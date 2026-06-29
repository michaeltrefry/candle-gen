//! SD3.5 inference-side adapter merge (sc-7881, epic 7982) — load a community / trained LoRA/LoKr
//! `.safetensors` and fold its delta into the dense **MMDiT** (`transformer/`) weights **before** the
//! [`crate::transformer::Sd3Transformer`] is built. The SD3.5 sibling of the well-exercised Z-Image
//! ([`candle_gen_z_image::merge_adapters`]) / Krea inference-merge seam, re-homed onto the SD3.5 key
//! namespace.
//!
//! **Merge, don't residual** (same rationale as Z-Image / Krea / SDXL): inference has no need to keep
//! the factors trainable, so it folds `W += δ` into the dense weight and reproduces the merged-weight
//! forward exactly. SD3.5's flow-match sampler is chaos-sensitive — `(W+δ)·x` ≠ `W·x + δ·x` to ~1 ULP —
//! so a live residual would drift. The delta is reconstructed with the **same** f32 math the trainer's
//! forward uses ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`]), so a candle-trained adapter
//! round-trips. The merge runs at the safetensors-key level on a CPU map; quantization (Q4/Q8) folds
//! the merged dense weights **afterwards** in [`crate::pipeline::Pipeline::load_components`].
//!
//! ## The SD3.5 naming gap (the hard part)
//!
//! Community SD3.5 LoRAs are overwhelmingly **kohya / sd-scripts `lora_sd3`** files, whose keys use the
//! original **MMDiT-native** module names (`lora_unet_joint_blocks_<i>_x_block_attn_qkv` …), NOT the
//! diffusers names the candle port ([`crate::transformer`]) reads. Two structural differences make the
//! generic kohya-flatten-resolve trick (used by Z-Image/Krea) insufficient, so SD3.5 needs an explicit
//! native→diffusers map:
//!
//! 1. **Different module tree.** Native `joint_blocks_<i>.x_block` / `.context_block` vs diffusers
//!    `transformer_blocks.<i>` image/text streams; `final_layer`/`x_embedder`/`t_embedder`/`y_embedder`
//!    vs `norm_out`/`proj_out`/`pos_embed.proj`/`time_text_embed.*`. The native names don't appear in
//!    the diffusers base-key table, so they can't be resolved by `_`-flattening against it.
//! 2. **Fused QKV.** The native checkpoint trains a **single fused** `attn_qkv` projection
//!    (down `[r, inner]`, up `[3·inner, r]`); the diffusers port has it **split** into
//!    `attn.to_q`/`to_k`/`to_v` (each `[inner, inner]`). One LoRA module therefore maps to **three**
//!    base weights: reconstruct the fused `[3·inner, inner]` delta and slice it row-wise (`q | k | v`,
//!    in that packing order). The context stream's fused projection splits into
//!    `add_q_proj`/`add_k_proj`/`add_v_proj` the same way.
//!
//! For robustness we still accept **diffusers-named** adapters too (PEFT / `peft.save_pretrained`,
//! bare-dotted candle-trainer output, and kohya `lora_transformer_<flat>` resolved against the base key
//! set) — those use split q/k/v already, so they map 1:1 with no fusion.
//!
//! Out-of-surface keys are **counted and surfaced** in [`MergeReport`], never silently dropped:
//! text-encoder LoRAs (this is a DiT-only merge), conv-shaped (4-D) factors (e.g. an `x_embedder` patch
//! conv — the merge adapts 2-D Linears only), and any native module the map doesn't know.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta};
use candle_gen::{CandleError, Result};

/// kohya / sd-scripts **`lora_sd3`** prefix — the MMDiT-native flattening (`lora_unet_<native flat>`).
/// The leading `lora_unet_` is sd-scripts' historical UNet tag; for SD3 the flattened stem after it is
/// the original MMDiT module path ([`map_native_stem`]).
const KOHYA_NATIVE_PREFIX: &str = "lora_unet_";

/// kohya / community flattened-module prefix in **diffusers** naming (the DiT analog of SDXL's
/// `lora_unet_`, as used by Z-Image/Krea). The `_`-flattened stem is resolved against the base DiT key
/// table (diffusers names contain `_`, so the flattening is ambiguous without it).
const KOHYA_DIFFUSERS_PREFIX: &str = "lora_transformer_";

/// PEFT key prefixes tolerated on read, longest-first. The candle trainers write **bare** dotted paths
/// (no prefix), but community adapters and `peft.save_pretrained()` wrap the DiT under one of these;
/// stripping them yields the same dotted module path. A key matching none is taken as-is (bare).
const PEFT_PREFIXES: [&str; 4] = [
    "base_model.model.transformer.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
];

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Outcome of merging the adapter specs into the base MMDiT tensor map: how many base weights were
/// updated, and how many keys/targets fell outside the merge surface (text-encoder / conv-shaped /
/// unresolved native module / missing base key — surfaced, not silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub merged: usize,
    pub skipped_keys: usize,
}

/// One destination of a LoRA module's delta. A diffusers-named (already-split) adapter maps each module
/// to a single full-weight target (`chunk = None`); a kohya-native **fused** `attn_qkv` maps one module
/// to three row-slice targets (`chunk = Some((i, 3))`) — `q | k | v` packed in that order along the
/// delta's output (row) axis.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Target {
    /// Dotted base path (without the trailing `.weight`).
    path: String,
    /// `Some((index, parts))` ⇒ take row-slice `index` of `parts` equal slices; `None` ⇒ the whole delta.
    chunk: Option<(usize, usize)>,
}

impl Target {
    fn single(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            chunk: None,
        }
    }
    fn chunk(path: impl Into<String>, index: usize, parts: usize) -> Self {
        Self {
            path: path.into(),
            chunk: Some((index, parts)),
        }
    }
}

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
}

#[derive(Default)]
struct LoraTriple {
    down: Option<Tensor>, // A: [rank, in]
    up: Option<Tensor>,   // B: [out, rank]
    alpha: Option<f32>,
}

impl LoraTriple {
    /// Number of factor tensors collected — used to surface the right skipped-key count for a module
    /// that doesn't resolve to a target (so the report never undercounts a silent drop).
    fn key_count(&self) -> usize {
        self.down.is_some() as usize + self.up.is_some() as usize + self.alpha.is_some() as usize
    }
}

/// A loaded adapter file: its tensors (CPU, native dtype) and the safetensors header metadata.
struct AdapterFile {
    tensors: HashMap<String, Tensor>,
    meta: HashMap<String, String>,
}

/// Read an adapter `.safetensors` once: tensors via candle's loader, metadata via the safetensors
/// header reader (candle's `load` drops the header `__metadata__`, which LoKr's `rank`/`alpha` and the
/// diffusers `lora_adapter_metadata` blob live in).
fn read_adapter(path: &Path) -> Result<AdapterFile> {
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("read adapter {}: {e}", path.display())))?;
    let tensors = cst::load_buffer(&bytes, &Device::Cpu)?;
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("read adapter metadata {}: {e}", path.display())))?;
    let meta = md.metadata().clone().unwrap_or_default();
    Ok(AdapterFile { tensors, meta })
}

/// Strip the longest matching PEFT prefix, or return the key unchanged (bare dotted path).
fn strip_peft_prefix(key: &str) -> &str {
    for p in PEFT_PREFIXES {
        if let Some(rem) = key.strip_prefix(p) {
            return rem;
        }
    }
    key
}

/// Build the kohya `flattened → dotted` lookup table from the base MMDiT's 2-D Linear weight keys
/// (`{dotted}.weight`). Used only for the **diffusers**-named kohya path (`lora_transformer_<flat>`);
/// the native `lora_unet_` path is mapped explicitly ([`map_native_stem`]).
fn build_kohya_table(base: &HashMap<String, Tensor>) -> BTreeMap<String, String> {
    base.iter()
        .filter_map(|(k, t)| {
            let dotted = k.strip_suffix(".weight")?;
            (t.dims().len() == 2).then(|| (dotted.replace('.', "_"), dotted.to_string()))
        })
        .collect()
}

/// Map one MMDiT-native (`lora_sd3`) module stem — the flattened path **after** `lora_unet_` — to the
/// diffusers port target(s), or `None` if it's outside the known surface. A fused `attn_qkv` expands to
/// three row-slice targets (`q | k | v`); everything else is a single full-weight target.
///
/// This is the SD3.5-specific heart of the merge: the native names (`joint_blocks_<i>_x_block_…`,
/// `final_layer_…`, `<x|t|y>_embedder_…`) don't appear in the diffusers base-key table, so they can't be
/// resolved by flattening — they're translated here.
fn map_native_stem(stem: &str) -> Option<Vec<Target>> {
    // Per-block modules: joint_blocks_<i>_{x_block|context_block}_<leaf>.
    if let Some(after) = stem.strip_prefix("joint_blocks_") {
        let sep = after.find('_')?;
        let i: usize = after[..sep].parse().ok()?;
        let leaf = &after[sep + 1..];
        // The image stream uses `to_*`/`ff`/`norm1`; the text stream uses `add_*_proj`/`to_add_out`/
        // `ff_context`/`norm1_context`.
        if let Some(m) = leaf.strip_prefix("x_block_") {
            return map_block_leaf(i, m, false);
        }
        if let Some(m) = leaf.strip_prefix("context_block_") {
            return map_block_leaf(i, m, true);
        }
        return None;
    }
    // Top-level modules.
    let one = |p: &str| Some(vec![Target::single(p)]);
    match stem {
        // `final_layer`: the AdaLayerNormContinuous output head + linear projection.
        "final_layer_linear" => one("proj_out"),
        "final_layer_adaLN_modulation_1" => one("norm_out.linear"),
        // The context (T5/CLIP) projection into the joint width.
        "context_embedder" => one("context_embedder"),
        // The patchify conv (4-D) — included for completeness; a 2-D LoRA on it is shape-skipped.
        "x_embedder_proj" => one("pos_embed.proj"),
        // CombinedTimestepTextProjEmbeddings: timestep MLP = t_embedder, pooled-text MLP = y_embedder.
        "t_embedder_mlp_0" => one("time_text_embed.timestep_embedder.linear_1"),
        "t_embedder_mlp_2" => one("time_text_embed.timestep_embedder.linear_2"),
        "y_embedder_mlp_0" => one("time_text_embed.text_embedder.linear_1"),
        "y_embedder_mlp_2" => one("time_text_embed.text_embedder.linear_2"),
        _ => None,
    }
}

/// Map a per-block native leaf (the part after `x_block_` / `context_block_`) to its diffusers
/// target(s) for joint block `i`. `context` selects the text-stream names.
fn map_block_leaf(i: usize, leaf: &str, context: bool) -> Option<Vec<Target>> {
    let attn = format!("transformer_blocks.{i}.attn");
    match (leaf, context) {
        // Fused QKV → three row-slices (q | k | v). Image stream = to_q/to_k/to_v; text = add_*_proj.
        ("attn_qkv", false) => Some(vec![
            Target::chunk(format!("{attn}.to_q"), 0, 3),
            Target::chunk(format!("{attn}.to_k"), 1, 3),
            Target::chunk(format!("{attn}.to_v"), 2, 3),
        ]),
        ("attn_qkv", true) => Some(vec![
            Target::chunk(format!("{attn}.add_q_proj"), 0, 3),
            Target::chunk(format!("{attn}.add_k_proj"), 1, 3),
            Target::chunk(format!("{attn}.add_v_proj"), 2, 3),
        ]),
        // Attention output projection.
        ("attn_proj", false) => Some(vec![Target::single(format!("{attn}.to_out.0"))]),
        ("attn_proj", true) => Some(vec![Target::single(format!("{attn}.to_add_out"))]),
        // Feed-forward (diffusers nests in/out at net.0.proj / net.2).
        ("mlp_fc1", false) => Some(vec![Target::single(format!(
            "transformer_blocks.{i}.ff.net.0.proj"
        ))]),
        ("mlp_fc2", false) => Some(vec![Target::single(format!(
            "transformer_blocks.{i}.ff.net.2"
        ))]),
        ("mlp_fc1", true) => Some(vec![Target::single(format!(
            "transformer_blocks.{i}.ff_context.net.0.proj"
        ))]),
        ("mlp_fc2", true) => Some(vec![Target::single(format!(
            "transformer_blocks.{i}.ff_context.net.2"
        ))]),
        // AdaLN-Zero modulation linear.
        ("adaLN_modulation_1", false) => Some(vec![Target::single(format!(
            "transformer_blocks.{i}.norm1.linear"
        ))]),
        ("adaLN_modulation_1", true) => Some(vec![Target::single(format!(
            "transformer_blocks.{i}.norm1_context.linear"
        ))]),
        _ => None,
    }
}

/// Resolve a LoRA module stem (the key with its `.lora_*`/`.alpha`/`.lokr_*` role suffix removed) to its
/// diffusers target(s). Tries, in order: kohya-native (`lora_unet_…`, the [`map_native_stem`] explicit
/// map with fused-QKV expansion); kohya-diffusers (`lora_transformer_<flat>`, resolved against the base
/// key table); then PEFT/bare diffusers dotted (1:1, existence checked at merge). `None` ⇒ out of surface.
fn resolve_targets(stem: &str, table: &BTreeMap<String, String>) -> Option<Vec<Target>> {
    if let Some(native) = stem.strip_prefix(KOHYA_NATIVE_PREFIX) {
        return map_native_stem(native);
    }
    if let Some(flat) = stem.strip_prefix(KOHYA_DIFFUSERS_PREFIX) {
        return table.get(flat).map(|d| vec![Target::single(d.clone())]);
    }
    // PEFT / bare dotted. Only a dotted path (or an exact base-key match) is a plausible DiT module; a
    // flat token like `lora_te1_…` (text-encoder) is rejected here so it's surfaced as skipped.
    let path = strip_peft_prefix(stem);
    if path.contains('.') || table.values().any(|v| v == path) {
        Some(vec![Target::single(path.to_string())])
    } else {
        None
    }
}

fn read_scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Split a LoRA key into `(module_stem, role)` — handling both kohya (`.lora_down.weight` /
/// `.lora_up.weight` / `.alpha`) and PEFT (`.lora_A[.default].weight` / `.lora_B[.default].weight` /
/// `.alpha`) factor suffixes. `None` ⇒ not a LoRA factor key.
fn split_lora_role(key: &str) -> Option<(&str, Role)> {
    for (suf, role) in [
        (".lora_down.weight", Role::Down),
        (".lora_up.weight", Role::Up),
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(stem) = key.strip_suffix(suf) {
            return Some((stem, role));
        }
    }
    None
}

/// Split a LoKr key into `(module_stem, factor_name)`. `None` ⇒ not a LoKr factor key.
fn split_lokr_factor(key: &str) -> Option<(&str, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            return Some((stem, &suf[1..])); // drop the leading '.'
        }
    }
    None
}

/// Merge `delta` (`[out, in]` f32) into the base weight at `key`, computing `W += δ` in f32 (the stored
/// f32 sum is cast to the MMDiT load dtype when the `VarBuilder` serves it). A missing key or a
/// shape-mismatched base (e.g. a 4-D conv weight, or a fused-split slice that doesn't match the port's
/// projection) is surfaced as skipped, never a hard error.
fn merge_into(
    base: &mut HashMap<String, Tensor>,
    key: &str,
    delta: &Tensor,
    report: &mut MergeReport,
) -> Result<()> {
    let merged = {
        let Some(w) = base.get(key) else {
            report.skipped_keys += 1;
            return Ok(());
        };
        if w.dims() != delta.dims() {
            report.skipped_keys += 1;
            return Ok(());
        }
        (w.to_dtype(DType::F32)? + delta)?
    };
    base.insert(key.to_string(), merged);
    report.merged += 1;
    Ok(())
}

/// Fold a reconstructed module `delta` (`[out, in]` f32) into every [`Target`] — the whole delta for a
/// single target, or the appropriate row-slice for a fused-QKV target. Each target merge is shape-checked
/// and surfaced independently in `report`.
fn merge_targets(
    base: &mut HashMap<String, Tensor>,
    targets: &[Target],
    delta: &Tensor,
    report: &mut MergeReport,
) -> Result<()> {
    for t in targets {
        let key = format!("{}.weight", t.path);
        match t.chunk {
            None => merge_into(base, &key, delta, report)?,
            Some((index, parts)) => {
                let total = delta.dim(0)?;
                if !total.is_multiple_of(parts) {
                    report.skipped_keys += 1; // fused out-dim not divisible by parts — surface, don't slice
                    continue;
                }
                let rows = total / parts;
                let slice = delta.narrow(0, index * rows, rows)?.contiguous()?;
                merge_into(base, &key, &slice, report)?;
            }
        }
    }
    Ok(())
}

/// Merge one LoRA file into `base` at `scale`: group factors by module stem, resolve each stem to its
/// target(s), reconstruct `δ = (alpha/rank)·scale·B·A` and fold it (whole, or row-sliced for fused QKV).
/// `rank` is `A`'s leading dim; `alpha` is the per-target `.alpha` tensor when present, else the
/// `lora_adapter_metadata` blob (diffusers / PEFT `save_lora_adapter` — sc-5374), else `rank`. Half-pairs,
/// conv-shaped (4-D) factors, and unresolved stems are surfaced as skipped.
fn merge_lora_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    let mut modules: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match split_lora_role(key) {
            Some((stem, Role::Down)) => {
                modules.entry(stem.to_string()).or_default().down = Some(t.clone())
            }
            Some((stem, Role::Up)) => {
                modules.entry(stem.to_string()).or_default().up = Some(t.clone())
            }
            Some((stem, Role::Alpha)) => {
                modules.entry(stem.to_string()).or_default().alpha = Some(read_scalar(t)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    // sc-5374: diffusers / PEFT `save_lora_adapter` files ship no per-target `.alpha` tensor —
    // `lora_alpha`/`r` (+ per-module overrides) live in the `lora_adapter_metadata` header blob.
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (stem, tr) in modules {
        let Some(targets) = resolve_targets(&stem, table) else {
            report.skipped_keys += tr.key_count(); // unresolved module — surface, don't silently drop
            continue;
        };
        let (Some(down), Some(up)) = (tr.down, tr.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // conv-shaped LoRA — out of surface
            continue;
        }
        // per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor rank (last resort).
        // The metadata pattern match keys off the (first) diffusers target path.
        let (cfg_alpha, cfg_rank) = cfg
            .as_ref()
            .map_or((None, None), |c| c.effective(&targets[0].path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = tr.alpha.or(cfg_alpha).unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_targets(base, &targets, &delta, report)?;
    }
    Ok(())
}

/// The full `[out, in]` shape a module's reconstructed delta must have, from its target(s): a single
/// target uses its base weight's shape; a fused-QKV module multiplies the per-slice base rows by the
/// number of slices. `None` (skip) if the (first) base key is missing or not 2-D (conv).
fn module_full_shape(base: &HashMap<String, Tensor>, targets: &[Target]) -> Option<(usize, usize)> {
    let first = targets.first()?;
    let w = base.get(&format!("{}.weight", first.path))?;
    if w.dims().len() != 2 {
        return None;
    }
    let (rows, cols) = (w.dims()[0], w.dims()[1]);
    let parts = first.chunk.map(|(_, n)| n).unwrap_or(1);
    Some((rows * parts, cols))
}

/// Merge one LoKr file into `base` at `scale`: `rank`/`alpha` from file metadata (alpha defaults to
/// rank), per-module factors grouped, `δ = (alpha/rank)·kron(w1,w2)·scale` reconstructed at the module's
/// full (possibly fused) `[out, in]` shape and folded (whole, or row-sliced for fused QKV).
fn merge_lokr_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    let rank = af
        .meta
        .get("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let alpha = af
        .meta
        .get("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);

    let mut modules: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match split_lokr_factor(key) {
            Some((stem, factor)) => {
                modules
                    .entry(stem.to_string())
                    .or_default()
                    .insert(factor, t.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (stem, f) in modules {
        let Some(targets) = resolve_targets(&stem, table) else {
            report.skipped_keys += f.len();
            continue;
        };
        let Some((out_f, in_f)) = module_full_shape(base, &targets) else {
            report.skipped_keys += 1; // missing base key or conv (4-D) — out of surface
            continue;
        };
        let delta = reconstruct_lokr_delta(
            f.get("lokr_w1"),
            f.get("lokr_w1_a"),
            f.get("lokr_w1_b"),
            f.get("lokr_w2"),
            f.get("lokr_w2_a"),
            f.get("lokr_w2_b"),
            alpha,
            rank,
            scale,
            (out_f, in_f),
        )?;
        merge_targets(base, &targets, &delta, report)?;
    }
    Ok(())
}

/// Whether the adapter file declares LoKr in its `networkType` metadata.
fn declares_lokr(af: &AdapterFile) -> bool {
    af.meta.get("networkType").map(String::as_str) == Some("lokr")
}

/// Fold every adapter spec in `specs` into the base MMDiT tensor `map` (CPU, native dtype) at each
/// spec's `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`). Returns the [`MergeReport`];
/// errors if a non-empty spec list matches **no** target (a format / prefix misconfiguration — the
/// worker should then fall back rather than render an unadapted image silently).
pub fn merge_adapters(
    map: &mut HashMap<String, Tensor>,
    specs: &[AdapterSpec],
) -> Result<MergeReport> {
    if specs.is_empty() {
        return Ok(MergeReport::default());
    }
    let table = build_kohya_table(map);
    let mut report = MergeReport::default();
    for spec in specs {
        let af = read_adapter(&spec.path)?;
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &table, &mut report)?,
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file has no lora_A/B keys
                // and would merge nothing; surface the mismatch loudly rather than no-op.
                if declares_lokr(&af) {
                    return Err(CandleError::Msg(format!(
                        "sd3: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &table, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(CandleError::Msg(format!(
            "sd3: no adapter target modules matched across {} file(s) — expected a kohya `lora_sd3` \
             file (`lora_unet_joint_blocks_<i>_<x|context>_block_…`, fused `attn_qkv`) or a \
             diffusers-named LoRA (`transformer_blocks.<i>.attn.<to_q|to_k|to_v|to_out.0>` …). \
             Conv-layer / text-encoder adapters are out of surface",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    fn max_abs(t: &Tensor) -> f32 {
        t.abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// A tiny stand-in for the base MMDiT tensor map at `inner = 4`: one full joint block's split
    /// attention (image + text), its FFN + AdaLN linears, the embedders + output head, and a conv
    /// (4-D) `pos_embed.proj` that a 2-D LoRA must never touch.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        let z = |r: usize, c: usize| Tensor::zeros((r, c), DType::BF16, &dev).unwrap();
        // Image-stream attention (split q/k/v + out).
        for k in ["to_q", "to_k", "to_v", "to_out.0"] {
            m.insert(format!("transformer_blocks.0.attn.{k}.weight"), z(4, 4));
        }
        // Text-stream attention (split add_q/k/v + to_add_out).
        for k in ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"] {
            m.insert(format!("transformer_blocks.0.attn.{k}.weight"), z(4, 4));
        }
        // FFN (image + context) — net.0.proj is [hidden, inner], net.2 is [inner, hidden]; hidden=8.
        m.insert("transformer_blocks.0.ff.net.0.proj.weight".into(), z(8, 4));
        m.insert("transformer_blocks.0.ff.net.2.weight".into(), z(4, 8));
        m.insert(
            "transformer_blocks.0.ff_context.net.0.proj.weight".into(),
            z(8, 4),
        );
        m.insert(
            "transformer_blocks.0.ff_context.net.2.weight".into(),
            z(4, 8),
        );
        // AdaLN linears (6·inner and 2·inner).
        m.insert("transformer_blocks.0.norm1.linear.weight".into(), z(24, 4));
        m.insert(
            "transformer_blocks.0.norm1_context.linear.weight".into(),
            z(24, 4),
        );
        // Embedders + output head.
        m.insert("context_embedder.weight".into(), z(4, 16));
        m.insert(
            "time_text_embed.timestep_embedder.linear_1.weight".into(),
            z(4, 4),
        );
        m.insert(
            "time_text_embed.text_embedder.linear_1.weight".into(),
            z(4, 8),
        );
        m.insert("norm_out.linear.weight".into(), z(8, 4));
        m.insert("proj_out.weight".into(), z(16, 4));
        // A conv (4-D) patch-embed weight — must never be merged by a 2-D LoRA.
        m.insert(
            "pos_embed.proj.weight".into(),
            Tensor::zeros((4, 16, 2, 2), DType::BF16, &dev).unwrap(),
        );
        m
    }

    /// The native `lora_sd3` names map to the diffusers port paths — including the fused `attn_qkv`
    /// expanding to the three split projections (image to_q/k/v, text add_*_proj) and the
    /// final_layer/embedder top-level modules.
    #[test]
    fn native_stem_mapping_covers_the_surface() {
        // Image stream.
        assert_eq!(
            map_native_stem("joint_blocks_7_x_block_attn_qkv").unwrap(),
            vec![
                Target::chunk("transformer_blocks.7.attn.to_q", 0, 3),
                Target::chunk("transformer_blocks.7.attn.to_k", 1, 3),
                Target::chunk("transformer_blocks.7.attn.to_v", 2, 3),
            ]
        );
        assert_eq!(
            map_native_stem("joint_blocks_7_x_block_attn_proj").unwrap(),
            vec![Target::single("transformer_blocks.7.attn.to_out.0")]
        );
        assert_eq!(
            map_native_stem("joint_blocks_3_x_block_mlp_fc1").unwrap(),
            vec![Target::single("transformer_blocks.3.ff.net.0.proj")]
        );
        assert_eq!(
            map_native_stem("joint_blocks_3_x_block_mlp_fc2").unwrap(),
            vec![Target::single("transformer_blocks.3.ff.net.2")]
        );
        assert_eq!(
            map_native_stem("joint_blocks_3_x_block_adaLN_modulation_1").unwrap(),
            vec![Target::single("transformer_blocks.3.norm1.linear")]
        );
        // Text/context stream.
        assert_eq!(
            map_native_stem("joint_blocks_2_context_block_attn_qkv").unwrap(),
            vec![
                Target::chunk("transformer_blocks.2.attn.add_q_proj", 0, 3),
                Target::chunk("transformer_blocks.2.attn.add_k_proj", 1, 3),
                Target::chunk("transformer_blocks.2.attn.add_v_proj", 2, 3),
            ]
        );
        assert_eq!(
            map_native_stem("joint_blocks_2_context_block_attn_proj").unwrap(),
            vec![Target::single("transformer_blocks.2.attn.to_add_out")]
        );
        assert_eq!(
            map_native_stem("joint_blocks_2_context_block_mlp_fc1").unwrap(),
            vec![Target::single("transformer_blocks.2.ff_context.net.0.proj")]
        );
        assert_eq!(
            map_native_stem("joint_blocks_2_context_block_adaLN_modulation_1").unwrap(),
            vec![Target::single("transformer_blocks.2.norm1_context.linear")]
        );
        // Top-level modules.
        for (native, diff) in [
            ("final_layer_linear", "proj_out"),
            ("final_layer_adaLN_modulation_1", "norm_out.linear"),
            ("context_embedder", "context_embedder"),
            ("x_embedder_proj", "pos_embed.proj"),
            (
                "t_embedder_mlp_0",
                "time_text_embed.timestep_embedder.linear_1",
            ),
            (
                "t_embedder_mlp_2",
                "time_text_embed.timestep_embedder.linear_2",
            ),
            ("y_embedder_mlp_0", "time_text_embed.text_embedder.linear_1"),
            ("y_embedder_mlp_2", "time_text_embed.text_embedder.linear_2"),
        ] {
            assert_eq!(
                map_native_stem(native).unwrap(),
                vec![Target::single(diff)],
                "native {native} must map to {diff}"
            );
        }
        // An unknown native module is out of surface.
        assert!(map_native_stem("joint_blocks_0_x_block_bogus").is_none());
        assert!(map_native_stem("mystery_module").is_none());
    }

    /// `resolve_targets` accepts the kohya-native prefix, the kohya-diffusers flatten, and PEFT/bare
    /// dotted names — and rejects a text-encoder token.
    #[test]
    fn resolve_targets_handles_every_format() {
        let table = build_kohya_table(&base_map());
        // kohya-native (the lora_sd3 portrait checkpoint format).
        assert_eq!(
            resolve_targets("lora_unet_joint_blocks_0_x_block_attn_qkv", &table).unwrap(),
            vec![
                Target::chunk("transformer_blocks.0.attn.to_q", 0, 3),
                Target::chunk("transformer_blocks.0.attn.to_k", 1, 3),
                Target::chunk("transformer_blocks.0.attn.to_v", 2, 3),
            ]
        );
        // kohya-diffusers flatten (resolved against the base key table; incl. to_out.0 → to_out_0).
        assert_eq!(
            resolve_targets(
                "lora_transformer_transformer_blocks_0_attn_to_out_0",
                &table
            )
            .unwrap(),
            vec![Target::single("transformer_blocks.0.attn.to_out.0")]
        );
        // PEFT-prefixed + bare dotted.
        assert_eq!(
            resolve_targets("transformer.transformer_blocks.0.attn.to_q", &table).unwrap(),
            vec![Target::single("transformer_blocks.0.attn.to_q")]
        );
        assert_eq!(
            resolve_targets("transformer_blocks.0.attn.to_k", &table).unwrap(),
            vec![Target::single("transformer_blocks.0.attn.to_k")]
        );
        // Text-encoder flat token — out of surface.
        assert!(resolve_targets(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj",
            &table
        )
        .is_none());
    }

    /// **The fused-QKV split is correct.** A single native `attn_qkv` LoRA (down `[r, inner]`, up
    /// `[3·inner, r]`) merges into the three split projections, each receiving the matching row-slice of
    /// the `[3·inner, inner]` reconstructed delta (`q | k | v` in that order). Base is zero, so the
    /// merged weight IS the slice.
    #[test]
    fn fused_qkv_lora_splits_into_three() {
        let dev = Device::Cpu;
        let mut map = base_map();
        // rank 2, inner 4 ⇒ down [2,4], up [12,2]; alpha 4 ⇒ effective (4/2)=2.0 at scale 1.
        let down = Tensor::randn(0f32, 1f32, (2, 4), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (12, 2), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight".to_string(),
                    down.clone(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_up.weight".to_string(),
                    up.clone(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        // Three projections merged, nothing skipped.
        assert_eq!(report.merged, 3, "fused qkv must merge q, k, v");
        assert_eq!(report.skipped_keys, 0);

        // The full delta, then each split slice, must match the merged weights exactly.
        let full = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap(); // [12,4]
        for (i, proj) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let merged = map
                .get(&format!("transformer_blocks.0.attn.{proj}.weight"))
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let expected = full.narrow(0, i * 4, 4).unwrap();
            assert!(
                max_abs(&(merged - expected).unwrap()) < 1e-4,
                "{proj} slice mismatch"
            );
        }
    }

    /// A bare/PEFT diffusers-named (already-split) LoRA on `to_q` merges 1:1 (no fusion) at
    /// `(alpha/rank)·scale`. Confirms the diffusers fallback path next to the native path.
    #[test]
    fn diffusers_named_lora_merges_one_to_one() {
        let dev = Device::Cpu;
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let down = Tensor::randn(0f32, 1f32, (2, 4), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_A.weight"), down.clone()),
                (format!("{path}.lora_B.weight"), up.clone()),
                (
                    format!("{path}.alpha"),
                    Tensor::from_vec(vec![4.0f32], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get(&format!("{path}.weight"))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        assert!(max_abs(&(merged - expected).unwrap()) < 1e-4);
    }

    /// A 2-D LoRA targeting the conv `x_embedder` (`pos_embed.proj`, a 4-D base) is surfaced as skipped,
    /// never merged into the conv weight.
    #[test]
    fn conv_target_is_skipped_not_merged() {
        let dev = Device::Cpu;
        let mut map = base_map();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_x_embedder_proj.lora_down.weight".to_string(),
                    Tensor::randn(0f32, 1f32, (2, 16), &dev).unwrap(),
                ),
                (
                    "lora_unet_x_embedder_proj.lora_up.weight".to_string(),
                    Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 0, "the conv pos_embed.proj must not merge");
        assert_eq!(report.skipped_keys, 1); // the (down,up) pair, shape-mismatched against the 4-D base
    }

    /// A non-empty spec list that resolves nothing is a loud error (not a silent unadapted render).
    #[test]
    fn unresolvable_specs_error() {
        let dev = Device::Cpu;
        let pid = std::process::id();
        let file = std::env::temp_dir().join(format!("sd3_adapt_none_{pid}.safetensors"));
        let tensors = HashMap::from([
            (
                "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"
                    .to_string(),
                Tensor::zeros((2, 4), DType::F32, &dev).unwrap(),
            ),
            (
                "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_string(),
                Tensor::zeros((4, 2), DType::F32, &dev).unwrap(),
            ),
        ]);
        cst::save(&tensors, &file).unwrap();
        let mut map = base_map();
        let res = merge_adapters(
            &mut map,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&file).ok();
        let Err(e) = res else {
            panic!("text-encoder-only adapter must error (nothing merged)")
        };
        assert!(e.to_string().contains("no adapter target modules matched"));
    }

    /// **AC: scale-0 merge is byte-exact with the base.** A fused-QKV LoRA folded at `scale = 0` adds a
    /// zero delta, so every targeted projection equals its original (nonzero) base bit-for-bit — a LoRA
    /// at strength 0 is a no-op render.
    #[test]
    fn scale_zero_merge_is_base_byte_exact() {
        let dev = Device::Cpu;
        let mut map = base_map();
        // Give the three image projections a nonzero base so "equals base" is a real assertion.
        let bases: Vec<(String, Tensor)> = ["to_q", "to_k", "to_v"]
            .iter()
            .map(|p| {
                let key = format!("transformer_blocks.0.attn.{p}.weight");
                let w = Tensor::randn(0f32, 1f32, (4, 4), &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap();
                map.insert(key.clone(), w.clone());
                (key, w.to_dtype(DType::F32).unwrap())
            })
            .collect();

        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight".to_string(),
                    Tensor::randn(0f32, 1f32, (2, 4), &dev).unwrap(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_up.weight".to_string(),
                    Tensor::randn(0f32, 1f32, (12, 2), &dev).unwrap(),
                ),
                (
                    "lora_unet_joint_blocks_0_x_block_attn_qkv.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 0.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 3, "targets still 'merge' a zero delta");
        for (key, original) in bases {
            let merged = map.get(&key).unwrap().to_dtype(DType::F32).unwrap();
            assert_eq!(
                max_abs(&(merged - original).unwrap()),
                0.0,
                "scale-0 merge must be byte-exact with the base at {key}"
            );
        }
    }

    /// LoKr merges `δ = (alpha/rank)·kron(w1,w2)` into a single diffusers-named target, reading rank/alpha
    /// from the file metadata.
    #[test]
    fn lokr_merges_kron_delta() {
        let mut map = base_map();
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
        let path = "transformer_blocks.0.attn.to_q";
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lokr_w1"), w1.clone()),
                (format!("{path}.lokr_w2"), w2.clone()),
            ]),
            meta: HashMap::from([
                ("networkType".to_string(), "lokr".to_string()),
                ("rank".to_string(), "2".to_string()),
                ("alpha".to_string(), "2".to_string()),
            ]),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lokr_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get(&format!("{path}.weight"))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            2.0,
            2.0,
            1.0,
            (4, 4),
        )
        .unwrap();
        assert!(max_abs(&(merged - expected).unwrap()) < 1e-2);
    }

    /// The keystone train→infer round-trip: a PEFT `.safetensors` written by the **actual trainer** path
    /// ([`candle_gen::train::lora::save_lora_peft`] with the DiT's empty prefix) is read back through the
    /// public [`merge_adapters`] entry, and the merged weight equals the trained delta.
    #[test]
    fn roundtrip_trainer_peft_file_merges() {
        use candle_gen::candle_nn::Linear;
        use candle_gen::train::lora::{build_lora_targets, save_lora_peft, LoraHost, LoraLinear};

        struct Host(LoraLinear);
        impl LoraHost for Host {
            fn visit_lora_mut(
                &mut self,
                f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
            ) -> candle_gen::Result<()> {
                f(&mut self.0)
            }
        }

        let dev = Device::Cpu;
        let path = "transformer_blocks.0.attn.to_v";
        let base_w = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        let mut host = Host(LoraLinear::from_linear(
            Linear::new(base_w, None),
            4,
            4,
            path.into(),
        ));
        // rank 2, alpha 4 ⇒ effective 2.0. Force B (vars[1]) nonzero so ΔW ≠ 0 (zero-init B no-ops).
        let set = build_lora_targets(&mut host, &["to_v".to_string()], 2, 4.0, 7, &dev).unwrap();
        let up_randn = Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap();
        set.vars[1].set(&up_randn).unwrap();

        let file = std::env::temp_dir().join(format!(
            "sd3_lora_roundtrip_{}.safetensors",
            std::process::id()
        ));
        save_lora_peft(&set, "", &HashMap::new(), &file).unwrap();

        let mut map = base_map();
        let report = merge_adapters(
            &mut map,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&file).ok();
        let report = report.unwrap();

        assert_eq!(report.merged, 1, "the trained to_v adapter must merge");
        let expected = reconstruct_lora_delta(
            set.vars[0].as_tensor(),
            set.vars[1].as_tensor(),
            4.0,
            2.0,
            1.0,
        )
        .unwrap();
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
        assert!(max_abs(&(&merged - &expected).unwrap()) < 1e-2);
        assert!(
            max_abs(&expected) > 0.0,
            "forced-nonzero B must be non-trivial"
        );
    }
}
