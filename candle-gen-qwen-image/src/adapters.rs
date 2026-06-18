//! Qwen-Image-Edit inference-side adapter merge (sc-6220, epic 5480) — fold a LoRA/LoKr
//! `.safetensors` delta into the dense MMDiT transformer weights **before** [`crate::transformer::
//! QwenTransformer`] is built. The candle twin of `mlx-gen-qwen-image`'s adapter consumption (the
//! `AdaptableHost for QwenTransformer` module map) realized in the by-key-merge style the candle
//! `candle-gen-sdxl::adapters` already uses.
//!
//! **Primary consumer:** the **Qwen-Image-Edit-2511-Lightning** few-step distill (lightx2v) — a LoRA
//! over the per-block joint-attention + stream-MLP linears, merged at scale 1.0 so the 4-step
//! lightning schedule produces a clean edit. General Qwen-family LoRA/LoKr ride the same path.
//!
//! **Merge, don't residual** (same rationale as the SDXL merge): the flow-match Euler denoise is
//! precision-sensitive, so folding the delta into the dense weight (`W += δ`) reproduces the merged
//! forward `(W+δ)·x` exactly, with no per-step residual op. The delta is reconstructed with the same
//! f32 math the trainer's forward uses ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`] /
//! [`reconstruct_loha_delta`]), so a candle-trained adapter round-trips.
//!
//! **Merge at the safetensors-key level.** The candle MMDiT reads the diffusers transformer keys 1:1
//! (`transformer_blocks.{i}.attn.to_q.weight`, `…img_mlp.net.0.proj.weight`, …), so a LoRA's
//! prefix-stripped dotted module path resolves `{path}.weight` directly — no per-module routing table.
//! Formats resolved (the Qwen-family conventions; `gen-core`'s [`wmeta::COMMON_LORA_PREFIXES`]):
//!  - **PEFT / diffusers / bare LoRA** — `‹prefix›‹path›.lora_A/B[.default].weight` **or**
//!    `‹prefix›‹path›.lora_down/up.weight` (+ optional `‹path›.alpha`), where `‹prefix›` is
//!    `transformer.` / `diffusion_model.` / none. `lora_down`==`lora_A`, `lora_up`==`lora_B`. The
//!    lightx2v Lightning LoRA is the **bare-path + down/up + per-module-`.alpha`** form. The scaling is
//!    the per-target `.alpha` tensor (kohya / candle-trainer) or — when absent — `lora_alpha`/`r`
//!    (+ `alpha_pattern`/`rank_pattern`) in the `lora_adapter_metadata` blob (diffusers
//!    `save_lora_adapter`), else `rank`.
//!  - **LoKr** — PEFT-stamped `‹path›.lokr_w1`/`lokr_w2` (+ low-rank `_a`/`_b`) with `networkType=lokr`
//!    and `rank`/`alpha` in file metadata, reconstructing `δ = (alpha/rank)·kron(w1,w2)`.
//!  - **Third-party LyCORIS** — untagged `lokr_*` / `hada_*` (no `networkType` stamp), reconstructed
//!    per-module at the lycoris scale.
//!
//! **Linear-only.** Every Qwen MMDiT target is a `Linear` (the model has no conv layers), so — unlike
//! the SDXL merge — there is no conv-LoRA surface; a non-2-D factor or a factor that resolves to no
//! module is surfaced in [`MergeReport`], never silently dropped.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{
    reconstruct_loha_delta, reconstruct_lokr_delta, reconstruct_lora_delta, LoraAdapterMeta,
};
use candle_gen::{CandleError, Result};

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Outcome of merging the adapter specs into the base transformer tensor map: how many base weights
/// were updated, and how many keys fell outside the merge surface (a non-MMDiT module, a conv-shaped
/// factor, a text-encoder key, …) — surfaced, not silently dropped.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub merged: usize,
    pub skipped_keys: usize,
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

/// A loaded adapter file: its tensors (CPU, native dtype) and the safetensors header metadata.
struct AdapterFile {
    tensors: HashMap<String, Tensor>,
    meta: HashMap<String, String>,
}

/// Read an adapter `.safetensors` once: tensors via candle's loader, metadata via the safetensors
/// header reader (candle's `load` drops the `__metadata__`, where LoKr's `rank`/`alpha` + the PEFT
/// `lora_adapter_metadata` blob live).
fn read_adapter(path: &Path) -> Result<AdapterFile> {
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("read adapter {}: {e}", path.display())))?;
    let tensors = cst::load_buffer(&bytes, &Device::Cpu)?;
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("read adapter metadata {}: {e}", path.display())))?;
    let meta = md.metadata().clone().unwrap_or_default();
    Ok(AdapterFile { tensors, meta })
}

/// Strip a leading Qwen LoRA namespace prefix (`transformer.` / `diffusion_model.`), if present —
/// leaving the bare diffusers module path that resolves directly against the base transformer keys.
/// A bare key (the lightx2v Lightning convention) and a LoKr factor key (always bare) pass through.
fn strip_lora_prefix(key: &str) -> &str {
    for p in wmeta::COMMON_LORA_PREFIXES {
        if let Some(rest) = key.strip_prefix(p) {
            return rest;
        }
    }
    key
}

/// Map one LoRA key to `(module_path, role)`, or `None` if outside the merge surface. Strips the
/// optional namespace prefix, then matches both the PEFT (`lora_A`/`lora_B`, optional `.default.`
/// infix) and the diffusers/kohya (`lora_down`/`lora_up`) factor namings, plus the per-module
/// `.alpha` (which is often bare even when the factors are prefixed).
fn classify_lora_key(key: &str) -> Option<(String, Role)> {
    let rem = strip_lora_prefix(key);
    for (suf, role) in [
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".lora_down.weight", Role::Down),
        (".lora_up.weight", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(path) = rem.strip_suffix(suf) {
            return Some((path.to_string(), role));
        }
    }
    None
}

/// Map one (PEFT-stamped) LoKr factor key to `(module_path, factor_name)`, or `None` if out of
/// surface. Strips the optional namespace prefix; the factor name keeps its leading `.` dropped.
fn classify_lokr_key(key: &str) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return Some((strip_lora_prefix(stem).to_string(), factor));
        }
    }
    None
}

fn read_scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Read a per-module `.alpha` scalar as `f32`, regardless of on-disk dtype or shape (`[]` or `[1]`),
/// returning `None` for a size-0 (malformed) tensor rather than panicking.
fn read_scalar_opt(t: &Tensor) -> Result<Option<f32>> {
    if t.elem_count() == 0 {
        return Ok(None);
    }
    Ok(t.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .first()
        .copied())
}

/// Merge `delta` (`[out, in]` f32) into the base weight at `key`, computing `W += δ` in f32 (the
/// stored f32 sum is cast to the transformer load dtype when the VarBuilder serves it). A missing key
/// or a shape mismatch is surfaced as skipped, never a hard error.
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

/// Merge one LoRA file into `base` at `scale`: classify every key, fold complete `(down, up)` pairs
/// into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` is the per-target `.alpha` tensor when
/// present, else the `lora_adapter_metadata` blob's `alpha_pattern`/`lora_alpha`, else `rank`.
/// Linear-only (the Qwen MMDiT has no convs): a non-2-D pair, a half-pair, or an unresolved module is
/// surfaced as skipped.
fn merge_lora_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key) {
            Some((path, Role::Down)) => triples.entry(path).or_default().down = Some(t.clone()),
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(t.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(t)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` blob. `None` for kohya / candle-
    // trainer / lightx2v files (those ship a `.alpha` tensor), in which case the per-target `.alpha`
    // or the factor rank is used.
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // Linear-only surface (the MMDiT has no conv weights)
            continue;
        }
        let base_key = format!("{path}.weight");
        if !base.contains_key(&base_key) {
            report.skipped_keys += 1;
            continue;
        }
        // Effective scaling: per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor
        // rank. The denominator is the blob `r`/`rank_pattern` when given, else `A`'s leading dim.
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Merge one (PEFT-stamped) LoKr file into `base` at `scale`: `rank`/`alpha` from file metadata
/// (alpha defaults to rank), per-module factors grouped, `δ = (alpha/rank)·kron(w1,w2)·scale`
/// reconstructed and merged. Linear-only.
fn merge_lokr_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    let (rank, alpha) = wmeta::parse_rank_alpha(
        af.meta.get("rank").map(String::as_str),
        af.meta.get("alpha").map(String::as_str),
    );

    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lokr_key(key) {
            Some((path, factor)) => {
                grouped.entry(path).or_default().insert(factor, t.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, f) in grouped {
        let base_key = format!("{path}.weight");
        let Some(w) = base.get(&base_key) else {
            report.skipped_keys += 1;
            continue;
        };
        if w.dims().len() != 2 {
            report.skipped_keys += 1; // Linear-only surface
            continue;
        }
        let (out_f, in_f) = (w.dims()[0], w.dims()[1]);
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
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Whether the adapter file declares LoKr in its `networkType` metadata (the SceneWorks/PEFT stamp).
/// An untagged LyCORIS LoKr has the `lokr_*` factors but no stamp — see [`merge_lokr_thirdparty`].
fn declares_lokr(af: &AdapterFile) -> bool {
    wmeta::is_lokr_network_type(af.meta.get("networkType").map(String::as_str))
}

// ---- Third-party LyCORIS LoKr / LoHa -------------------------------------------------------------
//
// kohya / ai-toolkit / lycoris-lib LoKr (`lokr_*`) and LoHa (`hada_*`) files ship the decomposition
// factors but NOT the `networkType=lokr` stamp, and derive rank/alpha/scale **per module** (vs the
// PEFT path's one global pair). We reuse `gen_core::weightsmeta` for the suffix tables + the shared
// f32 reconstruction; the Qwen factor keys are dotted (bare or namespace-prefixed), so resolution is
// a prefix strip — no flattened table. Linear-only: a factor that resolves to no MMDiT module, or a
// conv/tucker (`lokr_t2`/`hada_t1`/`hada_t2`) form, is surfaced as skipped.

/// One module's third-party LoKr factors (full `w1`/`w2`, low-rank `_a`/`_b`, optional `.alpha`).
#[derive(Default)]
struct ThirdPartyLokr {
    w1: Option<Tensor>,
    w1_a: Option<Tensor>,
    w1_b: Option<Tensor>,
    w2: Option<Tensor>,
    w2_a: Option<Tensor>,
    w2_b: Option<Tensor>,
    alpha: Option<f32>,
}

impl ThirdPartyLokr {
    /// The factorization rank (`lora_dim`): `lokr_w1_a` is `[shape0, dim]`; else `lokr_w2_a`. `None`
    /// when both factors are full — lycoris then forces `alpha = lora_dim` ⇒ scale 1, so rank is unused.
    fn rank(&self) -> Option<f32> {
        if let Some(a) = &self.w1_a {
            return Some(a.dims()[1] as f32);
        }
        self.w2_a.as_ref().map(|a| a.dims()[1] as f32)
    }

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`), EXCEPT both-full forces
    /// scale 1 (`LokrModule.__init__`: `if use_w1 and use_w2: alpha = lora_dim`).
    fn lycoris_scale(&self) -> f32 {
        match self.rank() {
            None => 1.0,
            Some(r) => self.alpha.unwrap_or(r) / r,
        }
    }

    /// Reconstruct this module's `[out, in]` delta (lycoris per-module scale × `user_scale` baked in).
    fn delta(&self, base_shape: (usize, usize), user_scale: f32) -> Result<Tensor> {
        reconstruct_lokr_delta(
            self.w1.as_ref(),
            self.w1_a.as_ref(),
            self.w1_b.as_ref(),
            self.w2.as_ref(),
            self.w2_a.as_ref(),
            self.w2_b.as_ref(),
            self.lycoris_scale(),
            1.0,
            user_scale,
            base_shape,
        )
    }
}

/// Group a third-party LoKr file's tensors by raw module key (the part before `.lokr_*`/`.alpha`).
fn parse_lokr_thirdparty(af: &AdapterFile) -> Result<BTreeMap<String, ThirdPartyLokr>> {
    let mut groups: BTreeMap<String, ThirdPartyLokr> = BTreeMap::new();
    for (key, t) in &af.tensors {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = read_scalar_opt(t)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        if let Some((path, factor)) = wmeta::split_factor_key(key, &wmeta::LOKR_TP_SUFFIXES) {
            let g = groups.entry(path.to_string()).or_default();
            match factor {
                "lokr_w1" => g.w1 = Some(t.clone()),
                "lokr_w1_a" => g.w1_a = Some(t.clone()),
                "lokr_w1_b" => g.w1_b = Some(t.clone()),
                "lokr_w2" => g.w2 = Some(t.clone()),
                "lokr_w2_a" => g.w2_a = Some(t.clone()),
                "lokr_w2_b" => g.w2_b = Some(t.clone()),
                "lokr_t2" => {} // tucker (conv-only) — out of the Linear surface; module skips below.
                _ => {}
            }
        }
    }
    Ok(groups)
}

/// One module's third-party LoHa factors — two low-rank Hadamard pairs + optional `.alpha`.
#[derive(Default)]
struct ThirdPartyLoha {
    w1_a: Option<Tensor>,
    w1_b: Option<Tensor>,
    w2_a: Option<Tensor>,
    w2_b: Option<Tensor>,
    alpha: Option<f32>,
}

impl ThirdPartyLoha {
    /// rank (`lora_dim`) = `hada_w1_b.shape[0]` (lycoris stores `hada_w1_b` as `[lora_dim, …]`).
    fn rank(&self) -> Option<f32> {
        self.w1_b.as_ref().map(|b| b.dims()[0] as f32)
    }

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`). LoHa is always decomposed.
    fn lycoris_scale(&self) -> f32 {
        match self.rank() {
            None => 1.0,
            Some(r) => self.alpha.unwrap_or(r) / r,
        }
    }

    /// Reconstruct this module's `[out, in]` Hadamard delta (lycoris scale × `user_scale` baked in).
    fn delta(&self, base_shape: (usize, usize), user_scale: f32) -> Result<Tensor> {
        let (w1_a, w1_b, w2_a, w2_b) = match (&self.w1_a, &self.w1_b, &self.w2_a, &self.w2_b) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => {
                return Err(CandleError::Msg(
                    "loha: a hada_w1/w2 a/b factor is missing".into(),
                ))
            }
        };
        reconstruct_loha_delta(
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            self.lycoris_scale() * user_scale,
            base_shape,
        )
    }
}

/// Group a third-party LoHa file's tensors by raw module key (the part before `.hada_*`/`.alpha`).
fn parse_loha_thirdparty(af: &AdapterFile) -> Result<BTreeMap<String, ThirdPartyLoha>> {
    let mut groups: BTreeMap<String, ThirdPartyLoha> = BTreeMap::new();
    for (key, t) in &af.tensors {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = read_scalar_opt(t)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        if let Some((path, factor)) = wmeta::split_factor_key(key, &wmeta::LOHA_TP_SUFFIXES) {
            let g = groups.entry(path.to_string()).or_default();
            match factor {
                "hada_w1_a" => g.w1_a = Some(t.clone()),
                "hada_w1_b" => g.w1_b = Some(t.clone()),
                "hada_w2_a" => g.w2_a = Some(t.clone()),
                "hada_w2_b" => g.w2_b = Some(t.clone()),
                "hada_t1" | "hada_t2" => {} // tucker (conv-only) — module skips at the shape gate.
                _ => {}
            }
        }
    }
    Ok(groups)
}

/// Merge one reconstructed `[out, in]` delta into the resolved Linear module `path` (`W += δ`).
/// Shared by the third-party LoKr + LoHa paths: prefix-strip → Linear-only shape gate → reconstruct →
/// merge. A missing weight or a non-2-D (conv) target is surfaced as skipped, never mis-merged.
fn merge_thirdparty(
    base: &mut HashMap<String, Tensor>,
    raw: &str,
    delta_at: impl FnOnce((usize, usize)) -> Result<Tensor>,
    report: &mut MergeReport,
) -> Result<()> {
    let base_key = format!("{}.weight", strip_lora_prefix(raw));
    let Some(w) = base.get(&base_key) else {
        report.skipped_keys += 1;
        return Ok(());
    };
    if w.dims().len() != 2 {
        report.skipped_keys += 1; // Linear-only surface
        return Ok(());
    }
    let (out_f, in_f) = (w.dims()[0], w.dims()[1]);
    let delta = delta_at((out_f, in_f))?;
    merge_into(base, &base_key, &delta, report)
}

/// Merge a third-party LyCORIS **LoKr** file (`lokr_*` keys, per-module `.alpha`, no `networkType`
/// stamp) into `base` at `scale`.
fn merge_lokr_thirdparty(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    for (raw, g) in parse_lokr_thirdparty(af)? {
        merge_thirdparty(base, &raw, |bs| g.delta(bs, scale), report)?;
    }
    Ok(())
}

/// Merge a third-party LyCORIS **LoHa** file (`hada_*` keys) into `base` at `scale`.
fn merge_loha_thirdparty(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    report: &mut MergeReport,
) -> Result<()> {
    for (raw, g) in parse_loha_thirdparty(af)? {
        merge_thirdparty(base, &raw, |bs| g.delta(bs, scale), report)?;
    }
    Ok(())
}

/// Fold every adapter spec in `specs` into the base MMDiT tensor `map` (CPU, native dtype) at each
/// spec's `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`). Returns the
/// [`MergeReport`]; errors if a non-empty spec list matches **no** target (a format / prefix
/// misconfiguration — the worker should then fail rather than render an unadapted image silently).
pub fn merge_adapters(
    map: &mut HashMap<String, Tensor>,
    specs: &[AdapterSpec],
) -> Result<MergeReport> {
    if specs.is_empty() {
        return Ok(MergeReport::default());
    }
    let mut report = MergeReport::default();
    for spec in specs {
        let af = read_adapter(&spec.path)?;
        // Untagged LyCORIS: `lokr_*` / `hada_*` keys without a `networkType=lokr` stamp, so the
        // caller's declared `kind` can't label them — detect + route by keys before the kind match.
        if !declares_lokr(&af) && wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str)) {
            merge_lokr_thirdparty(map, &af, spec.scale, &mut report)?;
            continue;
        }
        if wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)) {
            merge_loha_thirdparty(map, &af, spec.scale, &mut report)?;
            continue;
        }
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &mut report)?,
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file has no lora_A/B keys
                // and would merge nothing; surface the mismatch loudly rather than no-op.
                if declares_lokr(&af) {
                    return Err(CandleError::Msg(format!(
                        "qwen edit: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(CandleError::Msg(format!(
            "qwen edit: no adapter target modules matched across {} file(s) — expected diffusers/PEFT \
             `‹transformer.|diffusion_model.›‹path›.lora_A/B|lora_down/up.weight` (+ optional `.alpha`) \
             over the MMDiT `transformer_blocks.{{i}}.{{attn.*,img_mlp.*,txt_mlp.*}}` modules, \
             `‹path›.lokr_w1/w2` with networkType=lokr (LoKr), or untagged LyCORIS `lokr_*` / `hada_*`",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny stand-in for the base MMDiT tensor map: two per-block attention Linears + one MLP Linear.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        for key in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
        ] {
            m.insert(
                key.to_string(),
                Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
            );
        }
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// The lightx2v Lightning shape: bare dotted path + `lora_down`/`lora_up` + per-module `.alpha`.
    #[test]
    fn classify_resolves_bare_down_up_and_alpha() {
        assert!(matches!(
            classify_lora_key("transformer_blocks.0.attn.to_q.lora_down.weight").unwrap(),
            (p, Role::Down) if p == "transformer_blocks.0.attn.to_q"
        ));
        assert!(matches!(
            classify_lora_key("transformer_blocks.0.attn.to_q.lora_up.weight").unwrap(),
            (p, Role::Up) if p == "transformer_blocks.0.attn.to_q"
        ));
        assert!(matches!(
            classify_lora_key("transformer_blocks.0.attn.to_out.0.alpha").unwrap(),
            (p, Role::Alpha) if p == "transformer_blocks.0.attn.to_out.0"
        ));
    }

    /// PEFT spelling with a `transformer.` namespace prefix + `lora_A`/`lora_B` (+ `.default.` infix).
    #[test]
    fn classify_strips_namespace_prefix_and_peft_naming() {
        let (p, role) =
            classify_lora_key("transformer.transformer_blocks.5.img_mlp.net.2.lora_A.weight")
                .unwrap();
        assert_eq!(p, "transformer_blocks.5.img_mlp.net.2");
        assert!(matches!(role, Role::Down));
        assert!(matches!(
            classify_lora_key(
                "diffusion_model.transformer_blocks.5.txt_mlp.net.2.lora_B.default.weight"
            )
            .unwrap()
            .1,
            Role::Up
        ));
        // A non-LoRA key is out of surface.
        assert!(classify_lora_key("transformer_blocks.0.attn.norm_q.weight").is_none());
    }

    /// The Lightning merge: a bare down/up + per-module `.alpha` LoRA folds `W += (alpha/rank)·B·A`.
    #[test]
    fn merge_lightning_shape_folds_expected_delta() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4); // A [rank=2, in=4]
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2); // B [out=4, rank=2]
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_down.weight"), down.clone()),
                (format!("{path}.lora_up.weight"), up.clone()),
                (
                    format!("{path}.alpha"),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        assert_eq!(report.skipped_keys, 0);
        // alpha 4 / rank 2 = 2.0; base is zero, so the merged weight IS ΔW = 2·(B·A).
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-2, "merged weight off by {diff}"); // bf16 base round-trip tolerance
    }

    /// The user-`scale` knob: the same adapter at scale 0.5 yields half the delta.
    #[test]
    fn merge_honors_user_scale() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let down = Tensor::randn(0f32, 1f32, (2, 4), &Device::Cpu).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2), &Device::Cpu).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (format!("{path}.lora_down.weight"), down.clone()),
                (format!("{path}.lora_up.weight"), up.clone()),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 0.5, &mut report).unwrap();
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
        // alpha defaults to rank (2) ⇒ effective scale = (2/2)·0.5 = 0.5.
        let expected = reconstruct_lora_delta(&down, &up, 2.0, 2.0, 0.5).unwrap();
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-2, "scaled merge off by {diff}");
    }

    /// PEFT LoKr (`networkType=lokr`, rank/alpha in metadata) folds the kron delta into the dense weight.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
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
        let mut report = MergeReport::default();
        merge_lokr_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
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
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-2, "merged lokr weight off by {diff}");
    }

    /// An untagged third-party LyCORIS LoKr (no `networkType`) is detected by keys + merged.
    #[test]
    fn merge_thirdparty_lokr_routes_and_merges() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let af = AdapterFile {
            tensors: HashMap::from([
                // both-full ⇒ lycoris scale 1.0.
                (format!("{path}.lokr_w1"), t2(&[1.0, 0.0, 0.0, 1.0], 2, 2)),
                (format!("{path}.lokr_w2"), t2(&[0.5, 0.0, 0.0, 0.5], 2, 2)),
            ]),
            meta: HashMap::new(), // no stamp → third-party
        };
        assert!(!declares_lokr(&af));
        assert!(wmeta::keys_contain_lokr(
            af.tensors.keys().map(String::as_str)
        ));
        // `merge_adapters` reads the file from disk; drive the in-memory third-party path directly.
        let mut report = MergeReport::default();
        merge_lokr_thirdparty(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
    }

    /// A third-party LoHa (`hada_*`) routes through the Hadamard merge into the resolved Linear.
    #[test]
    fn merge_thirdparty_loha_routes_and_merges() {
        let mut map = base_map();
        let path = "transformer_blocks.0.attn.to_q";
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    format!("{path}.hada_w1_a"),
                    t2(&[0.5, 0.1, -0.2, 0.3], 4, 1),
                ),
                (
                    format!("{path}.hada_w1_b"),
                    t2(&[0.4, -0.1, 0.2, 0.6], 1, 4),
                ),
                (
                    format!("{path}.hada_w2_a"),
                    t2(&[0.2, 0.0, 0.1, -0.3], 4, 1),
                ),
                (
                    format!("{path}.hada_w2_b"),
                    t2(&[1.0, 0.5, -0.5, 0.25], 1, 4),
                ),
            ]),
            meta: HashMap::new(),
        };
        assert!(wmeta::keys_contain_loha(
            af.tensors.keys().map(String::as_str)
        ));
        let mut report = MergeReport::default();
        merge_loha_thirdparty(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map[&format!("{path}.weight")]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(merged.iter().all(|v| v.is_finite()));
    }

    /// An empty spec list merges nothing (no error); the production edit path.
    #[test]
    fn merge_adapters_empty_is_noop() {
        let mut map = base_map();
        let report = merge_adapters(&mut map, &[]).unwrap();
        assert_eq!(report, MergeReport::default());
    }

    /// A non-empty LoRA file that matches no MMDiT module merges nothing (the loud-error precondition).
    #[test]
    fn merge_lora_file_matches_nothing_when_off_surface() {
        let mut map = base_map();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "transformer_blocks.99.attn.to_q.lora_down.weight".to_string(),
                    t2(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 4),
                ),
                (
                    "transformer_blocks.99.attn.to_q.lora_up.weight".to_string(),
                    t2(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4, 2),
                ),
            ]),
            meta: HashMap::new(),
        };
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert!(report.skipped_keys >= 1);
    }
}
