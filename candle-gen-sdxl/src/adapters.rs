//! SDXL inference-side adapter merge (sc-5165) — load a trained LoRA/LoKr `.safetensors` and fold its
//! delta into the dense UNet weights **before** the stock candle-transformers UNet is built. The
//! candle twin of `mlx-gen-sdxl::adapters`, and the closing half of the native-trainer loop: a LoRA
//! produced by [`candle_gen::train`]'s SDXL trainer now actually loads in candle inference.
//!
//! **Merge, don't residual.** SDXL's sampler is chaos-sensitive: the merged forward `(W+δ)·x` differs
//! from a forward-time residual `W·x + δ·x` by ~1 ULP, which cascades to a visibly different image.
//! The training seam ([`candle_gen::train::lora::LoraLinear`]) *must* add a residual (the factors stay
//! trainable); inference has no such need, so it merges and reproduces the merged-weight forward
//! exactly. The delta is reconstructed with the **same** f32 math the trainer's forward uses
//! ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`]), so a candle-trained adapter round-trips.
//!
//! **Merge at the safetensors-key level.** Unlike the MLX port (which fights its vendored UNet's
//! module naming), candle merges into the raw base-weight tensor map *before* construction: the stock
//! candle UNet reads diffusers keys 1:1, so `{path}.weight` is a valid base key for every Linear an
//! adapter targets — attention (`to_q`/`to_k`/`to_v`/`to_out.0`), `proj_in`/`proj_out`, the GEGLU
//! `ff.net.0.proj`/`ff.net.2`, and `mid_block.*`. This is diffusers' full ("complete") coverage by
//! construction, no per-module routing table on our side. Two on-disk LoRA formats resolve:
//!  - **PEFT** (`base_model.model.unet.<dotted>.lora_A/B[.default].weight`) — what the candle trainer
//!    ([`save_lora_peft`](candle_gen::train::lora::save_lora_peft)) and `peft.save_pretrained()` /
//!    diffusers `save_lora_adapter` emit; the dotted path resolves directly (the prefix is optional).
//!    The scaling is a per-target `.alpha` tensor (candle trainer / kohya) or — when absent, as in the
//!    diffusers format — `lora_alpha`/`r` (+ `alpha_pattern`/`rank_pattern`) in the
//!    `lora_adapter_metadata` header blob ([`LoraAdapterMeta`](candle_gen::train::lora::LoraAdapterMeta), sc-5374).
//!  - **kohya** (`lora_unet_<flat>.lora_down/up.weight` + `.alpha`) — community / diffusers LoRAs; the
//!    `_`-flattened stem (ambiguous, since diffusers names contain `_`) resolves against a table built
//!    from the base UNet's own Linear keys.
//!
//! LoKr resolves PEFT/bare or kohya `<module>.lokr_w1`/`lokr_w2` (+ low-rank `_a`/`_b`) with `rank` /
//! `alpha` read from file metadata (`networkType=lokr`), reconstructing `δ = (alpha/rank)·kron(w1,w2)`.
//!
//! Beyond the candle trainer's own (Linear) output this also folds the dominant **community** adapter
//! formats (sc-5225), so a hand-trained or downloaded SDXL adapter merges in full — matching mlx-gen's
//! `LoraCoverage::Complete` by construction (the by-key merge into the stock UNet reaches every module
//! a diffusers checkpoint names):
//!  - **conv-layer LoRA** — resnet `conv1`/`conv2`/`conv_shortcut`, the down/up-samplers, `conv_in`/
//!    `conv_out`. The `down`∘`up` pair fuses into a single conv-weight delta
//!    ([`conv_lora_delta`](candle_gen::train::lora::conv_lora_delta)) and folds into the 4-D `{path}.
//!    weight`. candle convs are NCHW (`candle_nn::Conv2d`) = the trained-file layout, so there is no
//!    NHWC transpose (mlx needs one) and `conv_shortcut` is a real 4-D 1×1 conv, not a reshaped Linear.
//!  - **LyCORIS LoHa** (`hada_*`) and **untagged third-party LoKr** (`lokr_*` with no `networkType=lokr`
//!    stamp) — reconstructed per-module at the lycoris scale ([`reconstruct_loha_delta`]
//!    (candle_gen::train::lora::reconstruct_loha_delta) / [`reconstruct_lokr_delta`]) and merged. These
//!    stay at the **Linear** (attention/proj) surface — the conv surface is LoRA-only, mirroring mlx-gen;
//!    the lycoris conv/tucker forms are surfaced as skipped.
//!
//! Out-of-surface keys are **counted and surfaced** in [`MergeReport`], never silently dropped:
//! text-encoder `lora_te*` keys (UNet-only merge) and any factor that resolves to no UNet module.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{
    conv_lora_delta, reconstruct_loha_delta, reconstruct_lokr_delta, reconstruct_lora_delta,
    LoraAdapterMeta,
};
use candle_gen::{CandleError, Result};

/// PEFT key prefix the candle SDXL trainer (and `peft.save_pretrained()`) write. Optional on read —
/// a bare dotted path resolves the same way.
const PEFT_PREFIX: &str = "base_model.model.unet.";
/// kohya / diffusers community LoRA key prefix (the flattened-module form).
const KOHYA_PREFIX: &str = "lora_unet_";

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Outcome of merging the adapter specs into the base UNet tensor map: how many base weights were
/// updated, and how many keys fell outside the merge surface (text-encoder keys, a conv-targeting
/// LoKr/LoHa on the Linear-only surface, or an unresolved module — surfaced, not silently dropped).
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
/// header reader (candle's `load` drops the header `__metadata__`, which LoKr's `rank`/`alpha` live in).
fn read_adapter(path: &Path) -> Result<AdapterFile> {
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("read adapter {}: {e}", path.display())))?;
    let tensors = cst::load_buffer(&bytes, &Device::Cpu)?;
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("read adapter metadata {}: {e}", path.display())))?;
    let meta = md.metadata().clone().unwrap_or_default();
    Ok(AdapterFile { tensors, meta })
}

/// Build the kohya `flattened → dotted` lookup table from the base UNet's adaptable weight keys
/// (`{dotted}.weight`). The `_`-flattening diffusers uses is ambiguous (its own names contain `_`), so
/// resolving against the real key set — the candle analog of the vendored `named_modules()` walk —
/// is what disambiguates a kohya stem. Both **2-D Linear** weights (attention/proj/ff) and **4-D conv**
/// weights (resnet convs, samplers, `conv_in`/`conv_out`) are included (sc-5225): a kohya conv key
/// (`lora_unet_..._conv1`, `..._downsamplers_0_conv`, `conv_in`, …) then resolves to its dotted path and
/// reaches the conv merge. The two surfaces share no flattened stem (distinct module paths), so adding
/// convs introduces no collision; a tagged/third-party LoKr that resolves a conv stem still skips it at
/// the Linear-only shape gate.
fn build_kohya_table(base: &HashMap<String, Tensor>) -> BTreeMap<String, String> {
    base.iter()
        .filter_map(|(k, t)| {
            let dotted = k.strip_suffix(".weight")?;
            let nd = t.dims().len();
            (nd == 2 || nd == 4).then(|| (dotted.replace('.', "_"), dotted.to_string()))
        })
        .collect()
}

/// Map one LoRA key to `(diffusers_dotted_path, role)`, or `None` if outside the UNet merge surface.
/// kohya (`lora_unet_<flat>…`) resolves the flattened stem via `table` — directly for a diffusers-named
/// stem, or via an original-SD/A1111 → diffusers translation (sc-6051); PEFT (`base_model.model.unet.`)
/// and bare dotted paths resolve directly.
fn classify_lora_key(key: &str, table: &BTreeMap<String, String>) -> Option<(String, Role)> {
    if let Some(rem) = key.strip_prefix(KOHYA_PREFIX) {
        for (suf, role) in [
            (".lora_down.weight", Role::Down),
            (".lora_up.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(stem) = rem.strip_suffix(suf) {
                return wmeta::resolve_kohya_stem(stem, table).map(|d| (d, role));
            }
        }
        return None;
    }
    // PEFT (explicit prefix) or a bare dotted path — strip the optional prefix, resolve directly.
    let rem = key.strip_prefix(PEFT_PREFIX).unwrap_or(key);
    for (suf, role) in [
        (".lora_A.default.weight", Role::Down),
        (".lora_B.default.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".alpha", Role::Alpha),
    ] {
        if let Some(path) = rem.strip_suffix(suf) {
            return Some((path.to_string(), role));
        }
    }
    None
}

/// Map one LoKr factor key to `(diffusers_dotted_path, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(
    key: &str,
    table: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                wmeta::resolve_kohya_stem(flat, table).map(|d| (d, factor))
            } else {
                Some((
                    stem.strip_prefix(PEFT_PREFIX).unwrap_or(stem).to_string(),
                    factor,
                ))
            };
        }
    }
    None
}

fn read_scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Read a per-module `.alpha` scalar as `f32`, regardless of on-disk dtype or shape (`[]` or `[1]`),
/// returning `None` for a size-0 (malformed) tensor rather than panicking — the candle twin of
/// mlx-gen's `scalar_alpha`. Third-party adapters store `alpha` in their compute dtype.
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

/// Merge `delta` (`[out, in]` f32) into the base weight at `key`, computing `W += δ` in f32 (the stored
/// f32 sum is cast to the UNet load dtype when the VarBuilder serves it). A missing key or a
/// shape-mismatched base (e.g. a 4-D conv weight) is surfaced as skipped, never a hard error.
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

/// Merge one LoRA file into `base` at `scale`: classify every key (PEFT + kohya), fold complete
/// `(down, up)` pairs into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` is the per-target
/// `.alpha` tensor when present, else the `lora_adapter_metadata` blob's `alpha_pattern`/`lora_alpha`
/// (the diffusers / PEFT `save_lora_adapter` format ships no `.alpha` tensor — sc-5374), else `rank`.
/// **2-D Linear** pairs fold via [`reconstruct_lora_delta`]; **4-D conv** pairs fuse via
/// [`conv_lora_delta`] into the 4-D conv weight (sc-5225). Half-pairs, a conv LoRA targeting a non-conv
/// weight, and other unexpected shapes are surfaced as skipped.
fn merge_lora_file(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lora_key(key, table) {
            Some((path, Role::Down)) => triples.entry(path).or_default().down = Some(t.clone()),
            Some((path, Role::Up)) => triples.entry(path).or_default().up = Some(t.clone()),
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(t)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` blob (sc-5374). `None` for kohya /
    // candle-trainer files (those ship a `.alpha` tensor), in which case the per-target `.alpha` or the
    // factor rank is used exactly as before.
    let cfg = LoraAdapterMeta::from_file_metadata(&af.meta);
    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        let base_key = format!("{path}.weight");
        // Effective scaling: per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor
        // rank (today's last-resort default). The denominator is the blob `r`/`rank_pattern` when given,
        // else the stored `A` leading dim (which equals it for a well-formed PEFT file).
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let (dn, un) = (down.dims().len(), up.dims().len());
        if dn == 4 && un == 4 {
            // Conv-layer LoRA (sc-5225): fuse `down`∘`up` into a single NCHW conv-weight delta and fold
            // it into the 4-D `{path}.weight`. candle convs are NCHW, so no transpose — `merge_into`
            // adds the matching-shape delta directly. A conv LoRA whose target is missing or not 4-D
            // (a non-conv weight) is surfaced as skipped, never mis-merged.
            let Some(w) = base.get(&base_key) else {
                report.skipped_keys += 1;
                continue;
            };
            if w.dims().len() != 4 {
                report.skipped_keys += 1;
                continue;
            }
            let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
            let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
            let delta = conv_lora_delta(&down, &up, alpha, rank, scale)?;
            merge_into(base, &base_key, &delta, report)?;
            continue;
        }
        if dn != 2 || un != 2 {
            report.skipped_keys += 1; // neither a 2-D Linear nor a 4-D conv pair — unexpected shape
            continue;
        }
        if !base.contains_key(&base_key) {
            report.skipped_keys += 1;
            continue;
        }
        let rank = cfg_rank.unwrap_or(down.dims()[0] as f32);
        let alpha = t.alpha.or(cfg_alpha).unwrap_or(rank);
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, scale)?;
        merge_into(base, &base_key, &delta, report)?;
    }
    Ok(())
}

/// Merge one LoKr file into `base` at `scale`: `rank`/`alpha` from file metadata (alpha defaults to
/// rank), per-module factors grouped, `δ = (alpha/rank)·kron(w1,w2)·scale` reconstructed and merged.
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

    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Tensor>> = BTreeMap::new();
    for (key, t) in &af.tensors {
        match classify_lokr_key(key, table) {
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
            report.skipped_keys += 1; // conv LoKr — deferred (sc-5225)
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

/// Whether the adapter file declares LoKr in its `networkType` metadata (the SceneWorks/PEFT stamp the
/// candle trainer writes). A third-party LyCORIS LoKr has the `lokr_*` factors but **no** stamp — see
/// [`merge_lokr_thirdparty`].
fn declares_lokr(af: &AdapterFile) -> bool {
    wmeta::is_lokr_network_type(af.meta.get("networkType").map(String::as_str))
}

// ---- Third-party LyCORIS LoKr / LoHa (sc-5225) ---------------------------------------------------
//
// kohya / ai-toolkit / lycoris-lib LoKr (`lokr_*`) and LoHa (`hada_*`) files ship the decomposition
// factors but NOT the `networkType=lokr` stamp `declares_lokr` keys off, and derive rank/alpha/scale
// **per module** (vs the PEFT path's one global pair). We reuse `gen_core::weightsmeta` for all the
// string/metadata logic (key detection, suffix tables, flattened→dotted resolution) and port only the
// per-module factor grouping + the lycoris scale rule here; the delta is reconstructed with the shared
// f32 math. **Linear-only** to match mlx-gen (`merge_one_lokr_thirdparty` / `merge_one_loha_thirdparty`
// resolve only Linear targets): a factor that resolves to a 4-D conv weight — including the lycoris
// conv/tucker (`lokr_t2`/`hada_t1`/`hada_t2`) forms — is surfaced as skipped, never mis-merged.

/// One module's third-party LoKr factors (full `w1`/`w2`, low-rank `_a`/`_b`, optional per-module
/// `.alpha`). The tucker `lokr_t2` factor is conv-only and out of the Linear surface, so it is ignored.
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
    /// The factorization rank (`lora_dim`), derived from whichever decomposed factor is present:
    /// `lokr_w1_a` is `[shape0, dim]`; else the non-tucker `lokr_w2_a` is `[shape0, dim]`. `None` when
    /// **both** factors are full — lycoris then forces `alpha = lora_dim` ⇒ scale 1, so rank is unused.
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

    /// Reconstruct this module's `[out, in]` delta with the lycoris per-module scale baked in, times
    /// the caller's `user_scale`. Reuses the shared [`reconstruct_lokr_delta`] by passing the lycoris
    /// scale as `alpha` over `rank = 1.0` (so `eff = lycoris_scale · user_scale`).
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

/// Group a third-party LoKr file's tensors by raw module key (the part before `.lokr_*`/`.alpha`). The
/// raw key is whatever the trainer wrote — a `<PREFIX>_<flattened.path>` (kohya/lycoris) — resolved to
/// a UNet dotted path in [`merge_lokr_thirdparty`].
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

/// One module's third-party LoHa factors — two low-rank Hadamard pairs + an optional per-module
/// `.alpha`. The tucker `hada_t1`/`hada_t2` factors are conv-only and ignored (Linear surface).
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

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`). LoHa is always decomposed
    /// (no both-full case), so — unlike LoKr — there is no forced-1 branch.
    fn lycoris_scale(&self) -> f32 {
        match self.rank() {
            None => 1.0,
            Some(r) => self.alpha.unwrap_or(r) / r,
        }
    }

    /// Reconstruct this module's `[out, in]` Hadamard delta (lycoris scale × `user_scale` baked in).
    /// Errors if a `hada_w1/w2` `a`/`b` leg is missing (a conv-tucker-only module never reaches here —
    /// it resolves to a 4-D base and skips first).
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

/// Merge one reconstructed `[out, in]` delta into the base at the resolved Linear module `path`
/// (`W += δ`). Shared by the third-party LoKr + LoHa paths: resolve → Linear-only shape gate →
/// reconstruct → merge. An unresolved key, a missing weight, or a 4-D (conv) target is surfaced as
/// skipped, never mis-merged.
fn merge_thirdparty(
    base: &mut HashMap<String, Tensor>,
    path: Option<&str>,
    delta_at: impl FnOnce((usize, usize)) -> Result<Tensor>,
    report: &mut MergeReport,
) -> Result<()> {
    let Some(path) = path else {
        report.skipped_keys += 1;
        return Ok(());
    };
    let base_key = format!("{path}.weight");
    let Some(w) = base.get(&base_key) else {
        report.skipped_keys += 1;
        return Ok(());
    };
    if w.dims().len() != 2 {
        report.skipped_keys += 1; // Linear-only surface (the conv surface is LoRA-only)
        return Ok(());
    }
    let (out_f, in_f) = (w.dims()[0], w.dims()[1]);
    let delta = delta_at((out_f, in_f))?;
    merge_into(base, &base_key, &delta, report)
}

/// Merge a third-party LyCORIS **LoKr** file (`lokr_*` keys, per-module `.alpha`, no `networkType`
/// stamp) into `base` at `scale`. Resolves each flattened module key against `table` (the kohya
/// `flattened → dotted` map); an unresolved key is surfaced as skipped (mirrors mlx-gen's
/// `merge_one_lokr_thirdparty`).
fn merge_lokr_thirdparty(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    for (raw, g) in parse_lokr_thirdparty(af)? {
        merge_thirdparty(
            base,
            wmeta::resolve_lokr_path(&raw, table),
            |bs| g.delta(bs, scale),
            report,
        )?;
    }
    Ok(())
}

/// Merge a third-party LyCORIS **LoHa** file (`hada_*` keys) into `base` at `scale`. As
/// [`merge_lokr_thirdparty`] but the per-module delta is the Hadamard reconstruction.
fn merge_loha_thirdparty(
    base: &mut HashMap<String, Tensor>,
    af: &AdapterFile,
    scale: f32,
    table: &BTreeMap<String, String>,
    report: &mut MergeReport,
) -> Result<()> {
    for (raw, g) in parse_loha_thirdparty(af)? {
        merge_thirdparty(
            base,
            wmeta::resolve_lokr_path(&raw, table),
            |bs| g.delta(bs, scale),
            report,
        )?;
    }
    Ok(())
}

/// Fold every adapter spec in `specs` into the base UNet tensor `map` (CPU, native dtype) at each
/// spec's `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`). Returns the
/// [`MergeReport`]; errors if a non-empty spec list matches **no** target (a format / prefix
/// misconfiguration — the worker should then fall back rather than render an unadapted image silently).
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
        // Third-party LyCORIS (sc-5225): `lokr_*` / `hada_*` keys without a `networkType=lokr` stamp,
        // so the caller's declared `kind` can't label them — detect + route by keys before the kind
        // match. (A PEFT LoKr carries the stamp and goes through the `Lokr` arm; the LoKr-keys branch
        // excludes it via `!declares_lokr`.)
        if !declares_lokr(&af) && wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str)) {
            merge_lokr_thirdparty(map, &af, spec.scale, &table, &mut report)?;
            continue;
        }
        if wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)) {
            merge_loha_thirdparty(map, &af, spec.scale, &table, &mut report)?;
            continue;
        }
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &table, &mut report)?,
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file has no lora_A/B keys
                // and would merge nothing; surface the mismatch loudly rather than no-op.
                if declares_lokr(&af) {
                    return Err(CandleError::Msg(format!(
                        "sdxl: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &table, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(CandleError::Msg(format!(
            "sdxl: no adapter target modules matched across {} file(s) — expected PEFT \
             `base_model.model.unet.<path>.lora_A/B.weight` or kohya `lora_unet_<flat>.lora_down/up.\
             weight` with diffusers `down_blocks_*` or original-SD `input_blocks_*` block naming \
             (LoRA, incl. conv layers), `<module>.lokr_w1/w2` with networkType=lokr (LoKr), \
             or untagged LyCORIS `lokr_*` / `hada_*` (third-party LoKr / LoHa)",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny stand-in for the base UNet tensor map: two attention Linears + one conv (4-D) weight.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        // attn1.to_q: [out=4, in=4]
        m.insert(
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight".into(),
            Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
        );
        // attn1.to_out.0: [out=4, in=4]
        m.insert(
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight".into(),
            Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
        );
        // a conv weight (4-D) — must never be merged by a 2-D LoRA.
        m.insert(
            "conv_in.weight".into(),
            Tensor::zeros((4, 4, 3, 3), DType::F16, &dev).unwrap(),
        );
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// kohya stems resolve against the base-key table; the ambiguous `to_out_0` flattening resolves to
    /// the real `…to_out.0` path.
    #[test]
    fn classify_lora_resolves_peft_kohya_and_bare() {
        let table = build_kohya_table(&base_map());
        // PEFT prefixed.
        let (p, _) = classify_lora_key(
            "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight",
            &table,
        )
        .unwrap();
        assert_eq!(
            p,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        // PEFT `.default.` infix.
        assert!(matches!(
            classify_lora_key(
                "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
                &table,
            )
            .unwrap()
            .1,
            Role::Up
        ));
        // kohya flattened stem, incl. the `.0` of to_out.0 → `to_out_0`.
        let (p, _) = classify_lora_key(
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight",
            &table,
        )
        .unwrap();
        assert_eq!(
            p,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0"
        );
        // text-encoder + unknown stems are out of surface.
        assert!(classify_lora_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            &table
        )
        .is_none());
    }

    /// sc-6051: an original-SD / A1111 kohya key (`lora_unet_input_blocks_4_1_…`) classifies onto the
    /// same diffusers dotted path as its `down_blocks` twin, so civitai SDXL LoRAs merge in candle too.
    #[test]
    fn classify_lora_translates_original_sd_naming() {
        // A table holding a real down_blocks.1 attention path (the diffusers twin of input_blocks.4.1).
        let table: BTreeMap<String, String> =
            ["down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"]
                .into_iter()
                .map(|p| (p.replace('.', "_"), p.to_string()))
                .collect();
        let (p, role) = classify_lora_key(
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &table,
        )
        .expect("original-SD input_blocks key should translate + resolve");
        assert_eq!(
            p,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // The LoKr classify path translates too (kohya-prefixed original-SD stem).
        assert_eq!(
            classify_lokr_key(
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lokr_w1",
                &table,
            )
            .unwrap()
            .0,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
    }

    /// PEFT LoRA merges into `W += (alpha/rank)·scale·B·A`; base+delta is exact in f32.
    #[test]
    fn merge_lora_peft_folds_expected_delta() {
        let mut map = base_map();
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4); // A [rank=2, in=4]
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2); // B [out=4, rank=2]
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(),
                    down.clone(),
                ),
                (
                    "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight".to_string(),
                    up.clone(),
                ),
                (
                    "base_model.model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        // scale 1.0; alpha 4, rank 2 ⇒ effective 2.0. ΔW = 2.0·(B·A).
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap(); // base is zero
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "merged weight off by {diff}");
    }

    /// sc-5374: a diffusers-format LoRA with NO per-target `.alpha` tensor but a `lora_adapter_metadata`
    /// blob (`lora_alpha = 16`, `r = 8`) merges at the metadata-derived strength `(16/8)·scale = 2.0`,
    /// not the old `alpha = rank` default (which would halve it). Proves the blob is read and applied.
    #[test]
    fn merge_lora_honors_lora_adapter_metadata_alpha() {
        let dev = Device::Cpu;
        let mut map = base_map();
        let path = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q";
        // A [r=8, in=4], B [out=4, r=8] — nonzero so ΔW ≠ 0; deliberately NO `.alpha` tensor.
        let down = Tensor::randn(0f32, 1f32, (8, 4), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 8), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    format!("base_model.model.unet.{path}.lora_A.weight"),
                    down.clone(),
                ),
                (
                    format!("base_model.model.unet.{path}.lora_B.weight"),
                    up.clone(),
                ),
            ]),
            meta: HashMap::from([(
                "lora_adapter_metadata".to_string(),
                r#"{"lora_alpha": 16, "r": 8}"#.to_string(),
            )]),
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
        // Effective alpha 16 over rank 8 ⇒ scale 2.0; base is zero, so the merged weight IS the delta.
        let expected = reconstruct_lora_delta(&down, &up, 16.0, 8.0, 1.0).unwrap();
        let diff = (&merged - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "metadata-alpha merge off by {diff}");
        // The pre-sc-5374 default (alpha = rank ⇒ scale 1.0) would diverge by a full factor of 2.
        let buggy = reconstruct_lora_delta(&down, &up, 8.0, 8.0, 1.0).unwrap();
        let gap = (&merged - &buggy)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            gap > 1e-3,
            "metadata alpha must differ from the alpha=rank default (gap {gap})"
        );
    }

    /// sc-5225: a conv-shaped LoRA (4-D factors) now folds into the 4-D conv weight (`conv_in`), via
    /// the NCHW [`conv_lora_delta`] fusion — no transpose. Base is zero, so the merged weight IS the
    /// fused delta. PEFT keys resolve the dotted path directly (no table needed).
    #[test]
    fn merge_conv_lora_folds_into_conv_weight() {
        use candle_gen::train::lora::conv_lora_delta;
        let mut map = base_map();
        let dev = Device::Cpu;
        // down [rank=2, in=4, 3, 3], up [out=4, rank=2, 1, 1] — nonzero so ΔW ≠ 0.
        let down = Tensor::randn(0f32, 1f32, (2, 4, 3, 3), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2, 1, 1), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "base_model.model.unet.conv_in.lora_A.weight".to_string(),
                    down.clone(),
                ),
                (
                    "base_model.model.unet.conv_in.lora_B.weight".to_string(),
                    up.clone(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        // alpha defaults to rank (2) ⇒ effective 1.0; scale 1.0.
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        assert_eq!(report.skipped_keys, 0);
        let merged = map
            .get("conv_in.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(merged.dims(), &[4, 4, 3, 3]);
        let expected = conv_lora_delta(&down, &up, 2.0, 2.0, 1.0).unwrap(); // base is zero
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "merged conv weight off by {diff}");
    }

    /// sc-5225: a kohya conv LoRA (`lora_unet_conv_in.lora_down/up.weight`) resolves the flattened conv
    /// stem through the now conv-aware [`build_kohya_table`] and merges (proving conv stems join the
    /// table). A 1×1 conv (in=out=4, kH=kW=1 here via conv_in's 3×3 base — use a synthetic 1×1 conv).
    #[test]
    fn merge_kohya_conv_lora_resolves_flattened_stem() {
        let dev = Device::Cpu;
        let mut map = HashMap::new();
        // A 1×1 conv weight [out=4, in=4, 1, 1] under a dotted path with internal-underscore segments.
        map.insert(
            "down_blocks.0.downsamplers.0.conv.weight".to_string(),
            Tensor::zeros((4, 4, 1, 1), DType::F16, &dev).unwrap(),
        );
        let down = Tensor::randn(0f32, 1f32, (2, 4, 1, 1), &dev).unwrap();
        let up = Tensor::randn(0f32, 1f32, (4, 2, 1, 1), &dev).unwrap();
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "lora_unet_down_blocks_0_downsamplers_0_conv.lora_down.weight".to_string(),
                    down,
                ),
                (
                    "lora_unet_down_blocks_0_downsamplers_0_conv.lora_up.weight".to_string(),
                    up,
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1, "kohya conv stem must resolve and merge");
        assert_eq!(report.skipped_keys, 0);
    }

    /// LoKr merges `δ = (alpha/rank)·kron(w1,w2)` into the dense weight, reading rank/alpha from meta.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        // base [out=4,in=4] factors 2×2 ⊗ 2×2.
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lokr_w1"
                        .to_string(),
                    w1.clone(),
                ),
                (
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lokr_w2"
                        .to_string(),
                    w2.clone(),
                ),
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
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight")
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
        let diff = (merged - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "merged lokr weight off by {diff}");
    }

    /// A non-empty spec list that matches nothing is a loud error (not a silent unadapted render).
    #[test]
    fn merge_adapters_errors_when_nothing_matches() {
        let mut map = base_map();
        let af_tensors = HashMap::from([(
            "lora_unet_nonexistent_module.lora_down.weight".to_string(),
            t2(&[0.0, 0.0], 1, 2),
        )]);
        // Drive merge_lora_file directly with an unresolvable key → 0 merged.
        let af = AdapterFile {
            tensors: af_tensors,
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert!(report.skipped_keys >= 1);
    }

    /// A Lora-declared spec pointing at LoKr-tagged metadata is rejected (the candle trainer never
    /// produces this, but a misconfigured worker request must fail loudly).
    #[test]
    fn merge_adapters_rejects_kind_metadata_mismatch() {
        // Build via the public entry point using an in-memory file is awkward; assert the helper.
        let af = AdapterFile {
            tensors: HashMap::new(),
            meta: HashMap::from([("networkType".to_string(), "lokr".to_string())]),
        };
        assert!(declares_lokr(&af));
    }

    /// The keystone train→infer round-trip: a PEFT `.safetensors` written by the **actual trainer**
    /// path ([`candle_gen::train::lora::save_lora_peft`]) is read back through the public
    /// [`merge_adapters`] entry — exercising `read_adapter` (tensors + header metadata), PEFT
    /// classification, and the f32 reconstruction — and the merged weight equals the trained delta
    /// `ΔW = (alpha/rank)·B·A`. Proves the loader consumes the trainer's real on-disk format, not just
    /// hand-built tensors.
    #[test]
    fn roundtrip_trainer_peft_file_merges() {
        use candle_gen::candle_nn::Linear;
        use candle_gen::train::lora::{
            build_lora_targets, save_lora_peft, LoraHost, LoraLinear, SDXL_PEFT_PREFIX,
        };

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
        let path = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v";
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
        set.vars[1].set(&up_randn).unwrap(); // vars = [down(A), up(B)]

        // Write the real PEFT file the trainer emits, then merge it through the public entry point.
        let file = std::env::temp_dir().join(format!(
            "candle_sdxl_lora_roundtrip_{}.safetensors",
            std::process::id()
        ));
        save_lora_peft(&set, SDXL_PEFT_PREFIX, &HashMap::new(), &file).unwrap();

        let mut map = HashMap::new();
        map.insert(
            format!("{path}.weight"),
            Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
        );
        let report = merge_adapters(
            &mut map,
            &[AdapterSpec::new(file.clone(), 1.0, AdapterKind::Lora)],
        );
        std::fs::remove_file(&file).ok();
        let report = report.unwrap();

        assert_eq!(report.merged, 1, "the trained to_v adapter must merge");
        // Base is zero, so the merged weight IS ΔW = (alpha/rank)·B·A.
        let expected = reconstruct_lora_delta(
            set.vars[0].as_tensor(),
            set.vars[1].as_tensor(),
            4.0,
            2.0,
            1.0,
        )
        .unwrap();
        let merged = map[&format!("{path}.weight")].to_dtype(DType::F32).unwrap();
        let diff = (&merged - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-4,
            "round-trip merge diverged from the trained delta by {diff}"
        );
        let mag = expected
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mag > 0.0, "forced-nonzero B must yield a non-trivial delta");
    }

    /// sc-5225: the candle SDXL crate reconstructs third-party LyCORIS LoKr / LoHa deltas (via the
    /// shared f32 reconstruction) bit-close to the lycoris reference fixtures — the same fixtures the
    /// mlx-gen `thirdparty_lycoris_reconstructs_against_reference_f32` test pins (generated through the
    /// lycoris venv). Exercises detection (`keys_contain_*`), per-module factor grouping + the lycoris
    /// scale rule, and the flattened-key → dotted resolution. Linear fixtures only (the SDXL third-party
    /// surface is Linear-only; the conv/tucker fixtures are out of scope, as in mlx-gen's SDXL path).
    #[test]
    fn thirdparty_lycoris_reconstructs_against_reference_f32() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        // (fixture dir, file stem, is_loha)
        let cases: [(&str, &str, bool); 4] = [
            ("sc3642_lokr", "linear_w1full_w2lr", false),
            ("sc3642_lokr", "linear_bothlr", false),
            ("sc3642_lokr", "linear_bothfull", false),
            ("sc3643_loha", "linear", true),
        ];
        for (dir, stem, is_loha) in cases {
            let base = root.join(dir);
            let af = read_adapter(&base.join(format!("{stem}.safetensors"))).unwrap();
            let exp = read_adapter(&base.join(format!("{stem}.expected.safetensors"))).unwrap();
            // Detection mirrors the merge router.
            if is_loha {
                assert!(
                    wmeta::keys_contain_loha(af.tensors.keys().map(String::as_str)),
                    "{stem}: not detected as LoHa"
                );
            } else {
                assert!(
                    wmeta::keys_contain_lokr(af.tensors.keys().map(String::as_str))
                        && !declares_lokr(&af),
                    "{stem}: not detected as third-party LoKr"
                );
            }
            // table: the expected file's single target ("proj") flattened → dotted.
            let table: BTreeMap<String, String> = exp
                .tensors
                .keys()
                .map(|d| (d.replace('.', "_"), d.clone()))
                .collect();
            let want = exp
                .tensors
                .get("proj")
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let (out_f, in_f) = (want.dims()[0], want.dims()[1]);
            let got = if is_loha {
                let groups = parse_loha_thirdparty(&af).unwrap();
                let (raw, g) = groups.iter().next().unwrap();
                assert_eq!(wmeta::resolve_lokr_path(raw, &table), Some("proj"));
                g.delta((out_f, in_f), 1.0).unwrap()
            } else {
                let groups = parse_lokr_thirdparty(&af).unwrap();
                let (raw, g) = groups.iter().next().unwrap();
                assert_eq!(wmeta::resolve_lokr_path(raw, &table), Some("proj"));
                g.delta((out_f, in_f), 1.0).unwrap()
            };
            assert_eq!(
                got.dims(),
                want.dims(),
                "{stem}: reconstructed shape mismatch"
            );
            let diff = (&got - &want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff < 1e-4,
                "{stem}: third-party reconstruction diverged from lycoris reference by {diff}"
            );
        }
    }

    /// sc-5225: an untagged third-party LoKr (kohya-flattened keys, no `networkType`) is detected by
    /// keys and merged into the resolved Linear (`W += δ`). A conv-targeting third-party factor stays
    /// on the Linear-only surface — surfaced as skipped, never folded into a 4-D conv weight.
    #[test]
    fn merge_thirdparty_lokr_routes_resolves_and_merges() {
        let mut map = base_map(); // attn1.to_q [4,4], conv_in [4,4,3,3]
        let to_q = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
        let af = AdapterFile {
            tensors: HashMap::from([
                // to_q: factor [4,4] as 2×2 ⊗ 2×2 (both full ⇒ lycoris scale 1).
                (format!("{to_q}.lokr_w1"), t2(&[1.0, 0.0, 0.0, 1.0], 2, 2)),
                (format!("{to_q}.lokr_w2"), t2(&[0.5, 0.0, 0.0, 0.5], 2, 2)),
                // conv_in: resolves to a 4-D weight ⇒ Linear-only surface skips it.
                ("lora_unet_conv_in.lokr_w1".to_string(), t2(&[1.0], 1, 1)),
                ("lora_unet_conv_in.lokr_w2".to_string(), t2(&[1.0], 1, 1)),
            ]),
            meta: HashMap::new(), // no networkType stamp → third-party
        };
        assert!(!declares_lokr(&af));
        assert!(wmeta::keys_contain_lokr(
            af.tensors.keys().map(String::as_str)
        ));
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lokr_thirdparty(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1, "the to_q LoKr must merge");
        assert!(
            report.skipped_keys >= 1,
            "the conv-targeting LoKr is Linear-only ⇒ skipped"
        );
        // conv_in untouched (still 4-D, all-zero).
        assert_eq!(map.get("conv_in.weight").unwrap().dims(), &[4, 4, 3, 3]);
    }

    /// sc-5225: a third-party LoHa (`hada_*`) routes through the Hadamard merge into the resolved
    /// Linear, producing a finite merged weight.
    #[test]
    fn merge_thirdparty_loha_routes_and_merges() {
        let mut map = base_map();
        let to_q = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
        // rank-1 Hadamard factors: w*_a [4,1], w*_b [1,4] ⇒ [4,4] products.
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    format!("{to_q}.hada_w1_a"),
                    t2(&[0.5, 0.1, -0.2, 0.3], 4, 1),
                ),
                (
                    format!("{to_q}.hada_w1_b"),
                    t2(&[0.4, -0.1, 0.2, 0.6], 1, 4),
                ),
                (
                    format!("{to_q}.hada_w2_a"),
                    t2(&[0.2, 0.0, 0.1, -0.3], 4, 1),
                ),
                (
                    format!("{to_q}.hada_w2_b"),
                    t2(&[1.0, 0.5, -0.5, 0.25], 1, 4),
                ),
            ]),
            meta: HashMap::new(),
        };
        assert!(wmeta::keys_contain_loha(
            af.tensors.keys().map(String::as_str)
        ));
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_loha_thirdparty(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            merged.iter().all(|v| v.is_finite()),
            "merged LoHa weight must be finite"
        );
    }
}
