//! Z-Image inference-side adapter merge (sc-5166) — load a trained LoRA/LoKr `.safetensors` and fold
//! its delta into the dense DiT (`transformer/`) weights **before** the stock candle-transformers
//! `ZImageTransformer2DModel` is built. The candle twin of `mlx-gen-z-image::adapters`, and the
//! closing half of the native-trainer loop: a LoRA produced by [`crate::training`]'s Z-Image trainer
//! now actually loads in candle inference. Structurally identical to `candle-gen-sdxl::adapters`,
//! re-homed onto the DiT key namespace.
//!
//! **Merge, don't residual** (same rationale as SDXL): inference has no need to keep the factors
//! trainable, so it folds `W += δ` into the dense weight and reproduces the merged-weight forward
//! exactly. The delta is reconstructed with the **same** f32 math the trainer's forward uses
//! ([`reconstruct_lora_delta`] / [`reconstruct_lokr_delta`]), so a candle-trained adapter round-trips.
//!
//! **Merge at the safetensors-key level.** The stock DiT reads its `transformer/` keys 1:1, so
//! `{path}.weight` is a valid base key for every Linear an adapter targets — the attention
//! projections (`to_q`/`to_k`/`to_v`/`to_out.0`) the trainer adapts. The DiT family writes **bare
//! dotted** PEFT keys (no `base_model.model.unet.` prefix — that is SDXL-specific); on read we also
//! tolerate the common community prefixes ([`PEFT_PREFIXES`]) and a kohya `lora_transformer_<flat>`
//! flattening resolved against the base key set.
//!
//! Out-of-surface keys are **counted and surfaced** in [`MergeReport`], never silently dropped:
//! text-encoder LoRAs (DiT-only merge) and conv-shaped (4-D) factors (the trainer adapts Linear
//! attention projections only).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta};
use candle_gen::{CandleError, Result};

/// PEFT key prefixes tolerated on read, longest-first. The candle Z-Image trainer writes **bare**
/// dotted paths (no prefix), but community adapters and `peft.save_pretrained()` wrap the DiT under
/// one of these; stripping them yields the same dotted module path. A key matching none is taken
/// as-is (bare).
const PEFT_PREFIXES: [&str; 4] = [
    "base_model.model.transformer.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
];

/// kohya / community flattened-module LoRA prefix (the DiT analog of SDXL's `lora_unet_`). The
/// `_`-flattened stem is resolved against the base DiT key table (diffusers names contain `_`, so the
/// flattening is ambiguous without it).
const KOHYA_PREFIX: &str = "lora_transformer_";

/// LoKr per-module factor suffixes, longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Outcome of merging the adapter specs into the base DiT tensor map: how many base weights were
/// updated, and how many keys fell outside the merge surface (text-encoder / conv-shaped / unresolved
/// — surfaced, not silently dropped).
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

/// Strip the longest matching PEFT prefix, or return the key unchanged (bare dotted path).
fn strip_peft_prefix(key: &str) -> &str {
    for p in PEFT_PREFIXES {
        if let Some(rem) = key.strip_prefix(p) {
            return rem;
        }
    }
    key
}

/// Build the kohya `flattened → dotted` lookup table from the base DiT's 2-D Linear weight keys
/// (`{dotted}.weight`). The `_`-flattening is ambiguous (diffusers names contain `_`), so resolving
/// against the real key set is what disambiguates a kohya stem.
fn build_kohya_table(base: &HashMap<String, Tensor>) -> BTreeMap<String, String> {
    base.iter()
        .filter_map(|(k, t)| {
            let dotted = k.strip_suffix(".weight")?;
            (t.dims().len() == 2).then(|| (dotted.replace('.', "_"), dotted.to_string()))
        })
        .collect()
}

/// Map one LoRA key to `(dit_dotted_path, role)`, or `None` if outside the DiT merge surface. kohya
/// (`lora_transformer_<flat>…`) resolves the flattened stem via `table`; PEFT/bare resolve directly
/// after the optional prefix strip.
fn classify_lora_key(key: &str, table: &BTreeMap<String, String>) -> Option<(String, Role)> {
    if let Some(rem) = key.strip_prefix(KOHYA_PREFIX) {
        for (suf, role) in [
            (".lora_down.weight", Role::Down),
            (".lora_up.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(stem) = rem.strip_suffix(suf) {
                return table.get(stem).map(|d| (d.clone(), role));
            }
        }
        return None;
    }
    let rem = strip_peft_prefix(key);
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

/// Map one LoKr factor key to `(dit_dotted_path, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(
    key: &str,
    table: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                table.get(flat).map(|d| (d.clone(), factor))
            } else {
                Some((strip_peft_prefix(stem).to_string(), factor))
            };
        }
    }
    None
}

fn read_scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Merge `delta` (`[out, in]` f32) into the base weight at `key`, computing `W += δ` in f32 (the
/// stored f32 sum is cast to the DiT load dtype when the VarBuilder serves it). A missing key or a
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

/// Merge one LoRA file into `base` at `scale`: classify every key, fold complete `(down, up)` pairs
/// into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` is the per-target `.alpha` (default
/// `rank`). Half-pairs and conv-shaped (4-D) factors are surfaced as skipped.
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

    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            report.skipped_keys += 1; // half-pair (partner targeted a non-routable module)
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // conv-shaped LoRA — out of surface
            continue;
        }
        let base_key = format!("{path}.weight");
        if !base.contains_key(&base_key) {
            report.skipped_keys += 1;
            continue;
        }
        let rank = down.dims()[0] as f32;
        let alpha = t.alpha.unwrap_or(rank);
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
            report.skipped_keys += 1; // conv LoKr — out of surface
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

/// Whether the adapter file declares LoKr in its `networkType` metadata.
fn declares_lokr(af: &AdapterFile) -> bool {
    af.meta.get("networkType").map(String::as_str) == Some("lokr")
}

/// Fold every adapter spec in `specs` into the base DiT tensor `map` (CPU, native dtype) at each
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
        match spec.kind {
            AdapterKind::Lokr => merge_lokr_file(map, &af, spec.scale, &table, &mut report)?,
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file has no lora_A/B keys
                // and would merge nothing; surface the mismatch loudly rather than no-op.
                if declares_lokr(&af) {
                    return Err(CandleError::Msg(format!(
                        "z_image: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &table, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(CandleError::Msg(format!(
            "z_image: no adapter target modules matched across {} file(s) — expected bare/PEFT \
             `<path>.lora_A/B.weight` (LoRA) or `<module>.lokr_w1/w2` with networkType=lokr (LoKr) \
             over the DiT attention projections. Conv-layer / text-encoder adapters are out of \
             surface",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny stand-in for the base DiT tensor map: two attention Linears + one conv (4-D) weight.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        m.insert(
            "layers.0.attention.to_q.weight".into(),
            Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
        );
        m.insert(
            "layers.0.attention.to_out.0.weight".into(),
            Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
        );
        // a conv weight (4-D) — must never be merged by a 2-D LoRA.
        m.insert(
            "vae_like_conv.weight".into(),
            Tensor::zeros((4, 4, 3, 3), DType::BF16, &dev).unwrap(),
        );
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// Bare dotted (the trainer's format), prefixed PEFT, and kohya flattened all resolve to the same
    /// dotted DiT path.
    #[test]
    fn classify_lora_resolves_bare_peft_and_kohya() {
        let table = build_kohya_table(&base_map());
        // bare dotted (what `save_lora_peft(set, "", …)` writes for the DiT).
        let (p, _) = classify_lora_key("layers.0.attention.to_q.lora_A.weight", &table).unwrap();
        assert_eq!(p, "layers.0.attention.to_q");
        // PEFT-prefixed (community / peft.save_pretrained).
        let (p, r) =
            classify_lora_key("transformer.layers.0.attention.to_q.lora_B.weight", &table).unwrap();
        assert_eq!(p, "layers.0.attention.to_q");
        assert!(matches!(r, Role::Up));
        // `.default.` infix.
        assert!(matches!(
            classify_lora_key(
                "base_model.model.transformer.layers.0.attention.to_q.lora_B.default.weight",
                &table,
            )
            .unwrap()
            .1,
            Role::Up
        ));
        // kohya flattened stem, incl. the `.0` of to_out.0 → `to_out_0`.
        let (p, _) = classify_lora_key(
            "lora_transformer_layers_0_attention_to_out_0.lora_down.weight",
            &table,
        )
        .unwrap();
        assert_eq!(p, "layers.0.attention.to_out.0");
        // text-encoder keys are out of surface.
        assert!(classify_lora_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            &table
        )
        .is_none());
    }

    /// Bare-dotted LoRA merges into `W += (alpha/rank)·scale·B·A`; base+delta is exact in f32.
    #[test]
    fn merge_lora_bare_folds_expected_delta() {
        let mut map = base_map();
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4); // A [rank=2, in=4]
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2); // B [out=4, rank=2]
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "layers.0.attention.to_q.lora_A.weight".to_string(),
                    down.clone(),
                ),
                (
                    "layers.0.attention.to_q.lora_B.weight".to_string(),
                    up.clone(),
                ),
                (
                    "layers.0.attention.to_q.alpha".to_string(),
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
            .get("layers.0.attention.to_q.weight")
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
        assert!(diff < 1e-2, "merged weight off by {diff}"); // bf16 base round-trip tolerance
    }

    /// A conv-shaped LoRA (4-D factors) is surfaced as skipped, never merged into the conv weight.
    #[test]
    fn merge_skips_conv_shaped_lora() {
        let mut map = base_map();
        let dev = Device::Cpu;
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "vae_like_conv.lora_A.weight".to_string(),
                    Tensor::zeros((2, 4, 3, 3), DType::F32, &dev).unwrap(),
                ),
                (
                    "vae_like_conv.lora_B.weight".to_string(),
                    Tensor::zeros((4, 2, 1, 1), DType::F32, &dev).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert_eq!(report.skipped_keys, 1); // the (down,up) pair, dropped as a conv shape
    }

    /// LoKr merges `δ = (alpha/rank)·kron(w1,w2)` into the dense weight, reading rank/alpha from meta.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                ("layers.0.attention.to_q.lokr_w1".to_string(), w1.clone()),
                ("layers.0.attention.to_q.lokr_w2".to_string(), w2.clone()),
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
            .get("layers.0.attention.to_q.weight")
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
        assert!(diff < 1e-2, "merged lokr weight off by {diff}");
    }

    /// The keystone train→infer round-trip: a PEFT `.safetensors` written by the **actual trainer**
    /// path ([`candle_gen::train::lora::save_lora_peft`] with the DiT's empty prefix) is read back
    /// through the public [`merge_adapters`] entry, and the merged weight equals the trained delta.
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
        let path = "layers.3.attention.to_v";
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

        // Write the real PEFT file the DiT trainer emits (empty prefix → bare dotted keys), then
        // merge it through the public entry point.
        let file = std::env::temp_dir().join(format!(
            "candle_zimage_lora_roundtrip_{}.safetensors",
            std::process::id()
        ));
        save_lora_peft(&set, "", &HashMap::new(), &file).unwrap();

        let mut map = HashMap::new();
        map.insert(
            format!("{path}.weight"),
            Tensor::zeros((4, 4), DType::BF16, &dev).unwrap(),
        );
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
        let diff = (&merged - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-2,
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

    /// A non-empty spec list that matches nothing is a loud error (not a silent unadapted render).
    #[test]
    fn merge_lora_file_unresolvable_key_merges_nothing() {
        let mut map = base_map();
        let af = AdapterFile {
            tensors: HashMap::from([(
                "lora_transformer_nonexistent_module.lora_down.weight".to_string(),
                t2(&[0.0, 0.0], 1, 2),
            )]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 0);
        assert!(report.skipped_keys >= 1);
    }
}
