//! Wan inference-side adapter merge (sc-5167) — load a trained LoRA/LoKr `.safetensors` and fold its
//! delta into the dense DiT-expert weights **before** the stock [`WanTransformer`](crate::transformer)
//! is built. The Wan twin of [`candle-gen-sdxl::adapters`] / `candle-gen-z-image::adapters`, and the
//! closing half of the native-trainer loop: a LoRA/LoKr produced by [`crate::training`]'s Wan MoE
//! trainer now actually loads in candle inference.
//!
//! **Merge, don't residual** (the chaos-sensitive-sampler argument from the SDXL/Z-Image ports), at the
//! **safetensors-key level** before construction: load the expert's base weights into a
//! `HashMap<String,Tensor>` on CPU, add `δ` to `{path}.weight`, then `VarBuilder::from_tensors`. The
//! stock Wan DiT reads diffusers keys 1:1, so `{path}.weight` is a valid base key for every attention
//! projection an adapter targets (`blocks.{i}.attn1/attn2.{to_q,to_k,to_v,to_out.0}`). The delta is
//! reconstructed with the **same** f32 math the trainer's forward uses
//! ([`reconstruct_lora_delta`](candle_gen::train::lora::reconstruct_lora_delta) /
//! [`reconstruct_lokr_delta`](candle_gen::train::lora::reconstruct_lokr_delta)), so a candle-trained
//! adapter round-trips exactly.
//!
//! **MoE.** The A14B is two experts (`transformer/` high-noise, `transformer_2/` low-noise). A trained
//! Wan MoE LoRA ships as a `{stem}.high_noise` / `{stem}.low_noise` pair; the worker tags each
//! [`AdapterSpec`] with [`MoeExpert`](candle_gen::gen_core::MoeExpert) so the high file merges onto the
//! high expert and the low onto the low. This module merges whatever specs it is handed into one map;
//! the per-expert routing (filter by `moe_expert`) lives in [`crate::wan14b`].
//!
//! **Key conventions.** The candle trainer writes **bare** dotted PEFT/LoKr keys (no prefix). Community
//! Wan LoRAs carry a `diffusion_model.` / `transformer.` namespace (the diffusers/sd-scripts exports) or
//! the kohya `lora_unet_<flattened>` form; all resolve. Out-of-surface keys are counted in
//! [`MergeReport`] and surfaced, never silently dropped.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::{reconstruct_lokr_delta, reconstruct_lora_delta};
use candle_gen::{CandleError, Result};

/// LoRA-key namespace prefixes a Wan adapter may carry, longest-first so the more specific peft form
/// wins. The candle trainer writes bare keys (matched by the trailing `""`).
const LORA_PREFIXES: [&str; 5] = [
    "base_model.model.diffusion_model.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
    "",
];
/// kohya / sd-scripts community LoRA key prefix (the flattened-module form).
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

/// Outcome of merging adapter specs into one expert's tensor map: base weights updated, and keys that
/// fell outside the merge surface (text-encoder / unresolved — surfaced, not silently dropped).
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
    down: Option<Tensor>,
    up: Option<Tensor>,
    alpha: Option<f32>,
}

/// A loaded adapter file: its tensors (CPU, native dtype) and the safetensors header metadata.
struct AdapterFile {
    tensors: HashMap<String, Tensor>,
    meta: HashMap<String, String>,
}

/// Read an adapter `.safetensors` once: tensors via candle's loader, metadata via the header reader
/// (candle's `load` drops `__metadata__`, where LoKr `rank`/`alpha` live).
fn read_adapter(path: &Path) -> Result<AdapterFile> {
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("read adapter {}: {e}", path.display())))?;
    let tensors = cst::load_buffer(&bytes, &Device::Cpu)?;
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("read adapter metadata {}: {e}", path.display())))?;
    let meta = md.metadata().clone().unwrap_or_default();
    Ok(AdapterFile { tensors, meta })
}

/// Build the kohya `flattened → dotted` lookup from the expert's 2-D Linear weight keys
/// (`{dotted}.weight`). The `_`-flattening diffusers uses is ambiguous, so resolving against the real
/// key set is what disambiguates a kohya stem.
fn build_kohya_table(base: &HashMap<String, Tensor>) -> BTreeMap<String, String> {
    base.iter()
        .filter_map(|(k, t)| {
            let dotted = k.strip_suffix(".weight")?;
            (t.dims().len() == 2).then(|| (dotted.replace('.', "_"), dotted.to_string()))
        })
        .collect()
}

/// Strip the longest matching [`LORA_PREFIXES`] namespace from a dotted key (or return it unchanged for
/// a bare key).
fn strip_lora_prefix(key: &str) -> &str {
    for p in LORA_PREFIXES {
        if let Some(rem) = key.strip_prefix(p) {
            return rem;
        }
    }
    key
}

/// Map one LoRA key to `(diffusers_dotted_path, role)`, or `None` if outside the DiT merge surface.
/// kohya (`lora_unet_<flat>…`) resolves the flattened stem via `table`; the dotted forms (bare or
/// namespaced) resolve directly.
fn classify_lora_key(key: &str, table: &BTreeMap<String, String>) -> Option<(String, Role)> {
    // A bundled text-encoder adapter (`lora_te*` / `…text_encoder.…`) is never a DiT target — reject it
    // up front so the permissive dotted branch below (which accepts a bare path) can't mis-route it.
    if key.starts_with("lora_te") || key.contains("text_encoder") {
        return None;
    }
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

/// Map one LoKr factor key to `(diffusers_dotted_path, factor_name)`, or `None` if out of surface.
fn classify_lokr_key(
    key: &str,
    table: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..];
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                table.get(flat).map(|d| (d.clone(), factor))
            } else {
                Some((strip_lora_prefix(stem).to_string(), factor))
            };
        }
    }
    None
}

fn read_scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Merge `delta` (`[out, in]` f32) into the base weight at `key`, computing `W += δ` in f32. A missing
/// or shape-mismatched key is surfaced as skipped, never a hard error.
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
/// into `{path}.weight`. `rank` is `A`'s leading dim; `alpha` the per-target `.alpha` (default `rank`).
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
            report.skipped_keys += 1; // half-pair
            continue;
        };
        if down.dims().len() != 2 || up.dims().len() != 2 {
            report.skipped_keys += 1; // Wan adapts attention Linears only (no conv surface)
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

/// Merge one LoKr file into `base` at `scale`: `rank`/`alpha` from file metadata, per-module factors
/// grouped, `δ = (alpha/rank)·kron(w1,w2)·scale` reconstructed and merged.
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
            report.skipped_keys += 1;
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
    candle_gen::gen_core::weightsmeta::is_lokr_network_type(
        af.meta.get("networkType").map(String::as_str),
    )
}

/// Fold every adapter spec in `specs` into one expert's base DiT tensor `map` (CPU, native dtype) at
/// each spec's `scale` — LoRA and LoKr, merged into the dense weights (`W += δ`). Returns the
/// [`MergeReport`]; errors if a non-empty spec list matches **no** target (a format/prefix
/// misconfiguration — the worker should fall back rather than render an unadapted video silently).
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
                if declares_lokr(&af) {
                    return Err(CandleError::Msg(format!(
                        "wan: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_lora_file(map, &af, spec.scale, &table, &mut report)?;
            }
        }
    }
    if report.merged == 0 {
        return Err(CandleError::Msg(format!(
            "wan: no adapter target modules matched across {} file(s) — expected PEFT \
             `[diffusion_model.|transformer.]<path>.lora_A/B.weight` or kohya \
             `lora_unet_<flat>.lora_down/up.weight` (LoRA), or `<module>.lokr_w1/w2` with \
             networkType=lokr (LoKr), targeting `blocks.<i>.attn1/attn2.{{to_q,to_k,to_v,to_out.0}}`",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny stand-in for one expert's DiT tensor map: the four attention projections of block 0.
    fn base_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        for leaf in ["attn1.to_q", "attn1.to_out.0", "attn2.to_k"] {
            m.insert(
                format!("blocks.0.{leaf}.weight"),
                Tensor::zeros((4, 4), DType::F16, &dev).unwrap(),
            );
        }
        m
    }

    fn t2(data: &[f32], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (r, c), &Device::Cpu).unwrap()
    }

    /// The bare candle-trainer key, a `diffusion_model.`-namespaced community key, and a kohya
    /// flattened stem all resolve to the same dotted path.
    #[test]
    fn classify_lora_resolves_bare_namespaced_and_kohya() {
        let table = build_kohya_table(&base_map());
        let (p, _) = classify_lora_key("blocks.0.attn1.to_q.lora_A.weight", &table).unwrap();
        assert_eq!(p, "blocks.0.attn1.to_q");
        let (p, _) = classify_lora_key(
            "diffusion_model.blocks.0.attn1.to_q.lora_down.weight",
            &table,
        )
        .unwrap();
        assert_eq!(p, "blocks.0.attn1.to_q");
        let (p, _) =
            classify_lora_key("lora_unet_blocks_0_attn1_to_out_0.lora_up.weight", &table).unwrap();
        assert_eq!(p, "blocks.0.attn1.to_out.0");
        // text-encoder keys are out of the DiT surface.
        assert!(
            classify_lora_key("lora_te_text_model_layers_0_q.lora_down.weight", &table).is_none()
        );
    }

    /// PEFT LoRA merges into `W += (alpha/rank)·scale·B·A`; base is zero so the merged weight IS ΔW.
    #[test]
    fn merge_lora_folds_expected_delta() {
        let mut map = base_map();
        let down = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 4);
        let up = t2(&[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                (
                    "blocks.0.attn1.to_q.lora_A.weight".to_string(),
                    down.clone(),
                ),
                ("blocks.0.attn1.to_q.lora_B.weight".to_string(), up.clone()),
                (
                    "blocks.0.attn1.to_q.alpha".to_string(),
                    Tensor::from_vec(vec![4.0f32], (1,), &Device::Cpu).unwrap(),
                ),
            ]),
            meta: HashMap::new(),
        };
        let table = build_kohya_table(&map);
        let mut report = MergeReport::default();
        merge_lora_file(&mut map, &af, 1.0, &table, &mut report).unwrap();
        assert_eq!(report.merged, 1);
        let merged = map
            .get("blocks.0.attn1.to_q.weight")
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let expected = reconstruct_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
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

    /// LoKr merges `δ = (alpha/rank)·kron(w1,w2)` into the dense weight, reading rank/alpha from meta.
    #[test]
    fn merge_lokr_folds_kron_delta() {
        let mut map = base_map();
        let w1 = t2(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w2 = t2(&[0.5, 0.0, 0.0, 0.5], 2, 2);
        let af = AdapterFile {
            tensors: HashMap::from([
                ("blocks.0.attn2.to_k.lokr_w1".to_string(), w1.clone()),
                ("blocks.0.attn2.to_k.lokr_w2".to_string(), w2.clone()),
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
            .get("blocks.0.attn2.to_k.weight")
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

    /// A non-empty spec list that matches nothing surfaces as zero-merged (the public entry then errors).
    #[test]
    fn merge_lora_nothing_matched_is_zero() {
        let mut map = base_map();
        let af = AdapterFile {
            tensors: HashMap::from([(
                "blocks.99.attn1.to_q.lora_A.weight".to_string(),
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
