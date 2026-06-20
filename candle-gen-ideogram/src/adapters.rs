//! TurboTime LoRA merge for the Ideogram 4 **Turbo** path. The few-step ostris "continuous turbo"
//! LoRA is bundled in a turbo snapshot ([`crate::config::TURBO_LORA_FILE`]); this folds its delta
//! into the conditional DiT's weights at load (`W += eff·(up @ down)`, in f32, Linear-only), via the
//! loader's override layer — the candle analogue of `mlx-gen-ideogram`'s `apply_ideogram_adapters`.
//!
//! Key forms handled: `{ns}{module}.lora_{down,up}.weight` / `.lora_{A,B}.weight` (and the
//! `.weight`-less variants), namespace `ns` ∈ {`diffusion_model.`, `transformer.`, `model.`, none}
//! (sd-scripts / ai-toolkit exports). The `module` path (e.g. `layers.0.attention.qkv`) matches the
//! DiT's safetensors keys directly. An optional `{module}.alpha` applies `alpha/rank` scaling.

use std::collections::HashSet;
use std::path::Path;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Error, Result};

use crate::loader::Weights;

/// Recognized `(down, up)` suffix pairs, most-specific first.
const PAIRS: &[(&str, &str)] = &[
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down", ".lora_up"),
    (".lora_A", ".lora_B"),
];

/// Namespace prefixes stripped to recover the DiT module path.
const PREFIXES: &[&str] = &["diffusion_model.", "transformer.", "model."];

/// Merge the TurboTime LoRA at `lora_path` into `w`'s DiT weights (override layer). Returns the
/// number of merged target modules. Errors if the file is missing or **no** target matched (a wrong
/// key format / prefix), mirroring the MLX strict apply's no-silent-drop intent.
pub fn merge_turbo_lora(w: &mut Weights, lora_path: &Path, scale: f32) -> Result<usize> {
    if !lora_path.exists() {
        return Err(Error::Msg(format!(
            "ideogram turbo: TurboTime LoRA not found at {} (a turbo snapshot must ship it alongside transformer/)",
            lora_path.display()
        )));
    }
    // SAFETY: read-only mmap of the adapter file.
    let lora = unsafe { MmapedSafetensors::new(lora_path)? };
    let names: Vec<String> = lora.tensors().into_iter().map(|(n, _)| n).collect();
    let present: HashSet<&str> = names.iter().map(String::as_str).collect();

    let mut merged = 0usize;
    let mut skipped = 0usize;
    for name in &names {
        let Some((base_full, up_name)) = down_pair(name, &present) else {
            continue;
        };
        let module = strip_prefix(&base_full);
        let weight_key = format!("{module}.weight");
        if !w.contains(&weight_key) {
            skipped += 1;
            continue;
        }
        let down = lora.load(name, w.device())?.to_dtype(DType::F32)?; // [r, in]
        let up = lora.load(&up_name, w.device())?.to_dtype(DType::F32)?; // [out, r]
        if down.rank() != 2 || up.rank() != 2 {
            return Err(Error::Msg(format!(
                "ideogram turbo: LoRA {name} is not a 2D Linear adapter (rank {}/{})",
                up.rank(),
                down.rank()
            )));
        }
        let rank = down.dim(0)?;
        let eff = scale as f64
            * alpha_for(&lora, &base_full)
                .map(|a| a as f64 / rank as f64)
                .unwrap_or(1.0);
        let delta = up.contiguous()?.matmul(&down.contiguous()?)?; // [out, in]
        let base = w.get_f32(&weight_key)?;
        let merged_w = (base + (delta * eff)?)?.to_dtype(w.dtype())?;
        w.insert_override(weight_key, merged_w);
        merged += 1;
    }

    if merged == 0 {
        return Err(Error::Msg(format!(
            "ideogram turbo: no TurboTime LoRA targets matched the DiT (checked {} adapter tensors — wrong key format/prefix?)",
            names.len()
        )));
    }
    if skipped > 0 {
        eprintln!(
            "ideogram turbo: merged {merged} LoRA target(s), skipped {skipped} non-DiT key(s)"
        );
    }
    Ok(merged)
}

/// If `name` is a recognized "down"/"A" key whose paired "up"/"B" is also present, return
/// `(module_base_with_namespace, up_key)`.
fn down_pair(name: &str, present: &HashSet<&str>) -> Option<(String, String)> {
    for (down_suf, up_suf) in PAIRS {
        if let Some(base) = name.strip_suffix(down_suf) {
            let up = format!("{base}{up_suf}");
            if present.contains(up.as_str()) {
                return Some((base.to_string(), up));
            }
        }
    }
    None
}

/// Strip a known namespace prefix to recover the DiT module path.
fn strip_prefix(base: &str) -> &str {
    for p in PREFIXES {
        if let Some(rest) = base.strip_prefix(p) {
            return rest;
        }
    }
    base
}

/// Read an optional `{base}.alpha` scalar.
fn alpha_for(lora: &MmapedSafetensors, base_full: &str) -> Option<f32> {
    let t = lora
        .load(&format!("{base_full}.alpha"), &Device::Cpu)
        .ok()?;
    t.to_dtype(DType::F32)
        .ok()?
        .flatten_all()
        .ok()?
        .to_vec1::<f32>()
        .ok()?
        .first()
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_prefix_recovers_module_path() {
        assert_eq!(
            strip_prefix("diffusion_model.layers.0.attention.qkv"),
            "layers.0.attention.qkv"
        );
        assert_eq!(strip_prefix("transformer.input_proj"), "input_proj");
        assert_eq!(
            strip_prefix("layers.3.feed_forward.w1"),
            "layers.3.feed_forward.w1"
        );
    }

    #[test]
    fn down_pair_matches_known_suffixes() {
        let names = [
            "m.lora_down.weight".to_string(),
            "m.lora_up.weight".to_string(),
        ];
        let present: HashSet<&str> = names.iter().map(String::as_str).collect();
        assert_eq!(
            down_pair("m.lora_down.weight", &present),
            Some(("m".to_string(), "m.lora_up.weight".to_string()))
        );
        // The up half alone is not a "down" key.
        assert_eq!(down_pair("m.lora_up.weight", &present), None);
    }
}
