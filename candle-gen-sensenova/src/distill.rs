//! The 8-step **distill LoRA** merge for the `sensenova_u1_8b_fast` variant — the candle port of
//! `mlx-gen-sensenova`'s `distill.rs`.
//!
//! The reference ships an 8-NFE preview as a LoRA over the base checkpoint
//! (`sensenova/SenseNova-U1-8B-MoT-LoRAs` → `SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors`) that
//! is merged at load and then run at `cfg_scale=1.0` / `timestep_shift=3.0` / `num_steps=8`. The
//! merge is: for every base parameter `…W.weight` with a matching `…W.lora_down.weight` /
//! `…W.lora_up.weight` / `…W.alpha`, add `Δ = (alpha/rank)·(up @ down)` into the weight (`W += Δ`,
//! computed in f32 — the components already load f32 here).
//!
//! The distill LoRA touches **only** the generation path — every layer's `*_mot_gen` attention
//! projections (`{q,k,v,o}_proj_mot_gen`) and SwiGLU (`mlp_mot_gen.{gate,up,down}_proj`), plus the
//! two FM-head Linears (`fm_modules.fm_head.{0,2}`) — `7·layers + 2` targets.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{Linear, VarBuilder};
use candle_gen::{CandleError, Result};

/// The distill LoRA file name (the `--include` argument the reference docs download).
pub const DISTILL_LORA_FILE: &str = "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors";
/// The HF Hub repo the distill LoRA ships in (for the not-found error hint).
pub const DISTILL_LORA_REPO: &str = "sensenova/SenseNova-U1-8B-MoT-LoRAs";

/// A loaded distill-LoRA weight map (f32 mmap), keyed by the PEFT/diffusers `…lora_down`/`lora_up`/
/// `alpha` layout.
pub struct DistillLora {
    vb: VarBuilder<'static>,
}

impl DistillLora {
    /// mmap the LoRA `.safetensors` at f32 on `device`.
    pub fn from_file(path: &Path, device: &Device) -> Result<Self> {
        // SAFETY: mmap of a read-only weight file; the standard candle loading path.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.to_path_buf()], DType::F32, device)?
        };
        Ok(Self { vb })
    }

    /// The `[out, in]` merge delta for `target` (the base weight key **without** its `.weight`
    /// suffix, e.g. `…self_attn.q_proj_mot_gen`), or `None` if the LoRA does not carry that target.
    ///
    /// `Δ = (alpha/rank)·(up @ down)`, where `down` is `[rank, in]`, `up` is `[out, rank]`, and
    /// `rank = down.shape[0]`.
    pub fn delta(&self, target: &str) -> Result<Option<Tensor>> {
        let down_key = format!("{target}.lora_down.weight");
        if !self.vb.contains_tensor(&down_key) {
            return Ok(None);
        }
        let down = self.vb.get_unchecked(&down_key)?; // [rank, in]
        let up = self.vb.get_unchecked(&format!("{target}.lora_up.weight"))?; // [out, rank]
        let alpha = scalar_f32(&self.vb.get_unchecked(&format!("{target}.alpha"))?)?;
        let rank = down.dim(0)? as f32;
        if rank == 0.0 {
            // Zero rank (empty/malformed down factor) → non-finite scaling → NaN-poisoned GEN-path
            // merge that silently corrupts every generation. Reject instead (F-002).
            return Err(CandleError::Msg(format!(
                "distill LoRA: zero-rank factor at '{target}'"
            )));
        }
        let scaling = (alpha / rank) as f64;
        let delta = (up.matmul(&down)? * scaling)?; // [out, in]
        Ok(Some(delta))
    }

    /// Merge this LoRA's delta for `target` into `lin` (a bias-less or biased Linear), returning the
    /// merged Linear, or `None` if the LoRA carries no such target. `W += Δ`; the bias is untouched.
    pub fn merge_linear(&self, lin: &Linear, target: &str) -> Result<Option<Linear>> {
        match self.delta(target)? {
            Some(delta) => {
                let merged = (lin.weight() + delta)?;
                Ok(Some(Linear::new(merged, lin.bias().cloned())))
            }
            None => Ok(None),
        }
    }
}

/// Read a (possibly I32) scalar LoRA `alpha` as `f32`. The distill LoRA stores `alpha` as an `I32`
/// scalar; the f32 VarBuilder already converts it, but it may be rank-0 or `[1]` — flatten and take
/// the first element.
fn scalar_f32(a: &Tensor) -> Result<f32> {
    a.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .first()
        .copied()
        .ok_or_else(|| CandleError::Msg("distill LoRA: empty alpha scalar".into()))
}

/// Resolve the distill LoRA `.safetensors` for the `fast` variant. Resolution order:
/// 1. `$SENSENOVA_DISTILL_LORA` (explicit override / CI),
/// 2. co-located in the base snapshot `root`,
/// 3. the standard HF Hub cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`, or the user-home `.cache`).
///
/// Errors with a download hint if none resolve — the fast variant never silently falls back to the
/// un-merged base.
pub fn resolve_distill_lora(root: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("SENSENOVA_DISTILL_LORA") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Ok(p);
        }
        return Err(CandleError::Msg(format!(
            "SENSENOVA_DISTILL_LORA={} does not exist",
            p.display()
        )));
    }
    let co_located = root.join(DISTILL_LORA_FILE);
    if co_located.exists() {
        return Ok(co_located);
    }
    if let Some(p) = hf_cache_distill_lora() {
        return Ok(p);
    }
    Err(CandleError::Msg(format!(
        "sensenova_u1_8b_fast: distill LoRA `{DISTILL_LORA_FILE}` not found. Download it \
         (`huggingface-cli download {DISTILL_LORA_REPO} --include {DISTILL_LORA_FILE}`) or set \
         SENSENOVA_DISTILL_LORA to its path."
    )))
}

/// Locate `DISTILL_LORA_FILE` under the HF Hub cache for [`DISTILL_LORA_REPO`], scanning each
/// `snapshots/<rev>/` directory. Honours `$HF_HUB_CACHE` and `$HF_HOME` before the user-home
/// `.cache` default (`USERPROFILE` on Windows, `HOME` elsewhere).
fn hf_cache_distill_lora() -> Option<PathBuf> {
    let repo_dir = format!("models--{}", DISTILL_LORA_REPO.replace('/', "--"));
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(c) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(c));
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(h).join("hub"));
    }
    for home_var in ["USERPROFILE", "HOME"] {
        if let Ok(home) = std::env::var(home_var) {
            roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
        }
    }
    for snapshots in roots
        .into_iter()
        .map(|r| r.join(&repo_dir).join("snapshots"))
    {
        let Ok(revs) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for rev in revs.filter_map(|e| e.ok()) {
            let cand = rev.path().join(DISTILL_LORA_FILE);
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
}
