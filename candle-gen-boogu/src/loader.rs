//! Weight loading for the Boogu DiT + Qwen3-VL condition encoder — a thin shape-inferring wrapper
//! over candle's [`MmapedSafetensors`], mirroring `mlx-gen-boogu`'s `Weights`/`lin` interface (and
//! `candle-gen-ideogram`'s `loader::Weights`) so the port stays a near-1:1 translation. [`linear`]
//! builds a [`Linear`] from the actual `{base}.weight` (+ optional `{base}.bias`) tensor shapes, so
//! dims that aren't in the public config (the FFN inner width, the embedder MLP hidden) need no
//! hardcoding.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::Linear;

/// An mmaped component-directory of `.safetensors`, loading tensors at a fixed compute dtype.
pub struct Weights {
    st: MmapedSafetensors,
    device: Device,
    dtype: DType,
}

impl Weights {
    /// mmap every `*.safetensors` in `dir` (sorted; later files win on name collision).
    pub fn from_dir(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| {
                candle_gen::candle_core::Error::Msg(format!("boogu: read {}: {e}", dir.display()))
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "boogu: no .safetensors in {}",
                dir.display()
            )));
        }
        // SAFETY: read-only mmap of weight files; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
        })
    }

    /// Load `name` at the component dtype.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(self.dtype)
    }

    /// Load `name` forcing f32 (norm weights and other precision-sensitive scalars).
    pub fn get_f32(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(DType::F32)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.st.get(name).is_ok()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Build a [`Linear`] from `{base}.weight` (+ `{base}.bias` when `bias`), inferring in/out dims from
/// the stored tensor shape (`[out, in]`, PyTorch/HF convention).
pub fn linear(w: &Weights, base: &str, bias: bool) -> Result<Linear> {
    let weight = w.get(&format!("{base}.weight"))?;
    let bias = if bias {
        Some(w.get(&format!("{base}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// RMSNorm over the last dim with weight `w` (candle's fused op; eps as f32). Inference-only — the
/// fused kernel has no backward, which is irrelevant here.
pub(crate) fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    candle_gen::candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)
}

/// Plain LayerNorm over the last dim with **no affine** (LuminaLayerNormContinuous's inner norm,
/// eps 1e-6): `(x − mean) / sqrt(var + eps)`. Computed in f32 then cast back to `x`'s dtype.
pub(crate) fn layernorm_noaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(dt)
}
