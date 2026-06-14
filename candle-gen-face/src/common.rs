//! Shared leaf helpers for the face sub-models (SCRFD / ArcFace): a small safetensors weight map,
//! the BN-folded biased [`Conv`], and the bias-less [`ConvW`]. The candle twin of mlx-gen-face's
//! `common.rs` — but where MLX runs NHWC with OHWI conv weights, candle's `conv2d` is NCHW with OIHW
//! weights, so [`Conv::load`] transposes the stored OHWI `[out, kH, kW, in]` kernels to candle's
//! OIHW `[out, in, kH, kW]` at load. The weight FILES are shared with MLX (the same antelopev2
//! `tools/convert_*` output); only the in-memory layout differs.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{safetensors, DType, Device, Tensor};
use candle_gen::{CandleError, Result};

/// A loaded face-checkpoint weight map. The face sub-model files are small f32 safetensors; every
/// tensor is coerced to f32 on load (the reference is f32 and cosine parity is comfortable there).
pub(crate) struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    /// Load every tensor from a `.safetensors` file onto `device`, coercing to f32.
    pub fn from_file(path: &Path, device: &Device) -> Result<Self> {
        let raw = safetensors::load(path, device)?;
        let mut map = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            let v = if v.dtype() == DType::F32 {
                v
            } else {
                v.to_dtype(DType::F32)?
            };
            map.insert(k, v);
        }
        Ok(Self { map })
    }

    /// Fetch a required tensor, erroring (not panicking) when a checkpoint is missing a key.
    pub fn require(&self, key: &str) -> Result<Tensor> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CandleError::Msg(format!("missing tensor: {key}")))
    }

    /// A per-channel vector (`[C]`) reshaped to broadcast over an NCHW feature map: `[1, C, 1, 1]`.
    /// Used for the folded BN affines and PReLU slopes (the channel axis is 1 in candle, last in MLX).
    pub fn require_channel4d(&self, key: &str) -> Result<Tensor> {
        let t = self.require(key)?;
        let c = t.elem_count();
        Ok(t.reshape((1, c, 1, 1))?)
    }
}

/// Transpose a stored OHWI conv kernel `[out, kH, kW, in]` to candle's OIHW `[out, in, kH, kW]`.
pub(crate) fn ohwi_to_oihw(w: &Tensor) -> Result<Tensor> {
    Ok(w.permute((0, 3, 1, 2))?.contiguous()?)
}

/// A biased convolution (the BN-folded convs all carry a bias, folded in at conversion). The kernel
/// is stored OIHW (transposed from the file's OHWI by [`Conv::load`]); the `[O]` bias broadcasts over
/// the NCHW channel axis.
pub(crate) struct Conv {
    pub w: Tensor,
    pub b: Tensor,
}

impl Conv {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: ohwi_to_oihw(&w.require(&format!("{prefix}.weight"))?)?,
            b: w.require(&format!("{prefix}.bias"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
        let y = x.conv2d(&self.w, padding, stride, 1, 1)?;
        let b = self.b.reshape((1, self.b.elem_count(), 1, 1))?;
        Ok(y.broadcast_add(&b)?)
    }

    /// Biased conv → ReLU.
    pub fn forward_relu(&self, x: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
        Ok(self.forward(x, stride, padding)?.relu()?)
    }
}
