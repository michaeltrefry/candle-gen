//! DPT depth-estimation head — candle port of `DepthAnythingDepthEstimationHead` for the `head.*`
//! weights. The candle twin of `mlx-gen-depth`'s `head.rs`.
//!
//! `conv1` (3×3 pad-1, fusion_hidden_size→fusion_hidden_size/2) → bilinear upsample (align_corners
//! true) to the full `patch_grid · patch_size` resolution → `conv2` (3×3 pad-1 → head_hidden_size)
//! → ReLU → `conv3` (1×1 → 1) → ReLU (DA-V2 is a *relative*-depth model: the final activation is
//! ReLU, no sigmoid / max-depth scaling). Output `[B, H, W]` single-channel depth.

use candle_gen::candle_core::Tensor;
use candle_gen::Result;

use crate::common::{bilinear_resize, conv2d_nhwc, join, relu, Weights};
use crate::config::DepthAnythingConfig;

pub struct DepthHead {
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    conv3_w: Tensor,
    conv3_b: Tensor,
    patch_size: usize,
}

impl DepthHead {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &DepthAnythingConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            conv1_w: w.require(&p("conv1.weight"))?,
            conv1_b: w.require(&p("conv1.bias"))?,
            conv2_w: w.require(&p("conv2.weight"))?,
            conv2_b: w.require(&p("conv2.bias"))?,
            conv3_w: w.require(&p("conv3.weight"))?,
            conv3_b: w.require(&p("conv3.bias"))?,
            patch_size: cfg.patch_size,
        })
    }

    /// `fused`: the neck's fused NHWC map `[B, h, w, fusion_hidden]`. `patch_grid` is the backbone
    /// token-grid side (37 at the default size) — the head upsamples to `patch_grid · patch_size`
    /// (the input resolution). Returns `[B, H, W]`.
    pub fn forward(&self, fused: &Tensor, patch_grid: usize) -> Result<Tensor> {
        let x = conv2d_nhwc(fused, &self.conv1_w, Some(&self.conv1_b), 1, 1)?;
        let full = patch_grid * self.patch_size;
        let x = bilinear_resize(&x, full, full, true)?;
        let x = conv2d_nhwc(&x, &self.conv2_w, Some(&self.conv2_b), 1, 1)?;
        let x = relu(&x)?;
        let x = conv2d_nhwc(&x, &self.conv3_w, Some(&self.conv3_b), 1, 0)?;
        let x = relu(&x)?; // relative-depth: ReLU output, channel dim = 1.
        let (b, h, wd, _) = x.dims4()?;
        // [B, H, W, 1] → [B, H, W].
        Ok(x.reshape((b, h, wd))?)
    }
}
