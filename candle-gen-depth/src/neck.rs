//! DPT neck — reassemble stage + RefineNet feature-fusion stage. Faithful candle port of the HF
//! `transformers` `DepthAnythingNeck` / `DepthAnythingReassembleStage` /
//! `DepthAnythingFeatureFusionStage` for the `neck.*` weight tree. The candle twin of
//! `mlx-gen-depth`'s `neck.rs`.
//!
//! Stage 1 — **reassemble**: each of the four captured hidden states `[B, grid²+1, hidden]` has its
//! CLS token dropped, is reshaped to a 2-D map `[B, grid, grid, hidden]`, projected by a 1×1 conv to
//! `neck_hidden_sizes[i]`, then resized by `reassemble_factors[i]`:
//!   - factor > 1 → `ConvTranspose2d(kernel=factor, stride=factor)` (upsample),
//!   - factor == 1 → identity,
//!   - factor < 1 → `Conv2d(kernel=3, stride=1/factor, pad=1)` (downsample).
//!
//! Stage 2 — `convs`: a 3×3 (pad 1, **no bias**) conv projects each reassembled map to
//! `fusion_hidden_size` (64).
//!
//! Stage 3 — **feature fusion** (`fusion_stage`), processed deepest→shallowest: a pre-activation
//! residual unit refines each level; from the second level on the running fused map is bilinearly
//! resized to the incoming residual, summed, refined again, ×2 bilinearly upsampled, and 1×1
//! projected. The shallowest fused map is the head input.

use candle_gen::candle_core::Tensor;
use candle_gen::Result;

use crate::common::{bilinear_resize, conv2d_nhwc, conv_transpose2d_nhwc, join, relu, Weights};
use crate::config::DepthAnythingConfig;

/// How a reassemble layer resizes its projected map.
enum Resize {
    /// `ConvTranspose2d(kernel=stride=factor)`: IOHW weight + bias.
    Up { w: Tensor, b: Tensor, stride: usize },
    /// Identity (factor == 1).
    Same,
    /// `Conv2d(kernel=3, stride, pad=1)`: OIHW weight + bias.
    Down { w: Tensor, b: Tensor, stride: usize },
}

/// One reassemble layer: 1×1 projection + factor resize.
struct ReassembleLayer {
    proj_w: Tensor, // 1×1 conv OIHW
    proj_b: Tensor,
    resize: Resize,
}

impl ReassembleLayer {
    fn from_weights(w: &Weights, prefix: &str, factor: f32) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let resize = if factor > 1.0 {
            Resize::Up {
                w: w.require(&p("resize.weight"))?,
                b: w.require(&p("resize.bias"))?,
                stride: factor as usize,
            }
        } else if (factor - 1.0).abs() < f32::EPSILON {
            Resize::Same
        } else {
            Resize::Down {
                w: w.require(&p("resize.weight"))?,
                b: w.require(&p("resize.bias"))?,
                stride: (1.0 / factor).round() as usize,
            }
        };
        Ok(Self {
            proj_w: w.require(&p("projection.weight"))?,
            proj_b: w.require(&p("projection.bias"))?,
            resize,
        })
    }

    /// `hidden`: a captured backbone state `[B, grid²+1, hidden]` → an NHWC feature map.
    fn forward(&self, hidden: &Tensor, grid: usize, hidden_dim: usize) -> Result<Tensor> {
        // Drop CLS (index 0), reshape patch tokens to [B, grid, grid, hidden] (NHWC).
        let b = hidden.dim(0)?;
        let n = hidden.dim(1)?;
        let patches = hidden.narrow(1, 1, n - 1)?;
        let map = patches.reshape((b, grid, grid, hidden_dim))?;
        // 1×1 projection.
        let map = conv2d_nhwc(&map, &self.proj_w, Some(&self.proj_b), 1, 0)?;
        match &self.resize {
            Resize::Up { w, b: bias, stride } => conv_transpose2d_nhwc(&map, w, bias, *stride),
            Resize::Same => Ok(map),
            Resize::Down { w, b: bias, stride } => conv2d_nhwc(&map, w, Some(bias), *stride, 1),
        }
    }
}

/// Pre-activation residual unit (`PreActResidualLayer`): `ReLU → conv3×3 → ReLU → conv3×3`, added to
/// the input. Both convs are pad-1 stride-1 with bias.
struct PreActResidual {
    c1_w: Tensor,
    c1_b: Tensor,
    c2_w: Tensor,
    c2_b: Tensor,
}

impl PreActResidual {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            c1_w: w.require(&p("convolution1.weight"))?,
            c1_b: w.require(&p("convolution1.bias"))?,
            c2_w: w.require(&p("convolution2.weight"))?,
            c2_b: w.require(&p("convolution2.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = relu(x)?;
        let h = conv2d_nhwc(&h, &self.c1_w, Some(&self.c1_b), 1, 1)?;
        let h = relu(&h)?;
        let h = conv2d_nhwc(&h, &self.c2_w, Some(&self.c2_b), 1, 1)?;
        Ok(x.add(&h)?)
    }
}

/// One feature-fusion layer (`DepthAnythingFeatureFusionLayer`).
struct FusionLayer {
    res1: PreActResidual,
    res2: PreActResidual,
    proj_w: Tensor, // 1×1 conv OIHW
    proj_b: Tensor,
}

impl FusionLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            res1: PreActResidual::from_weights(w, &p("residual_layer1"))?,
            res2: PreActResidual::from_weights(w, &p("residual_layer2"))?,
            proj_w: w.require(&p("projection.weight"))?,
            proj_b: w.require(&p("projection.bias"))?,
        })
    }

    /// Fuse: when `residual` is present, refine it (res1) + add to the running map (bilinearly
    /// resized to the residual's HW when they differ, align_corners=False), then refine (res2),
    /// ×2 bilinear upsample (align_corners=True), 1×1 project.
    fn forward(&self, hidden: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let mut x = hidden.clone();
        if let Some(res) = residual {
            let (rh, rw) = (res.dim(1)?, res.dim(2)?);
            if x.dim(1)? != rh || x.dim(2)? != rw {
                x = bilinear_resize(&x, rh, rw, false)?;
            }
            x = x.add(&self.res1.forward(res)?)?;
        }
        x = self.res2.forward(&x)?;
        // ×2 upsample (align_corners=True).
        let (h, w) = (x.dim(1)?, x.dim(2)?);
        x = bilinear_resize(&x, h * 2, w * 2, true)?;
        // 1×1 projection.
        conv2d_nhwc(&x, &self.proj_w, Some(&self.proj_b), 1, 0)
    }
}

/// The full DPT neck: reassemble + project (`convs`) + fusion stage.
pub struct DptNeck {
    reassemble: Vec<ReassembleLayer>,
    // 3×3 pad-1 NO-bias projection per level (`neck.convs.{i}`).
    convs: Vec<Tensor>,
    fusion: Vec<FusionLayer>,
}

impl DptNeck {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &DepthAnythingConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let reassemble = (0..4)
            .map(|i| {
                ReassembleLayer::from_weights(
                    w,
                    &p(&format!("reassemble_stage.layers.{i}")),
                    cfg.reassemble_factors[i],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let convs = (0..4)
            .map(|i| w.require(&p(&format!("convs.{i}.weight"))))
            .collect::<Result<Vec<_>>>()?;
        let fusion = (0..4)
            .map(|i| FusionLayer::from_weights(w, &p(&format!("fusion_stage.layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            reassemble,
            convs,
            fusion,
        })
    }

    /// `hidden_states`: the four captured backbone states (shallow→deep), each `[B, grid²+1, hidden]`.
    /// Returns the fused NHWC feature map the head consumes.
    pub fn forward(
        &self,
        hidden_states: &[Tensor],
        grid: usize,
        hidden_dim: usize,
    ) -> Result<Tensor> {
        // Reassemble + project each level (shallow→deep order).
        let mut feats = Vec::with_capacity(4);
        for (i, hs) in hidden_states.iter().enumerate() {
            let re = self.reassemble[i].forward(hs, grid, hidden_dim)?;
            // 3×3 pad-1 no-bias projection to fusion_hidden_size.
            let f = conv2d_nhwc(&re, &self.convs[i], None, 1, 1)?;
            feats.push(f);
        }
        // Fusion runs deepest→shallowest. `transformers` reverses the feature list so `fusion[0]`
        // pairs with the deepest level (`feats[3]`) and fuses with no residual; each subsequent
        // `fusion[k]` folds in the next-shallower feature (`feats[3-k]`) as its residual.
        let mut fused = self.fusion[0].forward(&feats[3], None)?;
        for k in 1..4 {
            fused = self.fusion[k].forward(&fused, Some(&feats[3 - k]))?;
        }
        Ok(fused)
    }
}
