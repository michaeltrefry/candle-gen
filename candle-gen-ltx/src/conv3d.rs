//! **Causal / non-causal 3-D convolution** for the LTX-2.3 temporal VAE — candle ships no `conv3d`,
//! and because video has `T > 1` a `kt×kh×kw` kernel does not reduce to one conv2d. It is decomposed
//! into `kt` conv2d "taps": the temporal axis is padded (see below), then the output is
//! `Σ_{kd} conv2d(x_pad[:, :, kd : kd+T], W[:, :, kd])`.
//!
//! LTX padding differs from Wan's causal-zero pad: time is padded by **frame replication** (the
//! reference `CausalConv3d`), spatial by symmetric **zeros** `(k-1)/2`. `causal=true` replicates the
//! first frame `kt-1` times at the front (encoder); `causal=false` replicates the first frame
//! `(kt-1)/2` at the front *and* the last frame `(kt-1)/2` at the back (the decoder path, which is
//! all T2V needs). Weight is the checkpoint-native PyTorch layout `[O, I, kt, kh, kw]`.

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

/// A 3-D conv loaded from a `[O, I, kt, kh, kw]` weight. Temporal stride is always 1 in the LTX VAE;
/// spatial padding is "same" (`(kh-1)/2`); temporal padding is frame-replication (causal toggle).
pub struct CausalConv3d {
    weight: Tensor, // [O, I, kt, kh, kw]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kt: usize,
    spatial_pad: usize,
}

impl CausalConv3d {
    /// Load `{prefix}.weight` + `{prefix}.bias`, inferring channels + kernel dims from the weight
    /// shape `[O, I, kt, kh, kw]` (channels ride on the weights, not the config).
    pub fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        let w = vb
            .get_unchecked(&format!("{prefix}.weight"))?
            .contiguous()?;
        let dims = w.dims();
        let (out_c, kt, kh) = (dims[0], dims[2], dims[3]);
        let bias = vb
            .get_unchecked(&format!("{prefix}.bias"))?
            .reshape((1, out_c, 1, 1, 1))?;
        Ok(Self {
            weight: w,
            bias,
            kt,
            spatial_pad: (kh - 1) / 2,
        })
    }

    /// Replicate `frame` (`[B,C,1,H,W]`) `n` times along the temporal axis.
    fn repeat_frame(frame: &Tensor, n: usize) -> Result<Tensor> {
        let parts: Vec<Tensor> = (0..n).map(|_| frame.clone()).collect();
        Tensor::cat(&parts, 2)
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T, H, W]` (spatial "same", temporal frame-replicate).
    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let xpad = if self.kt > 1 {
            let first = x.narrow(2, 0, 1)?;
            if causal {
                let front = Self::repeat_frame(&first, self.kt - 1)?;
                Tensor::cat(&[&front, x], 2)?
            } else {
                let ps = (self.kt - 1) / 2;
                if ps > 0 {
                    let last = x.narrow(2, t - 1, 1)?;
                    let front = Self::repeat_frame(&first, ps)?;
                    let back = Self::repeat_frame(&last, ps)?;
                    Tensor::cat(&[&front, x, &back], 2)?
                } else {
                    x.clone()
                }
            }
        } else {
            x.clone()
        };

        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kt {
            // Tap W[:, :, kd] → [O, I, kh, kw].
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            // The T frames this tap convolves: x_pad[:, :, kd : kd+T].
            let frames = xpad.narrow(2, kd, t)?;
            // [B, C, T, H, W] → [B*T, C, H, W] for conv2d.
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * t, c, h, w))?
                .contiguous()?;
            let y = merged.conv2d(&wk, self.spatial_pad, 1, 1, 1)?;
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let y = acc.expect("kt >= 1");
        let (_, o, hp, wp) = y.dims4()?;
        let y = y
            .reshape((b, t, o, hp, wp))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        y.broadcast_add(&self.bias)
    }
}
