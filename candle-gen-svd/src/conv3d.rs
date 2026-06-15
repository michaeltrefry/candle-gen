//! **Temporal Conv3d** for the SVD `AutoencoderKLTemporalDecoder` — candle ships no `conv3d`. SVD's
//! temporal decoder uses only `(kD, 1, 1)` kernels (the `TemporalResnetBlock` `conv1`/`conv2` are
//! `(3,1,1)` with **symmetric** temporal padding 1, the `conv_shortcut` + `time_conv_out` are likewise
//! temporal-only). A `kD`-tap kernel is decomposed into `kD` pointwise (1×1) conv2d "taps" over the
//! merged `(B·T)` batch, mirroring diffusers' `Conv3d` over the frame axis with `padding=(1,0,0)`.
//!
//! This is the symmetric sibling of the Wan crate's causal `conv3d` (Wan left-pads `kD-1`; SVD's
//! temporal conv is non-causal "same").

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

/// A temporal Conv3d with a `(kD, 1, 1)` kernel and **symmetric** temporal padding `(kD-1)/2` (SD
/// "same"), spatial kernel/pad 1×1/0. Weight is stored diffusers-layout `[O, I, kD, 1, 1]`.
pub struct TemporalConv3d {
    weight: Tensor, // [O, I, kD, 1, 1]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kd: usize,
}

impl TemporalConv3d {
    /// Load `[O, I, kD, 1, 1]` weight + `[O]` bias from `vb`.
    pub fn load(in_c: usize, out_c: usize, kd: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_c, in_c, kd, 1, 1), "weight")?.contiguous()?;
        let bias = vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1, 1))?;
        Ok(Self { weight, bias, kd })
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T, H, W]` (temporal "same", symmetric zero-pad).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let pad = (self.kd - 1) / 2;
        let xpad = if pad > 0 {
            x.pad_with_zeros(2, pad, pad)?
        } else {
            x.clone()
        };
        debug_assert_eq!(
            xpad.dim(2)?,
            t + self.kd - 1,
            "temporal pad must yield t+(kD-1)"
        );
        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kd {
            // Tap weight W[:, :, kd] → [O, I, 1, 1] (a pointwise 1×1 conv2d).
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            // The T frames this tap convolves: x_pad[:, :, kd : kd+T].
            let frames = xpad.narrow(2, kd, t)?;
            // Merge (B, T) into the conv2d batch axis: [B, C, T, H, W] → [B·T, C, H, W].
            let merged = frames
                .permute((0, 2, 1, 3, 4))?
                .reshape((b * t, c, h, w))?
                .contiguous()?;
            let y = merged.conv2d(&wk, 0, 1, 1, 1)?; // [B·T, O, H, W]
            acc = Some(match acc {
                Some(a) => (a + y)?,
                None => y,
            });
        }
        let y = acc.expect("kD >= 1");
        let (_, o, hp, wp) = y.dims4()?;
        let y = y
            .reshape((b, t, o, hp, wp))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        y.broadcast_add(&self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// A `(1,1,1)` temporal kernel is a pointwise channel conv — equivalent to a per-frame matmul.
    #[test]
    fn kd1_is_pointwise() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (4usize, 3usize);
        let conv = TemporalConv3d {
            weight: Tensor::randn(0f32, 1.0, (o, i, 1, 1, 1), &dev)?,
            bias: Tensor::zeros((1, o, 1, 1, 1), DType::F32, &dev)?,
            kd: 1,
        };
        let x = Tensor::randn(0f32, 1.0, (1, i, 5, 2, 2), &dev)?;
        let y = conv.forward(&x)?;
        assert_eq!(y.dims(), &[1, o, 5, 2, 2]);
        Ok(())
    }

    /// A symmetric `(3,1,1)` kernel preserves the temporal length (pad 1 each side).
    #[test]
    fn kd3_preserves_temporal_len() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i) = (2usize, 2usize);
        let conv = TemporalConv3d {
            weight: Tensor::randn(0f32, 1.0, (o, i, 3, 1, 1), &dev)?,
            bias: Tensor::zeros((1, o, 1, 1, 1), DType::F32, &dev)?,
            kd: 3,
        };
        let x = Tensor::randn(0f32, 1.0, (1, i, 7, 3, 3), &dev)?;
        let y = conv.forward(&x)?;
        assert_eq!(y.dims(), &[1, o, 7, 3, 3]);
        Ok(())
    }
}
