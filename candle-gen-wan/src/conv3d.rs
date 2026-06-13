//! **Causal 3-D convolution** for the Wan temporal VAE — candle ships no `conv3d`, and because video
//! has `T > 1` the conv3d does *not* reduce to a single conv2d (unlike a single-image VAE). Instead a
//! `kD×kH×kW` kernel is decomposed into `kD` conv2d "taps": the temporal axis is causally left-padded
//! by `kD-1` zero frames, and the output is `Σ_{kd} conv2d(x_pad[:, :, kd : kd+T], W[:, :, kd])`.
//!
//! This reproduces diffusers' `WanCausalConv3d` exactly (its `_padding = (·, ·, ·, ·, 2·pad_t, 0)`
//! left-pad + VALID conv, temporal stride 1).
//!
//! **Two decode modes (sc-5176):**
//! - *Single pass* ([`Ctx::single_pass`]): one forward over all `T` frames with the causal
//!   left-padding — what we shipped originally. Correct, but the decoder's full-resolution
//!   activations for **every frame at once** spike VAE memory (~60 GB for a 320²×17 clip) → OOM.
//! - *Streaming* ([`Ctx::streaming`]): decode one latent frame at a time, each `CausalConv3d`
//!   carrying its last `kD-1` input frames as a `feat_cache` (diffusers' frame-by-frame path). The
//!   prepended cache replaces the would-be zero-pad, so streaming is **bit-equivalent** to the single
//!   pass (a causal conv over the whole clip == the cache-streamed one) while bounding peak memory to
//!   ~one frame's activations. The caller resets caches around the decode and flips `first_chunk`.

use std::sync::Mutex;

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

/// Decode context threaded through the VAE forward. `streaming` selects the per-frame `feat_cache`
/// path; `first_chunk` marks the first latent frame (the temporal-upsample "first frame un-doubled"
/// rule — used by the upsampler/dup in `vae.rs`, not here).
#[derive(Clone, Copy)]
pub struct Ctx {
    pub streaming: bool,
    pub first_chunk: bool,
}

impl Ctx {
    /// Whole-clip single pass (causal zero-pad over all frames).
    pub fn single_pass() -> Self {
        Self {
            streaming: false,
            first_chunk: true,
        }
    }
    /// One streaming chunk (one latent frame); `first` is the leading latent frame.
    pub fn streaming(first: bool) -> Self {
        Self {
            streaming: true,
            first_chunk: first,
        }
    }
}

/// A causal Conv3d loaded from a diffusers `[O, I, kD, kH, kW]` weight. Temporal stride is always 1
/// in the Wan decoder; spatial padding is "same" (`(kH-1)/2`), temporal padding is causal (left
/// `kD-1`). In streaming mode it carries the last `kD-1` **input** frames in `cache`.
pub struct CausalConv3d {
    weight: Tensor, // [O, I, kD, kH, kW]
    bias: Tensor,   // [1, O, 1, 1, 1]
    kd: usize,
    spatial_pad: usize,
    /// Streaming `feat_cache`: the last `kD-1` input frames of the previous chunk (≤ `kD-1` while
    /// warming up). `None` = first chunk / reset. `Mutex` (not `RefCell`) keeps the generator
    /// `Send + Sync` for the worker's generator cache (`Arc<WanVae>`); decode is single-threaded so the
    /// lock is always uncontended.
    cache: Mutex<Option<Tensor>>,
}

impl CausalConv3d {
    /// Load from `vb` with an explicit kernel `(kD, kH, kW)` (kH == kW) and channel counts.
    pub fn load(
        in_c: usize,
        out_c: usize,
        kernel: (usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kd, kh, kw) = kernel;
        let weight = vb.get((out_c, in_c, kd, kh, kw), "weight")?.contiguous()?;
        let bias = vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1, 1))?;
        Ok(Self {
            weight,
            bias,
            kd,
            spatial_pad: (kh - 1) / 2,
            cache: Mutex::new(None),
        })
    }

    /// Drop any streaming `feat_cache` (call before/after a streaming decode).
    pub fn reset_cache(&self) {
        *self.cache.lock().unwrap() = None;
    }

    /// `x`: `[B, C, T, H, W]` → `[B, O, T, H, W]` (spatial "same", temporal causal).
    pub fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        // Build the temporally-padded input `xpad` with exactly `t + (kD-1)` frames, so the kD-tap
        // VALID conv below yields `t` output frames.
        let xpad = if self.kd > 1 {
            let want = self.kd - 1; // left context frames needed
            if ctx.streaming {
                let old_cache = self.cache.lock().unwrap().clone();
                let old_n = old_cache
                    .as_ref()
                    .map(|c| c.dim(2))
                    .transpose()?
                    .unwrap_or(0);
                // `[old_cache ++ x]` is the visible input history; left-pad the still-missing
                // `want - old_n` frames with zeros (first chunks, before the cache has warmed).
                let cat_cx = match &old_cache {
                    Some(cache) => Tensor::cat(&[cache, x], 2)?,
                    None => x.clone(),
                };
                let xpad = if want > old_n {
                    cat_cx.pad_with_zeros(2, want - old_n, 0)?
                } else {
                    cat_cx.clone()
                };
                // Update the cache to the last `want` frames of the input history.
                let hn = cat_cx.dim(2)?;
                let keep = want.min(hn);
                *self.cache.lock().unwrap() =
                    Some(cat_cx.narrow(2, hn - keep, keep)?.contiguous()?);
                xpad
            } else {
                x.pad_with_zeros(2, want, 0)?
            }
        } else {
            x.clone()
        };
        let xpad_t = xpad.dim(2)?;
        debug_assert_eq!(
            xpad_t,
            t + self.kd - 1,
            "causal pad must yield t+(kD-1) frames"
        );
        let mut acc: Option<Tensor> = None;
        for kd in 0..self.kd {
            // Tap weight W[:, :, kd] → [O, I, kH, kW].
            let wk = self.weight.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
            // The T frames this tap convolves: x_pad[:, :, kd : kd+T].
            let frames = xpad.narrow(2, kd, t)?;
            // Merge (B, T) into the conv2d batch axis: [B, C, T, H, W] → [B*T, C, H, W].
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

    /// The streaming `feat_cache` path (one frame at a time) must be bit-equivalent to the single
    /// pass over the whole clip — the causal-conv identity sc-5176 relies on for VAE-decode parity.
    #[test]
    fn streaming_matches_single_pass() -> Result<()> {
        let dev = Device::Cpu;
        let (o, i, kd) = (5usize, 3usize, 3usize);
        let conv = CausalConv3d {
            weight: Tensor::randn(0f32, 1.0, (o, i, kd, 3, 3), &dev)?,
            bias: Tensor::randn(0f32, 1.0, o, &dev)?.reshape((1, o, 1, 1, 1))?,
            kd,
            spatial_pad: 1,
            cache: Mutex::new(None),
        };
        let t = 7usize;
        let x = Tensor::randn(0f32, 1.0, (1, i, t, 4, 4), &dev)?;

        let full = conv.forward(&x, &Ctx::single_pass())?;

        conv.reset_cache();
        let mut chunks = Vec::with_capacity(t);
        for f in 0..t {
            let xf = x.narrow(2, f, 1)?.contiguous()?;
            chunks.push(conv.forward(&xf, &Ctx::streaming(f == 0))?);
        }
        let streamed = Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?;

        assert_eq!(full.dims(), streamed.dims());
        let max_diff = (full - streamed)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        assert!(
            max_diff < 1e-5,
            "streaming vs single-pass max abs diff = {max_diff}"
        );
        Ok(())
    }
}
