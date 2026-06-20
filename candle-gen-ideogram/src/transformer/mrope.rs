//! Interleaved 3D MRoPE (`Ideogram4MRoPE`), ported from `mlx-gen-ideogram`. For each rotary
//! frequency index `d` (`0..head_dim/2`) the position used is the **t** axis by default, overridden
//! to **h** at `d ≡ 1 (mod 3)` for `d < section_h·3` and to **w** at `d ≡ 2 (mod 3)` for
//! `d < section_w·3` — exactly the upstream `freqs_t[..., idx] = freqs[axis][..., idx]` interleave.
//! The text-only TE path (1-D) is a special case where t = h = w; the DiT mixes image-grid (t,h,w)
//! positions so the 3 axes differ.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};

pub struct Ideogram4MRoPE {
    /// `[1, 1, head_dim/2]` inverse frequencies and 0/1 axis selectors (t/h/w) per frequency index.
    inv_freq: Tensor,
    mask_t: Tensor,
    mask_h: Tensor,
    mask_w: Tensor,
}

impl Ideogram4MRoPE {
    pub fn new(
        head_dim: usize,
        theta: f32,
        mrope_section: [usize; 3],
        device: &Device,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let mut inv = vec![0f32; half];
        let mut mt = vec![0f32; half];
        let mut mh = vec![0f32; half];
        let mut mw = vec![0f32; half];
        let (len_h, len_w) = (mrope_section[1] * 3, mrope_section[2] * 3);
        for d in 0..half {
            // arange(0, head_dim, 2)[d] / head_dim = 2d / head_dim.
            inv[d] = theta.powf(-(2.0 * d as f32) / head_dim as f32);
            let axis = if d % 3 == 1 && d < len_h {
                1
            } else if d % 3 == 2 && d < len_w {
                2
            } else {
                0
            };
            match axis {
                1 => mh[d] = 1.0,
                2 => mw[d] = 1.0,
                _ => mt[d] = 1.0,
            }
        }
        let shape = (1, 1, half);
        Ok(Self {
            inv_freq: Tensor::from_vec(inv, shape, device)?,
            mask_t: Tensor::from_vec(mt, shape, device)?,
            mask_h: Tensor::from_vec(mh, shape, device)?,
            mask_w: Tensor::from_vec(mw, shape, device)?,
        })
    }

    /// `position_ids`: `[B, L, 3]` (t, h, w). Returns `(cos, sin)` `[B, L, head_dim]` (f32).
    pub fn forward(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let pos = position_ids.to_dtype(DType::F32)?;
        // sel[b,l,d] = pos_axis_of(d)[b,l] — broadcast [B,L,1] · [1,1,half] → [B,L,half].
        let parts = pos.chunk(3, 2)?; // 3 × [B, L, 1]
        let sel = (parts[0].broadcast_mul(&self.mask_t)?
            + parts[1].broadcast_mul(&self.mask_h)?
            + parts[2].broadcast_mul(&self.mask_w)?)?;
        let freqs = sel.broadcast_mul(&self.inv_freq)?; // [B, L, half]
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [B, L, head_dim]
        Ok((emb.cos()?, emb.sin()?))
    }
}
