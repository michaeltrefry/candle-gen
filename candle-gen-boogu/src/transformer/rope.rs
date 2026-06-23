//! Boogu DiT 3-axis (t, h, w) unified RoPE — the OmniGen2 `BooguImageDoubleStreamRotaryPosEmbed`.
//! Port of `mlx-gen-boogu`'s `transformer/rope.rs`.
//!
//! Two things differ from the Qwen3-VL text encoder's RoPE and matter for parity:
//!  1. **Complex *interleaved* rotation** (`apply_rotary_emb(use_real=False)`, the "lumina" branch):
//!     adjacent dims `(2k, 2k+1)` form a complex pair `x[2k] + i·x[2k+1]` rotated by `e^{iθ_k}`
//!     (GPT-J / interleaved), *not* the text encoder's half-split `[x1, x2] → [-x2, x1]`.
//!  2. **Three position axes**: per token the rotary frequency index `k ∈ [0, 60)` is grouped into
//!     three contiguous blocks of 20 (`axes_dim_rope = [40,40,40]` ⇒ 20 complex freqs each), one per
//!     axis. Text tokens use position `(i, i, i)`; image patch tokens use `(cap_len, row, col)`.
//!
//! Each axis shares the same 20-vector of inverse frequencies `θ^(−2j/40)` (`θ = 10000`). The
//! `cos`/`sin` tables are built on the host in f32 (the reference builds the freqs in f32), uploaded
//! to the model device, and sliced into the text-only / image-only sub-ranges.

use candle_gen::candle_core::{Device, Result, Tensor, D};

/// Precomputed `cos`/`sin` rotary tables for one forward pass.
///
/// Layout is `[1, cap_len + ref_len + img_len, head_dim/2]` (f32) in the joint
/// `[instruct; ref-image; noise-image]` order. For text-to-image there is no reference image
/// (`ref_len == 0`) and the layout collapses to `[instruct; image]`.
pub struct RopeTables {
    cos: Tensor,
    sin: Tensor,
    cap_len: usize,
    ref_len: usize,
}

impl RopeTables {
    /// Build the joint table for a text-to-image forward (no reference images): `cap_len` text
    /// positions followed by an `h_tokens × w_tokens` image grid (row-major, `h` outer).
    #[allow(clippy::too_many_arguments)]
    pub fn build_t2i(
        cap_len: usize,
        h_tokens: usize,
        w_tokens: usize,
        axes_dim: usize,
        theta: f32,
        device: &Device,
    ) -> Result<Self> {
        let mut positions = Vec::with_capacity(cap_len + h_tokens * w_tokens);
        text_positions(&mut positions, cap_len);
        grid_positions(&mut positions, cap_len as f32, h_tokens, w_tokens);
        from_positions(&positions, axes_dim, theta, cap_len, 0, device)
    }

    /// Build the joint table for an **edit** forward with one or more reference images (the OmniGen2
    /// unified-RoPE multi-image scheme): `cap_len` text positions, then each reference's `rh × rw`
    /// grid placed at its own t-axis position `pe_shift` (starting at `cap_len` and advancing by
    /// `max(rh, rw)` after each reference), then the `h × w` target grid at the final
    /// `pe_shift = cap_len + Σ max(rh_j, rw_j)` — matching the `[instruct; ref₀; …; ref_{N-1}; noise]`
    /// packing the DiT runs the single-stream over. `ref_grids` are `(rh_tokens, rw_tokens)` per
    /// reference, in order; a single-element slice reproduces the single-reference table exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn build_edit(
        cap_len: usize,
        ref_grids: &[(usize, usize)],
        h_tokens: usize,
        w_tokens: usize,
        axes_dim: usize,
        theta: f32,
        device: &Device,
    ) -> Result<Self> {
        let ref_len: usize = ref_grids.iter().map(|(h, w)| h * w).sum();
        let mut positions = Vec::with_capacity(cap_len + ref_len + h_tokens * w_tokens);
        text_positions(&mut positions, cap_len);
        // Each reference grid at its own t-axis position; `pe_shift` advances by the reference's longer
        // side after each (the OmniGen2 `pe_shift += max(ref_H_tokens, ref_W_tokens)`), so references
        // occupy disjoint t-ranges and the noise grid follows the last one.
        let mut pe_shift = cap_len;
        for &(rh, rw) in ref_grids {
            grid_positions(&mut positions, pe_shift as f32, rh, rw);
            pe_shift += rh.max(rw);
        }
        grid_positions(&mut positions, pe_shift as f32, h_tokens, w_tokens);
        from_positions(&positions, axes_dim, theta, cap_len, ref_len, device)
    }

    /// `(cos, sin)` for the text tokens only (`context_refiner`).
    pub fn text(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.cos.narrow(1, 0, self.cap_len)?,
            self.sin.narrow(1, 0, self.cap_len)?,
        ))
    }

    /// `(cos, sin)` for **all** reference-image patch tokens (the full `[ref₀; …; ref_{N-1}]` block).
    pub fn ref_image(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.cos.narrow(1, self.cap_len, self.ref_len)?,
            self.sin.narrow(1, self.cap_len, self.ref_len)?,
        ))
    }

    /// `(cos, sin)` for one reference image's tokens — the sub-range at local offset `local_start`
    /// (relative to the start of the reference block) of length `len`. Used to refine each reference
    /// independently (the OmniGen2 per-image batched `ref_image_refiner`: no cross-image attention).
    pub fn ref_image_slice(&self, local_start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let start = self.cap_len + local_start;
        Ok((
            self.cos.narrow(1, start, len)?,
            self.sin.narrow(1, start, len)?,
        ))
    }

    /// `(cos, sin)` for the target (noise) patch tokens only (`noise_refiner`). These sit after the
    /// reference block, so the range is `[cap_len + ref_len, end)`.
    pub fn image(&self) -> Result<(Tensor, Tensor)> {
        let start = self.cap_len + self.ref_len;
        let len = self.cos.dim(1)? - start;
        Ok((
            self.cos.narrow(1, start, len)?,
            self.sin.narrow(1, start, len)?,
        ))
    }

    /// `(cos, sin)` for the combined image sequence `[ref; noise]` (the double-stream image
    /// self-attention). For T2I (`ref_len == 0`) this equals [`Self::image`].
    pub fn combined_image(&self) -> Result<(Tensor, Tensor)> {
        let start = self.cap_len;
        let len = self.cos.dim(1)? - start;
        Ok((
            self.cos.narrow(1, start, len)?,
            self.sin.narrow(1, start, len)?,
        ))
    }

    /// `(cos, sin)` for the full joint `[text; ref; noise]` sequence (double / single stream).
    pub fn joint(&self) -> (Tensor, Tensor) {
        (self.cos.clone(), self.sin.clone())
    }
}

/// Push `cap_len` text positions `(i, i, i)`.
fn text_positions(out: &mut Vec<(f32, f32, f32)>, cap_len: usize) {
    for i in 0..cap_len {
        out.push((i as f32, i as f32, i as f32));
    }
}

/// Push an `h × w` row-major image grid at a fixed t-axis position: `(t, row, col)`.
fn grid_positions(out: &mut Vec<(f32, f32, f32)>, t: f32, h: usize, w: usize) {
    for r in 0..h {
        for c in 0..w {
            out.push((t, r as f32, c as f32));
        }
    }
}

/// Build the `cos`/`sin` tables from 3-axis positions: each rotary freq index `k ∈ [0, 3·axes_dim/2)`
/// is grouped into three contiguous blocks of `axes_dim/2`, one per axis, all sharing the inverse
/// frequencies `θ^(−2j/axes_dim)`.
fn from_positions(
    positions: &[(f32, f32, f32)],
    axes_dim: usize,
    theta: f32,
    cap_len: usize,
    ref_len: usize,
    device: &Device,
) -> Result<RopeTables> {
    let half_axis = axes_dim / 2; // 20 complex freqs per axis
    let half = half_axis * 3; // 60 for head_dim 120
    let inv: Vec<f32> = (0..half_axis)
        .map(|j| theta.powf(-(2.0 * j as f32) / axes_dim as f32))
        .collect();

    let total = positions.len();
    let mut cos = vec![0f32; total * half];
    let mut sin = vec![0f32; total * half];
    for (t, &(p0, p1, p2)) in positions.iter().enumerate() {
        for k in 0..half {
            let p = match k / half_axis {
                0 => p0,
                1 => p1,
                _ => p2,
            };
            let angle = p * inv[k % half_axis];
            cos[t * half + k] = angle.cos();
            sin[t * half + k] = angle.sin();
        }
    }

    let cos = Tensor::from_vec(cos, (1, total, half), device)?;
    let sin = Tensor::from_vec(sin, (1, total, half), device)?;
    Ok(RopeTables {
        cos,
        sin,
        cap_len,
        ref_len,
    })
}

/// Apply the complex-interleaved rotary embedding to `x` in `[b, s, heads, head_dim]` layout.
///
/// `cos`/`sin` are `[1, s, head_dim/2]` (f32, broadcast over heads). For each adjacent pair
/// `(x[2k], x[2k+1])`:
///   `out[2k]   = x[2k]·cos_k − x[2k+1]·sin_k`
///   `out[2k+1] = x[2k]·sin_k + x[2k+1]·cos_k`
/// Computed in f32 (the reference upcasts), then cast back to `x`'s dtype.
pub fn apply_interleaved_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let (b, s, h, hd) = x.dims4()?;
    let half = hd / 2;

    // cos/sin: [1, s, half] → [1, s, 1, half] (broadcast over heads + the pair axis). They arrive as
    // `narrow`ed slices of the rope table, so contiguate before the reshape.
    let cos = cos.contiguous()?.reshape((1, s, 1, half))?;
    let sin = sin.contiguous()?.reshape((1, s, 1, half))?;

    // [b, s, h, hd] → [b, s, h, half, 2]; the last axis holds the (even, odd) complex pair.
    let xr = x
        .to_dtype(candle_gen::candle_core::DType::F32)?
        .reshape((b, s, h, half, 2))?;
    // `narrow` yields a strided view; contiguate before the reshape (candle reshape needs contiguous).
    let xe = xr.narrow(4, 0, 1)?.contiguous()?.reshape((b, s, h, half))?;
    let xo = xr.narrow(4, 1, 1)?.contiguous()?.reshape((b, s, h, half))?;

    let out_e = (xe.broadcast_mul(&cos)? - xo.broadcast_mul(&sin)?)?;
    let out_o = (xe.broadcast_mul(&sin)? + xo.broadcast_mul(&cos)?)?;

    // Re-interleave: stack on a new trailing axis → [b, s, h, half, 2] → [b, s, h, hd].
    let out = Tensor::stack(&[&out_e, &out_o], D::Minus1)?.reshape((b, s, h, hd))?;
    out.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    // head_dim 120 ⇒ axes_dim 40 per axis ⇒ half = 3·20 = 60.
    const AXES_DIM: usize = 40;
    const HALF: usize = 60;
    const THETA: f32 = 10000.0;

    #[test]
    fn build_edit_multi_ref_token_accounting() {
        let dev = Device::Cpu;
        let cap = 3;
        // Two references of different shapes (2×2 and 3×1 tokens) + a 4×4 noise grid.
        let refs = [(2usize, 2usize), (3usize, 1usize)];
        let (ht, wt) = (4usize, 4usize);
        let r = RopeTables::build_edit(cap, &refs, ht, wt, AXES_DIM, THETA, &dev).unwrap();

        let ref_len: usize = refs.iter().map(|(h, w)| h * w).sum(); // 4 + 3 = 7
        let noise_len = ht * wt; // 16
        let total = cap + ref_len + noise_len; // 26
        assert_eq!(r.joint().0.dims(), &[1, total, HALF]);

        assert_eq!(r.text().unwrap().0.dim(1).unwrap(), cap);
        assert_eq!(r.ref_image().unwrap().0.dim(1).unwrap(), ref_len);
        assert_eq!(r.image().unwrap().0.dim(1).unwrap(), noise_len);
        assert_eq!(
            r.combined_image().unwrap().0.dim(1).unwrap(),
            ref_len + noise_len
        );
        // Per-image ref slices partition the reference block in order.
        assert_eq!(r.ref_image_slice(0, 4).unwrap().0.dim(1).unwrap(), 4);
        assert_eq!(r.ref_image_slice(4, 3).unwrap().0.dim(1).unwrap(), 3);
    }

    #[test]
    fn build_edit_single_ref_matches_legacy_shape() {
        // A one-element `ref_grids` slice reproduces the single-reference table.
        let dev = Device::Cpu;
        let r = RopeTables::build_edit(5, &[(8, 8)], 16, 16, AXES_DIM, THETA, &dev).unwrap();
        assert_eq!(r.text().unwrap().0.dim(1).unwrap(), 5);
        assert_eq!(r.ref_image().unwrap().0.dim(1).unwrap(), 64);
        assert_eq!(r.image().unwrap().0.dim(1).unwrap(), 256);
    }
}
