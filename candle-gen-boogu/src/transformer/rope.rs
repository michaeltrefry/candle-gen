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

    /// Build the joint table for an **edit** forward (one reference image): `cap_len` text positions,
    /// then a `ref_h × ref_w` reference grid at t-axis `cap_len`, then the `h × w` target grid at
    /// t-axis `cap_len + max(ref_h, ref_w)` (the reference's `pe_shift` advance) — matching the
    /// `[instruct; ref; noise]` packing the DiT runs the single-stream over.
    #[allow(clippy::too_many_arguments)]
    pub fn build_edit(
        cap_len: usize,
        ref_h: usize,
        ref_w: usize,
        h_tokens: usize,
        w_tokens: usize,
        axes_dim: usize,
        theta: f32,
        device: &Device,
    ) -> Result<Self> {
        let ref_len = ref_h * ref_w;
        let mut positions = Vec::with_capacity(cap_len + ref_len + h_tokens * w_tokens);
        text_positions(&mut positions, cap_len);
        grid_positions(&mut positions, cap_len as f32, ref_h, ref_w);
        let noise_t = (cap_len + ref_h.max(ref_w)) as f32;
        grid_positions(&mut positions, noise_t, h_tokens, w_tokens);
        from_positions(&positions, axes_dim, theta, cap_len, ref_len, device)
    }

    /// `(cos, sin)` for the text tokens only (`context_refiner`).
    pub fn text(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.cos.narrow(1, 0, self.cap_len)?,
            self.sin.narrow(1, 0, self.cap_len)?,
        ))
    }

    /// `(cos, sin)` for the reference-image patch tokens only (`ref_image_refiner`).
    pub fn ref_image(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.cos.narrow(1, self.cap_len, self.ref_len)?,
            self.sin.narrow(1, self.cap_len, self.ref_len)?,
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
