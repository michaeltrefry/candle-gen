//! Shared nn helpers for the SeedVR2 candle port, matching the MLX reference semantics:
//! GroupNorm/RMSNorm compute in **f32** (cast back to the input dtype), dense scaled-dot-product
//! attention, a `[out,in]`-weight linear (`y = x·Wᵀ + b`), and the tanh-GELU.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;

/// `[out,in]`-weight dense layer: `y = x·Wᵀ (+ b)`. Flattens all leading dims into one 2-D GEMM and
/// reshapes back — candle's `matmul` rejects the non-contiguous broadcasted rhs that `broadcast_matmul`
/// produces for a high-rank `x` (e.g. the 5-D patchified tokens), and the flattened GEMM is faster.
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let wt = w.t()?.contiguous()?; // [in, out]
    let (in_dim, out_dim) = (wt.dim(0)?, wt.dim(1)?);
    let dims = x.dims().to_vec();
    let lead: usize = dims[..dims.len() - 1].iter().product();
    let y = x.contiguous()?.reshape((lead, in_dim))?.matmul(&wt)?; // [lead, out]
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(out_dim);
    let y = y.reshape(out_shape)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

/// GroupNorm over `[N, C, *spatial]` (channels in dim 1, any trailing rank), computed in f32 with a
/// learnable `[C]` weight/bias. Matches mlx's channels-last f32 GroupNorm bit-for-bit at f32.
pub fn group_norm(x: &Tensor, w: &Tensor, b: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    let sh = x.dims().to_vec();
    let (n, c) = (sh[0], sh[1]);
    let rest: usize = sh[2..].iter().product();
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let g = xf.reshape((n, groups, (c / groups) * rest))?;
    let mean = g.mean_keepdim(2)?;
    let centered = g.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?;
    let normed = centered
        .broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?
        .reshape(sh.clone())?;
    // affine: reshape [C] → [1, C, 1, …] to broadcast over the trailing dims.
    let mut ws = vec![1usize; sh.len()];
    ws[1] = c;
    let wv = w.to_dtype(DType::F32)?.reshape(ws.clone())?;
    let bv = b.to_dtype(DType::F32)?.reshape(ws)?;
    normed.broadcast_mul(&wv)?.broadcast_add(&bv)?.to_dtype(dt)
}

/// RMSNorm over the last dim with a `[dim]` weight, computed in f32.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&ms.affine(1.0, eps)?.sqrt()?)?;
    normed.broadcast_mul(&w.to_dtype(DType::F32)?)?.to_dtype(dt)
}

/// SiLU (x·sigmoid(x)).
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let sig = (x.neg()?.exp()? + 1.0)?.recip()?;
    x.mul(&sig)
}

/// tanh-approximation GELU (the reference `gelu_tanh`).
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

/// Dense scaled-dot-product attention over `[B, H, S, D]` (no mask): `softmax(q·kᵀ·scale)·v`.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let q = q.contiguous()?;
    let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?; // [B,H,D,S]
    let scores = (q.matmul(&kt)? * scale)?;
    let attn = softmax_last_dim(&scores)?;
    attn.matmul(&v.contiguous()?)
}
