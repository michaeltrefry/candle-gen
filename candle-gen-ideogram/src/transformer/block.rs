//! Ideogram 4 DiT block: attention + SwiGLU MLP with AdaLN "sandwich" norms (a pre-norm scaled by
//! `1+scale`, a post-norm gated by `tanh(gate)`), full segment-masked attention, per-head q/k
//! RMSNorm, and interleaved 3D MRoPE. Port of `Ideogram4Attention` / `Ideogram4MLP` /
//! `Ideogram4TransformerBlock`.

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, Module};

use super::rmsnorm;
use crate::loader::{linear, Weights};

/// Per-head q/k RMSNorm eps (upstream `Ideogram4Attention`, hardcoded 1e-5).
const ATTN_QK_EPS: f64 = 1e-5;

// ── Attention ────────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Attention {
    qkv: Linear,
    o: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl Ideogram4Attention {
    pub fn load(w: &Weights, prefix: &str, num_heads: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            qkv: linear(w, &format!("{prefix}.qkv"), false)?,
            o: linear(w, &format!("{prefix}.o"), false)?,
            norm_q: w.get(&format!("{prefix}.norm_q.weight"))?,
            norm_k: w.get(&format!("{prefix}.norm_k.weight"))?,
            num_heads,
            head_dim,
        })
    }

    /// `x`: `[B, L, emb]`; `cos`/`sin`: `[B, L, head_dim]`; `mask`: additive `[B, 1, L, L]`.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, hd) = (self.num_heads, self.head_dim);

        // qkv → [B, L, 3, H, hd] → q,k,v [B, L, H, hd]
        let qkv = self.qkv.forward(x)?.reshape((b, s, 3, nh, hd))?;
        let q = qkv.narrow(2, 0, 1)?.contiguous()?.reshape((b, s, nh, hd))?;
        let k = qkv.narrow(2, 1, 1)?.contiguous()?.reshape((b, s, nh, hd))?;
        let v = qkv.narrow(2, 2, 1)?.contiguous()?.reshape((b, s, nh, hd))?;

        // Per-head q/k RMSNorm over the head dim, before transpose + RoPE.
        let q = rmsnorm(&q, &self.norm_q, ATTN_QK_EPS)?;
        let k = rmsnorm(&k, &self.norm_k, ATTN_QK_EPS)?;

        // [B,L,H,hd] → [B,H,L,hd]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // [B,H,L,L]
        let scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?;
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?; // [B,H,L,hd]
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
        self.o.forward(&o)
    }
}

/// HF half-split RoPE in `[B, H, L, hd]` layout: `cos`/`sin` `[B, L, hd]` → broadcast over heads.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(1)?.to_dtype(x.dtype())?; // [B,1,L,hd]
    let sin = sin.unsqueeze(1)?.to_dtype(x.dtype())?;
    let chunks = x.chunk(2, D::Minus1)?;
    let x1 = chunks[0].contiguous()?;
    let x2 = chunks[1].contiguous()?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(&cos)? + rot.broadcast_mul(&sin)?
}

// ── SwiGLU MLP ───────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Ideogram4Mlp {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: linear(w, &format!("{prefix}.w1"), false)?,
            w2: linear(w, &format!("{prefix}.w2"), false)?,
            w3: linear(w, &format!("{prefix}.w3"), false)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.w1.forward(x)?.silu()? * self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }
}

// ── Block ────────────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Block {
    attention: Ideogram4Attention,
    feed_forward: Ideogram4Mlp,
    attention_norm1: Tensor,
    attention_norm2: Tensor,
    ffn_norm1: Tensor,
    ffn_norm2: Tensor,
    adaln_modulation: Linear,
    eps: f64,
}

impl Ideogram4Block {
    pub fn load(
        w: &Weights,
        prefix: &str,
        num_heads: usize,
        head_dim: usize,
        norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            attention: Ideogram4Attention::load(
                w,
                &format!("{prefix}.attention"),
                num_heads,
                head_dim,
            )?,
            feed_forward: Ideogram4Mlp::load(w, &format!("{prefix}.feed_forward"))?,
            attention_norm1: w.get(&format!("{prefix}.attention_norm1.weight"))?,
            attention_norm2: w.get(&format!("{prefix}.attention_norm2.weight"))?,
            ffn_norm1: w.get(&format!("{prefix}.ffn_norm1.weight"))?,
            ffn_norm2: w.get(&format!("{prefix}.ffn_norm2.weight"))?,
            adaln_modulation: linear(w, &format!("{prefix}.adaln_modulation"), true)?,
            eps: norm_eps,
        })
    }

    /// `x`: `[B, L, emb]`; `adaln_input`: `[B, 1, adaln_dim]`; `cos`/`sin`: `[B, L, head_dim]`;
    /// `mask`: additive `[B, 1, L, L]`.
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        adaln_input: &Tensor,
    ) -> Result<Tensor> {
        let mod_ = self.adaln_modulation.forward(adaln_input)?; // [B,1,4*emb]
        let chunks = mod_.chunk(4, D::Minus1)?;
        let scale_msa = (chunks[0].contiguous()? + 1.0)?;
        let gate_msa = chunks[1].contiguous()?.tanh()?;
        let scale_mlp = (chunks[2].contiguous()? + 1.0)?;
        let gate_mlp = chunks[3].contiguous()?.tanh()?;

        let normed = rmsnorm(x, &self.attention_norm1, self.eps)?.broadcast_mul(&scale_msa)?;
        let attn_out = self.attention.forward(&normed, cos, sin, mask)?;
        let x =
            (x + rmsnorm(&attn_out, &self.attention_norm2, self.eps)?.broadcast_mul(&gate_msa)?)?;

        let normed2 = rmsnorm(&x, &self.ffn_norm1, self.eps)?.broadcast_mul(&scale_mlp)?;
        let ff = self.feed_forward.forward(&normed2)?;
        let x = (&x + rmsnorm(&ff, &self.ffn_norm2, self.eps)?.broadcast_mul(&gate_mlp)?)?;
        Ok(x)
    }
}
