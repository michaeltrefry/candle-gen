//! SVD `TransformerSpatioTemporalModel` — a spatial `BasicTransformerBlock` (self-attn + cross-attn to
//! the CLIP `image_embeds` + GEGLU ff) and a `TemporalBasicTransformerBlock` (ff_in + self-attn over
//! the frame axis + cross-attn + ff), blended per layer by an `AlphaBlender`
//! (`merge_strategy="learned_with_images"`, no switch → `σ(mix)·spatial + (1−σ)·temporal`). candle port
//! of diffusers `transformer_temporal.TransformerSpatioTemporalModel`. NCHW I/O.

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::ops::{sigmoid, softmax_last_dim};
use candle_gen::candle_nn::{
    layer_norm, linear, linear_no_bias, LayerNorm, Linear, Module, VarBuilder,
};

use crate::embeddings::{sinusoidal_timestep, TimestepEmbedding};
use crate::vae::GroupNormW;

const GN_GROUPS: usize = 32;
const GN_EPS: f64 = 1e-6;
const LN_EPS: f64 = 1e-5;

/// GEGLU feed-forward: `proj` is `ff.net.0.proj` (`[2·inner, D]`), split into value/gate halves;
/// `out` is `ff.net.2`. `value · gelu(gate) → out`.
fn geglu(x: &Tensor, proj: &Linear, out: &Linear) -> Result<Tensor> {
    let p = proj.forward(x)?; // [..., 2·inner]
    let halves = p.chunk(2, D::Minus1)?;
    let y = (&halves[0] * halves[1].gelu_erf()?)?;
    out.forward(&y)
}

/// Multi-head attention: bias-free q/k/v, biased `to_out.0`, no mask. `head_dim = inner/heads`,
/// `scale = head_dim^-0.5`. Self-attn passes `context == x`; cross-attn passes the memory.
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    /// `query_dim` = the token channel dim; `kv_dim` = the context dim (== query_dim for self-attn);
    /// `inner` = `heads · head_dim` (== query_dim for SVD).
    fn load(
        query_dim: usize,
        kv_dim: usize,
        inner: usize,
        heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = inner / heads;
        Ok(Self {
            q: linear_no_bias(query_dim, inner, vb.pp("to_q"))?,
            k: linear_no_bias(kv_dim, inner, vb.pp("to_k"))?,
            v: linear_no_bias(kv_dim, inner, vb.pp("to_v"))?,
            out: linear(inner, query_dim, vb.pp("to_out").pp("0"))?,
            heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[B, Lq, Dq]`; `context`: `[B, Lk, Dkv]`.
    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, lq, _) = x.dims3()?;
        let lk = context.dim(1)?;
        let split = |t: Tensor, n: usize| -> Result<Tensor> {
            t.reshape((b, n, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q.forward(x)?, lq)?;
        let k = split(self.k.forward(context)?, lk)?;
        let v = split(self.v.forward(context)?, lk)?;
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn = softmax_last_dim(&attn)?;
        let o = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, lq, self.heads * self.head_dim))?;
        self.out.forward(&o)
    }
}

/// Spatial `BasicTransformerBlock`: pre-norm self-attn → pre-norm cross-attn → pre-norm GEGLU ff.
struct BasicBlock {
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    ff_proj: Linear,
    ff_out: Linear,
}

impl BasicBlock {
    fn load(dim: usize, cross_dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let ff_inner = ff_inner_dim(dim);
        Ok(Self {
            norm1: layer_norm(dim, LN_EPS, vb.pp("norm1"))?,
            attn1: Attention::load(dim, dim, dim, heads, vb.pp("attn1"))?,
            norm2: layer_norm(dim, LN_EPS, vb.pp("norm2"))?,
            attn2: Attention::load(dim, cross_dim, dim, heads, vb.pp("attn2"))?,
            norm3: layer_norm(dim, LN_EPS, vb.pp("norm3"))?,
            ff_proj: linear(dim, 2 * ff_inner, vb.pp("ff").pp("net").pp("0").pp("proj"))?,
            ff_out: linear(ff_inner, dim, vb.pp("ff").pp("net").pp("2"))?,
        })
    }

    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let y = self.norm1.forward(x)?;
        let x = (x + self.attn1.forward(&y, &y)?)?;
        let y = self.norm2.forward(&x)?;
        let x = (&x + self.attn2.forward(&y, context)?)?;
        let y = self.norm3.forward(&x)?;
        &x + geglu(&y, &self.ff_proj, &self.ff_out)?
    }
}

/// `TemporalBasicTransformerBlock`: reshape to attend over the frame axis, then ff_in (+residual) →
/// self-attn → cross-attn → ff (+residual), then reshape back.
struct TemporalBlock {
    norm_in: LayerNorm,
    ffin_proj: Linear,
    ffin_out: Linear,
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    ff_proj: Linear,
    ff_out: Linear,
}

impl TemporalBlock {
    fn load(dim: usize, cross_dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let ff_inner = ff_inner_dim(dim);
        Ok(Self {
            norm_in: layer_norm(dim, LN_EPS, vb.pp("norm_in"))?,
            ffin_proj: linear(
                dim,
                2 * ff_inner,
                vb.pp("ff_in").pp("net").pp("0").pp("proj"),
            )?,
            ffin_out: linear(ff_inner, dim, vb.pp("ff_in").pp("net").pp("2"))?,
            norm1: layer_norm(dim, LN_EPS, vb.pp("norm1"))?,
            attn1: Attention::load(dim, dim, dim, heads, vb.pp("attn1"))?,
            norm2: layer_norm(dim, LN_EPS, vb.pp("norm2"))?,
            attn2: Attention::load(dim, cross_dim, dim, heads, vb.pp("attn2"))?,
            norm3: layer_norm(dim, LN_EPS, vb.pp("norm3"))?,
            ff_proj: linear(dim, 2 * ff_inner, vb.pp("ff").pp("net").pp("0").pp("proj"))?,
            ff_out: linear(ff_inner, dim, vb.pp("ff").pp("net").pp("2"))?,
        })
    }

    /// `x`: `[B·F, seq, C]`; `context`: `[B·seq, ctx, Dkv]` (frame-0 memory, broadcast over seq).
    fn forward(&self, x: &Tensor, num_frames: usize, context: &Tensor) -> Result<Tensor> {
        let (bf, seq, c) = x.dims3()?;
        let b = bf / num_frames;
        // [B·F, seq, C] → [B, F, seq, C] → [B, seq, F, C] → [B·seq, F, C] (attend over frames).
        let h = x
            .reshape((b, num_frames, seq, c))?
            .transpose(1, 2)?
            .reshape((b * seq, num_frames, c))?
            .contiguous()?;
        let residual = h.clone();
        let n = self.norm_in.forward(&h)?;
        let h = (geglu(&n, &self.ffin_proj, &self.ffin_out)? + residual)?; // is_res
        let n = self.norm1.forward(&h)?;
        let h = (self.attn1.forward(&n, &n)? + h)?;
        let n = self.norm2.forward(&h)?;
        let h = (self.attn2.forward(&n, context)? + h)?;
        let n = self.norm3.forward(&h)?;
        let h = (geglu(&n, &self.ff_proj, &self.ff_out)? + h)?;
        // back to [B·F, seq, C].
        h.reshape((b, seq, num_frames, c))?
            .transpose(1, 2)?
            .reshape((bf, seq, c))?
            .contiguous()
    }
}

/// The full spatio-temporal transformer at one resolution.
pub struct TransformerSpatioTemporal {
    norm: GroupNormW,
    proj_in: Linear,
    blocks: Vec<BasicBlock>,
    temporal_blocks: Vec<TemporalBlock>,
    time_pos_embed: TimestepEmbedding,
    mix_factor: Tensor,
    proj_out: Linear,
    in_channels: usize,
}

impl TransformerSpatioTemporal {
    /// `vb` addresses the `attentions.{i}` module; `heads` is the block's head count; `cross_dim` is the
    /// CLIP image-embed dim (1024).
    pub fn load(
        in_channels: usize,
        cross_dim: usize,
        heads: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let blocks = (0..num_layers)
            .map(|i| {
                BasicBlock::load(
                    in_channels,
                    cross_dim,
                    heads,
                    vb.pp("transformer_blocks").pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let temporal_blocks = (0..num_layers)
            .map(|i| {
                TemporalBlock::load(
                    in_channels,
                    cross_dim,
                    heads,
                    vb.pp("temporal_transformer_blocks").pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            norm: GroupNormW::load(in_channels, GN_GROUPS, GN_EPS, vb.pp("norm"))?,
            proj_in: linear(in_channels, in_channels, vb.pp("proj_in"))?,
            blocks,
            temporal_blocks,
            // diffusers: TimestepEmbedding(in_channels, in_channels·4, out_dim=in_channels).
            time_pos_embed: TimestepEmbedding::load(
                in_channels,
                in_channels * 4,
                in_channels,
                vb.pp("time_pos_embed"),
            )?,
            mix_factor: vb.get(1, "time_mixer.mix_factor")?,
            proj_out: linear(in_channels, in_channels, vb.pp("proj_out"))?,
            in_channels,
        })
    }

    /// `x`: NCHW `[B·F, C, H, W]`; `context`: CLIP image memory `[B·F, ctx, Dkv]`.
    pub fn forward(&self, x: &Tensor, context: &Tensor, num_frames: usize) -> Result<Tensor> {
        let (bf, c, h_, w_) = x.dims4()?;
        let b = bf / num_frames;
        let seq = h_ * w_;

        // Temporal cross-attn memory: take frame-0's context, broadcast over the H·W tokens.
        let (_, ctx_seq, cd) = context.dims3()?;
        let tctx = context.reshape((b, num_frames, ctx_seq, cd))?;
        let tctx = tctx.narrow(1, 0, 1)?; // [B, 1, ctx, cd] (frame 0)
        let tctx = tctx
            .reshape((b, 1, ctx_seq, cd))?
            .broadcast_as((b, seq, ctx_seq, cd))?
            .reshape((b * seq, ctx_seq, cd))?
            .contiguous()?;

        let residual = x.clone();
        let n = self.norm.forward(x)?;
        // [B·F, C, H, W] → [B·F, seq, C] tokens.
        let tokens = n.reshape((bf, c, seq))?.transpose(1, 2)?.contiguous()?;
        let mut tokens = self.proj_in.forward(&tokens)?;

        // Per-frame position embedding: arange(F) tiled over the batch.
        let nframes: Vec<f32> = (0..b)
            .flat_map(|_| (0..num_frames).map(|f| f as f32))
            .collect();
        let nframes = Tensor::from_vec(nframes, bf, x.device())?;
        let emb = self.time_pos_embed.forward(&sinusoidal_timestep(
            &nframes,
            self.in_channels,
            x.device(),
        )?)?;
        let emb = emb.reshape((bf, 1, c))?; // [B·F, 1, C]

        let alpha = sigmoid(&self.mix_factor)?; // [1]
        let one_minus = alpha.affine(-1.0, 1.0)?; // 1 − alpha
        for (block, temporal) in self.blocks.iter().zip(&self.temporal_blocks) {
            tokens = block.forward(&tokens, context)?;
            let mix = tokens.broadcast_add(&emb)?;
            let mix = temporal.forward(&mix, num_frames, &tctx)?;
            // learned_with_images, no switch → α·spatial + (1−α)·temporal.
            tokens = tokens
                .broadcast_mul(&alpha)?
                .add(&mix.broadcast_mul(&one_minus)?)?;
        }

        let tokens = self.proj_out.forward(&tokens)?;
        // [B·F, seq, C] → [B·F, C, H, W].
        let out = tokens.transpose(1, 2)?.reshape((bf, c, h_, w_))?;
        out + residual
    }
}

/// diffusers feed-forward inner dim = `dim · 4` (the SVD blocks use `mult=4`, no `ff_inner_dim`
/// override). The GEGLU projection is `[2·inner, dim]`.
fn ff_inner_dim(dim: usize) -> usize {
    dim * 4
}
