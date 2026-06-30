//! The Qwen-Image **dual-stream MMDiT** (60 blocks). Port of `mlx-gen-qwen-image`'s `transformer/`,
//! run in candle bf16 (the native checkpoint dtype; ~41 GB).
//!
//! Shape anchors: `inner_dim = 3072` (24 heads × 128), `in_channels = 64`, `out_channels = 16`,
//! `joint_attention_dim = 3584`. Conditioning is **timestep-only** (no text pooling). Each block runs
//! both an image and a text stream with per-stream AdaLN modulation (`img_mod`/`txt_mod` → 2 sets of
//! shift/scale/gate), a JOINT attention over the **`[txt, img]`** sequence (text first) with
//! interleaved 3-axis RoPE (see [`crate::rope`]), and a GELU-tanh FFN per stream.
//!
//! Parity-load-bearing: all LayerNorms are affine-free, eps 1e-6; q/k RMSNorm is per-head (128-dim),
//! eps 1e-6; the top-level `txt_norm` is a standard RMSNorm applied before `txt_in`; `norm_out.linear`
//! is loaded **bias-less** (the checkpoint bias is ignored); the timestep proj scales by ×1000 inside
//! the sinusoid; all the other Linears are **biased**.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    linear, linear_no_bias, ops::softmax_last_dim, rms_norm, Linear, Module, RmsNorm, VarBuilder,
};

use crate::config::TransformerConfig;
use crate::rope::{apply_rope, QwenRope};

const EPS: f64 = 1e-6;

/// Affine-free LayerNorm over the last axis (dtype-preserving; computed in f32).
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + EPS)?.sqrt()?)?.to_dtype(dt)
}

/// Split a `[B, 3·inner]` modulation chunk into `(shift, scale, gate)`, each `[B, 1, inner]`.
fn chunk3(m: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let inner = m.dim(D::Minus1)? / 3;
    let shift = m.narrow(D::Minus1, 0, inner)?.unsqueeze(1)?;
    let scale = m.narrow(D::Minus1, inner, inner)?.unsqueeze(1)?;
    let gate = m.narrow(D::Minus1, 2 * inner, inner)?.unsqueeze(1)?;
    Ok((shift, scale, gate))
}

/// AdaLN-zero modulate: returns `(x·(1+scale) + shift, gate)`.
fn modulate(x: &Tensor, m: &Tensor) -> Result<(Tensor, Tensor)> {
    let (shift, scale, gate) = chunk3(m)?;
    let out = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    Ok((out, gate))
}

/// `modulate` with optional per-token timestep selection (`zero_cond_t`, Qwen-Image-Edit-2511). With
/// `index = None` this is exactly [`modulate`]. With `index = Some` the `m` chunk carries a doubled
/// batch `[real_t ; zero_t]` (`[2, 3·inner]`); each image token picks the real-`t` half where
/// `index == 0` (noise) and the `t 0` half where `index == 1` (conditioning) — the diffusers
/// `_modulate(index)`. Blended via `real + (zero − real)·index` (bit-equivalent for a 0/1 index).
fn modulate_sel(x: &Tensor, m: &Tensor, index: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
    let Some(index) = index else {
        return modulate(x, m);
    };
    let inner = m.dim(D::Minus1)? / 3;
    let blend = index.unsqueeze(2)?.to_dtype(m.dtype())?; // [1, seq, 1]
    let pick = |slot: usize| -> Result<Tensor> {
        let real = m
            .narrow(0, 0, 1)?
            .narrow(D::Minus1, slot * inner, inner)?
            .unsqueeze(1)?; // [1,1,inner]
        let zero = m
            .narrow(0, 1, 1)?
            .narrow(D::Minus1, slot * inner, inner)?
            .unsqueeze(1)?; // [1,1,inner]
        real.broadcast_add(&zero.broadcast_sub(&real)?.broadcast_mul(&blend)?) // [1, seq, inner]
    };
    let shift = pick(0)?;
    let scale = pick(1)?;
    let gate = pick(2)?;
    let out = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    Ok((out, gate))
}

/// `x + gate·y`.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x + y.broadcast_mul(gate)?
}

/// Sinusoidal timestep embedding `[1, dim]` from the raw sigma — the ×1000 scale is applied inside
/// the argument (diffusers `timestep · 1000`); `[cos | sin]`, base 10000.
fn timestep_embedding(sigma: f32, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let ln = 10000f32.ln();
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for k in 0..half {
        let freq = (-ln * k as f32 / half as f32).exp();
        let arg = sigma * 1000.0 * freq;
        cos[k] = arg.cos();
        sin[k] = arg.sin();
    }
    let cos = Tensor::from_vec(cos, (1, half), device)?;
    let sin = Tensor::from_vec(sin, (1, half), device)?;
    Tensor::cat(&[&cos, &sin], D::Minus1)
}

/// Max elements in a single attention scores tensor `[B,H,Sq,Sk]` before [`attention`] chunks over the
/// query rows. candle CUDA kernels index elements with **i32**, so a scores/probs tensor exceeding
/// `i32::MAX` (~2.147B) silently corrupts its tail. The Qwen MMDiT runs ONE joint attention over the
/// `[txt, noise(, ref)]` sequence (24 heads); the dual-latent edit path concatenates the reference
/// latents after the noise, so its joint sequence grows fastest and at >~1024² (≳1280²) `H·Sq·Sk`
/// exceeds the i32 limit → the trailing query rows get garbage attention → noise (sc-6217). 1.0B keeps
/// each chunk well under the limit while leaving the txt2img / control sizes (≤ ~0.5B at 1024²) a single
/// un-chunked pass, so those paths stay byte-identical.
const ATTN_SCORES_BUDGET: usize = 1_000_000_000;

/// SDPA over `[B,H,S,D]` q/k/v → `[B, S, H·D]`. scale = `head_dim^-0.5`. Chunks over the query rows when
/// the full `[B,H,Sq,Sk]` scores tensor would exceed [`ATTN_SCORES_BUDGET`] (the candle CUDA i32-index
/// limit). Each query row's softmax is over all keys and independent of the other rows, so the chunked
/// result is numerically identical to the single pass — only the long edit/joint sequences trip it.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> Result<Tensor> {
    attention_budgeted(q, k, v, head_dim, ATTN_SCORES_BUDGET)
}

/// [`attention`] with an explicit per-block scores-element budget (so the chunking is unit-testable with
/// a tiny budget that forces the chunked path on small tensors).
fn attention_budgeted(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    budget: usize,
) -> Result<Tensor> {
    let (b, h, s, d) = q.dims4()?;
    let scale = (head_dim as f64).powf(-0.5);
    let q = q.contiguous()?;
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;

    // The largest query block whose `[B,H,block,S]` scores tensor stays within budget (the whole `S` for
    // the txt2img sizes, so that path is the unchanged single matmul+softmax+matmul).
    let block = if b * h * s * s <= budget {
        s
    } else {
        (budget / (b * h * s)).max(1)
    };

    let o = if block >= s {
        let scores = (q.matmul(&k_t)? * scale)?;
        let probs = softmax_last_dim(&scores)?;
        probs.matmul(&v)? // [B,H,S,D]
    } else {
        let mut blocks = Vec::new();
        let mut start = 0;
        while start < s {
            let len = block.min(s - start);
            let scores = (q.narrow(2, start, len)?.matmul(&k_t)? * scale)?;
            let probs = softmax_last_dim(&scores)?;
            blocks.push(probs.matmul(&v)?); // [B,H,len,D]
            start += len;
        }
        Tensor::cat(&blocks, 2)? // [B,H,S,D]
    };
    o.transpose(1, 2)?.reshape((b, s, h * d))
}

/// Reshape `[B,S,inner]` → `[B,H,S,head_dim]`, applying per-head RMSNorm (over head_dim) for q/k.
fn to_heads(x: &Tensor, heads: usize, head_dim: usize, norm: Option<&RmsNorm>) -> Result<Tensor> {
    let (b, s, _) = x.dims3()?;
    let x = x.reshape((b, s, heads, head_dim))?;
    let x = match norm {
        Some(n) => n.forward(&x)?,
        None => x,
    };
    x.transpose(1, 2)?.contiguous()
}

struct TimeEmbed {
    linear_1: Linear,
    linear_2: Linear,
    channels: usize,
}

impl TimeEmbed {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let te = vb.pp("timestep_embedder");
        Ok(Self {
            linear_1: linear(cfg.timestep_channels, inner, te.pp("linear_1"))?,
            linear_2: linear(inner, inner, te.pp("linear_2"))?,
            channels: cfg.timestep_channels,
        })
    }

    fn forward(&self, sigma: f32, device: &Device, dtype: DType) -> Result<Tensor> {
        let emb = timestep_embedding(sigma, self.channels, device)?.to_dtype(dtype)?;
        let h = self.linear_1.forward(&emb)?.silu()?;
        self.linear_2.forward(&h)
    }
}

/// GELU-tanh feed-forward (`net.0.proj → gelu → net.2`).
struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    fn new(inner: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj_in: linear(inner, hidden, vb.pp("net").pp("0").pp("proj"))?,
            proj_out: linear(hidden, inner, vb.pp("net").pp("2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj_out.forward(&self.proj_in.forward(x)?.gelu()?)
    }
}

struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    to_add_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hd = cfg.head_dim;
        Ok(Self {
            to_q: linear(inner, inner, vb.pp("to_q"))?,
            to_k: linear(inner, inner, vb.pp("to_k"))?,
            to_v: linear(inner, inner, vb.pp("to_v"))?,
            to_out: linear(inner, inner, vb.pp("to_out").pp("0"))?,
            add_q: linear(inner, inner, vb.pp("add_q_proj"))?,
            add_k: linear(inner, inner, vb.pp("add_k_proj"))?,
            add_v: linear(inner, inner, vb.pp("add_v_proj"))?,
            to_add_out: linear(inner, inner, vb.pp("to_add_out"))?,
            norm_q: rms_norm(hd, EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, EPS, vb.pp("norm_k"))?,
            norm_added_q: rms_norm(hd, EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(hd, EPS, vb.pp("norm_added_k"))?,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (self.heads, self.head_dim);
        let txt_seq = txt.dim(1)?;

        let iq = apply_rope(
            &to_heads(&self.to_q.forward(img)?, h, hd, Some(&self.norm_q))?,
            img_cos,
            img_sin,
        )?;
        let ik = apply_rope(
            &to_heads(&self.to_k.forward(img)?, h, hd, Some(&self.norm_k))?,
            img_cos,
            img_sin,
        )?;
        let iv = to_heads(&self.to_v.forward(img)?, h, hd, None)?;
        let tq = apply_rope(
            &to_heads(&self.add_q.forward(txt)?, h, hd, Some(&self.norm_added_q))?,
            txt_cos,
            txt_sin,
        )?;
        let tk = apply_rope(
            &to_heads(&self.add_k.forward(txt)?, h, hd, Some(&self.norm_added_k))?,
            txt_cos,
            txt_sin,
        )?;
        let tv = to_heads(&self.add_v.forward(txt)?, h, hd, None)?;

        // Joint over the sequence, text first.
        let q = Tensor::cat(&[&tq, &iq], 2)?;
        let k = Tensor::cat(&[&tk, &ik], 2)?;
        let v = Tensor::cat(&[&tv, &iv], 2)?;
        // Chunk the joint attention over query rows when the [B,H,Sq,Sk] scores tensor would exceed the
        // candle CUDA i32-index limit (long edit/joint sequences >~1024²); numerically identical to a
        // single pass, and a no-op single pass for the txt2img / control sizes (sc-6217).
        let o = attention(&q, &k, &v, hd)?; // [B, seq, h·hd]
        let seq = o.dim(1)?;
        let txt_o = o.narrow(1, 0, txt_seq)?.contiguous()?;
        let img_o = o.narrow(1, txt_seq, seq - txt_seq)?.contiguous()?;
        Ok((
            self.to_out.forward(&img_o)?,
            self.to_add_out.forward(&txt_o)?,
        ))
    }
}

struct Block {
    img_mod: Linear,
    txt_mod: Linear,
    attn: JointAttention,
    img_ff: FeedForward,
    txt_ff: FeedForward,
}

impl Block {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let ff_hidden = inner * 4;
        Ok(Self {
            img_mod: linear(inner, 6 * inner, vb.pp("img_mod").pp("1"))?,
            txt_mod: linear(inner, 6 * inner, vb.pp("txt_mod").pp("1"))?,
            attn: JointAttention::new(cfg, vb.pp("attn"))?,
            img_ff: FeedForward::new(inner, ff_hidden, vb.pp("img_mlp"))?,
            txt_ff: FeedForward::new(inner, ff_hidden, vb.pp("txt_mlp"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        // `Some` only on the Qwen-Image-Edit-2511 `zero_cond_t` path: then `temb` is the doubled
        // `[real_t ; zero_t]` and the image stream selects modulation per token (0 = noise, 1 = cond).
        modulate_index: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let act = temb.silu()?; // [1, inner] (or [2, inner] under zero_cond_t)
        let img_mod = self.img_mod.forward(&act)?; // [1 or 2, 6·inner]
                                                   // The text stream always uses the real-timestep modulation (row 0 under zero_cond_t).
        let txt_act = match modulate_index {
            Some(_) => act.narrow(0, 0, 1)?,
            None => act.clone(),
        };
        let txt_mod = self.txt_mod.forward(&txt_act)?; // [1, 6·inner]
        let half = img_mod.dim(D::Minus1)? / 2;
        let (im0, im1) = (
            img_mod.narrow(D::Minus1, 0, half)?,
            img_mod.narrow(D::Minus1, half, half)?,
        );
        let (tm0, tm1) = (
            txt_mod.narrow(D::Minus1, 0, half)?,
            txt_mod.narrow(D::Minus1, half, half)?,
        );

        // attention path
        let (img_n, img_g1) = modulate_sel(&layer_norm(hidden)?, &im0, modulate_index)?;
        let (txt_n, txt_g1) = modulate(&layer_norm(encoder)?, &tm0)?;
        let (img_attn, txt_attn) = self
            .attn
            .forward(&img_n, &txt_n, img_cos, img_sin, txt_cos, txt_sin)?;
        let hidden = gated(hidden, &img_g1, &img_attn)?;
        let encoder = gated(encoder, &txt_g1, &txt_attn)?;

        // feed-forward path
        let (img_n2, img_g2) = modulate_sel(&layer_norm(&hidden)?, &im1, modulate_index)?;
        let hidden = gated(&hidden, &img_g2, &self.img_ff.forward(&img_n2)?)?;
        let (txt_n2, txt_g2) = modulate(&layer_norm(&encoder)?, &tm1)?;
        let encoder = gated(&encoder, &txt_g2, &self.txt_ff.forward(&txt_n2)?)?;

        Ok((encoder, hidden))
    }
}

/// AdaLayerNorm-Continuous output head: `silu(temb) → linear (bias-less) → (scale, shift)`, then
/// `(1+scale)·LN(x) + shift`.
struct NormOut {
    linear: Linear,
}

impl NormOut {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        // The checkpoint ships a bias, but the fork loads this bias-less.
        Ok(Self {
            linear: linear_no_bias(inner, 2 * inner, vb.pp("linear"))?,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let p = self.linear.forward(&temb.silu()?)?;
        let inner = p.dim(D::Minus1)? / 2;
        let scale = p.narrow(D::Minus1, 0, inner)?.unsqueeze(1)?;
        let shift = p.narrow(D::Minus1, inner, inner)?.unsqueeze(1)?;
        layer_norm(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)
    }
}

/// The Qwen-Image MMDiT.
pub struct QwenTransformer {
    img_in: Linear,
    txt_norm: RmsNorm,
    txt_in: Linear,
    time_embed: TimeEmbed,
    blocks: Vec<Block>,
    norm_out: NormOut,
    proj_out: Linear,
    rope: QwenRope,
    device: Device,
    dtype: DType,
}

impl QwenTransformer {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, vb.pp("transformer_blocks").pp(i))?);
        }
        Ok(Self {
            img_in: linear(cfg.in_channels, inner, vb.pp("img_in"))?,
            txt_norm: rms_norm(cfg.joint_attention_dim, cfg.eps, vb.pp("txt_norm"))?,
            txt_in: linear(cfg.joint_attention_dim, inner, vb.pp("txt_in"))?,
            time_embed: TimeEmbed::new(cfg, vb.pp("time_text_embed"))?,
            blocks,
            norm_out: NormOut::new(cfg, vb.pp("norm_out"))?,
            // proj_out maps to the packed velocity (patch²·out_channels = 64 = in_channels).
            proj_out: linear(inner, cfg.in_channels, vb.pp("proj_out"))?,
            rope: QwenRope::new(cfg),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Predict velocity. `hidden_states` `[1, img_seq, 64]`, `encoder_hidden_states`
    /// `[1, txt_seq, 3584]`, `timestep` = raw sigma, `(lat_h, lat_w)` = the packed token grid.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
    ) -> Result<Tensor> {
        // The plain path is `forward_control` with no residuals — byte-identical (the match below is
        // inert when `residuals = None`), so the txt2img parity path has a single source of truth.
        self.forward_control(
            hidden_states,
            encoder_hidden_states,
            timestep,
            lat_h,
            lat_w,
            None,
            0.0,
        )
    }

    /// [`forward`] with optional ControlNet residual injection (sc-5489): after base block `i` the
    /// residual `residuals[i / interval]` (pre-scaled by `control_scale`) is added to the image stream,
    /// where `interval = ceil(num_blocks / num_residuals)` (60 base blocks, 5 control residuals →
    /// interval 12) — the diffusers `QwenImageTransformer2DModel` `index_block // interval_control`
    /// pattern. `residuals = None` (or empty) is byte-identical to the plain forward.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_control(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
        residuals: Option<&[Tensor]>,
        control_scale: f32,
    ) -> Result<Tensor> {
        let temb = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        let mut hidden = self.img_in.forward(hidden_states)?;
        let encoder = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder = self.txt_in.forward(&encoder)?;

        let txt_seq = encoder.dim(1)?;
        let (img_cos, img_sin) = self.rope.img_cos_sin(lat_h, lat_w, &self.device)?;
        let (txt_cos, txt_sin) = self.rope.txt_cos_sin(txt_seq, lat_h, lat_w, &self.device)?;

        // Treat an empty slice as "no control" so the group index can't underflow. Pre-scale the (few)
        // control residuals once, before the 60-block loop.
        let residuals = residuals.filter(|r| !r.is_empty());
        let interval = residuals.map(|r| self.blocks.len().div_ceil(r.len().max(1)));
        let scaled: Option<Vec<Tensor>> = match residuals {
            Some(res) => Some(
                res.iter()
                    .map(|r| r * control_scale as f64)
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => None,
        };

        for (i, block) in self.blocks.iter().enumerate() {
            let (e, h) = block.forward(
                &hidden, &encoder, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin, None,
            )?;
            encoder = e;
            // After each base block, add the pre-scaled control residual for this block's group:
            // diffusers `hidden_states = hidden_states + controlnet_block_samples[i // interval]`.
            hidden = match (&scaled, interval) {
                (Some(res), Some(interval)) => {
                    let idx = (i / interval).min(res.len() - 1);
                    (h + &res[idx])?
                }
                _ => h,
            };
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }

    /// Qwen-Image-**Edit** dual-latent forward (sc-5487). `hidden_states` `[1, noise_seq + ref_seq, 64]`
    /// is the noise latents concatenated with the packed reference latents (the caller concatenates and
    /// slices back the noise prefix from the returned velocity); `cond_grids` lists each reference's
    /// `(latent_h, latent_w)` so the 3-axis RoPE spans `[noise] + references` (the grid index drives the
    /// frame axis). `zero_cond_t` (Edit-2511): double the timestep to `[t, 0]` and modulate the
    /// conditioning tokens as clean (t = 0) via the per-token `modulate_index`; `false` (the original
    /// Edit / 2509) runs a single timestep over the whole sequence. Returns the velocity over the
    /// **full** sequence `[1, noise_seq + ref_seq, 64]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_edit(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
        cond_grids: &[(usize, usize)],
        zero_cond_t: bool,
    ) -> Result<Tensor> {
        let img_seq = hidden_states.dim(1)?;
        let mut hidden = self.img_in.forward(hidden_states)?;
        let encoder = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder = self.txt_in.forward(&encoder)?;
        let txt_seq = encoder.dim(1)?;

        // 3-axis RoPE over the noise grid then each reference grid.
        let mut grids = Vec::with_capacity(1 + cond_grids.len());
        grids.push((lat_h, lat_w));
        grids.extend_from_slice(cond_grids);
        let (img_cos, img_sin) = self.rope.img_cos_sin_multi(&grids, &self.device)?;
        let (txt_cos, txt_sin) = self.rope.txt_cos_sin_multi(txt_seq, &grids, &self.device)?;

        // zero_cond_t: double the temb to [real_t ; zero_t] and build the per-token select index.
        let zc = zero_cond_t && !cond_grids.is_empty();
        let temb_real = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        let (temb, modulate_index) = if zc {
            let temb_zero = self.time_embed.forward(0.0, &self.device, self.dtype)?;
            let temb2 = Tensor::cat(&[&temb_real, &temb_zero], 0)?;
            let idx = build_modulate_index(lat_h * lat_w, cond_grids, img_seq, &self.device)?;
            (temb2, Some(idx))
        } else {
            (temb_real.clone(), None)
        };

        for block in &self.blocks {
            let (e, h) = block.forward(
                &hidden,
                &encoder,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                modulate_index.as_ref(),
            )?;
            encoder = e;
            hidden = h;
        }

        // norm_out uses only the real-timestep embedding (the fork's temb[:B]).
        let hidden = self.norm_out.forward(&hidden, &temb_real)?;
        self.proj_out.forward(&hidden)
    }
}

/// The per-token timestep selector for `zero_cond_t` (Qwen-Image-Edit-2511): `0` for the noise latent
/// tokens (`latent_h·latent_w`), `1` for every conditioning-image token (`Σ h·w` over the reference
/// grids). Shape `[1, img_seq]` f32 — diffusers `[[0]*prod(shapes[0]) + [1]*Σ prod(shapes[1:])]`.
fn build_modulate_index(
    noise_len: usize,
    cond_grids: &[(usize, usize)],
    img_seq: usize,
    device: &Device,
) -> Result<Tensor> {
    let cond_len: usize = cond_grids.iter().map(|(h, w)| h * w).sum();
    debug_assert_eq!(
        noise_len + cond_len,
        img_seq,
        "modulate index spans the full image sequence"
    );
    let mut row = vec![0f32; noise_len];
    row.extend(std::iter::repeat_n(1f32, cond_len));
    Tensor::from_vec(row, (1, img_seq), device)
}

/// The Qwen-Image **ControlNet-Union** control transformer (sc-5489) — the candle port of the InstantX
/// `Qwen-Image-ControlNet-Union` `QwenImageControlNetModel`. A small (5-block) partial copy of the base
/// MMDiT with its own input projections + a zero-init `controlnet_x_embedder` that adds the packed
/// VAE-encoded control image to `img_in(x)`; each block's output is projected by a zero-init
/// `controlnet_blocks[i]` into a residual. The residuals are injected into the frozen base transformer
/// at `interval = ceil(60/5) = 12` (see [`QwenTransformer::forward_control`]). The block math is the
/// **same** [`Block`] as the base (identical on-disk keys), so the loader reuses it.
pub struct QwenControlNet {
    img_in: Linear,
    txt_norm: RmsNorm,
    txt_in: Linear,
    time_embed: TimeEmbed,
    /// Zero-init projection of the packed control latent (`64 → inner`), added to `img_in(x)`.
    x_embedder: Linear,
    blocks: Vec<Block>,
    /// Zero-init per-block residual projections (`inner → inner`).
    controlnet_blocks: Vec<Linear>,
    rope: QwenRope,
    device: Device,
    dtype: DType,
}

impl QwenControlNet {
    /// Load the control branch (`num_layers` blocks — 5 for the InstantX Union) from its single-file
    /// checkpoint (the base block keys + `controlnet_x_embedder` + `controlnet_blocks.{i}`).
    pub fn new(cfg: &TransformerConfig, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mut blocks = Vec::with_capacity(num_layers);
        let mut controlnet_blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(Block::new(cfg, vb.pp("transformer_blocks").pp(i))?);
            controlnet_blocks.push(linear(inner, inner, vb.pp("controlnet_blocks").pp(i))?);
        }
        Ok(Self {
            img_in: linear(cfg.in_channels, inner, vb.pp("img_in"))?,
            txt_norm: rms_norm(cfg.joint_attention_dim, cfg.eps, vb.pp("txt_norm"))?,
            txt_in: linear(cfg.joint_attention_dim, inner, vb.pp("txt_in"))?,
            time_embed: TimeEmbed::new(cfg, vb.pp("time_text_embed"))?,
            x_embedder: linear(cfg.in_channels, inner, vb.pp("controlnet_x_embedder"))?,
            blocks,
            controlnet_blocks,
            rope: QwenRope::new(cfg),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// The number of control residuals (= control layers); drives the base injection interval.
    pub fn num_residuals(&self) -> usize {
        self.controlnet_blocks.len()
    }

    /// Run the control branch → the per-block residuals (pre-scale), one per control layer.
    /// `hidden_states`: the current packed noise latents `[1, seq, 64]`; `control_cond`: the packed
    /// VAE-encoded control image `[1, seq, 64]` (constant across steps); `encoder_hidden_states`: text
    /// `[1, txt_seq, 3584]`; `timestep`: the same raw sigma the base forward uses.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        control_cond: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
    ) -> Result<Vec<Tensor>> {
        let temb = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        // diffusers `hidden_states = self.img_in(x) + self.controlnet_x_embedder(controlnet_cond)`.
        let mut hidden =
            (self.img_in.forward(hidden_states)? + self.x_embedder.forward(control_cond)?)?;
        let encoder = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder = self.txt_in.forward(&encoder)?;

        let txt_seq = encoder.dim(1)?;
        let (img_cos, img_sin) = self.rope.img_cos_sin(lat_h, lat_w, &self.device)?;
        let (txt_cos, txt_sin) = self.rope.txt_cos_sin(txt_seq, lat_h, lat_w, &self.device)?;

        let mut residuals = Vec::with_capacity(self.blocks.len());
        for (block, cn) in self.blocks.iter().zip(&self.controlnet_blocks) {
            let (e, h) = block.forward(
                &hidden, &encoder, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin, None,
            )?;
            encoder = e;
            hidden = h;
            residuals.push(cn.forward(&hidden)?);
        }
        Ok(residuals)
    }
}

/// The Qwen-Image **2512-Fun-Controlnet-Union** VACE control branch (sc-8350) — the candle port of the
/// alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` `QwenImageControlTransformer2DModel` (the
/// `VideoX-Fun` VACE family, the same shape as the FLUX.2 / Z-Image Fun-Controlnet-Union branches),
/// which **replaces** the InstantX [`QwenControlNet`] shape on the Qwen control path.
///
/// Unlike the InstantX ControlNet (an independent mini-transformer with a zero-init
/// `controlnet_x_embedder` ADDed onto `img_in(x)`, emitting per-block residuals the base ADDs at a
/// fixed interval), the 2512-Fun branch is **VACE-style**: a `control_img_in` patch embedder
/// (`132 → inner`) feeds a control state `c` threaded through `N` control blocks that reuse the base
/// [`Block`] math (and the base modulation / RoPE / timestep), seeded at block 0 by
/// `c = before_proj(c) + img_embed`. Each control block emits a hint via a zero-init `after_proj`; the
/// base transformer adds `hints[n]·control_scale` into its image stream **after** the base block at
/// `control_layers[n]` (`[0, 12, 24, 36, 48]` — 5 hints across the 60-layer MMDiT). `control_scale = 0`
/// is byte-identical to the base forward (`+0`).
///
/// The control blocks are the *same* [`Block`] as the base (identical on-disk keys), so the loader
/// reuses [`Block::new`].
pub struct QwenFunControlBranch {
    /// `control_img_in`: 132 → inner. Biased patch embedder for the packed 132-ch control context.
    control_img_in: Linear,
    /// The `N` control blocks (same math as the base dual-stream block; reuse the base RoPE / temb).
    blocks: Vec<Block>,
    /// Zero-init per-block hint projection (`inner → inner`), one per control block (`after_proj`).
    after_proj: Vec<Linear>,
    /// Zero-init `before_proj` on control block 0 (`inner → inner`): `c = before_proj(c) + img_embed`.
    before_proj: Linear,
    /// Base block indices each control hint injects into (`control_layers`); `places[n]` is the base
    /// index for hint `n`.
    places: Vec<usize>,
}

impl QwenFunControlBranch {
    /// Load from the 2512-Fun checkpoint. `control_layers` must contain `0` (`before_proj` lives on
    /// control block 0). Keys: `control_img_in.{weight,bias}`, `control_blocks.{i}.*` (a base block +
    /// `after_proj` for every `i`, plus `before_proj` on `i == 0`).
    pub fn new(
        cfg: &TransformerConfig,
        control_layers: &[usize],
        control_in_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner = cfg.inner_dim();
        let n = control_layers.len();
        let mut blocks = Vec::with_capacity(n);
        let mut after_proj = Vec::with_capacity(n);
        for i in 0..n {
            let blk = vb.pp("control_blocks").pp(i);
            blocks.push(Block::new(cfg, blk.clone())?);
            after_proj.push(linear(inner, inner, blk.pp("after_proj"))?);
        }
        Ok(Self {
            control_img_in: linear(control_in_dim, inner, vb.pp("control_img_in"))?,
            before_proj: linear(
                inner,
                inner,
                vb.pp("control_blocks").pp(0).pp("before_proj"),
            )?,
            blocks,
            after_proj,
            places: control_layers.to_vec(),
        })
    }

    /// Number of control hints (= control layers); drives the base injection sites.
    pub fn num_hints(&self) -> usize {
        self.blocks.len()
    }

    /// The hint index injected at base block `idx`, or `None`. Mirrors the fork's
    /// `control_layers_mapping`.
    pub fn hint_index(&self, idx: usize) -> Option<usize> {
        self.places.iter().position(|&p| p == idx)
    }

    /// Run the VACE control stack → the per-block hints (pre-scale), one per control layer. The fork's
    /// `forward_control`: `c = control_img_in(control_context)`; block 0 seeds
    /// `c = before_proj(c) + img_embed`; each control block runs the *base* block math (reusing the base
    /// modulation / RoPE / timestep) and threads its own text stream to the next; `hint[i] =
    /// after_proj(c_after_block_i)`. The threaded text is local to the control stack — only the
    /// image-stream hints leave.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img_embed: &Tensor,
        encoder_embed: &Tensor,
        control_context: &Tensor,
        temb: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
    ) -> Result<Vec<Tensor>> {
        // c = control_img_in(control_context); seed block 0 with `before_proj(c) + img_embed`.
        let c0 = self.control_img_in.forward(control_context)?;
        let mut c = (self.before_proj.forward(&c0)? + img_embed)?;
        let mut encoder = encoder_embed.clone();
        let mut hints = Vec::with_capacity(self.blocks.len());
        for (block, ap) in self.blocks.iter().zip(&self.after_proj) {
            let (e, new_c) =
                block.forward(&c, &encoder, temb, img_cos, img_sin, txt_cos, txt_sin, None)?;
            encoder = e;
            // hint[i] = after_proj(c_after_block_i) (zero-init projection; the fork's `c_skip`).
            hints.push(ap.forward(&new_c)?);
            c = new_c;
        }
        Ok(hints)
    }
}

impl QwenTransformer {
    /// [`forward`](Self::forward) with the **2512-Fun-Controlnet-Union** VACE control branch (sc-8350).
    /// Identical to the T2I forward, plus: the control branch's per-block hints are computed once from
    /// the post-embedder image + text streams (reusing the base modulation / RoPE / timestep), and after
    /// base block `control_layers[n]` the hint `hints[n]·control_scale` is added to the image stream —
    /// the fork's `QwenImageControlTransformer2DModel.forward`. `control = None` is **byte-identical** to
    /// the plain forward, as is `control_scale = 0` (the zero-init `after_proj` + `+0` injection).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fun_control(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: f32,
        lat_h: usize,
        lat_w: usize,
        control: Option<(&QwenFunControlBranch, &Tensor)>,
        control_scale: f32,
    ) -> Result<Tensor> {
        let temb = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        let mut hidden = self.img_in.forward(hidden_states)?;
        let encoder = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder = self.txt_in.forward(&encoder)?;

        let txt_seq = encoder.dim(1)?;
        let (img_cos, img_sin) = self.rope.img_cos_sin(lat_h, lat_w, &self.device)?;
        let (txt_cos, txt_sin) = self.rope.txt_cos_sin(txt_seq, lat_h, lat_w, &self.device)?;

        // VACE control hints (sc-8350): computed once from the post-embedder image + text streams,
        // before the base block loop (the fork's `forward_control`), then injected per block. The hints
        // are pre-scaled by `control_scale` once (the scalar is the same across all hints and blocks);
        // `control = None` or `control_scale = 0` → no injection (byte-identical base).
        let scaled_hints: Option<Vec<Tensor>> = match control {
            Some((branch, cc)) => {
                let hints = branch.forward(
                    &hidden, &encoder, cc, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin,
                )?;
                Some(
                    hints
                        .iter()
                        .map(|h| h * control_scale as f64)
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            None => None,
        };

        for (i, block) in self.blocks.iter().enumerate() {
            let (e, h) = block.forward(
                &hidden, &encoder, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin, None,
            )?;
            encoder = e;
            // After base block `i`, add the pre-scaled hint for this block (if `i` is a control layer) —
            // the fork's `hidden_states = hidden_states + hints[block_id] * context_scale`.
            hidden = match (&scaled_hints, control) {
                (Some(hints), Some((branch, _))) => match branch.hint_index(i) {
                    Some(n) => (h + &hints[n])?,
                    None => h,
                },
                _ => h,
            };
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_attention_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single pass bit-for-bit — the guard for the i32-overflow fix (sc-6217).
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        // Huge budget → single pass; tiny budget (1) → chunked into single-row blocks.
        let single = attention_budgeted(&q, &k, &v, d, usize::MAX).unwrap();
        let chunked = attention_budgeted(&q, &k, &v, d, 1).unwrap();
        assert_eq!(single.dims(), chunked.dims());
        let a = single.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let c = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&c) {
            assert!(
                (x - y).abs() < 1e-6,
                "chunked attention diverged: {x} vs {y}"
            );
        }
    }

    use candle_gen::candle_nn::var_builder::SimpleBackend;
    use std::sync::Mutex;

    /// A deterministic random [`SimpleBackend`] for the no-weights control tests: returns a small
    /// reproducible normal tensor of the requested shape for any key (a fresh seeded RNG per
    /// `VarBuilder`, advanced per `get` so distinct keys get distinct — but reproducible — tensors).
    /// `Mutex` (not `RefCell`) because `SimpleBackend: Send + Sync`.
    struct RandomBackend {
        rng: Mutex<rand::rngs::StdRng>,
    }

    impl RandomBackend {
        fn new(seed: u64) -> Self {
            use rand::SeedableRng;
            Self {
                rng: Mutex::new(rand::rngs::StdRng::seed_from_u64(seed)),
            }
        }
    }

    impl SimpleBackend for RandomBackend {
        fn get(
            &self,
            s: candle_gen::candle_core::Shape,
            _name: &str,
            _h: candle_gen::candle_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> candle_gen::candle_core::Result<Tensor> {
            use rand_distr::{Distribution, StandardNormal};
            let n: usize = s.elem_count();
            let mut rng = self.rng.lock().unwrap();
            // Small magnitude keeps the tiny DiT numerically sane (and norm-out + RMSNorm stable).
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    let v: f32 = StandardNormal.sample(&mut *rng);
                    0.05f32 * v
                })
                .collect();
            Tensor::from_vec(data, s, dev)?.to_dtype(dtype)
        }

        fn get_unchecked(
            &self,
            _name: &str,
            _dtype: DType,
            _dev: &Device,
        ) -> candle_gen::candle_core::Result<Tensor> {
            candle_gen::candle_core::bail!("RandomBackend requires a shape; use get")
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    /// A tiny Qwen config (4 base blocks, 2 heads × 8) for the no-weights control wiring tests.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 8,
            out_channels: 2,
            num_layers: 4,
            num_heads: 2,
            head_dim: 8,
            joint_attention_dim: 12,
            timestep_channels: 16,
            axes_dim: [2, 3, 3],
            rope_theta: 10_000.0,
            eps: 1e-6,
        }
    }

    fn random_vb(seed: u64) -> VarBuilder<'static> {
        VarBuilder::from_backend(Box::new(RandomBackend::new(seed)), DType::F32, Device::Cpu)
    }

    /// scale=0 (and `control = None`) reproduce the plain base forward **byte-exact** — the zero-init
    /// `after_proj` plus the `+0` injection means the control branch contributes nothing. This is the
    /// load-bearing parity guarantee (the base T2I/Edit path is untouched by the new lane).
    #[test]
    fn fun_control_scale_zero_is_byte_exact_base() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        // Base transformer + a 2512-Fun control branch (2 control layers for the tiny model) from
        // independent random backends.
        let transformer = QwenTransformer::new(&cfg, random_vb(1)).unwrap();
        let control_layers = [0usize, 2];
        let control_in_dim = 132; // the real packed control-context width (independent of the tiny DiT)
        let branch =
            QwenFunControlBranch::new(&cfg, &control_layers, control_in_dim, random_vb(2)).unwrap();

        let (lat_h, lat_w) = (2usize, 3usize);
        let seq = lat_h * lat_w;
        let txt_seq = 5usize;
        let hidden = Tensor::randn(0f32, 1f32, (1, seq, cfg.in_channels), &dev).unwrap();
        let encoder =
            Tensor::randn(0f32, 1f32, (1, txt_seq, cfg.joint_attention_dim), &dev).unwrap();
        let control_cond = Tensor::randn(0f32, 1f32, (1, seq, control_in_dim), &dev).unwrap();
        let sigma = 0.7f32;

        let base = transformer
            .forward(&hidden, &encoder, sigma, lat_h, lat_w)
            .unwrap();

        // control = None ≡ base.
        let none = transformer
            .forward_fun_control(&hidden, &encoder, sigma, lat_h, lat_w, None, 0.0)
            .unwrap();
        // control = Some(branch) with scale 0 ≡ base (the injection is `+ hint·0`).
        let scaled0 = transformer
            .forward_fun_control(
                &hidden,
                &encoder,
                sigma,
                lat_h,
                lat_w,
                Some((&branch, &control_cond)),
                0.0,
            )
            .unwrap();

        let b = base.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let n = none.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let z = scaled0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(b, n, "control=None must be byte-exact base");
        assert_eq!(b, z, "control_scale=0 must be byte-exact base");

        // And a non-zero scale actually changes the output (the lane is wired, not inert).
        let scaled1 = transformer
            .forward_fun_control(
                &hidden,
                &encoder,
                sigma,
                lat_h,
                lat_w,
                Some((&branch, &control_cond)),
                1.0,
            )
            .unwrap();
        let s1 = scaled1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            b.iter().zip(&s1).any(|(x, y)| (x - y).abs() > 1e-6),
            "control_scale=1 must change the output (branch must be wired)"
        );
    }

    /// The branch wiring: 2 control layers → 2 hints, injected at base blocks `[0, 2]` (and only there).
    #[test]
    fn fun_control_branch_hint_wiring() {
        let cfg = tiny_cfg();
        let control_layers = [0usize, 2];
        let branch = QwenFunControlBranch::new(&cfg, &control_layers, 132, random_vb(3)).unwrap();
        assert_eq!(branch.num_hints(), 2);
        assert_eq!(branch.hint_index(0), Some(0));
        assert_eq!(branch.hint_index(2), Some(1));
        assert_eq!(branch.hint_index(1), None);
        assert_eq!(branch.hint_index(3), None);
    }
}
