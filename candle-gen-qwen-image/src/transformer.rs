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
        let scale = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v.contiguous()?)?; // [B,H,seq,hd]
        let (b, _, seq, _) = o.dims4()?;
        let o = o.transpose(1, 2)?.reshape((b, seq, h * hd))?;
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
    ) -> Result<(Tensor, Tensor)> {
        let act = temb.silu()?;
        let img_mod = self.img_mod.forward(&act)?;
        let txt_mod = self.txt_mod.forward(&act)?;
        let inner = img_mod.dim(D::Minus1)? / 2;
        let (im0, im1) = (
            img_mod.narrow(D::Minus1, 0, inner)?,
            img_mod.narrow(D::Minus1, inner, inner)?,
        );
        let (tm0, tm1) = (
            txt_mod.narrow(D::Minus1, 0, inner)?,
            txt_mod.narrow(D::Minus1, inner, inner)?,
        );

        // attention path
        let (img_n, img_g1) = modulate(&layer_norm(hidden)?, &im0)?;
        let (txt_n, txt_g1) = modulate(&layer_norm(encoder)?, &tm0)?;
        let (img_attn, txt_attn) = self
            .attn
            .forward(&img_n, &txt_n, img_cos, img_sin, txt_cos, txt_sin)?;
        let hidden = gated(hidden, &img_g1, &img_attn)?;
        let encoder = gated(encoder, &txt_g1, &txt_attn)?;

        // feed-forward path
        let (img_n2, img_g2) = modulate(&layer_norm(&hidden)?, &im1)?;
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
                &hidden, &encoder, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin,
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
                &hidden, &encoder, &temb, &img_cos, &img_sin, &txt_cos, &txt_sin,
            )?;
            encoder = e;
            hidden = h;
            residuals.push(cn.forward(&hidden)?);
        }
        Ok(residuals)
    }
}
