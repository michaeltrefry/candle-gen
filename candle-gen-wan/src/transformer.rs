//! The **`WanTransformer3DModel`** DiT (TI2V-5B, dense) — a port of diffusers `transformer_wan.py`.
//! 30 blocks, each: AdaLN-modulated self-attention (3-axis interleaved RoPE, full-dim qk-RMSNorm) →
//! ungated cross-attention to the UMT5 context → AdaLN-modulated gated GELU FFN. The per-block
//! 6-vector modulation is `scale_shift_table + time_proj`; the head uses a separate 2-vector.
//!
//! Runs in **bf16** (the 5B checkpoint's native dtype) with norms / modulation / RoPE upcast to f32,
//! mirroring diffusers' `FP32LayerNorm` + `.float()` modulation.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, Module, VarBuilder};

use crate::config::TransformerConfig;
use crate::rope::apply_rope;

pub(crate) fn linear(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((out_c, in_c), "weight")?,
        Some(vb.get(out_c, "bias")?),
    ))
}

/// LayerNorm over the last dim with no learnable affine, in f32.
pub(crate) fn ln_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

/// RMSNorm over the last dim (qk-norm "across heads") with affine weight, in f32.
pub(crate) fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(dt)
}

/// Scaled-dot-product attention. `q,k,v`: `[B, H, S*, d]`; softmax upcast to f32.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;
    let attn = softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
    attn.matmul(&v.contiguous()?)
}

struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
    is_cross: bool,
}

impl Attention {
    fn new(cfg: &TransformerConfig, vb: VarBuilder, is_cross: bool) -> Result<Self> {
        let inner = cfg.dim;
        Ok(Self {
            to_q: linear(cfg.dim, inner, vb.pp("to_q"))?,
            to_k: linear(cfg.dim, inner, vb.pp("to_k"))?,
            to_v: linear(cfg.dim, inner, vb.pp("to_v"))?,
            to_out: linear(inner, cfg.dim, vb.pp("to_out").pp("0"))?,
            norm_q: vb.pp("norm_q").get(inner, "weight")?,
            norm_k: vb.pp("norm_k").get(inner, "weight")?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            eps: cfg.eps,
            is_cross,
        })
    }

    /// `hidden`: `[B, S, dim]`; `context`: cross-attn K/V source (= hidden for self-attn). RoPE is
    /// applied only when `cos`/`sin` are given (self-attn).
    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, _) = hidden.dims3()?;
        let s_kv = context.dim(1)?;
        let q = rms(&self.to_q.forward(hidden)?, &self.norm_q, self.eps)?;
        let k = rms(&self.to_k.forward(context)?, &self.norm_k, self.eps)?;
        let v = self.to_v.forward(context)?;
        let to_heads = |t: &Tensor, len: usize| -> Result<Tensor> {
            t.reshape((b, len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let mut q = to_heads(&q, s)?; // [B,H,S,d]
        let mut k = to_heads(&k, s_kv)?;
        let v = to_heads(&v, s_kv)?;
        if let Some((cos, sin)) = rope {
            q = apply_rope(&q, cos, sin)?;
            k = apply_rope(&k, cos, sin)?;
        }
        let scale = (self.head_dim as f64).powf(-0.5);
        let out = sdpa(&q, &k, &v, scale)?; // [B,H,S,d]
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        let _ = self.is_cross;
        self.to_out.forward(&out)
    }
}

struct Ffn {
    proj: Linear, // net.0.proj
    out: Linear,  // net.2
}

impl Ffn {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: linear(cfg.dim, cfg.ffn_dim, vb.pp("net").pp("0").pp("proj"))?,
            out: linear(cfg.ffn_dim, cfg.dim, vb.pp("net").pp("2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.proj.forward(x)?.gelu()?)
    }
}

pub(crate) struct Block {
    scale_shift_table: Tensor, // [1,6,dim] f32
    attn1: Attention,
    norm2_w: Tensor, // affine cross-attn norm
    norm2_b: Tensor,
    attn2: Attention,
    ffn: Ffn,
    eps: f64,
}

impl Block {
    pub(crate) fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale_shift_table: vb
                .get((1, 6, cfg.dim), "scale_shift_table")?
                .to_dtype(DType::F32)?,
            attn1: Attention::new(cfg, vb.pp("attn1"), false)?,
            norm2_w: vb
                .pp("norm2")
                .get(cfg.dim, "weight")?
                .to_dtype(DType::F32)?,
            norm2_b: vb.pp("norm2").get(cfg.dim, "bias")?.to_dtype(DType::F32)?,
            attn2: Attention::new(cfg, vb.pp("attn2"), true)?,
            ffn: Ffn::new(cfg, vb.pp("ffn"))?,
            eps: cfg.eps,
        })
    }

    /// `hidden`: `[B,S,dim]` (bf16); `temb6`: `[B,6,dim]` (f32); `context`: `[B,S_ctx,dim]` (bf16).
    pub(crate) fn forward(
        &self,
        hidden: &Tensor,
        temb6: &Tensor,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let dt = hidden.dtype();
        // mods: scale_shift_table[1,6,dim] + temb6[B,6,dim] → 6 × [B,1,dim] (f32).
        let mods = self.scale_shift_table.broadcast_add(temb6)?;
        let m = |i: usize| -> Result<Tensor> { mods.narrow(1, i, 1) };
        let (shift_msa, scale_msa, gate_msa) = (m(0)?, m(1)?, m(2)?);
        let (c_shift, c_scale, c_gate) = (m(3)?, m(4)?, m(5)?);

        let hf = hidden.to_dtype(DType::F32)?;
        // 1. self-attention
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?
            .to_dtype(dt)?;
        let a = self.attn1.forward(&n, &n, Some((cos, sin)))?;
        let hf = (hf + a.to_dtype(DType::F32)?.broadcast_mul(&gate_msa)?)?;

        // 2. cross-attention (affine norm2, ungated)
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&self.norm2_w)?
            .broadcast_add(&self.norm2_b)?
            .to_dtype(dt)?;
        let a = self.attn2.forward(&n, context, None)?;
        let hf = (hf + a.to_dtype(DType::F32)?)?;

        // 3. feed-forward
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&(c_scale + 1.0)?)?
            .broadcast_add(&c_shift)?
            .to_dtype(dt)?;
        let f = self.ffn.forward(&n)?;
        let hf = (hf + f.to_dtype(DType::F32)?.broadcast_mul(&c_gate)?)?;
        hf.to_dtype(dt)
    }
}

/// Build the `[B, freq_dim]` sinusoidal timestep embedding (diffusers `Timesteps`,
/// `flip_sin_to_cos=True`, `downscale_freq_shift=0`): `[cos(t·ω) | sin(t·ω)]`.
pub(crate) fn timestep_sinusoid(t: f64, freq_dim: usize, b: usize, dev: &Device) -> Result<Tensor> {
    let half = freq_dim / 2;
    let mut row = vec![0f32; freq_dim];
    for i in 0..half {
        let freq = (-(10000f64.ln()) * i as f64 / half as f64).exp();
        let ang = t * freq;
        row[i] = ang.cos() as f32;
        row[half + i] = ang.sin() as f32;
    }
    let one = Tensor::from_vec(row, (1, freq_dim), dev)?;
    if b == 1 {
        Ok(one)
    } else {
        Ok(one.broadcast_as((b, freq_dim))?.contiguous()?)
    }
}

pub struct WanTransformer {
    patch_w: Tensor, // [dim,48,p_h,p_w]
    patch_b: Tensor, // [1,dim,1,1]
    text_l1: Linear,
    text_l2: Linear,
    time_l1: Linear,
    time_l2: Linear,
    time_proj: Linear,
    blocks: Vec<Block>,
    norm_out_eps: f64,
    proj_out: Linear,
    scale_shift_table: Tensor, // [1,2,dim] f32
    cfg: TransformerConfig,
    device: Device,
    dtype: DType,
}

impl WanTransformer {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let (pt, ph, pw) = cfg.patch;
        // patch_embedding is a Conv3d (1,2,2); temporal kernel 1 → squeeze to a per-frame conv2d.
        let pw_full = vb.get(
            (cfg.dim, cfg.in_channels, pt, ph, pw),
            "patch_embedding.weight",
        )?;
        let patch_w = pw_full.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [dim,48,ph,pw]
        let patch_b = vb
            .get(cfg.dim, "patch_embedding.bias")?
            .reshape((1, cfg.dim, 1, 1))?;

        let ce = vb.pp("condition_embedder");
        let text_l1 = linear(cfg.text_dim, cfg.dim, ce.pp("text_embedder").pp("linear_1"))?;
        let text_l2 = linear(cfg.dim, cfg.dim, ce.pp("text_embedder").pp("linear_2"))?;
        let time_l1 = linear(cfg.freq_dim, cfg.dim, ce.pp("time_embedder").pp("linear_1"))?;
        let time_l2 = linear(cfg.dim, cfg.dim, ce.pp("time_embedder").pp("linear_2"))?;
        let time_proj = linear(cfg.dim, 6 * cfg.dim, ce.pp("time_proj"))?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let proj_out = linear(cfg.dim, cfg.out_channels * pt * ph * pw, vb.pp("proj_out"))?;
        let scale_shift_table = vb
            .get((1, 2, cfg.dim), "scale_shift_table")?
            .to_dtype(DType::F32)?;

        Ok(Self {
            patch_w,
            patch_b,
            text_l1,
            text_l2,
            time_l1,
            time_l2,
            time_proj,
            blocks,
            norm_out_eps: cfg.eps,
            proj_out,
            scale_shift_table,
            cfg: *cfg,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Project UMT5 prompt embeds `[B,S,4096]` → cross-attn context `[B,S,dim]` (constant across the
    /// denoise loop). `gelu_tanh` between the two linears (PixArtAlphaTextProjection).
    pub fn embed_text(&self, prompt_embeds: &Tensor) -> Result<Tensor> {
        let x = prompt_embeds.to_dtype(self.dtype)?;
        self.text_l2.forward(&self.text_l1.forward(&x)?.gelu()?)
    }

    /// One DiT forward: `latents [B,48,F,Hl,Wl]`, projected `context [B,S,dim]`, scalar `t`,
    /// RoPE `cos`/`sin [L,64]` → predicted velocity `[B,48,F,Hl,Wl]`.
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        t: f64,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, _c, f, hl, wl) = latents.dims5()?;
        let (pt, ph, pw) = self.cfg.patch;
        let (ppf, pph, ppw) = (f / pt, hl / ph, wl / pw);

        // Patch embed: per-frame strided conv2d, then flatten to tokens (f outer, then h, w).
        let merged = latents
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * f, self.cfg.in_channels, hl, wl))?
            .contiguous()?
            .to_dtype(self.dtype)?;
        let y = merged.conv2d(&self.patch_w, 0, ph, 1, 1)?; // [B*F,dim,pph,ppw]
        let y = y.broadcast_add(&self.patch_b)?;
        let mut hidden = y
            .reshape((b, f, self.cfg.dim, pph, ppw))?
            .permute((0, 1, 3, 4, 2))? // [B,F,pph,ppw,dim]
            .reshape((b, ppf * pph * ppw, self.cfg.dim))?
            .contiguous()?;

        // Time embedding → temb [B,dim], and the per-block 6-vector temb6 [B,6,dim] (f32).
        let sinus =
            timestep_sinusoid(t, self.cfg.freq_dim, b, &self.device)?.to_dtype(self.dtype)?;
        let temb = self
            .time_l2
            .forward(&self.time_l1.forward(&sinus)?.silu()?)?; // [B,dim]
        let temb6 = self
            .time_proj
            .forward(&temb.silu()?)?
            .reshape((b, 6, self.cfg.dim))?
            .to_dtype(DType::F32)?;

        for blk in &self.blocks {
            hidden = blk.forward(&hidden, &temb6, context, cos, sin)?;
        }

        // Head: norm_out (non-affine) modulated by scale_shift_table + temb.
        let head_mod = self
            .scale_shift_table
            .broadcast_add(&temb.unsqueeze(1)?.to_dtype(DType::F32)?)?;
        let shift = head_mod.narrow(1, 0, 1)?;
        let scale = head_mod.narrow(1, 1, 1)?;
        let hf = hidden.to_dtype(DType::F32)?;
        let normed = ln_no_affine(&hf, self.norm_out_eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?
            .to_dtype(self.dtype)?;
        let out = self.proj_out.forward(&normed)?; // [B,L,out_c*patch]

        // Unpatchify → [B,48,F,Hl,Wl].
        let oc = self.cfg.out_channels;
        out.reshape(&[b, ppf, pph, ppw, pt, ph, pw, oc][..])?
            .permute(&[0usize, 7, 1, 4, 2, 5, 3, 6][..])?
            .reshape((b, oc, ppf * pt, pph * ph, ppw * pw))?
            .to_dtype(DType::F32)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
