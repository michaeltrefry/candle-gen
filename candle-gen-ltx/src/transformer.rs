//! LTX-2.3 **video DiT** (`AVTransformer3DModel`, video-only / gated path) — port of mlx-gen-ltx
//! `transformer.rs` (`LtxDiT`). patchify_proj (128→4096) → adaLN-single (timestep→9·dim) +
//! prompt-adaLN (→2·dim) → 48 gated blocks → affine-false LayerNorm output head + 2-row scale-shift
//! → proj_out (→128) velocity.
//!
//! Per-block (gated 9-row `scale_shift_table` + adaLN-single timestep; rows [shift,scale,gate] ×
//! {MSA 0:3, FF 3:6, text-cross-attn 6:9}): MSA self-attn (q/k RMSNorm over full inner, split 3-D
//! RoPE, **2·sigmoid** per-head gate) → prompt-modulated text cross-attn (no RoPE) → tanh-gelu FFN,
//! each adaLN-modulated (`x·(1+scale)+shift`) and gated (`x + out·gate`). Our checkpoint is dense
//! bf16; the whole forward runs bf16, with attention/norms/layernorm computed in f32 for fidelity.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    ops::rms_norm, ops::sigmoid, ops::softmax_last_dim, Linear, Module, VarBuilder,
};

use crate::config::TransformerConfig;
use crate::rope::apply_split_rope;

fn linear(vb: &VarBuilder, key: &str) -> Result<Linear> {
    let w = vb
        .get_unchecked(&format!("{key}.weight"))?
        .to_dtype(DType::BF16)?;
    let b = vb
        .get_unchecked(&format!("{key}.bias"))?
        .to_dtype(DType::BF16)?;
    Ok(Linear::new(w, Some(b)))
}

/// `x·(1+scale)+shift`; scale/shift `[B,1,inner]` broadcast over the token axis.
fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// `x + out·gate`; gate `[B,1,inner]` broadcasts over `out [B,S,inner]`.
fn gated(x: &Tensor, out: &Tensor, gate: &Tensor) -> Result<Tensor> {
    x + out.broadcast_mul(gate)?
}

/// Weightless RMSNorm (unit weight) over the last axis, in f32.
fn rms_noweight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let inv = (xf.sqr()?.mean_keepdim(D::Minus1)? + eps)?
        .sqrt()?
        .recip()?;
    let _ = dim;
    xf.broadcast_mul(&inv)?.to_dtype(x.dtype())
}

/// PixArt sinusoidal timestep embedding (flip_sin_to_cos, cos first), `[N,256]` f32. `ts` is `[N]`
/// f32 (already × timestep_scale_multiplier).
fn timestep_embedding(ts: &Tensor, device: &Device) -> Result<Tensor> {
    const TIME_PROJ_DIM: usize = 256;
    let half = TIME_PROJ_DIM / 2;
    let neg_ln = -(10000f64).ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (neg_ln * i as f64 / half as f64).exp() as f32)
        .collect();
    let n = ts.dim(0)?;
    let freq = Tensor::from_vec(freqs, (1, half), device)?;
    let emb = ts.reshape((n, 1))?.broadcast_mul(&freq)?; // (N, half)
    Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1) // (N, 256)
}

/// `table[row] + ts4[:,:,row,:]` for `row in [lo,hi)`; each result `[B,1,inner]`.
fn ada_values(table: &Tensor, ts_emb: &Tensor, lo: usize, hi: usize) -> Result<Vec<Tensor>> {
    let (num, inner) = table.dims2()?;
    let (b, s, _) = ts_emb.dims3()?;
    let ts4 = ts_emb.reshape((b, s, num, inner))?;
    let mut out = Vec::with_capacity(hi - lo);
    for row in lo..hi {
        let trow = table.narrow(0, row, 1)?.reshape((1, 1, inner))?;
        let tsrow = ts4.narrow(2, row, 1)?.squeeze(2)?; // (b,s,inner)
        out.push(trow.broadcast_add(&tsrow)?);
    }
    Ok(out)
}

struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Linear,
    heads: usize,
    dim_head: usize,
    eps: f64,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &TransformerConfig) -> Result<Self> {
        Ok(Self {
            to_q: linear(&vb, "to_q")?,
            to_k: linear(&vb, "to_k")?,
            to_v: linear(&vb, "to_v")?,
            to_out: linear(&vb, "to_out.0")?,
            q_norm: vb.get_unchecked("q_norm.weight")?.to_dtype(DType::BF16)?,
            k_norm: vb.get_unchecked("k_norm.weight")?.to_dtype(DType::BF16)?,
            gate: linear(&vb, "to_gate_logits")?,
            heads: cfg.num_heads,
            dim_head: cfg.head_dim,
            eps: cfg.norm_eps,
        })
    }

    fn to_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        x.reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)
    }

    /// Self-attn when `context` is `None` (RoPE applied); cross-attn otherwise (no RoPE).
    fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let ctx = context.unwrap_or(x);
        // q/k RMSNorm over the full inner dim (pre-head), then head reshape.
        let q = rms_norm(
            &self.to_q.forward(x)?.contiguous()?,
            &self.q_norm,
            self.eps as f32,
        )?;
        let k = rms_norm(
            &self.to_k.forward(ctx)?.contiguous()?,
            &self.k_norm,
            self.eps as f32,
        )?;
        let v = self.to_v.forward(ctx)?;
        let mut qh = self.to_heads(&q)?;
        let mut kh = self.to_heads(&k)?;
        let vh = self.to_heads(&v)?;
        if let Some((cos, sin)) = rope {
            qh = apply_split_rope(&qh, cos, sin)?;
            kh = apply_split_rope(&kh, cos, sin)?;
        }
        // Attention in f32.
        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let qf = qh.to_dtype(DType::F32)?.contiguous()?;
        let kf = kh.to_dtype(DType::F32)?.contiguous()?;
        let vf = vh.to_dtype(DType::F32)?.contiguous()?;
        let scores = (qf.matmul(&kf.transpose(2, 3)?)? * scale)?;
        let out = softmax_last_dim(&scores)?.matmul(&vf)?; // (b,h,s,d)
        let (b, s, _) = x.dims3()?;
        let inner = self.heads * self.dim_head;
        let mut out = out
            .transpose(1, 2)?
            .reshape((b, s, inner))?
            .to_dtype(DType::BF16)?;
        // Per-head gate: 2·sigmoid(logits) (zero-init → identity).
        let logits = self.gate.forward(x)?;
        let gates = (sigmoid(&logits)? * 2.0)?.reshape((b, s, self.heads, 1))?;
        out = out
            .reshape((b, s, self.heads, self.dim_head))?
            .broadcast_mul(&gates)?
            .reshape((b, s, inner))?;
        self.to_out.forward(&out)
    }
}

struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj_in: linear(&vb.pp("net.0"), "proj")?,
            proj_out: linear(&vb.pp("net"), "2")?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // tanh-approx gelu.
        self.proj_out.forward(&self.proj_in.forward(x)?.gelu()?)
    }
}

struct AdaLayerNormSingle {
    ts_lin1: Linear,
    ts_lin2: Linear,
    linear: Linear,
}

impl AdaLayerNormSingle {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ts_lin1: linear(&vb.pp("emb.timestep_embedder"), "linear_1")?,
            ts_lin2: linear(&vb.pp("emb.timestep_embedder"), "linear_2")?,
            linear: linear(&vb, "linear")?,
        })
    }

    /// `ts_flat` is `[N]` f32 (already scaled). Returns `(scale_shift [N, coeff·inner], embedded
    /// [N, inner])`, bf16.
    fn forward(&self, ts_flat: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
        let proj = timestep_embedding(ts_flat, device)?.to_dtype(DType::BF16)?;
        let h = self.ts_lin1.forward(&proj)?.silu()?;
        let embedded = self.ts_lin2.forward(&h)?;
        let scale_shift = self.linear.forward(&embedded.silu()?)?;
        Ok((scale_shift, embedded))
    }
}

struct VideoBlock {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    scale_shift_table: Tensor,        // (9, inner) bf16
    prompt_scale_shift_table: Tensor, // (2, inner) bf16
    eps: f64,
}

impl VideoBlock {
    fn load(vb: VarBuilder, cfg: &TransformerConfig) -> Result<Self> {
        Ok(Self {
            attn1: Attention::load(vb.pp("attn1"), cfg)?,
            attn2: Attention::load(vb.pp("attn2"), cfg)?,
            ff: FeedForward::load(vb.pp("ff"))?,
            scale_shift_table: vb
                .get_unchecked("scale_shift_table")?
                .to_dtype(DType::BF16)?,
            prompt_scale_shift_table: vb
                .get_unchecked("prompt_scale_shift_table")?
                .to_dtype(DType::BF16)?,
            eps: cfg.norm_eps,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        ts_emb: &Tensor,
        prompt_ts: &Tensor,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        // MSA (rows 0:3 = shift, scale, gate).
        let msa = ada_values(&self.scale_shift_table, ts_emb, 0, 3)?;
        let norm = modulate(&rms_noweight(x, self.eps)?, &msa[1], &msa[0])?;
        let attn = self.attn1.forward(&norm, None, Some((cos, sin)))?;
        let mut x = gated(x, &attn, &msa[2])?;

        // prompt-adaLN on the text context (rows 0,1 = shift, scale).
        let p = ada_values(&self.prompt_scale_shift_table, prompt_ts, 0, 2)?;
        let v_context = modulate(context, &p[1], &p[0])?;

        // Text cross-attention (rows 6:9).
        let ca = ada_values(&self.scale_shift_table, ts_emb, 6, 9)?;
        let norm_ca = modulate(&rms_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = self.attn2.forward(&norm_ca, Some(&v_context), None)?;
        x = gated(&x, &cross, &ca[2])?;

        // FeedForward (rows 3:6).
        let mlp = ada_values(&self.scale_shift_table, ts_emb, 3, 6)?;
        let norm_mlp = modulate(&rms_noweight(&x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff = self.ff.forward(&norm_mlp)?;
        gated(&x, &ff, &mlp[2])
    }
}

/// Affine-false LayerNorm over the last axis (computed in f32, cast back).
fn layer_norm_noaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(x.dtype())
}

pub struct LtxDiT {
    patchify_proj: Linear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: AdaLayerNormSingle,
    blocks: Vec<VideoBlock>,
    scale_shift_table: Tensor, // (2, inner) bf16
    proj_out: Linear,
    cfg: TransformerConfig,
    device: Device,
}

impl LtxDiT {
    /// Build from a VarBuilder rooted at `model.diffusion_model.`.
    pub fn new(vb: VarBuilder, cfg: &TransformerConfig) -> Result<Self> {
        let device = vb.device().clone();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(VideoBlock::load(
                vb.pp(format!("transformer_blocks.{i}")),
                cfg,
            )?);
        }
        Ok(Self {
            patchify_proj: linear(&vb, "patchify_proj")?,
            adaln: AdaLayerNormSingle::load(vb.pp("adaln_single"))?,
            prompt_adaln: AdaLayerNormSingle::load(vb.pp("prompt_adaln_single"))?,
            blocks,
            scale_shift_table: vb
                .get_unchecked("scale_shift_table")?
                .to_dtype(DType::BF16)?,
            proj_out: linear(&vb, "proj_out")?,
            cfg: cfg.clone(),
            device,
        })
    }

    /// Velocity forward.
    ///
    /// * `latent_tokens` — `[B, S, 128]` patchified latent tokens.
    /// * `sigma` — scalar σ (uniform per-token T2V timestep).
    /// * `context` — `[B, ctx, 4096]` connector embeddings.
    /// * `cos`/`sin` — split-RoPE tables `[1, heads, S, 64]` (f32).
    pub fn forward(
        &self,
        latent_tokens: &Tensor,
        sigma: f64,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let b = latent_tokens.dim(0)?;
        let inner = self.cfg.inner_dim();
        let coeff = self.cfg.adaln_coeff;

        let x = self
            .patchify_proj
            .forward(&latent_tokens.to_dtype(DType::BF16)?)?;

        // adaLN-single + prompt-adaLN. Timestep enters as f32 (× scale_multiplier in f32, sinusoid in
        // f32) so the bf16-rounding path is avoided (f32 input is the reference's "unaffected" case).
        let ts_scaled = (sigma * self.cfg.timestep_scale_multiplier) as f32;
        let ts_flat = Tensor::from_vec(vec![ts_scaled], (b,), &self.device)?;
        let (ts_emb, emb_ts) = self.adaln.forward(&ts_flat, &self.device)?;
        let ts_emb = ts_emb.reshape((b, 1, coeff * inner))?;
        let emb_ts = emb_ts.reshape((b, 1, inner))?;
        let (prompt_ts, _) = self.prompt_adaln.forward(&ts_flat, &self.device)?;
        let prompt_ts = prompt_ts.reshape((b, 1, 2 * inner))?;

        let context = context.to_dtype(DType::BF16)?;
        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h, &ts_emb, &prompt_ts, &context, cos, sin)?;
        }
        self.output_head(&h, &emb_ts)
    }

    fn output_head(&self, h: &Tensor, emb_ts: &Tensor) -> Result<Tensor> {
        let b = h.dim(0)?;
        let inner = self.cfg.inner_dim();
        let table = self.scale_shift_table.reshape((1, 1, 2, inner))?;
        let ss = table.broadcast_add(&emb_ts.reshape((b, 1, 1, inner))?)?; // (b,1,2,inner)
        let shift = ss.narrow(2, 0, 1)?.squeeze(2)?;
        let scale = ss.narrow(2, 1, 1)?.squeeze(2)?;
        let normed = layer_norm_noaffine(h, self.cfg.norm_eps)?;
        let out = modulate(&normed, &scale, &shift)?;
        self.proj_out.forward(&out)
    }
}
