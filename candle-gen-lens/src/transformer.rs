//! The Lens denoising **DiT** (`LensTransformer2DModel`, sc-5112) — a 48-layer dual-stream MMDiT with
//! joint image+text attention, complex axial RoPE on both streams, and SwiGLU MLPs. A from-scratch
//! candle port of the vendor `LensTransformer2DModel`, architecturally a near-twin of
//! [`candle-gen-qwen-image`]'s MMDiT (the RoPE, joint attention, AdaLN modulation and
//! `AdaLayerNormContinuous` all follow that seam). The Lens-specific pieces are:
//!
//! - a **multi-layer text front-end** — the 4 captured gpt-oss layers (each `[B, txt, 2880]`) get a
//!   per-layer affine RMSNorm (eps **1e-5**) then channel-concat (`2880·4 = 11520`) → `txt_in`;
//! - **fused** per-stream `img_qkv` / `txt_qkv` projections (split into q/k/v after the matmul);
//! - **`[img, txt]`** join order (image tokens first — the reference orders image first);
//! - **SwiGLU GateMLP** (`w2(silu(w1·x) · w3·x)`, hidden `inner/3·8 = 4096`);
//! - affine **RMSNorm** block norms (`rms_norm=True`, eps 1e-6) rather than affine-free LayerNorm;
//! - a **biased** `norm_out.linear` (the checkpoint's `AdaLayerNormContinuous` uses the bias).
//!
//! `[B, seq, dim]` tensors throughout. The model consumes already-patchified image latents
//! `[B, img_len, 128]` plus the 4 captured text-feature layers and predicts the patch-space velocity
//! `[B, img_len, 128]` (= `patch²·out_channels`). Run bf16 in production / f32 for the parity gate.

use candle_gen::candle_core::quantized::GgmlDType;
use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    linear, ops::softmax_last_dim, rms_norm, Linear, Module, RmsNorm, VarBuilder,
};
use candle_gen::gen_core::Quant;

use crate::quant::QLinear;
use crate::rope::{apply_rope, LensRope};

/// Block / QK-norm / norm_out epsilon (the reference builds its norms at eps 1e-6). Shared with the
/// trainable twin ([`crate::dit_train`]) so the two stay in lockstep.
pub const EPS: f64 = 1e-6;
/// The multi-layer text front-end RMSNorm epsilon (the `txt_norm` per-layer norms use eps 1e-5).
pub const TXT_NORM_EPS: f64 = 1e-5;

/// Which DiT projection a quant decision applies to ([`DitQuantPlan`]).
#[derive(Clone, Copy)]
enum DitRole {
    ImgIn,
    TxtIn,
    ProjOut,
    Attn,
    Mlp,
}

/// Per-role GGUF block type for [`LensTransformer::quantize`]. `None` for a role keeps those linears
/// dense (bf16).
///
/// **sc-7702 — the SwiGLU MLP stays at Q8 in the Q4 tier.** Uniform `Q4_0` (or `Q4_K`/`Q5_K`) across
/// the DiT makes the denoise **diverge to NaN** (a black render): the [`GateMlp`]'s *unbounded*
/// `silu(w1·x)·(w3·x)` product amplifies the 4/5-bit quant error and the latent blows up over the
/// denoise steps until a SwiGLU activation overflows bf16 (~step 3). Empirically (GPU sweep, sc-7545
/// box) **only Q8 on the MLP** holds — Q4_0/Q4_K/Q5_K MLP all diverge — while attention tolerates Q4
/// (its softmax-weighted output is bounded). So `Quant::Q4` ⇒ Q8 MLP + Q4_0 attention/`img_in`/
/// `txt_in`/`proj_out`; `Quant::Q8` ⇒ uniform Q8. Every DiT linear's `in_features` is ÷32, so both
/// `Q4_0` and `Q8_0` are always valid (no k-quant / fallback needed).
#[derive(Clone, Copy)]
struct DitQuantPlan {
    img_in: Option<GgmlDType>,
    txt_in: Option<GgmlDType>,
    proj_out: Option<GgmlDType>,
    attn: Option<GgmlDType>,
    mlp: Option<GgmlDType>,
}

impl DitQuantPlan {
    fn from_quant(quant: Quant) -> Self {
        match quant {
            // Q8 is uniformly stable.
            Quant::Q8 => {
                let q8 = Some(GgmlDType::Q8_0);
                Self {
                    img_in: q8,
                    txt_in: q8,
                    proj_out: q8,
                    attn: q8,
                    mlp: q8,
                }
            }
            // Q4: Q8 the divergence-prone SwiGLU MLP, Q4_0 everything else (sc-7702).
            Quant::Q4 => Self {
                img_in: Some(GgmlDType::Q4_0),
                txt_in: Some(GgmlDType::Q4_0),
                proj_out: Some(GgmlDType::Q4_0),
                attn: Some(GgmlDType::Q4_0),
                mlp: Some(GgmlDType::Q8_0),
            },
        }
    }

    /// The GGUF block type for `role` (`None` = keep dense).
    fn dtype(&self, role: DitRole) -> Option<GgmlDType> {
        match role {
            DitRole::ImgIn => self.img_in,
            DitRole::TxtIn => self.txt_in,
            DitRole::ProjOut => self.proj_out,
            DitRole::Attn => self.attn,
            DitRole::Mlp => self.mlp,
        }
    }
}

/// The Lens / Lens-Turbo `transformer/config.json` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensDitConfig {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub inner_dim: usize,
    /// gpt-oss hidden width per captured text layer (2880).
    pub enc_hidden_dim: usize,
    /// Number of captured gpt-oss layers (`selected_layer_index = [5, 11, 17, 23]`).
    pub num_text_layers: usize,
    /// Sinusoidal timestep-embedding width (256).
    pub timestep_channels: usize,
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f32,
}

impl LensDitConfig {
    pub fn lens() -> Self {
        Self {
            patch_size: 2,
            in_channels: 128,
            out_channels: 32,
            num_layers: 48,
            num_heads: 24,
            head_dim: 64,
            inner_dim: 1536,
            enc_hidden_dim: 2880,
            num_text_layers: 4,
            timestep_channels: 256,
            axes_dims_rope: [8, 28, 28],
            rope_theta: 10_000.0,
        }
    }

    /// SwiGLU GateMLP hidden width: `inner/3·8` (= 4096).
    pub fn mlp_hidden(&self) -> usize {
        self.inner_dim / 3 * 8
    }

    /// Concatenated text front-end width: `enc_hidden_dim · num_text_layers` (= 11520).
    pub fn txt_in_dim(&self) -> usize {
        self.enc_hidden_dim * self.num_text_layers
    }
}

/// Affine-free LayerNorm over the last axis (dtype-preserving; computed in f32).
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + EPS)?.sqrt()?)?.to_dtype(dt)
}

/// Split a `[B, 3·inner]` modulation chunk into `(shift, scale, gate)`, each `[B, 1, inner]` —
/// the reference `_modulate` layout is **(shift, scale, gate)**.
fn chunk3(m: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let inner = m.dim(D::Minus1)? / 3;
    let shift = m.narrow(D::Minus1, 0, inner)?.unsqueeze(1)?;
    let scale = m.narrow(D::Minus1, inner, inner)?.unsqueeze(1)?;
    let gate = m.narrow(D::Minus1, 2 * inner, inner)?.unsqueeze(1)?;
    Ok((shift, scale, gate))
}

/// AdaLN modulate: returns `(x·(1+scale) + shift, gate)`. Composable (no fused op), so the trainable
/// twin ([`crate::dit_train`]) reuses it verbatim.
pub fn modulate(x: &Tensor, m: &Tensor) -> Result<(Tensor, Tensor)> {
    let (shift, scale, gate) = chunk3(m)?;
    let out = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    Ok((out, gate))
}

/// `x + gate·y`. Composable; reused by [`crate::dit_train`].
pub fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x.broadcast_add(&y.broadcast_mul(gate)?)
}

/// Sinusoidal timestep embedding `[1, dim]` from the raw sigma (diffusers `Timesteps(dim,
/// flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)`): arg `= σ·1000·freq`, `[cos | sin]`.
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

/// `temb = linear_2(silu(linear_1(proj(t))))`, `[1] → [1, inner]`. Frozen + composable, so the
/// trainable twin ([`crate::dit_train`]) reuses it (the timestep embed is upstream of every adapter).
pub struct TimeEmbed {
    linear_1: Linear,
    linear_2: Linear,
    channels: usize,
}

impl TimeEmbed {
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let te = vb.pp("timestep_embedder");
        Ok(Self {
            linear_1: linear(cfg.timestep_channels, inner, te.pp("linear_1"))?,
            linear_2: linear(inner, inner, te.pp("linear_2"))?,
            channels: cfg.timestep_channels,
        })
    }

    pub fn forward(&self, sigma: f32, device: &Device, dtype: DType) -> Result<Tensor> {
        let emb = timestep_embedding(sigma, self.channels, device)?.to_dtype(dtype)?;
        let h = self.linear_1.forward(&emb)?.silu()?;
        self.linear_2.forward(&h)
    }
}

/// SwiGLU MLP (`GateMLP`): `w2(silu(w1·x) · w3·x)`, all bias-less. Hidden width `inner/3·8`. The three
/// projections are [`QLinear`] so they can be Q4/Q8-quantized (sc-5117).
struct GateMlp {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl GateMlp {
    fn new(inner: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: QLinear::linear_no_bias(inner, hidden, vb.pp("w1"))?,
            w2: QLinear::linear_no_bias(hidden, inner, vb.pp("w2"))?,
            w3: QLinear::linear_no_bias(inner, hidden, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&gate.mul(&up)?)
    }

    fn quantize(&mut self, plan: &DitQuantPlan) -> Result<()> {
        let dt = plan.dtype(DitRole::Mlp);
        for w in [&mut self.w1, &mut self.w2, &mut self.w3] {
            w.quantize_to(dt)?;
        }
        Ok(())
    }
}

/// Lens joint (dual-stream) attention. **Fused** `img_qkv`/`txt_qkv` (biased) split into per-stream
/// q/k/v, per-head q/k RMSNorm, interleaved-complex RoPE on both streams, then SDPA over the
/// **`[img, txt]`**-concatenated sequence (image first), split back and projected (`to_out.0` for
/// image, `to_add_out` for text).
struct JointAttention {
    img_qkv: QLinear,
    txt_qkv: QLinear,
    to_out: QLinear,
    to_add_out: QLinear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let hd = cfg.head_dim;
        Ok(Self {
            img_qkv: QLinear::linear(inner, 3 * inner, vb.pp("img_qkv"))?,
            txt_qkv: QLinear::linear(inner, 3 * inner, vb.pp("txt_qkv"))?,
            to_out: QLinear::linear(inner, inner, vb.pp("to_out").pp("0"))?,
            to_add_out: QLinear::linear(inner, inner, vb.pp("to_add_out"))?,
            norm_q: rms_norm(hd, EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, EPS, vb.pp("norm_k"))?,
            norm_added_q: rms_norm(hd, EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(hd, EPS, vb.pp("norm_added_k"))?,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    /// Quantize the four fused/output projections to Q4/Q8 (sc-5117). Called **after** any adapter
    /// merge (the merge folds `W += δ` into the dense weight before the DiT is built, so the quantized
    /// base already carries the adapter delta). The QK-norm weights stay full precision.
    fn quantize(&mut self, plan: &DitQuantPlan) -> Result<()> {
        let dt = plan.dtype(DitRole::Attn);
        for l in [
            &mut self.img_qkv,
            &mut self.txt_qkv,
            &mut self.to_out,
            &mut self.to_add_out,
        ] {
            l.quantize_to(dt)?;
        }
        Ok(())
    }

    /// Fused QKV → `(q, k, v)` each `[B, seq, heads, head_dim]`.
    fn qkv(&self, lin: &QLinear, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let (h, hd) = (self.heads, self.head_dim);
        let t = lin.forward(x)?.reshape((b, s, 3, h, hd))?;
        let q = t.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
        let k = t.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let v = t.narrow(2, 2, 1)?.squeeze(2)?.contiguous()?;
        Ok((q, k, v))
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
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, img_seq, _) = img.dims3()?;
        let txt_seq = txt.dim(1)?;
        let (h, hd) = (self.heads, self.head_dim);

        let (iq, ik, iv) = self.qkv(&self.img_qkv, img)?;
        let (tq, tk, tv) = self.qkv(&self.txt_qkv, txt)?;

        // QK RMSNorm over head_dim (in `[B, seq, heads, head_dim]`).
        let iq = self.norm_q.forward(&iq)?;
        let ik = self.norm_k.forward(&ik)?;
        let tq = self.norm_added_q.forward(&tq)?;
        let tk = self.norm_added_k.forward(&tk)?;

        // To heads-first `[B, heads, seq, head_dim]`, then interleaved RoPE on q/k.
        let bhsd = |x: &Tensor| -> Result<Tensor> { x.transpose(1, 2)?.contiguous() };
        let iq = apply_rope(&bhsd(&iq)?, img_cos, img_sin)?;
        let ik = apply_rope(&bhsd(&ik)?, img_cos, img_sin)?;
        let iv = bhsd(&iv)?;
        let tq = apply_rope(&bhsd(&tq)?, txt_cos, txt_sin)?;
        let tk = apply_rope(&bhsd(&tk)?, txt_cos, txt_sin)?;
        let tv = bhsd(&tv)?;

        // Joint `[img, txt]` (image first) over the sequence axis.
        let q = Tensor::cat(&[&iq, &tq], 2)?;
        let k = Tensor::cat(&[&ik, &tk], 2)?;
        let v = Tensor::cat(&[&iv, &tv], 2)?;
        let scale = (hd as f64).powf(-0.5);
        let mut scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v.contiguous()?)?; // [B, heads, joint, head_dim]
        let joint = img_seq + txt_seq;
        let o = o.transpose(1, 2)?.reshape((b, joint, h * hd))?;

        // Split back at the image/text boundary (image first).
        let img_o = o.narrow(1, 0, img_seq)?.contiguous()?;
        let txt_o = o.narrow(1, img_seq, txt_seq)?.contiguous()?;
        Ok((
            self.to_out.forward(&img_o)?,
            self.to_add_out.forward(&txt_o)?,
        ))
    }
}

/// Lens dual-stream MMDiT block. Each stream (image, text) gets two AdaLN modulations from the
/// timestep embedding — `mod1` around the joint attention, `mod2` around the SwiGLU MLP — with gated
/// residuals. Norms are affine RMSNorm (eps 1e-6). Public so the parity gate (and the Q4/Q8 quant
/// path, sc-5117) can drive a single block in isolation.
pub struct LensTransformerBlock {
    img_mod: Linear,
    txt_mod: Linear,
    img_norm1: RmsNorm,
    img_norm2: RmsNorm,
    txt_norm1: RmsNorm,
    txt_norm2: RmsNorm,
    attn: JointAttention,
    img_mlp: GateMlp,
    txt_mlp: GateMlp,
}

impl LensTransformerBlock {
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let hidden = cfg.mlp_hidden();
        Ok(Self {
            img_mod: linear(inner, 6 * inner, vb.pp("img_mod").pp("1"))?,
            txt_mod: linear(inner, 6 * inner, vb.pp("txt_mod").pp("1"))?,
            img_norm1: rms_norm(inner, EPS, vb.pp("img_norm1"))?,
            img_norm2: rms_norm(inner, EPS, vb.pp("img_norm2"))?,
            txt_norm1: rms_norm(inner, EPS, vb.pp("txt_norm1"))?,
            txt_norm2: rms_norm(inner, EPS, vb.pp("txt_norm2"))?,
            attn: JointAttention::new(cfg, vb.pp("attn"))?,
            img_mlp: GateMlp::new(inner, hidden, vb.pp("img_mlp"))?,
            txt_mlp: GateMlp::new(inner, hidden, vb.pp("txt_mlp"))?,
        })
    }

    /// Quantize the block's compute-heavy linears to Q4/Q8 (sc-5117): the joint-attention projections
    /// and both SwiGLU MLPs. The AdaLN modulations (`img_mod`/`txt_mod`) and the RMSNorm weights stay
    /// full precision (small, and precision-sensitive — the modulation drives every gated residual).
    fn quantize(&mut self, plan: &DitQuantPlan) -> Result<()> {
        self.attn.quantize(plan)?;
        self.img_mlp.quantize(plan)?;
        self.txt_mlp.quantize(plan)?;
        Ok(())
    }

    /// Returns `(encoder_hidden_states, hidden_states)` (text, image) — the reference block's order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden: &Tensor,  // image [B, img_seq, inner]
        encoder: &Tensor, // text  [B, txt_seq, inner]
        temb: &Tensor,    // [B, inner]
        img_cos: &Tensor,
        img_sin: &Tensor,
        txt_cos: &Tensor,
        txt_sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // SiLU'd timestep → per-stream 6·inner modulation, split into mod1 (around attn) / mod2 (MLP).
        let act = temb.silu()?;
        let img_mod = self.img_mod.forward(&act)?;
        let txt_mod = self.txt_mod.forward(&act)?;
        let n = img_mod.dim(D::Minus1)? / 2;
        let (im0, im1) = (
            img_mod.narrow(D::Minus1, 0, n)?,
            img_mod.narrow(D::Minus1, n, n)?,
        );
        let (tm0, tm1) = (
            txt_mod.narrow(D::Minus1, 0, n)?,
            txt_mod.narrow(D::Minus1, n, n)?,
        );

        // attention path
        let (img_n, img_g1) = modulate(&self.img_norm1.forward(hidden)?, &im0)?;
        let (txt_n, txt_g1) = modulate(&self.txt_norm1.forward(encoder)?, &tm0)?;
        let (img_attn, txt_attn) = self
            .attn
            .forward(&img_n, &txt_n, img_cos, img_sin, txt_cos, txt_sin, mask)?;
        let hidden = gated(hidden, &img_g1, &img_attn)?;
        let encoder = gated(encoder, &txt_g1, &txt_attn)?;

        // feed-forward path (SwiGLU)
        let (img_n2, img_g2) = modulate(&self.img_norm2.forward(&hidden)?, &im1)?;
        let hidden = gated(&hidden, &img_g2, &self.img_mlp.forward(&img_n2)?)?;
        let (txt_n2, txt_g2) = modulate(&self.txt_norm2.forward(&encoder)?, &tm1)?;
        let encoder = gated(&encoder, &txt_g2, &self.txt_mlp.forward(&txt_n2)?)?;

        Ok((encoder, hidden))
    }
}

/// `AdaLayerNormContinuous`: affine-free LayerNorm scaled/shifted by `linear(silu(temb))`. The Lens
/// checkpoint's `norm_out.linear` carries a **bias** the reference uses. `[scale | shift]` →
/// `(1+scale)·LN(x) + shift`.
pub struct NormOut {
    linear: Linear,
}

impl NormOut {
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        Ok(Self {
            linear: linear(inner, 2 * inner, vb.pp("linear"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let p = self.linear.forward(&temb.silu()?)?;
        let inner = p.dim(D::Minus1)? / 2;
        let scale = p.narrow(D::Minus1, 0, inner)?.unsqueeze(1)?;
        let shift = p.narrow(D::Minus1, inner, inner)?.unsqueeze(1)?;
        layer_norm(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)
    }
}

/// The Lens denoising DiT (`LensTransformer2DModel`).
pub struct LensTransformer {
    img_in: QLinear,
    txt_norm: Vec<RmsNorm>, // per-layer text front-end RMSNorm (eps 1e-5)
    txt_in: QLinear,
    time_embed: TimeEmbed,
    blocks: Vec<LensTransformerBlock>,
    norm_out: NormOut,
    proj_out: QLinear,
    rope: LensRope,
    cfg: LensDitConfig,
    device: Device,
    dtype: DType,
}

impl LensTransformer {
    /// Load from a diffusers `transformer/` weight set at `dtype` (bf16 production / f32 gate).
    pub fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let mut txt_norm = Vec::with_capacity(cfg.num_text_layers);
        for i in 0..cfg.num_text_layers {
            txt_norm.push(rms_norm(
                cfg.enc_hidden_dim,
                TXT_NORM_EPS,
                vb.pp("txt_norm").pp(i),
            )?);
        }
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(LensTransformerBlock::new(
                cfg,
                vb.pp("transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            img_in: QLinear::linear(cfg.in_channels, inner, vb.pp("img_in"))?,
            txt_norm,
            txt_in: QLinear::linear(cfg.txt_in_dim(), inner, vb.pp("txt_in"))?,
            time_embed: TimeEmbed::new(cfg, vb.pp("time_text_embed"))?,
            blocks,
            norm_out: NormOut::new(cfg, vb.pp("norm_out"))?,
            proj_out: QLinear::linear(
                inner,
                cfg.patch_size * cfg.patch_size * cfg.out_channels,
                vb.pp("proj_out"),
            )?,
            rope: LensRope::new(cfg.rope_theta, cfg.axes_dims_rope),
            cfg: *cfg,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Fold the DiT's compute-heavy linears to Q4/Q8 in place (sc-5117): `img_in`, `txt_in`,
    /// `proj_out`, and every block's attention projections + SwiGLU MLPs. The timestep embedder, the
    /// AdaLN modulations, `norm_out`, and all RMSNorm weights stay full precision (small and
    /// precision-sensitive). Call **after** any adapter merge — the merge folds `W += δ` into the dense
    /// weight before the DiT is built, so quantizing here transcodes the already-adapted base. Mirrors
    /// `mlx-gen-lens::dit::LensTransformer::quantize` (sc-3175).
    ///
    /// The per-linear precision is the [`DitQuantPlan`] for `quant`: `Quant::Q8` is uniform Q8, while
    /// `Quant::Q4` keeps the **SwiGLU MLP at Q8** (4/5-bit MLP makes the denoise diverge to NaN —
    /// sc-7702) and Q4_0s the rest.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        let plan = DitQuantPlan::from_quant(quant);
        self.img_in.quantize_to(plan.dtype(DitRole::ImgIn))?;
        self.txt_in.quantize_to(plan.dtype(DitRole::TxtIn))?;
        self.proj_out.quantize_to(plan.dtype(DitRole::ProjOut))?;
        for block in &mut self.blocks {
            block.quantize(&plan)?;
        }
        Ok(())
    }

    /// Forward.
    ///
    /// - `hidden_states`: `[B, img_len, in_channels]` patchified image latents (`img_len = frame·h·w`).
    /// - `text_feats`: the `num_text_layers` captured gpt-oss layers, each `[B, txt_len, enc_hidden_dim]`.
    /// - `text_valid`: optional `[B, txt_len]` (1 = valid) → additive joint attention mask; `None` =
    ///   all text valid (no padding), the single-prompt path.
    /// - `timestep`: the scalar sigma in `[0, 1]`.
    /// - `(frame, h, w)`: the latent grid shape (`img_len = frame·h·w`).
    ///
    /// Returns `[B, img_len, patch²·out_channels]` (= 128) patch-space velocity.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        text_feats: &[Tensor],
        text_valid: Option<&Tensor>,
        timestep: f32,
        frame: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        assert_eq!(
            text_feats.len(),
            self.cfg.num_text_layers,
            "expected {} text-feature layers, got {}",
            self.cfg.num_text_layers,
            text_feats.len()
        );
        let img_len = hidden_states.dim(1)?;
        let txt_len = text_feats[0].dim(1)?;

        let mut hidden = self.img_in.forward(hidden_states)?;

        // Multi-layer text front-end: per-layer RMSNorm (eps 1e-5) → channel-concat → txt_in.
        let mut normed = Vec::with_capacity(self.cfg.num_text_layers);
        for (i, feat) in text_feats.iter().enumerate() {
            normed.push(self.txt_norm[i].forward(feat)?);
        }
        let normed_refs: Vec<&Tensor> = normed.iter().collect();
        let mut encoder = self
            .txt_in
            .forward(&Tensor::cat(&normed_refs, D::Minus1)?)?;

        let temb = self
            .time_embed
            .forward(timestep, &self.device, self.dtype)?;
        let (img_cos, img_sin) = self.rope.img_cos_sin(frame, h, w, &self.device)?;
        let (txt_cos, txt_sin) = self.rope.txt_cos_sin(txt_len, h, w, &self.device)?;

        let mask = match text_valid {
            Some(valid) => Some(build_joint_mask(valid, img_len, self.dtype, &self.device)?),
            None => None,
        };

        for block in &self.blocks {
            let (e, hs) = block.forward(
                &hidden,
                &encoder,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                mask.as_ref(),
            )?;
            encoder = e;
            hidden = hs;
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

/// Additive joint attention mask `[B, 1, 1, img_len + txt_len]`: image tokens always valid; text
/// positions follow `text_valid` (1 = valid). Padded positions get a large-negative additive term so
/// the softmax masks them out (`(valid − 1)·1e9`, valid → 0). Composable; reused by [`crate::dit_train`].
pub fn build_joint_mask(
    text_valid: &Tensor,
    img_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let (b, txt_len) = text_valid.dims2()?;
    let img_ones = Tensor::ones((b, img_len), DType::F32, device)?;
    let valid = Tensor::cat(&[&img_ones, &text_valid.to_dtype(DType::F32)?], 1)?;
    let additive = ((valid - 1.0)? * 1e9)?; // valid → 0, invalid → -1e9
    additive
        .reshape((b, 1, 1, img_len + txt_len))?
        .to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_match_checkpoint() {
        let c = LensDitConfig::lens();
        assert_eq!(c.num_layers, 48);
        assert_eq!(c.inner_dim, c.num_heads * c.head_dim); // 1536 = 24·64
        assert_eq!(c.in_channels, 128);
        assert_eq!(c.out_channels, 32);
        assert_eq!(c.patch_size * c.patch_size * c.out_channels, 128); // proj_out width
        assert_eq!(c.mlp_hidden(), 4096); // inner/3·8
        assert_eq!(c.txt_in_dim(), 11520); // 2880·4
        assert_eq!(c.axes_dims_rope.iter().sum::<usize>(), c.head_dim); // 8+28+28 = 64
    }

    /// sc-7702 regression: the Q4 plan must keep the SwiGLU MLP at **Q8_0** (4/5-bit MLP diverges to
    /// NaN over the denoise → black render) while Q4_0-ing the rest; Q8 stays uniform. Guards the fix
    /// against a future "quantize everything uniformly" regression (the e2e check is the GPU
    /// `lens-render` example's degeneracy guard).
    #[test]
    fn q4_plan_protects_swiglu_mlp() {
        let q4 = DitQuantPlan::from_quant(Quant::Q4);
        assert_eq!(
            q4.dtype(DitRole::Mlp),
            Some(GgmlDType::Q8_0),
            "Q4 MLP must be Q8"
        );
        for role in [
            DitRole::ImgIn,
            DitRole::TxtIn,
            DitRole::ProjOut,
            DitRole::Attn,
        ] {
            assert_eq!(
                q4.dtype(role),
                Some(GgmlDType::Q4_0),
                "Q4 non-MLP must be Q4_0"
            );
        }
        let q8 = DitQuantPlan::from_quant(Quant::Q8);
        for role in [
            DitRole::ImgIn,
            DitRole::TxtIn,
            DitRole::ProjOut,
            DitRole::Attn,
            DitRole::Mlp,
        ] {
            assert_eq!(
                q8.dtype(role),
                Some(GgmlDType::Q8_0),
                "Q8 must be uniform Q8"
            );
        }
    }
}
