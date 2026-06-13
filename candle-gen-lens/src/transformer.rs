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

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    linear, ops::softmax_last_dim, rms_norm, Linear, Module, RmsNorm, VarBuilder,
};
use candle_gen::gen_core::Quant;

use crate::quant::QLinear;
use crate::rope::{apply_rope, LensRope};

/// Block / QK-norm / norm_out epsilon (the reference builds its norms at eps 1e-6).
const EPS: f64 = 1e-6;
/// The multi-layer text front-end RMSNorm epsilon (the `txt_norm` per-layer norms use eps 1e-5).
const TXT_NORM_EPS: f64 = 1e-5;

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
    fn mlp_hidden(&self) -> usize {
        self.inner_dim / 3 * 8
    }

    /// Concatenated text front-end width: `enc_hidden_dim · num_text_layers` (= 11520).
    fn txt_in_dim(&self) -> usize {
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

/// AdaLN modulate: returns `(x·(1+scale) + shift, gate)`.
fn modulate(x: &Tensor, m: &Tensor) -> Result<(Tensor, Tensor)> {
    let (shift, scale, gate) = chunk3(m)?;
    let out = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    Ok((out, gate))
}

/// `x + gate·y`.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
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

/// `temb = linear_2(silu(linear_1(proj(t))))`, `[1] → [1, inner]`.
struct TimeEmbed {
    linear_1: Linear,
    linear_2: Linear,
    channels: usize,
}

impl TimeEmbed {
    fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
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

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.w1.quantize(quant)?;
        self.w2.quantize(quant)?;
        self.w3.quantize(quant)?;
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
    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.img_qkv.quantize(quant)?;
        self.txt_qkv.quantize(quant)?;
        self.to_out.quantize(quant)?;
        self.to_add_out.quantize(quant)?;
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
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.attn.quantize(quant)?;
        self.img_mlp.quantize(quant)?;
        self.txt_mlp.quantize(quant)?;
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
struct NormOut {
    linear: Linear,
}

impl NormOut {
    fn new(cfg: &LensDitConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        Ok(Self {
            linear: linear(inner, 2 * inner, vb.pp("linear"))?,
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
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.img_in.quantize(quant)?;
        self.txt_in.quantize(quant)?;
        self.proj_out.quantize(quant)?;
        for block in &mut self.blocks {
            block.quantize(quant)?;
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
/// the softmax masks them out (`(valid − 1)·1e9`, valid → 0).
fn build_joint_mask(
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
}
