//! The FLUX.2 **MMDiT** transformer. Port of `mlx-gen-flux2`'s `transformer.rs`, run in candle f32.
//!
//! Shape anchors (klein-9b): `inner_dim = 4096`, `in/out_channels = 128`, `joint_attention_dim =
//! 12288`, `num_heads = 32`, `head_dim = 128`, 8 double (joint) blocks + 24 single (fused parallel)
//! blocks. The joint sequence order is **`[txt, img]`** in every concat / RoPE / attention. The
//! double block returns `(txt, img)`.
//!
//! Parity-load-bearing details (verified against the fork): LayerNorms are affine-free with
//! `eps = 1e-6`; the per-head q/k RMSNorm uses `eps = 1e-5`; `modulate = (1+scale)·norm + shift`
//! (strong f32 1); modulation is **global** (produced once from `temb`, shared across all blocks of
//! a stream); the RoPE is interleaved (see [`crate::pos_embed`]); the timestep fed in is the **scaled
//! sigma `σ·1000`** and the velocity is applied with a negative `dt` (no negation, in the pipeline).

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    linear_no_bias, ops::softmax_last_dim, rms_norm, Linear, Module, RmsNorm, VarBuilder,
};

use crate::config::Flux2Config;
use crate::pos_embed::Flux2PosEmbed;

const LN_EPS: f64 = 1e-6;
const RMS_EPS: f64 = 1e-5;

/// Affine-free LayerNorm over the last axis (eps 1e-6), in f32.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + LN_EPS)?.sqrt()?)
}

/// `(1 + scale)·norm + shift`, broadcasting modulation `[B,1,D]` over `[B,S,D]`.
fn modulate(norm: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let one_plus = (scale + 1.0)?;
    norm.broadcast_mul(&one_plus)?.broadcast_add(shift)
}

/// `x + gate·y`, broadcasting gate `[B,1,D]`.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x + y.broadcast_mul(gate)?
}

/// SwiGLU: split the last axis in half, `silu(a)·b`.
fn swiglu(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let a = x.narrow(D::Minus1, 0, half)?;
    let b = x.narrow(D::Minus1, half, half)?;
    a.silu()? * b
}

/// Sinusoidal timestep embedding `[1, dim]`: `[cos(args) | sin(args)]`, `args = t · 10000^{-i/half}`
/// (diffusers `flip_sin_to_cos = True`, cos first).
fn timestep_embedding(t: f32, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    let ln10000 = 10000f32.ln();
    for i in 0..half {
        let freq = (-ln10000 * i as f32 / half as f32).exp();
        let arg = t * freq;
        cos[i] = arg.cos();
        sin[i] = arg.sin();
    }
    let cos = Tensor::from_vec(cos, (1, half), device)?;
    let sin = Tensor::from_vec(sin, (1, half), device)?;
    Tensor::cat(&[&cos, &sin], D::Minus1)
}

/// Max elements in a single attention scores tensor `[B,H,Sq,Sk]` before [`attention`] chunks over the
/// query rows. candle CUDA kernels index elements with **i32**, so a scores/probs tensor exceeding
/// `i32::MAX` (~2.147B) silently corrupts its tail — at 1024² the FLUX.2 edit joint sequence
/// `[txt, target, ref]` (~8.7k tokens) makes a 2.42B-element scores tensor and the trailing query rows
/// get garbage attention → a near-zero/wrong velocity → noise (sc-5487). 1.0B keeps each block well under
/// the limit while leaving the txt2img attention (≤ ~0.68B at 1024²) a single un-chunked pass.
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

/// Reshape `[B,S,inner]` → `[B,H,S,head_dim]`, applying per-head RMSNorm (over head_dim) when `norm`
/// is given (q/k), none for v.
fn to_heads(x: &Tensor, heads: usize, head_dim: usize, norm: Option<&RmsNorm>) -> Result<Tensor> {
    let (b, s, _) = x.dims3()?;
    let x = x.reshape((b, s, heads, head_dim))?;
    let x = match norm {
        Some(n) => n.forward(&x)?,
        None => x,
    };
    x.transpose(1, 2)?.contiguous() // [B,H,S,head_dim]
}

/// A sinusoidal-scalar embedding MLP: `timestep_embedding → linear_1 → silu → linear_2` → `[1, inner]`.
/// Shared by the timestep and (dev) guidance branches of `time_guidance_embed`.
struct SinEmbed {
    linear_1: Linear,
    linear_2: Linear,
    channels: usize,
}

impl SinEmbed {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            linear_1: linear_no_bias(cfg.timestep_channels, inner, vb.pp("linear_1"))?,
            linear_2: linear_no_bias(inner, inner, vb.pp("linear_2"))?,
            channels: cfg.timestep_channels,
        })
    }

    fn forward(&self, scalar: f32, device: &Device) -> Result<Tensor> {
        let emb = timestep_embedding(scalar, self.channels, device)?;
        let h = self.linear_1.forward(&emb)?.silu()?;
        self.linear_2.forward(&h)
    }
}

/// FLUX.2 `time_guidance_embed`: the timestep embedder (`timestep_embedder.*`, always present) plus —
/// on the guidance-distilled **dev** checkpoint only — a guidance embedder (`guidance_embedder.*`).
/// `temb = time_emb(σ·1000) + guidance_emb(guidance·1000)` (diffusers `Flux2TimestepGuidanceEmbeddings`,
/// no pooled-CLIP term); klein has no guidance embedder, so `temb` is the timestep embedding alone.
struct TimeGuidanceEmbed {
    timestep: SinEmbed,
    guidance: Option<SinEmbed>,
}

impl TimeGuidanceEmbed {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let timestep = SinEmbed::new(cfg, vb.pp("timestep_embedder"))?;
        // The guidance embedder exists only on dev; gate on the weight (mirrors the mlx `w.get(...)`
        // presence check) so a klein checkpoint loads without looking for absent keys.
        let guidance = if vb.contains_tensor("guidance_embedder.linear_1.weight") {
            Some(SinEmbed::new(cfg, vb.pp("guidance_embedder"))?)
        } else {
            None
        };
        Ok(Self { timestep, guidance })
    }

    /// `timestep` is fed as σ·1000 (the caller scales it). `guidance` is the raw guidance scale (e.g.
    /// 4.0); it is scaled ×1000 here (the diffusers `guidance = guidance * 1000` step) and added only
    /// when this is a dev transformer. A `Some(guidance)` on klein (no embedder) is silently ignored.
    fn forward(&self, timestep: f32, guidance: Option<f32>, device: &Device) -> Result<Tensor> {
        let mut temb = self.timestep.forward(timestep, device)?;
        if let (Some(g), Some(gemb)) = (guidance, &self.guidance) {
            temb = (temb + gemb.forward(g * 1000.0, device)?)?;
        }
        Ok(temb)
    }
}

/// Global modulation: `silu(temb) → linear → split 3·sets` → `sets × (shift, scale, gate)` (each
/// `[B,1,inner]`).
struct Modulation {
    linear: Linear,
    sets: usize,
}

impl Modulation {
    fn new(cfg: &Flux2Config, sets: usize, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            linear: linear_no_bias(inner, 3 * sets * inner, vb.pp("linear"))?,
            sets,
        })
    }

    fn forward(&self, temb: &Tensor) -> Result<Vec<(Tensor, Tensor, Tensor)>> {
        let m = self.linear.forward(&temb.silu()?)?.unsqueeze(1)?; // [B,1,3·sets·inner]
        let inner = m.dim(D::Minus1)? / (3 * self.sets);
        let mut out = Vec::with_capacity(self.sets);
        for i in 0..self.sets {
            let base = 3 * i * inner;
            let shift = m.narrow(D::Minus1, base, inner)?;
            let scale = m.narrow(D::Minus1, base + inner, inner)?;
            let gate = m.narrow(D::Minus1, base + 2 * inner, inner)?;
            out.push((shift, scale, gate));
        }
        Ok(out)
    }
}

/// Joint attention for a double block: separate img/txt q/k/v with per-head q/k RMSNorm, attention
/// over the concatenated `[txt, img]` sequence with interleaved RoPE, split back.
struct DoubleAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    to_add_out: Linear,
    norm_added_q: RmsNorm,
    norm_added_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl DoubleAttention {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hd = cfg.head_dim;
        Ok(Self {
            to_q: linear_no_bias(inner, inner, vb.pp("to_q"))?,
            to_k: linear_no_bias(inner, inner, vb.pp("to_k"))?,
            to_v: linear_no_bias(inner, inner, vb.pp("to_v"))?,
            to_out: linear_no_bias(inner, inner, vb.pp("to_out").pp("0"))?,
            norm_q: rms_norm(hd, RMS_EPS, vb.pp("norm_q"))?,
            norm_k: rms_norm(hd, RMS_EPS, vb.pp("norm_k"))?,
            add_q: linear_no_bias(inner, inner, vb.pp("add_q_proj"))?,
            add_k: linear_no_bias(inner, inner, vb.pp("add_k_proj"))?,
            add_v: linear_no_bias(inner, inner, vb.pp("add_v_proj"))?,
            to_add_out: linear_no_bias(inner, inner, vb.pp("to_add_out"))?,
            norm_added_q: rms_norm(hd, RMS_EPS, vb.pp("norm_added_q"))?,
            norm_added_k: rms_norm(hd, RMS_EPS, vb.pp("norm_added_k"))?,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    /// `norm_img` / `norm_txt`: the modulated, normed streams. Returns `(img_out, txt_out)`.
    fn forward(
        &self,
        norm_img: &Tensor,
        norm_txt: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (self.heads, self.head_dim);
        let txt_seq = norm_txt.dim(1)?;

        // img stream q/k/v
        let iq = to_heads(&self.to_q.forward(norm_img)?, h, hd, Some(&self.norm_q))?;
        let ik = to_heads(&self.to_k.forward(norm_img)?, h, hd, Some(&self.norm_k))?;
        let iv = to_heads(&self.to_v.forward(norm_img)?, h, hd, None)?;
        // txt stream q/k/v
        let tq = to_heads(
            &self.add_q.forward(norm_txt)?,
            h,
            hd,
            Some(&self.norm_added_q),
        )?;
        let tk = to_heads(
            &self.add_k.forward(norm_txt)?,
            h,
            hd,
            Some(&self.norm_added_k),
        )?;
        let tv = to_heads(&self.add_v.forward(norm_txt)?, h, hd, None)?;

        // Concat [txt, img] along the sequence axis, apply RoPE to the full q/k.
        let q = Tensor::cat(&[&tq, &iq], 2)?;
        let k = Tensor::cat(&[&tk, &ik], 2)?;
        let v = Tensor::cat(&[&tv, &iv], 2)?;
        let q = Flux2PosEmbed::apply(&q, cos, sin)?;
        let k = Flux2PosEmbed::apply(&k, cos, sin)?;

        let o = attention(&q, &k, &v, hd)?; // [B, txt_seq+img_seq, inner]
        let txt_out = o.narrow(1, 0, txt_seq)?;
        let img_out = o.narrow(1, txt_seq, o.dim(1)? - txt_seq)?;
        let txt_out = self.to_add_out.forward(&txt_out.contiguous()?)?;
        let img_out = self.to_out.forward(&img_out.contiguous()?)?;
        Ok((img_out, txt_out))
    }
}

/// SwiGLU feed-forward: `linear_in → swiglu → linear_out`.
struct FeedForward {
    linear_in: Linear,
    linear_out: Linear,
}

impl FeedForward {
    fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_in: linear_no_bias(in_dim, 2 * hidden, vb.pp("linear_in"))?,
            linear_out: linear_no_bias(hidden, in_dim, vb.pp("linear_out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = swiglu(&self.linear_in.forward(x)?)?;
        self.linear_out.forward(&h)
    }
}

struct DoubleBlock {
    attn: DoubleAttention,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl DoubleBlock {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let ff_hidden = (cfg.mlp_ratio * inner as f32) as usize;
        Ok(Self {
            attn: DoubleAttention::new(cfg, vb.pp("attn"))?,
            ff: FeedForward::new(inner, ff_hidden, vb.pp("ff"))?,
            ff_context: FeedForward::new(inner, ff_hidden, vb.pp("ff_context"))?,
        })
    }

    /// Returns `(txt, img)` (note order).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod: &[(Tensor, Tensor, Tensor)],
        txt_mod: &[(Tensor, Tensor, Tensor)],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (shift_msa, scale_msa, gate_msa) = &img_mod[0];
        let (shift_mlp, scale_mlp, gate_mlp) = &img_mod[1];
        let (c_shift_msa, c_scale_msa, c_gate_msa) = &txt_mod[0];
        let (c_shift_mlp, c_scale_mlp, c_gate_mlp) = &txt_mod[1];

        let norm_img = modulate(&layer_norm(img)?, scale_msa, shift_msa)?;
        let norm_txt = modulate(&layer_norm(txt)?, c_scale_msa, c_shift_msa)?;
        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt, cos, sin)?;
        let mut img = gated(img, gate_msa, &img_attn)?;
        let mut txt = gated(txt, c_gate_msa, &txt_attn)?;

        let norm_img2 = modulate(&layer_norm(&img)?, scale_mlp, shift_mlp)?;
        let img_ff = self.ff.forward(&norm_img2)?;
        img = gated(&img, gate_mlp, &img_ff)?;

        let norm_txt2 = modulate(&layer_norm(&txt)?, c_scale_mlp, c_shift_mlp)?;
        let txt_ff = self.ff_context.forward(&norm_txt2)?;
        txt = gated(&txt, c_gate_mlp, &txt_ff)?;

        Ok((txt, img))
    }
}

/// Single (fused parallel attention + SwiGLU) block: one projection produces q/k/v and the MLP input.
struct SingleBlock {
    to_qkv_mlp: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    inner: usize,
    heads: usize,
    head_dim: usize,
}

impl SingleBlock {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mlp_hidden = cfg.single_mlp_hidden();
        let proj_out = 3 * inner + 2 * mlp_hidden;
        // The single block's projections nest under `attn.` in the diffusers checkpoint.
        let attn = vb.pp("attn");
        Ok(Self {
            to_qkv_mlp: linear_no_bias(inner, proj_out, attn.pp("to_qkv_mlp_proj"))?,
            to_out: linear_no_bias(inner + mlp_hidden, inner, attn.pp("to_out"))?,
            norm_q: rms_norm(cfg.head_dim, RMS_EPS, attn.pp("norm_q"))?,
            norm_k: rms_norm(cfg.head_dim, RMS_EPS, attn.pp("norm_k"))?,
            inner,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        m: &(Tensor, Tensor, Tensor),
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (shift, scale, gate) = m;
        let norm = modulate(&layer_norm(hidden)?, scale, shift)?;
        let proj = self.to_qkv_mlp.forward(&norm)?;
        let inner = self.inner;
        let q = proj.narrow(D::Minus1, 0, inner)?;
        let k = proj.narrow(D::Minus1, inner, inner)?;
        let v = proj.narrow(D::Minus1, 2 * inner, inner)?;
        let mlp = proj.narrow(D::Minus1, 3 * inner, proj.dim(D::Minus1)? - 3 * inner)?;

        let q = to_heads(&q, self.heads, self.head_dim, Some(&self.norm_q))?;
        let k = to_heads(&k, self.heads, self.head_dim, Some(&self.norm_k))?;
        let v = to_heads(&v, self.heads, self.head_dim, None)?;
        let q = Flux2PosEmbed::apply(&q, cos, sin)?;
        let k = Flux2PosEmbed::apply(&k, cos, sin)?;
        let attn = attention(&q, &k, &v, self.head_dim)?; // [B,S,inner]

        let mlp = swiglu(&mlp)?; // [B,S,mlp_hidden]
        let cat = Tensor::cat(&[&attn, &mlp], D::Minus1)?;
        let attn_output = self.to_out.forward(&cat)?;
        gated(hidden, gate, &attn_output)
    }
}

/// AdaLayerNorm-Continuous output head: `silu(temb) → linear → (scale, shift)`, then
/// `(1+scale)·LN(x) + shift`.
struct NormOut {
    linear: Linear,
}

impl NormOut {
    fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            linear: linear_no_bias(inner, 2 * inner, vb.pp("linear"))?,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let p = self.linear.forward(&temb.silu()?)?.unsqueeze(1)?; // [B,1,2·inner]
        let inner = p.dim(D::Minus1)? / 2;
        let scale = p.narrow(D::Minus1, 0, inner)?; // scale first
        let shift = p.narrow(D::Minus1, inner, inner)?;
        modulate(&layer_norm(x)?, &scale, &shift)
    }
}

/// The FLUX.2 MMDiT.
pub struct Flux2Transformer {
    x_embedder: Linear,
    context_embedder: Linear,
    time_embed: TimeGuidanceEmbed,
    mod_img: Modulation,
    mod_txt: Modulation,
    mod_single: Modulation,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out: NormOut,
    proj_out: Linear,
    pos_embed: Flux2PosEmbed,
    device: Device,
}

impl Flux2Transformer {
    pub fn new(cfg: &Flux2Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let mut double_blocks = Vec::with_capacity(cfg.num_double_layers);
        for i in 0..cfg.num_double_layers {
            double_blocks.push(DoubleBlock::new(cfg, vb.pp("transformer_blocks").pp(i))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        for i in 0..cfg.num_single_layers {
            single_blocks.push(SingleBlock::new(
                cfg,
                vb.pp("single_transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            x_embedder: linear_no_bias(cfg.in_channels, inner, vb.pp("x_embedder"))?,
            context_embedder: linear_no_bias(
                cfg.joint_attention_dim,
                inner,
                vb.pp("context_embedder"),
            )?,
            time_embed: TimeGuidanceEmbed::new(cfg, vb.pp("time_guidance_embed"))?,
            mod_img: Modulation::new(cfg, 2, vb.pp("double_stream_modulation_img"))?,
            mod_txt: Modulation::new(cfg, 2, vb.pp("double_stream_modulation_txt"))?,
            mod_single: Modulation::new(cfg, 1, vb.pp("single_stream_modulation"))?,
            double_blocks,
            single_blocks,
            norm_out: NormOut::new(cfg, vb.pp("norm_out"))?,
            proj_out: linear_no_bias(inner, cfg.out_channels, vb.pp("proj_out"))?,
            pos_embed: Flux2PosEmbed::new(cfg),
            device: vb.device().clone(),
        })
    }

    /// Predict velocity. `hidden_states` `[B, seq_img, 128]`, `encoder_hidden_states`
    /// `[B, seq_txt, joint]`, `img_ids`/`txt_ids` the 4-axis position ids, `timestep` = `σ·1000`.
    /// `guidance` is the raw embedded-guidance scale for the guidance-distilled **dev** path (e.g.
    /// 4.0), or `None` for klein (distilled / true-CFG); it is ignored unless this transformer carries
    /// the dev guidance embedder.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        img_ids: &[[i64; 4]],
        txt_ids: &[[i64; 4]],
        timestep: f32,
        guidance: Option<f32>,
    ) -> Result<Tensor> {
        let temb = self.time_embed.forward(timestep, guidance, &self.device)?;
        let mut img = self
            .x_embedder
            .forward(&hidden_states.to_dtype(DType::F32)?)?;
        let mut txt = self
            .context_embedder
            .forward(&encoder_hidden_states.to_dtype(DType::F32)?)?;

        // RoPE table over the [txt, img] sequence.
        let (txt_cos, txt_sin) = self.pos_embed.cos_sin(txt_ids, &self.device)?;
        let (img_cos, img_sin) = self.pos_embed.cos_sin(img_ids, &self.device)?;
        let cos = Tensor::cat(&[&txt_cos, &img_cos], 0)?;
        let sin = Tensor::cat(&[&txt_sin, &img_sin], 0)?;

        let img_mod = self.mod_img.forward(&temb)?;
        let txt_mod = self.mod_txt.forward(&temb)?;
        for block in &self.double_blocks {
            let (t, i) = block.forward(&img, &txt, &img_mod, &txt_mod, &cos, &sin)?;
            txt = t;
            img = i;
        }

        let txt_seq = txt.dim(1)?;
        let mut hidden = Tensor::cat(&[&txt, &img], 1)?;
        let single_mod = self.mod_single.forward(&temb)?;
        for block in &self.single_blocks {
            hidden = block.forward(&hidden, &single_mod[0], &cos, &sin)?;
        }

        let img_seq = hidden.dim(1)? - txt_seq;
        let img_out = hidden.narrow(1, txt_seq, img_seq)?;
        let img_out = self.norm_out.forward(&img_out.contiguous()?, &temb)?;
        self.proj_out.forward(&img_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_is_cos_then_sin_and_pos_zero_is_one_zero() {
        let emb = timestep_embedding(0.0, 256, &Device::Cpu).unwrap();
        assert_eq!(emb.dims(), &[1, 256]);
        let v = emb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // t=0: all args 0 → cos 1 (first half), sin 0 (second half).
        for c in &v[..128] {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in &v[128..] {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn swiglu_halves_last_dim() {
        let x = Tensor::ones((1, 2, 8), DType::F32, &Device::Cpu).unwrap();
        let y = swiglu(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2, 4]);
    }

    #[test]
    fn chunked_attention_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single pass bit-for-bit — the guard for the i32-overflow fix (sc-5487).
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

    #[test]
    fn modulate_is_one_plus_scale() {
        let norm = Tensor::ones((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let scale = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let shift = Tensor::ones((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        // (1+0)*1 + 1 = 2
        let out = modulate(&norm, &scale, &shift).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for x in v {
            assert!((x - 2.0).abs() < 1e-6);
        }
    }
}
