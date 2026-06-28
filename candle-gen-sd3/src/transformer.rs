//! The SD3.5 **MMDiT** (joint / double-stream) transformer (sc-7876, epic 7982).
//!
//! Port of the public diffusers `SD3Transformer2DModel` to candle, in f32. Adapted from the
//! candle-gen-flux2 MMDiT template ([`candle_gen_flux2`]'s `transformer.rs`) with the FLUX-specific
//! pieces swapped for SD3.5's:
//!  - **NO RoPE.** This is the #1 parity risk vs FLUX. SD3.5 adds a **learned 2D positional
//!    embedding** to the image tokens at patchify (cropped from a `pos_embed_max_size` grid) and the
//!    attention then runs *without* any rotary embedding. We do not build or apply any RoPE table.
//!  - **AdaLN-Zero modulation** derived from `temb = timestep_emb + pooled_proj` (the
//!    `CombinedTimestepTextProjEmbeddings`). Each joint block's image stream gets a 6-chunk
//!    modulation `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)` from its own
//!    `norm1.linear`; the text stream gets the same from `norm1_context.linear` — EXCEPT the LAST
//!    block, whose context stream is `context_pre_only` (2-chunk scale/shift, no gate, no ff_context).
//!  - **Modulation application order** (the mlx-port bug magnet, epic 7841): the AdaLN-Zero chunk
//!    split is `chunk(6) -> [shift, scale, gate_msa, shift, scale, gate_mlp]` and the apply is
//!    `x * (1 + scale) + shift` then `x + gate * sublayer(...)`. Pinned by [`tests`].
//!
//! Shape anchors (Large): `inner_dim = 2432`, `num_heads = 38`, `head_dim = 64`, 38 joint blocks,
//! `in/out_channels = 16`, `patch_size = 2`, `joint_attention_dim = 4096`. The joint attention runs
//! over the concatenated `[context(text), hidden(image)]` sequence (diffusers concats text-first in
//! `JointTransformerBlock`), splitting back after.
//!
//! The forward predicts the flow-match velocity in patchified space and unpatchifies to the
//! 16-channel latent. C1 builds the FULL Large forward at real shapes; C2 drives it from a pipeline.

use candle_gen::candle_core::{DType, Module, Result, Tensor, D};
use candle_gen::candle_nn::{self, LayerNorm, Linear, RmsNorm, VarBuilder};

use crate::config::Sd3Config;

/// Affine-free LayerNorm eps (diffusers `elementwise_affine=False, eps=1e-6` on the AdaLN norms).
const LN_EPS: f64 = 1e-6;
/// Per-head QK RMSNorm eps (diffusers `qk_norm="rms_norm"`, eps 1e-6).
const RMS_EPS: f64 = 1e-6;

/// Affine-free LayerNorm over the last axis (eps 1e-6), in f32 — the base norm inside every AdaLN.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + LN_EPS)?.sqrt()?)
}

/// AdaLN-continuous / AdaLN-Zero apply: `(1 + scale) * norm + shift`, broadcasting modulation
/// `[B,1,D]` over `[B,S,D]`. **This is the order the SD3.5 port must use** (epic 7841 flagged the
/// `(1+scale)` vs `scale` and the chunk-order as the bug magnet).
fn modulate(norm: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let one_plus = (scale + 1.0)?;
    norm.broadcast_mul(&one_plus)?.broadcast_add(shift)
}

/// `x + gate * y`, broadcasting gate `[B,1,D]`.
fn gated(x: &Tensor, gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    x + y.broadcast_mul(gate)?
}

/// Sinusoidal timestep embedding `[B, dim]` from a `[B]` timestep vector. diffusers
/// `get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0)`: `[cos(args) | sin(args)]`,
/// `args = t * exp(-ln(10000) * i / half)`. SD3 scales the timestep by 1000 before embedding (handled
/// by the caller).
fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let device = t.device();
    let half = dim / 2;
    let ln10000 = 10000f64.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-ln10000 * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), device)?; // [1, half]
    let t = t.to_dtype(DType::F32)?.reshape(((), 1))?; // [B, 1]
    let args = t.broadcast_mul(&freqs)?; // [B, half]
    Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1) // [B, dim] (cos first)
}

/// `timestep_embedding -> linear_1 -> silu -> linear_2`. Used by the timestep branch and the pooled
/// text-projection branch of [`CombinedTimestepTextEmbed`].
struct MlpEmbed {
    linear_1: Linear,
    linear_2: Linear,
}

impl MlpEmbed {
    fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: candle_nn::linear(in_dim, hidden, vb.pp("linear_1"))?,
            linear_2: candle_nn::linear(hidden, hidden, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear_2.forward(&self.linear_1.forward(x)?.silu()?)
    }
}

/// `CombinedTimestepTextProjEmbeddings`: `temb = time_mlp(sinusoid(σ·1000)) + text_mlp(pooled)`.
/// `pooled` is the 2048-wide aggregator output. Both MLPs project into `inner_dim`.
struct CombinedTimestepTextEmbed {
    timestep: MlpEmbed,
    text: MlpEmbed,
    timestep_channels: usize,
}

impl CombinedTimestepTextEmbed {
    fn new(cfg: &Sd3Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        Ok(Self {
            // `timestep_embedder.*` consumes the `timestep_channels` (256) sinusoid.
            timestep: MlpEmbed::new(cfg.timestep_channels, inner, vb.pp("timestep_embedder"))?,
            // `text_embedder.*` projects the pooled 2048 vector.
            text: MlpEmbed::new(cfg.pooled_dim, inner, vb.pp("text_embedder"))?,
            timestep_channels: cfg.timestep_channels,
        })
    }

    /// `timesteps` `[B]` (already ×1000), `pooled` `[B, pooled_dim]` → `temb [B, inner]`.
    fn forward(&self, timesteps: &Tensor, pooled: &Tensor) -> Result<Tensor> {
        let sin = timestep_embedding(timesteps, self.timestep_channels)?;
        let t_emb = self.timestep.forward(&sin)?;
        let p_emb = self.text.forward(&pooled.to_dtype(DType::F32)?)?;
        t_emb + p_emb
    }
}

/// The learned-positional-embedding patch embedder (`pos_embed`): a conv2d patchify (kernel =
/// stride = patch_size) projecting the 16-ch latent into `inner_dim` tokens, then **adds** the
/// learned 2D positional embedding cropped from a `pos_embed_max_size` grid.
///
/// The `pos_embed` parameter is stored flattened `[1, max^2, inner]`; for an `h×w` patch grid the
/// crop is the centred `h×w` window of the `max×max` grid (diffusers `cropped_pos_embed`). Cropping
/// (not interpolating) is the SD3.5 behaviour.
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    pos_embed: Tensor,
    max_size: usize,
    inner: usize,
}

impl PatchEmbed {
    fn new(cfg: &Sd3Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(
            cfg.in_channels,
            cfg.inner_dim,
            cfg.patch_size,
            conv_cfg,
            vb.pp("proj"),
        )?;
        // `[1, max_size*max_size, inner_dim]`.
        let pos_embed = vb.get(
            (
                1,
                cfg.pos_embed_max_size * cfg.pos_embed_max_size,
                cfg.inner_dim,
            ),
            "pos_embed",
        )?;
        Ok(Self {
            proj,
            pos_embed,
            max_size: cfg.pos_embed_max_size,
            inner: cfg.inner_dim,
        })
    }

    /// Centre-crop the `[1, max², inner]` table to an `h×w` token grid → `[1, h*w, inner]`. Mirrors
    /// diffusers `cropped_pos_embed`: reshape to `[1, max, max, inner]`, slice `[top:top+h,
    /// left:left+w]`, flatten.
    fn cropped_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        let top = (self.max_size - h) / 2;
        let left = (self.max_size - w) / 2;
        let grid = self
            .pos_embed
            .reshape((1, self.max_size, self.max_size, self.inner))?;
        grid.narrow(1, top, h)?
            .narrow(2, left, w)?
            .reshape((1, h * w, self.inner))
    }

    /// `latent [B, C, H, W]` → image tokens `[B, h*w, inner]` with the learned pos-embed added.
    /// `(h, w)` is the patchified grid (`H/patch_size`, `W/patch_size`).
    fn forward(&self, latent: &Tensor) -> Result<(Tensor, usize, usize)> {
        let latent = latent.to_dtype(DType::F32)?;
        // conv2d patchify -> [B, inner, h, w]
        let x = self.proj.forward(&latent)?;
        let (b, _c, h, w) = x.dims4()?;
        // -> [B, h*w, inner]
        let x = x.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let pos = self.cropped_pos_embed(h, w)?;
        let x = x.broadcast_add(&pos)?;
        Ok((x.reshape((b, h * w, self.inner))?, h, w))
    }
}

/// AdaLN-Zero: `silu(temb) -> linear -> n_chunks × [B,1,inner]`. The image norm uses 6 chunks; the
/// non-final context norm uses 6; the final context norm uses 2 (scale, shift; `context_pre_only`).
struct AdaLayerNormZero {
    linear: Linear,
    n_chunks: usize,
}

impl AdaLayerNormZero {
    fn new(inner: usize, n_chunks: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(inner, n_chunks * inner, vb.pp("linear"))?,
            n_chunks,
        })
    }

    /// Returns the `n_chunks` modulation slices, each `[B,1,inner]`.
    fn forward(&self, temb: &Tensor) -> Result<Vec<Tensor>> {
        let m = self.linear.forward(&temb.silu()?)?.unsqueeze(1)?; // [B,1,n*inner]
        let inner = m.dim(D::Minus1)? / self.n_chunks;
        (0..self.n_chunks)
            .map(|i| m.narrow(D::Minus1, i * inner, inner))
            .collect()
    }
}

/// Reshape `[B,S,inner]` → `[B,H,S,head_dim]`, applying per-head RMSNorm (over head_dim) when given
/// (q/k), none for v.
fn to_heads(x: &Tensor, heads: usize, head_dim: usize, norm: Option<&RmsNorm>) -> Result<Tensor> {
    let (b, s, _) = x.dims3()?;
    let x = x.reshape((b, s, heads, head_dim))?;
    let x = match norm {
        Some(n) => n.forward(&x)?,
        None => x,
    };
    x.transpose(1, 2)?.contiguous() // [B,H,S,head_dim]
}

/// SDPA over `[B,H,S,D]` q/k/v → `[B, S, H·D]`, scale `head_dim^-0.5`. No RoPE (SD3.5 uses learned
/// pos-embed only). Composable softmax so the math is portable.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> Result<Tensor> {
    let (b, h, s, d) = q.dims4()?;
    let scale = (head_dim as f64).powf(-0.5);
    let q = q.contiguous()?;
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;
    let scores = (q.matmul(&k_t)? * scale)?;
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let o = probs.matmul(&v)?; // [B,H,S,D]
    o.transpose(1, 2)?.reshape((b, s, h * d))
}

/// Joint attention for a SD3.5 joint block: separate image (`to_q/k/v`) and text (`add_q/k/v_proj`)
/// projections with per-head q/k RMSNorm, attention over the concatenated `[text, image]` sequence
/// (NO RoPE), split back. Returns `(image_out, text_out)`; `text_out` is `None` when the block is
/// `context_pre_only` (the final block has no `to_add_out`).
struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    to_add_out: Option<Linear>,
    norm_q: Option<RmsNorm>,
    norm_k: Option<RmsNorm>,
    norm_added_q: Option<RmsNorm>,
    norm_added_k: Option<RmsNorm>,
    heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn new(cfg: &Sd3Config, context_pre_only: bool, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let hd = cfg.head_dim;
        let (norm_q, norm_k, norm_added_q, norm_added_k) = if cfg.qk_norm {
            (
                Some(candle_nn::rms_norm(hd, RMS_EPS, vb.pp("norm_q"))?),
                Some(candle_nn::rms_norm(hd, RMS_EPS, vb.pp("norm_k"))?),
                Some(candle_nn::rms_norm(hd, RMS_EPS, vb.pp("norm_added_q"))?),
                Some(candle_nn::rms_norm(hd, RMS_EPS, vb.pp("norm_added_k"))?),
            )
        } else {
            (None, None, None, None)
        };
        // The image-stream output projection lives at `to_out.0`.
        let to_out = candle_nn::linear(inner, inner, vb.pp("to_out").pp("0"))?;
        // The text-stream output projection is absent on the `context_pre_only` (final) block.
        let to_add_out = if context_pre_only {
            None
        } else {
            Some(candle_nn::linear(inner, inner, vb.pp("to_add_out"))?)
        };
        Ok(Self {
            to_q: candle_nn::linear(inner, inner, vb.pp("to_q"))?,
            to_k: candle_nn::linear(inner, inner, vb.pp("to_k"))?,
            to_v: candle_nn::linear(inner, inner, vb.pp("to_v"))?,
            to_out,
            add_q: candle_nn::linear(inner, inner, vb.pp("add_q_proj"))?,
            add_k: candle_nn::linear(inner, inner, vb.pp("add_k_proj"))?,
            add_v: candle_nn::linear(inner, inner, vb.pp("add_v_proj"))?,
            to_add_out,
            norm_q,
            norm_k,
            norm_added_q,
            norm_added_k,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    /// `norm_img` / `norm_txt`: the modulated, normed streams. Returns `(img_out, Option<txt_out>)`.
    fn forward(&self, norm_img: &Tensor, norm_txt: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (h, hd) = (self.heads, self.head_dim);
        let txt_seq = norm_txt.dim(1)?;

        let iq = to_heads(&self.to_q.forward(norm_img)?, h, hd, self.norm_q.as_ref())?;
        let ik = to_heads(&self.to_k.forward(norm_img)?, h, hd, self.norm_k.as_ref())?;
        let iv = to_heads(&self.to_v.forward(norm_img)?, h, hd, None)?;
        let tq = to_heads(
            &self.add_q.forward(norm_txt)?,
            h,
            hd,
            self.norm_added_q.as_ref(),
        )?;
        let tk = to_heads(
            &self.add_k.forward(norm_txt)?,
            h,
            hd,
            self.norm_added_k.as_ref(),
        )?;
        let tv = to_heads(&self.add_v.forward(norm_txt)?, h, hd, None)?;

        // Concat [txt, img] along the sequence axis (NO RoPE).
        let q = Tensor::cat(&[&tq, &iq], 2)?;
        let k = Tensor::cat(&[&tk, &ik], 2)?;
        let v = Tensor::cat(&[&tv, &iv], 2)?;

        let o = attention(&q, &k, &v, hd)?; // [B, txt_seq+img_seq, inner]
        let txt_out = o.narrow(1, 0, txt_seq)?.contiguous()?;
        let img_out = o.narrow(1, txt_seq, o.dim(1)? - txt_seq)?.contiguous()?;

        let img_out = self.to_out.forward(&img_out)?;
        let txt_out = match &self.to_add_out {
            Some(p) => Some(p.forward(&txt_out)?),
            None => None,
        };
        Ok((img_out, txt_out))
    }
}

/// GELU feed-forward (diffusers `FeedForward` with `gelu` activation): `proj -> gelu -> out`.
struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        // diffusers FeedForward nests the input projection at `net.0.proj` and the output at `net.2`.
        Ok(Self {
            proj: candle_nn::linear(in_dim, hidden, vb.pp("net").pp("0").pp("proj"))?,
            out: candle_nn::linear(hidden, in_dim, vb.pp("net").pp("2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.proj.forward(x)?.gelu()?)
    }
}

/// One SD3.5 joint (double-stream) block. The image stream is always a full AdaLN-Zero block; the
/// text stream is full unless `context_pre_only` (the final block), where it only emits scale/shift
/// for its norm and has no attention output projection / ff_context.
struct JointBlock {
    norm1: AdaLayerNormZero,
    norm1_context: AdaLayerNormZero,
    attn: JointAttention,
    ff: FeedForward,
    ff_context: Option<FeedForward>,
    context_pre_only: bool,
}

impl JointBlock {
    fn new(cfg: &Sd3Config, context_pre_only: bool, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let ff_hidden = cfg.ff_hidden();
        // Image norm: 6 chunks (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        let norm1 = AdaLayerNormZero::new(inner, 6, vb.pp("norm1"))?;
        // Context norm: 6 chunks normally; 2 (scale, shift) when context_pre_only.
        let norm1_context = AdaLayerNormZero::new(
            inner,
            if context_pre_only { 2 } else { 6 },
            vb.pp("norm1_context"),
        )?;
        let attn = JointAttention::new(cfg, context_pre_only, vb.pp("attn"))?;
        let ff = FeedForward::new(inner, ff_hidden, vb.pp("ff"))?;
        let ff_context = if context_pre_only {
            None
        } else {
            Some(FeedForward::new(inner, ff_hidden, vb.pp("ff_context"))?)
        };
        Ok(Self {
            norm1,
            norm1_context,
            attn,
            ff,
            ff_context,
            context_pre_only,
        })
    }

    /// `(image, text, temb)` → updated `(image, text)`. For the final (`context_pre_only`) block the
    /// returned `text` is unchanged (only `image` matters downstream).
    fn forward(&self, img: &Tensor, txt: &Tensor, temb: &Tensor) -> Result<(Tensor, Tensor)> {
        // Image AdaLN-Zero: 6 chunks.
        let im = self.norm1.forward(temb)?;
        let (shift_msa, scale_msa, gate_msa) = (&im[0], &im[1], &im[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&im[3], &im[4], &im[5]);
        let norm_img = modulate(&layer_norm(img)?, scale_msa, shift_msa)?;

        // Context norm.
        let cm = self.norm1_context.forward(temb)?;
        let norm_txt = if self.context_pre_only {
            // 2-chunk scale/shift only.
            modulate(&layer_norm(txt)?, &cm[1], &cm[0])?
        } else {
            modulate(&layer_norm(txt)?, &cm[1], &cm[0])?
        };

        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt)?;

        // Image stream residual + ff.
        let mut img = gated(img, gate_msa, &img_attn)?;
        let norm_img2 = modulate(&layer_norm(&img)?, scale_mlp, shift_mlp)?;
        let img_ff = self.ff.forward(&norm_img2)?;
        img = gated(&img, gate_mlp, &img_ff)?;

        // Text stream: only on non-final blocks (final has no attn out + no ff_context).
        let txt = if self.context_pre_only {
            txt.clone()
        } else {
            let txt_attn = txt_attn.expect("non-final block produces a text attention output");
            let (c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
                (&cm[2], &cm[3], &cm[4], &cm[5]);
            let mut txt = gated(txt, c_gate_msa, &txt_attn)?;
            let norm_txt2 = modulate(&layer_norm(&txt)?, c_scale_mlp, c_shift_mlp)?;
            let txt_ff = self
                .ff_context
                .as_ref()
                .expect("non-final block has ff_context")
                .forward(&norm_txt2)?;
            txt = gated(&txt, c_gate_mlp, &txt_ff)?;
            txt
        };

        Ok((img, txt))
    }
}

/// AdaLayerNormContinuous output head: `silu(temb) -> linear -> (shift, scale)`, then
/// `(1+scale)·LN(x) + shift`. diffusers `AdaLayerNormContinuous` splits the linear output as
/// `chunk(2) -> [scale, shift]`? No — diffusers emits `[shift, scale]` for `norm_out`? It is
/// `emb = linear(silu(temb)); shift, scale = emb.chunk(2)`. We follow diffusers `norm_out`:
/// `scale, shift` order is `chunk(2)` with shift first. We pin the order in [`tests`].
struct AdaLayerNormContinuous {
    linear: Linear,
    norm: LayerNorm,
    has_affine: bool,
}

impl AdaLayerNormContinuous {
    fn new(inner: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(inner, 2 * inner, vb.pp("linear"))?;
        // diffusers `norm_out.norm` is affine-free LayerNorm; we keep a parameterless LayerNorm
        // (weight=1, bias=0) so `forward` is the plain normalization, then apply scale/shift.
        let weight = Tensor::ones(inner, DType::F32, vb.device())?;
        let bias = Tensor::zeros(inner, DType::F32, vb.device())?;
        let norm = LayerNorm::new(weight, bias, LN_EPS);
        Ok(Self {
            linear,
            norm,
            has_affine: false,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&temb.silu()?)?.unsqueeze(1)?; // [B,1,2*inner]
        let inner = emb.dim(D::Minus1)? / 2;
        // diffusers AdaLayerNormContinuous: `shift, scale = emb.chunk(2, dim=1)`; x = norm(x) * (1 +
        // scale) + shift.
        let shift = emb.narrow(D::Minus1, 0, inner)?;
        let scale = emb.narrow(D::Minus1, inner, inner)?;
        let normed = if self.has_affine {
            self.norm.forward(&x.to_dtype(DType::F32)?)?
        } else {
            layer_norm(x)?
        };
        modulate(&normed, &scale, &shift)
    }
}

/// The SD3.5 MMDiT transformer (Large by default; geometry from [`Sd3Config`]).
pub struct Sd3Transformer {
    pos_embed: PatchEmbed,
    time_text_embed: CombinedTimestepTextEmbed,
    context_embedder: Linear,
    blocks: Vec<JointBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: Linear,
    cfg: Sd3Config,
}

impl Sd3Transformer {
    pub fn new(cfg: &Sd3Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            // The LAST block is context_pre_only when the config flags it (SD3.5 default).
            let pre_only = cfg.context_pre_only_last && i == cfg.num_layers - 1;
            blocks.push(JointBlock::new(
                cfg,
                pre_only,
                vb.pp("transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            pos_embed: PatchEmbed::new(cfg, vb.pp("pos_embed"))?,
            time_text_embed: CombinedTimestepTextEmbed::new(cfg, vb.pp("time_text_embed"))?,
            context_embedder: candle_nn::linear(
                cfg.joint_attention_dim,
                inner,
                vb.pp("context_embedder"),
            )?,
            blocks,
            norm_out: AdaLayerNormContinuous::new(inner, vb.pp("norm_out"))?,
            proj_out: candle_nn::linear(inner, cfg.patch_dim(), vb.pp("proj_out"))?,
            cfg: cfg.clone(),
        })
    }

    /// Unpatchify image tokens `[B, h*w, patch_dim]` back to a `[B, C, H, W]` latent (the inverse of
    /// the conv2d patchify). `patch_dim = patch_size² · out_channels`.
    fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let p = self.cfg.patch_size;
        let c = self.cfg.in_channels;
        let b = x.dim(0)?;
        // [B, h, w, p, p, c]
        let x = x.reshape((b, h, w, p, p, c))?;
        // -> [B, c, h, p, w, p]
        let x = x.permute((0, 5, 1, 3, 2, 4))?;
        // -> [B, c, h*p, w*p]
        x.reshape((b, c, h * p, w * p))
    }

    /// Predict the flow-match velocity. `latent` `[B, C, H, W]`, `context` `[B, ctx_seq, joint]`,
    /// `pooled` `[B, pooled_dim]`, `timesteps` `[B]` (the caller scales by 1000). Returns `[B, C, H,
    /// W]`.
    pub fn forward(
        &self,
        latent: &Tensor,
        context: &Tensor,
        pooled: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        let temb = self.time_text_embed.forward(timesteps, pooled)?; // [B, inner]
        let (mut img, h, w) = self.pos_embed.forward(latent)?; // [B, h*w, inner]
        let mut txt = self
            .context_embedder
            .forward(&context.to_dtype(DType::F32)?)?; // [B, ctx_seq, inner]

        for block in &self.blocks {
            let (i, t) = block.forward(&img, &txt, &temb)?;
            img = i;
            txt = t;
        }

        let img = self.norm_out.forward(&img, &temb)?;
        let img = self.proj_out.forward(&img)?; // [B, h*w, patch_dim]
        self.unpatchify(&img, h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::{VarBuilder, VarMap};

    /// A tiny SD3.5-shaped config: 1 head at head_dim 8 (inner 8), a couple of joint blocks, small
    /// CLIP/T5 dims — enough to exercise every vendored path cheaply on CPU.
    fn tiny_cfg() -> Sd3Config {
        Sd3Config {
            in_channels: 16,
            patch_size: 2,
            pos_embed_max_size: 8,
            inner_dim: 16,
            num_heads: 2,
            head_dim: 8,
            num_layers: 3,
            mlp_ratio: 2.0,
            qk_norm: true,
            context_pre_only_last: true,
            pooled_dim: 2048,
            joint_attention_dim: 4096,
            clip_l_dim: 768,
            clip_g_dim: 1280,
            clip_concat_dim: 2048,
            clip_seq_len: 77,
            t5_seq_len: 8,
            t5_dim: 4096,
            timestep_channels: 16,
        }
    }

    /// `modulate` is `(1+scale)·norm + shift` — the order epic 7841 flagged as the bug magnet.
    #[test]
    fn modulate_is_one_plus_scale_then_shift() {
        let dev = Device::Cpu;
        let norm = Tensor::full(2f32, (1, 2, 4), &dev).unwrap();
        let scale = Tensor::full(3f32, (1, 1, 4), &dev).unwrap();
        let shift = Tensor::full(5f32, (1, 1, 4), &dev).unwrap();
        // (1+3)*2 + 5 = 13
        let out = modulate(&norm, &scale, &shift).unwrap();
        for x in out.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!((x - 13.0).abs() < 1e-6, "modulate order wrong: {x}");
        }
    }

    /// AdaLN-Zero produces exactly `n_chunks` modulation slices of width `inner`.
    #[test]
    fn adaln_zero_chunk_count_and_width() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let inner = 16;
        let ada = AdaLayerNormZero::new(inner, 6, vb.pp("norm1")).unwrap();
        let temb = Tensor::randn(0f32, 1f32, (1, inner), &dev).unwrap();
        let chunks = ada.forward(&temb).unwrap();
        assert_eq!(chunks.len(), 6);
        for c in &chunks {
            assert_eq!(c.dims(), &[1, 1, inner]);
        }
    }

    /// timestep_embedding: cos first, sin second; t=0 -> [1...1, 0...0].
    #[test]
    fn timestep_embedding_cos_then_sin() {
        let dev = Device::Cpu;
        let t = Tensor::zeros(1, DType::F32, &dev).unwrap();
        let emb = timestep_embedding(&t, 8).unwrap();
        assert_eq!(emb.dims(), &[1, 8]);
        let v = emb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in &v[..4] {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in &v[4..] {
            assert!(s.abs() < 1e-6);
        }
    }

    /// patchify -> unpatchify round-trips the spatial geometry: a `[B,C,H,W]` latent yields the
    /// patch grid `(H/p, W/p)`, and unpatchify of a `[B, h*w, patch_dim]` returns `[B,C,H,W]`.
    #[test]
    fn patchify_unpatchify_round_trip_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let pe = PatchEmbed::new(&cfg, vb.pp("pos_embed")).unwrap();
        // latent 16ch, 8x8 -> patch grid 4x4 = 16 tokens.
        let latent = Tensor::randn(0f32, 1f32, (1, 16, 8, 8), &dev).unwrap();
        let (tokens, h, w) = pe.forward(&latent).unwrap();
        assert_eq!((h, w), (4, 4));
        assert_eq!(tokens.dims(), &[1, 16, cfg.inner_dim]);

        // Build a transformer just to use its unpatchify; feed a [B, h*w, patch_dim] tensor.
        let t = Sd3Transformer::new(&cfg, vb).unwrap();
        let patched = Tensor::randn(0f32, 1f32, (1, h * w, cfg.patch_dim()), &dev).unwrap();
        let out = t.unpatchify(&patched, h, w).unwrap();
        assert_eq!(out.dims(), &[1, 16, 8, 8]);
    }

    /// The FULL tiny MMDiT forward runs on CPU and produces a velocity with the input latent's shape.
    /// Base weights are randomized (the VarMap is filled with random init by `candle_nn::linear`),
    /// so the patch conv is NOT zero-init — the output is a non-trivial function of the input.
    #[test]
    fn tiny_mmdit_forward_produces_latent_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let model = Sd3Transformer::new(&cfg, vb).unwrap();

        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 8, 8), &dev).unwrap();
        // context seq = clip 77 + t5 8 = 85 at the tiny config.
        let ctx_seq = cfg.context_seq_len();
        let context =
            Tensor::randn(0f32, 1f32, (1, ctx_seq, cfg.joint_attention_dim), &dev).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &dev).unwrap();
        let t = Tensor::full(0.5f32, 1, &dev).unwrap();

        let v = model.forward(&latent, &context, &pooled, &t).unwrap();
        assert_eq!(v.dims(), latent.dims());
        // Non-vacuous: a randomized base must not yield an all-zero velocity.
        let max = v
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max > 0.0,
            "velocity is vacuously zero — base weights not randomized?"
        );
    }

    /// The final block is `context_pre_only`: its attention has no `to_add_out` and the block has no
    /// `ff_context`, so it must still forward (returning the text stream unchanged).
    #[test]
    fn final_block_context_pre_only_forwards() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let block = JointBlock::new(&cfg, true, vb.pp("transformer_blocks").pp(0)).unwrap();
        assert!(block.context_pre_only);
        assert!(block.ff_context.is_none());
        assert!(block.attn.to_add_out.is_none());

        let img = Tensor::randn(0f32, 1f32, (1, 4, cfg.inner_dim), &dev).unwrap();
        let txt = Tensor::randn(0f32, 1f32, (1, 6, cfg.inner_dim), &dev).unwrap();
        let temb = Tensor::randn(0f32, 1f32, (1, cfg.inner_dim), &dev).unwrap();
        let (img_out, txt_out) = block.forward(&img, &txt, &temb).unwrap();
        assert_eq!(img_out.dims(), img.dims());
        // The text stream is returned unchanged on the final block.
        let same = (txt_out - txt)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            same < 1e-6,
            "context_pre_only block must leave text unchanged"
        );
    }
}
