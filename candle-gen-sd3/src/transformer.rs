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

use candle_gen::candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_gen::candle_nn::{self, Linear, RmsNorm, VarBuilder};
use candle_gen::gen_core::Quant;

use crate::config::Sd3Config;
use crate::quant::{conv2d_to, linear_to, rms_norm_to, QLinear};

/// Affine-free LayerNorm eps (diffusers `elementwise_affine=False, eps=1e-6` on the AdaLN norms).
const LN_EPS: f64 = 1e-6;
/// Per-head QK RMSNorm eps (diffusers `qk_norm="rms_norm"`, eps 1e-6).
const RMS_EPS: f64 = 1e-6;

/// Affine-free LayerNorm over the last axis (eps 1e-6) — the base norm inside every AdaLN. The
/// mean/variance reduction is computed in **f32** for numerical stability, then the normalized output
/// is cast **back to the input dtype** so the downstream modulation + projections all run in one
/// consistent dtype (the loaded weight dtype). On real bf16 weights, forcing the output to stay f32
/// produced an `F32 × BF16` matmul mismatch in the very first block (sc-7881, C6); the parameterless
/// reduction stays f32 but must not leak its dtype into the rest of the block.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let in_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + LN_EPS)?.sqrt()?)?
        .to_dtype(in_dtype)
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

    /// CPU-stage migration (sc-8504): move both dense linears to `device`.
    fn migrate_to(&mut self, device: &Device) -> Result<()> {
        self.linear_1 = linear_to(&self.linear_1, device)?;
        self.linear_2 = linear_to(&self.linear_2, device)?;
        Ok(())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // The sinusoid / pooled inputs arrive in f32; cast to the (loaded) weight dtype so the matmul
        // dtypes agree on real bf16 weights (sc-7881, C6).
        let x = x.to_dtype(self.linear_1.weight().dtype())?;
        self.linear_2.forward(&self.linear_1.forward(&x)?.silu()?)
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

    /// CPU-stage migration (sc-8504): move both dense MLP embedders to `device`.
    fn migrate_to(&mut self, device: &Device) -> Result<()> {
        self.timestep.migrate_to(device)?;
        self.text.migrate_to(device)
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

    /// CPU-stage migration (sc-8504): move the patchify conv + the learned pos-embed table to `device`.
    fn migrate_to(&mut self, device: &Device) -> Result<()> {
        self.proj = conv2d_to(&self.proj, device)?;
        self.pos_embed = self.pos_embed.to_device(device)?;
        Ok(())
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
        // Cast the latent to the conv weight dtype (the loaded weight dtype, e.g. bf16) so the
        // patchify conv + the learned-pos-embed add run in one consistent dtype (sc-7881, C6).
        let latent = latent.to_dtype(self.proj.weight().dtype())?;
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

    /// CPU-stage migration (sc-8504): move the dense modulation linear to `device`.
    fn migrate_to(&mut self, device: &Device) -> Result<()> {
        self.linear = linear_to(&self.linear, device)?;
        Ok(())
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
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    add_q: QLinear,
    add_k: QLinear,
    add_v: QLinear,
    to_add_out: Option<QLinear>,
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
        let to_out = QLinear::linear(inner, inner, vb.pp("to_out").pp("0"))?;
        // The text-stream output projection is absent on the `context_pre_only` (final) block.
        let to_add_out = if context_pre_only {
            None
        } else {
            Some(QLinear::linear(inner, inner, vb.pp("to_add_out"))?)
        };
        Ok(Self {
            to_q: QLinear::linear(inner, inner, vb.pp("to_q"))?,
            to_k: QLinear::linear(inner, inner, vb.pp("to_k"))?,
            to_v: QLinear::linear(inner, inner, vb.pp("to_v"))?,
            to_out,
            add_q: QLinear::linear(inner, inner, vb.pp("add_q_proj"))?,
            add_k: QLinear::linear(inner, inner, vb.pp("add_k_proj"))?,
            add_v: QLinear::linear(inner, inner, vb.pp("add_v_proj"))?,
            to_add_out,
            norm_q,
            norm_k,
            norm_added_q,
            norm_added_k,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    /// Fold every projection (image q/k/v/out + text add_q/k/v + to_add_out) to `Q4_0`/`Q8_0`
    /// **in place** on the weights' current device (the original on-device build path).
    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.to_q.quantize(quant)?;
        self.to_k.quantize(quant)?;
        self.to_v.quantize(quant)?;
        self.to_out.quantize(quant)?;
        self.add_q.quantize(quant)?;
        self.add_k.quantize(quant)?;
        self.add_v.quantize(quant)?;
        if let Some(p) = &mut self.to_add_out {
            p.quantize(quant)?;
        }
        Ok(())
    }

    /// CPU-stage path (sc-8504): quantize every projection **onto `device`** (dense never lands on the
    /// GPU) and migrate the dense-kept per-head q/k RMSNorms there too.
    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.to_q.quantize_onto(quant, device)?;
        self.to_k.quantize_onto(quant, device)?;
        self.to_v.quantize_onto(quant, device)?;
        self.to_out.quantize_onto(quant, device)?;
        self.add_q.quantize_onto(quant, device)?;
        self.add_k.quantize_onto(quant, device)?;
        self.add_v.quantize_onto(quant, device)?;
        if let Some(p) = &mut self.to_add_out {
            p.quantize_onto(quant, device)?;
        }
        self.migrate_norms_to(device)
    }

    /// Move the dense per-head q/k RMSNorms (image + text streams) to `device`.
    fn migrate_norms_to(&mut self, device: &Device) -> Result<()> {
        for rn in [
            &mut self.norm_q,
            &mut self.norm_k,
            &mut self.norm_added_q,
            &mut self.norm_added_k,
        ]
        .into_iter()
        .flatten()
        {
            *rn = rms_norm_to(rn, RMS_EPS, device)?;
        }
        Ok(())
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

/// The **MMDiT-X** second attention (`attn2`): an image-token-ONLY self-attention (diffusers
/// `Attention(cross_attention_dim=None, added_kv_proj_dim=None)`) present on the dual-attention
/// blocks. Same per-head q/k RMSNorm and `to_out.0` projection as the joint attention's image side,
/// but it does NOT touch the text stream and runs over the image sequence alone (NO RoPE, like the
/// rest of SD3.5). Only constructed for `dual_attention_layers` blocks.
struct SelfAttention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    norm_q: Option<RmsNorm>,
    norm_k: Option<RmsNorm>,
    heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(cfg: &Sd3Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let hd = cfg.head_dim;
        let (norm_q, norm_k) = if cfg.qk_norm {
            (
                Some(candle_nn::rms_norm(hd, RMS_EPS, vb.pp("norm_q"))?),
                Some(candle_nn::rms_norm(hd, RMS_EPS, vb.pp("norm_k"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            to_q: QLinear::linear(inner, inner, vb.pp("to_q"))?,
            to_k: QLinear::linear(inner, inner, vb.pp("to_k"))?,
            to_v: QLinear::linear(inner, inner, vb.pp("to_v"))?,
            to_out: QLinear::linear(inner, inner, vb.pp("to_out").pp("0"))?,
            norm_q,
            norm_k,
            heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    /// Fold the image-only `attn2` projections (q/k/v/out) to `Q4_0`/`Q8_0` in place.
    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.to_q.quantize(quant)?;
        self.to_k.quantize(quant)?;
        self.to_v.quantize(quant)?;
        self.to_out.quantize(quant)?;
        Ok(())
    }

    /// CPU-stage path (sc-8504): quantize the `attn2` projections **onto `device`** + migrate q/k norms.
    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.to_q.quantize_onto(quant, device)?;
        self.to_k.quantize_onto(quant, device)?;
        self.to_v.quantize_onto(quant, device)?;
        self.to_out.quantize_onto(quant, device)?;
        for rn in [&mut self.norm_q, &mut self.norm_k].into_iter().flatten() {
            *rn = rms_norm_to(rn, RMS_EPS, device)?;
        }
        Ok(())
    }

    /// `norm_img2`: the modulated, normed image stream (from the `shift_msa2/scale_msa2` chunks).
    /// Returns the image-only attention output `[B, img_seq, inner]` (before the `gate_msa2` gate).
    fn forward(&self, norm_img2: &Tensor) -> Result<Tensor> {
        let (h, hd) = (self.heads, self.head_dim);
        let q = to_heads(&self.to_q.forward(norm_img2)?, h, hd, self.norm_q.as_ref())?;
        let k = to_heads(&self.to_k.forward(norm_img2)?, h, hd, self.norm_k.as_ref())?;
        let v = to_heads(&self.to_v.forward(norm_img2)?, h, hd, None)?;
        let o = attention(&q, &k, &v, hd)?; // [B, img_seq, inner]
        self.to_out.forward(&o)
    }
}

/// GELU feed-forward (diffusers `FeedForward` with `gelu` activation): `proj -> gelu -> out`.
struct FeedForward {
    proj: QLinear,
    out: QLinear,
}

impl FeedForward {
    fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        // diffusers FeedForward nests the input projection at `net.0.proj` and the output at `net.2`.
        Ok(Self {
            proj: QLinear::linear(in_dim, hidden, vb.pp("net").pp("0").pp("proj"))?,
            out: QLinear::linear(hidden, in_dim, vb.pp("net").pp("2"))?,
        })
    }

    /// Fold both projections to `Q4_0`/`Q8_0` in place (the MLP is the largest per-block footprint).
    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.proj.quantize(quant)?;
        self.out.quantize(quant)?;
        Ok(())
    }

    /// CPU-stage path (sc-8504): quantize both projections **onto `device`** (no dense GPU transient).
    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.proj.quantize_onto(quant, device)?;
        self.out.quantize_onto(quant, device)?;
        Ok(())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.proj.forward(x)?.gelu()?)
    }
}

/// One SD3.5 joint (double-stream) block. The image stream is always a full AdaLN-Zero block; the
/// text stream is full unless `context_pre_only` (the final block), where it only emits scale/shift
/// for its norm and has no attention output projection / ff_context.
///
/// **MMDiT-X dual-attention blocks** (the diffusers `dual_attention_layers`, SD3.5 Medium blocks
/// 0..=12) additionally carry [`attn2`](SelfAttention) — a second, image-only self-attention. Their
/// `norm1` is a 9-chunk `SD35AdaLayerNormZeroX` (the usual 6 + `shift_msa2/scale_msa2/gate_msa2`),
/// and the block adds `gate_msa2 · attn2(modulate(LN(img), shift2, scale2))` to the image stream
/// *before* the joint-attn + mlp residuals (diffusers adds it right after computing the norms).
struct JointBlock {
    norm1: AdaLayerNormZero,
    norm1_context: AdaLayerNormZero,
    attn: JointAttention,
    /// The MMDiT-X second (image-only) attention; `Some` only on `dual_attention_layers` blocks.
    attn2: Option<SelfAttention>,
    ff: FeedForward,
    ff_context: Option<FeedForward>,
    context_pre_only: bool,
}

impl JointBlock {
    fn new(cfg: &Sd3Config, context_pre_only: bool, dual: bool, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let ff_hidden = cfg.ff_hidden();
        // Image norm: 6 chunks normally; 9 on a dual (MMDiT-X) block — the extra 3 are
        // shift_msa2/scale_msa2/gate_msa2 for the `attn2` path (diffusers `SD35AdaLayerNormZeroX`).
        let norm1 = AdaLayerNormZero::new(inner, if dual { 9 } else { 6 }, vb.pp("norm1"))?;
        // Context norm: 6 chunks normally; 2 (scale, shift) when context_pre_only.
        let norm1_context = AdaLayerNormZero::new(
            inner,
            if context_pre_only { 2 } else { 6 },
            vb.pp("norm1_context"),
        )?;
        let attn = JointAttention::new(cfg, context_pre_only, vb.pp("attn"))?;
        // The second image-only attention lives at `attn2.*` on dual blocks only.
        let attn2 = if dual {
            Some(SelfAttention::new(cfg, vb.pp("attn2"))?)
        } else {
            None
        };
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
            attn2,
            ff,
            ff_context,
            context_pre_only,
        })
    }

    /// Fold the block's compute-heavy projections (joint attention, the GELU MLP, the dual `attn2`,
    /// and the text-stream `ff_context`) to `Q4_0`/`Q8_0`. The small AdaLN modulation linears
    /// (`norm1.linear` / `norm1_context.linear`) stay dense — their footprint is negligible and they
    /// drive the chaos-sensitive modulation chunks.
    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.attn.quantize(quant)?;
        if let Some(a2) = &mut self.attn2 {
            a2.quantize(quant)?;
        }
        self.ff.quantize(quant)?;
        if let Some(ffc) = &mut self.ff_context {
            ffc.quantize(quant)?;
        }
        Ok(())
    }

    /// CPU-stage path (sc-8504): quantize the block's compute-heavy projections **onto `device`** and
    /// migrate the block's dense-kept leaves (the AdaLN modulation linears) there too. After this the
    /// whole block resides on `device` with NO dense projection ever having touched the GPU.
    fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        self.attn.quantize_onto(quant, device)?;
        if let Some(a2) = &mut self.attn2 {
            a2.quantize_onto(quant, device)?;
        }
        self.ff.quantize_onto(quant, device)?;
        if let Some(ffc) = &mut self.ff_context {
            ffc.quantize_onto(quant, device)?;
        }
        // Dense-kept leaves: the chaos-sensitive AdaLN modulation linears.
        self.norm1.migrate_to(device)?;
        self.norm1_context.migrate_to(device)?;
        Ok(())
    }

    /// `(image, text, temb)` → updated `(image, text)`. For the final (`context_pre_only`) block the
    /// returned `text` is unchanged (only `image` matters downstream). On a dual (MMDiT-X) block the
    /// image-only `attn2` residual is added before the joint-attn residual.
    fn forward(&self, img: &Tensor, txt: &Tensor, temb: &Tensor) -> Result<(Tensor, Tensor)> {
        // Image AdaLN-Zero: 6 chunks (standard) or 9 (dual / MMDiT-X). The first 6 are always
        // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
        let im = self.norm1.forward(temb)?;
        let (shift_msa, scale_msa, gate_msa) = (&im[0], &im[1], &im[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&im[3], &im[4], &im[5]);
        let ln_img = layer_norm(img)?;
        let norm_img = modulate(&ln_img, scale_msa, shift_msa)?;

        // Context norm.
        let cm = self.norm1_context.forward(temb)?;
        let norm_txt = if self.context_pre_only {
            // 2-chunk scale/shift only.
            modulate(&layer_norm(txt)?, &cm[1], &cm[0])?
        } else {
            modulate(&layer_norm(txt)?, &cm[1], &cm[0])?
        };

        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt)?;

        // MMDiT-X dual attention: a second, image-only self-attention. diffusers modulates the SAME
        // LayerNorm output with the (shift_msa2, scale_msa2) chunks, runs `attn2`, gates by
        // `gate_msa2`, and adds it to the image stream *before* the joint-attn residual.
        let mut img = img.clone();
        if let Some(attn2) = &self.attn2 {
            let (shift_msa2, scale_msa2, gate_msa2) = (&im[6], &im[7], &im[8]);
            let norm_img2 = modulate(&ln_img, scale_msa2, shift_msa2)?;
            let img_attn2 = attn2.forward(&norm_img2)?;
            img = gated(&img, gate_msa2, &img_attn2)?;
        }

        // Image stream residual + ff.
        let mut img = gated(&img, gate_msa, &img_attn)?;
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

/// AdaLayerNormContinuous output head (the diffusers `norm_out`): `emb = linear(silu(temb))`, then
/// `scale, shift = emb.chunk(2)` — **scale is the first half, shift the second** — and the result is
/// `(1 + scale)·LN(x) + shift`. This chunk order is the one diffusers `AdaLayerNormContinuous` uses
/// (`scale, shift = torch.chunk(emb, 2)`); getting it backwards swaps every output token's scale and
/// shift and scrambles the predicted velocity into spatial noise (sc-7881, C6 — the epic-7841 AdaLN
/// bug magnet, caught against real weights). Pinned by [`tests`].
struct AdaLayerNormContinuous {
    linear: Linear,
}

impl AdaLayerNormContinuous {
    fn new(inner: usize, vb: VarBuilder) -> Result<Self> {
        // diffusers `norm_out.norm` is an affine-free LayerNorm, so there are no norm weights to
        // load — `forward` runs the parameterless `layer_norm` then applies the AdaLN scale/shift.
        let linear = candle_nn::linear(inner, 2 * inner, vb.pp("linear"))?;
        Ok(Self { linear })
    }

    /// CPU-stage migration (sc-8504): move the dense AdaLN-continuous linear to `device`.
    fn migrate_to(&mut self, device: &Device) -> Result<()> {
        self.linear = linear_to(&self.linear, device)?;
        Ok(())
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&temb.silu()?)?.unsqueeze(1)?; // [B,1,2*inner]
        let inner = emb.dim(D::Minus1)? / 2;
        // diffusers `AdaLayerNormContinuous`: `scale, shift = emb.chunk(2)` — scale is the FIRST
        // half, shift the second; result is `(1 + scale)·LN(x) + shift`.
        let scale = emb.narrow(D::Minus1, 0, inner)?;
        let shift = emb.narrow(D::Minus1, inner, inner)?;
        modulate(&layer_norm(x)?, &scale, &shift)
    }
}

/// The SD3.5 MMDiT transformer (Large by default; geometry from [`Sd3Config`]).
pub struct Sd3Transformer {
    pos_embed: PatchEmbed,
    time_text_embed: CombinedTimestepTextEmbed,
    context_embedder: QLinear,
    blocks: Vec<JointBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: QLinear,
    cfg: Sd3Config,
}

impl Sd3Transformer {
    pub fn new(cfg: &Sd3Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            // The LAST block is context_pre_only when the config flags it (SD3.5 default).
            let pre_only = cfg.context_pre_only_last && i == cfg.num_layers - 1;
            // MMDiT-X: the early `dual_attention_layers` blocks carry the image-only `attn2`.
            let dual = cfg.is_dual_block(i);
            blocks.push(JointBlock::new(
                cfg,
                pre_only,
                dual,
                vb.pp("transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            pos_embed: PatchEmbed::new(cfg, vb.pp("pos_embed"))?,
            time_text_embed: CombinedTimestepTextEmbed::new(cfg, vb.pp("time_text_embed"))?,
            context_embedder: QLinear::linear(
                cfg.joint_attention_dim,
                inner,
                vb.pp("context_embedder"),
            )?,
            blocks,
            norm_out: AdaLayerNormContinuous::new(inner, vb.pp("norm_out"))?,
            proj_out: QLinear::linear(inner, cfg.patch_dim(), vb.pp("proj_out"))?,
            cfg: cfg.clone(),
        })
    }

    /// Fold every compute-heavy MMDiT projection to `Q4_0`/`Q8_0` **in place** — uniform across the
    /// `context_embedder`, every joint block (attention + GELU MLP + dual `attn2` + `ff_context`),
    /// and the `proj_out` head. Call **after** the dense weights load. Uniform Q4 is the sc-7702
    /// dequant-on-forward design: the int8 `QMatMul` activation-quant path is avoided so coarse Q4
    /// stays coherent. The small AdaLN/timestep/patch-embed leaves stay dense (negligible footprint).
    /// Mirrors `LensTransformer::quantize` (sc-5117).
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.context_embedder.quantize(quant)?;
        for block in &mut self.blocks {
            block.quantize(quant)?;
        }
        self.proj_out.quantize(quant)?;
        Ok(())
    }

    /// **CPU-stage → quantize_onto (sc-8504, FLUX.2-dev pattern).** Build the dense MMDiT on a CPU
    /// VarBuilder, then call this with the GPU `device`: every compute-heavy projection is quantized
    /// *onto* the GPU (so the dense projection weight never lands on the GPU at all — only the small
    /// `Q4_0`/`Q8_0` blocks do), and the dense-kept leaves (the patch-embed conv + learned pos-embed
    /// table, the timestep/text embedders, every block's AdaLN modulation linears + per-head q/k
    /// norms, the AdaLN-continuous head) are migrated to the GPU alongside. Afterwards the whole
    /// transformer resides on `device`.
    ///
    /// Numerically identical to the in-place [`Self::quantize`] (the quantizer routes through the CPU
    /// either way, so the blocks are bit-for-bit the same); the only difference is the dense on-device
    /// transient — and thus the load-time peak — is gone. Mirrors `Flux2Transformer::quantize`
    /// (sc-7457). Idempotent per `QLinear`.
    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        // Quantized projections onto the GPU.
        self.context_embedder.quantize_onto(quant, device)?;
        for block in &mut self.blocks {
            block.quantize_onto(quant, device)?;
        }
        self.proj_out.quantize_onto(quant, device)?;
        // Dense-kept leaves migrated to the GPU.
        self.pos_embed.migrate_to(device)?;
        self.time_text_embed.migrate_to(device)?;
        self.norm_out.migrate_to(device)?;
        Ok(())
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
                                                               // Run the whole DiT in the image stream's (loaded weight) dtype so every joint matmul agrees
                                                               // — the context arrives in f32 (aggregator) and must be cast to match (sc-7881, C6).
        let mut txt = self
            .context_embedder
            .forward(&context.to_dtype(img.dtype())?)?; // [B, ctx_seq, inner]

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
            dual_attention_layers: Vec::new(),
        }
    }

    /// A tiny MMDiT-X (Medium-shaped) config: like [`tiny_cfg`] but with the FIRST block flagged as a
    /// dual-attention (`attn2`) block, so the dual path is exercised.
    fn tiny_dual_cfg() -> Sd3Config {
        Sd3Config {
            dual_attention_layers: vec![0],
            ..tiny_cfg()
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

    /// `AdaLayerNormContinuous` splits the linear output as `scale, shift = chunk(2)` (scale FIRST,
    /// shift second — diffusers order), then applies `(1+scale)·LN(x) + shift`. Drive it with a
    /// hand-built linear so the chunk order is observable: a constant LN(x)=0 input isolates the
    /// shift. Getting this backwards is the sc-7881 real-weight bug (it produced spatial-noise renders).
    #[test]
    fn adaln_continuous_chunk_order_is_scale_then_shift() {
        let dev = Device::Cpu;
        let inner = 4usize;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut ada = AdaLayerNormContinuous::new(inner, vb.pp("norm_out")).unwrap();
        // Replace the linear with a known one: a zero weight + a bias of [scale(4)=7, shift(4)=2] so
        // emb = bias = [scale..., shift...] regardless of temb.
        let w = Tensor::zeros((2 * inner, inner), DType::F32, &dev).unwrap();
        let b = Tensor::from_vec(vec![7f32, 7., 7., 7., 2., 2., 2., 2.], 2 * inner, &dev).unwrap();
        ada.linear = Linear::new(w, Some(b));
        let temb = Tensor::randn(0f32, 1f32, (1, inner), &dev).unwrap();
        // A single token row of all-1s normalizes to 0 (zero variance) → LN(x)=0, so
        // out = (1+scale)·0 + shift = shift. With scale FIRST, shift is the SECOND half = 2.
        let x = Tensor::ones((1, 1, inner), DType::F32, &dev).unwrap();
        let out = ada.forward(&x, &temb).unwrap();
        for v in out.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!(
                (v - 2.0).abs() < 1e-4,
                "shift must be the SECOND chunk (got {v}, expected 2)"
            );
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
        let block = JointBlock::new(&cfg, true, false, vb.pp("transformer_blocks").pp(0)).unwrap();
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

    // ---- MMDiT-X dual-attention (sc-7878) -------------------------------------------------------

    /// A dual (MMDiT-X) block's `norm1` is a 9-chunk `SD35AdaLayerNormZeroX`: the usual 6
    /// (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) PLUS shift_msa2, scale_msa2,
    /// gate_msa2 for the `attn2` path. A standard block's `norm1` is 6 chunks.
    #[test]
    fn dual_block_norm1_emits_nine_chunks() {
        let dev = Device::Cpu;
        let cfg = tiny_dual_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let dual = JointBlock::new(&cfg, false, true, vb.pp("dual")).unwrap();
        assert_eq!(dual.norm1.n_chunks, 9, "dual norm1 must emit 9 chunks");
        assert!(dual.attn2.is_some(), "dual block must carry attn2");

        let std = JointBlock::new(&cfg, false, false, vb.pp("std")).unwrap();
        assert_eq!(std.norm1.n_chunks, 6, "standard norm1 must emit 6 chunks");
        assert!(std.attn2.is_none(), "standard block must NOT carry attn2");

        // The chunks really are produced in the documented order/width.
        let temb = Tensor::randn(0f32, 1f32, (1, cfg.inner_dim), &dev).unwrap();
        let chunks = dual.norm1.forward(&temb).unwrap();
        assert_eq!(chunks.len(), 9);
        for c in &chunks {
            assert_eq!(c.dims(), &[1, 1, cfg.inner_dim]);
        }
    }

    /// **The dual `attn2` path is NOT a no-op.** Build one dual block, run its forward (which adds the
    /// `gate_msa2 · attn2(...)` residual), then DROP its `attn2` (set to `None`) and run the same
    /// inputs again. The only behavioural difference between the two runs is the `attn2` residual, so
    /// a differing output proves the dual path actually executes and contributes (a vacuous/zero
    /// attn2 would leave the output unchanged). The norm1 9-chunk linear is identical across both
    /// runs (the extra `shift/scale/gate_msa2` chunks are only consumed when `attn2` is present).
    #[test]
    fn dual_attention_path_changes_output() {
        let dev = Device::Cpu;
        let cfg = tiny_dual_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        // Dual block (9-chunk norm1 + attn2).
        let mut block =
            JointBlock::new(&cfg, false, true, vb.pp("transformer_blocks").pp(0)).unwrap();
        assert!(block.attn2.is_some());

        let img = Tensor::randn(0f32, 1f32, (1, 4, cfg.inner_dim), &dev).unwrap();
        let txt = Tensor::randn(0f32, 1f32, (1, 6, cfg.inner_dim), &dev).unwrap();
        let temb = Tensor::randn(0f32, 1f32, (1, cfg.inner_dim), &dev).unwrap();

        // With attn2 active.
        let (img_dual, _) = block.forward(&img, &txt, &temb).unwrap();
        // Drop attn2 and re-run with identical weights/inputs — the ONLY difference is the missing
        // image-only attention residual.
        block.attn2 = None;
        let (img_no_attn2, _) = block.forward(&img, &txt, &temb).unwrap();

        let diff = (img_dual - img_no_attn2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-5,
            "dual attn2 path must change the image stream (diff={diff}); it is a no-op?"
        );
    }

    /// A small MMDiT-X (Medium-shaped) transformer with dual blocks runs end-to-end and produces a
    /// velocity with the latent's shape and non-vacuous magnitude.
    #[test]
    fn tiny_mmdit_x_forward_produces_latent_shape() {
        let dev = Device::Cpu;
        // 3 blocks, the first two dual (MMDiT-X), the last standard — exercises both code paths and
        // the dual/standard boundary like Medium's 13-dual / 11-standard split.
        let cfg = Sd3Config {
            dual_attention_layers: vec![0, 1],
            ..tiny_cfg()
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let model = Sd3Transformer::new(&cfg, vb).unwrap();
        // The dual flag really propagated to the blocks.
        assert!(model.blocks[0].attn2.is_some());
        assert!(model.blocks[1].attn2.is_some());
        assert!(model.blocks[2].attn2.is_none());

        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 8, 8), &dev).unwrap();
        let ctx_seq = cfg.context_seq_len();
        let context =
            Tensor::randn(0f32, 1f32, (1, ctx_seq, cfg.joint_attention_dim), &dev).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &dev).unwrap();
        let t = Tensor::full(0.5f32, 1, &dev).unwrap();

        let v = model.forward(&latent, &context, &pooled, &t).unwrap();
        assert_eq!(v.dims(), latent.dims());
        let max = v
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max > 0.0, "MMDiT-X velocity is vacuously zero?");
    }

    // ---- Q4/Q8 quantization (sc-7879) -----------------------------------------------------------

    /// Cosine similarity over all elements (f64).
    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// Build a tiny MMDiT and its full forward inputs (shared by the quant tests). The inner dim is
    /// bumped to 32 so every quantized contraction is at least one Q4_0/Q8_0 block wide.
    fn quant_harness(cfg: &Sd3Config) -> (VarMap, Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 8, 8), &dev).unwrap();
        let ctx_seq = cfg.context_seq_len();
        let context =
            Tensor::randn(0f32, 1f32, (1, ctx_seq, cfg.joint_attention_dim), &dev).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &dev).unwrap();
        let t = Tensor::full(0.5f32, 1, &dev).unwrap();
        (vm, latent, context, pooled, t)
    }

    /// A config like [`tiny_cfg`] but with inner=32 (one full Q4_0/Q8_0 block per contraction row).
    fn quant_cfg() -> Sd3Config {
        Sd3Config {
            inner_dim: 32,
            num_heads: 2,
            head_dim: 16,
            ..tiny_cfg()
        }
    }

    /// **Q8 forward is near-lossless; Q4 forward stays coherent** vs the dense MMDiT (CPU, random
    /// weights). Builds one transformer, captures the dense velocity, quantizes a *copy* of the same
    /// weights, and compares — the full-model analog of the per-linear `quant.rs` round-trip. Asserts
    /// finite output and a bounded cosine drop (not bit-equality).
    fn quant_forward_parity(quant: Quant, min_cos: f32) {
        let cfg = quant_cfg();
        let (vm, latent, context, pooled, t) = quant_harness(&cfg);
        let dev = Device::Cpu;
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        // Dense reference.
        let dense = Sd3Transformer::new(&cfg, vb.clone()).unwrap();
        let v_dense = dense.forward(&latent, &context, &pooled, &t).unwrap();

        // Quantized model over the SAME weights (re-read from the varmap), folded in place.
        let mut q = Sd3Transformer::new(&cfg, vb).unwrap();
        q.quantize(quant).unwrap();
        let v_q = q.forward(&latent, &context, &pooled, &t).unwrap();

        assert_eq!(v_q.dims(), v_dense.dims());
        for x in v_q.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!(
                x.is_finite(),
                "{quant:?} MMDiT produced a non-finite velocity"
            );
        }
        let cos = cosine(&v_dense, &v_q);
        assert!(
            cos > min_cos,
            "{quant:?} MMDiT forward cosine {cos:.5} ≤ {min_cos} vs dense"
        );
        // Non-vacuous: the quantized velocity is not all-zero.
        let max = v_q
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max > 0.0, "{quant:?} velocity vacuously zero");
    }

    #[test]
    fn q8_mmdit_forward_is_near_lossless() {
        quant_forward_parity(Quant::Q8, 0.99);
    }

    #[test]
    fn q4_mmdit_forward_stays_coherent() {
        quant_forward_parity(Quant::Q4, 0.85);
    }

    // ---- CPU-stage → quantize_onto (sc-8504) ----------------------------------------------------

    /// **CPU-staged quantization is numerically identical to the in-place path.** Build two MMDiTs
    /// from the SAME varmap weights; quantize one in place (`quantize`) and the other via the
    /// CPU-stage entry point (`quantize_onto`, target = CPU here). Their forwards must be **bit-exact**
    /// — the load-time-peak optimization must not perturb a single bit (the quantizer routes the
    /// source through the CPU in both paths, so the `Q4_0`/`Q8_0` blocks are identical). Run for both
    /// Q4 and Q8 and across the standard + MMDiT-X (dual) configs.
    fn cpu_stage_matches_in_place(quant: Quant, cfg: &Sd3Config) {
        let dev = Device::Cpu;
        let (vm, latent, context, pooled, t) = quant_harness(cfg);
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        // In-place quantize (the original on-device path; here device == CPU).
        let mut in_place = Sd3Transformer::new(cfg, vb.clone()).unwrap();
        in_place.quantize(quant).unwrap();
        let v_in_place = in_place.forward(&latent, &context, &pooled, &t).unwrap();

        // CPU-stage quantize_onto the SAME device (the dense weight round-trips through the CPU either
        // way, so the resulting blocks must be bit-identical to the in-place fold).
        let mut staged = Sd3Transformer::new(cfg, vb).unwrap();
        staged.quantize_onto(quant, &dev).unwrap();
        let v_staged = staged.forward(&latent, &context, &pooled, &t).unwrap();

        // The staged projections really transitioned to the Quantized arm.
        assert!(matches!(
            staged.blocks[0].attn.to_q,
            crate::quant::QLinear::Quantized { .. }
        ));

        let a = v_in_place.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = v_staged.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{quant:?} CPU-stage forward differs from in-place ({x} vs {y}) — not bit-exact"
            );
        }
    }

    #[test]
    fn cpu_stage_q4_bit_exact_vs_in_place() {
        cpu_stage_matches_in_place(Quant::Q4, &quant_cfg());
    }

    #[test]
    fn cpu_stage_q8_bit_exact_vs_in_place() {
        cpu_stage_matches_in_place(Quant::Q8, &quant_cfg());
    }

    /// The CPU-stage path also covers the MMDiT-X (dual-attention) config bit-exactly — its `attn2`
    /// projections + the 9-chunk AdaLN leaf migrate identically.
    #[test]
    fn cpu_stage_dual_bit_exact_vs_in_place() {
        let cfg = Sd3Config {
            dual_attention_layers: vec![0, 1],
            ..quant_cfg()
        };
        cpu_stage_matches_in_place(Quant::Q4, &cfg);
    }

    /// `quantize_onto` is idempotent (a re-run is a no-op per `QLinear`) and the staged model still
    /// forwards finite output.
    #[test]
    fn cpu_stage_is_idempotent_and_forwards() {
        let cfg = quant_cfg();
        let (vm, latent, context, pooled, t) = quant_harness(&cfg);
        let dev = Device::Cpu;
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut model = Sd3Transformer::new(&cfg, vb).unwrap();
        model.quantize_onto(Quant::Q8, &dev).unwrap();
        model.quantize_onto(Quant::Q8, &dev).unwrap(); // no-op, must not panic
        let v = model.forward(&latent, &context, &pooled, &t).unwrap();
        for x in v.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!(x.is_finite());
        }
    }

    /// **CUDA CPU-stage smoke (sc-8504).** Build the dense MMDiT on a **CPU** VarBuilder, then
    /// `quantize_onto` the GPU. Assert (a) the quantized leaves landed on the CUDA device with the
    /// expected `Q4_0`/`Q8_0` block dtype, (b) the dense-kept leaves (the patch-embed conv, the
    /// pos-embed table, an AdaLN modulation linear, a per-head q/k norm) are on the GPU too, and (c)
    /// the forward runs finite + non-zero on the GPU. This is the staging analog of the in-place
    /// `cuda_quant_forward_smoke`; it proves the dense projection never had to land on the GPU.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cpu_stage_quant_smoke() {
        use candle_gen::candle_core::quantized::GgmlDType;

        let gpu = Device::new_cuda(0).expect("CUDA device 0");
        let cpu = Device::Cpu;
        let cfg = quant_cfg();

        for quant in [Quant::Q4, Quant::Q8] {
            // Dense build on CPU (system RAM) — nothing on the GPU yet.
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F32, &cpu);
            let mut model = Sd3Transformer::new(&cfg, vb).unwrap();
            model.quantize_onto(quant, &gpu).unwrap();

            // (a) A quantized projection is on the GPU at the expected block dtype.
            match &model.blocks[0].attn.to_q {
                crate::quant::QLinear::Quantized { weight, .. } => {
                    assert!(weight.device().is_cuda(), "quantized leaf not on CUDA");
                    let expected = match quant {
                        Quant::Q4 => GgmlDType::Q4_0,
                        Quant::Q8 => GgmlDType::Q8_0,
                    };
                    assert_eq!(weight.dtype(), expected, "wrong GGUF block dtype");
                }
                _ => panic!("attn.to_q did not quantize"),
            }

            // (b) Dense-kept leaves migrated to the GPU.
            assert!(model.pos_embed.proj.weight().device().is_cuda());
            assert!(model.pos_embed.pos_embed.device().is_cuda());
            assert!(model.blocks[0].norm1.linear.weight().device().is_cuda());
            assert!(model.norm_out.linear.weight().device().is_cuda());
            if let Some(rn) = &model.blocks[0].attn.norm_q {
                assert!(rn.weight().device().is_cuda());
            }

            // (c) Forward runs finite + non-zero on the GPU.
            let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 8, 8), &gpu).unwrap();
            let ctx_seq = cfg.context_seq_len();
            let context =
                Tensor::randn(0f32, 1f32, (1, ctx_seq, cfg.joint_attention_dim), &gpu).unwrap();
            let pooled = Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &gpu).unwrap();
            let t = Tensor::full(0.5f32, 1, &gpu).unwrap();
            let v = model.forward(&latent, &context, &pooled, &t).unwrap();
            let vals = v
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(vals.iter().all(|x| x.is_finite()), "{quant:?} non-finite");
            assert!(
                vals.iter().fold(0f32, |m, &x| m.max(x.abs())) > 0.0,
                "{quant:?} all-zero (dequant no-op?)"
            );
        }
    }

    /// `quantize` transitions the model's projections to the `Quantized` arm (idempotently): a second
    /// pass is a no-op, and the forward still runs finite afterward.
    #[test]
    fn quantize_is_idempotent_and_still_forwards() {
        let cfg = quant_cfg();
        let (vm, latent, context, pooled, t) = quant_harness(&cfg);
        let dev = Device::Cpu;
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut model = Sd3Transformer::new(&cfg, vb).unwrap();
        model.quantize(Quant::Q4).unwrap();
        model.quantize(Quant::Q4).unwrap(); // no-op, must not panic
                                            // The image q-projection of block 0 is now quantized.
        assert!(matches!(
            model.blocks[0].attn.to_q,
            crate::quant::QLinear::Quantized { .. }
        ));
        let v = model.forward(&latent, &context, &pooled, &t).unwrap();
        for x in v.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!(x.is_finite());
        }
    }
}
