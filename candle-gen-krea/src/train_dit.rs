//! The **trainable** Krea 2 single-stream DiT (sc-7577) — the candle twin of `mlx-gen-krea`'s
//! training DiT, vendored alongside the inference [`crate::transformer::Krea2Transformer`] for the same
//! reason the Z-Image trainer vendors its DiT: candle's fused `softmax_last_dim` / `ops::rms_norm`
//! kernels are `CustomOp`s with **no backward** (they silently yield `None` grads), so any module the
//! gradient must flow *through* to reach a LoRA factor has to use composable ops instead.
//!
//! ## What is — and is NOT — re-implemented
//!
//! In the backward, the chain runs `loss → final_layer → block_{N-1} → … → block_0`. Every module on
//! that path must be differentiable, so the **single-stream blocks** ([`TrainBlock`]) and the
//! **`final_layer`** are re-implemented here with composable softmax ([`candle_nn::ops::softmax`]) and a
//! composable `+1` RMSNorm ([`rms_scale_diff`]). The LoRA seam lives in each block's attention
//! `to_q/to_k/to_v/to_out.0` projections, which become [`LoraLinear`]s; everything else in a block
//! (the `to_gate` projection, the SwiGLU FFN, the modulation table) is the **frozen** base.
//!
//! The **pre-main** front-end (`img_in`, the timestep MLP + `time_mod_proj`, `txt_in`, and the
//! `text_fusion` aggregator) is *upstream* of every adapter: it only produces the joint sequence that
//! enters block 0, and it holds no trainable factor. candle's `sorted_nodes` prunes any backward branch
//! that leads to no `Var`, so those modules are never differentiated — which means the trainer can
//! **reuse the inference crate's fused-op front-end structs verbatim** ([`TextFusionTransformer`],
//! [`RmsScale`], the `Linear`s), guaranteeing train/infer parity for the conditioning at zero cost.
//!
//! ## Velocity sign
//!
//! Unlike the Z-Image trainer (which negates the DiT output to match its inference pipeline's
//! `noise_pred.neg()`), Krea's inference pipeline consumes the **raw** velocity directly
//! (`x + v·Δσ`, [`crate::pipeline`]). So [`KreaTrainDit::forward`] returns the raw velocity and the
//! trainer regresses it toward `noise − x0` with no negation — the Lens convention.
//!
//! ## Gradient checkpointing
//!
//! Because the default training surface is the 28 single-stream blocks' attention, **all** adapters
//! live in the checkpointed main stack — there is no retained-pre-main adapter to stitch back (the
//! Z-Image complication). So the checkpointed path is the plain
//! [`checkpointed_backward`](candle_gen::train::gradient_checkpoint::checkpointed_backward): run
//! [`forward_pre_main`](KreaTrainDit::forward_pre_main) once (frozen, detached at the boundary),
//! checkpoint the [`main_layer_segments`](KreaTrainDit::main_layer_segments), and recompute the loss in
//! the final segment via [`velocity_out`](KreaTrainDit::velocity_out).

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::ops::{sigmoid, softmax};
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::train::gradient_checkpoint::Segment;
use candle_gen::train::lora::{LoraHost, LoraLinear};

use crate::config::Krea2Config;
use crate::loader::{linear, rms_scale_weight, Weights};
use crate::transformer::block::{RmsScale, SwiGlu, TextFusionTransformer};
use crate::transformer::rope::{apply_interleaved_rope, RopeTables};
use crate::transformer::{patchify, temb, unpatchify};

/// Default LoRA target suffixes — the single-stream blocks' attention projections (`to_out.0` is the
/// first element of diffusers' `to_out` `ModuleList`, so the suffix literally carries the `.0`). With
/// 28 blocks this is the **112-target** default surface the MLX trainer uses (sc-7577); `to_gate` and
/// the SwiGLU FFN are intentionally not in the default set.
pub const KREA_ATTN_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// Composable `+1` RMSNorm — the differentiable twin of [`crate::loader::rms_scale`] (which calls the
/// no-backward fused `ops::rms_norm`). `weight` is the pre-folded `scale + 1` f32 tensor; the reduction
/// runs in f32 (the reference upcasts) and the result is cast back to `x`'s dtype, so it is numerically
/// the same op the inference path applies — just one the autograd can traverse.
fn rms_scale_diff(x: &Tensor, weight_f32: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let hidden = xf.dim(D::Minus1)? as f64;
    let norm = (xf.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
    let y = xf.broadcast_div(&(norm + eps)?.sqrt()?)?;
    y.broadcast_mul(weight_f32)?.to_dtype(dt)
}

/// Repeat each kv head `groups` times consecutively (`[b,s,hkv,hd] → [b,s,hkv·groups,hd]`) — the
/// composable `repeat_interleave` matching the inference block's `repeat_kv` (reference `enable_gqa`).
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, s, hkv, hd) = x.dims4()?;
    x.unsqueeze(3)?
        .expand((b, s, hkv, groups, hd))?
        .contiguous()?
        .reshape((b, s, hkv * groups, hd))
}

/// Bidirectional, unmasked scaled-dot-product attention with a **composable** softmax (the inference
/// `sdpa` uses the no-backward fused `softmax_last_dim`). `q`/`k`/`v`: `[b, h, s, hd]`.
fn sdpa_diff(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    let probs = softmax(&scores, D::Minus1)?;
    probs.matmul(&v)
}

/// Build a frozen base `Linear` (no bias) from the mmap'd `Weights` and wrap it as a trainable
/// [`LoraLinear`], reading `in`/`out` from the on-disk shape (`[out, in]`) and recording `path` as the
/// PEFT module path the harness matches against.
fn lora_proj(w: &Weights, path: &str) -> Result<LoraLinear> {
    let base = linear(w, path, false)?;
    let (out_f, in_f) = base.weight().dims2()?;
    Ok(LoraLinear::from_linear(base, in_f, out_f, path.to_string()))
}

/// Sigmoid-gated GQA attention with the four attention projections as adaptable [`LoraLinear`]s and a
/// frozen `to_gate` / per-head `+1` RMSNorm — the trainable twin of [`crate::transformer::block`]'s
/// `GatedAttention`.
struct TrainAttention {
    q: LoraLinear,
    k: LoraLinear,
    v: LoraLinear,
    gate: Linear,
    o: LoraLinear,
    norm_q: Tensor, // f32, scale + 1
    norm_k: Tensor, // f32, scale + 1
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
    scale: f64,
}

impl TrainAttention {
    fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            q: lora_proj(w, &format!("{prefix}.to_q"))?,
            k: lora_proj(w, &format!("{prefix}.to_k"))?,
            v: lora_proj(w, &format!("{prefix}.to_v"))?,
            gate: linear(w, &format!("{prefix}.to_gate"), false)?,
            o: lora_proj(w, &format!("{prefix}.to_out.0"))?,
            norm_q: rms_scale_weight(w, &format!("{prefix}.norm_q.weight"))?,
            norm_k: rms_scale_weight(w, &format!("{prefix}.norm_k.weight"))?,
            heads,
            kv_heads,
            head_dim,
            eps,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Visit the four adaptable projections in install order.
    fn visit(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&mut self.q)?;
        f(&mut self.k)?;
        f(&mut self.v)?;
        f(&mut self.o)?;
        Ok(())
    }

    fn forward(&self, x: &Tensor, rope: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);

        let q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;
        let gate = self.gate.forward(x)?;

        let q = rms_scale_diff(&q, &self.norm_q, self.eps)?;
        let k = rms_scale_diff(&k, &self.norm_k, self.eps)?;
        let (q, k) = match rope {
            Some((cos, sin)) => (
                apply_interleaved_rope(&q, cos, sin)?,
                apply_interleaved_rope(&k, cos, sin)?,
            ),
            None => (q, k),
        };

        let groups = nh / nkv;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = sdpa_diff(&q, &k, &v, self.scale)?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;

        let gated = (o * sigmoid(&gate)?)?;
        self.o.forward(&gated)
    }
}

/// One trainable single-stream block (`DoubleSharedModulation`) — the differentiable twin of
/// [`crate::transformer::block`]'s `SingleStreamBlock`. The SwiGLU FFN is the inference crate's
/// (composable, frozen) [`SwiGlu`]; only the norms are swapped for the composable [`rms_scale_diff`].
struct TrainBlock {
    scale_shift_table: Tensor, // [1, 1, 6·hidden]
    prenorm: Tensor,           // f32, scale + 1
    postnorm: Tensor,          // f32, scale + 1
    attn: TrainAttention,
    mlp: SwiGlu,
    eps: f64,
}

impl TrainBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        eps: f64,
    ) -> Result<Self> {
        let sst = w
            .get(&format!("{prefix}.scale_shift_table"))?
            .reshape((1, 1, 6 * hidden))?;
        Ok(Self {
            scale_shift_table: sst,
            prenorm: rms_scale_weight(w, &format!("{prefix}.norm1.weight"))?,
            postnorm: rms_scale_weight(w, &format!("{prefix}.norm2.weight"))?,
            attn: TrainAttention::load(
                w,
                &format!("{prefix}.attn"),
                heads,
                kv_heads,
                head_dim,
                eps,
            )?,
            mlp: SwiGlu::load(w, &format!("{prefix}.ff"))?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor, tvec: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let m = tvec.broadcast_add(&self.scale_shift_table)?; // [b, 1, 6·hidden]
        let chunks = m.chunk(6, D::Minus1)?;
        let (prescale, preshift, pregate) = (&chunks[0], &chunks[1], &chunks[2]);
        let (postscale, postshift, postgate) = (&chunks[3], &chunks[4], &chunks[5]);

        let pre = rms_scale_diff(x, &self.prenorm, self.eps)?
            .broadcast_mul(&(prescale + 1.0)?)?
            .broadcast_add(preshift)?;
        let attn = self.attn.forward(&pre, Some((cos, sin)))?;
        let x = (x + attn.broadcast_mul(pregate)?)?;

        let post = rms_scale_diff(&x, &self.postnorm, self.eps)?
            .broadcast_mul(&(postscale + 1.0)?)?
            .broadcast_add(postshift)?;
        let mlp = self.mlp.forward(&post)?;
        &x + mlp.broadcast_mul(postgate)?
    }
}

/// The constants the single-stream stack + final layer need, computed once in
/// [`KreaTrainDit::forward_pre_main`] and threaded (cloned) into the per-block segments / the loss
/// segment. None of these carry a trainable factor, so they are the detached boundary of the
/// checkpointed backward.
pub struct MainCtx {
    tvec: Tensor, // [b, 1, 6·hidden] shared modulation
    rcos: Tensor, // joint RoPE cos table
    rsin: Tensor, // joint RoPE sin table
    t: Tensor,    // [b, 1, hidden] for the final SimpleModulation
    cap_len: usize,
    img_len: usize,
    ht: usize,
    wt: usize,
    latent_ch: usize,
    patch: usize,
}

/// The trainable Krea 2 single-stream DiT. Built from the same mmap'd `transformer/` `Weights` the
/// inference path loads — the frozen base is shared; only the attention projections grow a `Var`-backed
/// LoRA residual (installed by [`build_lora_targets`](candle_gen::train::lora::build_lora_targets)).
pub struct KreaTrainDit {
    cfg: Krea2Config,
    device: Device,
    dtype: DType,
    // --- pre-main front-end (frozen, fused-op, upstream of every adapter → reused verbatim) ---
    img_in: Linear,
    time_embed_l1: Linear,
    time_embed_l2: Linear,
    time_mod_proj: Linear,
    txt_in_norm: RmsScale,
    txt_in_l1: Linear,
    txt_in_l2: Linear,
    text_fusion: TextFusionTransformer,
    // --- trainable single-stream stack ---
    blocks: Vec<TrainBlock>,
    // --- final layer (composable; on the backward path to every adapter) ---
    final_norm: Tensor, // f32, scale + 1
    final_linear: Linear,
    final_sstable: Tensor, // [1, 2, hidden]
}

impl KreaTrainDit {
    /// Build from a loaded `transformer/` weight set at `w`'s compute dtype.
    pub fn load(w: &Weights, cfg: &Krea2Config) -> Result<Self> {
        let (heads, kv, hd, eps) = (
            cfg.num_attention_heads,
            cfg.num_kv_heads,
            cfg.attention_head_dim,
            cfg.norm_eps,
        );
        let (theads, tkv) = (cfg.text_num_attention_heads, cfg.text_num_kv_heads);
        let hidden = cfg.hidden_size;

        let final_sstable = w
            .get("final_layer.scale_shift_table")?
            .reshape((1, 2, hidden))?;

        Ok(Self {
            cfg: cfg.clone(),
            device: w.device().clone(),
            dtype: w.dtype(),
            img_in: linear(w, "img_in", true)?,
            time_embed_l1: linear(w, "time_embed.linear_1", true)?,
            time_embed_l2: linear(w, "time_embed.linear_2", true)?,
            time_mod_proj: linear(w, "time_mod_proj", true)?,
            txt_in_norm: RmsScale::load(w, "txt_in.norm.weight", eps)?,
            txt_in_l1: linear(w, "txt_in.linear_1", true)?,
            txt_in_l2: linear(w, "txt_in.linear_2", true)?,
            text_fusion: TextFusionTransformer::load(
                w,
                cfg.num_layerwise_text_blocks,
                cfg.num_refiner_text_blocks,
                theads,
                tkv,
                hd,
                eps,
            )?,
            blocks: (0..cfg.num_layers)
                .map(|i| {
                    TrainBlock::load(
                        w,
                        &format!("transformer_blocks.{i}"),
                        heads,
                        kv,
                        hd,
                        hidden,
                        eps,
                    )
                })
                .collect::<Result<_>>()?,
            final_norm: rms_scale_weight(w, "final_layer.norm.weight")?,
            final_linear: linear(w, "final_layer.linear", true)?,
            final_sstable,
        })
    }

    /// Run the frozen front-end: patch-embed the latent, build the shared modulation, aggregate +
    /// project the text conditioning, and fuse to the joint `[ctx; img]` sequence. Returns that joint
    /// sequence (the differentiable boundary entering block 0) plus the [`MainCtx`] the stack/final
    /// need. `latent`: `[b, 16, H, W]`; `timestep`: `[b]` (the raw flow σ); `context`:
    /// `[b, n_tok, num_text_layers, text_hidden]` (the stacked Qwen3-VL select layers).
    pub fn forward_pre_main(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
    ) -> Result<(Tensor, MainCtx)> {
        let cfg = &self.cfg;
        let p = cfg.patch_size;
        let dt = self.dtype;
        let (_, _, h, w) = latent.dims4()?;
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;
        let latent_ch = cfg.in_channels / (p * p);
        let cap_len = context.dim(1)?;
        let context = context.to_dtype(dt)?;

        let img = self.img_in.forward(&patchify(&latent.to_dtype(dt)?, p)?)?;

        let t_sin = temb(timestep, cfg.timestep_embed_dim, &self.device)?.to_dtype(dt)?;
        let t = self
            .time_embed_l2
            .forward(&self.time_embed_l1.forward(&t_sin)?.gelu()?)?;
        let tvec = self.time_mod_proj.forward(&t.gelu()?)?;

        let ctx = self.text_fusion.forward(&context)?;
        let ctx = self.txt_in_norm.forward(&ctx)?;
        let ctx = self
            .txt_in_l2
            .forward(&self.txt_in_l1.forward(&ctx)?.gelu()?)?;

        let combined = Tensor::cat(&[&ctx, &img], 1)?;
        let rope = RopeTables::build_t2i(
            cap_len,
            ht,
            wt,
            cfg.axes_dims_rope,
            cfg.rope_theta as f64,
            &self.device,
        )?;
        let (rcos, rsin) = rope.joint();
        Ok((
            combined,
            MainCtx {
                tvec,
                rcos,
                rsin,
                t,
                cap_len,
                img_len,
                ht,
                wt,
                latent_ch,
                patch: p,
            },
        ))
    }

    /// One [`Segment`] per single-stream block (for the checkpointed backward): each recomputes its
    /// block forward over the incoming joint sequence, threading the (constant) shared modulation +
    /// RoPE tables borrowed from `ctx`. The `Segment` lifetime ties to both `self` (the block refs) and
    /// `ctx`, so the trainer can push a `ctx`-borrowing loss segment after these.
    pub fn main_layer_segments<'a>(&'a self, ctx: &'a MainCtx) -> Vec<Segment<'a>> {
        self.blocks
            .iter()
            .map(|blk| -> Segment<'a> {
                Box::new(move |st: &[Tensor]| {
                    Ok(vec![blk.forward(&st[0], &ctx.tvec, &ctx.rcos, &ctx.rsin)?])
                })
            })
            .collect()
    }

    /// The continuous-AdaLN output head: `LastLayer` (SimpleModulation on `t`) over the joint sequence,
    /// then slice the image tokens and unpatchify back to a velocity `[b, 16, H, W]`. Composable (it is
    /// on the backward path to every block adapter).
    pub fn velocity_out(&self, combined: &Tensor, ctx: &MainCtx) -> Result<Tensor> {
        let m = ctx.t.broadcast_add(&self.final_sstable)?; // [b, 2, hidden]
        let scale = m.narrow(1, 0, 1)?;
        let shift = m.narrow(1, 1, 1)?;
        let normed = rms_scale_diff(combined, &self.final_norm, self.cfg.norm_eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        let out = self.final_linear.forward(&normed)?; // [b, cap+img_len, in_channels]
        let img_out = out.narrow(1, ctx.cap_len, ctx.img_len)?;
        unpatchify(&img_out, ctx.ht, ctx.wt, ctx.patch, ctx.latent_ch)
    }

    /// Dense (retained) velocity prediction — the same surface as the inference
    /// [`Krea2Transformer::forward`](crate::transformer::Krea2Transformer::forward), built from the
    /// composable trainable blocks. Returns the **raw** velocity `[b, 16, H, W]` (no negation).
    pub fn forward(&self, latent: &Tensor, timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (mut combined, ctx) = self.forward_pre_main(latent, timestep, context)?;
        for blk in &self.blocks {
            combined = blk.forward(&combined, &ctx.tvec, &ctx.rcos, &ctx.rsin)?;
        }
        self.velocity_out(&combined, &ctx)
    }
}

impl LoraHost for KreaTrainDit {
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for blk in &mut self.blocks {
            blk.attn.visit(f)?;
        }
        Ok(())
    }
}
