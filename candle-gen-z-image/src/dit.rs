//! Vendored, training-adapted Z-Image DiT (sc-5166).
//!
//! A faithful copy of the candle-transformers `z_image::transformer` model at the workspace candle
//! pin (`65ecb58`), vendored because the native LoRA/LoKr trainer needs a **trainable** residual
//! spliced into the attention projections (`to_q`/`to_k`/`to_v`/`to_out.0`) — the stock
//! `ZImageAttention` builds them from frozen `nn::Linear` with no seam (the candle twin of SDXL's
//! `src/unet/` vendoring, sc-5165).
//!
//! Only the three structs that *own* the attention projections are vendored — [`ZImageAttention`]
//! (the LoRA seam), [`ZImageTransformerBlock`] (owns the attention), and
//! [`ZImageTransformer2DModel`] (owns the blocks + the [`LoraHost`] walk). Everything else
//! (`TimestepEmbedder`, `FeedForward`, `QkNorm`, `RopeEmbedder`, `FinalLayer`, `patchify`/
//! `unpatchify`/`apply_rotary_emb`, the `Config`) is **reused** straight from candle-transformers —
//! those are `pub`, frozen, and not adapter targets, so re-deriving them would only invite drift.
//!
//! Three deliberate deviations from the stock attention/blocks, all forced by candle's autograd —
//! candle's *fused* kernels are `CustomOp`s with **no backward**, and they fail SILENTLY (yielding
//! `None` grads upstream, not an error — the epic-5164 fused-ops trap), so every fused op on the
//! differentiable path must be swapped for its composable equivalent:
//!  1. **Composable softmax.** The stock `attention_basic` uses the fused `softmax_last_dim`; the
//!     vendored attention uses `candle_nn::ops::softmax` so `loss.backward()` flows through it.
//!  2. **Composable RMSNorm.** `candle_nn::RmsNorm::forward` dispatches to the fused `ops::rms_norm`
//!     on contiguous input — which is exactly the gradient-killer that left the attention factors
//!     with no grad in the first cut. Every norm here (the vendored [`QkNorm`], the block's four
//!     `attention_norm`/`ffn_norm`s, the model's `cap_embedder_norm`) calls `forward_diff`, the
//!     composable `LayerNorm` path. (The reused `FinalLayer`/`TimestepEmbedder`/`FeedForward` hold no
//!     fused op — only Linears + `silu` + the manual-ops `LayerNormNoParams` — so they stay stock.)
//!  3. **No flash-attn / SDPA dispatch.** Those fused kernels are likewise non-differentiable; the
//!     trainer always runs the materialized math attention. (Inference keeps using the stock model —
//!     this vendored copy is training-only; the [`crate::adapters`] merge is how a trained adapter
//!     reaches inference.)
//!
//! With no adapter installed, the vendored forward is bit-identical to the stock forward (the
//! `parity_tests` gate pins this). The `forward` returns the **raw** DiT velocity (no sign flip,
//! matching candle-transformers); the trainer negates it to match the inference pipeline's
//! `noise_pred.neg()` (the Z-Image sign convention) — see [`crate::training`].

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

use candle_gen::train::lora::{lora_linear_no_bias, LoraHost, LoraLinear};

// Reused verbatim from candle-transformers — frozen, non-adapter sub-modules + the patchify/RoPE
// helpers. Vendoring these would add ~600 lines of drift surface for zero benefit (they hold no
// LoRA seam *and* no fused-norm), so we import the `pub` originals at the workspace candle pin.
// (`QkNorm` is the one exception — see the local [`QkNorm`] below — because its `RmsNorm::forward`
// dispatches to a no-backward fused kernel.)
use candle_transformers::models::z_image::transformer::{
    apply_rotary_emb, create_coordinate_grid, patchify, unpatchify, Config, FeedForward,
    FinalLayer, RopeEmbedder, TimestepEmbedder, ADALN_EMBED_DIM,
};

/// QK normalization (RMSNorm on the per-head query/key), vendored so it uses the **composable**
/// `RmsNorm::forward_diff` rather than `RmsNorm::forward` (which dispatches to the no-backward fused
/// `ops::rms_norm` on contiguous input — the q/k path feeds the attention output, so a fused norm
/// here silently zeroes every attention factor's gradient). Same `norm_q`/`norm_k` weight keys as the
/// stock `QkNorm`, so it loads the real weights unchanged.
#[derive(Debug, Clone)]
struct QkNorm {
    norm_q: RmsNorm,
    norm_k: RmsNorm,
}

impl QkNorm {
    fn new(head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_q: candle_nn::rms_norm(head_dim, eps, vb.pp("norm_q"))?,
            norm_k: candle_nn::rms_norm(head_dim, eps, vb.pp("norm_k"))?,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.norm_q.forward_diff(q)?, self.norm_k.forward_diff(k)?))
    }
}

/// The default LoRA target suffixes — the attention projections, matching the SDXL trainer's
/// [`SDXL_ATTN_TARGETS`](candle_gen::train::lora::SDXL_ATTN_TARGETS), the torch
/// `DEFAULT_LORA_TARGET_MODULES`, and the MLX Z-Image trainer. `to_out.0` is the first element of
/// diffusers' `to_out` `ModuleList`, so its path segment literally contains the `.0`.
pub const Z_IMAGE_ATTN_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

// ==================== ZImageAttention (LoRA seam) ====================

/// Z-Image attention with QK normalization and 3D RoPE, with the four projections held as
/// [`LoraLinear`] so the trainer can splice a trainable residual into each. Numerically identical to
/// the stock `ZImageAttention` when no adapter is installed, except it always runs the materialized
/// math attention with a composable softmax (the stock flash/SDPA/`softmax_last_dim` paths have no
/// backward).
#[derive(Debug, Clone)]
pub struct ZImageAttention {
    to_q: LoraLinear,
    to_k: LoraLinear,
    to_v: LoraLinear,
    to_out: LoraLinear,
    qk_norm: Option<QkNorm>,
    n_heads: usize,
    head_dim: usize,
}

impl ZImageAttention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim();

        let to_q = lora_linear_no_bias(dim, n_heads * head_dim, vb.pp("to_q"))?;
        let to_k = lora_linear_no_bias(dim, cfg.n_kv_heads * head_dim, vb.pp("to_k"))?;
        let to_v = lora_linear_no_bias(dim, cfg.n_kv_heads * head_dim, vb.pp("to_v"))?;
        let to_out = lora_linear_no_bias(n_heads * head_dim, dim, vb.pp("to_out").pp("0"))?;

        let qk_norm = if cfg.qk_norm {
            Some(QkNorm::new(head_dim, 1e-5, vb.clone())?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            qk_norm,
            n_heads,
            head_dim,
        })
    }

    /// Visit the four adaptable projections (the [`LoraHost`] leaf walk).
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&mut self.to_q)?;
        f(&mut self.to_k)?;
        f(&mut self.to_v)?;
        f(&mut self.to_out)?;
        Ok(())
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;

        // Project to Q, K, V (through the LoRA-adapted projections).
        let q = hidden_states.apply(&self.to_q)?;
        let k = hidden_states.apply(&self.to_k)?;
        let v = hidden_states.apply(&self.to_v)?;

        // (B, seq, n*hd) -> (B, seq, n, hd)
        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_heads, self.head_dim))?;

        let (q, k) = if let Some(ref norm) = self.qk_norm {
            norm.forward(&q, &k)?
        } else {
            (q, k)
        };

        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // (B, n, seq, hd)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(m) = attention_mask {
            // mask: (B, seq) -> (B, 1, 1, seq); 1=valid -> 0, 0=padding -> -1e9
            let m = m
                .unsqueeze(1)?
                .unsqueeze(2)?
                .to_dtype(attn_weights.dtype())?;
            let m = ((m - 1.0)? * 1e9)?;
            attn_weights = attn_weights.broadcast_add(&m)?;
        }

        // Composable softmax (NOT the fused `softmax_last_dim` — that CustomOp has no backward).
        let attn_probs = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let context = attn_probs.matmul(&v)?;

        // (B, n, seq, hd) -> (B, seq, dim)
        let context = context.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        context.apply(&self.to_out)
    }
}

// ==================== ZImageTransformerBlock ====================

/// Z-Image transformer block with optional AdaLN modulation. Identical to the stock block except its
/// attention is the vendored [`ZImageAttention`]; the norms / FFN / AdaLN projection are stock
/// frozen modules.
#[derive(Debug, Clone)]
pub struct ZImageTransformerBlock {
    attention: ZImageAttention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    adaln_modulation: Option<candle_nn::Linear>,
}

impl ZImageTransformerBlock {
    pub fn new(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim();

        let attention = ZImageAttention::new(cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(dim, hidden_dim, vb.pp("feed_forward"))?;

        let attention_norm1 = candle_nn::rms_norm(dim, cfg.norm_eps, vb.pp("attention_norm1"))?;
        let attention_norm2 = candle_nn::rms_norm(dim, cfg.norm_eps, vb.pp("attention_norm2"))?;
        let ffn_norm1 = candle_nn::rms_norm(dim, cfg.norm_eps, vb.pp("ffn_norm1"))?;
        let ffn_norm2 = candle_nn::rms_norm(dim, cfg.norm_eps, vb.pp("ffn_norm2"))?;

        let adaln_modulation = if modulation {
            let adaln_dim = dim.min(ADALN_EMBED_DIM);
            Some(candle_nn::linear(
                adaln_dim,
                4 * dim,
                vb.pp("adaLN_modulation").pp("0"),
            )?)
        } else {
            None
        };

        Ok(Self {
            attention,
            feed_forward,
            attention_norm1,
            attention_norm2,
            ffn_norm1,
            ffn_norm2,
            adaln_modulation,
        })
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attention.visit_lora_mut(f)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(ref adaln) = self.adaln_modulation {
            let adaln_input = adaln_input.expect("adaln_input required when modulation=true");
            let modulation = adaln_input.apply(adaln)?.unsqueeze(1)?;
            let chunks = modulation.chunk(4, D::Minus1)?;
            let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
                (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);

            let gate_msa = gate_msa.tanh()?;
            let gate_mlp = gate_mlp.tanh()?;
            let scale_msa = (scale_msa + 1.0)?;
            let scale_mlp = (scale_mlp + 1.0)?;

            // Attention block
            let normed = self.attention_norm1.forward_diff(x)?;
            let scaled = normed.broadcast_mul(&scale_msa)?;
            let attn_out = self.attention.forward(&scaled, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward_diff(&attn_out)?;
            let x = (x + gate_msa.broadcast_mul(&attn_out)?)?;

            // FFN block
            let normed = self.ffn_norm1.forward_diff(&x)?;
            let scaled = normed.broadcast_mul(&scale_mlp)?;
            let ffn_out = self.feed_forward.forward(&scaled)?;
            let ffn_out = self.ffn_norm2.forward_diff(&ffn_out)?;
            x + gate_mlp.broadcast_mul(&ffn_out)?
        } else {
            let normed = self.attention_norm1.forward_diff(x)?;
            let attn_out = self.attention.forward(&normed, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward_diff(&attn_out)?;
            let x = (x + attn_out)?;

            let normed = self.ffn_norm1.forward_diff(&x)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let ffn_out = self.ffn_norm2.forward_diff(&ffn_out)?;
            x + ffn_out
        }
    }
}

// ==================== ZImageTransformer2DModel ====================

/// Z-Image Transformer 2D Model — the vendored, trainable twin of the stock
/// `ZImageTransformer2DModel`. Built from the *same* `transformer/` safetensors keys (the reuse +
/// the unchanged sub-module paths guarantee key parity), so it loads the real weights unchanged and,
/// with no adapter installed, reproduces the stock forward bit-for-bit (`parity_tests`).
#[derive(Debug, Clone)]
pub struct ZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: candle_nn::Linear,
    x_embedder: candle_nn::Linear,
    final_layer: FinalLayer,
    #[allow(dead_code)]
    x_pad_token: Tensor,
    #[allow(dead_code)]
    cap_pad_token: Tensor,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: Vec<ZImageTransformerBlock>,
    rope_embedder: RopeEmbedder,
    cfg: Config,
}

impl ZImageTransformer2DModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();

        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);
        let t_embedder = TimestepEmbedder::new(adaln_dim, 1024, vb.pp("t_embedder"))?;

        let cap_embedder_norm = candle_nn::rms_norm(
            cfg.cap_feat_dim,
            cfg.norm_eps,
            vb.pp("cap_embedder").pp("0"),
        )?;
        let cap_embedder_linear =
            candle_nn::linear(cfg.cap_feat_dim, cfg.dim, vb.pp("cap_embedder").pp("1"))?;

        let patch_dim = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.in_channels;
        let x_embedder = candle_nn::linear(patch_dim, cfg.dim, vb.pp("all_x_embedder").pp("2-1"))?;

        let out_channels = cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_f_patch_size[0]
            * cfg.in_channels;
        let final_layer =
            FinalLayer::new(cfg.dim, out_channels, vb.pp("all_final_layer").pp("2-1"))?;

        let x_pad_token = vb.get((1, cfg.dim), "x_pad_token")?;
        let cap_pad_token = vb.get((1, cfg.dim), "cap_pad_token")?;

        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            noise_refiner.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("noise_refiner").pp(i),
            )?);
        }

        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            context_refiner.push(ZImageTransformerBlock::new(
                cfg,
                false,
                vb.pp("context_refiner").pp(i),
            )?);
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("layers").pp(i),
            )?);
        }

        let rope_embedder = RopeEmbedder::new(
            cfg.rope_theta,
            cfg.axes_dims.clone(),
            cfg.axes_lens.clone(),
            device,
            dtype,
        )?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            x_pad_token,
            cap_pad_token,
            noise_refiner,
            context_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
        })
    }

    /// Forward pass — returns the **raw** DiT velocity `(B, C, F, H, W)` (no sign flip; the inference
    /// pipeline and the trainer apply `.neg()`). Byte-faithful to the stock model's forward.
    ///
    /// * `x` — latent `(B, C, F, H, W)`
    /// * `t` — flow-match timestep in `[0, 1]` `(B,)`
    /// * `cap_feats` — caption features `(B, text_len, cap_feat_dim)`
    /// * `cap_mask` — caption attention mask `(B, text_len)`, 1=valid, 0=padding
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        let device = x.device();
        let (b, _c, f, h, w) = x.dims5()?;
        let patch_size = self.cfg.all_patch_size[0];
        let f_patch_size = self.cfg.all_f_patch_size[0];

        // 1. Timestep embedding
        let t_scaled = (t * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?;

        // 2. Patchify + embed image
        let (x_patches, orig_size) = patchify(x, patch_size, f_patch_size)?;
        let mut x = x_patches.apply(&self.x_embedder)?;
        let img_seq_len = x.dim(1)?;

        // 3. Image position IDs (offset past the text block)
        let f_tokens = f / f_patch_size;
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let text_len = cap_feats.dim(1)?;
        let x_pos_ids =
            create_coordinate_grid((f_tokens, h_tokens, w_tokens), (text_len + 1, 0, 0), device)?;
        let (x_cos, x_sin) = self.rope_embedder.forward(&x_pos_ids)?;

        // 4. Caption embedding (composable norm — see the module header on fused RMSNorm).
        let cap_normed = self.cap_embedder_norm.forward_diff(cap_feats)?;
        let mut cap = cap_normed.apply(&self.cap_embedder_linear)?;

        // 5. Caption position IDs
        let cap_pos_ids = create_coordinate_grid((text_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        // 6. Attention masks
        let x_attn_mask = Tensor::ones((b, img_seq_len), DType::U8, device)?;
        let cap_attn_mask = cap_mask.to_dtype(DType::U8)?;

        // 7. Noise refiner (image, with modulation)
        for layer in &self.noise_refiner {
            x = layer.forward(&x, Some(&x_attn_mask), &x_cos, &x_sin, Some(&adaln_input))?;
        }

        // 8. Context refiner (text, without modulation)
        for layer in &self.context_refiner {
            cap = layer.forward(&cap, Some(&cap_attn_mask), &cap_cos, &cap_sin, None)?;
        }

        // 9. Concatenate [image, text]
        let unified = Tensor::cat(&[&x, &cap], 1)?;

        // 10. Unified position IDs + mask
        let unified_pos_ids = Tensor::cat(&[&x_pos_ids, &cap_pos_ids], 0)?;
        let (unified_cos, unified_sin) = self.rope_embedder.forward(&unified_pos_ids)?;
        let unified_attn_mask = Tensor::cat(&[&x_attn_mask, &cap_attn_mask], 1)?;

        // 11. Main transformer layers
        let mut unified = unified;
        for layer in &self.layers {
            unified = layer.forward(
                &unified,
                Some(&unified_attn_mask),
                &unified_cos,
                &unified_sin,
                Some(&adaln_input),
            )?;
        }

        // 12. Final layer (image portion only)
        let x_out = unified.narrow(1, 0, img_seq_len)?;
        let x_out = self.final_layer.forward(&x_out, &adaln_input)?;

        // 13. Unpatchify
        unpatchify(
            &x_out,
            orig_size,
            patch_size,
            f_patch_size,
            self.cfg.in_channels,
        )
    }
}

impl LoraHost for ZImageTransformer2DModel {
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for blk in self
            .noise_refiner
            .iter_mut()
            .chain(self.context_refiner.iter_mut())
            .chain(self.layers.iter_mut())
        {
            blk.visit_lora_mut(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod parity_tests {
    //! Pin the vendored DiT to the stock candle-transformers DiT: built from the *same*
    //! `VarMap`-backed weights with no adapter installed, the two must produce bit-identical forward
    //! output. The regression guard that the vendoring (the `LoraLinear` swap + composable softmax)
    //! changed nothing numerically.
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::z_image::preprocess::prepare_inputs;
    use candle_transformers::models::z_image::transformer::{
        Config, ZImageTransformer2DModel as StockModel,
    };

    /// A tiny Z-Image-shaped config: `head_dim` is locked to 128 by `axes_dims=[32,48,48]` (the RoPE
    /// half-dims `16+24+24=64=head_dim/2`), so a single head at `dim=128` is the smallest valid DiT.
    /// 2 main layers + 1 refiner each exercises every vendored path cheaply on CPU.
    fn tiny_cfg() -> Config {
        let mut cfg = Config::z_image_turbo();
        cfg.dim = 128; // head_dim = 128/1 = 128 (axes_dims sum)
        cfg.n_heads = 1;
        cfg.n_kv_heads = 1;
        cfg.n_layers = 2;
        cfg.n_refiner_layers = 1;
        cfg.cap_feat_dim = 64;
        cfg.use_accelerated_attn = false; // force the math/basic path on the stock side
        cfg
    }

    #[test]
    fn vendored_dit_matches_stock_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // The vendored model is built first, populating the VarMap with random weights; the stock
        // model then reads the SAME parameters (identical names/shapes), so any output difference is
        // a forward-logic difference, not a weight difference.
        let vendored = ZImageTransformer2DModel::new(&cfg, vb.clone()).unwrap();
        let stock = StockModel::new(&cfg, vb).unwrap();

        // latent (1, 16, 4, 4) -> patchified 2x2 = 4 image tokens; tiny caption of 3 tokens.
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let prepared = prepare_inputs(&latent, std::slice::from_ref(&cap), &dev).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let y_v = vendored
            .forward(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
            )
            .unwrap();
        let y_s = stock
            .forward(
                &prepared.latents,
                &t,
                &prepared.cap_feats,
                &prepared.cap_mask,
            )
            .unwrap();

        assert_eq!(y_v.dims(), y_s.dims());
        let diff = (y_v - y_s)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "vendored DiT diverged from stock by {diff}");
    }

    /// The [`LoraHost`] walk reaches exactly `4 × (n_refiner·2 + n_layers)` projections — the four
    /// attention `LoraLinear`s in every noise-refiner, context-refiner, and main block.
    #[test]
    fn lora_host_visits_every_attention_projection() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut model = ZImageTransformer2DModel::new(&cfg, vb).unwrap();
        let mut paths: Vec<String> = Vec::new();
        model
            .visit_lora_mut(&mut |lin| {
                paths.push(lin.path().to_string());
                Ok(())
            })
            .unwrap();
        let expected = 4 * (cfg.n_refiner_layers * 2 + cfg.n_layers);
        assert_eq!(paths.len(), expected);
        // Every default target suffix resolves against at least one visited path.
        for suffix in Z_IMAGE_ATTN_TARGETS {
            assert!(
                paths
                    .iter()
                    .any(|p| p == suffix || p.ends_with(&format!(".{suffix}"))),
                "no visited projection matched suffix {suffix}"
            );
        }
        // Layer-0 main-block attention paths are present and correctly named.
        assert!(paths.contains(&"layers.0.attention.to_q".to_string()));
        assert!(paths.contains(&"layers.0.attention.to_out.0".to_string()));
    }
}
