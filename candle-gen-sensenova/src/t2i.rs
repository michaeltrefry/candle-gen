//! Text-to-image generation — the candle port of `mlx-gen-sensenova`'s `t2i.rs` `t2i_generate` spine,
//! scoped to the **non-think T2I** path the `Generator` contract drives (it2i / VQA / interleave /
//! think-mode are the understanding surface → Phase 6).
//!
//! The flow:
//! 1. Build the `neo1_0` query ([`build_neo1_query`] + [`SYSTEM_MESSAGE_FOR_GEN`] + the no-think
//!    `<think>\n\n</think>\n\n<img>` sentinel), tokenize, and **prefill** it into a KV cache on the
//!    understanding path. With CFG (`cfg_scale > 1`) a second, *uncondition* prefix (`<img>` after an
//!    empty prompt) is prefilled into its own cache.
//! 2. **Denoise** for `num_steps` over the standard flow-matching schedule
//!    ([`apply_time_schedule`]): each step embeds the current noisy image through the gen-path
//!    [`NeoVisionEmbedder`] (channel-first patches) + the timestep (and noise-scale) embedding, runs
//!    the **generation** path over `[cached prefix ++ image block]` use-only (`append=false`), maps
//!    the image hidden states through the [`FmHead`] to a patch latent `x_pred`, forms the
//!    [`velocity`], and takes an [`euler_step`]. CFG blends the condition/uncondition velocities.
//! 3. [`unpatchify`] the final latent → RGB `[1, 3, H, W]` (model space ≈ `[-1, 1]`).
//!
//! Deterministic, launch-portable initial noise from a fixed-algorithm CPU RNG (`StdRng`, sc-3673) —
//! same-backend determinism only; cross-backend pixel-equality vs `mlx-gen-sensenova` is NOT a goal.

use candle_gen::candle_core::{Device, IndexOp, Result as CResult, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{CancelFlag, Image, Progress};
use candle_gen::{CandleError, Result};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::config::NeoChatConfig;
use crate::distill::DistillLora;
use crate::fm::{
    apply_time_schedule, euler_step, patchify, patchify_channel_first, unpatchify, velocity,
    FmHead, TimestepEmbedder,
};
use crate::qwen3::{KvCache, Path, Qwen3Backbone, RopeMask};
use crate::runtime::{argmax, Sampler};
use crate::text::{
    build_neo1_query, image_indexes, text_indexes, tokens, SenseNovaTokenizer,
    SYSTEM_MESSAGE_FOR_GEN,
};
use crate::vision::NeoVisionEmbedder;

/// Classifier-free-guidance velocity-blend normalisation (`t2i_generate`'s `cfg_norm`). The T2I
/// `Generator` path always uses [`CfgNorm::None`]; the understanding it2i/interleave denoise honours
/// Global/Channel and rejects CFG-Zero* (a T2I-only blend mode), matching the reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CfgNorm {
    /// Plain blend `v_uncond + cfg·(v_cond − v_uncond)`.
    #[default]
    None,
    /// Rescale the blended velocity to the condition velocity's global norm.
    Global,
    /// Per-token rescale to the condition velocity's per-token norm.
    Channel,
    /// CFG-Zero* (T2I-only; rejected on the it2i/interleave path).
    CfgZeroStar,
}

/// Knobs for the T2I and understanding (it2i / VQA / interleave) denoise paths (the `t2i_generate`
/// arguments). The T2I `Generator` path drives only `cfg_scale` + plain CFG; the it2i/interleave
/// surface adds `img_cfg_scale` (dual guidance), `cfg_norm`, and `think_mode`.
#[derive(Clone, Copy, Debug)]
pub struct T2iOptions {
    pub cfg_scale: f32,
    /// Image-guidance scale for it2i / interleave (`img_cfg_scale`): edit ≈ 1.0, character ≈ 1.5.
    /// Unused by plain T2I.
    pub img_cfg_scale: f32,
    pub cfg_norm: CfgNorm,
    pub cfg_interval: (f32, f32),
    pub num_steps: usize,
    pub timestep_shift: f32,
    pub enable_timestep_shift: bool,
    pub t_eps: f32,
    pub seed: u64,
    /// Think-mode: interleave reasons (and may emit images) inside a `<think>…</think>` block before
    /// the final answer. Off ⇒ the non-think sentinel is primed so the model answers directly.
    pub think_mode: bool,
}

impl Default for T2iOptions {
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            img_cfg_scale: 1.0,
            cfg_norm: CfgNorm::None,
            cfg_interval: (0.0, 1.0),
            num_steps: 30,
            timestep_shift: 1.0,
            enable_timestep_shift: true,
            t_eps: 0.02,
            seed: 0,
            think_mode: false,
        }
    }
}

/// The result of a [`T2iModel::interleave_gen`] run: the composed text (with `<image>` placeholders
/// where images were generated) and the generated model-space images `[1,3,H,W]` in order.
pub struct InterleaveOutput {
    pub text: String,
    pub images: Vec<Tensor>,
}

/// The interleave resolution buckets (`examples/interleave/inference.py::SUPPORTED_RESOLUTIONS`) —
/// `(width, height)` per aspect ratio. Document Studio picks one of these. Default `"16:9"`.
pub const INTERLEAVE_RESOLUTIONS: &[(&str, (i32, i32))] = &[
    ("1:1", (1536, 1536)),
    ("16:9", (2048, 1152)),
    ("9:16", (1152, 2048)),
    ("3:2", (1888, 1248)),
    ("2:3", (1248, 1888)),
    ("4:3", (1760, 1312)),
    ("3:4", (1312, 1760)),
    ("1:2", (1088, 2144)),
    ("2:1", (2144, 1088)),
    ("1:3", (864, 2592)),
    ("3:1", (2592, 864)),
];

/// Look up an interleave resolution bucket by aspect-ratio key (e.g. `"16:9"`).
pub fn interleave_resolution_for(ratio: &str) -> Option<(i32, i32)> {
    INTERLEAVE_RESOLUTIONS
        .iter()
        .find(|(r, _)| *r == ratio)
        .map(|(_, wh)| *wh)
}

/// The T2I model: the dual-path backbone plus the flow-matching generation modules.
pub struct T2iModel {
    backbone: Qwen3Backbone,
    gen_vision: NeoVisionEmbedder,
    /// The **understanding**-path vision embedder (`vision_model.embeddings`) used to embed source /
    /// reference / re-encoded generated images for the it2i / VQA / interleave surfaces. `None` for
    /// T2I-only fixtures that omit `vision_model.*` (the registry T2I path never touches it).
    und_vision: Option<NeoVisionEmbedder>,
    fm_head: FmHead,
    timestep_embedder: TimestepEmbedder,
    noise_scale_embedder: Option<TimestepEmbedder>,
    patch_size: usize,
    merge_size: usize,
    noise_scale: f32,
    noise_scale_mode: String,
    noise_scale_base_image_seq_len: f32,
    noise_scale_max_value: f32,
    /// `<IMG_CONTEXT>` / `<img>` / `</img>` ids (the checkpoint constants).
    img_context_id: i32,
    img_start_id: i32,
    img_end_id: i32,
    device: Device,
}

impl T2iModel {
    /// Build from a loaded checkpoint VarBuilder (`language_model.*` + `fm_modules.*`, all f32).
    pub fn from_weights(vb: &VarBuilder, cfg: &NeoChatConfig) -> Result<Self> {
        // `noise_scale_embed` divides each step's conditioning by `noise_scale_max_value`; a
        // zero/negative value would inject NaN/Inf conditioning silently. Reject at load (F-012).
        if cfg.noise_scale_max_value <= 0.0 || cfg.noise_scale_max_value.is_nan() {
            return Err(CandleError::Msg(format!(
                "sensenova: noise_scale_max_value must be > 0 (got {})",
                cfg.noise_scale_max_value
            )));
        }
        let noise_scale_embedder = if cfg.add_noise_scale_embedding {
            Some(TimestepEmbedder::from_weights(
                vb,
                "fm_modules.noise_scale_embedder",
            )?)
        } else {
            None
        };
        // The understanding-path vision embedder is only needed by the it2i / VQA / interleave
        // surfaces; gate on its presence so T2I-only fixtures (no `vision_model.*`) still load.
        let und_vision = if vb.contains_tensor("vision_model.embeddings.patch_embedding.weight") {
            Some(NeoVisionEmbedder::from_weights(
                vb,
                cfg,
                "vision_model.embeddings",
            )?)
        } else {
            None
        };
        Ok(Self {
            backbone: Qwen3Backbone::from_weights(vb, cfg, "language_model")?,
            gen_vision: NeoVisionEmbedder::from_weights(
                vb,
                cfg,
                "fm_modules.vision_model_mot_gen.embeddings",
            )?,
            und_vision,
            fm_head: FmHead::from_weights(vb, "fm_modules.fm_head")?,
            timestep_embedder: TimestepEmbedder::from_weights(vb, "fm_modules.timestep_embedder")?,
            noise_scale_embedder,
            img_context_id: tokens::IMG_CONTEXT,
            img_start_id: tokens::IMG_START,
            img_end_id: tokens::IMG_END,
            patch_size: cfg.patch_size,
            merge_size: (1.0 / cfg.downsample_ratio).round() as usize,
            noise_scale: cfg.noise_scale,
            noise_scale_mode: cfg.noise_scale_mode.clone(),
            noise_scale_base_image_seq_len: cfg.noise_scale_base_image_seq_len as f32,
            noise_scale_max_value: cfg.noise_scale_max_value,
            device: vb.device().clone(),
        })
    }

    /// Merge the 8-step distill LoRA (the `fast` variant): the backbone generation-path projections
    /// (`7 · layers`) + the two FM-head Linears. Returns the total Linears merged.
    pub fn merge_distill_lora(&mut self, lora: &DistillLora) -> Result<usize> {
        let n = self.backbone.merge_distill_lora(lora, "language_model")?;
        Ok(n + self
            .fm_head
            .merge_distill_lora(lora, "fm_modules.fm_head")?)
    }

    /// The patch·merge cell — every image side must be a multiple of this.
    pub fn cell(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// The resolution-mode noise scale for a `grid_h × grid_w` patch grid (the `t2i_generate`
    /// formula), clamped to `noise_scale_max_value`.
    fn noise_scale_for(&self, grid_h: usize, grid_w: usize) -> f32 {
        let mut scale = self.noise_scale;
        if matches!(
            self.noise_scale_mode.as_str(),
            "resolution" | "dynamic" | "dynamic_sqrt"
        ) {
            let seq = (grid_h * grid_w) as f32 / (self.merge_size * self.merge_size) as f32;
            scale = (seq / self.noise_scale_base_image_seq_len).sqrt() * self.noise_scale;
            if self.noise_scale_mode == "dynamic_sqrt" {
                scale = scale.sqrt();
            }
        }
        scale.min(self.noise_scale_max_value)
    }

    /// Prefill a text query into a fresh understanding-path cache; returns the cache and prefix len.
    fn prefill(&self, ids: &[i32]) -> CResult<(KvCache, usize)> {
        let embeds = self.backbone.embed(ids)?;
        let (t, h, w) = text_indexes(ids.len());
        let mut cache = self.backbone.new_cache();
        // The returned hidden state is unused (non-think → no logits); we only need the populated
        // cache for the denoise loop's gen-path forwards.
        self.backbone
            .forward_cached(&embeds, &t, &h, &w, Path::Und, &mut cache, true)?;
        Ok((cache, ids.len()))
    }

    /// Build the gen-path [`RopeMask`] for the image block (`image_indexes` for a `token_h × token_w`
    /// grid after `text_len` prefix tokens, block mask at cache prefix `past`).
    fn prepare_gen(
        &self,
        token_h: usize,
        token_w: usize,
        text_len: usize,
        past: usize,
    ) -> CResult<RopeMask> {
        let (it, ih, iw) = image_indexes(token_h, token_w, text_len);
        self.backbone.prepare_rope_mask(&it, &ih, &iw, past)
    }

    /// Gen-path velocity prediction for one diffusion step against a prefilled cache: `forward_prepared`
    /// (Gen, use-only) over the conditioned image block, `fm_head` → `x_pred`, then the velocity.
    fn predict_v(
        &self,
        image_embeds: &Tensor,
        rm: &RopeMask,
        cache: &mut KvCache,
        z: &Tensor,
        t: f32,
        t_eps: f32,
    ) -> CResult<Tensor> {
        let hidden = self
            .backbone
            .forward_prepared(image_embeds, rm, Path::Gen, cache, false)?;
        let x_pred = self.fm_head.forward(&hidden)?;
        velocity(&x_pred, z, t, t_eps)
    }

    /// Build one denoise step's latent `z` (channel-last patchify at `cell`) and the conditioned image
    /// block `cond = gen_vision(image) + timestep_embed(t) [+ noise_scale_embed]` `[1, L, hidden]`.
    fn step_cond_embeds(
        &self,
        image: &Tensor,
        grid_h: usize,
        grid_w: usize,
        l: usize,
        t: f32,
        noise_embed: &Option<Tensor>,
    ) -> CResult<(Tensor, Tensor)> {
        let cell = self.cell();
        let z = patchify(image, cell)?;
        let pdim = 3 * self.patch_size * self.patch_size;
        let image_input =
            patchify_channel_first(image, self.patch_size)?.reshape((grid_h * grid_w, pdim))?;
        let vis = self.gen_vision.forward(&image_input, &[(grid_h, grid_w)])?;
        let hidden = vis.dim(1)?;
        let vis = vis.reshape((1, l, hidden))?;
        let t_in = Tensor::from_vec(vec![t; l], (l,), &self.device)?;
        let t_tok = self
            .timestep_embedder
            .forward(&t_in)?
            .reshape((1, l, hidden))?;
        let mut cond = (vis + t_tok)?;
        if let Some(ne) = noise_embed {
            cond = (cond + ne)?;
        }
        Ok((z, cond))
    }

    /// The constant noise-scale conditioning token `[1, L, hidden]` (only when a `noise_scale_embedder`
    /// is present), added to every step's timestep embedding.
    fn noise_scale_embed(&self, noise_scale: f32, l: usize) -> CResult<Option<Tensor>> {
        let Some(emb) = &self.noise_scale_embedder else {
            return Ok(None);
        };
        let ns = vec![noise_scale / self.noise_scale_max_value; l];
        let out = emb.forward(&Tensor::from_vec(ns, (l,), &self.device)?)?;
        let hidden = out.dim(1)?;
        Ok(Some(out.reshape((1, l, hidden))?))
    }

    /// Generate an image for `prompt` at `width × height` (both multiples of `patch·merge`). Emits
    /// per-step [`Progress`] and aborts on `cancel`. Returns the model-space image `[1, 3, H, W]`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        tokenizer: &SenseNovaTokenizer,
        prompt: &str,
        width: usize,
        height: usize,
        opts: &T2iOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let cell = self.cell();
        if !width.is_multiple_of(cell) || !height.is_multiple_of(cell) {
            return Err(CandleError::Msg(format!(
                "sensenova t2i: width/height must be multiples of {cell}, got {width}x{height}"
            )));
        }

        // ---- Condition prefix (no-think sentinel) ----
        let query_cond = format!(
            "{}<think>\n\n</think>\n\n<img>",
            build_neo1_query(prompt, SYSTEM_MESSAGE_FOR_GEN)
        );
        let ids_cond = tokenizer.encode_ids(&query_cond, true)?;
        let (mut cache_cond, text_len) = self.prefill(&ids_cond)?;

        // ---- Uncondition prefix (CFG) ----
        let needs_cfg = opts.cfg_scale > 1.0;
        let mut cache_uncond: Option<KvCache> = None;
        let mut uncond_text_len = 0;
        if needs_cfg {
            let query_uncond = format!("{}<img>", build_neo1_query("", ""));
            let ids_uncond = tokenizer.encode_ids(&query_uncond, true)?;
            let (cache, plen) = self.prefill(&ids_uncond)?;
            cache_uncond = Some(cache);
            uncond_text_len = plen;
        }

        let base_noise = gaussian((1, 3, height, width), opts.seed, &self.device)?;
        let image = self.denoise(
            &mut cache_cond,
            text_len,
            cache_uncond.as_mut(),
            uncond_text_len,
            width,
            height,
            &base_noise,
            opts,
            cancel,
            on_progress,
        )?;
        Ok(image)
    }

    /// The flow-matching denoise loop. Returns the final model-space image `[1,3,H,W]`.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        cache_cond: &mut KvCache,
        text_len: usize,
        mut cache_uncond: Option<&mut KvCache>,
        uncond_text_len: usize,
        width: usize,
        height: usize,
        base_noise: &Tensor,
        opts: &T2iOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let cell = self.cell();
        let token_h = height / cell;
        let token_w = width / cell;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let l = token_h * token_w;

        let noise_scale = self.noise_scale_for(grid_h, grid_w);
        let mut image = (base_noise * noise_scale as f64)?;

        let steps = opts.num_steps;
        let timesteps = step_schedule(opts)?;
        let noise_embed = self.noise_scale_embed(noise_scale, l)?;

        let needs_cfg = opts.cfg_scale > 1.0 && cache_uncond.is_some();
        // RoPE tables + block mask are invariant across steps for a given cache — build once (F-139).
        let rm_cond = self.prepare_gen(token_h, token_w, text_len, cache_cond.len())?;
        let rm_uncond = match cache_uncond.as_deref() {
            Some(cu) => Some(self.prepare_gen(token_h, token_w, uncond_text_len, cu.len())?),
            None => None,
        };

        for i in 0..steps {
            if cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let t = timesteps[i];
            let t_next = timesteps[i + 1];

            let (z, cond) = self.step_cond_embeds(&image, grid_h, grid_w, l, t, &noise_embed)?;
            let v_cond = self.predict_v(&cond, &rm_cond, cache_cond, &z, t, opts.t_eps)?;

            // CFG-interval gate (inclusive both ends — the reference T2I gate).
            let v_pred = if needs_cfg && t >= opts.cfg_interval.0 && t <= opts.cfg_interval.1 {
                let cache_u = cache_uncond
                    .as_deref_mut()
                    .expect("needs_cfg ⇒ uncond cache");
                let rm_u = rm_uncond.as_ref().expect("needs_cfg ⇒ uncond rope-mask");
                let v_uncond = self.predict_v(&cond, rm_u, cache_u, &z, t, opts.t_eps)?;
                cfg_blend(&v_cond, &v_uncond, opts.cfg_scale)?
            } else {
                v_cond
            };

            image = unpatchify(&euler_step(&v_pred, &z, t, t_next)?, cell, token_h, token_w)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total: steps as u32,
            });
        }
        Ok(image)
    }

    // ===================== understanding path (VQA + interleave) =====================

    /// ImageNet-normalise a source RGB image `[3, H, W]` (f32 in `[0, 1]`) and channel-first patchify
    /// to `[grid_h·grid_w, 3·ps²]` for the understanding vision embedder. Returns the patches and the
    /// `(grid_h, grid_w)` patch grid. `H`/`W` must be multiples of `patch·merge` (use
    /// [`smart_resize`] upstream). The input may live on any device — it is moved to the model's
    /// device first (the worker builds VQA / interleave inputs on CPU; candle treats every
    /// `Device::new_cuda(0)` handle as distinct, so mixing a foreign-device input with the model's
    /// tensors would otherwise error). Mirrors the reference `load_image_native` (ToTensor +
    /// Normalize) + `preprocess_pixel_values`.
    pub fn preprocess_image(&self, rgb: &Tensor) -> CResult<(Tensor, (usize, usize))> {
        let (c, h, w) = rgb.dims3()?;
        if c != 3 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "sensenova: expected source image [3,H,W], got [{c},{h},{w}]"
            )));
        }
        let cell = self.cell();
        if !h.is_multiple_of(cell) || !w.is_multiple_of(cell) {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "sensenova: source image H/W must be multiples of {cell}, got {h}x{w}"
            )));
        }
        // Relocate to the model's device + dtype (a no-op for the on-device append path).
        let rgb = rgb
            .to_device(&self.device)?
            .to_dtype(candle_gen::candle_core::DType::F32)?;
        let mean = Tensor::from_vec(vec![0.485f32, 0.456, 0.406], (3, 1, 1), &self.device)?;
        let std = Tensor::from_vec(vec![0.229f32, 0.224, 0.225], (3, 1, 1), &self.device)?;
        let norm = rgb.broadcast_sub(&mean)?.broadcast_div(&std)?;
        let (gh, gw) = (h / self.patch_size, w / self.patch_size);
        let patches = patchify_channel_first(&norm.reshape((1, 3, h, w))?, self.patch_size)?
            .reshape((gh * gw, 3 * self.patch_size * self.patch_size))?;
        Ok((patches, (gh, gw)))
    }

    /// Understanding-path vision features for source-image patches `[Σ tokens, llm_hidden]`.
    pub fn und_vision_features(
        &self,
        pixel_values: &Tensor,
        grids: &[(usize, usize)],
    ) -> CResult<Tensor> {
        let und = self.und_vision.as_ref().ok_or_else(|| {
            candle_gen::candle_core::Error::Msg("sensenova: vision_model not loaded".into())
        })?;
        und.forward(pixel_values, grids)
    }

    /// (t, h, w) position rows for a prefix containing source-image blocks (the reference
    /// `get_thw_indexes`): text tokens advance temporal by one; an image-context block shares one
    /// temporal index and carries its **merged-grid** `(row, col)` as `(h, w)`. `grids` are the full
    /// patch grids `(grid_h, grid_w)` per image, in order.
    fn get_thw_indexes(
        &self,
        ids: &[i32],
        grids: &[(usize, usize)],
    ) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let n = ids.len();
        let mut t = Vec::with_capacity(n);
        let mut acc = 0i32;
        for i in 0..n {
            let shift = i32::from(i > 0 && ids[i - 1] == self.img_start_id);
            let not_img = i32::from(ids[i] != self.img_context_id);
            acc += shift + not_img;
            t.push(acc - 1);
        }
        // Merged-grid (row=y, col=x) coordinates, concatenated across images in order.
        let merge = self.merge_size;
        let mut abs = Vec::new();
        for &(gh, gw) in grids {
            let (mh, mw) = (gh / merge, gw / merge);
            for idx in 0..(mh * mw) {
                abs.push(((idx / mw) as i32, (idx % mw) as i32));
            }
        }
        let mut h = vec![0i32; n];
        let mut w = vec![0i32; n];
        let mut k = 0usize;
        for i in 0..n {
            if ids[i] == self.img_context_id {
                let (y, x) = abs[k];
                h[i] = y;
                w[i] = x;
                k += 1;
            }
        }
        (t, h, w)
    }

    /// Embed `ids` and splice the understanding vision features into the `<IMG_CONTEXT>` positions
    /// (the reference `_build_it2i_inputs`). Returns the prefix embeds `[1, S, hidden]` and its
    /// `(t, h, w)` rows. Scatter is a one-hot selection matmul (no in-place index assignment).
    #[allow(clippy::type_complexity)]
    fn build_it2i_prefix(
        &self,
        ids: &[i32],
        pixel_values: Option<&Tensor>,
        grids: &[(usize, usize)],
    ) -> CResult<(Tensor, Vec<i32>, Vec<i32>, Vec<i32>)> {
        let s = ids.len();
        let mut embeds = self.backbone.embed(ids)?; // [1, S, H]
        let (t, h, w) = self.get_thw_indexes(ids, grids);

        if let Some(pv) = pixel_values {
            let vit = self.und_vision_features(pv, grids)?; // [n_ctx, H]
            let hidden = embeds.dim(2)?;
            let ctx: Vec<usize> = ids
                .iter()
                .enumerate()
                .filter(|(_, &id)| id == self.img_context_id)
                .map(|(i, _)| i)
                .collect();
            let n_ctx = vit.dim(0)?;
            if ctx.len() != n_ctx {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "sensenova it2i: {} <IMG_CONTEXT> tokens but {n_ctx} vision tokens",
                    ctx.len()
                )));
            }
            // P [S, n_ctx] one-hot: row = sequence position, col = vision-token index.
            let mut p = vec![0f32; s * n_ctx];
            let mut mask = vec![0f32; s];
            for (k, &pos) in ctx.iter().enumerate() {
                p[pos * n_ctx + k] = 1.0;
                mask[pos] = 1.0;
            }
            let p_arr = Tensor::from_vec(p, (s, n_ctx), &self.device)?;
            let vit_full = p_arr.matmul(&vit)?; // [S, H], 0 off-context
            let keep: Vec<f32> = mask.iter().map(|m| 1.0 - m).collect();
            let keep_mask = Tensor::from_vec(keep, (s, 1), &self.device)?;
            let e2d = embeds.reshape((s, hidden))?;
            embeds = (e2d.broadcast_mul(&keep_mask)? + vit_full)?.reshape((1, s, hidden))?;
        }
        Ok((embeds, t, h, w))
    }

    /// Prefill a prepared prefix (embeds + positions) on the understanding path. Returns the cache,
    /// the last-position logits, and the image-block temporal index (`max(t) + 1`). The last hidden
    /// row is sliced before `lm_head` so the projection materializes one `[1,1,vocab]` row, not the
    /// whole `[1,S,vocab]` prefix.
    fn prefill_prefix(
        &self,
        embeds: &Tensor,
        t: &[i32],
        h: &[i32],
        w: &[i32],
    ) -> CResult<(KvCache, Vec<f32>, usize)> {
        let mut cache = self.backbone.new_cache();
        let hidden = self
            .backbone
            .forward_cached(embeds, t, h, w, Path::Und, &mut cache, true)?;
        let s = t.len();
        let last_hidden = hidden.narrow(1, s - 1, 1)?; // [1, 1, H]
        let logits = self.backbone.lm_head(&last_hidden)?; // [1, 1, vocab]
        let vocab = logits.dim(2)?;
        let last = logits.reshape((vocab,))?.to_vec1::<f32>()?;
        let img_temporal = (*t.iter().max().unwrap_or(&0) + 1) as usize;
        Ok((cache, last, img_temporal))
    }

    /// Build + prefill an it2i/VQA prefix (image-conditioned or text-only) and return the cache, the
    /// last-position logits, and the next-token temporal index (`max(t)` — decode starts at `+1`).
    pub fn prefill_it2i_logits(
        &self,
        ids: &[i32],
        pixel_values: Option<&Tensor>,
        grids: &[(usize, usize)],
    ) -> CResult<(KvCache, Vec<f32>, usize)> {
        let (embeds, t, h, w) = self.build_it2i_prefix(ids, pixel_values, grids)?;
        let (cache, last, img_temporal) = self.prefill_prefix(&embeds, &t, &h, &w)?;
        Ok((cache, last, img_temporal - 1))
    }

    /// Build the it2i condition prefix from a `base_query` + per-image patch grids, replacing each
    /// `<image>` marker with `<img><IMG_CONTEXT>×n</img>` (n = merged-grid token count). Mirrors the
    /// reference `_build_it2i_query`.
    fn build_it2i_query_ids(
        &self,
        tokenizer: &SenseNovaTokenizer,
        base_query: &str,
        grids: &[(usize, usize)],
    ) -> Result<Vec<i32>> {
        let mut query = base_query.to_string();
        for &(gh, gw) in grids {
            let n = (gh / self.merge_size) * (gw / self.merge_size);
            let block = format!("<img>{}</img>", "<IMG_CONTEXT>".repeat(n));
            query = query.replacen("<image>", &block, 1);
        }
        tokenizer.encode_ids(&query, true)
    }

    /// Greedy/sampled understanding-path text decode from a prefilled cache. `first_logits` are the
    /// prefix's last-position logits; `t_idx` the prefix's max temporal index. Returns the generated
    /// token ids (stop ids excluded).
    pub fn decode_text(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: usize,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
    ) -> CResult<Vec<i32>> {
        self.backbone.generate(
            first_logits,
            cache,
            t_idx as i32,
            eos,
            max_new_tokens,
            sampler,
        )
    }

    /// VQA / understanding (`chat` / `answer_question`): image(s) + question → understanding-path AR
    /// text generation → answer. The question prefix (empty system message, `<image>` markers
    /// auto-prepended, no-think `<think></think>` primed) is built and prefilled exactly like the
    /// it2i condition prefix, then decoded to the `<|im_end|>` stop. `images` are decoded RGB
    /// `[3,H,W]` in `[0,1]` (sized to multiples of `patch·merge`); an empty slice gives a text-only
    /// question. Returns the decoded answer (special tokens stripped, trimmed).
    pub fn vqa(
        &self,
        tokenizer: &SenseNovaTokenizer,
        question: &str,
        images: &[Tensor],
        max_new_tokens: usize,
        sampler: Sampler,
    ) -> Result<String> {
        let mut pv_parts = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for img in images {
            let (p, g) = self.preprocess_image(img)?;
            pv_parts.push(p);
            grids.push(g);
        }
        let pixel_values = if pv_parts.is_empty() {
            None
        } else {
            let refs: Vec<&Tensor> = pv_parts.iter().collect();
            Some(Tensor::cat(&refs, 0)?)
        };

        // Auto-prepend `<image>` markers (the reference `chat` prepends one per missing image).
        let count = question.matches("<image>").count();
        let mut q = question.to_string();
        if images.len() > count {
            let pre = "<image>\n".repeat(images.len() - count);
            q = format!("{pre}{q}");
        }
        // Empty system message + a primed empty `<think></think>` block so the model answers
        // directly without a chain-of-thought (the reference `chat(think=False)`).
        let base = format!("{}<think>\n\n</think>\n\n", build_neo1_query(&q, ""));
        let ids = if images.is_empty() {
            tokenizer.encode_ids(&base, true)?
        } else {
            self.build_it2i_query_ids(tokenizer, &base, &grids)?
        };

        let (mut cache, last_logits, t_idx) =
            self.prefill_it2i_logits(&ids, pixel_values.as_ref(), &grids)?;
        let toks = self.decode_text(
            &last_logits,
            &mut cache,
            t_idx,
            &[tokens::IM_END],
            max_new_tokens,
            sampler,
        )?;
        let u32s: Vec<u32> = toks.iter().map(|&i| i as u32).collect();
        Ok(tokenizer.decode(&u32s, true)?.trim().to_string())
    }

    /// Re-encode a generated image through the **understanding** vision embedder and append it (plus
    /// the `</img>` token) to a text cache, so subsequent text generation attends to it (the
    /// reference `interleave_gen`'s inner `append_image_to_cache`). The generated `image` `[1,3,H,W]`
    /// is mapped model-space→`[0,1]` (`·0.5+0.5`) then ImageNet-normalised. Image tokens take temporal
    /// `t_idx+1` with their merged-grid `(h,w)`; `</img>` takes `t_idx+2`. Returns the next-token
    /// logits and the advanced `t_idx` (`+2`).
    pub fn append_generated_image(
        &self,
        image: &Tensor,
        token_h: usize,
        token_w: usize,
        t_idx: usize,
        cache: &mut KvCache,
    ) -> CResult<(Vec<f32>, usize)> {
        let (_, _, h, w) = image.dims4()?;
        let raw = image.affine(0.5, 0.5)?.reshape((3, h, w))?; // model-space [-1,1] → [0,1]
        let (patches, (gh, gw)) = self.preprocess_image(&raw)?;
        let vit = self.und_vision_features(&patches, &[(gh, gw)])?; // [n_img, H]
        let n_img = vit.dim(0)?;
        let hidden = vit.dim(1)?;
        let end = self
            .backbone
            .embed(&[self.img_end_id])?
            .reshape((1, hidden))?;
        let embeds = Tensor::cat(&[&vit, &end], 0)?.reshape((1, n_img + 1, hidden))?;

        let ti = t_idx as i32;
        let mut t = vec![ti + 1; n_img];
        t.push(ti + 2);
        let (hh, ww) = merged_grid_position_ids(n_img, token_h, token_w)?;
        let hs = self
            .backbone
            .forward_cached(&embeds, &t, &hh, &ww, Path::Und, cache, true)?;
        // Slice the kept `</img>` hidden row (index `n_img`) before `lm_head` (F-129).
        let last_hidden = hs.narrow(1, n_img, 1)?; // [1, 1, H]
        let logits = self.backbone.lm_head(&last_hidden)?;
        let vocab = logits.dim(2)?;
        let last = logits.reshape((vocab,))?.to_vec1::<f32>()?;
        Ok((last, t_idx + 2))
    }

    /// The dual-guidance denoise loop (`it2i_generate`'s body) shared by the interleave image steps.
    /// `cache_img` / `cache_uncond` (with their image-block temporal indices) are the optional
    /// image-condition / uncondition caches; the per-step blend follows the reference's
    /// `cfg_scale`/`img_cfg_scale` cases, then optional `cfg_norm`. Returns the final image
    /// `[1,3,H,W]`. Aborts on `cancel` and reports per-step [`Progress`].
    #[allow(clippy::too_many_arguments)]
    fn it2i_denoise(
        &self,
        cache_cond: &mut KvCache,
        cond_t: usize,
        mut cache_img: Option<&mut KvCache>,
        img_t: usize,
        mut cache_uncond: Option<&mut KvCache>,
        uncond_t: usize,
        width: usize,
        height: usize,
        base_noise: &Tensor,
        opts: &T2iOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let cell = self.cell();
        let token_h = height / cell;
        let token_w = width / cell;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let l = token_h * token_w;

        // The denoise guard's cache requirements must match the caller's `needs_*` flags (F-126).
        let (needs_img, needs_uncond) = it2i_cache_requirements(opts.cfg_scale, opts.img_cfg_scale);
        if needs_img && cache_img.is_none() {
            return Err(img_cache_err());
        }
        if needs_uncond && cache_uncond.is_none() {
            return Err(uncond_cache_err());
        }
        // CFG-Zero* is a T2I-only blend mode; reject it on the it2i/interleave path (F-131).
        if opts.cfg_norm == CfgNorm::CfgZeroStar {
            return Err(CandleError::Msg(
                "sensenova it2i: cfg_norm=cfg_zero_star is T2I-only — the it2i/interleave path \
                 supports only none/global/channel"
                    .into(),
            ));
        }

        let noise_scale = self.noise_scale_for(grid_h, grid_w);
        let mut image = (base_noise * noise_scale as f64)?;
        let steps = opts.num_steps;
        let timesteps = step_schedule(opts)?;
        let noise_embed = self.noise_scale_embed(noise_scale, l)?;

        let (cfg, img_cfg) = (opts.cfg_scale, opts.img_cfg_scale);
        let (i0, i1) = opts.cfg_interval;
        // RoPE tables + block mask are invariant across steps for a given cache — build once (F-139).
        let rm_cond = self.prepare_gen(token_h, token_w, cond_t, cache_cond.len())?;
        let rm_img = match cache_img.as_deref() {
            Some(c) => Some(self.prepare_gen(token_h, token_w, img_t, c.len())?),
            None => None,
        };
        let rm_uncond = match cache_uncond.as_deref() {
            Some(c) => Some(self.prepare_gen(token_h, token_w, uncond_t, c.len())?),
            None => None,
        };

        for i in 0..steps {
            if cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let t = timesteps[i];
            let t_next = timesteps[i + 1];
            // it2i CFG-interval gate: **exclusive** `(i0, i1)` OR an `i0 == 0` always-on override —
            // the reference `modeling_neo_chat.py` it2i gate (the T2I `denoise` uses the inclusive
            // form; they diverge only for a custom `cfg_interval`, F-130).
            let use_cfg = (t > i0 && t < i1) || i0 == 0.0;

            let (z, cond_emb) =
                self.step_cond_embeds(&image, grid_h, grid_w, l, t, &noise_embed)?;
            let out_cond = self.predict_v(&cond_emb, &rm_cond, cache_cond, &z, t, opts.t_eps)?;

            let mut v_pred = if !use_cfg || (cfg == 1.0 && img_cfg == 1.0) {
                out_cond.clone()
            } else if img_cfg == 1.0 {
                let rm_i = rm_img.as_ref().ok_or_else(img_cache_err)?;
                let c = cache_img.as_deref_mut().ok_or_else(img_cache_err)?;
                let oi = self.predict_v(&cond_emb, rm_i, c, &z, t, opts.t_eps)?;
                (&oi + ((&out_cond - &oi)? * cfg as f64)?)?
            } else if cfg == img_cfg {
                let rm_u = rm_uncond.as_ref().ok_or_else(uncond_cache_err)?;
                let c = cache_uncond.as_deref_mut().ok_or_else(uncond_cache_err)?;
                let ou = self.predict_v(&cond_emb, rm_u, c, &z, t, opts.t_eps)?;
                (&ou + ((&out_cond - &ou)? * cfg as f64)?)?
            } else {
                let oi = {
                    let rm_i = rm_img.as_ref().ok_or_else(img_cache_err)?;
                    let c = cache_img.as_deref_mut().ok_or_else(img_cache_err)?;
                    self.predict_v(&cond_emb, rm_i, c, &z, t, opts.t_eps)?
                };
                let ou = {
                    let rm_u = rm_uncond.as_ref().ok_or_else(uncond_cache_err)?;
                    let c = cache_uncond.as_deref_mut().ok_or_else(uncond_cache_err)?;
                    self.predict_v(&cond_emb, rm_u, c, &z, t, opts.t_eps)?
                };
                let a = ((&out_cond - &oi)? * cfg as f64)?;
                let b = ((&oi - &ou)? * img_cfg as f64)?;
                ((&ou + &a)? + &b)?
            };

            if (cfg > 1.0 || img_cfg > 1.0) && use_cfg {
                v_pred = apply_cfg_norm(v_pred, &out_cond, opts.cfg_norm)?;
            }

            image = unpatchify(&euler_step(&v_pred, &z, t, t_next)?, cell, token_h, token_w)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total: steps as u32,
            });
        }
        Ok(image)
    }

    /// Interleaved text-image generation (`interleave_gen`) — the **Document Studio** deliverable. A
    /// single rollout that alternates understanding-path text generation and gen-path flow-matching
    /// image generation: text streams until the model emits `<img>`, an image is generated (3-cache
    /// CFG: condition / text-uncondition / image-uncondition) and re-encoded back into the text
    /// caches, then text resumes. `input_images` are optional source images (`[3,H,W]` in `[0,1]`,
    /// 32-aligned); `system_message` is normally [`INTERLEAVE_SYSTEM_MESSAGE`]. Returns the composed
    /// text (with `<image>` placeholders) and the generated images in order.
    #[allow(clippy::too_many_arguments)]
    pub fn interleave_gen(
        &self,
        tokenizer: &SenseNovaTokenizer,
        prompt: &str,
        input_images: &[Tensor],
        width: usize,
        height: usize,
        opts: &T2iOptions,
        system_message: &str,
        max_new_tokens: usize,
        max_images: usize,
        cancel: &CancelFlag,
    ) -> Result<InterleaveOutput> {
        let cell = self.cell();
        if !width.is_multiple_of(cell) || !height.is_multiple_of(cell) {
            return Err(CandleError::Msg(format!(
                "sensenova interleave: width/height must be multiples of {cell}, got {width}x{height}"
            )));
        }
        let token_h = height / cell;
        let token_w = width / cell;

        // Source images (optional).
        let mut pv_parts = Vec::with_capacity(input_images.len());
        let mut grids = Vec::with_capacity(input_images.len());
        for img in input_images {
            let (p, g) = self.preprocess_image(img)?;
            pv_parts.push(p);
            grids.push(g);
        }
        let pixel_values = if pv_parts.is_empty() {
            None
        } else {
            let refs: Vec<&Tensor> = pv_parts.iter().collect();
            Some(Tensor::cat(&refs, 0)?)
        };

        // ---- Three prefixes / caches ----
        let mut cond_query = build_neo1_query(prompt, system_message);
        if !opts.think_mode {
            cond_query.push_str("<think>\n\n</think>\n\n");
        }
        let cond_ids = if input_images.is_empty() {
            tokenizer.encode_ids(&cond_query, true)?
        } else {
            self.build_it2i_query_ids(tokenizer, &cond_query, &grids)?
        };
        let (mut cache_cond, cond_logits, mut t_cond) =
            self.prefill_it2i_logits(&cond_ids, pixel_values.as_ref(), &grids)?;

        let tu_query = build_neo1_query(&"<image>".repeat(input_images.len()), "");
        let tu_ids = if input_images.is_empty() {
            tokenizer.encode_ids(&tu_query, true)?
        } else {
            self.build_it2i_query_ids(tokenizer, &tu_query, &grids)?
        };
        let (mut cache_tu, _, mut t_tu) =
            self.prefill_it2i_logits(&tu_ids, pixel_values.as_ref(), &grids)?;

        let iu_query = format!("{}<img>", build_neo1_query("", ""));
        let iu_ids = tokenizer.encode_ids(&iu_query, true)?;
        let (mut cache_iu, _, iu_max) = self.prefill_it2i_logits(&iu_ids, None, &[])?;

        let mut text = String::new();
        let mut images: Vec<Tensor> = Vec::new();
        let mut total_tokens = 0usize;
        let mut next = argmax(&cond_logits);

        loop {
            // ---- Text generation on the condition cache ----
            let mut gen_tokens = Vec::new();
            let mut hit_max = false;
            loop {
                if cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                if next == tokens::IM_END || next == self.img_start_id {
                    break;
                }
                gen_tokens.push(next);
                total_tokens += 1;
                let logits =
                    self.backbone
                        .decode_logits(next, (t_cond + 1) as i32, &mut cache_cond)?;
                t_cond += 1;
                next = argmax(&logits);
                if total_tokens >= max_new_tokens {
                    hit_max = true;
                    break;
                }
            }
            if !gen_tokens.is_empty() {
                let u32s: Vec<u32> = gen_tokens.iter().map(|&i| i as u32).collect();
                text.push_str(&tokenizer.decode(&u32s, true)?);
            }
            if next == tokens::IM_END || hit_max || images.len() >= max_images {
                break;
            }
            if next != self.img_start_id {
                break;
            }

            // ---- Image generation ----
            text.push_str("<image>");
            // Append `<img>` to the condition + text-uncondition caches.
            self.backbone
                .decode_logits(self.img_start_id, (t_cond + 1) as i32, &mut cache_cond)?;
            t_cond += 1;
            self.backbone
                .decode_logits(self.img_start_id, (t_tu + 1) as i32, &mut cache_tu)?;
            t_tu += 1;

            let base_noise = gaussian(
                (1, 3, height, width),
                opts.seed.wrapping_add(images.len() as u64),
                &self.device,
            )?;
            let mut sink = |_p: Progress| {};
            let image = self.it2i_denoise(
                &mut cache_cond,
                t_cond + 1,
                Some(&mut cache_tu),
                t_tu + 1,
                Some(&mut cache_iu),
                iu_max + 1,
                width,
                height,
                &base_noise,
                opts,
                cancel,
                &mut sink,
            )?;

            // Re-encode the generated image back into the condition + text-uncondition caches.
            let (cond_next, nt_cond) =
                self.append_generated_image(&image, token_h, token_w, t_cond, &mut cache_cond)?;
            t_cond = nt_cond;
            let (_, nt_tu) =
                self.append_generated_image(&image, token_h, token_w, t_tu, &mut cache_tu)?;
            t_tu = nt_tu;
            images.push(image);
            next = argmax(&cond_next);
        }

        Ok(InterleaveOutput { text, images })
    }
}

/// `(needs_img, needs_uncond)`: which extra caches [`T2iModel::it2i_denoise`] requires for the given
/// guidance scales, so the denoise guard and the interleave cache construction agree (F-126).
fn it2i_cache_requirements(cfg_scale: f32, img_cfg_scale: f32) -> (bool, bool) {
    let needs_cfg = !(cfg_scale == 1.0 && img_cfg_scale == 1.0);
    let needs_img = needs_cfg && (img_cfg_scale == 1.0 || cfg_scale != img_cfg_scale);
    let needs_uncond = needs_cfg && img_cfg_scale != 1.0;
    (needs_img, needs_uncond)
}

fn img_cache_err() -> CandleError {
    CandleError::Msg(
        "sensenova it2i: image-CFG guidance needs an image-conditioned cache, but none was supplied"
            .into(),
    )
}

fn uncond_cache_err() -> CandleError {
    CandleError::Msg("sensenova it2i: guidance needs an uncond cache, but none was supplied".into())
}

/// Apply the post-blend `cfg_norm` rescale (`it2i_generate`): clamp the guided velocity's norm to the
/// condition velocity's (global = whole-tensor, channel = per-token). `None` is a no-op; `CfgZeroStar`
/// is rejected (a T2I-only blend mode, not a post-rescale).
fn apply_cfg_norm(v: Tensor, out_cond: &Tensor, norm: CfgNorm) -> CResult<Tensor> {
    match norm {
        CfgNorm::None => Ok(v),
        CfgNorm::Global => {
            let nc = frobenius(out_cond)?;
            let nv = frobenius(&v)?;
            let s = (nc / (nv + 1e-8)).clamp(0.0, 1.0);
            v * s as f64
        }
        CfgNorm::Channel => {
            let nc = l2_last(out_cond)?;
            let nv = l2_last(&v)?;
            let ratio = nc.broadcast_div(&(nv + 1e-8)?)?;
            let s = ratio.clamp(0f32, 1f32)?;
            v.broadcast_mul(&s)
        }
        CfgNorm::CfgZeroStar => Err(candle_gen::candle_core::Error::Msg(
            "sensenova it2i: cfg_norm=cfg_zero_star is T2I-only".into(),
        )),
    }
}

/// `‖x‖₂` over the whole tensor (the reference `torch.norm(v, dim=(1,2))` for batch 1).
fn frobenius(x: &Tensor) -> CResult<f32> {
    Ok(x.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt())
}

/// Per-token L2 norm over the last axis, keeping dims: `[1,L,D] → [1,L,1]`.
fn l2_last(x: &Tensor) -> CResult<Tensor> {
    let last = x.rank() - 1;
    x.sqr()?.sum_keepdim(last)?.sqrt()
}

/// 2D position ids for an appended image's `n_img` merged-grid tokens followed by the trailing
/// `img_end` token: row-major `(i / token_w, i % token_w)` for the image rows, then `(0, 0)` for the
/// end token. Cross-checks the merged-grid token count against `token_h × token_w` (F-135).
fn merged_grid_position_ids(
    n_img: usize,
    token_h: usize,
    token_w: usize,
) -> CResult<(Vec<i32>, Vec<i32>)> {
    if n_img != token_h * token_w {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "sensenova append_generated_image: re-encoded image has {n_img} merged tokens but the \
             caller's grid is {token_h}×{token_w} ({} expected)",
            token_h * token_w
        )));
    }
    let mut hh = Vec::with_capacity(n_img + 1);
    let mut ww = Vec::with_capacity(n_img + 1);
    for i in 0..n_img {
        hh.push((i / token_w) as i32);
        ww.push((i % token_w) as i32);
    }
    hh.push(0);
    ww.push(0);
    Ok((hh, ww))
}

/// `smart_resize` (Qwen2.5-VL, the vendored `utils.smart_resize`): round `height`/`width` to
/// multiples of `factor` (use `patch·merge = 32`) with total pixels held in `[min_pixels,
/// max_pixels]`. Returns `(height, width)`.
pub fn smart_resize(
    height: i32,
    width: i32,
    factor: i32,
    min_pixels: i64,
    max_pixels: i64,
) -> (i32, i32) {
    let round_by = |n: f64| ((n / factor as f64).round() as i32) * factor;
    let floor_by = |n: f64| ((n / factor as f64).floor() as i32) * factor;
    let ceil_by = |n: f64| ((n / factor as f64).ceil() as i32) * factor;
    let (hf, wf) = (height as f64, width as f64);
    let mut h_bar = factor.max(round_by(hf));
    let mut w_bar = factor.max(round_by(wf));
    let area = (h_bar as i64) * (w_bar as i64);
    if area > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = factor.max(floor_by(hf / beta));
        w_bar = factor.max(floor_by(wf / beta));
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = ceil_by(hf * beta);
        w_bar = ceil_by(wf * beta);
    }
    (h_bar, w_bar)
}

/// The flow-matching step schedule: `linspace(0, 1, steps+1)`, optionally through
/// [`apply_time_schedule`]. The boundary grid is ascending `0 → 1` (length `steps + 1`).
fn step_schedule(opts: &T2iOptions) -> CResult<Vec<f32>> {
    let steps = opts.num_steps;
    if steps == 0 {
        return Err(candle_gen::candle_core::Error::Msg(
            "sensenova: num_steps must be >= 1".into(),
        ));
    }
    let lin: Vec<f32> = (0..=steps).map(|i| i as f32 / steps as f32).collect();
    if opts.enable_timestep_shift {
        let lin_t = Tensor::from_vec(lin, (steps + 1,), &Device::Cpu)?;
        apply_time_schedule(&lin_t, opts.timestep_shift)?.to_vec1::<f32>()
    } else {
        Ok(lin)
    }
}

/// Plain CFG velocity blend: `v_uncond + scale·(v_cond − v_uncond)` (`t2i_generate`'s `cfg_norm=none`,
/// the only mode the Generator path uses).
fn cfg_blend(v_cond: &Tensor, v_uncond: &Tensor, scale: f32) -> CResult<Tensor> {
    let diff = (v_cond - v_uncond)?;
    v_uncond + (diff * scale as f64)?
}

/// sc-3673 deterministic, launch-portable standard-normal noise: N(0,1) from a fixed-algorithm CPU
/// RNG (`StdRng`, ChaCha) seeded by `seed`, moved to the device. (The candle slice uses `StdRng`, not
/// the mlx `SplitMix64` Box–Muller — cross-backend pixel-equality is not a goal; per-seed
/// reproducibility within candle is.)
fn gaussian(shape: (usize, usize, usize, usize), seed: u64, device: &Device) -> CResult<Tensor> {
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    let mut rng = StdRng::seed_from_u64(seed);
    let v: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Tensor::from_vec(v, shape, &Device::Cpu)?.to_device(device)
}

/// Map a model-space image `[1,3,H,W]` (≈ `[-1,1]`) to an RGB8 [`Image`] (`x·0.5+0.5`, clamp, ×255).
pub fn tensor_to_image(img: &Tensor) -> CResult<Image> {
    let img = ((img * 0.5)? + 0.5)?.clamp(0f32, 1f32)?;
    let img = (img * 255.)?
        .to_dtype(candle_gen::candle_core::DType::U8)?
        .i(0)?
        .to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "sensenova: expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn step_schedule_guards_zero_and_applies_shift() {
        let base = T2iOptions::default();
        assert!(step_schedule(&T2iOptions {
            num_steps: 0,
            ..base
        })
        .is_err());

        let linear = step_schedule(&T2iOptions {
            num_steps: 4,
            enable_timestep_shift: false,
            ..base
        })
        .unwrap();
        assert_eq!(linear, vec![0.0, 0.25, 0.5, 0.75, 1.0]);

        let shifted = step_schedule(&T2iOptions {
            num_steps: 4,
            enable_timestep_shift: true,
            timestep_shift: 3.0,
            ..base
        })
        .unwrap();
        assert_eq!(shifted.first(), Some(&0.0));
        assert_eq!(shifted.last(), Some(&1.0));
    }

    #[test]
    fn cfg_blend_is_linear_extrapolation() {
        let dev = Device::Cpu;
        let v_cond = Tensor::from_vec(vec![2.0f32, 4.0], (1, 1, 2), &dev).unwrap();
        let v_uncond = Tensor::from_vec(vec![1.0f32, 1.0], (1, 1, 2), &dev).unwrap();
        // v_uncond + 3·(v_cond − v_uncond): 1 + 3·(2−1) = 4 ; 1 + 3·(4−1) = 10
        let out = cfg_blend(&v_cond, &v_uncond, 3.0).unwrap();
        assert_eq!(flat(&out), vec![4.0, 10.0]);
    }

    #[test]
    fn gaussian_is_deterministic_per_seed() {
        let dev = Device::Cpu;
        let a = gaussian((1, 3, 4, 4), 42, &dev).unwrap();
        let b = gaussian((1, 3, 4, 4), 42, &dev).unwrap();
        let c = gaussian((1, 3, 4, 4), 43, &dev).unwrap();
        assert_eq!(flat(&a), flat(&b), "same seed → same noise");
        assert_ne!(flat(&a), flat(&c), "different seed → different noise");
        assert_eq!(a.dims(), &[1, 3, 4, 4]);
    }

    #[test]
    fn smart_resize_upscales_and_keeps_in_range() {
        // 100×100 rounds to 96×96 (< min) → upscaled to the 256² bucket.
        assert_eq!(smart_resize(100, 100, 32, 65536, 4_194_304), (256, 256));
        // In-range stays put; non-multiples round to the nearest factor.
        assert_eq!(smart_resize(512, 512, 32, 65536, 4_194_304), (512, 512));
        assert_eq!(smart_resize(500, 500, 32, 65536, 4_194_304), (512, 512));
    }

    #[test]
    fn interleave_resolutions_are_32_aligned_and_looked_up() {
        assert_eq!(interleave_resolution_for("16:9"), Some((2048, 1152)));
        assert_eq!(interleave_resolution_for("1:1"), Some((1536, 1536)));
        assert_eq!(interleave_resolution_for("nope"), None);
        for (_, (w, h)) in INTERLEAVE_RESOLUTIONS {
            assert_eq!(w % 32, 0, "interleave bucket width not 32-aligned");
            assert_eq!(h % 32, 0, "interleave bucket height not 32-aligned");
        }
    }

    #[test]
    fn merged_grid_position_ids_rows_and_guard() {
        // Row-major (i/token_w, i%token_w) for the grid, then (0,0) for the trailing `img_end`.
        let (hh, ww) = merged_grid_position_ids(6, 2, 3).unwrap();
        assert_eq!(hh, vec![0, 0, 0, 1, 1, 1, 0]);
        assert_eq!(ww, vec![0, 1, 2, 0, 1, 2, 0]);
        // A token count that doesn't match the caller's grid is rejected (F-135).
        assert!(merged_grid_position_ids(5, 2, 3).is_err());
        assert!(merged_grid_position_ids(6, 3, 3).is_err());
    }

    #[test]
    fn it2i_cache_requirements_match_guidance_scales() {
        assert_eq!(it2i_cache_requirements(1.0, 1.0), (false, false)); // no guidance
        assert_eq!(it2i_cache_requirements(4.0, 1.0), (true, false)); // image-CFG only
        assert_eq!(it2i_cache_requirements(4.0, 4.0), (false, true)); // uncond only
        assert_eq!(it2i_cache_requirements(4.0, 2.0), (true, true)); // dual guidance
        assert!(img_cache_err().to_string().contains("image"));
        assert!(uncond_cache_err().to_string().contains("uncond"));
    }

    #[test]
    fn apply_cfg_norm_none_is_noop_and_global_never_amplifies() {
        let dev = Device::Cpu;
        let v = Tensor::from_vec(vec![2.0f32, 4.0], (1, 1, 2), &dev).unwrap();
        let cond = Tensor::from_vec(vec![1.0f32, 1.0], (1, 1, 2), &dev).unwrap();
        // None is a pass-through.
        assert_eq!(
            flat(&apply_cfg_norm(v.clone(), &cond, CfgNorm::None).unwrap()),
            vec![2.0, 4.0]
        );
        // Global clamps the blended velocity's norm to ≤ the condition velocity's norm.
        let g = apply_cfg_norm(v, &cond, CfgNorm::Global).unwrap();
        let gn = (flat(&g).iter().map(|x| x * x).sum::<f32>()).sqrt();
        let cn = (2.0f32).sqrt();
        assert!(
            gn <= cn + 1e-4,
            "global-norm output {gn} exceeds cond norm {cn}"
        );
        // CFG-Zero* is rejected on the it2i/interleave path (T2I-only blend mode).
        assert!(apply_cfg_norm(cond.clone(), &cond, CfgNorm::CfgZeroStar).is_err());
    }
}
