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
use crate::text::{
    build_neo1_query, image_indexes, text_indexes, SenseNovaTokenizer, SYSTEM_MESSAGE_FOR_GEN,
};
use crate::vision::NeoVisionEmbedder;

/// Knobs for [`T2iModel::generate`] (the T2I subset of `t2i_generate`'s arguments).
///
/// The `Generator` contract only ever drives **plain** CFG (`cfg_norm = none` in the reference), so
/// the candle slice omits the Global/Channel/CFG-Zero* blend modes (mlx `T2iModel`-only diagnostics).
#[derive(Clone, Copy, Debug)]
pub struct T2iOptions {
    pub cfg_scale: f32,
    pub cfg_interval: (f32, f32),
    pub num_steps: usize,
    pub timestep_shift: f32,
    pub enable_timestep_shift: bool,
    pub t_eps: f32,
    pub seed: u64,
}

impl Default for T2iOptions {
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            cfg_interval: (0.0, 1.0),
            num_steps: 30,
            timestep_shift: 1.0,
            enable_timestep_shift: true,
            t_eps: 0.02,
            seed: 0,
        }
    }
}

/// The T2I model: the dual-path backbone plus the flow-matching generation modules.
pub struct T2iModel {
    backbone: Qwen3Backbone,
    gen_vision: NeoVisionEmbedder,
    fm_head: FmHead,
    timestep_embedder: TimestepEmbedder,
    noise_scale_embedder: Option<TimestepEmbedder>,
    patch_size: usize,
    merge_size: usize,
    noise_scale: f32,
    noise_scale_mode: String,
    noise_scale_base_image_seq_len: f32,
    noise_scale_max_value: f32,
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
        Ok(Self {
            backbone: Qwen3Backbone::from_weights(vb, cfg, "language_model")?,
            gen_vision: NeoVisionEmbedder::from_weights(
                vb,
                cfg,
                "fm_modules.vision_model_mot_gen.embeddings",
            )?,
            fm_head: FmHead::from_weights(vb, "fm_modules.fm_head")?,
            timestep_embedder: TimestepEmbedder::from_weights(vb, "fm_modules.timestep_embedder")?,
            noise_scale_embedder,
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
}
