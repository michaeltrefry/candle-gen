//! The candle Kolors **txt2img** pipeline — ChatGLM3-6B prompt encode → the SDXL-family Kolors UNet
//! (real CFG over the leading-Euler schedule) → the SDXL VAE, driven through the backend-neutral
//! [`gen_core::Generator`] contract and parity-matched to the macOS `mlx-gen-kolors` provider.
//!
//! Parity choices (grounded in the mlx `model.rs` + diffusers `KolorsPipeline`):
//! - **Conditioning**: each prompt is tokenized to the fixed 256-len left-padded form and run through
//!   ChatGLM3 with its own padding mask + `position_ids`; `context = hidden[-2]` `[1, 256, 4096]`,
//!   `pooled = hidden[-1]` last-position `[1, 4096]`. The two prompts' results are CFG-batched
//!   `[uncond, cond]` (candle's chunk convention), so the encode itself stays B==1.
//! - **`time_ids`** = `(H, W, 0, 0, H, W)` per row (SDXL `_get_add_time_ids`, original == target, no crop).
//! - **Sampler**: the leading EulerDiscrete over the 1100-step `scaled_linear` schedule
//!   ([`crate::sampler`]); `scale_model_input` divides by `√(σ²+1)`, the Euler step adds `ε·(σ_next−σ)`.
//! - **CFG**: `pred = uncond + g·(cond − uncond)`; `g ≤ 1` skips the negative branch (single forward).
//! - **Deterministic seeding (sc-3673)**: initial noise from a fixed-algorithm CPU RNG (`StdRng`,
//!   ChaCha) seeded by `seed`, moved to the device — launch-portable per seed.
//!
//! Components load at **f32** (the candle port recipe — single matmul dtype; = mlx's "f32 activations
//! over bf16 weights"); the SDXL VAE is f32-stable so it needs no fp16-fix.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::{
    schedule_sigmas, AlphaSchedule, DiscreteModelSampling, Scheduler, Solver,
};
use candle_gen::gen_core::{self, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::chatglm3::ChatGlmModel;
use crate::config::{ChatGlmConfig, DEFAULT_GUIDANCE, DEFAULT_STEPS};
use crate::sampler::KolorsEulerSampler;
use crate::tokenizer::KolorsTokenizer;
use crate::unet::KolorsUNet;

/// diffusers SDXL VAE `scaling_factor` (Kolors reuses it). The latents are divided by this before
/// decode — the diffusers-correct SDXL value (NOT candle's hardcoded SD1.5 0.18215). `pub(crate)` so
/// the IP-Adapter provider (sc-5488) shares the exact decode scale.
pub(crate) const VAE_SCALE: f64 = 0.13025;

/// Kolors' `scaled_linear` β endpoints + train-step count — the diffusers `EulerDiscreteScheduler`
/// config the native [`KolorsEulerSampler`](crate::sampler) is built from (β₁ = **0.014**, NOT SDXL's
/// 0.012; N = **1100**, NOT SDXL's 1000). The curated [`DiscreteModelSampling`] σ-table (sc-7124) is
/// built from these same values so the ε/DDPM menu integrates over Kolors' own noise schedule.
const KOLORS_BETA_START: f32 = 0.00085;
const KOLORS_BETA_END: f32 = 0.014;
const KOLORS_TRAIN_STEPS: usize = crate::sampler::NUM_TRAIN_TIMESTEPS;

/// A light pipeline handle: the snapshot `root` and compute device. Heavy components load via
/// [`load_components`](Self::load_components) and are owned/cached by the generator.
pub(crate) struct Pipeline {
    root: PathBuf,
    device: Device,
}

/// The loaded Kolors components, `Arc`-shared so the generator can cache them across `generate` calls.
/// All four are immutable in the forward (no per-call mutable state), so no interior locking is needed.
#[derive(Clone)]
pub(crate) struct Components {
    tokenizer: Arc<KolorsTokenizer>,
    chatglm: Arc<ChatGlmModel>,
    unet: Arc<KolorsUNet>,
    vae: Arc<AutoEncoderKL>,
}

impl Pipeline {
    pub(crate) fn load(root: &Path, device: &Device) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    /// Load the four heavy components from the Kolors-diffusers snapshot (`tokenizer/`, `text_encoder/`
    /// ChatGLM3-6B, `unet/` SDXL-family UNet, `vae/` SDXL VAE), all at f32.
    pub(crate) fn load_components(&self) -> Result<Components> {
        let tokenizer = KolorsTokenizer::from_dir(self.root.join("tokenizer"))?;
        let chatglm = ChatGlmModel::new(
            ChatGlmConfig::chatglm3_6b(),
            self.f32_vb(&self.root.join("text_encoder"))?,
        )?;
        let unet = KolorsUNet::new(self.f32_vb(&self.root.join("unet"))?, false)?;
        let vae = AutoEncoderKL::new(
            self.f32_vb(&self.root.join("vae"))?,
            3,
            3,
            sdxl_vae_config(),
        )?;
        Ok(Components {
            tokenizer: Arc::new(tokenizer),
            chatglm: Arc::new(chatglm),
            unet: Arc::new(unet),
            vae: Arc::new(vae),
        })
    }

    /// mmap an f32 [`VarBuilder`] over every `.safetensors` in `dir` (the ChatGLM3 encoder + UNet ship
    /// sharded or single-file).
    fn f32_vb(&self, dir: &Path) -> Result<VarBuilder<'static>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| CandleError::Msg(format!("kolors: read {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "kolors: no .safetensors found in {} (expected a Kolors-diffusers snapshot)",
                dir.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; the standard candle loading path.
        Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &self.device)? })
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. One image per `req.count` (each at seed `base_seed + index`).
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS as usize);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let use_guide = guidance > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let (h, w) = (req.height, req.width);

        // sc-7124 (epic 7114 P4): a curated solver name (≠ the native `euler_discrete` default / None)
        // routes the unified `Sampler` over `DiscreteModelSampling` (EPS) as a NEW path. The native
        // leading-Euler default stays byte-exact (N1) — Kolors' `steps_offset=1` leading timesteps can't
        // be bit-reproduced by `DiscreteModelSampling::timestep`, so this is ADDITIVE, not a replacement.
        let curated: Option<&str> = req
            .sampler
            .as_deref()
            .filter(|n| Solver::from_name(n).is_some() && *n != crate::config::DEFAULT_SAMPLER);

        let sampler = KolorsEulerSampler::new(steps).map_err(CandleError::Msg)?;

        // Conditioning is seed-independent — encode once. CFG batch is [uncond, cond] (candle's chunk
        // order); without guidance only the positive branch is built.
        let (pos_ctx, pos_pooled) = self.encode(components, &req.prompt)?;
        let (context, pooled) = if use_guide {
            let (neg_ctx, neg_pooled) = self.encode(components, negative)?;
            (
                Tensor::cat(&[&neg_ctx, &pos_ctx], 0)?,
                Tensor::cat(&[&neg_pooled, &pos_pooled], 0)?,
            )
        } else {
            (pos_ctx, pos_pooled)
        };
        let batch = if use_guide { 2 } else { 1 };
        let time_ids = self.build_time_ids(batch, h, w)?;

        let (lat_h, lat_w) = ((h / 8) as usize, (w / 8) as usize);
        let total = sampler.num_steps() as u32;
        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = image_seed(base_seed, index);
            let noise = self.initial_noise(seed, lat_h, lat_w)?;

            let latents = if let Some(name) = curated {
                self.denoise_curated(
                    req,
                    name,
                    &noise,
                    components,
                    &context,
                    &pooled,
                    &time_ids,
                    steps,
                    use_guide,
                    guidance,
                    seed,
                    on_progress,
                )?
            } else {
                let mut latents = (&noise * sampler.init_noise_sigma() as f64)?;
                for i in 0..sampler.num_steps() {
                    if req.cancel.is_cancelled() {
                        return Err(CandleError::Canceled);
                    }
                    let scaled = (&latents / sampler.scale_in(i) as f64)?;
                    let model_in = if use_guide {
                        Tensor::cat(&[&scaled, &scaled], 0)?
                    } else {
                        scaled
                    };
                    let eps = components.unet.forward(
                        &model_in,
                        sampler.timestep(i) as f64,
                        &context,
                        &pooled,
                        &time_ids,
                    )?;
                    let eps = if use_guide {
                        let ch = eps.chunk(2, 0)?;
                        let (uncond, cond) = (&ch[0], &ch[1]);
                        (uncond + ((cond - uncond)? * guidance as f64)?)?
                    } else {
                        eps
                    };
                    latents = (&latents + (eps * sampler.step_dt(i) as f64)?)?;
                    on_progress(Progress::Step {
                        current: i as u32 + 1,
                        total,
                    });
                }
                latents
            };

            on_progress(Progress::Decoding);
            images.push(self.decode(&components.vae, &latents)?);
        }
        Ok(images)
    }

    /// The **curated** ε/DDPM denoise (epic 7114 P4, sc-7124) — an ADDITIVE option alongside the native
    /// leading-Euler default. Drives the unified [`gen_core::sampling`] solver menu (`euler` /
    /// `euler_ancestral` / `heun` / `dpmpp_2m` / `dpmpp_sde` / `uni_pc` / `lcm` / `ddim`) over a
    /// [`DiscreteModelSampling`] (Kolors ε-prediction, `scaled_linear` β over the 1100 train steps), with
    /// the `scheduler` axis (`normal` default / `karras` / `sgm_uniform` / …) picking the σ schedule via
    /// [`candle_gen::resolve_schedule`]. Latents live in k-diffusion VE σ-space (prior = unit noise ·
    /// σ_max), kept f32 like the native path; the [`DiscreteModelSampling`] recombines ε → x0 and supplies
    /// the `1/√(σ²+1)` input scaling, so the `predict` closure just runs the UNet + CFG and returns raw ε.
    ///
    /// The native leading-Euler default is untouched, so this never affects the N1 default-parity gate —
    /// Kolors' `steps_offset=1` leading timesteps aren't bit-reproducible by `DiscreteModelSampling`, so a
    /// curated request is its own (ComfyUI-style trailing/normal) path, not a re-derivation of the default.
    #[allow(clippy::too_many_arguments)]
    fn denoise_curated(
        &self,
        req: &GenerationRequest,
        sampler: &str,
        init: &Tensor,
        components: &Components,
        context: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
        steps: usize,
        use_guide: bool,
        guidance: f32,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let sched =
            AlphaSchedule::scaled_linear(KOLORS_TRAIN_STEPS, KOLORS_BETA_START, KOLORS_BETA_END)
                .map_err(|e| CandleError::Msg(format!("kolors curated schedule: {e}")))?;
        let ms = DiscreteModelSampling::sdxl(&sched);
        // Native curated schedule = ComfyUI's default (`normal`); the scheduler axis overrides it.
        let native = schedule_sigmas(Scheduler::Normal, &ms, steps);
        let sigmas = candle_gen::resolve_schedule(req.scheduler.as_deref(), &ms, steps, &native);
        // VE prior: unit noise · σ_max (sigmas[0]); kept f32 through the sampler.
        let latents = (init * sigmas[0] as f64)?;
        let out = candle_gen::run_curated_sampler(
            Some(sampler),
            &ms,
            &sigmas,
            latents,
            seed,
            &req.cancel,
            on_progress,
            |x_in, t| -> Result<Tensor> {
                // `x_in` is already `1/√(σ²+1)`-scaled by `denoise()`; `t` is the nearest training-step
                // index the UNet embeds. CFG batches/combines exactly like the native leading-Euler path.
                let model_in = if use_guide {
                    Tensor::cat(&[x_in, x_in], 0)?
                } else {
                    x_in.clone()
                };
                let eps = components
                    .unet
                    .forward(&model_in, t as f64, context, pooled, time_ids)?;
                let eps = if use_guide {
                    let ch = eps.chunk(2, 0)?;
                    let (uncond, cond) = (&ch[0], &ch[1]);
                    (uncond + ((cond - uncond)? * guidance as f64)?)?
                } else {
                    eps
                };
                // Raw ε in f32 so the DiscreteModelSampling x0 recombine + solver math stay f32.
                Ok(eps.to_dtype(DType::F32)?)
            },
        )?;
        // The shared `decode` consumes the compute dtype (f32 for Kolors), like the native latents.
        Ok(out.to_dtype(DType::F32)?)
    }

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])` via the ChatGLM3 encoder.
    fn encode(&self, components: &Components, prompt: &str) -> Result<(Tensor, Tensor)> {
        let tokens = components.tokenizer.encode(prompt)?;
        Ok(components.chatglm.encode_prompt(&tokens)?)
    }

    /// The SDXL micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)` per row, f32 `[batch, 6]`.
    fn build_time_ids(&self, batch: usize, height: u32, width: u32) -> Result<Tensor> {
        let (hf, wf) = (height as f32, width as f32);
        let row = [hf, wf, 0.0, 0.0, hf, wf];
        let mut v = Vec::with_capacity(batch * 6);
        for _ in 0..batch {
            v.extend_from_slice(&row);
        }
        Ok(Tensor::from_vec(v, (batch, 6), &self.device)?)
    }

    /// sc-3673 deterministic, launch-portable initial noise `[1, 4, lat_h, lat_w]`: N(0,1) from a
    /// fixed-algorithm CPU RNG seeded by `seed`, moved to the device.
    fn initial_noise(&self, seed: u64, lat_h: usize, lat_w: usize) -> Result<Tensor> {
        let n = 4 * lat_h * lat_w;
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        Ok(Tensor::from_vec(noise, (1, 4, lat_h, lat_w), &Device::Cpu)?.to_device(&self.device)?)
    }

    /// VAE-decode latents `[1, 4, H/8, W/8]` → an RGB8 [`Image`] (un-scale by [`VAE_SCALE`],
    /// `x/2 + 0.5`, clamp, ×255).
    fn decode(&self, vae: &AutoEncoderKL, latents: &Tensor) -> Result<Image> {
        let unscaled = (latents / VAE_SCALE)?;
        let img = vae.decode(&unscaled)?;
        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?
            .to_dtype(DType::U8)?
            .i(0)?
            .to_device(&Device::Cpu)?;
        let (c, h, w) = img.dims3()?;
        if c != 3 {
            return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
        }
        let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        Ok(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        })
    }
}

/// The SDXL VAE config (`stabilityai/stable-diffusion-xl-base-1.0/vae/config.json`) — Kolors reuses it.
/// `pub(crate)` so the IP-Adapter provider (sc-5488) builds the identical VAE.
pub(crate) fn sdxl_vae_config() -> AutoEncoderKLConfig {
    AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

/// Per-image seed within a batch: image `index` renders at `base_seed + index` (wrapping), so the
/// *n*-th image reproduces in isolation at that derived seed (mlx `seed + i`).
pub(crate) fn image_seed(base_seed: u64, index: u32) -> u64 {
    base_seed.wrapping_add(index as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_seed_is_base_plus_index() {
        assert_eq!(image_seed(42, 0), 42);
        assert_eq!(image_seed(42, 7), 49);
        assert_eq!(image_seed(u64::MAX, 1), 0);
    }

    #[test]
    fn sdxl_vae_config_pins_canonical_values() {
        let c = sdxl_vae_config();
        assert_eq!(c.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(c.latent_channels, 4);
        assert_eq!(c.norm_num_groups, 32);
        assert!(c.use_quant_conv);
        assert!(c.use_post_quant_conv);
    }
}
