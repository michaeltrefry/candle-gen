//! The candle SDXL **txt2img** pipeline (sc-3675) — the proven epic-3494 prototype
//! (`D:\sceneworks-candle-spike\src\bin\candle_sdxl.rs`) lifted out of its standalone CLI/PNG shell
//! and into the backend-neutral [`gen_core::Generator`] contract.
//!
//! What changed vs the spike, and what deliberately did **not**:
//! - **Same numerics** (the GO-validated path): dual CLIP (CLIP-L + CLIP-bigG) loaded **f32** and
//!   encoded, embeddings cast to **f16**; UNet **f16**; VAE **f16** with the
//!   `madebyollin/sdxl-vae-fp16-fix` (f16 SDXL VAE NaNs without it); VAE scale **0.13025** (the
//!   diffusers SDXL value, not candle's hardcoded SD1.5 0.18215); euler-ancestral scheduler.
//! - **Seeding is still the spike's `device.set_seed` + `Tensor::randn`** — the env-fragile path the
//!   sc-3498 findings flagged. Replacing it with deterministic CPU-seeded noise + a non-ancestral
//!   scheduler is **sc-3673** (the conformance suite's seed-determinism check, sc-4481, gates it);
//!   it is intentionally NOT pre-empted here. See the `// sc-3673` seam in [`Pipeline::generate`].
//! - **CLI/`emit_event`/PNG/sidecar removed**: progress is `on_progress(Progress::Step/Decoding)`,
//!   cancellation is `req.cancel` → typed [`gen_core::Error::Canceled`], and each image is returned as a
//!   `gen_core::Image` (RGB8) — the worker owns asset writes (no candle-specific worker code).
//! - **Weights come from `spec.weights` (the SDXL snapshot dir)**, not a hardcoded HF repo: UNet +
//!   both text encoders load from the snapshot's component subdirs. The two **model-agnostic** inputs
//!   — the fp16-VAE-fix and the CLIP-L/bigG `tokenizer.json`s — still resolve via `hf-hub` (cached),
//!   exactly as the spike.
//!
//! Component loading is hoisted above the per-image `count` loop (the spike's shape) but runs once
//! **per `generate` call**; caching loaded components across calls is a follow-up (sc-3674).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_gen::gen_core::{self, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};
use tokenizers::Tokenizer;

/// diffusers SDXL VAE `scaling_factor` (candle's example hardcodes the SD1.5 value 0.18215 for `Xl`;
/// 0.13025 is the diffusers-correct one and is what produced correctly-exposed output in the spike).
const VAE_SCALE: f64 = 0.13025;
/// Production SDXL defaults (the SceneWorks `sdxl` row): 30 steps, CFG 7.0 — used when the request
/// omits them.
const DEFAULT_STEPS: usize = 30;
const DEFAULT_GUIDANCE: f64 = 7.0;

/// The fp16-stable SDXL VAE (the base VAE NaNs in f16). Model-agnostic across every SDXL checkpoint,
/// so it is fetched by repo id rather than read from the per-model snapshot.
const VAE_FIX_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";
const VAE_FIX_FILE: &str = "diffusion_pytorch_model.safetensors";

/// Which of the two SDXL CLIP encoders — selects the tokenizer repo, the snapshot weights subpath,
/// and which `StableDiffusionConfig` clip config to use.
enum Clip {
    /// CLIP-L (`text_encoder/`) — `openai/clip-vit-large-patch14` tokenizer.
    L,
    /// OpenCLIP bigG (`text_encoder_2/`) — `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` tokenizer.
    BigG,
}

impl Clip {
    /// `(tokenizer repo, snapshot weights subpath)`.
    fn sources(&self) -> (&'static str, &'static str) {
        match self {
            Clip::L => (
                "openai/clip-vit-large-patch14",
                "text_encoder/model.fp16.safetensors",
            ),
            Clip::BigG => (
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "text_encoder_2/model.fp16.safetensors",
            ),
        }
    }
}

/// Resolve a file from a (cached) HF repo — used only for the model-agnostic tokenizers + fp16-VAE-fix.
fn hf_get(repo: &str, path: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;
    Api::new()
        .and_then(|api| api.model(repo.to_string()).get(path))
        .map_err(|e| CandleError::Msg(format!("hf-hub fetch {repo}/{path}: {e}")))
}

/// A loaded txt2img pipeline: the dual CLIP encoders + their tokenizers, the UNet, and the f16 VAE,
/// plus the `StableDiffusionConfig` (carries the per-request latent dims) and the compute device/dtype.
pub(crate) struct Pipeline {
    config: StableDiffusionConfig,
    device: Device,
    dtype: DType,
    tokenizer_l: Tokenizer,
    tokenizer_g: Tokenizer,
    text_model_l: stable_diffusion::clip::ClipTextTransformer,
    text_model_g: stable_diffusion::clip::ClipTextTransformer,
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: stable_diffusion::vae::AutoEncoderKL,
}

impl Pipeline {
    /// Build the pipeline from the SDXL snapshot `root` at the given device/dtype (f16). UNet + both
    /// text encoders read the snapshot's component subdirs; the f16-VAE-fix + CLIP tokenizers resolve
    /// via `hf-hub`. CLIP weights load **f32** (matching candle's reference), embeddings cast to `dtype`.
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        dtype: DType,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // The config's only request-dependent fields are the latent dims; the component configs
        // (clip/clip2/unet/autoencoder) are fixed for SDXL.
        let config =
            StableDiffusionConfig::sdxl(None, Some(height as usize), Some(width as usize));

        let load_clip = |which: Clip| -> Result<(Tokenizer, stable_diffusion::clip::ClipTextTransformer)> {
            let (tok_repo, weights_sub) = which.sources();
            let tokenizer = Tokenizer::from_file(hf_get(tok_repo, "tokenizer.json")?)
                .map_err(|e| CandleError::Msg(format!("load tokenizer {tok_repo}: {e}")))?;
            let clip_cfg = match which {
                Clip::L => &config.clip,
                Clip::BigG => config
                    .clip2
                    .as_ref()
                    .ok_or_else(|| CandleError::Msg("sdxl config missing clip2".into()))?,
            };
            // CLIP loads f32 even though the weights file is fp16 (candle reference behavior).
            let model = stable_diffusion::build_clip_transformer(
                clip_cfg,
                snapshot_file(root, weights_sub)?,
                device,
                DType::F32,
            )?;
            Ok((tokenizer, model))
        };

        let (tokenizer_l, text_model_l) = load_clip(Clip::L)?;
        let (tokenizer_g, text_model_g) = load_clip(Clip::BigG)?;

        let vae = config.build_vae(hf_get(VAE_FIX_REPO, VAE_FIX_FILE)?, device, dtype)?;
        let unet = config.build_unet(
            snapshot_file(root, "unet/diffusion_pytorch_model.fp16.safetensors")?,
            device,
            4,
            // sc-3674 wires candle-flash-attn; the spike (and this faithful port) run unfused.
            false,
            dtype,
        )?;

        Ok(Self {
            config,
            device: device.clone(),
            dtype,
            tokenizer_l,
            tokenizer_g,
            text_model_l,
            text_model_g,
            unet,
            vae,
        })
    }

    /// SDXL dual-CLIP conditioning: encode `prompt` (cond) and `uncond` through both encoders, stack
    /// `[uncond, cond]` on the batch axis, and concatenate the two encoders on the feature axis —
    /// shape `[2, tokens, 2048]`, cast to the compute dtype. Mirrors the spike's `text_embeddings`.
    fn text_embeddings(&self, prompt: &str, uncond: &str) -> Result<Tensor> {
        let l = self.encode_one(Clip::L, prompt, uncond)?;
        let g = self.encode_one(Clip::BigG, prompt, uncond)?;
        Ok(Tensor::cat(&[l, g], D::Minus1)?)
    }

    /// Encode `[uncond, cond]` through one CLIP encoder, padded to its `max_position_embeddings`.
    fn encode_one(&self, which: Clip, prompt: &str, uncond: &str) -> Result<Tensor> {
        let (clip_cfg, tokenizer, text_model) = match which {
            Clip::L => (&self.config.clip, &self.tokenizer_l, &self.text_model_l),
            Clip::BigG => (
                self.config
                    .clip2
                    .as_ref()
                    .ok_or_else(|| CandleError::Msg("sdxl config missing clip2".into()))?,
                &self.tokenizer_g,
                &self.text_model_g,
            ),
        };
        let vocab = tokenizer.get_vocab(true);
        let pad_token = clip_cfg.pad_with.clone().unwrap_or_else(|| "<|endoftext|>".into());
        let pad_id = *vocab
            .get(pad_token.as_str())
            .ok_or_else(|| CandleError::Msg(format!("pad token {pad_token:?} not in vocab")))?;

        let encode = |text: &str| -> Result<Tensor> {
            let mut tokens = tokenizer
                .encode(text, true)
                .map_err(|e| CandleError::Msg(format!("tokenize: {e}")))?
                .get_ids()
                .to_vec();
            let max = clip_cfg.max_position_embeddings;
            if tokens.len() > max {
                return Err(CandleError::Msg(format!(
                    "prompt too long: {} tokens > {max}",
                    tokens.len()
                )));
            }
            while tokens.len() < max {
                tokens.push(pad_id);
            }
            Ok(Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?)
        };

        let cond = text_model.forward(&encode(prompt)?)?;
        let uncond = text_model.forward(&encode(uncond)?)?;
        Ok(Tensor::cat(&[uncond, cond], 0)?.to_dtype(self.dtype)?)
    }

    /// Run txt2img for `req`, emitting per-step progress and honoring `req.cancel`. Returns one
    /// `gen_core::Image` per `req.count` (each with seed `base_seed + index`).
    pub(crate) fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS);
        let guidance = req.guidance.map(|g| g as f64).unwrap_or(DEFAULT_GUIDANCE);
        let use_guide = guidance > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let total = steps as u32;

        // Seed-independent conditioning, hoisted above the count loop (the dual-CLIP forward draws no
        // RNG); only the per-image init noise depends on the seed.
        let text_embeddings = self.text_embeddings(&req.prompt, negative)?;
        let (lat_h, lat_w) = (self.config.height / 8, self.config.width / 8);

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = base_seed.wrapping_add(index as u64);
            // sc-3673: env-fragile seeding (CUDA `set_seed` + on-device `randn`). To be replaced by a
            // deterministic CPU-seeded noise draw + a non-ancestral scheduler.
            self.device.set_seed(seed)?;

            let mut scheduler = self.config.build_scheduler(steps)?;
            let timesteps = scheduler.timesteps().to_vec();
            let init = Tensor::randn(0f32, 1f32, (1, 4, lat_h, lat_w), &self.device)?;
            let mut latents = (init * scheduler.init_noise_sigma())?.to_dtype(self.dtype)?;

            for (step_i, &timestep) in timesteps.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                let model_in = if use_guide {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };
                let model_in = scheduler.scale_model_input(model_in, timestep)?;
                let noise_pred = self.unet.forward(&model_in, timestep as f64, &text_embeddings)?;
                let noise_pred = if use_guide {
                    let chunks = noise_pred.chunk(2, 0)?;
                    let (uncond, cond) = (&chunks[0], &chunks[1]);
                    (uncond + ((cond - uncond)? * guidance)?)?
                } else {
                    noise_pred
                };
                latents = scheduler.step(&noise_pred, timestep, &latents)?;
                on_progress(Progress::Step {
                    current: step_i as u32 + 1,
                    total,
                });
            }

            on_progress(Progress::Decoding);
            images.push(self.decode(&latents)?);
        }
        Ok(images)
    }

    /// VAE-decode latents to an RGB8 [`Image`] (un-scale by [`VAE_SCALE`], `x/2 + 0.5`, clamp, ×255).
    fn decode(&self, latents: &Tensor) -> Result<Image> {
        let img = self.vae.decode(&(latents / VAE_SCALE)?)?;
        let img = ((img / 2.)? + 0.5)?.clamp(0f32, 1f32)?;
        let img = (img * 255.)?.to_dtype(DType::U8)?.i(0)?.to_device(&Device::Cpu)?;
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

/// Resolve a component file inside the SDXL snapshot dir, erroring clearly if absent (e.g. a
/// single-file RealVisXL checkpoint that lacks the diffusers multi-component tree — sc-3677).
fn snapshot_file(root: &Path, sub: &str) -> Result<PathBuf> {
    let p = root.join(sub);
    if !p.is_file() {
        return Err(CandleError::Msg(format!(
            "sdxl snapshot is missing {sub} (expected a diffusers multi-component snapshot at {})",
            root.display()
        )));
    }
    Ok(p)
}
