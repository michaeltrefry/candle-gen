//! The candle SDXL **txt2img** pipeline (sc-3675) — the proven epic-3494 prototype
//! (`D:\sceneworks-candle-spike\src\bin\candle_sdxl.rs`) lifted out of its standalone CLI/PNG shell
//! and into the backend-neutral [`gen_core::Generator`] contract.
//!
//! What changed vs the spike, and what deliberately did **not**:
//! - **Components** (the GO-validated path): dual CLIP (CLIP-L + CLIP-bigG) loaded **f16** (sc-3674;
//!   the spike used f32) and encoded; UNet **f16**; VAE **f16** with the `madebyollin/sdxl-vae-fp16-fix`
//!   (f16 SDXL VAE NaNs without it); VAE scale **0.13025** (the diffusers SDXL value, not candle's
//!   hardcoded SD1.5 0.18215).
//! - **Perf (sc-3674)**: the UNet attention runs through fused **flash-attention** when the crate is
//!   built `--features flash-attn` AND the runtime toggle ([`crate::set_flash_attn`], default on) is
//!   set — on Blackwell sm_120 that cut steady-state from ~0.32 to ~0.21 s/step and peak VRAM ~21.6→18
//!   GiB. The build feature is the opt-in; the toggle is what the SceneWorks UI exposes.
//! - **Peak VRAM (sc-4987)**: two structural levers on top of sc-3674's 18 GiB high-water mark, both
//!   targeting torch-parity (~9 GiB) at 1024². (1) **Staged sequential load** — each CLIP encoder is
//!   loaded, run, and **dropped** before the next, and *both* are gone before the UNet/VAE even load
//!   (text embeddings are seed-independent, computed once up front), so the dual CLIP (~1.6 GiB f16)
//!   never sits resident through denoise/decode. (2) **VAE tiling** — the VAE decode at 1024² is the
//!   tallest single allocation; [`tile_blend_decode`] splits the latent into overlapping 64² latent
//!   tiles (512² output), decodes each, and trapezoidally blends the seams (diffusers'
//!   `enable_vae_tiling`), bounding the decode peak to one tile. Gated by [`crate::vae_tiling_enabled`]
//!   (default on) and only *fires* above 512² output (the geometry policy lives in [`gen_core::tiling`]).
//! - **Deterministic seeding + non-ancestral scheduler (sc-3673)**: initial noise is drawn from a
//!   fixed-algorithm CPU RNG (`StdRng`) seeded by `seed` and moved to the device — NOT candle's CUDA
//!   `device.set_seed`, whose seed→noise mapping was not portable across launch environments and
//!   occasionally collapsed the sample (sc-3498). The sampler is **DDIM (eta=0)**, non-ancestral, so
//!   there is no per-step stochastic noise. Net: generation is a pure function of `(seed, request)`.
//! - **CLI/`emit_event`/PNG/sidecar removed**: progress is `on_progress(Progress::Step/Decoding)`,
//!   cancellation is `req.cancel` → typed [`gen_core::Error::Canceled`], and each image is returned as a
//!   `gen_core::Image` (RGB8) — the worker owns asset writes (no candle-specific worker code).
//! - **Weights come from `spec.weights` (the SDXL snapshot dir)**, not a hardcoded HF repo: UNet +
//!   both text encoders load from the snapshot's component subdirs. The two **model-agnostic** inputs
//!   — the fp16-VAE-fix and the CLIP-L/bigG `tokenizer.json`s — still resolve via `hf-hub` (cached),
//!   exactly as the spike.
//!
//! Component *caching across* `generate` calls (the spike's per-call reload) stays a follow-up — and
//! is in tension with the sc-4987 staged load, which deliberately frees components mid-call to bound
//! peak VRAM; a cache would hold UNet+VAE resident between calls (a latency, not a peak-VRAM, win).

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_gen::gen_core::tiling::{TilingConfig, VaeTiling};
use candle_gen::gen_core::{self, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::stable_diffusion::ddim::DDIMSchedulerConfig;
use candle_transformers::models::stable_diffusion::schedulers::SchedulerConfig;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
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

/// The SDXL VAE's tiling geometry (sc-4987): the decoder upsamples latents ×8 spatially, and an image
/// VAE has **no temporal axis** — so temporal scale 1, non-causal (the `[B, 4, h, w]` latent is tiled
/// on the two spatial axes only, with the singleton temporal axis a no-op in [`TilingConfig::plan`]).
const SDXL_VAE_TILING: VaeTiling = VaeTiling {
    spatial_scale: 8,
    temporal_scale: 1,
    causal_temporal: false,
};

/// The SDXL VAE tiling policy (sc-4987) — diffusers' `enable_vae_tiling` defaults: **512² output
/// tiles (64² latent) with 128 px overlap (16 latent, the 0.25 overlap-factor)**. `needs_tiling` then
/// fires only when an output axis exceeds 512 px, so 512² renders stay monolithic (latent 64 is not
/// `> 64`) and 1024² tiles into a 3×3 grid stepping 48 latent — bounding the decode peak to one 512²
/// tile while the 16-latent overlap + trapezoidal blend keeps seams invisible.
fn sdxl_tiling_config() -> TilingConfig {
    TilingConfig::spatial_only(512, 128)
}

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

/// A txt2img pipeline handle. sc-4987 made loading **staged**: this carries only the
/// `StableDiffusionConfig` (the per-request latent dims), the snapshot `root`, and the compute
/// device/dtype — the heavy components (CLIP, UNet, VAE) are loaded *inside* [`generate`] in the
/// order they are needed and dropped as soon as they are not, so the dual CLIP is freed before the
/// UNet/VAE ever allocate. (Pre-sc-4987 this struct held all four components resident at once.)
pub(crate) struct Pipeline {
    config: StableDiffusionConfig,
    root: PathBuf,
    device: Device,
    dtype: DType,
}

impl Pipeline {
    /// Build the (light) pipeline handle for the SDXL snapshot `root` at the given device/dtype (f16)
    /// and request dims. This does **no** weight I/O — the config's only request-dependent fields are
    /// the latent dims; the heavy components load lazily in [`generate`].
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        dtype: DType,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // The config's only request-dependent fields are the latent dims; the component configs
        // (clip/clip2/unet/autoencoder) are fixed for SDXL.
        let config = StableDiffusionConfig::sdxl(None, Some(height as usize), Some(width as usize));
        Ok(Self {
            config,
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
        })
    }

    /// SDXL dual-CLIP conditioning: encode `prompt` (cond) and `uncond` through both encoders, stack
    /// `[uncond, cond]` on the batch axis, and concatenate the two encoders on the feature axis —
    /// shape `[2, tokens, 2048]`, cast to the compute dtype. Mirrors the spike's `text_embeddings`.
    ///
    /// sc-4987: each encoder is loaded, run, and dropped **inside** [`encode_one`] before the next is
    /// loaded — so the two CLIP encoders are never co-resident, and both are gone when this returns
    /// (before the UNet/VAE load). The embeddings it returns are the only thing that outlives them.
    fn text_embeddings(&self, prompt: &str, uncond: &str) -> Result<Tensor> {
        let l = self.encode_one(Clip::L, prompt, uncond)?;
        let g = self.encode_one(Clip::BigG, prompt, uncond)?;
        Ok(Tensor::cat(&[l, g], D::Minus1)?)
    }

    /// Load one CLIP encoder, encode `[uncond, cond]` through it (padded to its
    /// `max_position_embeddings`), and return the embeddings — the encoder weights are loaded into a
    /// local and **dropped when this function returns** (sc-4987), freeing its VRAM before the next
    /// encoder / the UNet load.
    fn encode_one(&self, which: Clip, prompt: &str, uncond: &str) -> Result<Tensor> {
        let (tok_repo, weights_sub) = which.sources();
        let clip_cfg = match which {
            Clip::L => &self.config.clip,
            Clip::BigG => self
                .config
                .clip2
                .as_ref()
                .ok_or_else(|| CandleError::Msg("sdxl config missing clip2".into()))?,
        };
        let tokenizer = Tokenizer::from_file(hf_get(tok_repo, "tokenizer.json")?)
            .map_err(|e| CandleError::Msg(format!("load tokenizer {tok_repo}: {e}")))?;
        // sc-3674: load CLIP at the compute dtype (f16), not the spike's F32. The fp16 safetensors
        // load directly, the forward runs f16 (diffusers loads CLIP fp16 too), and it halves the
        // text-encoder VRAM (CLIP-bigG ~2.8→1.4 GiB) with no visible quality change. The embeddings
        // are cast to `dtype` below.
        let text_model = stable_diffusion::build_clip_transformer(
            clip_cfg,
            snapshot_file(&self.root, weights_sub)?,
            &self.device,
            self.dtype,
        )?;

        let vocab = tokenizer.get_vocab(true);
        let pad_token = clip_cfg
            .pad_with
            .clone()
            .unwrap_or_else(|| "<|endoftext|>".into());
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
        // `text_model` + `tokenizer` drop here, freeing this encoder before the caller loads the next.
    }

    /// Load the UNet (f16) from the snapshot, routing attention through fused flash-attention when the
    /// crate is built `--features flash-attn` AND the runtime toggle ([`crate::set_flash_attn`]) is on.
    fn load_unet(&self) -> Result<stable_diffusion::unet_2d::UNet2DConditionModel> {
        // sc-3674: the build feature compiles the CUTLASS kernels in; the runtime toggle (which the
        // SceneWorks UI exposes) decides whether a flash-capable build actually uses them.
        let use_flash_attn = cfg!(feature = "flash-attn") && crate::flash_attn_enabled();
        Ok(self.config.build_unet(
            snapshot_file(&self.root, "unet/diffusion_pytorch_model.fp16.safetensors")?,
            &self.device,
            4,
            use_flash_attn,
            self.dtype,
        )?)
    }

    /// Load the f16-stable VAE (the `madebyollin/sdxl-vae-fp16-fix` weights, resolved via `hf-hub`).
    fn load_vae(&self) -> Result<AutoEncoderKL> {
        Ok(self.config.build_vae(
            hf_get(VAE_FIX_REPO, VAE_FIX_FILE)?,
            &self.device,
            self.dtype,
        )?)
    }

    /// Run txt2img for `req`, emitting per-step progress and honoring `req.cancel`. Returns one
    /// `gen_core::Image` per `req.count` (each with seed `base_seed + index`).
    ///
    /// sc-4987 staging: the seed-independent text embeddings are computed first (which loads and frees
    /// both CLIP encoders), *then* the UNet + VAE load — so the dual CLIP is never resident through the
    /// denoise/decode that follow.
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
        // RNG); only the per-image init noise depends on the seed. Computing it first also frees both
        // CLIP encoders (sc-4987) before the UNet/VAE below allocate.
        let text_embeddings = self.text_embeddings(&req.prompt, negative)?;
        let (lat_h, lat_w) = (self.config.height / 8, self.config.width / 8);

        // Heavy components, loaded once per call AFTER CLIP is freed and reused across the count loop.
        let unet = self.load_unet()?;
        let vae = self.load_vae()?;

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = base_seed.wrapping_add(index as u64);

            // sc-3673 — deterministic, launch-portable initial noise: draw N(0,1) from a
            // fixed-algorithm CPU RNG (`StdRng`, ChaCha-based) seeded by `seed`, build the latent on
            // CPU, then move it to the compute device. This replaces candle's CUDA `device.set_seed`
            // + on-device `randn`, whose seed→noise mapping was NOT portable across launch
            // environments and occasionally collapsed the sample to garbage (sc-3498). Paired with the
            // non-ancestral DDIM scheduler below (no per-step stochastic noise), the whole generation
            // is now a pure function of `(seed, request)` — same seed ⇒ same image, any launch.
            let n = 4 * lat_h * lat_w;
            let mut rng = StdRng::seed_from_u64(seed);
            let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
            let init = Tensor::from_vec(noise, (1, 4, lat_h, lat_w), &Device::Cpu)?
                .to_device(&self.device)?;

            // DDIM (eta=0): non-ancestral / deterministic, vs candle's default Euler-ancestral (which
            // injects fresh noise every step). SceneWorks/diffusers SDXL defaults to EulerDiscrete —
            // also non-ancestral, deterministic; DDIM is the closest deterministic solver candle ships
            // and gives portable, collapse-free output. Its config defaults ARE the SDXL values
            // (scaled_linear β 0.00085→0.012, epsilon prediction, 1000 train steps).
            let mut scheduler = DDIMSchedulerConfig::default().build(steps)?;
            let timesteps = scheduler.timesteps().to_vec();
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
                let noise_pred = unet.forward(&model_in, timestep as f64, &text_embeddings)?;
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
            images.push(self.decode(&vae, &latents)?);
        }
        Ok(images)
    }

    /// VAE-decode latents to an RGB8 [`Image`] (un-scale by [`VAE_SCALE`], `x/2 + 0.5`, clamp, ×255).
    /// The decode itself runs tiled or monolithic per [`decode_image`](Self::decode_image).
    fn decode(&self, vae: &AutoEncoderKL, latents: &Tensor) -> Result<Image> {
        let unscaled = (latents / VAE_SCALE)?;
        let img = self.decode_image(vae, &unscaled)?;
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

    /// Decode the already-unscaled latent to an image tensor `[1, 3, H, W]`. Tiled (sc-4987) when
    /// [`crate::vae_tiling_enabled`] is set AND the output exceeds the tiling threshold (512²);
    /// otherwise the monolithic `AutoEncoderKL::decode`. The non-tiling path is byte-identical to
    /// pre-sc-4987, so 512² renders and the conformance suite are unaffected.
    fn decode_image(&self, vae: &AutoEncoderKL, unscaled: &Tensor) -> Result<Tensor> {
        if crate::vae_tiling_enabled() {
            let cfg = sdxl_tiling_config();
            let (_, _, h, w) = unscaled.dims4()?;
            if cfg.needs_tiling(SDXL_VAE_TILING, 1, h as i32, w as i32) {
                return tile_blend_decode(unscaled, SDXL_VAE_TILING, &cfg, |tile| {
                    Ok(vae.decode(tile)?)
                });
            }
        }
        Ok(vae.decode(unscaled)?)
    }
}

/// Tiled VAE decode with trapezoidal seam blending (sc-4987) — the candle port of mlx-gen's
/// `tile_decode_accumulate`, specialized to a 4-D image latent `[B, C, h, w]` (no temporal axis).
///
/// Splits `unscaled` (the already-`/VAE_SCALE` latent) into the overlapping spatial tiles planned by
/// [`TilingConfig::plan`], decodes each via `decode_tile`, and accumulates `Σ(maskᵢ·decodeᵢ)` and
/// `Σ maskᵢ` into full-size output/weight buffers, returning `output / max(weights, 1e-8)`. Because
/// the tiles overlap and the per-axis masks are a partition of unity, the blend is exact for an
/// identity decode (the CPU unit test) and seam-free for the real VAE (the overlap absorbs the
/// boundary-conv mismatch). Peak memory is bounded by **one tile's** decode — the win — plus the two
/// full-size (but f32, ~12 MiB at 1024²) accumulators.
///
/// Accumulation is in f32: `decode_tile` runs f16, but the blend divide wants the mask precision and
/// f32 at output resolution is negligible. The returned tensor is `[1, 3, out_h, out_w]` f32, which
/// the caller's `/2 + 0.5 / clamp / ×255` post-processing consumes identically to the f16 mono path.
fn tile_blend_decode(
    unscaled: &Tensor,
    vae_tiling: VaeTiling,
    cfg: &TilingConfig,
    decode_tile: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let device = unscaled.device();
    let (_b, _c, h, w) = unscaled.dims4()?;
    // f = 1: an image latent has no temporal axis, so the plan's single temporal tile is a no-op and
    // we iterate the spatial (h × w) tiles only.
    let plan = cfg.plan(vae_tiling, 1, h as i32, w as i32);
    let (out_h, out_w) = (plan.out_h as usize, plan.out_w as usize);

    let mut output: Option<Tensor> = None; // [1, 3, out_h, out_w] f32
    let mut weights: Option<Tensor> = None; // [1, 1, out_h, out_w] f32
    for hh in &plan.h {
        for ww in &plan.w {
            let tile = unscaled
                .narrow(2, hh.start as usize, (hh.end - hh.start) as usize)?
                .narrow(3, ww.start as usize, (ww.end - ww.start) as usize)?;
            let dec = decode_tile(&tile)?.to_dtype(DType::F32)?;

            // Clip the decoded tile + masks to the planned output span (guards the VAE returning a
            // pixel or two over/under the latent×scale span; for SDXL's exact ×8 this is a no-op).
            let (_, _, dh, dw) = dec.dims4()?;
            let ah = dh.min((hh.out_stop - hh.out_start) as usize);
            let aw = dw.min((ww.out_stop - ww.out_start) as usize);
            let dec = dec.narrow(2, 0, ah)?.narrow(3, 0, aw)?;

            // 1-D trapezoidal masks → outer product, each broadcasting along its own (h / w) axis.
            let hm = Tensor::from_slice(&hh.mask[..ah], (1, 1, ah, 1), device)?;
            let wm = Tensor::from_slice(&ww.mask[..aw], (1, 1, 1, aw), device)?;
            let blend = hm.broadcast_mul(&wm)?; // [1, 1, ah, aw]
            let weighted = dec.broadcast_mul(&blend)?; // [1, 3, ah, aw]

            // Place each tile at its (out_start) offset by zero-padding to the full output shape, then
            // add — the bounded-peak accumulate (mirrors the reference's full-size output+weights).
            let (pad_top, pad_bottom) =
                (hh.out_start as usize, out_h - (hh.out_start as usize + ah));
            let (pad_left, pad_right) =
                (ww.out_start as usize, out_w - (ww.out_start as usize + aw));
            let weighted_full = weighted
                .pad_with_zeros(2, pad_top, pad_bottom)?
                .pad_with_zeros(3, pad_left, pad_right)?;
            let blend_full = blend
                .pad_with_zeros(2, pad_top, pad_bottom)?
                .pad_with_zeros(3, pad_left, pad_right)?;

            output = Some(match output {
                None => weighted_full,
                Some(acc) => (acc + weighted_full)?,
            });
            weights = Some(match weights {
                None => blend_full,
                Some(acc) => (acc + blend_full)?,
            });
        }
    }

    let output = output.ok_or_else(|| CandleError::Msg("vae tiling produced no tiles".into()))?;
    let weights = weights.ok_or_else(|| CandleError::Msg("vae tiling produced no tiles".into()))?;
    // Normalize by the summed blend weight (floored to avoid a divide-by-zero at any gap; the plan's
    // coverage invariant guarantees weights > 0 everywhere, so the floor never actually engages).
    Ok(output.broadcast_div(&weights.clamp(1e-8f32, f32::MAX)?)?)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiled blend (slice → mask → pad → accumulate → normalize) must exactly reconstruct the
    /// input under an **identity** decode at spatial-scale 1 — every output position is
    /// `Σ(maskᵢ·xᵢ) / Σ maskᵢ = x`, regardless of the (overlapping) trapezoidal mask values. This
    /// covers the candle accumulation math on CPU without a GPU/VAE; the per-axis tiling geometry
    /// itself is unit-tested in `gen_core::tiling`.
    #[test]
    fn tile_blend_identity_roundtrip() {
        let device = Device::Cpu;
        // 1×1 spatial scale so out dims == latent dims and an identity decode is shape-preserving.
        let vae = VaeTiling {
            spatial_scale: 1,
            temporal_scale: 1,
            causal_temporal: false,
        };
        // A small grid with overlapping tiles: 4-wide tiles, 2 overlap, over a 10×10 field → 4 tiles
        // per axis, exercising left/right ramps and the interior all-ones region.
        let cfg = TilingConfig::spatial_only(4, 2);
        let (h, w) = (10usize, 10usize);
        let vals: Vec<f32> = (0..(h * w) as i64).map(|i| i as f32).collect();
        let input = Tensor::from_vec(vals.clone(), (1, 1, h, w), &device).unwrap();

        // Sanity: tiling actually fires for this config/size.
        assert!(cfg.needs_tiling(vae, 1, h as i32, w as i32));

        let out = tile_blend_decode(&input, vae, &cfg, |tile| Ok(tile.clone())).unwrap();
        assert_eq!(out.dims4().unwrap(), (1, 1, h, w));
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (g, e) in got.iter().zip(vals.iter()) {
            assert!((g - e).abs() < 1e-4, "blend reconstruction off: {g} vs {e}");
        }
    }

    /// Below the tiling threshold (a 64² latent → 512² output, the conformance render size) the plan
    /// produces a **single** tile, so the tiled path is a no-op pass-through identical to a monolithic
    /// decode — the guarantee that 512² output is unchanged by sc-4987.
    #[test]
    fn no_tiling_below_threshold() {
        let cfg = sdxl_tiling_config();
        // 64² latent = 512² output: not > the 64-latent tile, so tiling must NOT fire.
        assert!(!cfg.needs_tiling(SDXL_VAE_TILING, 1, 64, 64));
        // 128² latent = 1024² output: must fire.
        assert!(cfg.needs_tiling(SDXL_VAE_TILING, 1, 128, 128));
    }
}
