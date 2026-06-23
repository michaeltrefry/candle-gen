//! Boogu Base + Turbo text-to-image pipelines — tokenize → condition-encode → flow-match denoise →
//! VAE decode. Port of `mlx-gen-boogu`'s `pipeline.rs` (T2I paths; the Edit path lands in sc-7523).
//!
//! - **Base** (`boogu_image`): true-CFG, 50-step rectified-flow Euler over the snapshot's static-v1
//!   shift schedule (`mu = lin(seq_len) = 1.15`), routed through the unified curated-sampler framework
//!   (epic 7114). The DiT is fed the shifted clean-fraction timestep `t = 1 − σ` (OneMinusSigma) and
//!   predicts the velocity in clean-fraction time, so `predict` negates it into `run_flow_sampler`'s
//!   noise-fraction FLOW convention. True-CFG: `pred = cond + (scale − 1)·(cond − uncond)`.
//! - **Turbo** (`boogu_image_turbo`): the DMD student few-step loop (CFG-free) over the
//!   `linspace(conditioning_sigma, 1, steps+1)[:-1]` clean-fraction grid — predict the clean estimate
//!   `x += (1 − σ)·v`, then renoise to the next level with fresh noise.
//!
//! Per-sample `B = 1`; the DiT runs once per condition. Deterministic CPU-seeded initial noise
//! (sc-3673 parity), exactly as the z-image/ideogram providers.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{Module, VarBuilder};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{self, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::z_image::sampling::postprocess_image;
use candle_transformers::models::z_image::vae::{AutoEncoderKL, Encoder, VaeConfig};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::config::BooguConfig;
use crate::loader::Weights;
use crate::text_encoder::{BooguTextEncoder, BooguTextEncoderConfig};
use crate::tokenizer::BooguTokenizer;
use crate::transformer::BooguTransformer;
use crate::vision::preprocess::preprocess_image;
use crate::vision::{VisionConfig, VisionTower};

/// Qwen3-VL image placeholder token (`mllm/config.json::image_token_id`) — the position the vision
/// tower's merged embeds are spliced into for image-conditioned editing.
const IMAGE_TOKEN_ID: u32 = 151655;

/// Base/Edit default steps + guidance (reference `__call__`: 50-step true-CFG, guidance 4.0).
pub(crate) const DEFAULT_STEPS: usize = 50;
pub(crate) const DEFAULT_GUIDANCE: f32 = 4.0;
/// Turbo default steps (DMD student few-step) + the lowest sigma in the DMD schedule.
pub(crate) const DEFAULT_TURBO_STEPS: usize = 4;
pub(crate) const DEFAULT_TURBO_SIGMA: f32 = 0.001;

/// VAE spatial downscale (the latent is image/8 per side) and latent channel count.
const SPATIAL_SCALE: u32 = 8;
const LATENT_CHANNELS: usize = 16;

/// Max prompt tokens the Qwen3-VL RoPE table is sized for (generous; Boogu prompts are short).
const MAX_TEXT_TOKENS: usize = 1280;

/// Component compute dtypes. The Qwen3-VL TE runs in **f32** (parity-grade for this encoder, shared
/// with the ideogram port); the 10 B DiT runs **bf16** (native on candle's CUDA backend); the small
/// FLUX.1 VAE runs **f32** (decode-precision-sensitive).
const TE_DTYPE: DType = DType::F32;
const DIT_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;

/// The loaded Boogu components, `Arc`-shared so the generator caches them across `generate` calls.
pub(crate) struct Components {
    tok: BooguTokenizer,
    te: BooguTextEncoder,
    dit: BooguTransformer,
    vae: Arc<AutoEncoderKL>,
}

/// Load the text-to-image components from a Boogu snapshot (`mllm/ transformer/ vae/`).
pub(crate) fn load_components(root: &Path, device: &Device) -> Result<Components> {
    let tok = BooguTokenizer::from_snapshot(root, device)?;

    let te_w = Weights::from_dir(&root.join("mllm"), device, TE_DTYPE)?;
    let te = BooguTextEncoder::load(
        &te_w,
        "model.language_model",
        &BooguTextEncoderConfig::qwen3_vl_8b(),
        MAX_TEXT_TOKENS,
    )?;

    let cfg = BooguConfig::from_snapshot(root)?;
    let dit_w = Weights::from_dir(&root.join("transformer"), device, DIT_DTYPE)?;
    let dit = BooguTransformer::load(&dit_w, &cfg)?;

    let vae_vb = vae_varbuilder(&root.join("vae"), device)?;
    let vae = AutoEncoderKL::new(&VaeConfig::z_image(), vae_vb)?;

    Ok(Components {
        tok,
        te,
        dit,
        vae: Arc::new(vae),
    })
}

/// Build a [`VarBuilder`] over every `.safetensors` in the snapshot's `vae/` dir at the VAE dtype.
fn vae_varbuilder(dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("boogu: read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "boogu: no .safetensors in {}",
            dir.display()
        )));
    }
    // SAFETY: read-only mmap of weight files; the standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, VAE_DTYPE, device)? };
    Ok(vb)
}

/// Render the **Base** (true-CFG) text-to-image path for `req`.
pub(crate) fn render_base(
    comps: &Components,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let steps = req.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS);
    let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

    // Condition encoding (seed-independent): positive instruction + CFG-negative (empty) instruction.
    let cond = comps.te.last_hidden(&comps.tok.encode_t2i(&req.prompt)?)?;
    let do_cfg = guidance > 1.0;
    let uncond = if do_cfg {
        Some(comps.te.last_hidden(&comps.tok.encode_negative()?)?)
    } else {
        None
    };

    let native = base_native_sigmas(steps);
    let sigmas = candle_gen::resolve_flow_schedule(
        req.scheduler.as_deref(),
        base_shift_mu(),
        steps,
        &native,
    );

    let mut images = Vec::with_capacity(req.count as usize);
    for index in 0..req.count {
        let seed = base_seed.wrapping_add(index as u64);
        let noise = init_noise(req.height, req.width, seed, 0, device)?;
        let lat = candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::OneMinusSigma,
            &sigmas,
            noise,
            seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let cond_v = comps.dit.forward(x, &t, &cond)?;
                let pred = match &uncond {
                    Some(u_hidden) => {
                        let uncond_v = comps.dit.forward(x, &t, u_hidden)?;
                        // pred = cond + (scale − 1)·(cond − uncond)
                        (&cond_v + ((&cond_v - &uncond_v)? * (guidance - 1.0) as f64)?)?
                    }
                    None => cond_v,
                };
                Ok(pred.to_dtype(DType::F32)?.neg()?)
            },
        )?;
        on_progress(Progress::Decoding);
        images.push(decode(&comps.vae, &lat)?);
    }
    Ok(images)
}

/// Render the **Turbo** (DMD student few-step, CFG-free) text-to-image path for `req`.
pub(crate) fn render_turbo(
    comps: &Components,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let steps = req.steps.map(|s| s as usize).unwrap_or(DEFAULT_TURBO_STEPS);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    let cond = comps.te.last_hidden(&comps.tok.encode_t2i(&req.prompt)?)?;
    let sigmas = dmd_sigmas(DEFAULT_TURBO_SIGMA, steps);

    let mut images = Vec::with_capacity(req.count as usize);
    for index in 0..req.count {
        let seed = base_seed.wrapping_add(index as u64);
        let mut lat = init_noise(req.height, req.width, seed, 0, device)?;
        for i in 0..steps {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let sigma = sigmas[i];
            let t = Tensor::from_vec(vec![sigma], (1,), device)?;
            let pred = comps.dit.forward(&lat, &t, &cond)?;
            // Predict (clean estimate): x += (1 − sigma)·v, in f32.
            lat =
                (lat.to_dtype(DType::F32)? + (pred.to_dtype(DType::F32)? * (1.0 - sigma) as f64)?)?;
            // Renoise to the next sigma level with fresh noise (all but the final step).
            if i + 1 < steps {
                let sigma_next = sigmas[i + 1];
                let noise = init_noise(req.height, req.width, seed, (i + 1) as u64, device)?;
                lat = ((noise * (1.0 - sigma_next) as f64)? + (&lat * sigma_next as f64)?)?;
            }
            on_progress(Progress::Step {
                current: (i + 1) as u32,
                total: steps as u32,
            });
        }
        on_progress(Progress::Decoding);
        images.push(decode(&comps.vae, &lat)?);
    }
    Ok(images)
}

// ── Edit (single-reference TI2I) path (sc-7523) ──────────────────────────────────────────────────

/// Edit-only components, lazily loaded on the first edit so the T2I paths keep their footprint: the
/// Qwen3-VL **vision tower** (image-conditioned instruction features) and a standalone VAE
/// **encoder** (the reference → clean spatial latent). Both run f32.
pub(crate) struct EditComponents {
    vision: VisionTower,
    vae_encoder: Encoder,
}

/// Load the Edit-only components from a Boogu snapshot: the Qwen3-VL vision tower (`mllm/model.visual.*`)
/// and the FLUX.1 VAE encoder (`vae/encoder.*`), both f32.
pub(crate) fn load_edit_components(root: &Path, device: &Device) -> Result<EditComponents> {
    let mllm_w = Weights::from_dir(&root.join("mllm"), device, VAE_DTYPE)?;
    let vision = VisionTower::load(&mllm_w, VisionConfig::qwen3_vl(), "model.visual")?;
    let vae_vb = vae_varbuilder(&root.join("vae"), device)?;
    let vae_encoder = Encoder::new(&VaeConfig::z_image(), vae_vb.pp("encoder"))?;
    Ok(EditComponents {
        vision,
        vae_encoder,
    })
}

/// Render the **Edit** (single-reference TI2I, true-CFG) path for `req` with source `reference`.
///
/// Mirrors `mlx-gen-boogu`'s `generate_edit`: VAE-encode the reference into a clean spatial latent,
/// build image-conditioned instruction features (Qwen3-VL vision tower → MLLM splice + deepstack), and
/// flow-match denoise with the reference threaded through the DiT's `forward_edit` (the reference
/// shapes the DiT image sequence; the instruction drives the edit). Same static-v1 scheduler /
/// true-CFG as the Base path. The CFG-negative is the text-only empty/drop instruction
/// (`use_input_images_4_neg_instruct = false`, the reference default).
pub(crate) fn render_edit(
    comps: &Components,
    edit: &EditComponents,
    req: &GenerationRequest,
    reference: &Image,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    // The reference is VAE-encoded at its own dimensions; the latent must be patchify-able (p=2 over
    // an /8 latent ⇒ multiple of 16), matching the mlx twin's `validate_multiple_of_16(reference)`.
    if !reference.width.is_multiple_of(16) || !reference.height.is_multiple_of(16) {
        return Err(CandleError::Msg(format!(
            "boogu_image_edit: reference dims must be multiples of 16 (got {}x{})",
            reference.width, reference.height
        )));
    }
    let steps = req.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS);
    let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

    // Reference → clean VAE latent [1, 16, rH/8, rW/8] (seed-independent).
    let ref_latent = vae_encode(&edit.vae_encoder, reference, device)?;

    // Condition encoding (seed-independent): image-conditioned edit instruction + text-only
    // CFG-negative (empty/drop instruction). Both DiT passes carry the same reference latent.
    let cond = encode_image_instruction(comps, edit, reference, &req.prompt, device)?;
    let do_cfg = guidance > 1.0;
    let uncond = if do_cfg {
        Some(comps.te.last_hidden(&comps.tok.encode_negative()?)?)
    } else {
        None
    };

    let native = base_native_sigmas(steps);
    let sigmas = candle_gen::resolve_flow_schedule(
        req.scheduler.as_deref(),
        base_shift_mu(),
        steps,
        &native,
    );

    let mut images = Vec::with_capacity(req.count as usize);
    for index in 0..req.count {
        let seed = base_seed.wrapping_add(index as u64);
        let noise = init_noise(req.height, req.width, seed, 0, device)?;
        let lat = candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::OneMinusSigma,
            &sigmas,
            noise,
            seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let cond_v = comps.dit.forward_edit(x, &ref_latent, &t, &cond)?;
                let pred = match &uncond {
                    Some(u_hidden) => {
                        let uncond_v = comps.dit.forward_edit(x, &ref_latent, &t, u_hidden)?;
                        // pred = cond + (scale − 1)·(cond − uncond)
                        (&cond_v + ((&cond_v - &uncond_v)? * (guidance - 1.0) as f64)?)?
                    }
                    None => cond_v,
                };
                Ok(pred.to_dtype(DType::F32)?.neg()?)
            },
        )?;
        on_progress(Progress::Decoding);
        images.push(decode(&comps.vae, &lat)?);
    }
    Ok(images)
}

/// Image-conditioned instruction features for the edit path: preprocess the reference, run the
/// Qwen3-VL vision tower, render the chat template with the reference image block, and run the
/// image-conditioned MLLM forward. Returns `[1, L, 4096]` (f32) — the `<|image_pad|>` positions now
/// carry the vision tower's merged embeds + deepstack injections.
fn encode_image_instruction(
    comps: &Components,
    edit: &EditComponents,
    reference: &Image,
    instruction: &str,
    device: &Device,
) -> Result<Tensor> {
    let (pixel_values, grid) = preprocess_image(
        &reference.pixels,
        reference.height as usize,
        reference.width as usize,
        device,
    )?;
    let (image_embeds, deepstack) = edit.vision.forward(&pixel_values, &[grid])?;

    // Chat template with N = merged vision tokens worth of `<|image_pad|>` placeholders, then the
    // image-conditioned MLLM forward (vision splice + 3-D MRoPE + deepstack injection).
    let n = image_embeds.dim(0)?;
    let ids = comps.tok.encode_edit_with_image(instruction, n)?;
    Ok(comps
        .te
        .last_hidden_with_image(&ids, &image_embeds, &deepstack, grid, IMAGE_TOKEN_ID)?)
}

/// VAE-encode an RGB8 reference [`Image`] → clean latent `[1, 16, H/8, W/8]` (f32). Takes the latent
/// distribution **mean** (first half of the encoder channels), then maps to latent space as
/// `(mean − shift) · scale` — exactly the mlx `Vae::encode`, NOT the candle `AutoEncoderKL::encode`
/// (which *samples* the diagonal Gaussian; the Edit path needs the deterministic mode).
fn vae_encode(encoder: &Encoder, reference: &Image, device: &Device) -> Result<Tensor> {
    let pixels = image_to_pixels(reference, device)?; // [1, 3, H, W] in [-1, 1], f32
    let moments = encoder.forward(&pixels)?; // [1, 2C, H/8, W/8]
    let two_c = moments.dim(1)?;
    if two_c % 2 != 0 {
        return Err(CandleError::Msg(format!(
            "boogu edit: VAE encoder produced an odd channel count ({two_c}), expected 2·C"
        )));
    }
    let c = two_c / 2;
    let mean = moments.narrow(1, 0, c)?; // first C channels (the distribution mean)
    let cfg = VaeConfig::z_image();
    Ok(((mean - cfg.shift_factor)? * cfg.scaling_factor)?)
}

/// RGB8 [`Image`] (HWC, `[0, 255]`) → the VAE encoder's `[1, 3, H, W]` f32 tensor in `[-1, 1]` — the
/// inverse of `postprocess_image`'s `x·0.5 + 0.5` denormalize.
fn image_to_pixels(img: &Image, device: &Device) -> Result<Tensor> {
    let (h, w) = (img.height as usize, img.width as usize);
    let expected = h * w * 3;
    if img.pixels.len() != expected {
        return Err(CandleError::Msg(format!(
            "boogu: reference pixel buffer {} bytes != {}x{}x3 ({expected})",
            img.pixels.len(),
            img.width,
            img.height
        )));
    }
    let f: Vec<f32> = img
        .pixels
        .iter()
        .map(|&p| (p as f32 / 255.0) * 2.0 - 1.0)
        .collect();
    // HWC → CHW (batched): build [1, H, W, 3] then permute to [1, 3, H, W].
    let nhwc = Tensor::from_vec(f, (1, h, w, 3), device)?;
    Ok(nhwc.permute((0, 3, 1, 2))?.contiguous()?)
}

/// Seeded initial/renoise latent noise `[1, 16, H/8, W/8]` (f32). `step` derives a distinct RNG key
/// per renoise. Deterministic, launch-portable CPU RNG (sc-3673 parity).
fn init_noise(height: u32, width: u32, seed: u64, step: u64, device: &Device) -> Result<Tensor> {
    let (lat_h, lat_w) = (
        (height / SPATIAL_SCALE) as usize,
        (width / SPATIAL_SCALE) as usize,
    );
    let n = LATENT_CHANNELS * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(step));
    let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Ok(
        Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(device)?,
    )
}

/// VAE-decode a final latent `[1, 16, H/8, W/8]` → RGB8 [`Image`]. The z-image `AutoEncoderKL::decode`
/// applies its own `/scaling + shift` un-scale internally; `postprocess_image` maps `[-1, 1]` → u8.
fn decode(vae: &AutoEncoderKL, lat: &Tensor) -> Result<Image> {
    let decoded = vae.decode(lat)?.to_dtype(DType::F32)?; // [1, 3, H, W] in [-1, 1]
    let img = postprocess_image(&decoded)?.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "boogu: expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

// ── Static-v1 flow-match schedule (pure math; port of mlx-gen-boogu's pipeline helpers) ──────────

/// Static-v1 time-shift parameters from the snapshot `scheduler/scheduler_config.json`
/// (`base_shift 0.5`, `max_shift 1.15`, `seq_len 4096`). The linear map saturates at `seq_len = 4096`,
/// so `mu` is the constant `max_shift`.
const SEQ_LEN: f64 = 4096.0;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// The Base/Edit static-shift `mu` (`lin_mu(4096) = 1.15`), fed to the epic-7114 scheduler axis so a
/// curated schedule re-shapes σ over the SAME shift the native schedule uses.
fn base_shift_mu() -> f32 {
    lin_mu(SEQ_LEN) as f32
}

/// The Base/Edit native sigma schedule (noise-fraction, descending to a trailing `0.0`) — the
/// `OneMinusSigma` view of the v1 shifted clean-fraction timesteps (`σ_i = 1 − ts_i`).
fn base_native_sigmas(steps: usize) -> Vec<f32> {
    build_timesteps_v1(steps)
        .iter()
        .map(|&t| 1.0 - t as f32)
        .collect()
}

/// Build the static-v1 shifted timestep schedule plus the trailing `1.0` (length `steps + 1`).
fn build_timesteps_v1(steps: usize) -> Vec<f64> {
    let mu = lin_mu(SEQ_LEN);
    let mut ts: Vec<f64> = (0..steps)
        .map(|i| time_shift_v1(i as f64 / steps as f64, mu))
        .collect();
    ts.push(1.0);
    ts
}

/// Reference `_get_lin_function(x1=256,y1=base_shift,x2=4096,y2=max_shift)(seq_len)` → `mu`.
fn lin_mu(seq_len: f64) -> f64 {
    let (x1, y1, x2, y2) = (256.0, BASE_SHIFT, 4096.0, MAX_SHIFT);
    let m = (y2 - y1) / (x2 - x1);
    let b = y1 - m * x1;
    m * seq_len + b
}

/// Reference `_time_shift_v1(t, mu, sigma=1.0)`: `t1 = 1 − t` (clipped); `y = e^mu / (e^mu + (1/t1 − 1))`;
/// return `1 − y`.
fn time_shift_v1(t: f64, mu: f64) -> f64 {
    let eps = 1e-8;
    let t1 = (1.0 - t).clamp(eps, 1.0 - eps);
    let num = mu.exp();
    let denom = num + (1.0 / t1 - 1.0);
    1.0 - num / denom
}

/// DMD sigma schedule: `linspace(conditioning_sigma, 1.0, steps+1)[:-1]` — `steps` ascending
/// **clean-fraction** sigmas from `conditioning_sigma` toward (but excluding) `1.0`.
fn dmd_sigmas(conditioning_sigma: f32, steps: usize) -> Vec<f32> {
    let span = 1.0 - conditioning_sigma;
    (0..steps)
        .map(|k| conditioning_sigma + span * (k as f32) / (steps as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_mu_is_static_max_shift() {
        // `base_shift_mu()` is f32, so the f64 round-trip carries ~1e-7 rounding.
        assert!((base_shift_mu() as f64 - MAX_SHIFT).abs() < 1e-5);
    }

    #[test]
    fn base_sigmas_descend_to_zero() {
        let s = base_native_sigmas(50);
        assert_eq!(s.len(), 51);
        assert!((s[50]).abs() < 1e-6, "terminal sigma must be 0");
        for w in s.windows(2) {
            assert!(w[0] >= w[1], "sigmas must be non-increasing: {s:?}");
        }
    }

    #[test]
    fn dmd_grid_is_ascending_clean_fraction() {
        let s = dmd_sigmas(0.001, 4);
        assert_eq!(s.len(), 4);
        assert!((s[0] - 0.001).abs() < 1e-6);
        for w in s.windows(2) {
            assert!(w[1] > w[0], "dmd sigmas ascend: {s:?}");
        }
        assert!(s[3] < 1.0);
    }
}
