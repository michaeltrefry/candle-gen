//! SDXL **img2img / inpaint / outpaint** edit provider (sc-6037, epic 5480) — pixel-conditioned
//! editing on SDXL/RealVisXL, the candle (Windows/CUDA) sibling of the `mlx-gen-sdxl` edit path and
//! the **provider half** that unblocks the worker wiring (sc-5487). It reuses the InstantID/IP-Adapter
//! stack ([`crate::loaders`] + [`crate::denoise`] + [`crate::sampler`]) **without** an IP-Adapter or a
//! ControlNet: the vendored UNet runs through [`UNet2DConditionModel::forward_instantid`] with the IP
//! branch inert (never installed / set), i.e. plain SDXL with the `add_embedding` micro-conditioning.
//!
//! Three modes, selected by the entry point + the request shape (mirroring the mlx `Conditioning`
//! presence):
//!  - **img2img** ([`SdxlEdit::generate`]) — VAE-encode the source to its clean latent mean, add noise
//!    at `start_time = max_time·strength`, denoise the reduced `round(steps·strength)`-step schedule,
//!    decode. Default strength [`DEFAULT_EDIT_STRENGTH`] (0.8).
//!  - **inpaint** ([`SdxlEdit::generate_masked`]) — img2img plus a binarized, nearest-8×-downsampled
//!    latent mask; after each sampler step the kept (black) region is pinned to the source re-noised to
//!    that step's σ: `latents = (1−mask)·init_noised + mask·latents` (white = repaint, black = keep).
//!    Default strength [`DEFAULT_INPAINT_STRENGTH`] (0.85).
//!  - **outpaint** — provider-side identical to inpaint; the worker pads the canvas and paints the new
//!    border white, then calls [`SdxlEdit::generate_masked`].
//!
//! **Determinism** is the candle-lane contract (sc-3673): one seeded `StdRng` keyed by `seed` draws the
//! init noise (also the fixed noise the inpaint blend re-applies every step), a second keyed by
//! `seed + STEP_RNG_SALT` draws each ancestral step's noise — generation is a pure function of
//! `(seed, request, source[, mask])`, launch-portable. **CFG is uncond-first** (`[negative, prompt]`),
//! matching the InstantID/IP-Adapter ports and the candle txt2img convention. `generate` takes `&self`
//! (no per-call UNet mutation — there is no IP context to set), so one loaded model serves many edits.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::imageops::{resize_lanczos_u8, resize_nearest_u8};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::conditioning::SdxlConditioner;
use crate::denoise::{decode_image, text_time_ids, SPATIAL_SCALE};
use crate::loaders::{load_instantid_unet, load_sdxl_vae, load_sdxl_vae_encoder};
use crate::sampler::EulerAncestralSampler;
use crate::unet::{UNet2DConditionModel, VaeMomentsEncoder};
use crate::AutoEncoderKL;

/// The edit compute dtype — fp16, matching the production SDXL path (the f16-stable VAE + UNet).
const DTYPE: DType = DType::F16;

/// img2img default denoise strength — the worker's plain-edit value (the torch `SdxlDiffusersAdapter`
/// uses ~0.6–0.8); 0.8 keeps prompt freedom while still honoring the source structure.
pub const DEFAULT_EDIT_STRENGTH: f32 = 0.8;
/// inpaint / outpaint default strength — higher, so the repaint region is substantially regenerated
/// (the torch adapter's `use_inpaint` / `outpaint` value).
pub const DEFAULT_INPAINT_STRENGTH: f32 = 0.85;

/// Offset so the per-step ancestral-noise RNG stream is distinct from the init-noise stream (init keyed
/// by `seed`, steps by `seed + STEP_RNG_SALT`) — the launch-portable determinism the IP/InstantID/PuLID
/// ports use.
const STEP_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// SDXL works in latent space at /8: both render dims must be multiples of 8.
const SIZE_MULTIPLE: u32 = 8;

/// Paths to the SDXL edit checkpoints — just the SDXL base snapshot (the f16-fix VAE is resolved via
/// `hf-hub`, exactly as the txt2img / IP-Adapter paths). No IP-Adapter / ControlNet / face checkpoints.
pub struct SdxlEditPaths {
    /// SDXL base snapshot dir (`unet/`, `text_encoder{,_2}/`, …).
    pub sdxl_base: PathBuf,
}

/// One SDXL edit request.
#[derive(Clone)]
pub struct SdxlEditRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Classifier-free guidance scale.
    pub guidance: f32,
    /// Denoise strength in `[0, 1]`: the fraction of the schedule run (`start_time = max_time·strength`,
    /// `round(steps·strength)` steps). 1.0 ≈ full regeneration; 0.0 ≈ the source unchanged.
    pub strength: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for SdxlEditRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 5.0,
            strength: DEFAULT_EDIT_STRENGTH,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Loaded SDXL edit model: the vendored SDXL UNet (`add_embedding`, **no** IP-Adapter installed) + the
/// dual-CLIP conditioner + the f16 VAE decoder + the deterministic VAE moments-encoder (img2img init) +
/// the ancestral sampler.
pub struct SdxlEdit {
    conditioner: SdxlConditioner,
    unet: UNet2DConditionModel,
    vae: AutoEncoderKL,
    vae_encoder: VaeMomentsEncoder,
    sampler: EulerAncestralSampler,
    device: Device,
}

impl SdxlEdit {
    /// Load the SDXL backbone: UNet (+ `add_embedding`), dual CLIP, the f16 VAE *decode*, and the f16
    /// VAE *encode* (the deterministic moments-encoder for the img2img init). No IP-Adapter K/V install
    /// and no `set_ip_context`, so [`UNet2DConditionModel::forward_instantid`] runs plain SDXL.
    pub fn load(paths: &SdxlEditPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.sdxl_base.as_path();
        let conditioner = SdxlConditioner::load(root, &device, DTYPE)?;
        let unet = load_instantid_unet(root, &device, DTYPE)?;
        let vae = load_sdxl_vae(&device, DTYPE)?;
        let vae_encoder = load_sdxl_vae_encoder(&device, DTYPE)?;
        Ok(Self {
            conditioner,
            unet,
            vae,
            vae_encoder,
            sampler: EulerAncestralSampler::sdxl(),
            device,
        })
    }

    /// **img2img**: regenerate `source` toward `req.prompt` at `req.strength`.
    pub fn generate(
        &self,
        req: &SdxlEditRequest,
        source: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        self.run(req, source, None, on_progress)
    }

    /// **inpaint / outpaint**: regenerate only the `mask`'s white region (black = keep `source`).
    pub fn generate_masked(
        &self,
        req: &SdxlEditRequest,
        source: &Image,
        mask: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        self.run(req, source, Some(mask), on_progress)
    }

    fn run(
        &self,
        req: &SdxlEditRequest,
        source: &Image,
        mask: Option<&Image>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(CandleError::Msg(format!(
                "sdxl edit: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        let cfg_on = req.guidance > 1.0;

        // Text conditioning (uncond-first under CFG) + the SDXL micro-conditioning time_ids.
        let (conditioning, pooled) = self
            .conditioner
            .encode(&req.prompt, &req.negative, cfg_on)?;
        let batch = conditioning.dim(0)?;
        let time_ids = text_time_ids(batch, &self.device, DTYPE)?;

        // The clean source latent (deterministic mean × 0.13025), then the strength-scaled init noising.
        let x0 = self.encode_source(source, req.width, req.height)?;
        let (_, lat_c, lat_h, lat_w) = x0.dims4()?;
        let strength = req.strength.clamp(0.0, 1.0) as f64;
        let start_time = self.sampler.max_time() * strength;
        let mut init_rng = StdRng::seed_from_u64(req.seed);
        let init_noise = draw_noise(&mut init_rng, lat_c, lat_h, lat_w, &self.device)?;
        let x_t = self.sampler.add_noise(&x0, &init_noise, start_time)?;

        // The latent mask (inpaint / outpaint), if any.
        let latent_mask = match mask {
            Some(m) => Some(encode_mask(m, req.width, req.height, &self.device)?),
            None => None,
        };

        // The reduced schedule: `round(steps·strength)` ancestral steps from `start_time` down to 0.
        let eff = (req.steps as f64 * strength).round() as usize;
        let steps = self.sampler.timesteps(eff, start_time);

        let mut step_rng = StdRng::seed_from_u64(req.seed.wrapping_add(STEP_RNG_SALT));
        let latents = self.denoise_edit(
            x_t,
            &x0,
            &init_noise,
            latent_mask.as_ref(),
            &conditioning,
            &pooled,
            &time_ids,
            req.guidance as f64,
            &steps,
            &mut step_rng,
            &req.cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        decode_image(&self.vae, &latents)
    }

    /// VAE-encode `source` (resized to the render size, LANCZOS, normalized to `[-1,1]` NCHW) to the
    /// clean latent mean `[1, 4, h/8, w/8]` × 0.13025 — the deterministic img2img/inpaint init.
    fn encode_source(&self, source: &Image, width: u32, height: u32) -> Result<Tensor> {
        let (iw, ih) = (source.width as usize, source.height as usize);
        if source.pixels.len() != iw * ih * 3 {
            return Err(CandleError::Msg(format!(
                "sdxl edit: source pixel buffer {} != {iw}x{ih}x3",
                source.pixels.len()
            )));
        }
        let (rw, rh) = (width as usize, height as usize);
        let resized = resize_lanczos_u8(&source.pixels, ih, iw, rh, rw); // HWC f32 [0,255]
                                                                         // [0,255] → [-1,1], then HWC → NCHW.
        let data: Vec<f32> = resized.iter().map(|&v| v / 127.5 - 1.0).collect();
        let hwc = Tensor::from_vec(data, (rh, rw, 3), &self.device)?;
        let nchw = hwc
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?
            .to_dtype(DTYPE)?;
        Ok(self.vae_encoder.encode_mean(&nchw)?)
    }

    /// The edit denoise loop: plain-SDXL `forward_instantid` (IP inert, no controls) + CFG + the
    /// ancestral step, with an optional per-step inpaint blend. Mirrors
    /// [`crate::denoise::denoise_ip_multi_control`] minus the ControlNet/IP machinery (the edit path has
    /// neither), plus the mask compositing.
    #[allow(clippy::too_many_arguments)]
    fn denoise_edit(
        &self,
        mut latents: Tensor,
        x0: &Tensor,
        init_noise: &Tensor,
        mask: Option<&Tensor>,
        conditioning: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
        cfg: f64,
        steps: &[(f64, f64)],
        rng: &mut StdRng,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        // An empty schedule (img2img at strength ≤ 1/steps) is a no-op: the lightly-noised source. For
        // inpaint that leaves the source untouched (no repaint), the honest result of a zero-step edit.
        if steps.is_empty() {
            return Ok(latents);
        }
        let cfg_on = cfg > 1.0;
        let total = steps.len() as u32;
        let (_, lat_c, lat_h, lat_w) = latents.dims4()?;

        for (i, &(t, t_prev)) in steps.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let x_unet = if cfg_on {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };
            // Plain SDXL forward (no control residuals, IP branch inert): the add_embedding
            // micro-conditioning only.
            let eps = self.unet.forward_instantid(
                &x_unet,
                t,
                conditioning,
                pooled,
                time_ids,
                None,
                None,
            )?;
            // Classifier-free guidance: row 0 = uncond, row 1 = cond.
            let eps = if cfg_on {
                let chunks = eps.chunk(2, 0)?;
                (&chunks[0] + ((&chunks[1] - &chunks[0])? * cfg)?)?
            } else {
                eps
            };
            let noise = draw_noise(rng, lat_c, lat_h, lat_w, &self.device)?;
            latents = self.sampler.step(&eps, &latents, t, t_prev, &noise)?;

            // Inpaint: pin the kept (mask=0) region to the source re-noised to this step's σ, leaving
            // the repaint (mask=1) region freely denoised. At the final step (t_prev=0) `init_noised =
            // x0`, so the kept region ends exactly at the source. The 1-channel mask broadcasts over the
            // 4 latent channels.
            if let Some(mask) = mask {
                let init_noised = self.sampler.add_noise(x0, init_noise, t_prev)?;
                let keep = init_noised.broadcast_mul(&mask.affine(-1.0, 1.0)?)?; // (1−mask)·init_noised
                let repaint = latents.broadcast_mul(mask)?; // mask·latents
                latents = (keep + repaint)?;
            }
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }
        Ok(latents)
    }
}

/// Draw a `[1, C, h, w]` unit-normal tensor from the seeded `rng` on CPU (so the draw sequence is
/// device- and launch-independent — sc-3673), then move to `device`. The init noise and each ancestral
/// step's noise come from this.
fn draw_noise(rng: &mut StdRng, c: usize, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let n = c * h * w;
    let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(rng)).collect();
    Ok(Tensor::from_vec(noise, (1, c, h, w), &Device::Cpu)?.to_device(device)?)
}

/// Build the latent inpaint mask `[1, 1, h/8, w/8]` (f16, values 0/1): resize the mask to the render
/// size with **nearest** (no interpolation grays), binarize the luma at 0.5 (white ≥ 0.5 ⇒ 1.0 =
/// repaint, black ⇒ 0.0 = keep), then 8× nearest-downsample (the top-left pixel of each 8×8 block) —
/// matching the mlx `preprocess_mask`. Masks arrive RGB8 (the worker decodes via `load_reference_image`
/// → `to_rgb8`), so the luma is taken over the three channels.
fn encode_mask(mask: &Image, width: u32, height: u32, device: &Device) -> Result<Tensor> {
    let (iw, ih) = (mask.width as usize, mask.height as usize);
    if mask.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "sdxl edit: mask pixel buffer {} != {iw}x{ih}x3",
            mask.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    let resized = resize_nearest_u8(&mask.pixels, ih, iw, rh, rw); // HWC f32 [0,255]
    let scale = SPATIAL_SCALE as usize;
    let (lh, lw) = (rh / scale, rw / scale);
    let mut latent = Vec::with_capacity(lh * lw);
    for ly in 0..lh {
        for lx in 0..lw {
            // The top-left pixel of each 8×8 block (nearest downsample).
            let (py, px) = (ly * scale, lx * scale);
            let idx = (py * rw + px) * 3;
            let luma = 0.299 * resized[idx] + 0.587 * resized[idx + 1] + 0.114 * resized[idx + 2];
            latent.push(if luma / 255.0 >= 0.5 { 1.0f32 } else { 0.0 });
        }
    }
    Ok(Tensor::from_vec(latent, (1, 1, lh, lw), device)?.to_dtype(DTYPE)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unet::{BlockConfig, UNet2DConditionModelConfig};
    use candle_nn::{VarBuilder, VarMap};

    /// The request defaults match the SDXL edit production knobs (1024², 30 steps, strength 0.8).
    #[test]
    fn request_defaults() {
        let r = SdxlEditRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert_eq!(r.strength, DEFAULT_EDIT_STRENGTH);
        assert!(!r.cancel.is_cancelled());
    }

    /// `encode_mask`: a left-half-white / right-half-black RGB mask binarizes + 8×-nearest-downsamples
    /// to a `[1, 1, h/8, w/8]` latent mask whose left column is 1.0 (repaint) and right is 0.0 (keep).
    #[test]
    fn mask_binarize_and_downsample() {
        let dev = Device::Cpu;
        let (w, h) = (16u32, 8u32); // → latent 2×1 (h/8=1, w/8=2)
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = if x < 8 { 255u8 } else { 0u8 }; // left half white (repaint), right black (keep)
                pixels.extend_from_slice(&[v, v, v]);
            }
        }
        let img = Image {
            width: w,
            height: h,
            pixels,
        };
        let m = encode_mask(&img, w, h, &dev).unwrap();
        assert_eq!(m.dims(), &[1, 1, 1, 2]);
        let v = m
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Latent column 0 samples src x=0 (white → 1.0); column 1 samples src x=8 (black → 0.0).
        assert_eq!(v, vec![1.0, 0.0]);
    }

    /// The central design assumption: `forward_instantid` runs as **plain SDXL** when the IP-Adapter is
    /// never installed and no IP context is set (the edit path) — the add_embedding micro-conditioning
    /// alone, no panic on an absent IP branch. Built on a tiny add_embedding UNet (no `install_ip_adapter`).
    #[test]
    fn forward_instantid_runs_without_ip_install() {
        const ADD_TIME_DIM: usize = 8;
        const PROJ_DIM: usize = 32; // pooled(16) + time_ids_len(2)·8
        const CROSS_DIM: usize = 16;
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cfg = UNet2DConditionModelConfig {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig {
                    out_channels: 32,
                    use_cross_attn: None,
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 64,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
            ],
            layers_per_block: 1,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: CROSS_DIM,
            sliced_attention_size: None,
            use_linear_projection: false,
        };
        // add_embedding loaded, but NO install_ip_adapter / set_ip_context — the plain-SDXL edit path.
        let unet = UNet2DConditionModel::new(vb.clone(), 4, 4, false, cfg)
            .unwrap()
            .with_add_embedding(vb, ADD_TIME_DIM, PROJ_DIM)
            .unwrap();

        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap();
        let ehs = Tensor::randn(0f32, 1f32, (1, 5, CROSS_DIM), &dev).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, CROSS_DIM), &dev).unwrap();
        let time_ids = Tensor::randn(0f32, 1f32, (1, 2), &dev).unwrap();

        let eps = unet
            .forward_instantid(&x, 500.0, &ehs, &pooled, &time_ids, None, None)
            .expect("forward_instantid must run plain-SDXL without an installed IP-Adapter");
        assert_eq!(eps.dims(), &[1, 4, 8, 8]);
        assert!(eps
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
}
