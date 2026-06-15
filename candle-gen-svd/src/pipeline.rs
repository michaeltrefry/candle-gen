//! SVD image-to-video pipeline — the `StableVideoDiffusionPipeline` orchestration over the
//! components: a frame-wise CFG denoise loop (EDM v-prediction Euler, image-latent channel-concat)
//! with `guidance_scale = linspace(min, max, num_frames)`; chunked temporal VAE decode → frames.
//! candle port of `mlx-gen-svd`'s `pipeline.rs`. Latents are `[1, F, 4, h, w]` (B=1; CFG doubles to
//! B=2 inside the step). Deterministic CPU-seeded noise (sc-3673 convention, matching candle-gen-wan).

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{CancelFlag, Image};
use candle_gen::{CandleError, Result as CResult};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::config::SchedulerConfig;
use crate::scheduler::{euler_step, scale_model_input, v_pred_denoised, EdmSchedule};
use crate::unet::SvdUnet;
use crate::vae::SvdVae;

/// Image-to-video generation parameters (the `StableVideoDiffusionPipeline.__call__` knobs).
#[derive(Clone, Debug)]
pub struct SvdParams {
    pub num_frames: usize,
    pub num_inference_steps: usize,
    pub min_guidance_scale: f32,
    pub max_guidance_scale: f32,
    /// Motion-conditioning cadence (the `fps_id` SVD was trained on) — distinct from output fps.
    pub fps: u32,
    pub motion_bucket_id: f32,
    pub noise_aug_strength: f32,
    /// Frames decoded per temporal VAE pass (diffusers default = `num_frames`).
    pub decode_chunk_size: usize,
}

impl Default for SvdParams {
    fn default() -> Self {
        Self {
            num_frames: 25,
            num_inference_steps: 25,
            min_guidance_scale: 1.0,
            max_guidance_scale: 3.0,
            fps: 7,
            motion_bucket_id: 127.0,
            noise_aug_strength: 0.02,
            decode_chunk_size: 25,
        }
    }
}

/// Deterministic N(0,1) latent noise `[1, F, 4, h, w]` (f32) — CPU `StdRng` (ChaCha), launch-portable
/// per seed (matches the candle-gen-wan convention).
pub fn create_noise(
    seed: u64,
    num_frames: usize,
    h: usize,
    w: usize,
    device: &Device,
) -> CResult<Tensor> {
    let n = num_frames * 4 * h * w;
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Ok(Tensor::from_vec(data, (1, num_frames, 4, h, w), device)?)
}

/// Deterministic N(0,1) noise of an arbitrary shape (the image-latent noise augmentation).
pub fn seeded_normal(
    seed: u64,
    shape: (usize, usize, usize, usize),
    device: &Device,
) -> CResult<Tensor> {
    let (a, b, c, d) = shape;
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..a * b * c * d)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    Ok(Tensor::from_vec(data, shape, device)?)
}

/// The `added_time_ids` micro-conditioning row `[1, 3]` = `[fps − 1, motion_bucket_id,
/// noise_aug_strength]` (the SVD pipeline reduces fps by 1 — the model was trained on fps−1).
pub fn added_time_ids(params: &SvdParams, device: &Device) -> CResult<Tensor> {
    let v = vec![
        (params.fps as f32) - 1.0,
        params.motion_bucket_id,
        params.noise_aug_strength,
    ];
    Ok(Tensor::from_vec(v, (1, 3), device)?)
}

/// The frame-wise CFG schedule `linspace(min, max, F)` shaped `[1, F, 1, 1, 1]` to broadcast over the
/// `[1, F, 4, h, w]` latents.
fn guidance_schedule(
    num_frames: usize,
    min_g: f32,
    max_g: f32,
    device: &Device,
) -> CResult<Tensor> {
    let f = num_frames.max(1);
    let vals: Vec<f32> = (0..f)
        .map(|i| {
            if f == 1 {
                min_g
            } else {
                min_g + (max_g - min_g) * (i as f32) / ((f - 1) as f32)
            }
        })
        .collect();
    Ok(Tensor::from_vec(vals, (1, f, 1, 1, 1), device)?)
}

/// The frame-wise CFG v-prediction Euler denoise loop. Inputs are the **conditional** rows (`[1, …]`);
/// the uncond CFG branch zeros `image_embeds`/`image_latents` (the diffusers SVD uncond). Returns the
/// final `[1, F, h, w, 4]`-ordered (`[1, F, 4, h, w]`) latents.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    unet: &SvdUnet,
    scheduler: &SchedulerConfig,
    latents: &Tensor,       // [1, F, 4, h, w] (init noise · init_noise_sigma)
    image_embeds: &Tensor,  // [1, ctx, 1024]
    image_latents: &Tensor, // [1, F, 4, h, w]
    added_time_ids: &Tensor,
    num_frames: usize,
    steps: usize,
    min_g: f32,
    max_g: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> CResult<Tensor> {
    let device = latents.device().clone();
    let sched = EdmSchedule::karras(steps, scheduler);

    // CFG conditioning batches (constant across steps): row 0 = uncond (zeros), row 1 = cond.
    let zeros_e = image_embeds.zeros_like()?;
    let embeds2 = Tensor::cat(&[&zeros_e, image_embeds], 0)?; // [2, ctx, 1024]
    let zeros_l = image_latents.zeros_like()?;
    let img_lat2 = Tensor::cat(&[&zeros_l, image_latents], 0)?; // [2, F, 4, h, w]
    let atid2 = Tensor::cat(&[added_time_ids, added_time_ids], 0)?; // [2, 3]
    let guidance = guidance_schedule(num_frames, min_g, max_g, &device)?;

    let mut latents = latents.clone();
    for i in 0..steps {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let sigma = sched.sigmas[i];
        let sigma_next = sched.sigmas[i + 1];
        let t = sched.timesteps[i];

        let scaled = scale_model_input(&latents, sigma)?; // [1, F, 4, h, w]
        let lat2 = Tensor::cat(&[&scaled, &scaled], 0)?; // [2, F, 4, h, w]
        let inp = Tensor::cat(&[&lat2, &img_lat2], 2)?; // [2, F, 8, h, w] (channel concat)

        let pred = unet.forward(&inp, t, &embeds2, &atid2, num_frames)?; // [2, F, 4, h, w]
        let uncond = pred.narrow(0, 0, 1)?;
        let cond = pred.narrow(0, 1, 1)?;
        // noise_pred = uncond + guidance · (cond − uncond), frame-wise.
        let noise_pred = uncond.add(&guidance.broadcast_mul(&(cond - &uncond)?)?)?;

        let denoised = v_pred_denoised(&noise_pred, &latents, sigma)?;
        latents = euler_step(&latents, &denoised, sigma, sigma_next)?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// Chunked temporal VAE decode (diffusers `decode_latents`): divide by `scaling_factor`, decode in
/// `chunk`-frame windows, concat. `latents` `[1, F, 4, h, w]` → frames `[1, F, 3, H, W]` (roughly
/// `[-1, 1]`; the caller maps to `[0, 1]` for display).
pub fn decode(vae: &SvdVae, latents: &Tensor, num_frames: usize, chunk: usize) -> CResult<Tensor> {
    let (b, f, c, h, w) = latents.dims5()?;
    if b != 1 {
        return Err(CandleError::Msg(format!(
            "svd decode: batch size must be 1 (got {b})"
        )));
    }
    // [1, F, 4, h, w] → [F, 4, h, w], divide by scaling_factor.
    let z = latents
        .reshape((f, c, h, w))?
        .affine(1.0 / vae.scaling_factor() as f64, 0.0)?;
    let chunk = chunk.max(1);

    let mut start = 0usize;
    let mut chunks: Vec<Tensor> = Vec::new();
    while start < num_frames {
        let n = chunk.min(num_frames - start);
        let zc = z.narrow(0, start, n)?; // [n, 4, h, w]
        chunks.push(vae.decode(&zc, n)?); // [n, 3, H, W]
        start += n;
    }
    let refs: Vec<&Tensor> = chunks.iter().collect();
    let frames = Tensor::cat(&refs, 0)?; // [F, 3, H, W]
    let (_, oc, oh, ow) = frames.dims4()?;
    Ok(frames.reshape((1, num_frames, oc, oh, ow))?)
}

/// Decoded frames `[1, F, 3, H, W]` (roughly `[-1, 1]`) → `Vec<Image>` (`clip(x·0.5+0.5)·255`).
pub fn frames_to_images(decoded: &Tensor) -> CResult<Vec<Image>> {
    let u8s = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .to_device(&Device::Cpu)?;
    let (_b, f, c, h, w) = u8s.dims5()?;
    debug_assert_eq!(c, 3);
    let frames = u8s.squeeze(0)?; // [F, 3, H, W]
    let mut out = Vec::with_capacity(f);
    for fi in 0..f {
        let frame = frames.narrow(0, fi, 1)?.squeeze(0)?; // [3, H, W]
        let pixels = frame.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        });
    }
    Ok(out)
}
