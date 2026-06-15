//! # candle-gen-svd
//!
//! **Stable Video Diffusion (img2vid-xt)** image-to-video provider for [`candle-gen`](candle_gen) —
//! the candle (Windows/CUDA) sibling of `mlx-gen-svd`. SVD has **no** `candle-transformers` reference:
//! the `UNetSpatioTemporalConditionModel` ([`unet`]), the `AutoencoderKLTemporalDecoder` temporal VAE
//! ([`vae`], built on a from-scratch causal conv3d since candle ships none), the OpenCLIP ViT-H
//! `CLIPVisionModelWithProjection` image encoder ([`image_encoder`]), and the EDM `EulerDiscreteScheduler`
//! ([`scheduler`]) are all ported here from the `stabilityai/stable-video-diffusion-img2vid-xt`
//! checkpoint.
//!
//! **img2vid (sc-5493):** a single [`Conditioning::Reference`] image is CLIP-encoded for the UNet
//! cross-attention conditioning and (noise-augmented) VAE-encoded into the per-frame image latent that
//! is channel-concatenated into the UNet input. `motion_bucket_id` / `noise_aug_strength` /
//! `conditioning_fps` / `decode_chunk_size` / `frames` / `steps` / the CFG ceiling come from the
//! request; `req.fps` is the decoupled output/playback cadence.
//!
//! **Dtypes:** the UNet + image encoder run **fp16** (SVD's production dtype); the VAE always stays
//! **f32** (`force_upcast=True`). `backend = "candle"`, `mac_only = false`.

pub mod config;
pub mod conv3d;
pub mod embeddings;
pub mod image_encoder;
pub mod pipeline;
pub mod preprocess;
pub mod scheduler;
pub mod transformer;
pub mod unet;
pub mod vae;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use config::{
    ImageEncoderConfig, SchedulerConfig, UnetConfig, VaeConfig, MODEL_ID, SIZE_ALIGN, VAE_SCALE,
};
use image_encoder::SvdImageEncoder;
use pipeline::SvdParams;
use scheduler::EdmSchedule;
use unet::SvdUnet;
use vae::SvdVae;

/// OpenCLIP ViT-H image-normalization mean/std (the SVD `feature_extractor`).
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];
const CLIP_SIZE: usize = 224;

/// The lazily-loaded SVD components (image encoder + VAE + UNet), cached behind the generator's
/// `Mutex` for the worker's `Arc<dyn Generator>` reuse.
#[derive(Clone)]
struct Components {
    image_encoder: Arc<SvdImageEncoder>,
    vae: Arc<SvdVae>,
    unet: Arc<SvdUnet>,
}

impl Components {
    /// Load every component from a checkpoint snapshot dir (`vae/` + `unet/` + `image_encoder/`). The
    /// UNet + image encoder run **fp16** on CUDA (SVD's production dtype) / **f32** on CPU; the VAE
    /// always stays f32 (`force_upcast=True`).
    fn load(root: &Path, device: &Device) -> CResult<Self> {
        let dense = if device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        let vae = SvdVae::new(
            &VaeConfig::default(),
            component_vb(root, "vae", "diffusion_pytorch_model", DType::F32, device)?,
        )?;
        let unet = SvdUnet::new(
            &UnetConfig::default(),
            component_vb(root, "unet", "diffusion_pytorch_model", dense, device)?,
        )?;
        let image_encoder = SvdImageEncoder::new(
            &ImageEncoderConfig::default(),
            component_vb(root, "image_encoder", "model", dense, device)?,
        )?;
        Ok(Self {
            image_encoder: Arc::new(image_encoder),
            vae: Arc::new(vae),
            unet: Arc::new(unet),
        })
    }
}

/// Build a `VarBuilder` over a component subdir's safetensors, preferring the on-disk `.fp16` variant
/// when loading at `DType::F16` (half the load IO).
fn component_vb(
    root: &Path,
    sub: &str,
    stem: &str,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "svd_xt: snapshot is missing the {sub}/ dir (expected a \
             stable-video-diffusion-img2vid-xt snapshot at {})",
            root.display()
        )));
    }
    let fp16 = dir.join(format!("{stem}.fp16.safetensors"));
    let full = dir.join(format!("{stem}.safetensors"));
    let path = if dtype == DType::F16 && fp16.exists() {
        fp16
    } else if full.exists() {
        full
    } else if fp16.exists() {
        fp16
    } else {
        return Err(CandleError::Msg(format!(
            "svd_xt: no {stem}.safetensors in {sub}/ (at {})",
            dir.display()
        )));
    };
    // SAFETY: mmap of read-only weight files; standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    Ok(vb)
}

/// Upper bound on a `Reference` image's dimensions (caps host allocations on the input buffer + the
/// resize's f32 intermediates). 8192 is far above any real photo (F-164).
const MAX_REFERENCE_DIM: u32 = 8192;
/// Upper bound on requested output `frames` — SVD-XT is the 25-frame variant; per-frame latents +
/// `added_time_ids` scale linearly, so cap the allocation.
const MAX_FRAMES: u32 = 64;
/// Upper bound on requested denoise `steps` (guards a pathological value pinning the GPU).
const MAX_STEPS: u32 = 200;

/// SVD-XT img2vid descriptor — image→video via a single `Reference`, a frame-wise guidance ramp
/// (`req.guidance` overrides the ceiling), no negative prompt / sampler / scheduler / LoRA / quant.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "svd",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: false,
            supports_lokr: false,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 1024,
            max_count: 1,
            mac_only: false,
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supported_quants: &[],
        },
    }
}

/// The lazy candle SVD generator. Components (image encoder + VAE + UNet) are loaded on first
/// `generate` and cached behind a `Mutex` for the worker's `Arc<dyn Generator>` cache.
pub struct SvdGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

/// The SVD-specific request validation the core `Capabilities::validate_request` leaves to each model
/// (size alignment + the allocation/compute knob bounds) — F-165.
fn validate_output_params(req: &GenerationRequest) -> gen_core::Result<()> {
    if !req.width.is_multiple_of(SIZE_ALIGN) || !req.height.is_multiple_of(SIZE_ALIGN) {
        return Err(gen_core::Error::Msg(format!(
            "svd_xt: {}x{} must be a multiple of {SIZE_ALIGN} (VAE 8× × UNet 8×)",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        if frames == 0 || frames > MAX_FRAMES {
            return Err(gen_core::Error::Msg(format!(
                "svd_xt: frames {frames} out of range 1..={MAX_FRAMES}"
            )));
        }
    }
    if let Some(steps) = req.steps {
        if steps == 0 || steps > MAX_STEPS {
            return Err(gen_core::Error::Msg(format!(
                "svd_xt: steps {steps} out of range 1..={MAX_STEPS}"
            )));
        }
    }
    Ok(())
}

/// Reject a `Reference` image with zero/oversized dims or a buffer that isn't `w*h*3` RGB8 (usize math
/// so the length never wraps — F-164).
fn validate_reference_image(img: &Image) -> gen_core::Result<()> {
    if img.width == 0 || img.height == 0 {
        return Err(gen_core::Error::Msg(format!(
            "svd_xt: reference image has a zero dimension ({}x{})",
            img.width, img.height
        )));
    }
    if img.width > MAX_REFERENCE_DIM || img.height > MAX_REFERENCE_DIM {
        return Err(gen_core::Error::Msg(format!(
            "svd_xt: reference image {}x{} exceeds the {MAX_REFERENCE_DIM}px dimension cap",
            img.width, img.height
        )));
    }
    if img.pixels.len() != img.width as usize * img.height as usize * 3 {
        return Err(gen_core::Error::Msg(format!(
            "svd_xt: reference image pixel buffer {} != {}x{}x3 (RGB8)",
            img.pixels.len(),
            img.width,
            img.height
        )));
    }
    Ok(())
}

impl SvdGenerator {
    /// Resolve the single conditioning reference image (image→video input).
    fn reference<'a>(&self, req: &'a GenerationRequest) -> gen_core::Result<&'a Image> {
        req.conditioning
            .iter()
            .find_map(|c| match c {
                Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            })
            .ok_or_else(|| {
                gen_core::Error::Msg("svd_xt: image→video requires a Reference image".into())
            })
    }

    /// Lazily load + cache the SVD components.
    fn components(&self) -> CResult<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("svd components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = Components::load(&self.root, &self.device)?;
        *guard = Some(c.clone());
        Ok(c)
    }

    /// CLIP `image_embeds` `[1, 1, 1024]` from the reference: diffusers `_resize_with_antialiasing` to
    /// 224 (gaussian-blur + align-corners bicubic, in `[-1,1]`) → CLIP mean/std normalize.
    fn clip_embeds(&self, comps: &Components, img: &Image) -> CResult<Tensor> {
        let unit = preprocess::resize_with_antialiasing_unit(
            &img.pixels,
            img.height as usize,
            img.width as usize,
            CLIP_SIZE,
            CLIP_SIZE,
        ); // HWC [224,224,3] in [0,1]
        let plane = CLIP_SIZE * CLIP_SIZE;
        let mut chw = vec![0f32; 3 * plane];
        for y in 0..CLIP_SIZE {
            for x in 0..CLIP_SIZE {
                for c in 0..3 {
                    let v = unit[(y * CLIP_SIZE + x) * 3 + c];
                    chw[c * plane + y * CLIP_SIZE + x] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
                }
            }
        }
        let pix = Tensor::from_vec(chw, (1, 3, CLIP_SIZE, CLIP_SIZE), &self.device)?;
        let embeds = comps.image_encoder.image_embeds(&pix)?; // [1, 1024]
        let d = embeds.dim(1)?;
        Ok(embeds.reshape((1, 1, d))?)
    }

    /// Per-frame VAE image latent `[1, F, 4, h, w]`: lanczos resize to the output size, scale to
    /// `[-1,1]`, add `noise_aug·N(0,1)`, VAE-encode (`mode()`), repeat over frames.
    #[allow(clippy::too_many_arguments)]
    fn image_latents(
        &self,
        comps: &Components,
        img: &Image,
        height: u32,
        width: u32,
        num_frames: usize,
        noise_aug: f32,
        seed: u64,
    ) -> CResult<Tensor> {
        let (oh, ow) = (height as usize, width as usize);
        let resized = candle_gen::gen_core::imageops::resize_lanczos_u8(
            &img.pixels,
            img.height as usize,
            img.width as usize,
            oh,
            ow,
        ); // HWC [0,255] f32
        let plane = oh * ow;
        let mut chw = vec![0f32; 3 * plane];
        for y in 0..oh {
            for x in 0..ow {
                for c in 0..3 {
                    chw[c * plane + y * ow + x] = resized[(y * ow + x) * 3 + c] / 255.0;
                }
            }
        }
        let unit = Tensor::from_vec(chw, (1, 3, oh, ow), &self.device)?;
        let centered = unit.affine(2.0, -1.0)?; // [-1,1]
        let noise = pipeline::seeded_normal(seed.wrapping_add(7), (1, 3, oh, ow), &self.device)?;
        let augmented = (centered + noise.affine(noise_aug as f64, 0.0)?)?;
        let latent = comps.vae.encode_mode(&augmented)?; // [1, 4, h, w]
        let (b, c, lh, lw) = latent.dims4()?;
        latent
            .reshape((b, 1, c, lh, lw))?
            .broadcast_as((b, num_frames, c, lh, lw))?
            .contiguous()
            .map_err(Into::into)
    }
}

impl Generator for SvdGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Shared capability floor: size range (256..=1024), count, unsupported negative-prompt /
        // true_cfg / sampler / scheduler, and conditioning (`Reference` only). `guidance` IS supported
        // — it overrides the frame-wise CFG ceiling.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        validate_output_params(req)?;
        let img = self.reference(req)?;
        validate_reference_image(img)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let img = self.reference(req)?;
        let comps = self.components()?;

        let mut params = SvdParams::default();
        if let Some(f) = req.frames {
            params.num_frames = f as usize;
            // Default the decode chunk to the full clip unless the request overrides it below.
            params.decode_chunk_size = f as usize;
        }
        if let Some(s) = req.steps {
            params.num_inference_steps = s as usize;
        }
        // `params.fps` is the MOTION-conditioning cadence (from `conditioning_fps`), distinct from
        // `req.fps` (the output/playback cadence applied at return time).
        if let Some(cfps) = req.conditioning_fps {
            params.fps = cfps;
        }
        if let Some(g) = req.guidance {
            params.max_guidance_scale = g;
        }
        if let Some(m) = req.motion_bucket_id {
            params.motion_bucket_id = m;
        }
        if let Some(n) = req.noise_aug_strength {
            params.noise_aug_strength = n;
        }
        if let Some(c) = req.decode_chunk_size {
            params.decode_chunk_size = c as usize;
        }
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);

        // Conditioning.
        let image_embeds = self.clip_embeds(&comps, img)?;
        let image_latents = self.image_latents(
            &comps,
            img,
            req.height,
            req.width,
            params.num_frames,
            params.noise_aug_strength,
            seed,
        )?;
        let atid = pipeline::added_time_ids(&params, &self.device)?;

        // Seeded init noise scaled by `init_noise_sigma`.
        let sched_cfg = SchedulerConfig::default();
        let sched = EdmSchedule::karras(params.num_inference_steps, &sched_cfg);
        let lh = (req.height / VAE_SCALE) as usize;
        let lw = (req.width / VAE_SCALE) as usize;
        let noise = pipeline::create_noise(seed, params.num_frames, lh, lw, &self.device)?;
        let latents = noise
            .affine(sched.init_noise_sigma() as f64, 0.0)
            .map_err(CandleError::from)?;

        let total = params.num_inference_steps as u32;
        on_progress(Progress::Step { current: 0, total });
        let final_latents = pipeline::denoise(
            &comps.unet,
            &sched_cfg,
            &latents,
            &image_embeds,
            &image_latents,
            &atid,
            params.num_frames,
            params.num_inference_steps,
            params.min_guidance_scale,
            params.max_guidance_scale,
            &req.cancel,
            &mut |step| {
                on_progress(Progress::Step {
                    current: step as u32,
                    total,
                })
            },
        )?;

        on_progress(Progress::Decoding);
        let decoded = pipeline::decode(
            &comps.vae,
            &final_latents,
            params.num_frames,
            params.decode_chunk_size,
        )?;
        let frames = pipeline::frames_to_images(&decoded)?;

        Ok(GenerationOutput::Video {
            frames,
            // Output/playback cadence = `req.fps` (decoupled from the motion-conditioning fps); falls
            // back to the conditioning fps when unset.
            fps: req.fps.unwrap_or(params.fps),
            audio: None,
        })
    }
}

/// Construct a lazy candle SVD generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// `stabilityai/stable-video-diffusion-img2vid-xt` snapshot (`vae/` + `unet/` + `image_encoder/`).
/// Adapters / quantization / control overlays are rejected (SVD is image→video only).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "svd_xt: expected a checkpoint directory (vae/ + unet/ + image_encoder/), not a \
                 single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle svd does not support LoRA/LoKr".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle svd does not support quantization".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle svd does not support control / IP-adapter overlays".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(SvdGenerator {
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("svd is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "svd");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.samplers.is_empty());
        assert_eq!(d.capabilities.min_size, 256);
        assert_eq!(d.capabilities.max_size, 1024);
    }

    fn ref_req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            width: w,
            height: h,
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: w,
                    height: h,
                    pixels: vec![0u8; w as usize * h as usize * 3],
                },
                strength: None,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn validate_accepts_img2vid_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        // 1024×576 = 16×64 / 9×64 with a well-formed reference passes.
        assert!(g.validate(&ref_req(1024, 576)).is_ok());
        // Missing reference image.
        assert!(g
            .validate(&GenerationRequest {
                width: 512,
                height: 512,
                ..Default::default()
            })
            .is_err());
        // Unaligned size (not a multiple of 64).
        assert!(g.validate(&ref_req(700, 704)).is_err());
        // Out-of-range frames.
        assert!(g
            .validate(&GenerationRequest {
                frames: Some(MAX_FRAMES + 1),
                ..ref_req(512, 512)
            })
            .is_err());
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("checkpoint directory"), "got: {err}");
    }
}
