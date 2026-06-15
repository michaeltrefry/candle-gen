//! SDXL **IP-Adapter-Plus** provider (sc-5488, epic 5480) — reference-image (identity) conditioning
//! on SDXL/RealVisXL, the candle (Windows/CUDA) sibling of the `mlx-gen-sdxl` IP-Adapter path. It is
//! the [`crate::ip_adapter`] + [`crate::denoise`] stack the InstantID port (sc-5491) built, composed
//! **without** a face embedder and **without** a ControlNet:
//!
//! - the reference image's identity tokens come from the **CLIP ViT-H/14 image encoder**
//!   ([`ClipVisionEncoder`]) → the IP-Adapter "plus" [`Resampler`] (`image_proj.*`), not ArcFace;
//! - the decoupled cross-attention K/V (`ip_adapter.*`) are installed into the vendored UNet exactly
//!   as for InstantID;
//! - the denoise is [`denoise_ip_multi_control`] with an **empty** control set — pure IP, no
//!   IdentityNet/OpenPose residuals.
//!
//! The two candle divergences carried from sc-5491 hold: **CFG is uncond-first** (`[negative,
//! prompt]`), and the IP tokens live on the UNet ([`UNet2DConditionModel::set_ip_context`], set once
//! before the denoise — so [`generate`](IpAdapterSdxl::generate) takes `&mut self`). The one
//! IP-Adapter-specific difference from InstantID: the uncond IP row is **literal zero tokens**
//! ([`IpImageEncoder::zeros_tokens`]), not `Resampler(zeros)`.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::denoise::{
    decode_image, denoise_ip_multi_control, seeded_prior, text_time_ids, Denoiser,
};
use crate::ip_adapter::{load_ip_kv_pairs, IpImageEncoder, Resampler, ResamplerConfig};
use crate::loaders::{load_instantid_unet, load_sdxl_vae};
use crate::sampler::EulerAncestralSampler;
use crate::unet::UNet2DConditionModel;
use crate::vision_encoder::{check_layer_count, ClipVisionEncoder, VisionConfig};
use crate::weights::Weights;
use crate::{conditioning::SdxlConditioner, AutoEncoderKL};

/// The IP-Adapter compute dtype — fp16, matching the production SDXL path (the VAE is the f16-stable
/// `madebyollin/sdxl-vae-fp16-fix`; the CLIP image encoder runs at this dtype too).
const DTYPE: DType = DType::F16;

/// Default `ip_adapter_scale` for SDXL IP-Adapter-Plus (the worker's `ipAdapterScale` default, matching
/// the torch `SdxlDiffusersAdapter`).
pub const DEFAULT_IP_ADAPTER_SCALE: f32 = 0.7;

/// A fixed offset so the per-step ancestral-noise RNG stream is distinct from the prior-noise stream
/// (prior keyed by `seed`, steps by `seed + STEP_RNG_SALT`) — the same launch-portable determinism the
/// InstantID port uses.
const STEP_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Paths to the SDXL IP-Adapter-Plus checkpoints.
pub struct IpAdapterSdxlPaths {
    /// SDXL base snapshot dir (`unet/`, `text_encoder{,_2}/`, …).
    pub sdxl_base: PathBuf,
    /// The IP-Adapter-Plus bundle (`ip-adapter-plus_sdxl_vit-h.safetensors`: `image_proj.*` Resampler +
    /// `ip_adapter.*` K/V pairs).
    pub ip_adapter: PathBuf,
    /// The CLIP ViT-H/14 image encoder — a dir (`model(.fp16).safetensors`) or the file directly
    /// (`h94/IP-Adapter` `models/image_encoder`).
    pub image_encoder: PathBuf,
}

/// One SDXL IP-Adapter generation request.
#[derive(Clone)]
pub struct IpAdapterSdxlRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Classifier-free guidance scale.
    pub guidance: f32,
    /// IP-Adapter scale (the decoupled cross-attn weight on the image tokens).
    pub ip_adapter_scale: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for IpAdapterSdxlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 5.0,
            ip_adapter_scale: DEFAULT_IP_ADAPTER_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Resolve the CLIP image-encoder weight file from a dir-or-file path: a file is used directly; a dir
/// resolves `model.safetensors` then `model.fp16.safetensors` (the diffusers `image_encoder/` layout).
fn resolve_image_encoder(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    for name in ["model.safetensors", "model.fp16.safetensors"] {
        let p = path.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "ip-adapter: CLIP image encoder not found under {} (expected a model.safetensors or a \
         direct .safetensors file)",
        path.display()
    )))
}

/// Loaded SDXL IP-Adapter model: the vendored SDXL UNet (with the IP K/V pairs installed + the
/// `add_embedding` head) + the dual-CLIP conditioner + the CLIP image encoder/Resampler token source +
/// the f16 VAE + the ancestral sampler.
pub struct IpAdapterSdxl {
    conditioner: SdxlConditioner,
    unet: UNet2DConditionModel,
    ip_encoder: IpImageEncoder,
    vae: AutoEncoderKL,
    sampler: EulerAncestralSampler,
    device: Device,
}

impl IpAdapterSdxl {
    /// Load the SDXL backbone + dual-CLIP conditioner + CLIP ViT-H image encoder + IP-Adapter-Plus
    /// Resampler, installing the decoupled-cross-attn K/V pairs into the UNet.
    pub fn load(paths: &IpAdapterSdxlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.sdxl_base.as_path();

        let conditioner = SdxlConditioner::load(root, &device, DTYPE)?;
        let mut unet = load_instantid_unet(root, &device, DTYPE)?;

        // IP-Adapter-Plus bundle: the Resampler (`image_proj.*`) + the decoupled K/V pairs
        // (`ip_adapter.*`), both at the UNet dtype.
        let ipa = Weights::from_file(&paths.ip_adapter, &device, DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "ip-adapter: load bundle {:?}: {e}",
                paths.ip_adapter
            ))
        })?;
        let resampler =
            Resampler::from_weights(&ipa, "image_proj", &ResamplerConfig::plus_sdxl_vit_h())?;
        unet.install_ip_adapter(load_ip_kv_pairs(&ipa)?)?;

        // CLIP ViT-H/14 image encoder (`vision_model.*`).
        let enc_cfg = VisionConfig::vit_h_14();
        let enc_path = resolve_image_encoder(&paths.image_encoder)?;
        let enc_w = Weights::from_file(&enc_path, &device, DTYPE).map_err(|e| {
            CandleError::Msg(format!(
                "ip-adapter: load CLIP image encoder {enc_path:?}: {e}"
            ))
        })?;
        check_layer_count(&enc_w, &enc_cfg)?;
        let encoder = ClipVisionEncoder::from_weights(&enc_w, &enc_cfg)?;
        let ip_encoder = IpImageEncoder::new(encoder, resampler, enc_cfg.image_size);

        let vae = load_sdxl_vae(&device, DTYPE)?;
        Ok(Self {
            conditioner,
            unet,
            ip_encoder,
            vae,
            sampler: EulerAncestralSampler::sdxl(),
            device,
        })
    }

    /// Reference-image T2I: condition the SDXL generation on `reference`'s CLIP-ViT-H identity tokens
    /// at `req.ip_adapter_scale` (no ControlNet — pure IP).
    pub fn generate(
        &mut self,
        req: &IpAdapterSdxlRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let cfg_on = req.guidance > 1.0;

        // Everything that borrows `&self`, computed into owned values BEFORE the `&mut self.unet`
        // `set_ip_context` (so the disjoint-field borrows don't overlap — the InstantID pattern).
        let (conditioning, pooled) = self
            .conditioner
            .encode(&req.prompt, &req.negative, cfg_on)?;
        let batch = conditioning.dim(0)?;
        let time_ids = text_time_ids(batch, &self.device, DTYPE)?;
        let ip_tokens = self.ip_tokens(reference, cfg_on)?;
        let prior = self.seeded_prior_with(req.seed, req.width, req.height)?;

        // Set the IP image tokens on the UNet (constant across the denoise — phase 2c/2e design).
        self.unet
            .set_ip_context(Some(&ip_tokens), req.ip_adapter_scale as f64)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &self.sampler,
        };
        let steps = self.sampler.timesteps(req.steps, self.sampler.max_time());
        let mut rng = StdRng::seed_from_u64(req.seed.wrapping_add(STEP_RNG_SALT));
        let latents = denoise_ip_multi_control(
            &d,
            prior,
            &conditioning,
            &pooled,
            &time_ids,
            req.guidance as f64,
            &steps,
            &mut rng,
            &req.cancel,
            on_progress,
            &[],           // pure IP — no ControlNet branches
            &conditioning, // controlnet_encoder is unused with no controls
        )?;
        on_progress(Progress::Decoding);
        decode_image(&self.vae, &latents)
    }

    /// Build the CFG-batched IP tokens from the reference image. **Uncond-first**: under CFG the uncond
    /// row is literal **zero tokens** (the IP-Adapter convention — `IPAdapter` zeros the image-embed
    /// output, not the Resampler input) stacked *before* the positive row.
    fn ip_tokens(&self, reference: &Image, cfg_on: bool) -> Result<Tensor> {
        let tokens = self.ip_encoder.tokens(reference, &self.device)?; // [1, 16, 2048]
        if cfg_on {
            let zeros = self.ip_encoder.zeros_tokens(&self.device)?;
            Ok(Tensor::cat(&[&zeros, &tokens], 0)?) // uncond (zeros) first, then cond
        } else {
            Ok(tokens)
        }
    }

    /// Seed a `StdRng` and sample the prior latents for a `width × height` render (the prior stream is
    /// keyed by `seed`; the per-step ancestral noise stream by `seed + STEP_RNG_SALT`).
    fn seeded_prior_with(&self, seed: u64, width: u32, height: u32) -> Result<Tensor> {
        let mut rng = StdRng::seed_from_u64(seed);
        seeded_prior(&self.sampler, &mut rng, width, height, &self.device, DTYPE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the SDXL IP-Adapter production knobs (1024², 30 steps, ip scale 0.7).
    #[test]
    fn request_defaults() {
        let r = IpAdapterSdxlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert_eq!(r.ip_adapter_scale, DEFAULT_IP_ADAPTER_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    /// `resolve_image_encoder`: a directory resolves `model.safetensors`; a missing dir errors loudly.
    #[test]
    fn image_encoder_resolution() {
        let dir = std::env::temp_dir().join(format!(
            "candle_ipadapter_enc_{}_{}",
            std::process::id(),
            "t"
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // No weight file yet → error.
        assert!(resolve_image_encoder(&dir).is_err());
        // Create a model.safetensors stand-in → resolves to it.
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_image_encoder(&dir).unwrap(), f);
        // A direct file path is used as-is.
        assert_eq!(resolve_image_encoder(&f).unwrap(), f);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
