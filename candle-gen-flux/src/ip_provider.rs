//! The XLabs FLUX **IP-Adapter** provider (sc-5872, epic 5480) — reference-image (identity)
//! conditioning on FLUX.1 [dev]/[schnell], the candle (Windows/CUDA) sibling of `mlx-gen-flux`'s XLabs
//! IP path. It composes the reused FLUX text encoders / VAE / flow-match schedule with the forked DiT
//! ([`crate::ip_dit::IpFlux`], the only FLUX DiT with an IP seam) + the XLabs adapter
//! ([`crate::ip_adapter`]) + the pooled CLIP-ViT-L image encoder ([`crate::ip_image_encoder`]).
//!
//! **Single distilled forward** (no true-CFG): FLUX is guidance/timestep-distilled, so — like the
//! candle txt2img path — each denoise step is a single DiT forward (dev embeds the guidance scalar;
//! schnell ignores it), with the XLabs IP residual injected per double block. The reference's identity
//! tokens are computed **once** (constant across the denoise) and bound into a [`FluxIpInjector`] at
//! `ip_adapter_scale`; at `scale = 0` the forked DiT is byte-identical to the stock FLUX path — the
//! no-IP arm of the validation ablation ([`crate::ip_validate`]).
//!
//! The provider is a plain struct driven **directly** by the worker (a bespoke reference stream, like
//! `candle_gen_sdxl::IpAdapterSdxl`), not a gen-core-registered [`Generator`](gen_core::Generator) — the
//! registered `flux1_*` descriptors stay txt2img-only.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::text_model::ClipTextTransformer;
use candle_transformers::models::flux::autoencoder::AutoEncoder;
use candle_transformers::models::flux::sampling::{get_schedule, State};
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};
use candle_gen_sdxl::weights::Weights;

use crate::ip_adapter::{FluxIpAdapter, FluxIpInjector};
use crate::ip_dit::IpFlux;
use crate::ip_image_encoder::FluxIpImageEncoder;
use crate::pipeline::{ae_config, clip_config, decode_latents, encode_text, flux_config};
use crate::Variant;

/// FLUX runs at bf16.
const DTYPE: DType = DType::BF16;
/// FLUX latent channel count (the raw VAE latent / initial noise; the DiT packs it 2×2 to 64).
const LATENT_CHANNELS: usize = 16;
/// FLUX dev's resolution-dependent flow-match time-shift endpoints (matching the txt2img pipeline).
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Default `ip_adapter_scale` for the XLabs FLUX IP-Adapter (mlx-gen-flux `DEFAULT_IP_SCALE`).
pub const DEFAULT_IP_SCALE: f32 = 0.7;

/// Paths to the FLUX IP-Adapter checkpoints.
pub struct IpAdapterFluxPaths {
    /// The black-forest-labs FLUX.1 snapshot dir (`flux1-{dev,schnell}.safetensors`, `ae.safetensors`,
    /// `text_encoder/`, `text_encoder_2/`, `tokenizer_2/`). The variant is detected from which DiT file
    /// is present.
    pub flux_base: PathBuf,
    /// The XLabs adapter (`XLabs-AI/flux-ip-adapter` `ip_adapter.safetensors`: `ip_adapter_proj_model.*`
    /// + `double_blocks.{0..18}.processor.ip_adapter_double_stream_{k,v}_proj.*`).
    pub ip_adapter: PathBuf,
    /// The CLIP ViT-L/14 image encoder (`openai/clip-vit-large-patch14`) — a dir (`model.safetensors`)
    /// or the file directly.
    pub image_encoder: PathBuf,
}

/// One FLUX IP-Adapter generation request.
#[derive(Clone)]
pub struct IpAdapterFluxRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Guidance scale — embedded by the dev DiT, inert on schnell.
    pub guidance: f32,
    /// IP-Adapter scale (the decoupled-cross-attn weight on the reference image tokens).
    pub ip_adapter_scale: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for IpAdapterFluxRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 25,
            guidance: 3.5,
            ip_adapter_scale: DEFAULT_IP_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Resolve the CLIP image-encoder weight file from a dir-or-file path (a file is used directly; a dir
/// resolves `model.safetensors` then `model.fp16.safetensors`).
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
        "flux ip-adapter: CLIP image encoder not found under {} (expected a model.safetensors or a \
         direct .safetensors file)",
        path.display()
    )))
}

/// Detect the FLUX variant from the snapshot by which DiT checkpoint is present (dev preferred if both).
fn detect_variant(flux_base: &Path) -> Result<Variant> {
    if flux_base.join(Variant::Dev.transformer_file()).is_file() {
        Ok(Variant::Dev)
    } else if flux_base
        .join(Variant::Schnell.transformer_file())
        .is_file()
    {
        Ok(Variant::Schnell)
    } else {
        Err(CandleError::Msg(format!(
            "flux ip-adapter: no flux1-dev/flux1-schnell .safetensors in {} (expected a \
             black-forest-labs FLUX.1 snapshot)",
            flux_base.display()
        )))
    }
}

/// mmap a [`VarBuilder`] over `files` at `dtype`/`device`, erroring if any is missing.
fn mmap_vb(files: &[PathBuf], dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    for f in files {
        if !f.is_file() {
            return Err(CandleError::Msg(format!(
                "flux ip-adapter snapshot is missing {}",
                f.display()
            )));
        }
    }
    // SAFETY: mmap of read-only weight files; standard candle loading path.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(files, dtype, device)? };
    Ok(vb)
}

/// Sorted list of every `.safetensors` in `dir` (sharded T5 checkpoints). Errors if none are found.
fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("flux ip-adapter: read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "flux ip-adapter: no .safetensors found in {}",
            dir.display()
        )));
    }
    Ok(files)
}

/// The loaded FLUX IP-Adapter model: the reused FLUX text encoders + VAE, the forked IP DiT, the XLabs
/// adapter, and the CLIP ViT-L image encoder.
pub struct IpAdapterFlux {
    variant: Variant,
    /// The snapshot root (for the T5 tokenizer in `encode_text`).
    root: PathBuf,
    device: Device,
    dtype: DType,
    clip: ClipTextTransformer,
    /// Behind a `Mutex` because `T5EncoderModel::forward` takes `&mut self` while `generate` is `&self`;
    /// locked only for the once-per-request text encode.
    t5: Mutex<T5EncoderModel>,
    transformer: IpFlux,
    vae: AutoEncoder,
    ip_encoder: FluxIpImageEncoder,
    adapter: FluxIpAdapter,
}

impl IpAdapterFlux {
    /// Load the FLUX backbone (text encoders + forked DiT + VAE) + the XLabs adapter + the CLIP ViT-L
    /// image encoder from a FLUX snapshot, the XLabs `ip_adapter.safetensors`, and a CLIP image encoder.
    pub fn load(paths: &IpAdapterFluxPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let dtype = DTYPE;
        let root = paths.flux_base.clone();
        let variant = detect_variant(&root)?;

        // CLIP-L (text) under `text_encoder/`.
        let clip_vb = mmap_vb(
            &[root.join("text_encoder/model.safetensors")],
            dtype,
            &device,
        )?;
        let clip = ClipTextTransformer::new(clip_vb.pp("text_model"), &clip_config())?;

        // T5-XXL under `text_encoder_2/` (sharded; config.json alongside).
        let t5_dir = root.join("text_encoder_2");
        let t5_cfg: T5Config = {
            let cfg = std::fs::read_to_string(t5_dir.join("config.json")).map_err(|e| {
                CandleError::Msg(format!(
                    "flux ip-adapter: read text_encoder_2/config.json: {e}"
                ))
            })?;
            serde_json::from_str(&cfg).map_err(|e| {
                CandleError::Msg(format!("flux ip-adapter: parse T5 config.json: {e}"))
            })?
        };
        let t5_vb = mmap_vb(&safetensors_in(&t5_dir)?, dtype, &device)?;
        let t5 = T5EncoderModel::load(t5_vb, &t5_cfg)?;

        // The forked FLUX DiT (the IP seam) from the root BFL checkpoint.
        let dit_vb = mmap_vb(&[root.join(variant.transformer_file())], dtype, &device)?;
        let transformer = IpFlux::new(&flux_config(variant), dit_vb)?;

        // FLUX AutoEncoder (`ae.safetensors`).
        let vae_vb = mmap_vb(&[root.join("ae.safetensors")], dtype, &device)?;
        let vae = AutoEncoder::new(&ae_config(variant), vae_vb)?;

        // XLabs adapter weights (`ip_adapter.safetensors`).
        let ipa = Weights::from_file(&paths.ip_adapter, &device, dtype).map_err(|e| {
            CandleError::Msg(format!(
                "flux ip-adapter: load adapter {:?}: {e}",
                paths.ip_adapter
            ))
        })?;
        let adapter = FluxIpAdapter::from_weights(&ipa)?;
        if adapter.num_blocks() != transformer.num_double_blocks() {
            return Err(CandleError::Msg(format!(
                "flux ip-adapter: adapter has {} double-block pairs but the DiT has {} double blocks",
                adapter.num_blocks(),
                transformer.num_double_blocks()
            )));
        }

        // CLIP ViT-L/14 image encoder (`vision_model.*` + `visual_projection.*`).
        let enc_path = resolve_image_encoder(&paths.image_encoder)?;
        let enc_w = Weights::from_file(&enc_path, &device, dtype).map_err(|e| {
            CandleError::Msg(format!(
                "flux ip-adapter: load CLIP image encoder {enc_path:?}: {e}"
            ))
        })?;
        let ip_encoder = FluxIpImageEncoder::from_weights(&enc_w)?;

        Ok(Self {
            variant,
            root,
            device,
            dtype,
            clip,
            t5: Mutex::new(t5),
            transformer,
            vae,
            ip_encoder,
            adapter,
        })
    }

    /// Reference-image T2I: condition the FLUX generation on `reference`'s CLIP-ViT-L identity tokens at
    /// `req.ip_adapter_scale` (a single distilled forward per step — no true-CFG).
    pub fn generate(
        &self,
        req: &IpAdapterFluxRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // Conditioning: text (T5 seq + CLIP pooled) and the reference image tokens (computed once).
        let (t5_emb, clip_emb) = encode_text(
            self.variant,
            &self.root,
            &self.device,
            self.dtype,
            &self.clip,
            &self.t5,
            &req.prompt,
        )?;
        let embeds = self
            .ip_encoder
            .image_embeds(reference)?
            .to_dtype(self.dtype)?;
        let tokens = self.adapter.tokens(&embeds)?;
        let injector = FluxIpInjector::new(&self.adapter, tokens, req.ip_adapter_scale as f64);

        // candle's get_noise geometry: latent is /8 of a multiple-of-16 request.
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;
        let n = LATENT_CHANNELS * lat_h * lat_w;
        // sc-3673 parity: deterministic, launch-portable CPU-seeded initial noise.
        let mut rng = StdRng::seed_from_u64(req.seed);
        let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;

        let state = State::new(&t5_emb, &clip_emb, &noise)?;
        let timesteps = if self.variant.is_dev() {
            get_schedule(req.steps, Some((state.img.dim(1)?, BASE_SHIFT, MAX_SHIFT)))
        } else {
            get_schedule(req.steps, None)
        };
        let guidance: f64 = if self.variant.supports_guidance() {
            req.guidance as f64
        } else {
            0.0
        };

        let latents = self.denoise(
            &state,
            &timesteps,
            guidance,
            &injector,
            &req.cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        decode_latents(&self.vae, &latents, req.height as usize, req.width as usize)
    }

    /// The flow-match Euler denoise with the XLabs IP injector — the txt2img `denoise` calling the
    /// forked [`IpFlux::forward`] (`Some(injector)`) instead of the stock FLUX. `img += pred·(t_prev −
    /// t_curr)` over the **descending** schedule.
    fn denoise(
        &self,
        state: &State,
        timesteps: &[f64],
        guidance: f64,
        injector: &FluxIpInjector,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let guidance_t = Tensor::full(guidance as f32, b_sz, &self.device)?;
        let total = timesteps.len().saturating_sub(1) as u32;
        let mut img = state.img.clone();
        for (i, window) in timesteps.windows(2).enumerate() {
            if cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let (t_curr, t_prev) = (window[0], window[1]);
            let t_vec = Tensor::full(t_curr as f32, b_sz, &self.device)?;
            let pred = self.transformer.forward(
                &img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &t_vec,
                &state.vec,
                Some(&guidance_t),
                Some(injector),
            )?;
            img = (img + (pred * (t_prev - t_curr))?)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }
        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the FLUX dev IP-Adapter knobs (1024², 25 steps, guidance 3.5, ip 0.7).
    #[test]
    fn request_defaults() {
        let r = IpAdapterFluxRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 25);
        assert_eq!(r.guidance, 3.5);
        assert_eq!(r.ip_adapter_scale, DEFAULT_IP_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    /// `resolve_image_encoder`: a directory resolves `model.safetensors`; a missing dir errors loudly;
    /// a direct file is used as-is.
    #[test]
    fn image_encoder_resolution() {
        let dir = std::env::temp_dir().join(format!("flux_ip_enc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(resolve_image_encoder(&dir).is_err());
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_image_encoder(&dir).unwrap(), f);
        assert_eq!(resolve_image_encoder(&f).unwrap(), f);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `detect_variant` keys off the DiT checkpoint filename and errors when neither is present.
    #[test]
    fn variant_detection() {
        let dir = std::env::temp_dir().join(format!("flux_ip_var_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(detect_variant(&dir).is_err());
        std::fs::write(dir.join(Variant::Schnell.transformer_file()), b"x").unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Schnell);
        std::fs::write(dir.join(Variant::Dev.transformer_file()), b"x").unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Dev); // dev preferred if both
        let _ = std::fs::remove_dir_all(&dir);
    }
}
