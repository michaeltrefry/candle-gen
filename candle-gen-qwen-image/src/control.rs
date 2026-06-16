//! The Qwen-Image **ControlNet (strict pose)** provider (sc-5489, epic 5480) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-qwen-image`'s `QwenImageControl`. Reference-pose conditioning on
//! `qwen_image` via the InstantX `Qwen-Image-ControlNet-Union` checkpoint (DWPose-trained): a rendered
//! OpenPose skeleton is VAE-encoded + packed once, then each denoise step the 5-block control branch
//! ([`QwenControlNet`]) emits 5 per-block residuals injected into the frozen 60-layer base MMDiT
//! (`interval = 12`, scaled by the control scale — [`QwenTransformer::forward_control`]).
//!
//! v1 is **pose-only** (the worker renders the skeleton; this provider takes the conditioning image).
//! True CFG with norm-rescale matches the base txt2img path (the control branch runs on both the pos
//! and neg pass). The provider is a plain struct driven **directly** by the worker (a bespoke stream,
//! like `candle_gen_sdxl::IpAdapterSdxl`), not a gen-core-registered generator — the registered
//! `qwen_image` descriptor stays txt2img-only.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};

use crate::config::{TextEncoderConfig, TransformerConfig, NEGATIVE_FALLBACK};
use crate::pipeline;
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::{QwenControlNet, QwenTransformer};
use crate::vae::{QwenVae, QwenVaeEncoder};

/// The transformer + control branch run bf16 (native dtype); the encoder + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
/// The InstantX `Qwen-Image-ControlNet-Union` ships a 5-block control transformer.
const CONTROL_LAYERS: usize = 5;
/// Default ControlNet conditioning scale (the strict-pose tier).
pub const DEFAULT_CONTROL_SCALE: f32 = 1.0;

/// Paths to the Qwen-Image ControlNet checkpoints.
pub struct QwenControlPaths {
    /// The `Qwen/Qwen-Image` diffusers snapshot dir (`text_encoder/`, `transformer/`, `vae/`,
    /// `tokenizer/`).
    pub qwen_base: PathBuf,
    /// The InstantX `Qwen-Image-ControlNet-Union` checkpoint — a single `.safetensors` file or a dir
    /// (`diffusion_pytorch_model.safetensors`).
    pub controlnet: PathBuf,
}

/// One Qwen-Image ControlNet (strict-pose) generation request.
#[derive(Clone)]
pub struct QwenControlRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// True-CFG guidance scale.
    pub guidance: f32,
    /// ControlNet conditioning scale on the pose residuals.
    pub control_scale: f32,
    pub seed: u64,
    pub cancel: CancelFlag,
}

impl Default for QwenControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 4.0,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Resolve the ControlNet weight file from a dir-or-file path.
fn resolve_controlnet_file(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    for name in [
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.safetensors",
    ] {
        let p = path.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "qwen control: ControlNet weights not found under {} (expected a \
         diffusion_pytorch_model.safetensors or a direct .safetensors file)",
        path.display()
    )))
}

/// mmap a [`VarBuilder`] over every `.safetensors` in `root/sub` at `dtype`.
fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "qwen control: snapshot is missing the {sub}/ dir (at {})",
            root.display()
        )));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CandleError::Msg(format!("qwen control: read {sub}/: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "qwen control: no .safetensors in {sub}/ (at {})",
            dir.display()
        )));
    }
    // SAFETY: mmap of read-only weight files; standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? })
}

/// The loaded Qwen-Image ControlNet model: the reused base text encoder / DiT / VAE-decoder, plus the
/// VAE encoder (to encode the pose skeleton) and the InstantX control branch.
pub struct QwenControl {
    te_cfg: TextEncoderConfig,
    root: PathBuf,
    device: Device,
    te: QwenTextEncoder,
    transformer: QwenTransformer,
    controlnet: QwenControlNet,
    vae: QwenVae,
    vae_encoder: QwenVaeEncoder,
}

impl QwenControl {
    /// Load the base Qwen-Image components + the VAE encoder + the InstantX control branch.
    pub fn load(paths: &QwenControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.qwen_base.clone();
        let te_cfg = TextEncoderConfig::qwen_image();
        let dit_cfg = TransformerConfig::qwen_image();

        let te = QwenTextEncoder::new(
            &te_cfg,
            component_vb(&root, "text_encoder", ENC_DTYPE, &device)?,
        )?;
        let transformer = QwenTransformer::new(
            &dit_cfg,
            component_vb(&root, "transformer", DIT_DTYPE, &device)?,
        )?;
        let vae = QwenVae::new(component_vb(&root, "vae", ENC_DTYPE, &device)?)?;
        let vae_encoder = QwenVaeEncoder::new(component_vb(&root, "vae", ENC_DTYPE, &device)?)?;

        let cn_file = resolve_controlnet_file(&paths.controlnet)?;
        // SAFETY: mmap of a read-only weight file.
        let cn_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[cn_file], DIT_DTYPE, &device)? };
        let controlnet = QwenControlNet::new(&dit_cfg, CONTROL_LAYERS, cn_vb)?;

        Ok(Self {
            te_cfg,
            root,
            device,
            te,
            transformer,
            controlnet,
            vae,
            vae_encoder,
        })
    }

    /// Tokenize + encode `prompt` → `prompt_embeds` `[1, seq, 3584]` at the DiT dtype (bf16). Mirrors
    /// the txt2img `Pipeline::encode`.
    fn encode(&self, prompt: &str) -> Result<Tensor> {
        let tok = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.te_cfg.max_length,
                pad_token_id: self.te_cfg.pad_token_id,
                chat_template: ChatTemplate::QwenImage,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("qwen control: load tokenizer: {e}")))?;
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("qwen control: tokenize: {e}")))?;
        let len = out.ids.len();
        let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        Ok(self.te.prompt_embeds(&input_ids)?.to_dtype(DIT_DTYPE)?)
    }

    /// Strict-pose generation: condition the base MMDiT on `skeleton` (a rendered OpenPose image at the
    /// request size) via the InstantX control branch. The worker renders the skeleton; this VAE-encodes
    /// + packs it once, then runs the control denoise (the control branch runs on both CFG passes).
    pub fn generate(
        &self,
        req: &QwenControlRequest,
        skeleton: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);
        let total = req.steps as u32;

        let pos = self.encode(&req.prompt)?;
        let neg = if req.guidance > 1.0 {
            let n = if req.negative.trim().is_empty() {
                NEGATIVE_FALLBACK
            } else {
                req.negative.as_str()
            };
            Some(self.encode(n)?)
        } else {
            None
        };

        // VAE-encode + pack the pose skeleton → the control latent `[1, seq, 64]` (constant across steps).
        let control_img = preprocess_control_image(skeleton, req.width, req.height, &self.device)?;
        let control_latent = self.vae_encoder.encode(&control_img)?;
        let control_cond =
            pipeline::pack_latents(&control_latent, req.width, req.height)?.to_dtype(DIT_DTYPE)?;

        let sigmas = pipeline::qwen_sigmas(req.steps, req.width, req.height);
        let mut latents = pipeline::create_noise(req.seed, req.width, req.height, &self.device)?
            .to_dtype(DIT_DTYPE)?;

        for i in 0..req.steps {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let sigma = sigmas[i];
            let pos_res =
                self.controlnet
                    .forward(&latents, &control_cond, &pos, sigma, lat_h, lat_w)?;
            let pos_v = self.transformer.forward_control(
                &latents,
                &pos,
                sigma,
                lat_h,
                lat_w,
                Some(&pos_res),
                req.control_scale,
            )?;
            let v = match &neg {
                Some(neg) => {
                    let neg_res = self.controlnet.forward(
                        &latents,
                        &control_cond,
                        neg,
                        sigma,
                        lat_h,
                        lat_w,
                    )?;
                    let neg_v = self.transformer.forward_control(
                        &latents,
                        neg,
                        sigma,
                        lat_h,
                        lat_w,
                        Some(&neg_res),
                        req.control_scale,
                    )?;
                    pipeline::compute_guided_noise(&pos_v, &neg_v, req.guidance)?
                }
                None => pos_v,
            };
            latents = pipeline::euler_step(&latents, &v, &sigmas, i)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }

        on_progress(Progress::Decoding);
        let lat = pipeline::unpack_latents(&latents, req.width, req.height)?;
        let decoded = self.vae.decode(&lat)?;
        to_image(&decoded)
    }
}

/// A pre-rendered RGB8 control image (the OpenPose skeleton, at the request size) → `[1, 3, H, W]` f32
/// in `[-1, 1]` (the VAE encoder's input range). Requires `image` already at `width × height` (the
/// worker renders the skeleton at the target size — no silent stretch).
fn preprocess_control_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    if image.width != width || image.height != height {
        return Err(CandleError::Msg(format!(
            "qwen control: control image {}x{} must match the request {width}x{height}",
            image.width, image.height
        )));
    }
    let (w, h) = (width as usize, height as usize);
    if image.pixels.len() != w * h * 3 {
        return Err(CandleError::Msg(format!(
            "qwen control: control image buffer {} != {w}x{h}x3",
            image.pixels.len()
        )));
    }
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = image.pixels[(y * w + x) * 3 + c] as f32 / 127.5 - 1.0;
                data[c * h * w + y * w + x] = v;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, h, w), device)?)
}

/// VAE output `[1, 3, H, W]` in `[-1, 1]` → an RGB8 [`Image`].
fn to_image(decoded: &Tensor) -> Result<Image> {
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = QwenControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    #[test]
    fn controlnet_file_resolution() {
        let dir = std::env::temp_dir().join(format!("qwen_cn_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(resolve_controlnet_file(&dir).is_err());
        let f = dir.join("diffusion_pytorch_model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_controlnet_file(&dir).unwrap(), f);
        assert_eq!(resolve_controlnet_file(&f).unwrap(), f);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn control_preprocess_shape_and_range() {
        let img = Image {
            width: 16,
            height: 8,
            pixels: vec![255u8; 16 * 8 * 3],
        };
        let t = preprocess_control_image(&img, 16, 8, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 8, 16]);
        // 255 → 255/127.5 - 1 = 1.0
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-4));
        // size mismatch errors loudly.
        assert!(preprocess_control_image(&img, 32, 8, &Device::Cpu).is_err());
    }
}
