//! Kolors **ControlNet (strict pose)** provider (sc-5489, epic 5480) — reference-pose conditioning on
//! Kolors, the candle (Windows/CUDA) sibling of the `mlx-gen-kolors` ControlNet path. It reuses the
//! same vendored SDXL stack the Kolors IP-Adapter slice ([`crate::ip_provider`]) stands on, swapping the
//! IP-Adapter overlay for a diffusers **`ControlNetModel`**:
//!
//! - a rendered OpenPose skeleton (the worker draws it at the request size) is normalized to `[0,1]`
//!   and embedded ONCE by the ControlNet's `controlnet_cond_embedding` conv stack
//!   ([`ControlNet::embed_cond`], step-invariant);
//! - each denoise step the Kolors ControlNet — an SDXL-family encoder copy ([`ControlNet`], built from
//!   [`ControlNetConfig::kolors`]) — emits the per-down-block + mid [`ControlResiduals`] (scaled by
//!   `control_scale`), which the vendored SDXL [`UNet2DConditionModel::forward_instantid`] adds into its
//!   skip connections + mid output (the same residual seam InstantID rides on);
//! - the text path is **ChatGLM3-6B** (the Kolors encoder); the denoise runs the Kolors **leading-Euler**
//!   sampler ([`KolorsEulerSampler`]) — NOT the SDXL EulerAncestral — so the numerics match Kolors txt2img.
//!
//! The Kolors `ControlNetModel` carries its **own** `encoder_hid_proj` (4096→2048), trained separately
//! from the UNet's — so the raw ChatGLM3 context is projected **twice**: once by the UNet's
//! `encoder_hid_proj` (for the base cross-attentions) and once by the ControlNet's (for the control
//! branch's cross-attentions). No IP-Adapter K/V is installed, so [`UNet2DConditionModel::forward_instantid`]
//! runs as a plain SDXL UNet + control residuals (the decoupled-attn branch is `None`-guarded).
//!
//! Like the SDXL/InstantID/Kolors-IP lanes, **CFG is uncond-first** (`[negative, prompt]`, the Kolors
//! txt2img convention), and the control branch runs on **both** CFG passes (the diffusers
//! `guess_mode=False` rule). The provider is a plain struct driven **directly** by the worker (a bespoke
//! pose stream, like [`crate::ip_provider::IpAdapterKolors`]), not a gen-core-registered generator — the
//! registered `kolors` descriptor stays txt2img-only.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{self as nn, Linear, Module, VarBuilder};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use candle_gen_sdxl::{
    preprocess_control_image, sdxl_unet_config, ControlNet, ControlNetConfig, UNet2DConditionModel,
};

use crate::chatglm3::ChatGlmModel;
use crate::config::ChatGlmConfig;
use crate::pipeline::{sdxl_vae_config, VAE_SCALE};
use crate::sampler::KolorsEulerSampler;
use crate::tokenizer::KolorsTokenizer;

/// The control compute dtype. Kolors runs the whole stack at **f32** (the candle port recipe — a single
/// matmul dtype), so the vendored UNet, the ControlNet, and the SDXL VAE all load at f32 here too.
const DTYPE: DType = DType::F32;

/// Kolors `add_embedding` dims (the Kolors `unet/config.json` AND the `Kolors-ControlNet-*/config.json`):
/// `addition_time_embed_dim = 256`, `projection_class_embeddings_input_dim = 5632` (pooled 4096 + 6·256)
/// — vs SDXL's 2816. The vendored UNet needs the `add_embedding` head the plain `forward` omits; the
/// ControlNet builds its matching head from [`ControlNetConfig::kolors`].
const ADDITION_TIME_EMBED_DIM: usize = 256;
const PROJECTION_INPUT_DIM: usize = 5632;

/// ChatGLM3 context width (the `encoder_hid_proj` in-features, on BOTH the UNet and the ControlNet).
const CONTEXT_DIM: usize = 4096;
/// The SDXL/Kolors UNet cross-attention width (the `encoder_hid_proj` out-features).
const CROSS_ATTENTION_DIM: usize = 2048;

/// Default ControlNet conditioning scale (the strict-pose tier — parity with the Qwen control slice and
/// the mlx Kolors ControlNet path).
pub const DEFAULT_CONTROL_SCALE: f32 = 1.0;

/// Paths to the Kolors ControlNet checkpoints.
pub struct KolorsControlPaths {
    /// The `Kwai-Kolors/Kolors-diffusers` snapshot dir (`tokenizer/`, `text_encoder/` ChatGLM3-6B,
    /// `unet/` SDXL-family UNet, `vae/` SDXL VAE).
    pub kolors_base: PathBuf,
    /// The `Kwai-Kolors/Kolors-ControlNet-Pose` checkpoint — a single `.safetensors` file or a dir
    /// (`diffusion_pytorch_model.safetensors`).
    pub controlnet: PathBuf,
}

/// One Kolors ControlNet (strict-pose) generation request.
#[derive(Clone)]
pub struct KolorsControlRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Classifier-free guidance scale.
    pub guidance: f32,
    /// ControlNet conditioning scale on the pose residuals.
    pub control_scale: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for KolorsControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 50,
            guidance: 5.0,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// mmap an f32 [`VarBuilder`] over every `.safetensors` in `dir` (the ChatGLM3 encoder + UNet ship
/// sharded or single-file) — mirrors the txt2img pipeline / IP-Adapter loaders.
fn f32_vb(dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("kolors-control: read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "kolors-control: no .safetensors found in {} (expected a Kolors-diffusers snapshot)",
            dir.display()
        )));
    }
    // SAFETY: mmap of read-only weight files; the standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, DTYPE, device)? })
}

/// Resolve the ControlNet weight **file** from a dir-or-file path (the diffusers `ControlNetModel`
/// layout: a single `diffusion_pytorch_model(.fp16).safetensors`).
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
        "kolors-control: ControlNet weights not found under {} (expected a \
         diffusion_pytorch_model.safetensors or a direct .safetensors file)",
        path.display()
    )))
}

/// Loaded Kolors ControlNet model: the ChatGLM3 tokenizer + encoder, the UNet's `encoder_hid_proj`
/// context projection, the vendored SDXL UNet (NO IP installed — plain SDXL + control residuals), the
/// Kolors ControlNet + its OWN `encoder_hid_proj`, and the f32 SDXL VAE.
pub struct KolorsControl {
    tokenizer: KolorsTokenizer,
    chatglm: ChatGlmModel,
    /// The UNet's ChatGLM3 context projection (4096 → 2048), applied before the base cross-attentions
    /// (the vendored UNet has no `encoder_hid_proj`, unlike [`crate::unet::KolorsUNet`]).
    encoder_hid_proj: Linear,
    unet: UNet2DConditionModel,
    /// The ControlNet's OWN ChatGLM3 context projection (4096 → 2048), trained separately from the
    /// UNet's, applied before the control branch's cross-attentions.
    cn_encoder_hid_proj: Linear,
    controlnet: ControlNet,
    vae: AutoEncoderKL,
    device: Device,
}

impl KolorsControl {
    /// Load the Kolors backbone (ChatGLM3 + SDXL-family UNet into the vendored stack + SDXL VAE) + the
    /// Kolors `ControlNetModel` (encoder copy + its own `encoder_hid_proj`). No IP-Adapter K/V is
    /// installed — the control branch is the only conditioning overlay.
    pub fn load(paths: &KolorsControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let base = paths.kolors_base.as_path();

        let tokenizer = KolorsTokenizer::from_dir(base.join("tokenizer"))?;
        let chatglm = ChatGlmModel::new(
            ChatGlmConfig::chatglm3_6b(),
            f32_vb(&base.join("text_encoder"), &device)?,
        )?;

        // Vendored SDXL UNet from the Kolors `unet/` weights + the 5632 `add_embedding` head + the UNet's
        // `encoder_hid_proj` (all in the same checkpoint). NOTE: no `install_ip_adapter` — `forward_instantid`
        // then runs as a plain SDXL UNet (its decoupled-attn branch is `None`-guarded) + control residuals.
        let vs = f32_vb(&base.join("unet"), &device)?;
        let unet = UNet2DConditionModel::new(vs.clone(), 4, 4, false, sdxl_unet_config())?
            .with_add_embedding(vs.clone(), ADDITION_TIME_EMBED_DIM, PROJECTION_INPUT_DIM)?;
        let encoder_hid_proj =
            nn::linear(CONTEXT_DIM, CROSS_ATTENTION_DIM, vs.pp("encoder_hid_proj"))?;

        // Kolors ControlNet (a diffusers SDXL-family `ControlNetModel`) + its OWN `encoder_hid_proj`.
        let cn_file = resolve_controlnet_file(&paths.controlnet)?;
        // SAFETY: mmap of a read-only weight file.
        let cn_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[cn_file], DTYPE, &device)? };
        let cn_encoder_hid_proj = nn::linear(
            CONTEXT_DIM,
            CROSS_ATTENTION_DIM,
            cn_vb.pp("encoder_hid_proj"),
        )?;
        let controlnet = ControlNet::new(cn_vb, &ControlNetConfig::kolors())?;

        let vae = AutoEncoderKL::new(f32_vb(&base.join("vae"), &device)?, 3, 3, sdxl_vae_config())?;

        Ok(Self {
            tokenizer,
            chatglm,
            encoder_hid_proj,
            unet,
            cn_encoder_hid_proj,
            controlnet,
            vae,
            device,
        })
    }

    /// Strict-pose T2I: condition the Kolors generation on `skeleton` (a rendered OpenPose image at the
    /// request size) via the Kolors ControlNet, denoising with the Kolors leading-Euler sampler. The
    /// worker renders the skeleton; this embeds it once, then runs the control denoise (the control
    /// branch runs on both CFG passes).
    pub fn generate(
        &self,
        req: &KolorsControlRequest,
        skeleton: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let use_guide = req.guidance > 1.0;

        // CFG batch is [neg, pos] = uncond-first (the Kolors txt2img convention); without guidance only
        // the positive branch is built.
        let (pos_ctx, pos_pooled) = self.encode(&req.prompt)?;
        let (context, pooled) = if use_guide {
            let (neg_ctx, neg_pooled) = self.encode(&req.negative)?;
            (
                Tensor::cat(&[&neg_ctx, &pos_ctx], 0)?,
                Tensor::cat(&[&neg_pooled, &pos_pooled], 0)?,
            )
        } else {
            (pos_ctx, pos_pooled)
        };
        let batch = if use_guide { 2 } else { 1 };

        // Two SEPARATE ChatGLM3 → cross-attention projections: the UNet's `encoder_hid_proj` feeds the
        // base cross-attentions; the ControlNet's own (separately-trained) `encoder_hid_proj` feeds the
        // control branch's. Both project the raw 4096-wide context to 2048 up front.
        let projected = self.encoder_hid_proj.forward(&context)?;
        let cn_context = self.cn_encoder_hid_proj.forward(&context)?;
        let time_ids = self.build_time_ids(batch, req.height, req.width)?;

        // The pose skeleton → `[batch, 3, H, W]` in `[0,1]` (the diffusers control-image normalization,
        // NOT a VAE's `[-1,1]`), CFG-batched (same control on both rows). `embed_cond` is step-invariant,
        // so the conditioning embedding is computed ONCE here.
        let control = preprocess_control_image(skeleton, req.width, req.height, &self.device)?
            .to_dtype(DTYPE)?;
        let control = if use_guide {
            Tensor::cat(&[&control, &control], 0)?
        } else {
            control
        };
        let cond_embed = self.controlnet.embed_cond(&control)?;

        let sampler = KolorsEulerSampler::new(req.steps).map_err(CandleError::Msg)?;
        let (lat_h, lat_w) = ((req.height / 8) as usize, (req.width / 8) as usize);
        let noise = self.initial_noise(req.seed, lat_h, lat_w)?;
        let mut latents = (noise * sampler.init_noise_sigma() as f64)?;

        let control_scale = req.control_scale as f64;
        let total = sampler.num_steps() as u32;
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
            let t = sampler.timestep(i) as f64;
            // Control residuals from the Kolors ControlNet (its own context projection), scaled by
            // `control_scale`, then added into the UNet skip + mid via `forward_instantid`.
            let res = self.controlnet.forward(
                &model_in,
                &cond_embed,
                t,
                &cn_context,
                &pooled,
                &time_ids,
                control_scale,
            )?;
            let eps = self.unet.forward_instantid(
                &model_in,
                t,
                &projected,
                &pooled,
                &time_ids,
                Some(res.down.as_slice()),
                Some(&res.mid),
            )?;
            let eps = if use_guide {
                let ch = eps.chunk(2, 0)?;
                let (uncond, cond) = (&ch[0], &ch[1]);
                (uncond + ((cond - uncond)? * req.guidance as f64)?)?
            } else {
                eps
            };
            latents = (&latents + (eps * sampler.step_dt(i) as f64)?)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }

        on_progress(Progress::Decoding);
        self.decode(&latents)
    }

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])` via the ChatGLM3 encoder.
    fn encode(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let tokens = self.tokenizer.encode(prompt)?;
        Ok(self.chatglm.encode_prompt(&tokens)?)
    }

    /// The SDXL micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)` per row, f32 `[batch, 6]` (the
    /// Kolors txt2img value — original == target, no crop).
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
    /// fixed-algorithm CPU RNG seeded by `seed`, moved to the device (matches the txt2img pipeline).
    fn initial_noise(&self, seed: u64, lat_h: usize, lat_w: usize) -> Result<Tensor> {
        let n = 4 * lat_h * lat_w;
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        Ok(Tensor::from_vec(noise, (1, 4, lat_h, lat_w), &Device::Cpu)?.to_device(&self.device)?)
    }

    /// VAE-decode latents `[1, 4, H/8, W/8]` → an RGB8 [`Image`] (un-scale by [`VAE_SCALE`], `x/2 + 0.5`,
    /// clamp, ×255) — the txt2img pipeline's decode.
    fn decode(&self, latents: &Tensor) -> Result<Image> {
        let unscaled = (latents / VAE_SCALE)?;
        let img = self.vae.decode(&unscaled)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the Kolors production knobs (1024², 50 steps, CFG 5.0, control 1.0).
    #[test]
    fn request_defaults() {
        let r = KolorsControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 50);
        assert_eq!(r.guidance, 5.0);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert_eq!(DEFAULT_CONTROL_SCALE, 1.0);
        assert!(!r.cancel.is_cancelled());
    }

    /// The Kolors `add_embedding` projection input is 5632 (pooled 4096 + 6·256) — vs SDXL's 2816 —
    /// shared by the vendored UNet AND the ControlNet's matching head (`ControlNetConfig::kolors`).
    #[test]
    fn kolors_add_embedding_dims() {
        assert_eq!(ADDITION_TIME_EMBED_DIM, 256);
        assert_eq!(PROJECTION_INPUT_DIM, 4096 + 6 * 256);
        assert_eq!(CONTEXT_DIM, 4096);
        assert_eq!(CROSS_ATTENTION_DIM, 2048);
        assert_eq!(
            ControlNetConfig::kolors().projection_class_embeddings_input_dim,
            PROJECTION_INPUT_DIM
        );
    }

    /// `resolve_controlnet_file`: a directory resolves `diffusion_pytorch_model.safetensors`; a direct
    /// file is used as-is; a missing dir errors loudly.
    #[test]
    fn controlnet_file_resolution() {
        let dir = std::env::temp_dir().join(format!("candle_kolors_cn_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(resolve_controlnet_file(&dir).is_err());
        let f = dir.join("diffusion_pytorch_model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_controlnet_file(&dir).unwrap(), f);
        assert_eq!(resolve_controlnet_file(&f).unwrap(), f);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
