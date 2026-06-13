//! # candle-gen-ltx
//!
//! The **LTX-2.3 (distilled 22B)** text-to-video provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-ltx`. LTX has **no** `candle-transformers` reference: the
//! `AVTransformer3DModel` video DiT ([`transformer`]), the `CausalVideoAutoencoder` temporal VAE
//! ([`vae`], on a from-scratch [`conv3d`]), the **Gemma-3-12B** text encoder ([`gemma`]) +
//! per-token-RMS aggregation + 8-layer learnable-register connector ([`text_encoder`], [`connector`]),
//! and the rectified-flow distilled scheduler ([`scheduler`]) are all ported here.
//!
//! **txt2video (sc-3698), first slice:** [`LtxGenerator::generate`] runs Gemma-3-12B → text projection
//! → connector → the 48-layer video DiT (split 3-D RoPE, per-head gated attention, adaLN-single) → the
//! temporal VAE decoder, emitting `GenerationOutput::Video`. Registered under `"ltx_2_3_distilled"`.
//! Single-stage distilled denoise (no CFG). **Deferred** to follow-up stories: the audio stack
//! (audio-VAE + vocoder + AV-joint DiT paths), the 2-stage latent upsampler, I2V conditioning,
//! prompt-enhance, LoRA/IC-LoRA, and fp8/quant.
//!
//! **Dtypes:** the DiT, connector, text projection, and Gemma encoder run **bf16** (the checkpoint's
//! native dtype; 22B+12B does not fit f32 on a single 96 GB GPU); the VAE runs **f32**; attention and
//! norms upcast to f32. `backend = "candle"`, `mac_only = false`.
//!
//! **Weights:** `spec.weights` points at an LTX-2.3 snapshot dir (the
//! `ltx-2.3-22b-distilled.safetensors` single-file checkpoint bundling DiT + VAE + projection +
//! connector). The Gemma-3-12B encoder + its `tokenizer.json` live in a separate snapshot, located via
//! the `LTX_GEMMA_DIR` env var (falling back to `<root>/text_encoder`).

pub mod config;
pub mod connector;
pub mod conv3d;
pub mod gemma;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use config::{
    ConnectorConfig, GemmaConfig, TransformerConfig, DEFAULT_FPS, DEFAULT_FRAMES, DEFAULT_HEIGHT,
    DEFAULT_WIDTH, MODEL_ID, STAGE1_SIGMAS, TEXT_MAX_LENGTH,
};
use gemma::GemmaEncoder;
use text_encoder::LtxTextEncoder;
use transformer::LtxDiT;
use vae::LtxVideoVae;

const DIT_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;
const SIZE_MULTIPLE: u32 = config::SPATIAL_SCALE as u32;

#[derive(Clone)]
struct Components {
    te: Arc<LtxTextEncoder>,
    dit: Arc<LtxDiT>,
    vae: Arc<LtxVideoVae>,
    tokenizer: Arc<tokenizers::Tokenizer>,
}

struct Pipeline {
    dit_cfg: TransformerConfig,
    gemma_cfg: GemmaConfig,
    conn_cfg: ConnectorConfig,
    root: PathBuf,
    device: Device,
}

impl Pipeline {
    fn load(root: &Path, device: &Device) -> Self {
        Self {
            dit_cfg: TransformerConfig::ltx_2_3(),
            gemma_cfg: GemmaConfig::gemma_3_12b(),
            conn_cfg: ConnectorConfig::ltx_2_3(),
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    /// The single LTX-2.3 checkpoint file in `root` (the distilled 22B model, not a LoRA/upscaler).
    fn ltx_checkpoint(&self) -> CResult<PathBuf> {
        let mut cands: Vec<PathBuf> = std::fs::read_dir(&self.root)
            .map_err(|e| CandleError::Msg(format!("ltx: read snapshot dir: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.ends_with(".safetensors")
                    && name.contains("distilled")
                    && !name.contains("lora")
                    && !name.contains("upscaler")
            })
            .collect();
        cands.sort();
        cands.into_iter().next().ok_or_else(|| {
            CandleError::Msg(format!(
                "ltx: no `ltx-2.3-*-distilled.safetensors` in {} (expected an LTX-2.3 snapshot)",
                self.root.display()
            ))
        })
    }

    /// The Gemma-3-12B encoder snapshot dir (`LTX_GEMMA_DIR`, or `<root>/text_encoder`).
    fn gemma_dir(&self) -> CResult<PathBuf> {
        if let Ok(p) = std::env::var("LTX_GEMMA_DIR") {
            return Ok(PathBuf::from(p));
        }
        let fallback = self.root.join("text_encoder");
        if fallback.is_dir() {
            return Ok(fallback);
        }
        Err(CandleError::Msg(
            "ltx: set LTX_GEMMA_DIR to a google/gemma-3-12b-it snapshot (or place it at \
             <root>/text_encoder)"
                .into(),
        ))
    }

    fn safetensors_in(dir: &Path) -> CResult<Vec<PathBuf>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| CandleError::Msg(format!("ltx: read {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "ltx: no .safetensors in {}",
                dir.display()
            )));
        }
        Ok(files)
    }

    fn load_components(&self) -> CResult<Components> {
        let ltx_file = self.ltx_checkpoint()?;
        let gemma_dir = self.gemma_dir()?;
        let gemma_files = Self::safetensors_in(&gemma_dir)?;

        // Two builders over the single LTX file: bf16 (DiT + projection + connector), f32 (VAE).
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let ltx_files = [ltx_file];
        let vb_bf16 =
            unsafe { VarBuilder::from_mmaped_safetensors(&ltx_files, DIT_DTYPE, &self.device)? };
        let vb_f32 =
            unsafe { VarBuilder::from_mmaped_safetensors(&ltx_files, VAE_DTYPE, &self.device)? };
        let gemma_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&gemma_files, DIT_DTYPE, &self.device)? }
                .pp("language_model.model");

        let dit_vb = vb_bf16.pp("model.diffusion_model");
        let dit = LtxDiT::new(dit_vb.clone(), &self.dit_cfg)?;
        let te = LtxTextEncoder::new(
            gemma_vb,
            vb_bf16.clone(),
            dit_vb,
            &self.gemma_cfg,
            &self.conn_cfg,
        )?;
        let vae = LtxVideoVae::new(vb_f32.pp("vae"), config::LATENT_CHANNELS, 4)?;

        let tok_path = gemma_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| CandleError::Msg(format!("ltx: load gemma tokenizer: {e}")))?;

        Ok(Components {
            te: Arc::new(te),
            dit: Arc::new(dit),
            vae: Arc::new(vae),
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Tokenize `prompt` with the Gemma tokenizer (BOS, right-truncate then **left-pad** to
    /// `TEXT_MAX_LENGTH`), returning `(input_ids [1, 256] u32, mask01 [256])`.
    fn tokenize(&self, tok: &tokenizers::Tokenizer, prompt: &str) -> CResult<(Tensor, Vec<u32>)> {
        let enc = tok
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("ltx: tokenize: {e}")))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let max = TEXT_MAX_LENGTH;
        if ids.len() > max {
            ids.truncate(max);
        }
        let nv = ids.len();
        let pad = max - nv;
        let mut padded = vec![0u32; pad];
        padded.extend_from_slice(&ids);
        let mut mask = vec![0u32; pad];
        mask.extend(std::iter::repeat_n(1u32, nv));
        let input_ids = Tensor::from_vec(padded, (1, max), &self.device)?;
        Ok((input_ids, mask))
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES);
        let fps = req.fps.unwrap_or(DEFAULT_FPS);
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);

        // Text encode → context (1, 256, 4096).
        let (input_ids, mask01) = self.tokenize(&comps.tokenizer, &req.prompt)?;
        let context = comps.te.encode(&input_ids, &mask01)?;

        // Latent geometry + split-RoPE for the token grid.
        let (t_lat, h_lat, w_lat) = pipeline::latent_dims(frames, req.width, req.height);
        let (cos, sin) =
            rope::video_rope(&self.dit_cfg, t_lat, h_lat, w_lat, fps as f32, &self.device)?;

        let mut latents = pipeline::create_noise(seed, t_lat, h_lat, w_lat, &self.device)?;
        let steps = STAGE1_SIGMAS.len() - 1;
        for i in 0..steps {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let (sigma, sigma_next) = (STAGE1_SIGMAS[i] as f64, STAGE1_SIGMAS[i + 1] as f64);
            let flat = pipeline::flatten_latent(&latents)?;
            let vel = comps.dit.forward(&flat, sigma, &context, &cos, &sin)?;
            let vel = pipeline::unflatten_latent(&vel.to_dtype(DType::F32)?, t_lat, h_lat, w_lat)?;
            let denoised = scheduler::to_denoised(&latents, &vel, sigma)?;
            latents = scheduler::euler_step(&latents, &denoised, sigma, sigma_next)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total: steps as u32,
            });
        }

        on_progress(Progress::Decoding);
        let decoded = comps.vae.decode(&latents)?;
        let images = pipeline::frames_to_images(&decoded)?;
        Ok((images, fps))
    }
}

pub struct LtxGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

impl LtxGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("ltx components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = pipe.load_components()?;
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for LtxGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg("ltx: prompt must not be empty".into()));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "ltx: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % config::TEMPORAL_SCALE as u32 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "ltx: frames must satisfy frames % {} == 1 (got {f})",
                    config::TEMPORAL_SCALE
                )));
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = Pipeline::load(&self.root, &self.device);
        let components = self.components(&pipe)?;
        let (frames, fps) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }
}

/// LTX-2.3 distilled txt2video descriptor — single-stage rectified-flow (no CFG / negative prompt;
/// guidance is distilled in). Audio / I2V / upsampler / LoRA / quant deferred.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ltx",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["rectified-flow"],
            schedulers: vec![],
            min_size: SIZE_MULTIPLE,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Construct a lazy candle LTX-2.3 generator. `spec.weights` is an LTX-2.3 snapshot dir (the
/// `ltx-2.3-22b-distilled.safetensors` checkpoint); the Gemma encoder is located via `LTX_GEMMA_DIR`.
/// Adapters / quantization / conditioning are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => p
            .parent()
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| p.clone()),
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle ltx does not support LoRA/LoKr yet".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle ltx does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle ltx does not support image / I2V conditioning yet (txt2video only)".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(LtxGenerator {
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

#[allow(dead_code)]
fn _defaults_referenced() {
    let _ = (DEFAULT_WIDTH, DEFAULT_HEIGHT, GemmaEncoder::forward);
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("ltx is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "ltx");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(d.capabilities.samplers.contains(&"rectified-flow"));
    }

    #[test]
    fn validate_accepts_txt2video_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 704,
            height: 480,
            frames: Some(49),
            sampler: Some("rectified-flow".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                frames: Some(48), // not ≡ 1 (mod 8)
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 700, // not a multiple of 32
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }
}
