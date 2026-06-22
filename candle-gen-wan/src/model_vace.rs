//! The **Wan-VACE** controllable-video provider (`wan_vace`, Wan2.1-VACE-14B) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-wan`'s `wan_vace`. Registers as `backend = "candle"`,
//! [`Modality::Video`].
//!
//! VACE is **mode-agnostic at the engine boundary**, exactly like diffusers `WanVACEPipeline`: the
//! SceneWorks worker builds the per-mode control video + mask (replace_person = the person-region-
//! neutralized clip + the person mask; extend/bridge = the source frames at the kept positions + a
//! generated-span mask) and passes them as one [`Conditioning::ControlClip`]. This provider
//! VAE-encodes the inactive/reactive split + unfolds the mask into the 96-channel control latent
//! ([`crate::vace::prepare_video_latents`] / [`prepare_masks`](crate::vace::prepare_masks)) and runs
//! the CFG VACE denoise loop ([`denoise_vace`](crate::vace::denoise_vace)). Reference images (from
//! [`Conditioning::Reference`]) become leading latent frames and are dropped after denoise (diffusers
//! `latents[:, :, num_reference_images:]`).
//!
//! **Snapshot layout** (diffusers): `transformer/` (the VACE DiT, diffusers tensor names), `text_encoder/`
//! (UMT5-XXL), `vae/` (the z16 Wan VAE — needs the encoder for the control encode), `tokenizer/`.
//!
//! **Dtypes:** UMT5 + VAE run **f32**; the VACE DiT runs **bf16** (norms/modulation upcast to f32) —
//! the candle Wan regime. LoRA / on-the-fly quantization are **deferred** (rejected at load).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use crate::config::{
    TextEncoderConfig, Vae16Config, WanVaceConfig, DEFAULT_FPS_VACE, DEFAULT_GUIDANCE_VACE,
    DEFAULT_STEPS_VACE, MODEL_ID_VACE, NEGATIVE_FALLBACK, SIZE_MULTIPLE_14B, VACE_FLOW_SHIFT,
    VAE16_STRIDE_TEMPORAL,
};
use crate::pipeline::{create_noise, frames_to_images};
use crate::rope::WanRope;
use crate::scheduler::Sampler;
use crate::text_encoder::Umt5Encoder;
use crate::vace::{
    build_vace_control, denoise_vace, prepare_masks, prepare_video_latents, WanVaceTransformer,
};
use crate::vae16::WanVae16;
use crate::wan14b::preprocess_i2v_image;

const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
const VAE_DTYPE: DType = DType::F32;
const Z_DIM: usize = 16;
const VAE_T: usize = VAE16_STRIDE_TEMPORAL as usize;

#[derive(Clone)]
struct Components {
    te: Arc<Umt5Encoder>,
    dit: Arc<WanVaceTransformer>,
    vae: Arc<WanVae16>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    vace_cfg: WanVaceConfig,
    vae_cfg: Vae16Config,
    root: PathBuf,
    device: Device,
}

impl Pipeline {
    fn load(root: &Path, device: &Device) -> Self {
        Self {
            te_cfg: TextEncoderConfig::umt5_xxl(),
            vace_cfg: WanVaceConfig::vace_14b(),
            vae_cfg: Vae16Config::wan21(),
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "wan-vace snapshot is missing the {sub}/ dir (expected a Wan2.1-VACE-14B diffusers \
                 snapshot at {})",
                self.root.display()
            )));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("wan-vace: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "wan-vace: no .safetensors in {sub}/ (at {})",
                dir.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &self.device)? };
        Ok(vb)
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Umt5Encoder::new(&self.te_cfg, self.component_vb("text_encoder", ENC_DTYPE)?)?;
        let dit =
            WanVaceTransformer::new(&self.vace_cfg, self.component_vb("transformer", DIT_DTYPE)?)?;
        // The control encode needs the VAE encoder.
        let vae = WanVae16::new_with_encoder(&self.vae_cfg, self.component_vb("vae", VAE_DTYPE)?)?;
        Ok(Components {
            te: Arc::new(te),
            dit: Arc::new(dit),
            vae: Arc::new(vae),
        })
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, 512, 4096]` (f32), zero-padded to `max_length` (the DiT
    /// cross-attends over the 512-padded context — the same rule as the base Wan).
    fn encode(&self, te: &Umt5Encoder, prompt: &str) -> CResult<Tensor> {
        let tok = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.te_cfg.max_length,
                pad_token_id: self.te_cfg.pad_token_id,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("wan-vace: load tokenizer: {e}")))?;
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("wan-vace: tokenize: {e}")))?;
        let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        if ids.is_empty() {
            // Empty prompt → zero ids → a degenerate `(1,1)` tensor (the old `.max(1)` padded the
            // shape, not the data) whose 0-element f32 embedding gather reads out of bounds on CUDA
            // (`CUDA_ERROR_ILLEGAL_ADDRESS`, surfacing as a misleading cublas failure). Emit one pad
            // token so a 0-length sequence never reaches the gather. (sc-7078)
            ids.push(self.te_cfg.pad_token_id as u32);
        }
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let embeds = te.encode(&input_ids)?;
        let max_len = self.te_cfg.max_length;
        let dim = embeds.dim(2)?;
        match len.cmp(&max_len) {
            std::cmp::Ordering::Less => {
                let pad = Tensor::zeros((1, max_len - len, dim), embeds.dtype(), &self.device)?;
                Ok(Tensor::cat(&[&embeds, &pad], 1)?)
            }
            std::cmp::Ordering::Greater => Ok(embeds.narrow(1, 0, max_len)?),
            std::cmp::Ordering::Equal => Ok(embeds),
        }
    }

    /// Stack a list of frame [`Image`]s → a `[1, 3, F, H, W]` clip in `[-1, 1]` (the Wan VAE input
    /// convention), via the per-frame cover-fit resize + center-crop.
    fn preprocess_clip(&self, frames: &[Image], width: u32, height: u32) -> CResult<Tensor> {
        if frames.is_empty() {
            return Err(CandleError::Msg(
                "wan-vace: control clip has no frames".into(),
            ));
        }
        let planes: Vec<Tensor> = frames
            .iter()
            .map(|im| preprocess_i2v_image(im, width, height, &self.device)) // [1,3,1,H,W]
            .collect::<CResult<_>>()?;
        let refs: Vec<&Tensor> = planes.iter().collect();
        Ok(Tensor::cat(&refs, 2)?) // [1,3,F,H,W]
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let clip = req
            .control_clip()
            .ok_or_else(|| CandleError::Msg("wan-vace: requires a ControlClip".into()))?;
        let width = req.width;
        let height = req.height;
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS_VACE as usize);
        let shift = req
            .scheduler_shift
            .map(|s| s as f64)
            .unwrap_or(VACE_FLOW_SHIFT);
        let sampler = Sampler::parse(req.sampler.as_deref());
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE_VACE) as f64;
        let cfg_disabled = guidance <= 1.0;
        let fps = req.fps.unwrap_or(DEFAULT_FPS_VACE);

        // Control video [-1,1] + mask [0,1] (diffusers `clamp((m+1)/2)`), each [1,3,F,H,W].
        let control_video = self.preprocess_clip(clip.frames, width, height)?;
        let mask = self.preprocess_clip(clip.mask, width, height)?;
        let mask = ((mask + 1.0)? * 0.5)?; // (m+1)/2 ∈ [0,1]

        // Reference images (optional) → [1,3,1,H,W] each.
        let references: Vec<Tensor> = req
            .conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            })
            .map(|im| preprocess_i2v_image(im, width, height, &self.device))
            .collect::<CResult<_>>()?;
        let num_ref = references.len();

        // Stage 1: UMT5 text encode + project to the DiT context.
        let pos = self.encode(&comps.te, &req.prompt)?;
        let ctx_pos = comps.dit.embed_text(&pos)?;
        let ctx_neg = if cfg_disabled {
            None
        } else {
            let neg = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
            Some(comps.dit.embed_text(&self.encode(&comps.te, neg)?)?)
        };

        // Stage 2: z16 VAE-encode the control + mask → the 96-ch control latent.
        let patch_h = self.vace_cfg.base.patch.1;
        let video_latents = prepare_video_latents(&comps.vae, &control_video, &mask, &references)?;
        let mask_latents = prepare_masks(&mask, patch_h, num_ref)?;
        let control = build_vace_control(&video_latents, &mask_latents)?;
        let (_, _, t_total, h_lat, w_lat) = control.dims5()?;

        // Per-vace-layer control scale (diffusers `conditioning_scale`); default 1.0.
        let scales = vec![req.control_scale.unwrap_or(1.0); self.vace_cfg.vace_layers.len()];

        // RoPE for the (ref-extended) token grid.
        let (pt, ph, pw) = self.vace_cfg.base.patch;
        let (ppf, pph, ppw) = (t_total / pt, h_lat / ph, w_lat / pw);
        let (cos, sin) = WanRope::new(&self.vace_cfg.base).cos_sin(ppf, pph, ppw, &self.device)?;

        // Seeded init noise [1, 16, T_total, h, w].
        let init_noise = create_noise(seed, Z_DIM, t_total, h_lat, w_lat, &self.device)?;

        // Stage 3: CFG VACE denoise.
        let total = steps as u32;
        let mut on_step = |i: usize| {
            on_progress(Progress::Step {
                current: i as u32,
                total,
            })
        };
        let latents = denoise_vace(
            &comps.dit,
            &control,
            &scales,
            sampler,
            steps,
            shift,
            guidance,
            &ctx_pos,
            ctx_neg.as_ref(),
            &init_noise,
            &cos,
            &sin,
            &req.cancel,
            &mut on_step,
        )?;

        // Drop the leading reference latent frames (diffusers `latents[:, :, num_reference_images:]`).
        let latents = if num_ref > 0 {
            latents.narrow(2, num_ref, t_total - num_ref)?
        } else {
            latents
        };

        // Stage 4: z16 VAE decode → RGB frames.
        on_progress(Progress::Decoding);
        let decoded = comps.vae.decode(&latents)?;
        let images = frames_to_images(&decoded)?;
        Ok((images, fps))
    }
}

/// A loaded Wan-VACE generator. Heavy components (UMT5, the VACE DiT, the z16 VAE) are loaded lazily on
/// the first `generate` and cached.
pub struct WanVaceGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

impl WanVaceGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("wan-vace components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = pipe.load_components()?;
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for WanVaceGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID_VACE, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "wan-vace: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg("wan-vace: steps must be >= 1".into()));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE_14B)
            || !req.height.is_multiple_of(SIZE_MULTIPLE_14B)
        {
            return Err(gen_core::Error::Msg(format!(
                "wan-vace: width/height must be multiples of {SIZE_MULTIPLE_14B} (got {}x{})",
                req.width, req.height
            )));
        }
        let clip = req.control_clip().ok_or_else(|| {
            gen_core::Error::Msg(
                "wan-vace: needs a ControlClip (the masked control video — the worker builds it per \
                 mode: replace_person / extend / bridge)"
                    .into(),
            )
        })?;
        if clip.frames.len() != clip.mask.len() {
            return Err(gen_core::Error::Msg(format!(
                "wan-vace: control frames ({}) and mask frames ({}) length mismatch",
                clip.frames.len(),
                clip.mask.len()
            )));
        }
        // Control clip frame count must be 1 + 4·k (one z16 VAE temporal chunk + groups of 4).
        if clip.frames.len() % VAE_T != 1 {
            return Err(gen_core::Error::Msg(format!(
                "wan-vace: control clip frame count must be 1 + 4·k (got {})",
                clip.frames.len()
            )));
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

/// Wan-VACE descriptor — CFG (guidance + negative prompt), UniPC/Euler samplers, a `ControlClip` (the
/// universal VACE form the worker builds per mode) + optional `Reference` images. LoRA / quant deferred.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID_VACE,
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::ControlClip, ConditioningKind::Reference],
            supports_lora: false,
            supports_lokr: false,
            // sc-7296: curated `uni_pc` (native UniPC) + `euler`; legacy `unipc` alias for old recipes.
            samplers: vec!["uni_pc", "euler", "unipc"],
            schedulers: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Construct a lazy candle Wan-VACE generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a Wan2.1-VACE-14B diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`, `tokenizer/`).
/// LoRA / on-the-fly quantization are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "wan_vace expects a snapshot directory (text_encoder/ transformer/ vae/ tokenizer/), \
                 not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle wan_vace does not support LoRA/LoKr yet".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle wan_vace does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(WanVaceGenerator {
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::ReplacementMode;

    fn control_req() -> GenerationRequest {
        let frame = Image {
            width: 64,
            height: 64,
            pixels: vec![0u8; 64 * 64 * 3],
        };
        GenerationRequest {
            prompt: "a person walking".into(),
            width: 64,
            height: 64,
            guidance: Some(5.0),
            conditioning: vec![Conditioning::ControlClip {
                frames: vec![
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                ],
                mask: vec![
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    frame,
                ],
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn registers_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID_VACE, &spec).expect("wan_vace is registered");
        assert_eq!(g.descriptor().id, MODEL_ID_VACE);
        assert_eq!(g.descriptor().family, "wan");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.accepts(ConditioningKind::ControlClip));
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.samplers.contains(&"uni_pc")); // sc-7296 curated
        assert!(d.capabilities.samplers.contains(&"unipc")); // legacy alias
        assert!(!d.capabilities.supports_lora);
    }

    #[test]
    fn validate_accepts_control_clip_and_rejects_bad_shapes() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID_VACE, &spec).unwrap();
        assert!(g.validate(&control_req()).is_ok());

        // No control clip.
        let mut no_clip = control_req();
        no_clip.conditioning.clear();
        assert!(g.validate(&no_clip).is_err());

        // Frame count not 1 + 4·k (4 frames).
        let frame = Image {
            width: 64,
            height: 64,
            pixels: vec![0u8; 64 * 64 * 3],
        };
        let mut bad_count = control_req();
        bad_count.conditioning = vec![Conditioning::ControlClip {
            frames: vec![frame.clone(), frame.clone(), frame.clone(), frame.clone()],
            mask: vec![frame.clone(), frame.clone(), frame.clone(), frame.clone()],
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        }];
        assert!(g.validate(&bad_count).is_err());

        // Size not a multiple of 16.
        let mut bad_size = control_req();
        bad_size.width = 70;
        assert!(g.validate(&bad_size).is_err());
    }

    #[test]
    fn load_rejects_adapters_quant_and_single_file() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
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
        let file = LoadSpec::new(WeightsSource::File("/w.safetensors".into()));
        assert!(load(&file).is_err());
    }
}
