//! # candle-gen-qwen-image
//!
//! The **Qwen-Image** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-qwen-image`. Qwen-Image has **no** `candle-transformers` reference: the
//! 60-layer dual-stream MMDiT ([`transformer`]), the causal-Conv3d VAE ([`vae`]), and the Qwen2.5-VL
//! prompt-embeds path ([`text_encoder`]) are all ported here from the macOS provider.
//!
//! **txt2img (sc-3696):** [`QwenImageGenerator::generate`] runs Qwen2.5-VL (last normed hidden state,
//! 34 template tokens dropped → 3584-wide `prompt_embeds`) → the MMDiT (interleaved 3-axis RoPE,
//! dynamic-μ flow-match Euler, **true CFG** with norm-rescale) → the AutoencoderKLQwenImage decoder,
//! registered under `"qwen_image"`. Deterministic CPU-seeded noise (sc-3673); tokenization reuses
//! gen-core's [`TextTokenizer`] with [`ChatTemplate::QwenImage`].
//!
//! **Dtypes:** the Qwen2.5-VL encoder runs in **f32** (the fork rounds only the embeds to bf16) and
//! the 20B MMDiT in **bf16** (its native checkpoint dtype) — ~74 GB resident, which fits the 96 GB
//! Blackwell where an all-f32 load (~113 GB) would not.
//!
//! **First-slice surface:** txt2img only. The mlx provider's img2img / Edit / ControlNet / Lightning
//! / LoRA / quantization surface is **deferred** and rejected. `backend = "candle"`, `mac_only = false`.

// Qwen-Image-Edit inference adapter merge (sc-6220, epic 5480): fold a LoRA/LoKr `.safetensors` delta
// into the dense MMDiT weights at load — the Qwen-Image-Edit-2511-Lightning few-step distill, plus
// general Qwen-family LoRA/LoKr. Consumed by `edit::QwenEdit::load`.
pub mod adapters;
pub mod config;
// Qwen-Image ControlNet (strict pose) — the candle reference-pose lane (sc-5489, epic 5480). The pose
// skeleton is VAE-encoded + packed and fed to the InstantX control branch, whose per-block residuals
// inject into the frozen base MMDiT. A bespoke provider the worker drives directly (the registered
// `qwen_image` descriptor stays txt2img-only).
pub mod control;
// Qwen-Image **2512-Fun-Controlnet-Union** (VACE) control — the candle structural-control lane
// (sc-8350, mirrors mlx sc-8267). A `control_img_in` patch embedder feeds a control state threaded
// through 5 VACE control blocks (seeded by `before_proj`), each emitting a zero-init `after_proj` hint
// the base 2512 MMDiT adds at `control_layers = [0, 12, 24, 36, 48]`. Input-agnostic (pose/canny/depth
// share one path, no mode index). A bespoke provider the worker drives directly. The InstantX lane
// (`control`) is kept intact; its retirement is Phase B (sc-8246, the worker repo).
pub mod control_fun;
// Qwen-Image-Edit (img2img / reference) — the candle edit lane (sc-5487, epic 5480). The Qwen2.5-VL
// vision tower + image processor + VL splice turn a reference image + edit prompt into vision-
// conditioned prompt embeds (Slice A); the dual-latent `QwenEdit` provider (Slice B) VAE-encodes each
// reference, concatenates it after the noise, and denoises with the reference grids in the RoPE.
pub mod edit;
pub mod image_processor;
pub mod pipeline;
pub mod rope;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vision;
pub mod vision_language;
pub mod vl_tokenizer;

pub use control::{QwenControl, QwenControlPaths, QwenControlRequest, DEFAULT_CONTROL_SCALE};
pub use control_fun::{
    QwenFunControl, QwenFunControlPaths, QwenFunControlRequest, CONTROL_IN_DIM, CONTROL_LAYERS,
};
pub use edit::{QwenEdit, QwenEditPaths, QwenEditRequest};
pub use vision_language::{load_vision_language_encoder, QwenVisionLanguageEncoder};

/// Qwen-Image ControlNet (strict-pose) real-weight GPU validation (sc-5489) — env-driven, `#[ignore]`d.
#[cfg(test)]
mod control_validate;

/// Qwen-Image 2512-Fun-Controlnet-Union (VACE) real-weight GPU validation (sc-8350) — env-driven,
/// `#[ignore]`d.
#[cfg(test)]
mod control_fun_validate;

/// Qwen-Image-Edit vision-language encoder real-weight GPU validation (sc-5487) — env-driven, `#[ignore]`d.
#[cfg(test)]
mod vision_validate;

/// Qwen-Image-Edit full provider real-weight GPU validation (sc-5487) — env-driven, `#[ignore]`d.
#[cfg(test)]
mod edit_validate;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use config::{
    TextEncoderConfig, TransformerConfig, DEFAULT_GUIDANCE, DEFAULT_STEPS, MODEL_ID,
    NEGATIVE_FALLBACK, SIZE_MULTIPLE,
};
use text_encoder::QwenTextEncoder;
use transformer::QwenTransformer;
use vae::QwenVae;

/// The transformer is the 20B bottleneck — keep it bf16 (native dtype). The encoder + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;

#[derive(Clone)]
struct Components {
    te: Arc<QwenTextEncoder>,
    transformer: Arc<QwenTransformer>,
    vae: Arc<QwenVae>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    dit_cfg: TransformerConfig,
    root: PathBuf,
    device: Device,
}

impl Pipeline {
    fn load(root: &Path, device: &Device) -> Self {
        Self {
            te_cfg: TextEncoderConfig::qwen_image(),
            dit_cfg: TransformerConfig::qwen_image(),
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "qwen-image snapshot is missing the {sub}/ dir (expected a Qwen-Image diffusers \
                 snapshot at {})",
                self.root.display()
            )));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("qwen-image: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "qwen-image: no .safetensors in {sub}/ (at {})",
                dir.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &self.device)? };
        Ok(vb)
    }

    fn load_components(&self) -> CResult<Components> {
        let te = QwenTextEncoder::new(&self.te_cfg, self.component_vb("text_encoder", ENC_DTYPE)?)?;
        let transformer =
            QwenTransformer::new(&self.dit_cfg, self.component_vb("transformer", DIT_DTYPE)?)?;
        let vae = QwenVae::new(self.component_vb("vae", ENC_DTYPE)?)?;
        Ok(Components {
            te: Arc::new(te),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
        })
    }

    /// Tokenize + encode `prompt` → `prompt_embeds` `[1, seq, 3584]` at the DiT dtype (bf16).
    fn encode(&self, te: &QwenTextEncoder, prompt: &str) -> CResult<Tensor> {
        let tok = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.te_cfg.max_length,
                pad_token_id: self.te_cfg.pad_token_id,
                chat_template: ChatTemplate::QwenImage,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("qwen-image: load tokenizer: {e}")))?;
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("qwen-image: tokenize: {e}")))?;
        let len = out.ids.len();
        let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        Ok(te.prompt_embeds(&input_ids)?.to_dtype(DIT_DTYPE)?)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS as usize);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);

        let pos_embeds = self.encode(&comps.te, &req.prompt)?;
        // True CFG: build the negative branch unless guidance is a no-op (≤ 1.0).
        let neg_embeds = if guidance > 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
            Some(self.encode(&comps.te, neg)?)
        } else {
            None
        };

        // Routed through the unified curated sampler/scheduler framework (epic 7114 P4, sc-7123): the
        // `scheduler` axis picks the σ schedule over the production dynamic-μ shift (`native` = the
        // legacy `qwen_sigmas`), the `sampler` axis picks the integrator. The DEFAULT (`euler` over the
        // native schedule) is the N1 no-op — algebraically the legacy `euler_step` loop. The model is
        // fed the raw sigma (`Sigma` convention), and Qwen-Image is **true CFG** (a positive + negative
        // forward + norm-rescaled blend per step), so the whole pos/neg/blend lives inside the `predict`
        // closure — a multi-eval solver re-runs the whole closure.
        let native = pipeline::qwen_sigmas(steps, req.width, req.height);
        let mu = pipeline::qwen_mu(req.width, req.height);
        let sigmas =
            candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = base_seed.wrapping_add(index as u64);
            let latents = pipeline::create_noise(seed, req.width, req.height, &self.device)?
                .to_dtype(DIT_DTYPE)?;

            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                gen_core::sampling::TimestepConvention::Sigma,
                &sigmas,
                latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, sigma| -> CResult<Tensor> {
                    let pos =
                        comps
                            .transformer
                            .forward(latents, &pos_embeds, sigma, lat_h, lat_w)?;
                    match &neg_embeds {
                        Some(neg) => {
                            let neg = comps
                                .transformer
                                .forward(latents, neg, sigma, lat_h, lat_w)?;
                            Ok(pipeline::compute_guided_noise(&pos, &neg, guidance)?)
                        }
                        None => Ok(pos),
                    }
                },
            )?;

            on_progress(Progress::Decoding);
            let lat = pipeline::unpack_latents(&latents, req.width, req.height)?;
            let decoded = comps.vae.decode(&lat)?;
            images.push(to_image(&decoded)?);
        }
        Ok(images)
    }
}

fn to_image(decoded: &Tensor) -> CResult<Image> {
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

pub struct QwenImageGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

impl QwenImageGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("qwen-image components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = pipe.load_components()?;
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for QwenImageGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "qwen_image: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "qwen_image: steps must be >= 1".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "qwen_image: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
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
        let images = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Qwen-Image txt2img descriptor — the surface sc-3696 wires: true-CFG txt2img with a negative
/// prompt; no conditioning (img2img/Edit deferred), no LoRA/quant, no Lightning sampler.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["flow_match_euler"],
            ),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: true,
        },
    }
}

/// Construct a lazy candle Qwen-Image generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `Qwen/Qwen-Image` diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`,
/// `tokenizer/`). Adapters / quantization / control overlays are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "qwen_image expects a snapshot directory (text_encoder/ transformer/ vae/ \
                 tokenizer/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle qwen_image does not support LoRA/LoKr yet".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle qwen_image does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle qwen_image does not support control / Edit yet (txt2img only)".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(QwenImageGenerator {
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! { descriptor => load }

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("qwen_image is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "qwen-image");
        assert_eq!(g.descriptor().backend, "candle");
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.requires_sigma_shift);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.supports_lora);
        // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123): the full curated sampler menu,
        // and the curated scheduler menu plus the legacy `flow_match_euler` alias (N3 fallback).
        assert_eq!(
            d.capabilities.samplers,
            candle_gen::curated_sampler_names(),
            "samplers expose the curated menu"
        );
        assert!(
            d.capabilities.schedulers.contains(&"flow_match_euler"),
            "schedulers keep the legacy alias"
        );
        for s in candle_gen::curated_scheduler_names() {
            assert!(
                d.capabilities.schedulers.contains(&s),
                "scheduler menu missing {s}"
            );
        }
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            guidance: Some(4.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
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
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    /// sc-8647: a `Qwen/Qwen-Image-2512` snapshot is a structural drop-in for the original
    /// `Qwen/Qwen-Image` — same diffusers layout (`text_encoder/ transformer/ vae/ tokenizer/`),
    /// same 60-layer MMDiT, same Qwen2.5-VL text encoder, same Qwen2 BPE tokenizer (the worker's
    /// `DERIVED_TOKENIZER_OVERLAYS` materializes `tokenizer/tokenizer.json` for 2512 too). The
    /// candle loader keys nothing on the repo string — it loads the dir structurally — so a
    /// 2512-shaped snapshot is accepted exactly like the base. Pin that: a synthetic 2512 snapshot
    /// dir loads, and the per-release config used is byte-identical to the base config.
    #[test]
    fn loads_qwen_image_2512_shaped_snapshot() {
        // The 2512 base reuses the base config verbatim (sc-8271 parity); the candle loader uses
        // these for the DiT + text encoder regardless of which snapshot dir is supplied.
        assert_eq!(
            TransformerConfig::qwen_image_2512(),
            TransformerConfig::qwen_image(),
            "2512 MMDiT config must be a verbatim drop-in (same 60-layer dual-stream MMDiT)"
        );
        assert_eq!(
            TextEncoderConfig::qwen_image_2512(),
            TextEncoderConfig::qwen_image(),
            "2512 text-encoder config must be a verbatim drop-in (same Qwen2.5-VL + BPE tokenizer)"
        );

        // A 2512 snapshot ships the identical diffusers directory layout; the worker overlays a
        // built `tokenizer/tokenizer.json`. Build that shape and confirm the loader accepts it (no
        // repo-string gate rejects 2512) and that `Pipeline::load` resolves the tokenizer path that
        // `encode` reads.
        let tmp = std::env::temp_dir().join(format!("qwen2512_snap_{}", std::process::id()));
        for sub in ["text_encoder", "transformer", "vae", "tokenizer"] {
            std::fs::create_dir_all(tmp.join(sub)).unwrap();
        }
        std::fs::write(tmp.join("tokenizer/tokenizer.json"), b"{}").unwrap();

        let spec = LoadSpec::new(WeightsSource::Dir(tmp.clone()));
        let g = load(&spec).expect("a 2512-shaped snapshot dir must load like the base");
        assert_eq!(g.descriptor().id, MODEL_ID);

        let pipe = Pipeline::load(&tmp, &Device::Cpu);
        assert!(
            pipe.root.join("tokenizer/tokenizer.json").is_file(),
            "loader must resolve the overlaid tokenizer.json under tokenizer/"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }
}
