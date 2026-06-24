//! # candle-gen-krea
//!
//! The **Krea 2** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-krea`. Registers one engine id:
//!
//! * **`krea_2_turbo`** — the user-facing text-to-image model: a 12B **dense single-stream**
//!   rectified-flow / v-param DiT (28 gated single-stream blocks, hidden 6144, GQA 48Q/12KV, head_dim
//!   128, SwiGLU 16384, 3-axis interleaved RoPE `[32,48,48]`, `DoubleSharedModulation`, and a
//!   `text_fusion` front-end that aggregates the 12 selected Qwen3-VL hidden layers) driven by a
//!   Qwen3-VL-4B condition encoder and the Qwen-Image VAE. TDM-distilled few-step (8 steps),
//!   **CFG-free** (guidance inert), up to 2048².
//!
//! **Reuse:** the VAE is `candle_gen_qwen_image::vae::QwenVae` (the exact `AutoencoderKLQwenImage`
//! Qwen-Image ships — per-channel `latents_mean`/`latents_std` de-norm) — reused verbatim, as
//! `mlx-gen-krea` reuses `mlx-gen-qwen-image`'s `QwenVae`. The Qwen3-VL-4B condition encoder
//! ([`text_encoder`]), the single-stream DiT ([`transformer`]), and the rectified-flow sampler
//! ([`schedule`]) are ported here.
//!
//! `backend = "candle"`, `mac_only = false`. Apache-2.0; Krea 2 Community License (non-commercial use
//! satisfies it). The Q4/Q8 turnkey + worker quant gating is sc-7581; the Story-1 slice loads dense
//! bf16.

pub mod adapters;
pub mod config;
pub mod convert;
pub mod loader;
pub mod pipeline;
pub mod schedule;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vae;

// The candle Krea LoRA/LoKr trainer (sc-7577) + its vendored composable-op trainable DiT. Private
// (reached through gen-core's trainer registry by id, like the SDXL/Z-Image trainers); the
// `inventory::submit!` in `training` is kept linked by [`force_link`].
mod train_dit;
mod training;

pub use adapters::{merge_adapters, merge_into_weights, MergeReport};
pub use config::Krea2Config;
pub use pipeline::Components;
pub use schedule::{krea_sigmas, turbo_sigmas, TURBO_MU, TURBO_STEPS};
pub use text_encoder::{KreaTeConfig, KreaTextEncoder};
pub use tokenizer::KreaTokenizer;
pub use transformer::Krea2Transformer;
pub use vae::{load_vae, QwenVae};

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Progress, Quant, WeightsSource,
};

/// Registry id for the Krea 2 Turbo text-to-image variant. Matches the SceneWorks worker's
/// `payload.model` and the manifest `engine_id` (sc-7572).
pub const KREA_2_TURBO_ID: &str = "krea_2_turbo";

/// patch_size(2)·vae_downsample(8) = 16 — patchify requires latent dims divisible by this.
const SIZE_MULTIPLE: u32 = 16;
/// Resolution bounds (W/H). Turbo renders up to 2048²; the catalog/worker gate the UI options tighter.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 2048;
/// Max images per request (the image-model standard, shared with the other families).
const MAX_COUNT: u32 = 8;

/// A lazily-loaded Krea 2 Turbo generator. The components (tokenizer + Qwen3-VL-4B TE + single-stream
/// DiT + Qwen-Image VAE) load on the first `generate` and are cached.
pub struct KreaGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// LoRA/LoKr adapters merged into the DiT weights at component-load (sc-7836). Fixed for this
    /// generator instance; empty ⇒ the stock unadapted build.
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Arc<Components>>>,
}

impl KreaGenerator {
    fn components(&self) -> gen_core::Result<Arc<Components>> {
        let mut guard = self
            .components
            .lock()
            .expect("krea components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = Arc::new(pipeline::load_components(
            &self.root,
            &self.device,
            &self.adapters,
        )?);
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for KreaGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
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
        let comps = self.components()?;
        let images = pipeline::render(&comps, req, &self.device, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Krea 2 Turbo identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement). Distilled few-step text-to-image: **CFG-free** (the TDM
/// distillation baked the guided velocity into the weights, so no guidance / unconditional branch), no
/// user negative prompt, no img2img/control conditioning on the Turbo checkpoint.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: KREA_2_TURBO_ID,
        family: "krea_2",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // CFG-free distilled student (like Ideogram Turbo / Boogu Turbo / SDXL-Lightning).
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            // LoRA/LoKr wired (sc-7836): a trained `krea_2_raw` adapter merges into the dense DiT
            // attention projections at load ([`adapters::merge_into_weights`]), closing the candle
            // train→infer loop. On-the-fly Q4/Q8 quantization is still deferred (rejected at load).
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow v-param over the unified curated-sampler framework (epic 7114). The
            // native distilled loop stays the byte-exact default (`req.sampler == None`).
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: false,
            // Story-1 slice is dense bf16; the Q4/Q8 turnkey + load-time quant gating is sc-7581.
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

fn build(spec: &LoadSpec, descriptor: ModelDescriptor) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a snapshot directory (transformer/ text_encoder/ vae/ tokenizer/), not a \
                 single .safetensors file",
                descriptor.id
            )));
        }
    };
    // LoRA/LoKr adapters are accepted and merged into the DiT at first `generate` (sc-7836); the merge
    // (`adapters::merge_into_weights`) is lazy, so a nonexistent adapter path still loads here.
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not yet support on-the-fly Q4/Q8 quantization (load bf16 weights); the \
             packed turnkey is sc-7581",
            descriptor.id
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not support ControlNet / IP-Adapter overlays",
            descriptor.id
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(KreaGenerator {
        descriptor,
        root,
        device,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Krea 2 **Turbo** generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a candle-readable (bf16) Krea 2 snapshot (`transformer/ text_encoder/ vae/ tokenizer/`).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor())
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

/// Force-link hook (keeps the `inventory::submit!` registrations — the `krea_2_turbo` generator and the
/// `krea_2_raw` trainer — from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_krea_2_turbo_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_TURBO_ID, &spec).expect("krea_2_turbo is registered");
        assert_eq!(g.descriptor().id, KREA_2_TURBO_ID);
        assert_eq!(g.descriptor().family, "krea_2");
        assert_eq!(g.descriptor().backend, "candle");
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn descriptor_surface_is_cfg_free_turbo() {
        let d = descriptor();
        assert_eq!(d.id, KREA_2_TURBO_ID);
        assert_eq!(d.modality, Modality::Image);
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.conditioning.is_empty());
        // LoRA/LoKr merge wired (sc-7836); on-the-fly quant still deferred.
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert!(d.capabilities.supported_quants.is_empty());
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(TURBO_STEPS, 8);
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_bad() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_TURBO_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                height: 1024,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 1024,
                height: 1024,
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn validate_rejects_guidance_and_negative_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(KREA_2_TURBO_ID, &spec).unwrap();
        let base = GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            ..Default::default()
        };
        assert!(g
            .validate(&GenerationRequest {
                guidance: Some(3.5),
                ..base.clone()
            })
            .is_err());
        assert!(g
            .validate(&GenerationRequest {
                negative_prompt: Some("y".into()),
                ..base
            })
            .is_err());
    }

    #[test]
    fn load_accepts_lora_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let file = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        assert!(load(&file).is_err());
        // LoRA/LoKr now wired (sc-7836): a LoRA `LoadSpec` is accepted (lazily — the merge happens at
        // first `generate`), so `load` resolves rather than rejecting.
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load(&lora).is_ok(), "LoRA load is wired + lazy (sc-7836)");
        // On-the-fly quantization is still deferred — a typed `Unsupported` so the worker can fall back.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
