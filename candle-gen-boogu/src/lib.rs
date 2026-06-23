//! # candle-gen-boogu
//!
//! The **Boogu-Image-0.1** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-boogu`. Registers two text-to-image engine ids:
//!
//! * **`boogu_image`** — the Base variant: a 10.3B Lumina-Image-2.0 / OmniGen2-lineage hybrid MMDiT
//!   (8 double + 32 single + 2 refiner layers, GQA, 3-axis interleaved RoPE) with true-CFG, driven by
//!   a Qwen3-VL-8B condition encoder and a FLUX.1 16-channel VAE. 50-step rectified-flow Euler over a
//!   static-shift (`mu = 1.15`) schedule, routed through the unified curated-sampler framework.
//! * **`boogu_image_turbo`** — the same Base weights-arch + a DMD-distilled few-step (4) sampler loop,
//!   CFG-free (guidance inert). The default fast surface.
//!
//! **Reuse:** the FLUX.1 VAE is `candle-transformers`' `z_image::vae::AutoEncoderKL` (the exact 16-ch
//! AutoencoderKL Z-Image ships, scaling 0.3611 / shift 0.1159) — reused verbatim, as `mlx-gen-boogu`
//! reuses `mlx-gen-z-image`'s `Vae`. The Qwen3-VL-8B condition encoder + the hybrid DiT are ported
//! here. The Edit (single-reference TI2I) variant lands in sc-7523.
//!
//! `backend = "candle"`, `mac_only = false`. Apache-2.0, ungated.

pub mod config;
pub mod loader;
pub mod pipeline;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};

use pipeline::Components;

/// Registry id for the Base text-to-image variant (true-CFG).
pub const BOOGU_IMAGE_ID: &str = "boogu_image";
/// Registry id for the Turbo variant (DMD few-step, CFG-free).
pub const BOOGU_IMAGE_TURBO_ID: &str = "boogu_image_turbo";

/// Patch(2)·ae_scale(8) = 16 — `patchify` requires latent dims divisible by this.
const SIZE_MULTIPLE: u32 = 16;

/// The curated samplers the Turbo DMD student stays coherent under (the stochastic / re-noising
/// solvers — `lcm` most of all). The deterministic ODE solvers feed the few-step student
/// out-of-regime latents, so they stay off the menu. Mirrors `mlx-gen-boogu`'s `TURBO_SAMPLERS`.
const TURBO_SAMPLERS: &[&str] = &["lcm", "euler_ancestral", "dpmpp_sde"];

/// A lazily-loaded Boogu generator. `turbo` selects the CFG-free DMD few-step path; otherwise the
/// true-CFG Base path. Components are loaded on the first `generate` and cached.
pub struct BooguGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    turbo: bool,
    device: Device,
    components: Mutex<Option<Arc<Components>>>,
}

impl BooguGenerator {
    fn components(&self) -> gen_core::Result<Arc<Components>> {
        let mut guard = self
            .components
            .lock()
            .expect("boogu components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = Arc::new(pipeline::load_components(&self.root, &self.device)?);
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for BooguGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
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
        let images = if self.turbo {
            pipeline::render_turbo(&comps, req, &self.device, on_progress)?
        } else {
            pipeline::render_base(&comps, req, &self.device, on_progress)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// Boogu Base descriptor — true-CFG text-to-image; no user negative prompt (the CFG-negative is the
/// model's own fixed empty/drop instruction); no img2img/control conditioning on the Base checkpoint.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: BOOGU_IMAGE_ID,
        family: "boogu",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            // Base is rectified-flow Euler over the static-shift schedule, routed through the unified
            // curated-sampler framework (epic 7114).
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            // Story-1 slice is dense bf16; load-time Q4/Q8 quant gating is sc-7524 worker wiring.
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Boogu Turbo descriptor — same base, CFG-free DMD few-step; guidance is inert. The advertised
/// sampler menu is the DMD-compatible stochastic subset ([`TURBO_SAMPLERS`]).
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_TURBO_ID;
    d.capabilities.supports_guidance = false;
    d.capabilities.samplers = TURBO_SAMPLERS.to_vec();
    d
}

fn build(
    spec: &LoadSpec,
    descriptor: ModelDescriptor,
    turbo: bool,
) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a snapshot directory (mllm/ transformer/ vae/), not a single \
                 .safetensors file",
                descriptor.id
            )));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not accept user LoRA/LoKr adapters",
            descriptor.id
        )));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not yet support on-the-fly Q4/Q8 quantization (load bf16 weights)",
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
    Ok(Box::new(BooguGenerator {
        descriptor,
        root,
        turbo,
        device,
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Boogu **Base** generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a candle-readable (bf16) Boogu snapshot (`mllm/ transformer/ vae/`).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor(), false)
}

/// Construct a lazy candle Boogu **Turbo** generator (DMD few-step, CFG-free).
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor_turbo(), true)
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}
inventory::submit! {
    ModelRegistration {
        descriptor: descriptor_turbo,
        load: load_turbo,
    }
}

/// Force-link hook (keeps the `inventory::submit!` registrations from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_both_ids_as_candle() {
        for id in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g = registry::load(id, &spec).unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "boogu");
            assert_eq!(g.descriptor().backend, "candle");
            assert!(!g.descriptor().capabilities.mac_only);
        }
    }

    #[test]
    fn descriptor_surfaces() {
        let b = descriptor();
        assert!(b.capabilities.supports_guidance);
        assert!(!b.capabilities.supports_negative_prompt);
        assert!(b.capabilities.conditioning.is_empty());
        let t = descriptor_turbo();
        assert_eq!(t.id, BOOGU_IMAGE_TURBO_ID);
        assert!(!t.capabilities.supports_guidance);
        assert_eq!(t.capabilities.samplers, TURBO_SAMPLERS.to_vec());
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_bad() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(BOOGU_IMAGE_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            guidance: Some(4.0),
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
    fn load_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let file = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        assert!(load(&file).is_err());
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
}
