//! # candle-gen-ideogram
//!
//! The **Ideogram 4** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-ideogram`. Registers two engine ids:
//!
//! * **`ideogram_4`** — the quality variant: a 9.3B single-stream flow-matching DiT with **asymmetric
//!   two-DiT CFG** (a conditional + an unconditional transformer) + a Qwen3-VL-8B text encoder.
//!   `V4_QUALITY_48` default (48 steps, guidance 7.0).
//! * **`ideogram_4_turbo`** — the same base + the bundled ostris **TurboTime LoRA** installed at load
//!   (single DiT, CFG-free, ~8 steps; guidance inert).
//!
//! **Reuse:** Ideogram's VAE is the FLUX.2 `AutoencoderKLFlux2`, reused verbatim from
//! [`candle_gen_flux2`] (`Flux2Vae`), exactly as the MLX provider reuses `mlx-gen-flux2`. The Qwen3-VL
//! text path ([`text_encoder`]) is adapted from flux2's Qwen3 encoder (θ=5e6, 13 interleaved
//! captured states). The single-stream DiT + the denoise pipeline are ported here.
//!
//! **Status (sc-6596):** both `ideogram_4` (quality, asymmetric two-DiT CFG) and `ideogram_4_turbo`
//! (CFG-free single DiT + the bundled TurboTime LoRA merged at load, [`adapters`]) are wired
//! end-to-end — Qwen3-VL text encode → single-stream DiT → VAE decode — pending GPU validation vs MLX
//! and the bf16 weight provisioning. Remix/edit is sc-6598. `backend = "candle"`, `mac_only = false`.

pub mod adapters;
pub mod config;
pub mod loader;
pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};

use config::{MODEL_ID, MODEL_ID_TURBO, SIZE_MULTIPLE};
use pipeline::Components;

/// A lazily-loaded Ideogram 4 generator. `turbo` selects the CFG-free single-DiT + TurboTime LoRA
/// path; otherwise the asymmetric two-DiT quality path. Components are loaded on the first
/// `generate` and cached.
pub struct Ideogram4Generator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    turbo: bool,
    device: Device,
    components: Mutex<Option<Arc<Components>>>,
}

impl Ideogram4Generator {
    fn components(&self) -> gen_core::Result<Arc<Components>> {
        let mut guard = self
            .components
            .lock()
            .expect("ideogram components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let components = if self.turbo {
            pipeline::load_components_turbo(&self.root, &self.device)?
        } else {
            pipeline::load_components(&self.root, &self.device)?
        };
        let c = Arc::new(components);
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for Ideogram4Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{}: prompt must not be empty",
                self.descriptor.id
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!(
                "{}: steps must be >= 1",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                self.descriptor.id, req.width, req.height
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

/// Ideogram 4 (quality) descriptor — asymmetric two-DiT CFG; no text negative prompt (the negative
/// branch is the trained unconditional DiT). T2I only for now; Remix/edit conditioning is sc-6598.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ideogram",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec![],
            schedulers: vec!["flow_match_euler"],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Ideogram 4 Turbo descriptor — same base, CFG-free single DiT + the bundled TurboTime LoRA;
/// guidance is inert (`supports_guidance = false`).
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = MODEL_ID_TURBO;
    d.capabilities.supports_guidance = false;
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
                "{} expects a snapshot directory (transformer/ [unconditional_transformer/] \
                 text_encoder/ vae/ tokenizer/), not a single .safetensors file",
                descriptor.id
            )));
        }
    };
    // User adapters / on-the-fly quant / control overlays are not wired (the turbo LoRA is bundled
    // in the snapshot and installed internally; edit conditioning is sc-6598).
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not accept user LoRA/LoKr (the TurboTime LoRA is bundled)",
            descriptor.id
        )));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not support on-the-fly Q4/Q8 quantization (load bf16 weights)",
            descriptor.id
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not support control / Edit yet (txt2img only; Remix/edit is sc-6598)",
            descriptor.id
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(Ideogram4Generator {
        descriptor,
        root,
        turbo,
        device,
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Ideogram 4 (quality) generator. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a candle-readable (bf16) Ideogram 4 snapshot.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor(), false)
}

/// Construct a lazy candle Ideogram 4 **Turbo** generator (CFG-free single DiT + bundled TurboTime
/// LoRA). The snapshot must additionally carry [`config::TURBO_LORA_FILE`].
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
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn registers_both_ids_as_candle() {
        for id in [MODEL_ID, MODEL_ID_TURBO] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g = registry::load(id, &spec).unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "ideogram");
            assert_eq!(g.descriptor().backend, "candle");
            assert!(!g.descriptor().capabilities.mac_only);
        }
    }

    #[test]
    fn descriptor_surfaces() {
        let q = descriptor();
        assert!(q.capabilities.supports_guidance);
        assert!(!q.capabilities.supports_negative_prompt);
        assert!(q.capabilities.conditioning.is_empty());
        assert!(!q.capabilities.accepts(ConditioningKind::Reference));
        let t = descriptor_turbo();
        assert_eq!(t.id, MODEL_ID_TURBO);
        assert!(!t.capabilities.supports_guidance);
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a neon city skyline at dusk".into(),
            guidance: Some(7.0),
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
    fn both_variants_reach_pipeline() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ideogram-snap".into()));
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // Both quality and turbo are wired — generate() passes the capability gate and fails on the
        // missing snapshot dir (a load error, NOT Unsupported), proving the pipeline path is reached.
        for id in [MODEL_ID, MODEL_ID_TURBO] {
            let g = registry::load(id, &spec).unwrap();
            let err = g.generate(&req, &mut |_| {}).unwrap_err();
            assert!(
                !matches!(err, gen_core::Error::Unsupported(_)),
                "{id} should reach the pipeline, got {err:?}"
            );
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
