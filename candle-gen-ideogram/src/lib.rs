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
//! **Status (sc-6596):** scaffold + text encoder + scheduler + config in place; the single-stream DiT
//! (`transformer`) and the denoise `pipeline` are WIP, so [`Ideogram4Generator::generate`] currently
//! returns [`gen_core::Error::Unsupported`]. T2I (both variants), then Remix/edit (sc-6598), follow.
//! `backend = "candle"`, `mac_only = false`.

pub mod config;
pub mod scheduler;
pub mod text_encoder;

use std::path::PathBuf;

use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};

use config::{MODEL_ID, MODEL_ID_TURBO, SIZE_MULTIPLE};

/// The two-DiT quality DiT is the bottleneck — bf16 (native checkpoint dtype). Encoder + VAE run f32.
#[allow(dead_code)]
const DIT_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::BF16;
#[allow(dead_code)]
const ENC_DTYPE: candle_gen::candle_core::DType = candle_gen::candle_core::DType::F32;

/// A lazily-loaded Ideogram 4 generator. `turbo` selects the CFG-free single-DiT + TurboTime LoRA
/// path; otherwise the asymmetric two-DiT quality path.
pub struct Ideogram4Generator {
    descriptor: ModelDescriptor,
    #[allow(dead_code)]
    root: PathBuf,
    #[allow(dead_code)]
    turbo: bool,
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
        _on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        // WIP (sc-6596): the single-stream DiT + denoise pipeline are not wired yet. Config,
        // scheduler, and the Qwen3-VL text encoder are in place; the transformer + pipeline land
        // next, then GPU parity validation vs MLX.
        Err(gen_core::Error::Unsupported(format!(
            "candle {} render pipeline not yet implemented (single-stream DiT + pipeline WIP, sc-6596)",
            self.descriptor.id
        )))
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

fn build(spec: &LoadSpec, descriptor: ModelDescriptor, turbo: bool) -> gen_core::Result<Box<dyn Generator>> {
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
    Ok(Box::new(Ideogram4Generator {
        descriptor,
        root,
        turbo,
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
    fn generate_is_wip_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(matches!(
            g.generate(&req, &mut |_| {}).err().expect("WIP error"),
            gen_core::Error::Unsupported(_)
        ));
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
