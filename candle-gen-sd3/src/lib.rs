//! # candle-gen-sd3
//!
//! The **Stable Diffusion 3.5** provider crate for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-sd3` (epic 7841). It ports the SD3.5 architecture to candle:
//! the **MMDiT** (joint / double-stream) transformer ([`transformer`]), the **triple text-encoder
//! aggregator** ([`conditioning`]) that combines CLIP-L + OpenCLIP bigG + T5-XXL into the pooled
//! (2048) + context (333×4096) conditioning, and the 16-channel **VAE** wiring ([`vae`]).
//!
//! **C1 scope (sc-7876):** this story is the *architecture foundation* — the FULL SD3.5 Large
//! forward at real shapes, the aggregator, the VAE config, and crate registration, with
//! shape/structural parity tests. The actual T2I **pipeline** (text encoder forward + flow-match
//! sampler + VAE decode) and the CUDA real-weight smoke land in **C2** (sc-7877); the registered
//! descriptor here therefore advertises the txt2img surface but [`generate`](Generator::generate)
//! returns a typed `Unsupported` until C2 wires the pipeline. This keeps the worker honest (it falls
//! back rather than calling an un-wired path) while the registration + architecture are in place.
//!
//! **Parity approach (re-validated):** the macOS `mlx-gen-sd3` reference is NOT present on this
//! Windows machine, so correctness is established the way epic 7841 did — line-by-line against the
//! PUBLIC diffusers `SD3Transformer2DModel` / `StableDiffusion3Pipeline` and the SD3 paper, plus
//! structural/shape unit tests (the `#[cfg(test)]` modules in each sub-module). Any cross-engine
//! numeric A/B harness is left `#[ignore]`d (reference unavailable here); coherent real-weight
//! rendering is validated in C2/C6.
//!
//! **Key SD3.5 vs FLUX parity note:** SD3.5's MMDiT uses a **learned 2D positional embedding** added
//! at patchify and runs attention with **NO RoPE** — do not copy FLUX's rotary embedding. The AdaLN
//! modulation apply order (`x·(1+scale)+shift`, with the correct chunk split) is the documented bug
//! magnet; see [`transformer`].

pub mod conditioning;
pub mod config;
pub mod transformer;
pub mod vae;

pub use config::Sd3Config;

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};

/// Registry id — matches the SceneWorks worker's `payload.model` and the macOS `mlx-gen-sd3`
/// SD3.5-Large descriptor.
pub const MODEL_ID: &str = "stable_diffusion_3_5_large";

/// SD3.5 works in latent space at /8 and the MMDiT patchifies that at /2, so both image dims must be
/// multiples of **16** for a clean patchify.
const SIZE_MULTIPLE: u32 = 16;

/// A loaded candle SD3.5 generator. C1 holds the load context (snapshot root + device/dtype); the
/// heavy components and the render loop are wired in C2.
pub struct Sd3Generator {
    descriptor: ModelDescriptor,
    #[allow(dead_code)]
    root: PathBuf,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl Generator for Sd3Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor (count/size/negative/guidance/conditioning/sampler).
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "stable_diffusion_3_5_large: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "stable_diffusion_3_5_large: steps must be >= 1".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "stable_diffusion_3_5_large: width/height must be multiples of {SIZE_MULTIPLE} \
                 (got {}x{})",
                req.width, req.height
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
        // The architecture (MMDiT + aggregator + VAE) is built in C1; the txt2img pipeline that
        // drives them (text-encoder forward, flow-match sampler, VAE decode) + the CUDA real-weight
        // smoke is C2 (sc-7877). Refuse loudly until then so the worker falls back rather than
        // calling an un-wired render path.
        Err(gen_core::Error::Unsupported(
            "candle stable_diffusion_3_5_large: the txt2img pipeline is not wired yet (C2, \
             sc-7877); C1 (sc-7876) lands the architecture + aggregator + VAE only"
                .into(),
        ))
    }
}

/// SD3.5 Large's identity + the surface C2 will wire: txt2img with CFG + negative prompt (SD3.5 is
/// NOT guidance-distilled — it uses classifier-free guidance, unlike Z-Image-Turbo). Conditioning /
/// LoRA / quantization stay off the advertised surface until their own stories wire them.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "stable-diffusion-3",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SD3.5 uses classifier-free guidance with a negative prompt (the base, non-distilled
            // checkpoint). The Turbo variant (C2) is guidance-distilled, but the Large id here is the
            // CFG model.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // txt2img only for now (img2img / control land later); empty list ⇒ shared
            // `validate_request` rejects any conditioning.
            conditioning: vec![],
            // LoRA/LoKr not wired in C1 (a later story); refuse rather than silently drop.
            supports_lora: false,
            supports_lokr: false,
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            supported_quants: &[],
            supports_kv_cache: false,
            // SD3.5 is a flow-match model; the resolution-dependent σ-shift is applied by the C2
            // pipeline, so it does not require the loader to pre-shift.
            requires_sigma_shift: false,
        },
    }
}

/// Construct the candle SD3.5 generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `stabilityai/stable-diffusion-3.5-large`-layout diffusers
/// snapshot (`text_encoder/`, `text_encoder_2/`, `text_encoder_3/`, `transformer/`, `vae/`, plus the
/// tokenizers). Quantization / control / adapters are refused (not wired in C1).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "stable_diffusion_3_5_large expects a snapshot directory (text_encoder/ \
                 text_encoder_2/ text_encoder_3/ transformer/ vae/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle stable_diffusion_3_5_large does not support on-the-fly Q4/Q8 quantization yet \
             (C4, sc-7879)"
                .into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle stable_diffusion_3_5_large does not support control / IP-adapter overlays \
             (txt2img only)"
                .into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle stable_diffusion_3_5_large does not support LoRA/LoKr adapters yet".into(),
        ));
    }
    // SD3.5 is a bf16 model; the device is the backend selected at compile time.
    let device = candle_gen::default_device()?;
    Ok(Box::new(Sd3Generator {
        descriptor: descriptor(),
        root,
        device,
        dtype: DType::BF16,
    }))
}

// Link-time self-registration into gen-core's model registry.
candle_gen::register_generators! { descriptor => load }

/// Force-link hook (see `candle_gen_z_image::force_link`): a consumer that reaches this provider only
/// through the registry references nothing here directly, so the linker can drop the rlib and its
/// `inventory::submit!`. Referencing this no-op keeps it linked.
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

    #[test]
    fn sd3_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("candle sd3 is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "stable-diffusion-3");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_cfg_txt2img_surface() {
        let d = descriptor();
        // SD3.5 Large uses classifier-free guidance with a negative prompt (NOT distilled).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.supports_lora);
        assert!(d.capabilities.supported_quants.is_empty());
        assert_eq!(d.capabilities.min_size, 256);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
        assert!(!descriptor()
            .capabilities
            .accepts(ConditioningKind::Reference));
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 1000, // not multiple of 16
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                conditioning: vec![Conditioning::Reference {
                    image: Image::default(),
                    strength: None,
                }],
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn generate_refuses_until_c2_pipeline() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let err = g
            .generate(&req, &mut progress)
            .expect_err("C1 generate is un-wired");
        assert!(
            matches!(err, gen_core::Error::Unsupported(_)),
            "got: {err:?}"
        );
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        // `Box<dyn Generator>` is not `Debug`, so `unwrap_err()` won't compile here; bind the error
        // with a `let Err(..) else { panic! }` instead.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        let Err(e) = load(&quant) else {
            panic!("quant load must be refused")
        };
        assert!(matches!(e, gen_core::Error::Unsupported(_)), "got: {e:?}");

        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        let Err(e) = load(&lora) else {
            panic!("lora load must be refused")
        };
        assert!(matches!(e, gen_core::Error::Unsupported(_)), "got: {e:?}");
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sd3.safetensors".into()));
        let Err(e) = load(&spec) else {
            panic!("single-file source must be refused")
        };
        assert!(e.to_string().contains("snapshot directory"), "got: {e}");
    }
}
