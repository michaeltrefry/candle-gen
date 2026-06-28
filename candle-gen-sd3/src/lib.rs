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
pub mod pipeline;
pub mod transformer;
pub mod vae;

pub use config::Sd3Config;
pub use pipeline::Variant;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};

use pipeline::{Components, Pipeline};

/// Registry id for SD3.5 **Large** — matches the SceneWorks worker's `payload.model` and the macOS
/// `mlx-gen-sd3` SD3.5-Large descriptor.
pub const MODEL_ID: &str = "stable_diffusion_3_5_large";

/// Registry id for SD3.5 **Large Turbo** — the guidance-distilled 4-step sibling.
pub const MODEL_ID_TURBO: &str = "stable_diffusion_3_5_large_turbo";

/// SD3.5 works in latent space at /8 and the MMDiT patchifies that at /2, so both image dims must be
/// multiples of **16** for a clean patchify.
const SIZE_MULTIPLE: u32 = 16;

/// A loaded candle SD3.5 generator. Loading is **lazy**: `load` does no file I/O, and the heavy
/// components (triple text encoders + MMDiT + VAE) are built on the first
/// [`generate`](Generator::generate) call and then **cached** so back-to-back requests skip the disk
/// re-read. The [`Variant`] (Large vs Large Turbo) is fixed at load from the registered id.
pub struct Sd3Generator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    variant: Variant,
    /// Cached components. `Mutex` because `Generator` is shared and `generate` takes `&self`; the lock
    /// is held only to read/populate the cache, never across the denoise.
    components: Mutex<Option<Components>>,
}

impl Sd3Generator {
    /// The registry id for this generator's variant (used in validation error prefixes).
    fn model_id(&self) -> &'static str {
        match self.variant {
            Variant::Large => MODEL_ID,
            Variant::LargeTurbo => MODEL_ID_TURBO,
        }
    }

    /// Get the cached components, loading (and caching) them on a miss.
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("sd3 components cache mutex poisoned");
        if let Some(comps) = guard.as_ref() {
            return Ok(comps.clone());
        }
        let comps = pipe.load_components()?;
        *guard = Some(comps.clone());
        Ok(comps)
    }
}

impl Generator for Sd3Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.model_id();
        // The shared capability floor (count/size/negative/guidance/conditioning/sampler).
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
        // The light `Pipeline` handle carries the snapshot/device/variant; the heavy components come
        // from the cache. The rich-`CandleError` tail (including the typed `Canceled`) bridges into
        // `gen_core::Error` via `?`.
        let pipe = Pipeline::load(&self.root, &self.device, self.dtype, self.variant);
        let components = self.components(&pipe)?;
        let images = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// SD3.5 **Large**'s identity + the wired surface: txt2img with CFG + negative prompt (SD3.5 Large is
/// NOT guidance-distilled — it uses classifier-free guidance, unlike Turbo). Conditioning / LoRA /
/// quantization stay off the advertised surface until their own stories wire them.
pub fn descriptor() -> ModelDescriptor {
    descriptor_for(Variant::Large)
}

/// SD3.5 **Large Turbo**'s identity + wired surface: guidance-distilled txt2img (no CFG, no negative
/// prompt) — the 4-step sibling. Same architecture/encoders as Large; only the schedule + CFG-off
/// differ.
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(Variant::LargeTurbo)
}

/// Build the descriptor for `variant`. Large advertises CFG + negative prompt; Turbo (distilled)
/// advertises neither — so the shared `validate_request` rejects guidance/negative on Turbo (the
/// distilled-model honesty the Z-Image provider uses), keeping the worker from promising a path
/// `generate` ignores.
pub fn descriptor_for(variant: Variant) -> ModelDescriptor {
    let (id, cfg) = match variant {
        Variant::Large => (MODEL_ID, true),
        Variant::LargeTurbo => (MODEL_ID_TURBO, false),
    };
    ModelDescriptor {
        id,
        family: "stable-diffusion-3",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Large uses classifier-free guidance with a negative prompt; Turbo is guidance-distilled
            // (no CFG, no negative branch).
            supports_negative_prompt: cfg,
            supports_guidance: cfg,
            supports_true_cfg: false,
            // txt2img only for now (img2img / control land later); empty list ⇒ shared
            // `validate_request` rejects any conditioning.
            conditioning: vec![],
            // LoRA/LoKr not wired yet (a later story); refuse rather than silently drop.
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
            // SD3.5 is a flow-match model; the resolution-independent σ-shift is applied by the
            // pipeline, so it does not require the loader to pre-shift.
            requires_sigma_shift: false,
        },
    }
}

/// Construct the candle SD3.5 **Large** generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `stabilityai/stable-diffusion-3.5-large`-layout diffusers
/// snapshot (`text_encoder/`, `text_encoder_2/`, `text_encoder_3/`, `transformer/`, `vae/`, plus the
/// tokenizers). Quantization / control / adapters are refused (not wired yet).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Large)
}

/// Construct the candle SD3.5 **Large Turbo** generator (the guidance-distilled 4-step sibling) from a
/// `stabilityai/stable-diffusion-3.5-large-turbo`-layout snapshot.
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::LargeTurbo)
}

/// Shared loader for both variants. Lazy — no file I/O until the first `generate`.
fn load_variant(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Box<dyn Generator>> {
    let id = match variant {
        Variant::Large => MODEL_ID,
        Variant::LargeTurbo => MODEL_ID_TURBO,
    };
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (text_encoder/ text_encoder_2/ text_encoder_3/ \
                 transformer/ vae/), not a single .safetensors file"
            )));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support on-the-fly Q4/Q8 quantization yet (C4, sc-7879)"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter overlays (txt2img only)"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support LoRA/LoKr adapters yet"
        )));
    }
    // SD3.5 is a bf16 model; the device is the backend selected at compile time.
    let device = candle_gen::default_device()?;
    Ok(Box::new(Sd3Generator {
        descriptor: descriptor_for(variant),
        root,
        device,
        dtype: DType::BF16,
        variant,
        components: Mutex::new(None),
    }))
}

// Link-time self-registration into gen-core's model registry — both the Large and Turbo ids.
candle_gen::register_generators! { descriptor => load }
candle_gen::register_generators! { descriptor_turbo => load_turbo }

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

    /// C2 wires `generate`, so it now passes validation and proceeds to component load. Against a
    /// nonexistent snapshot it fails at load with a *non-`Unsupported`* error (the dir is missing) —
    /// i.e. the pipeline path is reached, not the old typed-`Unsupported` stub.
    #[test]
    fn generate_is_wired_and_loads_components() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let err = g
            .generate(&req, &mut progress)
            .expect_err("missing snapshot dir must error at load");
        // The pipeline IS wired now — the error is the missing component dir, not the C1 stub.
        assert!(
            !matches!(err, gen_core::Error::Unsupported(_)),
            "generate should reach the pipeline (load error), not the un-wired stub: {err:?}"
        );
    }

    /// Both the Large and Turbo ids register and resolve as candle generators with distinct surfaces.
    #[test]
    fn turbo_registers_and_advertises_distilled_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID_TURBO, &spec).expect("candle sd3 turbo is registered");
        assert_eq!(g.descriptor().id, MODEL_ID_TURBO);
        assert_eq!(g.descriptor().family, "stable-diffusion-3");
        assert_eq!(g.descriptor().backend, "candle");

        let d = descriptor_turbo();
        // Turbo is guidance-distilled: no CFG, no negative prompt.
        assert!(!d.capabilities.supports_guidance, "turbo is distilled");
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);

        // Turbo validate rejects guidance + negative prompt (distilled-model honesty).
        for bad in [
            GenerationRequest {
                prompt: "x".into(),
                guidance: Some(5.0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                negative_prompt: Some("blurry".into()),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "turbo should reject: {bad:?}");
        }
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
