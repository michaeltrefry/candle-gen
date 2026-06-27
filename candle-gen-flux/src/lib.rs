//! # candle-gen-flux
//!
//! The **FLUX.1** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling
//! of `mlx-gen-flux`. It implements the backend-neutral [`gen_core::Generator`] contract and
//! self-registers via `inventory` for **both** FLUX.1 variants, so linking this crate makes
//! `gen_core::load("flux1_schnell", …)` / `gen_core::load("flux1_dev", …)` resolve the candle FLUX
//! generators.
//!
//! **txt2img (sc-3694):** [`FluxGenerator::generate`] adapts the `candle-transformers` `flux`
//! reference model ([`pipeline`]) through the contract: dual **CLIP-L + T5-XXL** text encoders → the
//! FLUX **DiT** (flow-match Euler) → FLUX **AutoEncoder** VAE, emitting `Progress` and honoring
//! `req.cancel`, with **deterministic CPU-seeded noise** (sc-3673) so output is launch-portable per
//! seed. The two variants:
//!
//! - **`flux1_schnell`** — Apache-2.0, timestep-distilled: a fixed **4-step** schedule, **no
//!   guidance** (the DiT has no guidance embedding), no negative prompt.
//! - **`flux1_dev`** — guidance-distilled: **25 steps** by default with a resolution-dependent
//!   time-shifted schedule and an embedded **guidance** scale (default 3.5, mlx parity). FLUX.1[dev]
//!   is a **gated** model (a non-commercial license + an accepted HF license agreement); the engine
//!   consumes already-staged weights and does not itself perform credential/license gating — that
//!   stays upstream in the worker's weight-staging layer, **consistent with the mlx provider** (which
//!   likewise carries no gating flag on the descriptor).
//!
//! The descriptors advertise **only** the wired txt2img surface — NOT the full mlx-gen-flux
//! Reference/IP-adapter, LoRA, or Q4/Q8 surface — so the worker routes the rest to the Python
//! fallback rather than the candle backend silently dropping a control (the false-capability trap,
//! exactly as the SDXL and Z-Image slices did). `backend` is `"candle"` and `mac_only` is `false`.

mod pipeline;

// XLabs FLUX IP-Adapter (sc-5872, epic 5480) — reference-image (identity) conditioning. `ip_dit` is the
// forked FLUX DiT carrying the per-double-block decoupled-cross-attn seam (the stock candle-transformers
// `Flux` has none); `ip_adapter` is the XLabs projector + K/V weights; `ip_image_encoder` is the pooled
// CLIP-ViT-L tower; `ip_provider` composes them into the bespoke reference stream the worker drives
// directly (not gen-core-registered — the `flux1_*` descriptors stay txt2img-only).
pub mod ip_adapter;
mod ip_dit;
pub mod ip_image_encoder;
pub mod ip_provider;
pub use ip_provider::{IpAdapterFlux, IpAdapterFluxPaths, IpAdapterFluxRequest, DEFAULT_IP_SCALE};

// The vendored FLUX DiT + its post-block image-stream injector seam, re-exported for the PuLID-FLUX
// provider (`candle-gen-pulid`, sc-5492), which composes the FLUX backbone with the EVA-CLIP tower +
// IDFormer + the 20 PerceiverAttentionCA modules driven through [`DitImageInjector`]. `Config` is the
// candle-transformers FLUX config the fork reuses (so it cannot drift on hyperparameters).
pub use ip_dit::{Config as FluxConfig, DitImageInjector, IpFlux};
// FLUX backbone helpers shared with the PuLID provider so the two never drift on the parity-critical
// tokenization / VAE decode / config (the candle twin of `mlx-gen-flux`'s shared `Flux1` surface). The
// IP-Adapter provider reaches these as `pub(crate)`; PuLID is a separate crate, hence `pub`.
pub use pipeline::{ae_config, clip_config, decode_latents, encode_text, flux_config};

/// FLUX XLabs IP-Adapter real-weight GPU validation (sc-5872) — env-driven, `#[ignore]`d integration
/// test (the analog of the SDXL/Kolors IP-Adapter Phase-5 harnesses).
#[cfg(test)]
mod ip_validate;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};

use pipeline::{Components, Pipeline};

/// Registry id for FLUX.1 [schnell] — matches the SceneWorks worker's engine id and the macOS
/// `mlx-gen-flux` descriptor.
pub const FLUX1_SCHNELL_ID: &str = "flux1_schnell";
/// Registry id for FLUX.1 [dev].
pub const FLUX1_DEV_ID: &str = "flux1_dev";

/// FLUX works in the VAE's /8 latent and the DiT packs that 2×2, so both image dims must be multiples
/// of **16** for a clean pack. Enforced in [`validate`](Generator::validate).
const SIZE_MULTIPLE: u32 = 16;

/// The two FLUX.1 variants. Carries the parity-critical per-variant metadata (id, step/guidance
/// defaults, T5 length, checkpoint filename) as primitives so `lib.rs` stays candle-light — the
/// pipeline maps the variant onto candle's `flux`/`autoencoder` configs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Schnell,
    Dev,
}

impl Variant {
    /// The registry / engine id.
    pub const fn model_id(self) -> &'static str {
        match self {
            Variant::Schnell => FLUX1_SCHNELL_ID,
            Variant::Dev => FLUX1_DEV_ID,
        }
    }

    /// Distilled default step count (mlx parity): schnell 4, dev 25.
    pub const fn default_steps(self) -> u32 {
        match self {
            Variant::Schnell => 4,
            Variant::Dev => 25,
        }
    }

    /// Whether the DiT embeds a guidance scale. schnell is timestep-distilled (no guidance); dev is
    /// guidance-distilled. Drives both the descriptor's `supports_guidance` and the denoise.
    pub const fn supports_guidance(self) -> bool {
        matches!(self, Variant::Dev)
    }

    /// Default guidance scale when a dev request omits one (mlx `DEFAULT_GUIDANCE`). Inert for schnell.
    pub const fn default_guidance(self) -> f32 {
        3.5
    }

    /// T5 sequence length the prompt is padded to (diffusers FluxPipeline default): schnell 256,
    /// dev 512. FLUX attends every T5 position, so this is parity-critical.
    pub const fn t5_max_len(self) -> usize {
        match self {
            Variant::Schnell => 256,
            Variant::Dev => 512,
        }
    }

    /// The root BFL DiT checkpoint filename in the snapshot.
    pub const fn transformer_file(self) -> &'static str {
        match self {
            Variant::Schnell => "flux1-schnell.safetensors",
            Variant::Dev => "flux1-dev.safetensors",
        }
    }

    /// Whether this is the dev variant (guidance + time-shifted schedule).
    pub const fn is_dev(self) -> bool {
        matches!(self, Variant::Dev)
    }
}

/// A loaded candle FLUX generator (one per variant). Loading is **lazy**: `load` does no file I/O,
/// and the heavy components (CLIP + T5 + DiT + VAE) are built on the first
/// [`generate`](Generator::generate) call and then **cached** in `components` so back-to-back
/// requests skip the disk re-read.
pub struct FluxGenerator {
    variant: Variant,
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// Cached components. `Mutex` because `Generator` is shared and `generate` takes `&self`; the lock
    /// is held only to read/populate the cache, never across the denoise.
    components: Mutex<Option<Components>>,
}

impl FluxGenerator {
    /// Get the cached components, loading (and caching) them on a miss.
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("flux components cache mutex poisoned");
        if let Some(comps) = guard.as_ref() {
            return Ok(comps.clone());
        }
        let comps = pipe.load_components()?;
        *guard = Some(comps.clone());
        Ok(comps)
    }
}

impl Generator for FluxGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor: the descriptor advertises no conditioning and (for schnell) no
        // guidance / negative prompt, so any of those is rejected here (distilled-model honesty).
        self.descriptor
            .capabilities
            .validate_request(self.variant.model_id(), req)?;
        let id = self.variant.model_id();
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        // An explicit `steps: Some(0)` would VAE-decode pure noise — reject loudly (txt2img-only).
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
            )));
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
        // The rich-`CandleError` tail — including the typed `Canceled` — bridges into
        // `gen_core::Error` via `?`. The light `Pipeline` handle carries the snapshot/device; the
        // heavy components come from the cache.
        let pipe = Pipeline::load(self.variant, &self.root, &self.device, self.dtype);
        let components = self.components(&pipe)?;
        let images = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// The descriptor for a FLUX.1 `variant` — the surface sc-3694 actually wires: txt2img only (no
/// conditioning / LoRA / quantization advertised — those are the Python fallback's job until candle
/// wires them), dev exposes guidance (schnell does not), no negative prompt / true-CFG. `backend` is
/// `"candle"` and `mac_only` is `false` (the two backend-correct deviations from `mlx-gen-flux`).
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    ModelDescriptor {
        id: variant.model_id(),
        family: "flux",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Negative prompt / true-CFG ride the mlx Reference path (dev + reference = real CFG),
            // which this txt2img slice does not wire — so neither is advertised on either variant.
            supports_negative_prompt: false,
            supports_guidance: variant.supports_guidance(),
            supports_true_cfg: false,
            // txt2img only in sc-3694 — Reference/IP-adapter lands later; an empty list means the
            // shared `validate_request` rejects any conditioning and the worker keeps those shapes on
            // the Python path.
            conditioning: vec![],
            // LoRA/LoKr (mlx supports both) and Q4/Q8 quantization are deferred to a later slice; not
            // advertised, and rejected at load rather than silently dropped.
            supports_lora: false,
            supports_lokr: false,
            // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123): the denoise routes
            // through the shared driver, so the per-generation `sampler`/`scheduler` knob can select any
            // curated integrator/schedule. The DEFAULT (None/None) reproduces the native flow-match
            // Euler path (N1). FLUX had no legacy sampler/scheduler aliases, so no `menu_with_aliases`.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            mac_only: false,
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// FLUX.1 [schnell] descriptor (registry).
pub fn descriptor_schnell() -> ModelDescriptor {
    descriptor_for(Variant::Schnell)
}

/// FLUX.1 [dev] descriptor (registry).
pub fn descriptor_dev() -> ModelDescriptor {
    descriptor_for(Variant::Dev)
}

/// Construct a lazy candle FLUX generator for `variant` from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a black-forest-labs `FLUX.1-{schnell,dev}` snapshot (root
/// `flux1-*.safetensors` + `ae.safetensors`, plus the `text_encoder/`, `text_encoder_2/`,
/// `tokenizer_2/` subdirs). LoRA adapters, on-the-fly quantization, and control/IP-adapter overlays
/// are rejected — none are wired in this slice, so refusing is more honest than silently dropping
/// them (the worker falls back to Python).
fn load_variant(variant: Variant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.model_id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (flux1-*.safetensors, ae.safetensors, \
                 text_encoder/, text_encoder_2/, tokenizer_2/), not a single .safetensors file"
            )));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support LoRA/LoKr yet — refusing to silently drop the adapters"
        )));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support on-the-fly Q4/Q8 quantization yet"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter overlays yet (txt2img only)"
        )));
    }
    // FLUX is a bf16 model; load at bf16 regardless of the CPU-default dtype. The device is the
    // backend selected at compile time (CUDA on Windows, Metal/CPU on Mac).
    let device = candle_gen::default_device()?;
    Ok(Box::new(FluxGenerator {
        variant,
        descriptor: descriptor_for(variant),
        root,
        device,
        dtype: DType::BF16,
        components: Mutex::new(None),
    }))
}

/// Registry entry point for FLUX.1 [schnell].
pub fn load_schnell(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Variant::Schnell, spec)
}

/// Registry entry point for FLUX.1 [dev].
pub fn load_dev(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Variant::Dev, spec)
}

// Link-time self-registration into gen-core's model registry — one descriptor per variant. Linking
// this crate makes `gen_core::load("flux1_schnell"/"flux1_dev", …)` resolve the candle generators
// with no central match to edit.
candle_gen::register_generators! {
    descriptor_schnell => load_schnell,
    descriptor_dev => load_dev,
}

/// Force-link hook. A consumer that only reaches this provider *through* the `gen_core` registry
/// references nothing in this crate directly, so the linker (MSVC on a release build in particular)
/// can discard the whole rlib — taking the `inventory::submit!` registrations with it. Referencing
/// this no-op from the consumer keeps the crate linked. (Same pattern as `candle_gen_sdxl::force_link`.)
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

    /// Both variants register and resolve as candle generators through gen-core's registry (their
    /// `inventory::submit!`s are linked into the test binary). `load` is lazy, so a nonexistent
    /// weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn both_variants_register_and_resolve_as_candle() {
        for (id, family) in [(FLUX1_SCHNELL_ID, "flux"), (FLUX1_DEV_ID, "flux")] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g =
                registry::load(id, &spec).unwrap_or_else(|_| panic!("candle {id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, family);
            assert_eq!(g.descriptor().backend, "candle");
            assert_eq!(g.descriptor().modality, Modality::Image);
        }
    }

    /// schnell advertises no guidance (timestep-distilled); dev advertises guidance. Neither
    /// advertises negative prompt, true-CFG, conditioning, LoRA, or quantization, and neither is
    /// Mac-only.
    #[test]
    fn descriptors_advertise_only_wired_txt2img_surface() {
        let schnell = descriptor_schnell();
        let dev = descriptor_dev();
        assert!(
            !schnell.capabilities.supports_guidance,
            "schnell is distilled"
        );
        assert!(
            dev.capabilities.supports_guidance,
            "dev is guidance-distilled"
        );
        for d in [&schnell, &dev] {
            assert!(!d.capabilities.supports_negative_prompt);
            assert!(!d.capabilities.supports_true_cfg);
            assert!(!d.capabilities.mac_only);
            assert!(d.capabilities.conditioning.is_empty());
            assert!(!d.capabilities.supports_lora);
            assert!(!d.capabilities.supports_lokr);
            assert!(d.capabilities.supported_quants.is_empty());
            assert_eq!(d.capabilities.min_size, 256);
            assert_eq!(d.capabilities.max_size, 2048);
            assert_eq!(d.capabilities.max_count, 8);
            // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123) — the denoise routes
            // through the shared driver, so both variants now advertise the full curated vocabulary.
            assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
            assert_eq!(
                d.capabilities.schedulers,
                candle_gen::curated_scheduler_names()
            );
        }
    }

    /// A txt2img request passes validation; unsupported shapes are rejected clearly. dev accepts a
    /// guidance value (advertised), schnell rejects it (not advertised). Lazy generator → no weights.
    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let schnell = registry::load(FLUX1_SCHNELL_ID, &spec).unwrap();
        let dev = registry::load(FLUX1_DEV_ID, &spec).unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(schnell.validate(&ok).is_ok());
        assert!(dev.validate(&ok).is_ok());

        // dev advertises guidance, so a guidance request is accepted; schnell rejects it.
        let with_guidance = GenerationRequest {
            prompt: "x".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        assert!(dev.validate(&with_guidance).is_ok());
        assert!(
            schnell.validate(&with_guidance).is_err(),
            "schnell advertises no guidance"
        );

        // Shapes rejected on both variants.
        for g in [&schnell, &dev] {
            for bad in [
                GenerationRequest::default(), // empty prompt
                GenerationRequest {
                    prompt: "x".into(),
                    negative_prompt: Some("blurry".into()),
                    ..Default::default()
                },
                GenerationRequest {
                    prompt: "x".into(),
                    width: 1000, // not a multiple of 16
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
        // Neither variant advertises img2img Reference.
        assert!(!descriptor_schnell()
            .capabilities
            .accepts(ConditioningKind::Reference));
        assert!(!descriptor_dev()
            .capabilities
            .accepts(ConditioningKind::Reference));
    }

    /// LoRA adapters / quantization / control overlays are rejected at load as typed `Unsupported`
    /// (both variants), so the worker can fall back to Python rather than the backend silently
    /// dropping them.
    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        for load in [load_schnell, load_dev] {
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

            let control = LoadSpec::new(WeightsSource::Dir("/snap".into()))
                .with_control(WeightsSource::Dir("/ctrl".into()));
            assert!(matches!(
                load(&control).err().expect("err"),
                gen_core::Error::Unsupported(_)
            ));
        }
    }

    #[test]
    fn load_rejects_single_file_source() {
        for load in [load_schnell, load_dev] {
            let spec = LoadSpec::new(WeightsSource::File("/tmp/flux.safetensors".into()));
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(err.contains("snapshot directory"), "got: {err}");
        }
    }
}
