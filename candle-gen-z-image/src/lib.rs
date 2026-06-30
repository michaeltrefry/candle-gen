//! # candle-gen-z-image
//!
//! The **Z-Image** (Tongyi `Z-Image-Turbo`) provider crate for [`candle-gen`](candle_gen) — the
//! candle (Windows/CUDA) sibling of `mlx-gen-z-image`. It implements the backend-neutral
//! [`gen_core::Generator`] contract and self-registers via `inventory`, so linking this crate makes
//! `gen_core::load("z_image_turbo", …)` resolve the candle Z-Image generator.
//!
//! **txt2img (sc-3693):** [`ZImageGenerator::generate`] adapts the `candle-transformers` `z_image`
//! reference model ([`pipeline`]) through the contract: Qwen3 text encoder → DiT (flow-match Euler,
//! distilled 4-step, **no CFG**) → AutoencoderKL VAE, emitting `Progress` and honoring `req.cancel`,
//! with **deterministic CPU-seeded noise** (sc-3673) so output is launch-portable per seed. The
//! prompt's Qwen chat-template wrapping reuses gen-core's [`TextTokenizer`] — the same template the
//! mlx provider uses (the epic-3692 "carries over via gen-core" reuse).
//!
//! The descriptor advertises **only** the wired txt2img surface — NOT the full mlx-gen-z-image
//! img2img / LoRA / quantization surface — so the worker routes the rest to the Python fallback
//! rather than the candle backend silently dropping a control (the false-capability trap, exactly as
//! the SDXL slice sc-3675 did). The descriptor's `backend` is `"candle"` and `mac_only` is `false`.
//!
//! Z-Image-Turbo is guidance-distilled: no classifier-free guidance, no negative prompt; the wired
//! sampler is the model's static-shift-3.0 flow-match Euler schedule. See [`pipeline`] for the parity
//! choices reconciled against the macOS `mlx-gen-z-image` provider.

mod adapters;
mod dit;
mod pipeline;
mod training;

// Base (non-Turbo) `z_image` text-to-image generator (sc-8414, the candle sibling of mlx sc-8320).
// Registers its own engine id `z_image` via `inventory` alongside the Turbo `z_image_turbo` below; it
// reuses the identical DiT/VAE/encoder + [`pipeline`] components, differing only in the render path —
// real classifier-free guidance over the static **shift=6.0** flow-match schedule (vs Turbo's
// CFG-free 4-step shift-3.0 distillation). The Turbo path is completely untouched (additive).
pub mod base;

// Fun-ControlNet (strict-pose) provider (sc-5489, epic 5480) — VACE-style dual-injection control on
// the vendored DiT (`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`). Invoked directly by the
// worker (a bespoke pose stream), not gen-core-registered — the `z_image_turbo` descriptor stays
// txt2img-only.
pub mod control;

// Z-Image Fun-ControlNet real-weight GPU validation (sc-5489) — env-driven, `#[ignore]`d integration
// test (with-control vs no-control pixel diff + mid-denoise cancel).
#[cfg(test)]
mod control_validate;

// Z-Image **img2img / edit** (sc-6595, epic 5480) — the candle sibling of the MLX `z_image_turbo`
// `Conditioning::Reference` route. A bespoke provider driven directly by the worker (a `z_image_edit` /
// `z_image_turbo`+`edit_image` stream), like the strict-pose control above; the registered
// `z_image_turbo` descriptor stays txt2img-only (it can't promise img2img through the registry path).
pub mod edit;

// Z-Image img2img real-weight GPU validation (sc-6595) — env-driven, `#[ignore]`d integration test
// (strength ablation + the strength-1.0 source round-trip + mid-denoise cancel).
#[cfg(test)]
mod edit_validate;

pub use adapters::{merge_adapters, MergeReport};
// Base (non-Turbo) `z_image` generator (sc-8414). Its `descriptor`/`load`/`MODEL_ID` share the names
// of the Turbo model's free functions below, so reach them through the `base` module path (consumers
// use the registry id `"z_image"`).
pub use base::ZImageBaseGenerator;
pub use control::{ZImageControl, ZImageControlPaths, ZImageControlRequest, DEFAULT_CONTROL_SCALE};
pub use edit::{ZImageEdit, ZImageEditPaths, ZImageEditRequest, DEFAULT_EDIT_STRENGTH};

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Progress, WeightsSource,
};

use pipeline::{Components, Pipeline};

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TABLE["z_image_turbo"]`)
/// and the macOS `mlx-gen-z-image` descriptor.
pub const MODEL_ID: &str = "z_image_turbo";

/// Z-Image works in latent space at /8 and the DiT patchifies that at /2, so both image dims must be
/// multiples of **16** for a clean patchify. Enforced in [`validate`](Generator::validate).
pub(crate) const SIZE_MULTIPLE: u32 = 16;

/// Process-global accelerated-attention runtime toggle (the Z-Image analogue of the SDXL flash-attn
/// switch, sc-3674). The DiT's fused attention dispatch (CUDA flash-attn / Metal SDPA) is a **build
/// opt-in** (`--features flash-attn`); this switch decides whether a capable build actually *uses*
/// it, so the SceneWorks UI can expose it (defaulted on) and the worker flips it from settings
/// without recompiling. ANDed with `cfg!(feature = "flash-attn")` at load, so on a build without the
/// feature it is inert (the reference's manual attention path always runs). Default **on**.
static ACCEL_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable accelerated attention for subsequently-loaded pipelines. Process-global; the worker
/// calls this from its backend setting at startup. No effect on a build without the `flash-attn`
/// feature (the fused kernels aren't compiled in).
pub fn set_accel_attn(on: bool) {
    ACCEL_ATTN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether accelerated attention is currently enabled (the runtime toggle, [`set_accel_attn`]). The
/// pipeline gates this behind `cfg!(feature = "flash-attn")`, so this returning `true` on a non-flash
/// build does not enable anything.
pub fn accel_attn_enabled() -> bool {
    ACCEL_ATTN.load(std::sync::atomic::Ordering::Relaxed)
}

/// A loaded candle Z-Image generator. Loading is **lazy**: `load` does no file I/O, and the heavy
/// components (Qwen3 encoder + DiT + VAE) are built on the first [`generate`](Generator::generate)
/// call and then **cached** in `components` (keyed by the accelerated-attention setting) so
/// back-to-back requests skip the disk re-read.
pub struct ZImageGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// LoRA/LoKr adapters merged into the DiT weights at component-load (sc-5166). Fixed for this
    /// generator instance; empty ⇒ the stock unadapted build.
    adapters: Vec<AdapterSpec>,
    /// Cached components + the accel-attn flag they were built with. `Mutex` because `Generator` is
    /// shared and `generate` takes `&self`; the lock is held only to read/populate the cache, never
    /// across the denoise.
    components: Mutex<Option<(bool, Components)>>,
}

impl ZImageGenerator {
    /// Get the cached components, loading (and caching) them on a miss. Keyed by the effective
    /// accel-attn setting (baked into the DiT config at build), so flipping [`set_accel_attn`] between
    /// calls rebuilds rather than serving a stale DiT.
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let accel = cfg!(feature = "flash-attn") && accel_attn_enabled();
        let mut guard = self
            .components
            .lock()
            .expect("z-image components cache mutex poisoned");
        if let Some((cached_accel, comps)) = guard.as_ref() {
            if *cached_accel == accel {
                return Ok(comps.clone());
            }
        }
        let comps = pipe.load_components(accel)?;
        *guard = Some((accel, comps.clone()));
        Ok(comps)
    }
}

impl Generator for ZImageGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor: since the descriptor advertises no conditioning, no guidance,
        // and no negative prompt, any of those is rejected here (distilled-model honesty).
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        // Model-specific floor on top (mirrors mlx-gen-z-image::validate_request).
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "z_image_turbo: prompt must not be empty".into(),
            ));
        }
        // An explicit `steps: Some(0)` would VAE-decode pure noise — reject loudly (txt2img-only).
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "z_image_turbo: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "z_image_turbo: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
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
        let pipe = Pipeline::load(&self.root, &self.device, self.dtype, &self.adapters);
        let components = self.components(&pipe)?;
        let images = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Z-Image-Turbo's identity + the wired surface: distilled txt2img (no CFG, no negative prompt) plus
/// LoRA/LoKr adapter merge (sc-5166). img2img conditioning + Q4/Q8 quantization stay the Python
/// fallback's job until candle wires them, so the descriptor never promises a path `generate` can't
/// serve. Two backend-correct deviations from `mlx-gen-z-image`: `backend = "candle"` and
/// `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        // The tensor backend whose provider crate registered this engine (sc-3723). MLX sets "mlx".
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Turbo is guidance-distilled: no CFG, no negative prompt.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // txt2img only in sc-3693 — img2img (the mlx provider's `Reference`) lands later; an
            // empty list means the shared `validate_request` rejects any conditioning and the worker
            // keeps those shapes on the Python path.
            conditioning: vec![],
            // LoRA/LoKr now wired (sc-5166): a trained adapter merges into the dense DiT weights at
            // load ([`crate::adapters::merge_adapters`]), closing the candle train→infer loop. Q4/Q8
            // quantization is still deferred (rejected at load, not silently dropped).
            supports_lora: true,
            supports_lokr: true,
            // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123). Z-Image-Turbo is
            // guidance-distilled (4 steps, `euler` recommended), but the curated integrators +
            // σ-schedules are exposed for ComfyUI parity; the default (`euler` over the native linear
            // flow-match schedule) is the byte-faithful N1 no-op.
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

/// Construct the (lazy) candle Z-Image generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `Tongyi-MAI/Z-Image-Turbo`-layout snapshot (the diffusers
/// multi-component tree: `tokenizer/`, `text_encoder/`, `transformer/`, `vae/`). LoRA/LoKr adapters
/// are accepted and merged into the DiT at first `generate` (sc-5166); on-the-fly quantization and
/// control/IP-adapter overlays are still rejected — not wired, so refusing is more honest than
/// silently dropping them (the worker falls back to Python).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "z_image_turbo expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image_turbo does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image_turbo does not support control / IP-adapter overlays yet (txt2img only)"
                .into(),
        ));
    }
    // Z-Image is a bf16 model; load at bf16 regardless of the CPU-default dtype. The device is the
    // backend selected at compile time (CUDA on Windows, Metal/CPU on Mac).
    let device = candle_gen::default_device()?;
    Ok(Box::new(ZImageGenerator {
        descriptor: descriptor(),
        root,
        device,
        dtype: DType::BF16,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

// Link-time self-registration into gen-core's model registry. Linking this crate makes
// `gen_core::load("z_image_turbo", …)` resolve the candle generator — no central match to edit.
candle_gen::register_generators! { descriptor => load }

/// Force-link hook. A consumer that only reaches this provider *through* the `gen_core` registry
/// references nothing in this crate directly, so the linker (MSVC on a release build in particular)
/// can discard the whole rlib — taking the `inventory::submit!` registration with it. Referencing
/// this no-op from the consumer keeps the crate linked. (Same pattern as `candle_gen_sdxl::force_link`.)
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

    /// The seam under test: this provider's `inventory::submit!` is linked into the test binary, so
    /// resolving `"z_image_turbo"` through gen-core's registry returns OUR candle generator. `load`
    /// is lazy, so a nonexistent weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn z_image_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("z_image_turbo", &spec).expect("candle z-image is registered");
        assert_eq!(g.descriptor().id, "z_image_turbo");
        assert_eq!(g.descriptor().family, "z-image");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    /// The descriptor advertises only the wired distilled-txt2img surface: no CFG, no negative
    /// prompt, no conditioning, no LoRA — and is not Mac-only.
    #[test]
    fn descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor();
        assert!(
            !d.capabilities.supports_guidance,
            "turbo is guidance-distilled"
        );
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        // LoRA/LoKr wired (sc-5166) — merged into the DiT at load.
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert!(d.capabilities.supported_quants.is_empty());
        assert_eq!(d.capabilities.min_size, 256);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(d.capabilities.max_count, 8);
        // Curated sampler/scheduler menu (epic 7114 P4, sc-7123): full vocabulary, euler the default.
        assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
        assert_eq!(
            d.capabilities.schedulers,
            candle_gen::curated_scheduler_names()
        );
    }

    /// A txt2img request passes validation; unsupported shapes are rejected clearly (not silently
    /// served). Uses the lazy generator so no weights are needed.
    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("z_image_turbo", &spec).unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        for bad in [
            // empty prompt
            GenerationRequest::default(),
            // guidance on a distilled model (descriptor advertises no guidance)
            GenerationRequest {
                prompt: "x".into(),
                guidance: Some(5.0),
                ..Default::default()
            },
            // negative prompt (not supported)
            GenerationRequest {
                prompt: "x".into(),
                negative_prompt: Some("blurry".into()),
                ..Default::default()
            },
            // non-multiple-of-16 size
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                ..Default::default()
            },
            // explicit 0 steps
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
            // any conditioning (none advertised)
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
        // Sanity: img2img Reference is a kind the candle slice does not advertise.
        assert!(!descriptor()
            .capabilities
            .accepts(ConditioningKind::Reference));
    }

    /// Quantization / control overlays are rejected at load as typed `Unsupported`, so the worker
    /// can fall back to Python rather than the backend silently dropping them. LoRA/LoKr are now
    /// wired (sc-5166), so a LoRA `LoadSpec` is **accepted** (lazily — the merge happens at generate).
    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load(&lora).is_ok(), "LoRA load is wired + lazy (sc-5166)");

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

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    /// The accel-attn runtime toggle defaults on and round-trips (what the worker/UI drive).
    #[test]
    fn accel_attn_toggle_roundtrips() {
        assert!(
            accel_attn_enabled(),
            "accel-attn runtime toggle defaults on"
        );
        set_accel_attn(false);
        assert!(!accel_attn_enabled());
        set_accel_attn(true);
        assert!(accel_attn_enabled());
    }
}
