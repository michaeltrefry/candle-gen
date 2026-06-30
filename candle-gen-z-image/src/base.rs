//! `ZImageBaseGenerator` — the **base** (non-distilled, full-CFG) candle Z-Image generator (sc-8414,
//! the candle sibling of `mlx-gen-z-image::model_base`, mlx sc-8320). Registered as its own engine id
//! `z_image`, coexisting in the same crate with the distilled `z_image_turbo` ([`crate`]'s top-level
//! descriptor/load) — a distinct id + a separate `inventory` registration, no clash.
//!
//! The base and Turbo share the **identical** `ZImageTransformer2DModel` architecture (n_layers=30,
//! dim=3840, n_heads=30, cap_feat_dim=2560, qk_norm, rope_theta=256, t_scale=1000), so this generator
//! reuses [`crate::pipeline`]'s components, loader, VAE, and text encoder unchanged — even the DiT
//! config (`Config::z_image_turbo()`) is shared. The deltas (all from the base model card /
//! `scheduler/scheduler_config.json`) are:
//!
//! * **Scheduler shift = 6.0** (Turbo = 3.0) — static, resolution-independent. See
//!   [`crate::pipeline::base_scheduler_config`].
//! * **Default steps = 50** (Turbo = 4) — the base is undistilled.
//! * **Real classifier-free guidance** (Turbo is guidance-distilled → CFG-free). The base supports
//!   full CFG (`guidance` 3.0–5.0, default 4.0) + a negative prompt: each step runs the DiT twice
//!   (cond + uncond) and combines `v = v_uncond + guidance·(v_cond − v_uncond)`. `guidance == 1.0`
//!   collapses to a single cond forward (Turbo-equivalent cost). See
//!   [`crate::pipeline::Pipeline::render_base`].
//!
//! [`load`] assembles the model from a `Tongyi-MAI/Z-Image` snapshot directory (the same diffusers
//! multi-component tree the Turbo loader consumes; the base weights repo is `Tongyi-MAI/Z-Image`, the
//! Turbo's is `Tongyi-MAI/Z-Image-Turbo`). The Turbo path is **completely untouched** — this is
//! additive.

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Progress, WeightsSource,
};

use crate::pipeline::{Components, Pipeline};
use crate::{accel_attn_enabled, SIZE_MULTIPLE};

/// Registry id for the **base** Z-Image (non-Turbo). Matches the SceneWorks catalog `z_image` entry
/// (added by mlx sc-8320) and the macOS `mlx-gen-z-image::model_base` descriptor. Coexists with
/// `z_image_turbo` — a distinct id, a separate `inventory` registration, no clash.
pub const MODEL_ID: &str = "z_image";

/// A loaded candle **base** Z-Image generator. Loading is **lazy** (no file I/O in [`load`]); the heavy
/// components (Qwen3 encoder + DiT + VAE) are built on the first [`generate`](Generator::generate) call
/// and cached (keyed by the accelerated-attention setting), exactly as the Turbo generator. The base
/// reuses the Turbo's [`Pipeline`] + [`Components`] verbatim — only the render path (real CFG, shift
/// 6.0) differs.
pub struct ZImageBaseGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// LoRA/LoKr adapters merged into the DiT weights at component-load (sc-5166). Fixed for this
    /// generator instance; empty ⇒ the stock unadapted build.
    adapters: Vec<AdapterSpec>,
    /// Cached components + the accel-attn flag they were built with. `Mutex` because `Generator` is
    /// shared and `generate` takes `&self`; the lock is held only to read/populate the cache.
    components: Mutex<Option<(bool, Components)>>,
}

impl ZImageBaseGenerator {
    /// Get the cached components, loading (and caching) them on a miss. Keyed by the effective
    /// accel-attn setting (baked into the DiT config at build), identical to the Turbo generator.
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let accel = cfg!(feature = "flash-attn") && accel_attn_enabled();
        let mut guard = self
            .components
            .lock()
            .expect("z-image base components cache mutex poisoned");
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

impl Generator for ZImageBaseGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor: the base advertises guidance + negative prompt, so those are
        // accepted; anything outside the advertised set (e.g. conditioning) is rejected here.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "z_image: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "z_image: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "z_image: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
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
        let pipe = Pipeline::load(&self.root, &self.device, self.dtype, &self.adapters);
        let components = self.components(&pipe)?;
        let images = pipe.render_base(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Base Z-Image's identity + capabilities — constructible without loading weights. Unlike Turbo, the
/// base is a non-distilled foundation model: real CFG (guidance + negative prompt) is supported. Two
/// backend-correct deviations from `mlx-gen-z-image::model_base`: `backend = "candle"` and
/// `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Base is undistilled → full classifier-free guidance + negative prompting (the model
            // card's headline capabilities), unlike the guidance-distilled Turbo.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // txt2img only in this slice (sc-8414): `render_base` does not consume a Reference image,
            // so the descriptor advertises NO conditioning rather than silently dropping one (the
            // false-capability trap the Turbo crate documents). The mlx base provider DOES expose
            // img2img `Reference`; wiring it on the candle base path is a follow-up. An empty list
            // means the shared `validate_request` rejects any conditioning and the worker keeps those
            // shapes on the Python path.
            conditioning: vec![],
            // LoRA/LoKr merge into the dense DiT at load (sc-5166), shared with Turbo.
            supports_lora: true,
            supports_lokr: true,
            // Curated unified-framework integrator menu (epic 7114). An unset `req.sampler` is the
            // curated Euler over the static shift=6.0 schedule; an unset `req.scheduler` is the
            // byte-exact shift=6.0 σ table.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            mac_only: false,
            // On-the-fly Q4/Q8 not wired on the candle base path yet (rejected at load, not dropped).
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Construct the (lazy) candle **base** Z-Image generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `Tongyi-MAI/Z-Image`-layout snapshot (the diffusers
/// multi-component tree: `tokenizer/`, `text_encoder/`, `transformer/`, `vae/`). LoRA/LoKr adapters are
/// accepted and merged into the DiT at first `generate` (sc-5166); on-the-fly quantization and
/// control/IP-adapter overlays are rejected (not wired — refusing is more honest than silently
/// dropping; the worker falls back to Python).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "z_image expects a snapshot directory (tokenizer/ text_encoder/ transformer/ vae/), \
                 not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image does not support control / IP-adapter overlays yet (txt2img only)"
                .into(),
        ));
    }
    // Z-Image is a bf16 model; load at bf16 regardless of the CPU-default dtype.
    let device = candle_gen::default_device()?;
    Ok(Box::new(ZImageBaseGenerator {
        descriptor: descriptor(),
        root,
        device,
        dtype: DType::BF16,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

// Link-time self-registration into gen-core's model registry. A distinct id (`z_image`) → no clash
// with the `z_image_turbo` submission in this same crate (`inventory::submit!` emits anonymous
// statics). Linking this crate makes `gen_core::load("z_image", …)` resolve the candle base generator.
candle_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image};

    /// The seam under test: this provider's base `inventory::submit!` is linked into the test binary,
    /// so resolving `"z_image"` through gen-core's registry returns OUR candle base generator. `load`
    /// is lazy, so a nonexistent weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn base_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("z_image", &spec).expect("candle base z-image is registered");
        assert_eq!(g.descriptor().id, "z_image");
        assert_eq!(g.descriptor().family, "z-image");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    /// The base descriptor advertises the undistilled-CFG surface: guidance, negative prompt, true CFG
    /// — the delta vs the guidance-distilled Turbo. And it is not Mac-only (candle is Windows/CUDA).
    #[test]
    fn base_descriptor_advertises_cfg_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance, "base is undistilled CFG");
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
        assert_eq!(
            d.capabilities.schedulers,
            candle_gen::curated_scheduler_names()
        );
    }

    /// The base differs from Turbo where it must (CFG support, distinct id) and agrees on the shared
    /// envelope (family/backend/modality/size). Turbo must stay guidance-distilled (untouched).
    #[test]
    fn base_differs_from_turbo_only_in_cfg() {
        let base = descriptor();
        let turbo = crate::descriptor();
        assert_eq!(base.family, turbo.family);
        assert_eq!(base.backend, turbo.backend);
        assert_eq!(base.modality, turbo.modality);
        assert_eq!(base.capabilities.min_size, turbo.capabilities.min_size);
        assert_eq!(base.capabilities.max_size, turbo.capabilities.max_size);
        assert_ne!(base.id, turbo.id);
        // Turbo is guidance-distilled (CFG off); base is full-CFG. Turbo untouched by sc-8414.
        assert!(!turbo.capabilities.supports_guidance);
        assert!(base.capabilities.supports_guidance);
        assert!(!turbo.capabilities.supports_negative_prompt);
        assert!(base.capabilities.supports_negative_prompt);
    }

    /// A txt2img request with guidance + a negative prompt passes base validation (the Turbo descriptor
    /// rejects them); unsupported shapes are still rejected clearly. Uses the lazy generator (no GPU).
    #[test]
    fn validate_accepts_cfg_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("z_image", &spec).unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            guidance: Some(4.0),
            negative_prompt: Some("blurry, low quality".into()),
            ..Default::default()
        };
        assert!(
            g.validate(&ok).is_ok(),
            "base accepts guidance + negative prompt"
        );

        for bad in [
            // empty prompt
            GenerationRequest::default(),
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
            // any conditioning — txt2img-only slice advertises none (img2img is a follow-up)
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
        // Sanity: img2img Reference is a kind the candle base slice does not advertise.
        assert!(!descriptor()
            .capabilities
            .accepts(ConditioningKind::Reference));
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::Quant;
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
