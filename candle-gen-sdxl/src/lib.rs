//! # candle-gen-sdxl
//!
//! The **Stable Diffusion XL** provider crate for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-sdxl`. It implements the backend-neutral
//! [`gen_core::Generator`] contract and self-registers via `inventory`, so linking this crate
//! makes `gen_core::load("sdxl", …)` resolve the candle SDXL generator.
//!
//! **txt2img (sc-3675 + sc-3673):** [`SdxlGenerator::generate`] runs the GO-validated epic-3494
//! prototype ([`pipeline`]) through the contract: dual CLIP → UNet (real CFG) → f16 VAE, emitting
//! `Progress` and honoring `req.cancel`, with **deterministic CPU-seeded noise + the non-ancestral
//! DDIM sampler** (sc-3673) so output is launch-portable per seed. The descriptor advertises **only**
//! the wired surface (txt2img + negative prompt + guidance, `ddim`) — NOT the full mlx-gen-sdxl
//! conditioning/LoRA/accel-sampler surface — so the worker can route the rest to the Python fallback
//! (sc-3678) rather than the candle backend silently dropping a control. The descriptor's `backend`
//! is `"candle"` and `mac_only` is `false` (Windows/CUDA target).
//!
//! Perf (sc-3674): CLIP loads f16 and the UNet attention runs through fused **flash-attention** when
//! built `--features flash-attn` and the runtime toggle ([`set_flash_attn`], default on) is set.
//!
//! Peak VRAM (sc-4987): the dual CLIP is loaded/run/freed before the UNet+VAE load (staged
//! sequential load), and the VAE decode tiles + blends above 512² output ([`set_vae_tiling`], default
//! on) — together targeting torch-parity peak VRAM at 1024².
//!
//! Carried forward as named follow-ups: component caching across `generate` calls (a latency win, in
//! tension with sc-4987's mid-call frees), RealVisXL + parity (sc-3677).

mod pipeline;

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};

use pipeline::Pipeline;

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TABLE["sdxl"]`). The
/// worker maps both `sdxl` and `realvisxl` onto engine id `"sdxl"`, so — exactly like
/// `mlx-gen-sdxl` — this crate registers a SINGLE descriptor under `"sdxl"`.
pub const MODEL_ID: &str = "sdxl";

/// SDXL works in latent space at /8: both dims must be multiples of 8.
const SIZE_MULTIPLE: u32 = 8;

/// Process-global flash-attention runtime toggle (sc-3674). The **fused CUTLASS kernels are a build
/// opt-in** (`--features flash-attn`); this switch decides whether a flash-attn-capable build
/// actually *uses* them, so the SceneWorks UI can expose it (defaulted on) and the worker flips it
/// from settings — without recompiling. Mirrors `mlx-gen-sdxl::set_compile_glue`. Read at pipeline
/// load via [`flash_attn_enabled`]; the pipeline ANDs it with `cfg!(feature = "flash-attn")`, so on a
/// build without the feature this is inert (the unfused path always runs). Default **on**.
static FLASH_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable flash-attention for subsequently-loaded pipelines (sc-3674). Process-global; the
/// worker calls this from its `backend_candle`/flash setting at startup. No effect on a build without
/// the `flash-attn` feature (the kernels aren't compiled in).
pub fn set_flash_attn(on: bool) {
    FLASH_ATTN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether flash-attention is currently enabled (the runtime toggle, [`set_flash_attn`]). The
/// pipeline still gates this behind `cfg!(feature = "flash-attn")`, so this returning `true` on a
/// non-flash build does not enable anything.
pub fn flash_attn_enabled() -> bool {
    FLASH_ATTN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Process-global VAE-tiling runtime toggle (sc-4987). When on, the VAE decode tiles the latent into
/// overlapping 64²-latent (512²-output) tiles and trapezoidally blends the seams — bounding the
/// decode's peak VRAM to one tile (the tallest single allocation at 1024², for torch-parity). Unlike
/// flash-attn there is no build feature: it is pure candle, so the switch alone decides. It only
/// *fires* above 512² output (smaller renders stay monolithic), so leaving it on is free at/below
/// 512². The SceneWorks worker/UI drives it; default **on** to hit the <12 GiB target out of the box.
static VAE_TILING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable VAE tiling for subsequent decodes (sc-4987). Process-global; the worker drives it
/// from its backend setting. Off restores the monolithic single-pass decode (higher peak VRAM).
pub fn set_vae_tiling(on: bool) {
    VAE_TILING.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether VAE tiling is currently enabled (the runtime toggle, [`set_vae_tiling`]). The pipeline
/// additionally only tiles when the output exceeds the 512² threshold, so this returning `true` does
/// not change ≤512² output.
pub fn vae_tiling_enabled() -> bool {
    VAE_TILING.load(std::sync::atomic::Ordering::Relaxed)
}

/// A loaded candle SDXL generator. Loading is **lazy**: this carries only the resolved snapshot
/// `root` + the compute device/dtype (so `load` does no file I/O and registry introspection against a
/// missing path still resolves). The heavy components ([`Pipeline`]) are built at the top of
/// [`generate`](Generator::generate); caching them across calls is sc-3674.
pub struct SdxlGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
}

impl Generator for SdxlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor (count/size range/guidance/negative/sampler/conditioning):
        // since the descriptor advertises NO conditioning, any conditioning entry is rejected here.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        // Model-specific floor on top of the shared one (mirrors mlx-gen-sdxl::validate_request).
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "sdxl: prompt must not be empty".into(),
            ));
        }
        // An explicit `steps: Some(0)` would VAE-decode pure scaled noise — reject loudly (a derived
        // 0 from img2img strength would be a legitimate no-op, but this is txt2img-only).
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "sdxl: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "sdxl: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
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
        // Build the pipeline (once per call) and run it. The rich-`CandleError` tail — including the
        // typed `Canceled` — bridges into `gen_core::Error` via `?`/`map_err` (the From bridge).
        let pipe = Pipeline::load(&self.root, &self.device, self.dtype, req.width, req.height)?;
        let images = pipe.generate(req, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// SDXL's identity + the surface sc-3675 actually wires: real classifier-free guidance (negative
/// prompt + CFG scale), txt2img only, `euler_ancestral`. No conditioning / LoRA / acceleration
/// samplers are advertised — those are the Python fallback's job (sc-3678) until candle wires them —
/// so the descriptor never promises a path `generate` can't serve (the false-capability trap). Two
/// backend-correct deviations from `mlx-gen-sdxl`: `backend = "candle"` and `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        // The tensor backend whose provider crate registered this engine (sc-3723). MLX sets "mlx".
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // txt2img only in sc-3675 — img2img/inpaint/control land later; advertising none means
            // the shared `validate_request` rejects any conditioning, and the worker keeps those
            // shapes on the Python path (sc-3678).
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            // DDIM (eta=0) — the deterministic, launch-portable sampler wired in sc-3673 (replacing
            // the spike's Euler-ancestral). The few-step accel samplers need their acceleration LoRAs
            // (not yet supported), so they are not advertised. The worker sends no `sampler` for SDXL,
            // so this list is capability introspection (`validate` only rejects a *named* sampler not
            // in it).
            samplers: vec!["ddim"],
            schedulers: vec!["discrete"],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            mac_only: false,
            // No on-the-fly quantization wired yet (sc-3674 territory).
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Construct the (lazy) candle SDXL generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `stabilityai/stable-diffusion-xl-base-1.0`-layout snapshot
/// (the diffusers multi-component tree: `text_encoder/`, `text_encoder_2/`, `unet/`, …). LoRA
/// adapters are rejected — candle SDXL LoRA is not wired (it would otherwise be silently dropped).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "sdxl expects a snapshot directory (text_encoder/ text_encoder_2/ unet/ …), not a \
                 single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle sdxl does not support LoRA/LoKr yet — refusing to silently drop the adapters"
                .into(),
        ));
    }
    // SDXL is fp16 (the production reference dtype) regardless of the CPU-default dtype; the device
    // is the backend selected at compile time (CUDA on Windows, Metal/CPU on Mac).
    let device = candle_gen::default_device()?;
    Ok(Box::new(SdxlGenerator {
        descriptor: descriptor(),
        root,
        device,
        dtype: DType::F16,
    }))
}

// Link-time self-registration into gen-core's model registry. Linking this crate makes
// `gen_core::load("sdxl", …)` resolve the candle generator — no central match statement to edit.
inventory::submit! {
    ModelRegistration { descriptor, load }
}

/// Force-link hook. A consumer that only reaches this provider *through* the `gen_core` registry
/// references nothing in this crate directly, so the linker (MSVC in particular, on a release
/// build) discards the whole rlib — taking the `inventory::submit!` registration above with it, and
/// `gen_core::load("sdxl", …)` then fails with "no generator registered". Referencing this no-op
/// from the consumer keeps the crate linked so the registration survives. The SceneWorks worker
/// force-links each provider crate for exactly this reason (e.g. `sensenova_jobs`).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

    /// The seam under test: this provider's `inventory::submit!` is linked into the test binary,
    /// so resolving `"sdxl"` through gen-core's registry returns OUR candle generator. `load` is
    /// lazy, so a nonexistent weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn sdxl_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("sdxl", &spec).expect("candle sdxl is registered");
        assert_eq!(g.descriptor().id, "sdxl");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        // sc-3675 is txt2img-only: no conditioning / LoRA / accel samplers advertised.
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.supports_lora);
        assert!(!d.capabilities.supports_lokr);
        // sc-3673: the wired sampler is the deterministic DDIM (not the spike's euler-ancestral).
        assert_eq!(d.capabilities.samplers, vec!["ddim"]);
    }

    /// sc-3674: the flash-attn runtime toggle defaults on and round-trips (what the worker/UI drive).
    #[test]
    fn flash_attn_toggle_roundtrips() {
        assert!(
            flash_attn_enabled(),
            "flash-attn runtime toggle defaults on"
        );
        set_flash_attn(false);
        assert!(!flash_attn_enabled());
        set_flash_attn(true);
        assert!(flash_attn_enabled());
    }

    /// sc-4987: the VAE-tiling runtime toggle defaults on (to hit the <12 GiB target out of the box)
    /// and round-trips — what the worker/UI drive.
    #[test]
    fn vae_tiling_toggle_roundtrips() {
        assert!(
            vae_tiling_enabled(),
            "vae-tiling runtime toggle defaults on"
        );
        set_vae_tiling(false);
        assert!(!vae_tiling_enabled());
        set_vae_tiling(true);
        assert!(vae_tiling_enabled());
    }

    /// A txt2img request passes validation; unsupported shapes are rejected clearly (not silently
    /// served). Uses the lazy generator so no weights are needed.
    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load("sdxl", &spec).unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            guidance: Some(7.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        // Empty prompt, non-multiple-of-8 size, explicit 0 steps, and any conditioning are rejected.
        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 1020,
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
        // Sanity: the rejected conditioning above is a kind the descriptor does not advertise.
        assert!(!descriptor()
            .capabilities
            .accepts(ConditioningKind::Reference));
    }

    /// LoRA adapters are rejected at load (candle SDXL LoRA is not wired) — as a typed `Unsupported`,
    /// so the worker can fall back to Python rather than the backend silently dropping the adapter.
    #[test]
    fn load_rejects_lora_adapters() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        let err = load(&spec).err().expect("expected an error");
        assert!(matches!(err, gen_core::Error::Unsupported(_)));
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sdxl.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
