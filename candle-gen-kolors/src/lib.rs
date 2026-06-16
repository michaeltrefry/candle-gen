//! # candle-gen-kolors
//!
//! The **Kolors** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling
//! of `mlx-gen-kolors`. It implements the backend-neutral [`gen_core::Generator`] contract and
//! self-registers via `inventory`, so linking this crate makes `gen_core::load("kolors", …)` resolve
//! the candle Kolors generator.
//!
//! **txt2img:** Kolors is a bilingual (Chinese/English) SDXL-family T2I model — the SDXL UNet + SDXL
//! VAE with a **ChatGLM3-6B** text encoder in place of dual CLIP. [`pipeline`] runs it through the
//! contract: ChatGLM3 encode (penultimate hidden state → cross-attention context, last-token
//! last-layer state → pooled add-embedding) → the Kolors UNet (real CFG over the leading-Euler
//! 1100-step schedule) → the SDXL VAE, emitting `Progress`, honoring `req.cancel`, with
//! **deterministic CPU-seeded noise** (sc-3673) so output is launch-portable per seed.
//!
//! The descriptor advertises **only** the wired txt2img surface (negative prompt + CFG guidance, but
//! NOT LoRA/LoKr, quantization, ControlNet-pose, or IP-Adapter — all wired in the mlx provider) — so
//! the worker routes the rest to the Python fallback rather than the candle backend silently dropping
//! a control (the false-capability trap, exactly as the SDXL / FLUX / Z-Image / Chroma slices did).
//! `backend` is `"candle"` and `mac_only` is `false`.

mod chatglm3;
mod config;
mod pipeline;
mod sampler;
mod tokenizer;
mod unet;

// IP-Adapter-Plus reference-image (identity) provider (sc-5488, epic 5480) — CLIP ViT-L/14-336 image
// tokens injected into the vendored SDXL `UNet2DConditionModel` (candle-gen-sdxl) alongside the
// encoder_hid_proj-projected ChatGLM3 text path, denoised with the Kolors leading-Euler sampler.
// Invoked directly by the worker (a bespoke reference stream), not gen-core-registered.
pub mod ip_provider;

// ControlNet (strict-pose) provider (sc-5489, epic 5480) — a rendered OpenPose skeleton drives the
// `Kwai-Kolors/Kolors-ControlNet-Pose` SDXL-family `ControlNetModel`, whose per-block residuals are
// added into the vendored SDXL UNet (no IP installed). Invoked directly by the worker (a bespoke pose
// stream), not gen-core-registered.
pub mod control;

// Kolors IP-Adapter-Plus real-weight GPU validation (sc-5488) — env-driven, `#[ignore]`d integration
// test (the Kolors sibling of the SDXL IP-Adapter Phase-5 harness).
#[cfg(test)]
mod ip_validate;

// Kolors ControlNet (strict-pose) real-weight GPU validation (sc-5489) — env-driven, `#[ignore]`d
// integration test (with-control vs no-control pixel diff + mid-denoise cancel).
#[cfg(test)]
mod control_validate;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor, Progress,
    WeightsSource,
};

pub use config::{descriptor, MODEL_ID, SIZE_MULTIPLE};
pub use control::{KolorsControl, KolorsControlPaths, KolorsControlRequest, DEFAULT_CONTROL_SCALE};
pub use ip_provider::{
    IpAdapterKolors, IpAdapterKolorsPaths, IpAdapterKolorsRequest, DEFAULT_IP_ADAPTER_SCALE,
};
use sampler::NUM_TRAIN_TIMESTEPS;

use pipeline::{Components, Pipeline};

/// A loaded candle Kolors generator. Loading is **lazy**: `load` does no file I/O (registry
/// introspection against a missing path still resolves), and the heavy components (ChatGLM3 + UNet +
/// VAE) are built on the first [`generate`](Generator::generate) call and then cached.
pub struct KolorsGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

impl KolorsGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("kolors components cache mutex poisoned");
        if let Some(comps) = guard.as_ref() {
            return Ok(comps.clone());
        }
        let comps = pipe.load_components()?;
        *guard = Some(comps.clone());
        Ok(comps)
    }
}

impl Generator for KolorsGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor (count/size range, negative_prompt + guidance; since the
        // descriptor advertises NO conditioning, any conditioning entry is rejected here).
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(
                "kolors: prompt must not be empty".into(),
            ));
        }
        // `steps == 0` would VAE-decode undenoised noise; `steps > NUM_TRAIN_TIMESTEPS` collapses the
        // leading schedule (every timestep maps to one value). Reject both (the sampler errors too).
        if let Some(steps) = req.steps {
            if steps == 0 || steps as usize > NUM_TRAIN_TIMESTEPS {
                return Err(gen_core::Error::Msg(format!(
                    "kolors: steps must be in 1..={NUM_TRAIN_TIMESTEPS} (got {steps})"
                )));
            }
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "kolors: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
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
        let pipe = Pipeline::load(&self.root, &self.device);
        let components = self.components(&pipe)?;
        let images = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Construct the (lazy) candle Kolors generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `Kwai-Kolors/Kolors-diffusers` snapshot (`text_encoder/`,
/// `tokenizer/`, `unet/`, `vae/`, with `tokenizer/tokenizer.json` materialized). LoRA adapters,
/// quantization, and control / IP-adapter overlays are rejected — none are wired in this slice, so
/// refusing is more honest than silently dropping them.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "kolors expects a Kolors-diffusers snapshot directory (text_encoder/ tokenizer/ \
                 unet/ vae/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle kolors does not support LoRA/LoKr yet — refusing to silently drop the adapters"
                .into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle kolors does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle kolors does not support control / IP-adapter overlays yet (txt2img only)"
                .into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(KolorsGenerator {
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

// Link-time self-registration into gen-core's model registry. Linking this crate makes
// `gen_core::load("kolors", …)` resolve the candle generator — no central match statement to edit.
inventory::submit! {
    ModelRegistration { descriptor, load }
}

/// Force-link hook. A consumer that only reaches this provider *through* the `gen_core` registry
/// references nothing in this crate directly, so the linker (MSVC on a release build in particular)
/// can discard the whole rlib — taking the `inventory::submit!` registration with it. Referencing this
/// no-op from the consumer keeps the crate linked. (Same pattern as `candle_gen_chroma::force_link`.)
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, Conditioning, Image, Modality, Quant};

    #[test]
    fn kolors_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("candle kolors is registered");
        assert_eq!(g.descriptor().id, "kolors");
        assert_eq!(g.descriptor().family, "kolors");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();

        let ok = GenerationRequest {
            prompt: "一只猫 / a cat holding a lit candle".into(),
            guidance: Some(5.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 1020, // not a multiple of 8
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(NUM_TRAIN_TIMESTEPS as u32 + 1),
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
    fn load_rejects_unwired_surfaces_and_single_file() {
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

        let single = LoadSpec::new(WeightsSource::File("/x.safetensors".into()));
        let err = load(&single).err().expect("err").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
