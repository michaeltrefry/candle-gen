//! # candle-gen-seedvr2
//!
//! The **SeedVR2** provider crate for [`candle-gen`](candle_gen) — the Windows/CUDA sibling of
//! `mlx-gen-seedvr2` (epic 4811 / sc-5157). A native-candle port of the ByteDance one-step
//! diffusion-transformer super-resolution upscaler:
//!
//! 1. **DiT** ([`dit`]) — a dual-stream MMDiT with adaptive **spatiotemporal window attention**
//!    (`window=(T,H,W)=(4,3,3)`, shifted on odd layers), 3D axial RoPE, QK-norm, SwiGLU, AdaLN.
//! 2. **3D causal video VAE** ([`vae`]) — `CausalConv3d` (candle has no conv3d → conv2d temporal-sum,
//!    see [`conv3d`]) encoder/decoder with `temporal_down/up_blocks=2`, GroupNorm, per-frame attention.
//! 3. **One-step Euler** + a precomputed negative-prompt embedding (bundled, no runtime text encoder).
//!
//! **Surface (`Modality::Both`): a one-step super-resolution upscaler over image AND video**,
//! dispatched on the request's conditioning — a `Reference` LR image → `GenerationOutput::Images`
//! (sc-5157, the 3B engine), or a `VideoClip` LR frame sequence → `GenerationOutput::Video` (sc-5926:
//! the 5-D temporal pass with chunking/overlap cross-fade, a VRAM-budgeted chunk sizer, and HD spatial
//! tiling; see [`video`] + [`pipeline`]). 7B + int8/int4 quant is sc-5927; worker wiring/gating is
//! sc-5928. `backend = "candle"`, `mac_only = false`.

pub mod color;
pub mod config;
pub mod conv3d;
pub mod convert;
pub mod dit;
pub mod nn;
pub mod pipeline;
pub mod quant;
pub mod vae;
pub mod video;
pub mod weights;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, default_seed, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress,
    Quant, WeightsSource,
};

use config::{DitConfig, VAE_SCALE};
use pipeline::Seedvr2Pipeline;

pub const MODEL_ID: &str = "seedvr2";
pub const MODEL_ID_3B: &str = "seedvr2_3b";
pub const MODEL_ID_7B: &str = "seedvr2_7b";
const DIT_FILE_3B: &str = "seedvr2_ema_3b_fp16.safetensors";
const DIT_FILE_7B: &str = "seedvr2_ema_7b_fp16.safetensors";
/// Output fps when the request omits one (the worker normally supplies the source cadence).
const DEFAULT_FPS: u32 = 24;

/// The DiT checkpoint file + transformer config for a registered id (3B default; 7B is the
/// pixel-mode-RoPE variant — sc-5927). The VAE is shared across both.
fn variant(id: &str) -> (&'static str, DitConfig) {
    if id == MODEL_ID_7B {
        (DIT_FILE_7B, DitConfig::seedvr2_7b())
    } else {
        (DIT_FILE_3B, DitConfig::seedvr2_3b())
    }
}

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "seedvr2",
        backend: "candle",
        modality: Modality::Both, // image (Reference) + video (VideoClip) upscaling
        capabilities: Capabilities {
            supports_negative_prompt: false, // precomputed neg-embed; no prompt surface
            supports_guidance: false,        // one-step, guidance fixed at 1.0
            supports_true_cfg: false,
            // the LR input image (image upscale) or LR frame sequence (video upscale)
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::VideoClip],
            supports_lora: false,
            supports_lokr: false,
            // Bespoke-by-architecture (epic 7114 P4, sc-7123): SeedVR2 is a ONE-STEP restoration
            // transformer (fixed timestep 1000.0, `denoised = noise − dit_out`), NOT a multi-step
            // rectified-flow ODE — there is no sigma schedule to integrate and no sampler/scheduler axis
            // to expose. It keeps its native single-step `seedvr2_euler` descriptor verbatim; the unified
            // curated solvers/schedulers do not apply.
            samplers: vec!["seedvr2_euler"],
            schedulers: vec!["seedvr2_euler"],
            supported_guidance_methods: vec![],
            min_size: VAE_SCALE,
            max_size: 4096,
            max_count: 8,
            mac_only: false,
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supported_quants: &[Quant::Q4, Quant::Q8], // Linear-only DiT quant (sc-5927)
        },
    }
}

pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}
pub fn descriptor_3b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_3B)
}
pub fn descriptor_7b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_7B)
}

/// The lazy candle SeedVR2 generator (one-step image/video upscaler). The pipeline is loaded on first
/// `generate` and cached behind a `Mutex` for the worker's `Arc<dyn Generator>` reuse. `dit_file`/`cfg`
/// select the 3B (default) or 7B variant; `quant` applies int8/int4 DiT quantization at load (sc-5927).
pub struct Seedvr2Generator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    dit_file: &'static str,
    cfg: DitConfig,
    quant: Option<Quant>,
    pipe: Mutex<Option<std::sync::Arc<Seedvr2Pipeline>>>,
}

/// The LR input image carried by the request's `Reference` conditioning.
fn reference_image(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

impl Seedvr2Generator {
    fn pipeline(&self) -> gen_core::Result<std::sync::Arc<Seedvr2Pipeline>> {
        let mut guard = self.pipe.lock().expect("seedvr2 pipeline mutex poisoned");
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let mut p = Seedvr2Pipeline::load(
            &self.root,
            self.dit_file,
            &self.cfg,
            self.dtype,
            &self.device,
        )?;
        // sc-5927: int8/int4 quantize the DiT Linears at load (the VAE stays dense).
        if let Some(q) = self.quant {
            p.quantize(q)?;
        }
        let p = std::sync::Arc::new(p);
        *guard = Some(p.clone());
        Ok(p)
    }
}

impl Generator for Seedvr2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        let has_video = req.video_clips().iter().any(|c| !c.frames.is_empty());
        if !has_video && reference_image(req).is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{}: requires a Reference image (image upscale) or a non-empty VideoClip frame \
                 sequence (video upscale)",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(VAE_SCALE) || !req.height.is_multiple_of(VAE_SCALE) {
            return Err(gen_core::Error::Msg(format!(
                "{}: width/height must be multiples of {VAE_SCALE} (got {}x{})",
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
        let pipe = self.pipeline()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let softness = req.softness.unwrap_or(0.0);

        // Video upscale: a VideoClip carries the LR source frame sequence → one upscaled clip
        // (temporal chunking + overlap cross-fade + VRAM-budgeted chunk sizer + HD tiling).
        if let Some(clip) = req.video_clips().into_iter().find(|c| !c.frames.is_empty()) {
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            on_progress(Progress::Step {
                current: 1,
                total: 1,
            });
            let frames = pipe.generate_video(
                clip.frames,
                req.width as usize,
                req.height as usize,
                base_seed,
                softness,
                None,
            )?;
            on_progress(Progress::Decoding);
            return Ok(GenerationOutput::Video {
                frames,
                fps: req.fps.unwrap_or(DEFAULT_FPS),
                audio: None,
            });
        }

        let image = reference_image(req).expect("validated");
        let mut out = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            on_progress(Progress::Step {
                current: 1,
                total: 1,
            });
            let seed = base_seed.wrapping_add(i as u64);
            let img = pipe.generate(
                image,
                req.width as usize,
                req.height as usize,
                seed,
                softness,
            )?;
            on_progress(Progress::Decoding);
            out.push(img);
        }
        Ok(GenerationOutput::Images(out))
    }
}

/// Construct a lazy candle SeedVR2 generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a raw `numz/SeedVR2_comfyUI` checkpoint dir (`ema_vae_fp16.safetensors` + the 3B/7B DiT file).
/// `spec.quantize` (Q4/Q8) int8/int4-quantizes the DiT Linears at load (sc-5927). Adapters / control
/// overlays are rejected.
fn load_with(spec: &LoadSpec, id: &'static str) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id}: expects a numz/SeedVR2_comfyUI checkpoint directory, not a single file"
            )))
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: LoRA/LoKr adapters are not part of SeedVR2"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: ControlNet / IP-Adapter conditioning is not part of SeedVR2"
        )));
    }
    let device = candle_gen::default_device()?;
    // Precision::Bf16 is the "native dense" sentinel → bf16 on GPU, f32 on CPU (the parity dtype);
    // Fp32 forces full precision everywhere.
    let dtype = match spec.precision {
        Precision::Fp32 => DType::F32,
        Precision::Bf16 if device.is_cpu() => DType::F32,
        Precision::Bf16 => DType::BF16,
    };
    let (dit_file, cfg) = variant(id);
    Ok(Box::new(Seedvr2Generator {
        descriptor: descriptor_for(id),
        root,
        device,
        dtype,
        dit_file,
        cfg,
        quant: spec.quantize,
        pipe: Mutex::new(None),
    }))
}

fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID)
}
fn load_registered_3b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_3B)
}
fn load_registered_7b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_7B)
}

candle_gen::register_generators! {
    descriptor => load_registered,
    descriptor_3b => load_registered_3b,
    descriptor_7b => load_registered_7b,
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn descriptor_is_seedvr2_image_and_video() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "seedvr2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Both); // image (Reference) + video (VideoClip)
        assert!(!d.capabilities.mac_only);
        assert!(!d.capabilities.supports_guidance);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::VideoClip));
        assert_eq!(d.capabilities.min_size, VAE_SCALE);
        // sc-5927: int8/int4 DiT quant is now advertised.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
    }

    #[test]
    fn all_ids_resolve_in_registry() {
        for id in [MODEL_ID, MODEL_ID_3B, MODEL_ID_7B] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/seedvr2".into()));
            let g = registry::load(id, &spec).expect("seedvr2 is registered");
            assert_eq!(g.descriptor().family, "seedvr2");
            assert_eq!(g.descriptor().backend, "candle");
        }
    }

    #[test]
    fn variant_selects_7b_config_and_file() {
        let (file_3b, cfg_3b) = variant(MODEL_ID_3B);
        assert_eq!(file_3b, DIT_FILE_3B);
        assert_eq!(cfg_3b.num_layers, 32);
        assert!(!cfg_3b.rope_pixel);

        let (file_7b, cfg_7b) = variant(MODEL_ID_7B);
        assert_eq!(file_7b, DIT_FILE_7B);
        assert_eq!(cfg_7b.num_layers, 36);
        assert_eq!(cfg_7b.vid_dim, 3072);
        assert!(cfg_7b.rope_pixel); // pixel-mode RoPE — the 7B delta (sc-5197)
        assert!(!cfg_7b.rope_on_text);
        assert!(!cfg_7b.swiglu_mlp); // GELU MLP
    }

    #[test]
    fn load_accepts_quant_rejects_single_file() {
        use candle_gen::gen_core::Quant;
        // Quant is now wired (sc-5927) — load succeeds lazily and carries the level.
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(q);
            let g = load_with(&spec, MODEL_ID).expect("quant is wired");
            assert_eq!(g.descriptor().family, "seedvr2");
        }
        // A single-file weights source is still rejected.
        let file = LoadSpec::new(WeightsSource::File("/w.safetensors".into()));
        assert!(load_with(&file, MODEL_ID).is_err());
    }
}
