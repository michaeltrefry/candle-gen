//! SCAIL-2 provider: capability surface, registration, snapshot/config resolution, and the
//! [`Generator`] entrypoint — the candle (Windows/CUDA) sibling of `mlx-gen-scail2`'s pipeline.
//!
//! [`Generator::generate`] maps the [`GenerationRequest`] conditioning onto the SCAIL-2 inputs and runs
//! the live [`crate::generate`] denoise pipeline: the primary **reference character** is a
//! [`Conditioning::Reference`] image paired with its color-coded [`Conditioning::Mask`]; the **driving
//! video + per-frame color masks** are a `ControlClip`; `video_mode == "replacement"` toggles the
//! cross-identity `replace_flag` (else animation). Inference adapters (`spec.adapters`) — LoRA / LoKr /
//! LoHa, the lightx2v lightning diff-patch, and the Bias-Aware DPO refinement LoRA — are folded into the
//! dense DiT before build ([`crate::adapters`], sc-6838). Multi-reference awaits the worker request
//! contract; [`crate::generate`] already supports extra characters via [`crate::generate::CharacterRef`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant,
    WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_wan::config::{TextEncoderConfig, Vae16Config};
use candle_gen_wan::scheduler::Sampler;
use candle_gen_wan::text_encoder::Umt5Encoder;
use candle_gen_wan::vae16::WanVae16;

use crate::clip::{ClipVisionConfig, ScailClip};
use crate::config::Scail2Config;
use crate::generate::{CharacterRef, Components, Scail2Job};
use crate::model::Scail2Dit;

/// Default driving-segment window + clean-history overlap (upstream `scail.py` defaults).
const SEGMENT_LEN: usize = 81;
const SEGMENT_OVERLAP: usize = 5;
/// Upstream `generate()` sampler defaults: 40 steps, shift 5.0, guide 5.0, 16 fps.
const DEFAULT_STEPS: u32 = 40;
const DEFAULT_SHIFT: f32 = 5.0;
const DEFAULT_GUIDANCE: f32 = 5.0;
const DEFAULT_FPS: u32 = 16;

/// SceneWorks/engine model id (matches `mlx-gen-scail2` so a consumer resolves the same engine across
/// backends). A still image is `num_frames == 1`.
pub const MODEL_ID: &str = "scail2_14b";

/// Stable identity + advertised capabilities for SCAIL-2 (Wan2.1-14B I2V end-to-end character
/// animation: reference image + driving video + color-coded masks → animated/identity-replaced video;
/// plain single-scale CFG; packed-token conditioning + per-source RoPE + CLIP image cross-attn).
/// `backend = "candle"`, `mac_only = false` (the off-Mac CUDA lane).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "scail2",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Reference character image (Reference) + its color-coded segmentation mask (Mask); extra
            // characters (MultiReference, experimental); the driving video + its per-frame color masks
            // map to ControlClip.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::MultiReference,
                ConditioningKind::ControlClip,
            ],
            // Inference LoRA / LoKr / LoHa + the lightx2v lightning diff-patch + the Bias-Aware DPO
            // refinement LoRA, merged into the dense DiT before build (sc-6838,
            // [`crate::adapters::merge_adapters`]).
            supports_lora: true,
            supports_lokr: true,
            // candle's FlowScheduler is UniPC/Euler; "dpm++" resolves to UniPC (bh2). Advertised to
            // match the mlx-gen-scail2 descriptor for cross-backend routing parity.
            samplers: vec!["unipc", "dpm++"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 32,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Load all `.safetensors` in the snapshot subdir `sub` as one f32 mmapped [`VarBuilder`].
fn component_vb(root: &Path, device: &Device, sub: &str) -> CResult<VarBuilder<'static>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "scail2 snapshot is missing the {sub}/ dir (expected text_encoder/ transformer/ vae/ \
             clip/ tokenizer/ at {})",
            root.display()
        )));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CandleError::Msg(format!("scail2: read {sub}/: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "scail2: no .safetensors in {sub}/ (at {})",
            dir.display()
        )));
    }
    // SAFETY: mmap of read-only weight files; standard candle loading path. All components run f32.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, device)? };
    Ok(vb)
}

/// The loaded SCAIL-2 model: resolved config + snapshot dir, with the heavy components (DiT / VAE /
/// UMT5 / CLIP) loaded lazily on first generate and cached.
pub struct Scail2 {
    descriptor: ModelDescriptor,
    config: Scail2Config,
    root: PathBuf,
    device: Device,
    /// Inference adapters (LoRA / LoKr / LoHa / lightx2v lightning diff-patch) folded into the DiT
    /// before build; empty for the stock path (sc-6838).
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Arc<Components>>>,
}

impl Scail2 {
    /// Build the DiT [`VarBuilder`] over the `transformer/` snapshot. With no adapters this is the
    /// stock f32 mmap build — **byte-identical** to the pre-sc-6838 path (the empty-adapter regression
    /// gate). With adapters, the base tensors are loaded to a CPU map, each delta is folded in
    /// ([`crate::adapters::merge_adapters`], f32 math — merge not residual, the chaos-sensitive-sampler
    /// rationale), the **whole map is cast to f32 on the CPU**, then the DiT is built from it.
    ///
    /// The host-side f32 cast is load-bearing for memory: SCAIL-2's DiT is f32, so a bf16 base tensor
    /// served through `from_tensors(F32, gpu)` would cast bf16→f32 *on the GPU*, and candle's CUDA
    /// caching allocator retains the freed bf16 staging blocks — ~28 GiB piled on top of the ~56 GiB
    /// f32 DiT, OOM-ing at the VAE-decode peak even on a 96 GiB card. Casting host-side (host RAM is
    /// ample, the map is transient) makes `get` a pure f32 host→device move, so the GPU footprint
    /// matches the stock mmap path exactly. (The Wan-14B merge path doesn't need this — its DiT is
    /// bf16, so `from_tensors` never casts on the GPU.)
    fn transformer_vb(&self) -> CResult<VarBuilder<'static>> {
        if self.adapters.is_empty() {
            return component_vb(&self.root, &self.device, "transformer");
        }
        let dir = self.root.join("transformer");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("scail2: read transformer/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "scail2: no .safetensors in transformer/ (at {})",
                dir.display()
            )));
        }
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for f in &files {
            let part = candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?;
            tensors.extend(part);
        }
        let report = crate::adapters::merge_adapters(&mut tensors, &self.adapters)?;
        eprintln!(
            "[scail2] merged {} adapter file(s): {} weight/bias deltas applied, {} keys off-surface/skipped",
            self.adapters.len(),
            report.merged,
            report.skipped_keys
        );
        // Cast host-side so `from_tensors` does no GPU-side bf16→f32 staging (see the doc note above).
        for v in tensors.values_mut() {
            if v.dtype() != DType::F32 {
                *v = v.to_dtype(DType::F32)?;
            }
        }
        Ok(VarBuilder::from_tensors(tensors, DType::F32, &self.device))
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Umt5Encoder::new(
            &TextEncoderConfig::umt5_xxl(),
            component_vb(&self.root, &self.device, "text_encoder")?,
        )?;
        let dit = Scail2Dit::new(self.transformer_vb()?, &self.config)?;
        let vae = WanVae16::new_with_encoder(
            &Vae16Config::wan21(),
            component_vb(&self.root, &self.device, "vae")?,
        )?;
        let clip = ScailClip::new(
            component_vb(&self.root, &self.device, "clip")?,
            &ClipVisionConfig::vit_h_14(),
        )?;
        Ok(Components { te, dit, vae, clip })
    }

    fn components(&self) -> CResult<Arc<Components>> {
        let mut guard = self
            .components
            .lock()
            .expect("scail2 components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = Arc::new(self.load_components()?);
        *guard = Some(c.clone());
        Ok(c)
    }
}

/// Construct a candle SCAIL-2 generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// snapshot with `text_encoder/`, `transformer/` (the converted SCAIL2Model DiT), `vae/` (z16 Wan VAE
/// with encoder), `clip/` (open-CLIP ViT-H/14 visual tower), and `tokenizer/tokenizer.json`. Inference
/// adapters (`spec.adapters` — LoRA / LoKr / LoHa / lightx2v lightning diff-patch / Bias-Aware DPO) are
/// merged into the dense DiT before build (sc-6838); on-the-fly quantization is still rejected.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "scail2: expected a snapshot directory (text_encoder/ transformer/ vae/ clip/ \
                 tokenizer/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle scail2 does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if !root.exists() {
        return Err(gen_core::Error::Msg(format!(
            "scail2: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    let config = Scail2Config::from_model_dir(&root)?;
    let device = candle_gen::default_device()?;
    Ok(Box::new(Scail2 {
        descriptor: descriptor(),
        config,
        root,
        device,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! { descriptor => load }

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

impl Generator for Scail2 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        Ok(self.run(req, on_progress)?)
    }
}

/// The first conditioning input matching `f`.
fn find_conditioning<'a, T>(
    req: &'a GenerationRequest,
    f: impl Fn(&'a Conditioning) -> Option<T>,
) -> Option<T> {
    req.conditioning.iter().find_map(f)
}

impl Scail2 {
    /// Map the request conditioning onto a [`Scail2Job`] and run the denoise pipeline.
    fn run(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<GenerationOutput> {
        let reference = find_conditioning(req, |c| match c {
            Conditioning::Reference { image, .. } => Some(image),
            _ => None,
        })
        .ok_or_else(|| {
            CandleError::Msg("scail2: a Reference character image is required".into())
        })?;
        let ref_mask = find_conditioning(req, |c| match c {
            Conditioning::Mask { image } => Some(image),
            _ => None,
        })
        .ok_or_else(|| {
            CandleError::Msg(
                "scail2: a Mask (the reference character's color-coded segmentation mask) is required"
                    .into(),
            )
        })?;
        let driving = req.control_clip().ok_or_else(|| {
            CandleError::Msg(
                "scail2: a ControlClip (driving video frames + per-frame color masks) is required"
                    .into(),
            )
        })?;

        let first: &Image = driving.frames.first().ok_or_else(|| {
            CandleError::Msg("scail2: the ControlClip has no driving frames".into())
        })?;
        let width = if req.width > 0 {
            req.width
        } else {
            first.width
        };
        let height = if req.height > 0 {
            req.height
        } else {
            first.height
        };

        let neg = req.negative_prompt.clone().unwrap_or_default();
        let job = Scail2Job {
            prompt: &req.prompt,
            negative_prompt: &neg,
            width,
            height,
            reference: CharacterRef {
                image: reference,
                mask: ref_mask,
            },
            additional: Vec::new(),
            driving_frames: driving.frames,
            driving_masks: driving.mask,
            replace_flag: req.video_mode.as_deref() == Some("replacement"),
            seed: req.seed.unwrap_or_else(gen_core::default_seed),
            steps: req.steps.unwrap_or(DEFAULT_STEPS) as usize,
            shift: req.scheduler_shift.unwrap_or(DEFAULT_SHIFT) as f64,
            guidance: req.guidance.unwrap_or(DEFAULT_GUIDANCE) as f64,
            sampler: Sampler::parse(req.sampler.as_deref()),
            fps: req.fps.unwrap_or(DEFAULT_FPS),
            segment_len: SEGMENT_LEN,
            segment_overlap: SEGMENT_OVERLAP,
        };
        let comps = self.components()?;
        let te_cfg = TextEncoderConfig::umt5_xxl();
        crate::generate::generate(
            &self.root,
            &comps,
            &te_cfg,
            &job,
            &|| req.cancel.is_cancelled(),
            on_progress,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // The snapshot dir doesn't exist, so `load` errors — but the engine must be REGISTERED (the
        // registry resolves the id to this provider's `load`).
        let err = registry::load(MODEL_ID, &spec).err().expect("dir missing");
        assert!(
            err.to_string().contains("does not exist"),
            "expected a missing-dir error from the scail2 loader, got: {err}"
        );
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "scail2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Video);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::Mask));
        assert!(d.capabilities.accepts(ConditioningKind::ControlClip));
        assert!(d.capabilities.samplers.contains(&"unipc"));
    }

    #[test]
    fn load_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // single-file source
        let f = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        assert!(load(&f).is_err());
        // LoRA adapters are now ACCEPTED (sc-6838) — `load` proceeds past the adapter check and fails
        // only on the missing snapshot dir, NOT with an Unsupported("LoRA") error.
        let lora = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        let err = load(&lora).err().expect("missing dir");
        assert!(
            !matches!(err, gen_core::Error::Unsupported(_)),
            "got: {err}"
        );
        assert!(err.to_string().contains("does not exist"), "got: {err}");
        // on-the-fly quant is still rejected
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
