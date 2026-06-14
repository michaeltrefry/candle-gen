//! # candle-gen-sensenova
//!
//! The **SenseNova-U1** (NEO-Unify) provider crate for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-sensenova`. It implements the backend-neutral
//! [`gen_core::Generator`] contract and self-registers via `inventory`, so linking this crate makes
//! `gen_core::load("sensenova_u1_8b", …)` (and `…_fast`) resolve the candle generator.
//!
//! **txt2img:** SenseNova-U1 is a *unified* multimodal model — a dense dual-path Qwen3 "MoT" backbone
//! (understanding + generation paths) with a flow-matching image head; there is no separate VAE or
//! text encoder. This slice wires the **non-think T2I** path the `Generator` contract drives: build
//! the `neo1_0` query, prefill it on the understanding path, then run the flow-matching denoise loop
//! on the generation path ([`crate::t2i`]) and unpatchify to RGB. Deterministic CPU-seeded noise
//! (sc-3673) makes output launch-portable per seed.
//!
//! Two registered ids share the loader: **`sensenova_u1_8b`** (50 NFE, CFG 4.0) and
//! **`sensenova_u1_8b_fast`** (8 NFE, CFG 1.0 — its loader merges the 8-step distill LoRA into the
//! dense generation path). Both advertise **only** the wired T2I surface — image-edit / Character
//! Studio (it2i), VQA, interleave, think-mode, user LoRAs, and quantization (all in the mlx provider)
//! are NOT advertised and are rejected rather than silently dropped. `backend` is `"candle"` and
//! `mac_only` is `false`.

mod config;
mod distill;
mod fm;
mod qwen3;
mod t2i;
mod text;
mod vision;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};
use candle_gen::{CandleError, Result};

use config::NeoChatConfig;
use distill::{resolve_distill_lora, DistillLora, DISTILL_LORA_FILE};
use t2i::{tensor_to_image, T2iModel, T2iOptions};
use text::SenseNovaTokenizer;

/// Registry id — the base 8B-MoT variant.
pub const MODEL_ID: &str = "sensenova_u1_8b";
/// The 8-step distilled variant (same base weights, distill LoRA merged at load).
pub const MODEL_ID_FAST: &str = "sensenova_u1_8b_fast";

const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 4.0;
/// Distilled defaults (`docs/base_vs_distill.md`): 8 NFE at CFG 1.0 (guidance off).
const DEFAULT_STEPS_FAST: u32 = 8;
const DEFAULT_GUIDANCE_FAST: f32 = 1.0;
const DEFAULT_TIMESTEP_SHIFT: f32 = 3.0;
/// Cell = patch·merge: every side must be a multiple of this (the patchify grid).
pub const SIZE_MULTIPLE: u32 = 32;

/// The base descriptor (`sensenova_u1_8b`).
pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}

/// The 8-step distilled descriptor (`sensenova_u1_8b_fast`). Identical capability surface to the
/// base — only the id and the generation defaults differ.
pub fn descriptor_fast() -> ModelDescriptor {
    descriptor_for(MODEL_ID_FAST)
}

/// SenseNova-U1's identity + the surface this candle slice wires: classifier-free guidance over the
/// prompt, **txt2img only**. it2i/edit (Reference), true-CFG image guidance, VQA, interleave, user
/// LoRA, and quantization (all wired in the mlx provider) are NOT advertised — they stay the Python
/// fallback's job until candle wires them, so the descriptor never promises a path `generate` can't
/// serve. Two backend-correct deviations from `mlx-gen-sensenova`: `backend = "candle"` and
/// `mac_only = false`.
fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "sensenova-u1",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: true,
            // it2i image-CFG (true_cfg) and reference conditioning are Phase 6 (understanding surface).
            supports_true_cfg: false,
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            // No on-the-fly quantization wired in the candle slice (dense f32).
            supported_quants: &[],
            // The backbone uses a KV cache for the AR prefix + denoise.
            supports_kv_cache: true,
            // Flow-match schedule uses a timestep shift (mapped from scheduler_shift).
            requires_sigma_shift: true,
        },
    }
}

/// The loaded SenseNova-U1 components, `Arc`-shared so the generator can cache them across calls.
#[derive(Clone)]
struct Components {
    tokenizer: Arc<SenseNovaTokenizer>,
    model: Arc<T2iModel>,
}

/// A loaded candle SenseNova-U1 generator. Loading is **lazy**: `load` does no file I/O (registry
/// introspection against a missing path still resolves), and the heavy unified model is built on the
/// first [`generate`](Generator::generate) call and then cached.
pub struct SenseNovaGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// The 8-step distilled variant — merges the distill LoRA at build + applies distilled defaults.
    fast: bool,
    components: Mutex<Option<Components>>,
}

impl SenseNovaGenerator {
    fn components(&self) -> Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("sensenova components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let cfg = NeoChatConfig::from_dir(&self.root)?;
        let vb = f32_vb(&self.root, &self.device)?;
        let mut model = T2iModel::from_weights(&vb, &cfg)?;
        if self.fast {
            // Merge the 8-step distill LoRA into the dense generation path. Assert full coverage —
            // `7 · layers` gen-path projections + the 2 FM-head Linears — so a stale/mismatched LoRA
            // fails loudly rather than silently merging a subset.
            let lora_path = resolve_distill_lora(&self.root)?;
            let lora = DistillLora::from_file(&lora_path, &self.device)?;
            let applied = model.merge_distill_lora(&lora)?;
            let expected = cfg.llm.num_hidden_layers * 7 + 2;
            if applied != expected {
                return Err(CandleError::Msg(format!(
                    "{}: distill LoRA merged {applied} targets, expected {expected} \
                     (7·{} gen-path linears + 2 fm_head) — wrong LoRA file?",
                    self.descriptor.id, cfg.llm.num_hidden_layers
                )));
            }
        }
        let tokenizer = SenseNovaTokenizer::from_dir(&self.root)?;
        let comps = Components {
            tokenizer: Arc::new(tokenizer),
            model: Arc::new(model),
        };
        *guard = Some(comps.clone());
        Ok(comps)
    }

    /// Map a request to [`T2iOptions`] (distilled vs base defaults; explicit request values win).
    fn options(&self, req: &GenerationRequest, seed: u64) -> T2iOptions {
        let (def_steps, def_guidance) = if self.fast {
            (DEFAULT_STEPS_FAST, DEFAULT_GUIDANCE_FAST)
        } else {
            (DEFAULT_STEPS, DEFAULT_GUIDANCE)
        };
        T2iOptions {
            cfg_scale: req.guidance.unwrap_or(def_guidance),
            num_steps: req.steps.unwrap_or(def_steps) as usize,
            timestep_shift: req.scheduler_shift.unwrap_or(DEFAULT_TIMESTEP_SHIFT),
            seed,
            ..Default::default()
        }
    }

    /// The rich-`Result` body behind [`Generator::generate`].
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let comps = self.components()?;
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let (w, h) = (req.width as usize, req.height as usize);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // A 50-step 8B run is multi-minute; check cancellation between images too (the per-step
            // check lives in the denoise loop).
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let opts = self.options(req, base_seed.wrapping_add(i as u64));
            let img = comps.model.generate(
                &comps.tokenizer,
                &req.prompt,
                w,
                h,
                &opts,
                &req.cancel,
                on_progress,
            )?;
            images.push(tensor_to_image(&img)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

impl Generator for SenseNovaGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        // Capability floor (count/size range, guidance; the empty `conditioning` rejects any
        // conditioning entry).
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        // `steps == 0` builds an empty denoise trajectory; `None` falls back to the variant default.
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

/// mmap an f32 [`VarBuilder`] over the SenseNova-U1 checkpoint shards (the flat `*.safetensors` under
/// `root`, excluding the optional co-located distill LoRA).
fn f32_vb(root: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| CandleError::Msg(format!("sensenova: read {}: {e}", root.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .filter(|p| p.file_name().and_then(|n| n.to_str()) != Some(DISTILL_LORA_FILE))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "sensenova: no .safetensors found in {} (expected a SenseNova-U1-8B-MoT snapshot)",
            root.display()
        )));
    }
    // SAFETY: mmap of read-only weight files; the standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, device)? })
}

/// Construct the (lazy) base candle SenseNova-U1 generator (`sensenova_u1_8b`).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_inner(spec, false)
}

/// Construct the (lazy) 8-step distilled generator (`sensenova_u1_8b_fast`).
pub fn load_fast(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_inner(spec, true)
}

fn load_inner(spec: &LoadSpec, fast: bool) -> gen_core::Result<Box<dyn Generator>> {
    let id = if fast { MODEL_ID_FAST } else { MODEL_ID };
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a SenseNova-U1-8B-MoT snapshot directory, not a single .safetensors file"
            )));
        }
    };
    // User-supplied LoRAs are unsupported on both ids — the distill LoRA is merged internally by the
    // fast loader, never stacked via `spec.adapters`.
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: user-supplied adapters are not supported (supports_lora=false)"
        )));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: on-the-fly Q4/Q8 quantization is not wired in the candle slice yet (dense f32 only)"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: control / IP-adapter overlays are not supported (txt2img only)"
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(SenseNovaGenerator {
        descriptor: descriptor_for(id),
        root,
        device,
        fast,
        components: Mutex::new(None),
    }))
}

// Link-time self-registration of both ids into gen-core's model registry.
inventory::submit! {
    ModelRegistration { descriptor, load }
}
inventory::submit! {
    ModelRegistration { descriptor: descriptor_fast, load: load_fast }
}

/// Force-link hook. A consumer that only reaches this provider *through* the `gen_core` registry
/// references nothing here directly, so the linker (MSVC on a release build in particular) can discard
/// the whole rlib — taking the `inventory::submit!` registrations with it. Referencing this no-op from
/// the consumer keeps the crate linked. (Same pattern as `candle_gen_chroma::force_link`.)
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, Conditioning, Image, Quant};

    #[test]
    fn registers_both_ids_as_candle() {
        let ids: Vec<&str> = registry::generators()
            .map(|r| (r.descriptor)().id)
            .collect();
        assert!(ids.contains(&MODEL_ID), "{MODEL_ID} not registered");
        assert!(
            ids.contains(&MODEL_ID_FAST),
            "{MODEL_ID_FAST} not registered"
        );

        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("candle sensenova is registered");
        assert_eq!(g.descriptor().id, "sensenova_u1_8b");
        assert_eq!(g.descriptor().family, "sensenova-u1");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_only_wired_t2i_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.supports_lora);
        assert!(!d.capabilities.supports_lokr);
        assert!(d.capabilities.supported_quants.is_empty());
        assert!(d.capabilities.supports_kv_cache);
        assert!(d.capabilities.requires_sigma_shift);
        // The fast variant shares the capability surface; only id + defaults differ.
        let f = descriptor_fast();
        assert_eq!(f.id, MODEL_ID_FAST);
        assert_eq!(f.family, d.family);
        assert_eq!(f.capabilities.max_size, d.capabilities.max_size);
    }

    #[test]
    fn validate_accepts_t2i_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();

        let ok = GenerationRequest {
            prompt: "a cat holding a lit candle".into(),
            width: 512,
            height: 512,
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        for bad in [
            GenerationRequest {
                width: 512,
                height: 512,
                ..Default::default()
            }, // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 300, // not a multiple of 32
                height: 512,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 512,
                height: 512,
                steps: Some(0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 512,
                height: 512,
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
        // The fast loader rejects user adapters too (its distill LoRA is internal, not user-supplied).
        assert!(matches!(
            load_fast(&lora).err().expect("err"),
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
