//! # candle-gen-wan
//!
//! The **Wan2.2 TI2V-5B** text-to-video provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-wan`. Wan has **no** `candle-transformers` reference: the
//! `WanTransformer3DModel` DiT ([`transformer`]), the causal-Conv3d `AutoencoderKLWan` temporal VAE
//! ([`vae`], built on a from-scratch [`conv3d`] since candle ships none), the UMT5-XXL encoder
//! ([`text_encoder`]), and the UniPC flow-match scheduler ([`scheduler`]) are all ported here from
//! the diffusers checkpoint.
//!
//! **txt2video (sc-3697):** [`WanGenerator::generate`] runs UMT5-XXL → the 30-layer DiT (3-axis
//! interleaved RoPE, AdaLN modulation, cross-attention to text, classifier-free guidance, UniPC) →
//! the temporal VAE decoder, emitting `GenerationOutput::Video`. Registered under `"wan2_2_ti2v_5b"`.
//!
//! **Dtypes:** UMT5 + VAE run **f32**; the 5B DiT runs **bf16** (its native dtype), norms/modulation
//! upcast to f32. `backend = "candle"`, `mac_only = false`.
//!
//! **First-slice surface:** txt2video only. The mlx provider's image-conditioning (TI2V / I2V),
//! VACE, LoRA, and quantization surface is **deferred**. The z48 vae22 decode is memory-bounded:
//! the temporal axis streams per-frame ([`vae::WanVae::decode`]) and a budgeted **spatial** tiler
//! ([`vae::WanVae::decode_budgeted`], sc-7111) caps a single high-res frame's VRAM spike.

pub mod adapters;
pub mod config;
pub mod conv3d;
pub mod dit_train;
pub mod model_vace;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod training;
pub mod transformer;
pub mod vace;
pub mod vae;
pub mod vae16;
pub mod wan14b;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use candle_gen::gen_core::sampling::TimestepConvention;
use config::{
    TextEncoderConfig, TransformerConfig, VaeConfig, DEFAULT_FPS, DEFAULT_FRAMES, DEFAULT_GUIDANCE,
    DEFAULT_STEPS, MODEL_ID, NEGATIVE_FALLBACK, SIZE_MULTIPLE,
};
use rope::WanRope;
use scheduler::{flow_shift, FlowScheduler, Sampler};
use text_encoder::Umt5Encoder;
use transformer::WanTransformer;
use vae::WanVae;

/// The 5B DiT runs bf16 (native checkpoint dtype); the UMT5 encoder and the VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
const VAE_DTYPE: DType = DType::F32;
const Z_DIM: usize = 48;

#[derive(Clone)]
struct Components {
    te: Arc<Umt5Encoder>,
    dit: Arc<WanTransformer>,
    vae: Arc<WanVae>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    dit_cfg: TransformerConfig,
    vae_cfg: VaeConfig,
    root: PathBuf,
    device: Device,
}

impl Pipeline {
    fn load(root: &Path, device: &Device) -> Self {
        Self {
            te_cfg: TextEncoderConfig::umt5_xxl(),
            dit_cfg: TransformerConfig::ti2v_5b(),
            vae_cfg: VaeConfig::ti2v_5b(),
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "wan snapshot is missing the {sub}/ dir (expected a Wan2.2-TI2V-5B diffusers \
                 snapshot at {})",
                self.root.display()
            )));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("wan: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "wan: no .safetensors in {sub}/ (at {})",
                dir.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &self.device)? };
        Ok(vb)
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Umt5Encoder::new(&self.te_cfg, self.component_vb("text_encoder", ENC_DTYPE)?)?;
        let dit = WanTransformer::new(&self.dit_cfg, self.component_vb("transformer", DIT_DTYPE)?)?;
        let vae = WanVae::new(&self.vae_cfg, self.component_vb("vae", VAE_DTYPE)?)?;
        Ok(Components {
            te: Arc::new(te),
            dit: Arc::new(dit),
            vae: Arc::new(vae),
        })
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, S, 4096]` (f32).
    fn encode(&self, te: &Umt5Encoder, prompt: &str) -> CResult<Tensor> {
        let tok = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.te_cfg.max_length,
                pad_token_id: self.te_cfg.pad_token_id,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("wan: load tokenizer: {e}")))?;
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("wan: tokenize: {e}")))?;
        let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        if ids.is_empty() {
            // The gen_core tokenizer short-circuits an empty prompt to zero ids, but UMT5/T5 encode
            // the empty string as a single token. A 0-length sequence here would build a degenerate
            // `(1,1)` tensor (the old `.max(1)` padded the *shape* but not the data), and the f32
            // embedding gather over zero indices is a 0-element CUDA `index_select` that reads out of
            // bounds → `CUDA_ERROR_ILLEGAL_ADDRESS` (it surfaces deferred at the next cublas call as a
            // misleading `CUBLAS_STATUS_EXECUTION_FAILED`). Emit one pad token so a 0-length sequence
            // never reaches the gather. (sc-7078)
            ids.push(self.te_cfg.pad_token_id as u32);
        }
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let embeds = te.encode(&input_ids)?; // [1, S, 4096]

        // The Wan DiT cross-attends over a context **zero-padded to `max_length` (512)** — the
        // reference `WanPipeline` pads the UMT5 embeds to 512 before the transformer (the model was
        // trained that way). Feeding only the real tokens silently breaks conditioning. (sc-3697)
        let max_len = self.te_cfg.max_length;
        let dim = embeds.dim(2)?;
        match len.cmp(&max_len) {
            std::cmp::Ordering::Less => {
                let pad = Tensor::zeros((1, max_len - len, dim), embeds.dtype(), &self.device)?;
                Ok(Tensor::cat(&[&embeds, &pad], 1)?)
            }
            std::cmp::Ordering::Greater => Ok(embeds.narrow(1, 0, max_len)?),
            std::cmp::Ordering::Equal => Ok(embeds),
        }
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS as usize);
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES);
        let fps = req.fps.unwrap_or(DEFAULT_FPS);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE) as f64;
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let sampler = Sampler::parse(req.sampler.as_deref());
        let shift = flow_shift(req.scheduler_shift);

        // Text encode (pos + optional neg for CFG), then project to the DiT context once.
        let pos_embeds = self.encode(&comps.te, &req.prompt)?;
        let ctx_pos = comps.dit.embed_text(&pos_embeds)?;
        let ctx_neg = if guidance > 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
            Some(comps.dit.embed_text(&self.encode(&comps.te, neg)?)?)
        } else {
            None
        };

        // Latent geometry + RoPE for the token grid.
        let (t_lat, h_lat, w_lat) = pipeline::latent_dims(frames, req.width, req.height);
        let (pt, ph, pw) = self.dit_cfg.patch;
        let (ppf, pph, ppw) = (t_lat / pt, h_lat / ph, w_lat / pw);
        let (cos, sin) = WanRope::new(&self.dit_cfg).cos_sin(ppf, pph, ppw, &self.device)?;

        let latents0 = pipeline::create_noise(seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;

        // epic 7114 P4 (sc-7124) Wan fold-in: the gen-core-only curated solvers (euler_ancestral /
        // heun / dpmpp_sde / ddim) run over Wan's NATIVE flow σ schedule via the shared driver — one
        // solver library. Wan's native UniPC (curated `uni_pc`, sc-7296) / `euler` (the diffusers
        // FLOW-SNR multistep + flow Euler) stay the byte-exact default path; gen-core's VE-space
        // `uni_pc`/`dpmpp_2m` are deliberately NOT routed through the fold-in (they would diverge from
        // Wan's diffusers parity). The DiT timestep is `σ·N` (Sigma convention, ×N applied in the
        // closure); the model output is the velocity (CFG combined inside).
        const FOLDIN: &[&str] = &["euler_ancestral", "heun", "dpmpp_sde", "ddim"];
        let latents = if let Some(name) = req.sampler.as_deref().filter(|n| FOLDIN.contains(n)) {
            let native = scheduler::flow_sigmas(steps, shift);
            let n_train = config::NUM_TRAIN_TIMESTEPS as f64;
            candle_gen::run_flow_sampler(
                Some(name),
                TimestepConvention::Sigma,
                &native,
                latents0,
                seed,
                &req.cancel,
                on_progress,
                |latents, t| -> CResult<Tensor> {
                    let ts = t as f64 * n_train;
                    let v_pos = comps.dit.forward(latents, &ctx_pos, ts, &cos, &sin)?;
                    let v = match &ctx_neg {
                        Some(neg) => {
                            let v_neg = comps.dit.forward(latents, neg, ts, &cos, &sin)?;
                            pipeline::cfg(&v_pos, &v_neg, guidance)?
                        }
                        None => v_pos,
                    };
                    Ok(v)
                },
            )?
        } else {
            // Native FlowScheduler (UniPC default / flow Euler) — the byte-exact N1 path, untouched.
            let mut latents = latents0;
            let mut sched = FlowScheduler::new(sampler, steps, shift);
            let total = steps as u32;
            for i in 0..steps {
                if req.cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                let t = sched.timestep(i);
                let v_pos = comps.dit.forward(&latents, &ctx_pos, t, &cos, &sin)?;
                let v = match &ctx_neg {
                    Some(neg) => {
                        let v_neg = comps.dit.forward(&latents, neg, t, &cos, &sin)?;
                        pipeline::cfg(&v_pos, &v_neg, guidance)?
                    }
                    None => v_pos,
                };
                latents = sched.step(&v, &latents)?;
                on_progress(Progress::Step {
                    current: i as u32 + 1,
                    total,
                });
            }
            latents
        };

        on_progress(Progress::Decoding);
        // Memory-bounded z48 vae22 decode (sc-7111): the per-frame streaming `decode` already bounds
        // the temporal axis; `decode_budgeted` adds budgeted **spatial** tiling so a single high-res
        // frame can't spike VRAM, and returns a catchable error rather than OOM-ing when over budget.
        let decoded = comps.vae.decode_budgeted(&latents)?;
        let images = pipeline::frames_to_images(&decoded)?;
        Ok((images, fps))
    }
}

pub struct WanGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

impl WanGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("wan components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = pipe.load_components()?;
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for WanGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg("wan: prompt must not be empty".into()));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg("wan: steps must be >= 1".into()));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "wan: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % 4 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "wan: frames must satisfy frames % 4 == 1 (got {f})"
                )));
            }
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
        let (frames, fps) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }
}

/// Wan2.2 TI2V-5B txt2video descriptor — the surface sc-3697 wires: CFG txt2video with a negative
/// prompt, UniPC / Euler samplers; no conditioning (image / VACE deferred), no LoRA/quant.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            // Native flow samplers (curated `uni_pc` default / `euler`) + the epic 7114 P4 (sc-7124)
            // curated fold-in: the gen-core-only solvers over Wan's native flow σ schedule. The curated
            // `uni_pc` (sc-7296) is honored by Wan's OWN native UniPC; gen-core's VE-space `uni_pc`/
            // `dpmpp_2m` solvers are NOT routed through the fold-in (they would diverge from Wan's
            // diffusers FLOW-SNR parity). Legacy `unipc` retained as an alias for recipe back-compat. No
            // scheduler axis (the flow shift is the `scheduler_shift` knob).
            samplers: vec![
                "uni_pc",
                "euler",
                "euler_ancestral",
                "heun",
                "dpmpp_sde",
                "ddim",
                "unipc",
            ],
            schedulers: vec![],
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

/// Construct a lazy candle Wan generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing at
/// a `Wan-AI/Wan2.2-TI2V-5B-Diffusers` snapshot (`text_encoder/`, `transformer/`, `vae/`,
/// `tokenizer/`). Adapters / quantization / control overlays are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "wan expects a snapshot directory (text_encoder/ transformer/ vae/ tokenizer/), \
                 not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle wan does not support LoRA/LoKr yet".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle wan does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle wan does not support image / VACE conditioning yet (txt2video only)".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(WanGenerator {
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! { descriptor => load }

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).expect("wan is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "wan");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.requires_sigma_shift);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.samplers.contains(&"uni_pc")); // curated spelling (sc-7296)
        assert!(d.capabilities.samplers.contains(&"unipc")); // legacy alias retained
        assert!(d.capabilities.samplers.contains(&"euler"));
    }

    #[test]
    fn validate_accepts_txt2video_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(MODEL_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 256,
            height: 256,
            guidance: Some(5.0),
            negative_prompt: Some("blurry".into()),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        // Legacy `unipc` spelling stays accepted (sc-7296 alias).
        assert!(g
            .validate(&GenerationRequest {
                sampler: Some("unipc".into()),
                ..ok.clone()
            })
            .is_ok());
        for bad in [
            // empty prompt
            GenerationRequest::default(),
            // frames not ≡ 1 (mod 4)
            GenerationRequest {
                prompt: "x".into(),
                frames: Some(16),
                ..Default::default()
            },
            // size not a multiple of 32
            GenerationRequest {
                prompt: "x".into(),
                width: 300,
                ..Default::default()
            },
            // zero steps
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
            // unadvertised sampler
            GenerationRequest {
                prompt: "x".into(),
                sampler: Some("dpmpp2m".into()),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
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
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
