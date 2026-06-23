//! # candle-gen-flux2
//!
//! The **FLUX.2** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-flux2`. Unlike FLUX.1 (sc-3694), FLUX.2 has **no** `candle-transformers` reference: the
//! MMDiT transformer ([`transformer`]), the 32-channel 2×2-patchify VAE ([`vae`]), the decoder-LM
//! prompt-embeds text path ([`text_encoder`]), the 4-axis RoPE ([`pos_embed`]) and the flow-match
//! geometry ([`pipeline`]) are all ported here from the macOS provider.
//!
//! **Two txt2img variants** are registered, selected by [`config::Flux2Variant`]:
//! - **`flux2_klein_9b`** (sc-3695): Qwen3 TE (hidden states 9/18/27 → 12288-wide `prompt_embeds`) →
//!   the MMDiT (8 joint + 24 fused-single blocks) → the AutoencoderKL-Flux2 decoder. Distilled
//!   **4-step** flow-match Euler, CFG-free at guidance 1.0 (>1 runs a classifier-free negative pass).
//! - **`flux2_dev`** (epic 6564 sc-7457): the 32B flagship. **Mistral** TE (layers 10/20/30 →
//!   15360-wide `prompt_embeds`) → a wider/deeper MMDiT (8 joint + **48** single blocks, **48** heads,
//!   joint 15360). Guidance-**distilled** (embedded scalar, the FLUX.1-dev pattern): ~28 steps at
//!   guidance ~4 via a single forward feeding the DiT's guidance embedder — **not** true CFG.
//!
//! Same deterministic CPU-seeded-noise contract (sc-3673). Tokenization reuses gen-core's
//! [`TextTokenizer`]: klein with [`ChatTemplate::QwenInstructNoThink`], dev with
//! [`ChatTemplate::Flux2DevMistral`].
//!
//! **Sampling (epic 7114 P4, sc-7123):** both denoise loops (txt2img [`Pipeline::render`] and the edit
//! path [`Flux2Edit`]) route through the unified curated sampler/scheduler driver
//! (`candle_gen::run_flow_sampler` / `resolve_flow_schedule`). FLUX.2 is a rectified-flow engine using
//! the `Sigma` convention but embeds σ×1000, so the predict closure feeds `sigma * 1000.0` to the
//! transformer; the klein guidance>1 CFG blend / the dev embedded-guidance scalar (and, on the edit
//! path, the joint `[target, refs]` concat) live inside that closure. The descriptor advertises the
//! curated sampler/scheduler menus; the default (unset sampler/scheduler) path is the N1 no-op — euler
//! over the native empirical-mu flow-match schedule.
//!
//! **Surface:** txt2img for both variants. The mlx provider's edit variants (`flux2_klein_9b_edit`,
//! `flux2_klein_9b_kv_edit`, `flux2_dev_edit`/`flux2_dev_control`), LoRA/LoKr, and the **Q4 pre-quant
//! packed loader** dev needs to fit under the memory ceiling are **not** wired yet (epic 6564 stories
//! 1 follow-on / 4); until the packed loader lands, dev loads **dense** (fixture-only — the full 32B
//! needs the packed path). `backend = "candle"`, `mac_only = false`.

pub mod config;
pub mod convert;
pub mod edit_provider;
pub mod pipeline;
pub mod pos_embed;
pub mod quant;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use convert::convert_and_assemble;
pub use edit_provider::{Flux2Edit, Flux2EditPaths, Flux2EditRequest};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use config::{Flux2Config, Flux2Variant, SIZE_MULTIPLE};
use text_encoder::Qwen3TextEncoder;
use transformer::Flux2Transformer;
use vae::Flux2Vae;

/// Qwen3 `<|endoftext|>` pad token id (klein FLUX.2 text encoder).
const QWEN_PAD_TOKEN_ID: i32 = 151643;
/// Mistral `<pad>` pad token id (dev FLUX.2 text encoder).
const MISTRAL_PAD_TOKEN_ID: i32 = 11;

/// The loaded FLUX.2 components, `Arc`-shared so the generator caches them across `generate` calls.
#[derive(Clone)]
struct Components {
    te: Arc<Qwen3TextEncoder>,
    transformer: Arc<Flux2Transformer>,
    vae: Arc<Flux2Vae>,
}

/// A txt2img pipeline handle: snapshot root + device + the f32 compute dtype. `pub(crate)` so the
/// edit provider ([`edit_provider`]) reuses the snapshot mmap + prompt-encode scaffolding.
pub(crate) struct Pipeline {
    pub(crate) variant: Flux2Variant,
    pub(crate) cfg: Flux2Config,
    pub(crate) root: PathBuf,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
    /// When `Some`, the quantizable components (TE + DiT) are staged dense in CPU RAM and quantized
    /// onto `device` (the dev 32B path; klein leaves this `None`).
    pub(crate) quant: Option<Quant>,
}

impl Pipeline {
    pub(crate) fn load(
        variant: Flux2Variant,
        quant: Option<Quant>,
        root: &Path,
        device: &Device,
    ) -> Self {
        Self {
            variant,
            cfg: variant.config(),
            root: root.to_path_buf(),
            device: device.clone(),
            // FLUX.2 runs the reference math in f32 (the TE + the MMDiT). The weights are large but
            // the math is parity-sensitive; a bf16 pass is a follow-up optimization.
            dtype: DType::F32,
            quant,
        }
    }

    /// mmap a VarBuilder over every `.safetensors` in the snapshot subdir `sub`, on `self.device`.
    pub(crate) fn component_vb(&self, sub: &str) -> CResult<VarBuilder<'static>> {
        self.component_vb_on(sub, &self.device)
    }

    /// [`Self::component_vb`] but on an explicit `device` — the quant path stages the TE + DiT on the
    /// CPU (system RAM) before quantizing onto the GPU, so the dense 32B never lands on the GPU.
    pub(crate) fn component_vb_on(
        &self,
        sub: &str,
        device: &Device,
    ) -> CResult<VarBuilder<'static>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "flux2 snapshot is missing the {sub}/ component dir (expected a FLUX.2 \
                 diffusers snapshot at {})",
                self.root.display()
            )));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("flux2: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "flux2: no .safetensors in {sub}/ (at {})",
                dir.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, self.dtype, device)? };
        Ok(vb)
    }

    fn load_components(&self) -> CResult<Components> {
        let (te, transformer) = match self.quant {
            // Quant path (dev 32B): stage the TE + DiT dense in CPU RAM (~105 GB; the box has 512),
            // then quantize each projection onto the GPU and move the small dense leaves there too —
            // the dense weights never land on the GPU. The VAE is small and loads directly on-device.
            Some(q) => {
                let cpu = Device::Cpu;
                let mut te =
                    Qwen3TextEncoder::new(&self.cfg, self.component_vb_on("text_encoder", &cpu)?)?;
                te.quantize(q, &self.device)?;
                let mut transformer =
                    Flux2Transformer::new(&self.cfg, self.component_vb_on("transformer", &cpu)?)?;
                transformer.quantize(q, &self.device)?;
                (te, transformer)
            }
            // Dense path (klein, and dev on a fixture small enough to fit): load on-device directly.
            None => (
                Qwen3TextEncoder::new(&self.cfg, self.component_vb("text_encoder")?)?,
                Flux2Transformer::new(&self.cfg, self.component_vb("transformer")?)?,
            ),
        };
        let vae = Flux2Vae::new(self.component_vb("vae")?)?;
        Ok(Components {
            te: Arc::new(te),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
        })
    }

    /// Tokenize + encode the prompt to `prompt_embeds` `[1, 512, 3·hidden]` (f32). The tokenizer
    /// (pad token + chat template) is variant-specific: klein uses the Qwen2 `<|endoftext|>` pad +
    /// the Qwen no-think chat template; dev uses the Mistral `<pad>` + the `[INST]…[/INST]` template.
    pub(crate) fn encode(&self, te: &Qwen3TextEncoder, prompt: &str) -> CResult<Tensor> {
        let (pad_token_id, chat_template) = if self.variant.is_dev() {
            (MISTRAL_PAD_TOKEN_ID, ChatTemplate::Flux2DevMistral)
        } else {
            (QWEN_PAD_TOKEN_ID, ChatTemplate::QwenInstructNoThink)
        };
        let tok = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.cfg.max_sequence_length,
                pad_token_id,
                chat_template,
                pad_to_max_length: true,
            },
        )
        .map_err(|e| CandleError::Msg(format!("flux2: load tokenizer: {e}")))?;
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("flux2: tokenize: {e}")))?;
        let len = out.ids.len();
        let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        let mask: Vec<i64> = out.mask.iter().map(|&m| m as i64).collect();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let attn_mask = Tensor::from_vec(mask, (1, len), &self.device)?;
        Ok(te.prompt_embeds(&input_ids, &attn_mask)?)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(self.variant.default_steps() as usize);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let guidance = req.guidance.unwrap_or(self.variant.default_guidance());
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);

        // Prompt embeds are seed-independent: encode once.
        let prompt_embeds = self.encode(&comps.te, &req.prompt)?;
        dbg_stats("prompt_embeds", &prompt_embeds);
        // Two guidance regimes. dev is guidance-distilled: a single forward feeds the guidance scalar
        // to the DiT's embedded-guidance embedder (no negative pass). klein is distilled / true-CFG:
        // a classifier-free negative pass only when guidance > 1 (it runs CFG-free at 1.0).
        let embedded_guidance = self.variant.uses_embedded_guidance();
        let negative = if !embedded_guidance && guidance > 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or(" ");
            Some(self.encode(&comps.te, neg)?)
        } else {
            None
        };

        let img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        let txt_ids = pipeline::prepare_text_ids(self.cfg.max_sequence_length);

        // Curated sampler/scheduler routing (epic 7114 P4, sc-7123). The NATIVE schedule is the legacy
        // empirical-mu flow-match sigmas (descending, trailing 0.0); the same `mu` feeds the curated
        // scheduler axis so `normal`/`karras`/etc. honor the resolution-dependent shift. The default path
        // (sampler/scheduler unset) is the N1 no-op — euler over the native schedule reproduces the legacy
        // `euler_step` flow-match loop within tolerance.
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), steps);
        let (native, _timesteps) = pipeline::schedule(steps, req.width, req.height);
        let sigmas =
            candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = base_seed.wrapping_add(index as u64);
            let latents =
                pipeline::create_noise(&self.cfg, seed, req.width, req.height, &self.device)?;

            // The driver does cancel + progress + the euler/curated integrator step. The forward (and the
            // guidance>1 CFG blend) lives inside `predict` so a multi-eval solver re-runs it. FLUX.2 uses
            // the Sigma convention but the model embeds σ×1000, so feed `sigma * 1000.0` to the transformer.
            let mut dbg_first = true;
            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::Sigma,
                &sigmas,
                latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, sigma| -> CResult<Tensor> {
                    let ts = sigma * 1000.0;
                    let out = if embedded_guidance {
                        // dev: single forward feeding the embedded guidance scalar to the DiT.
                        comps.transformer.forward(
                            latents,
                            &prompt_embeds,
                            &img_ids,
                            &txt_ids,
                            ts,
                            Some(guidance),
                        )?
                    } else {
                        let v = comps.transformer.forward(
                            latents,
                            &prompt_embeds,
                            &img_ids,
                            &txt_ids,
                            ts,
                            None,
                        )?;
                        match &negative {
                            Some(neg) => {
                                let vn = comps
                                    .transformer
                                    .forward(latents, neg, &img_ids, &txt_ids, ts, None)?;
                                // vn + guidance·(v − vn)
                                (&vn + ((&v - &vn)? * guidance as f64)?)?
                            }
                            None => v,
                        }
                    };
                    if dbg_first {
                        dbg_stats("latents_in@step0", latents);
                        dbg_stats("velocity@step0", &out);
                        dbg_first = false;
                    }
                    Ok(out)
                },
            )?;
            dbg_stats("final_latents", &latents);

            on_progress(Progress::Decoding);
            let packed = pipeline::unpack_latents(&latents, req.width, req.height)?;
            let decoded = comps.vae.decode_packed(&packed)?; // [1,3,H,W] in [-1,1]
            dbg_stats("decoded", &decoded);
            images.push(to_image(&decoded)?);
        }
        Ok(images)
    }
}

/// Debug probe (sc-7457 dev-quant black-image hunt): print tensor stats to stderr when `FLUX2_DEBUG`
/// is set. Localizes the first non-finite / degenerate stage without changing normal-run behavior.
pub(crate) fn dbg_stats(name: &str, t: &Tensor) {
    if std::env::var_os("FLUX2_DEBUG").is_none() {
        return;
    }
    let dims = t.dims().to_vec();
    let r = (|| -> CResult<(f32, f32, f64, bool)> {
        let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let (mut mn, mut mx, mut s, mut bad) = (f32::INFINITY, f32::NEG_INFINITY, 0f64, false);
        for &x in &v {
            if x.is_finite() {
                mn = mn.min(x);
                mx = mx.max(x);
            } else {
                bad = true;
            }
            s += x as f64;
        }
        Ok((mn, mx, s / v.len().max(1) as f64, bad))
    })();
    match r {
        Ok((mn, mx, me, bad)) => eprintln!(
            "[dbg] {name} shape={dims:?} min={mn:.4} max={mx:.4} mean={me:.4} nonfinite={bad}"
        ),
        Err(e) => eprintln!("[dbg] {name}: stats err {e}"),
    }
}

/// Map a decoded `[1, 3, H, W]` tensor in `[-1, 1]` to an RGB8 [`Image`].
pub(crate) fn to_image(decoded: &Tensor) -> CResult<Image> {
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// A loaded candle FLUX.2 generator. Loading is lazy; components build on the first `generate` and
/// are cached. `variant` selects klein vs dev (config, text encoder, tokenizer, guidance regime).
pub struct Flux2Generator {
    variant: Flux2Variant,
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// `Some` ⇒ CPU-stage → quantize-onto-GPU at load (dev Q4/Q8); `None` ⇒ dense.
    quant: Option<Quant>,
    components: Mutex<Option<Components>>,
}

impl Flux2Generator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("flux2 components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = pipe.load_components()?;
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for Flux2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
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
        let pipe = Pipeline::load(self.variant, self.quant, &self.root, &self.device);
        let components = self.components(&pipe)?;
        let images = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// The txt2img descriptor for `variant`. **klein**: guidance advertised (defaults to 1.0 / CFG-free,
/// but >1.0 runs a classifier-free negative pass), so `supports_negative_prompt`. **dev**: guidance is
/// the embedded scalar (single forward, no negative pass), so `supports_negative_prompt = false`.
/// Both: txt2img only (edit/Reference deferred to epic 6564 story 4), no LoRA, no on-the-fly quant.
fn descriptor(variant: Flux2Variant) -> ModelDescriptor {
    ModelDescriptor {
        id: variant.id(),
        family: "flux2",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // dev is guidance-distilled (embedded scalar, no negative pass); klein runs a
            // classifier-free negative pass when guidance > 1.
            supports_negative_prompt: !variant.uses_embedded_guidance(),
            supports_guidance: true,
            supports_true_cfg: false,
            // txt2img only in this slice — the mlx edit/Reference surface is deferred.
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            // Curated sampler/scheduler menu (epic 7114 P4, sc-7123). The legacy `flow_match_euler`
            // scheduler alias is retained and falls back to the native schedule via the N3 path.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["flow_match_euler"],
            ),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            // dev quantizes (CPU-stage → quantize-onto-GPU) to fit the 32B under the memory ceiling;
            // klein is small and runs dense.
            supported_quants: if variant.is_dev() {
                &[Quant::Q4, Quant::Q8]
            } else {
                &[] as &[Quant]
            },
            supports_kv_cache: false,
            // FLUX.2 uses the empirical-mu shifted flow-match schedule.
            requires_sigma_shift: true,
        },
    }
}

/// FLUX.2-klein-9b txt2img descriptor (the surface sc-3695 wired).
pub fn descriptor_klein() -> ModelDescriptor {
    descriptor(Flux2Variant::Klein9b)
}

/// FLUX.2-dev txt2img descriptor (epic 6564 story 1): the guidance-distilled 32B flagship.
pub fn descriptor_dev() -> ModelDescriptor {
    descriptor(Flux2Variant::Dev)
}

/// Construct a lazy candle FLUX.2 generator for `variant`. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`,
/// `tokenizer/`) — klein at `black-forest-labs/FLUX.2-klein-9B`, dev at `black-forest-labs/FLUX.2-dev`
/// (whose `text_encoder/` is the Mistral3 checkpoint). Adapters / control overlays are rejected (not
/// wired). `spec.quantize` (Q4/Q8) is honored for **dev** — the 32B is staged dense in CPU RAM and
/// quantized onto the GPU at load (it does not fit the GPU dense); for **klein** quantization is not
/// wired and is rejected. dev without quant loads dense (fixture-only — the full 32B needs the quant).
fn load_variant(variant: Flux2Variant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (text_encoder/ transformer/ vae/ tokenizer/), \
                 not a single .safetensors file"
            )));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support LoRA/LoKr yet"
        )));
    }
    // dev honors Q4/Q8 (CPU-stage → quantize-onto-GPU); klein has no candle quant path yet.
    let quant = if variant.is_dev() {
        spec.quantize
    } else if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support on-the-fly Q4/Q8 quantization yet"
        )));
    } else {
        None
    };
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter / edit yet (txt2img only)"
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(Flux2Generator {
        variant,
        descriptor: descriptor(variant),
        root,
        device,
        quant,
        components: Mutex::new(None),
    }))
}

/// Registry load hook for `flux2_klein_9b`.
pub fn load_klein(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

/// Registry load hook for `flux2_dev`.
pub fn load_dev(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Dev, spec)
}

// Link-time self-registration into gen-core's model registry — one per txt2img variant.
inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein, load: load_klein }
}
inventory::submit! {
    ModelRegistration { descriptor: descriptor_dev, load: load_dev }
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped when the
/// crate is reached only through the registry). Same pattern as the other providers.
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX2_DEV_ID, FLUX2_KLEIN_9B_ID};
    use candle_gen::gen_core::registry;
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(FLUX2_KLEIN_9B_ID, &spec).expect("flux2 is registered");
        assert_eq!(g.descriptor().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn klein_descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor_klein();
        assert_eq!(d.id, FLUX2_KLEIN_9B_ID);
        assert!(d.capabilities.supports_guidance);
        // klein runs a classifier-free negative pass when guidance > 1.
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.requires_sigma_shift);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.supports_lora);
        assert!(!d.capabilities.supports_kv_cache);
        assert!(d.capabilities.supported_quants.is_empty());
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn dev_registers_and_advertises_embedded_guidance_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(FLUX2_DEV_ID, &spec).expect("flux2_dev is registered");
        assert_eq!(g.descriptor().id, FLUX2_DEV_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
        let d = descriptor_dev();
        assert!(d.capabilities.supports_guidance);
        // dev is guidance-distilled (embedded scalar) — no negative pass, no true-CFG, not mac-only.
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(d.capabilities.requires_sigma_shift);
        // dev advertises Q4/Q8 (CPU-stage → quantize-onto-GPU); klein advertises none.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(descriptor_klein().capabilities.supported_quants.is_empty());
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry::load(FLUX2_KLEIN_9B_ID, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
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
            load_klein(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load_klein(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        // klein has no candle quant path — on-the-fly quant is rejected.
        let klein_quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(matches!(
            load_klein(&klein_quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        // dev DOES accept Q4/Q8 (CPU-stage → quantize-onto-GPU); the generator builds lazily, so this
        // succeeds without touching the (nonexistent) weights.
        let dev_quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(load_dev(&dev_quant).is_ok());
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/flux2.safetensors".into()));
        let err = load_klein(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
