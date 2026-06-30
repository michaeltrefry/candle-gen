//! The candle **Krea 2 LoRA/LoKr trainer** (sc-7577) — the candle twin of `mlx-gen-krea`'s
//! `KreaRawTrainer`, implementing the backend-neutral [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer)
//! with `backend = "candle"` and reusing the shared [`candle_gen::train`] harness the SDXL/Z-Image
//! stories established. It trains on the **Krea-2-Raw** 12B base (the undistilled checkpoint; sc-7576)
//! and the adapter cross-applies to **Krea-2-Turbo** at inference by the family-match policy
//! (`baseModel: krea_2_raw`, `family: krea_2`).
//!
//! Since sc-7787 the cache → loop → save scaffolding lives in the shared single-model flow-match
//! driver ([`candle_gen::train::flow_match`]); this module supplies the Krea-specific hooks via
//! [`FlowMatchTrainer`] — caching, DiT construction, the parity-critical [`compute_loss_grads`], and the
//! provenance-stamping [`save`](FlowMatchTrainer::save).
//!
//! ## Cache → loop → save, on the flow-match objective
//!
//!  1. **Cache** — for each captioned image: decode/crop/resize to a VAE-input tensor
//!     ([`load_image_tensor`]), encode the **deterministic latent mean** through the Qwen-Image
//!     [`QwenVaeEncoder`] (the `(mean − latents_mean)/latents_std` the DiT consumes — `encode` already
//!     skips the `DiagonalGaussian` draw), and encode the caption through the Qwen3-VL-4B text encoder
//!     with the *exact* tokenizer + select-layer stack inference uses → `(L, num_text_layers,
//!     text_hidden)`. The VAE encoder + text encoder are dropped after caching.
//!  2. **Loop** (driver-owned) — sample a flow-match `σ ∈ [1e-3, 1−1e-3]`
//!     ([`sample_unit_timestep`](candle_gen::train::flow_match::sample_unit_timestep)), form
//!     `x_t = (1−σ)·x0 + σ·noise`, predict the velocity through the vendored trainable DiT
//!     ([`KreaTrainDit`]) at timestep `σ` (the raw flow time the DiT's `temb` scales ×1000 — the
//!     [`TimestepConvention::Sigma`](candle_gen::gen_core::sampling) inference uses), and regress it
//!     toward `noise − x0`.
//!  3. **Save** — a PEFT `.safetensors` (`save_lora_peft` with the DiT's **bare** key prefix /
//!     `save_lokr`) stamped with `baseModel`/`family` provenance, the on-disk format the Turbo
//!     inference-side merge (sc-7578) reads back.
//!
//! **Velocity sign.** Krea's inference pipeline consumes the **raw** DiT velocity (`x + v·Δσ`,
//! [`crate::pipeline`]) — unlike Z-Image it does not negate — so [`KreaTrainDit::forward`] returns the
//! raw velocity and the trainer regresses it toward `noise − x0` directly (the Lens convention). The
//! timestep fed to the DiT is the raw `σ` (NOT `1−σ`), matching the inference `TimestepConvention::Sigma`
//! — which (with the absence of a pre-main adapter stitch) is why [`compute_loss_grads`] stays local.
//!
//! **The eager-`Var` simplification** (inherited from the SDXL/Z-Image harness): the adapter factors
//! are storage-sharing `Var`s installed once; each forward re-reads the current factor storage and
//! `loss.backward()` attributes grads straight to the `Var`s.
//!
//! **Gradient checkpointing** (`config.gradient_checkpointing`) routes the backward through
//! [`checkpointed_backward`] over the DiT's single-stream `blocks`. Because the default surface is the
//! 28 blocks' attention, **every** adapter lives in that checkpointed stack — there is no
//! retained-pre-main adapter to stitch (the Z-Image complication), so the frozen front-end is simply
//! run once and detached at the joint-sequence boundary. Both paths yield the same grads (the
//! `dense_and_checkpoint_grads_match` gate pins this).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, IndexOp, Tensor, Var};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, Image, LoadSpec, Modality, Progress, WeightsSource};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{
    self, run_flow_match_training, validate_flow_match_request, velocity_loss, FlowMatchTrainer,
    SamplePlan,
};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::train::lora::LoraSet;
use candle_gen::{CandleError, Result};

use candle_gen_qwen_image::vae::{QwenVae, QwenVaeEncoder};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::config::Krea2Config;
use crate::loader::Weights;
use crate::schedule::{turbo_sigmas, TURBO_MU};
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::tokenizer::KreaTokenizer;
use crate::train_dit::{KreaTrainDit, KREA_ATTN_TARGETS};

/// VAE spatial downscale (the latent is image/8 per side) and latent channel count — the Turbo
/// inference constants ([`crate::pipeline`]) the preview-sample render mirrors.
const SPATIAL_SCALE: u32 = 8;
const LATENT_CHANNELS: usize = 16;

/// Max preview prompts pre-encoded + rendered per sample cadence (sc-8650). Matches the
/// `SAMPLE_PROMPT_CAP` the shared preview contract documents.
const SAMPLE_PROMPT_CAP: usize = 4;

/// Registry id for the trainable Krea 2 **Raw** base (the undistilled 12B checkpoint LoRAs train on),
/// distinct from the `krea_2_turbo` inference id — mirrors the MLX trainer (sc-7577).
pub const KREA_2_RAW_ID: &str = "krea_2_raw";

/// Error-message prefix shared by [`validate_flow_match_request`] and the driver's `no usable dataset
/// items` guard.
const LABEL: &str = "krea trainer";

/// Max prompt tokens the Qwen3-VL RoPE table is sized for during caption caching (matches the pipeline).
const MAX_TEXT_TOKENS: usize = 1024;

/// `(x_t, target, timestep)` for one sample at flow-match `σ`: delegates the latent mix
/// (`x_t = (1−σ)·x0 + σ·noise`, `target = noise − x0`) to the shared
/// [`flow_match::build_batch`](candle_gen::train::flow_match::build_batch) and appends Krea's raw-σ
/// timestep convention `timestep = σ` (the inference `TimestepConvention::Sigma`, NOT Z-Image's `1 − σ`).
fn build_batch(x0: &Tensor, noise: &Tensor, sigma: f32) -> Result<(Tensor, Tensor, f32)> {
    let (x_t, target) = flow_match::build_batch(x0, noise, sigma as f64)?;
    Ok((x_t, target, sigma))
}

/// One micro-step's forward+backward over the installed adapter `Var`s: build the noised latent at
/// `sigma`, predict the **raw** velocity through the trainable DiT, regress it toward `noise − x0`, and
/// return `(loss, grads)` keyed by the adapter `Var`s. A free function so the parity tests can drive it
/// against a tiny DiT.
///
/// `x0`/`noise` are the cached `[1, 16, h, w]` clean latent + the per-step noise (f32); `cap` is the
/// cached `(L, num_text_layers, text_hidden)` caption stack — unsqueezed to the DiT's batched `context`.
///
/// `use_checkpoint` selects the gradient-checkpointed backward over the dense `loss.backward()`. Because
/// every adapter lives in the checkpointed `blocks` stack, the frozen front-end is run once via
/// [`KreaTrainDit::forward_pre_main`] and detached at the joint-sequence boundary — no pre-main stitch.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &KreaTrainDit,
    lora_vars: &[Var],
    x0: &Tensor,
    cap: &Tensor,
    sigma: f32,
    noise: &Tensor,
    mae: bool,
    compute_dtype: DType,
    use_checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let device = x0.device();
    let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
    let x_t = x_t.to_dtype(compute_dtype)?;
    let context = cap.unsqueeze(0)?; // (L, n, d) -> (1, L, n, d)
    let t = Tensor::from_vec(vec![timestep], (1,), device)?;

    if use_checkpoint {
        let (combined, ctx) = dit.forward_pre_main(&x_t, &t, &context)?;
        let mut segs = dit.main_layer_segments(&ctx);
        // Final segment: the post-main head + the (raw-velocity) flow-match regression -> [loss].
        let target_owned = target.clone();
        let ctx_ref = &ctx;
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], ctx_ref)?;
            Ok(vec![velocity_loss(&v, &target_owned, mae)?])
        }));
        // Seed the checkpointed chain with the detached joint-sequence boundary (no adapters upstream).
        let combined_d = combined.detach();
        let (loss_val, grads) =
            checkpointed_backward(&segs, std::slice::from_ref(&combined_d), lora_vars)?;
        Ok((loss_val, grads))
    } else {
        // Dense backward: one monolithic `loss.backward()`. Raw DiT velocity (no negation).
        let v = dit.forward(&x_t, &t, &context)?;
        let loss = velocity_loss(&v, &target, mae)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        Ok((loss_val, grads))
    }
}

/// Tokenize `caption` + encode it through the Qwen3-VL text encoder to the cached conditioning stack
/// `(L, num_text_layers, text_hidden)` at f32 — the exact tokenizer + select-layer stack the inference
/// [`crate::pipeline`] uses (parity), minus the device-dtype cast (caching keeps f32).
fn encode_caption(tok: &KreaTokenizer, te: &KreaTextEncoder, caption: &str) -> Result<Tensor> {
    let ids = tok.encode_prompt(caption)?;
    let enc = te.forward(&ids)?; // (1, L, num_text_layers, text_hidden)
    Ok(enc.squeeze(0)?.to_dtype(DType::F32)?)
}

/// The Krea preview-sample render state (sc-8650) — everything [`KreaTrainer::render_sample`] needs to
/// run the family's CFG-free Turbo denoise on the **in-progress** trainable DiT, built once in
/// [`KreaTrainer::cache`] while the text encoder is still resident:
///
///  * `contexts` — the per-prompt pre-encoded Qwen3-VL conditioning, each `(L, num_text_layers,
///    text_hidden)` at f32 (exactly what [`encode_caption`] returns and [`KreaTrainDit::forward`]
///    consumes once unsqueezed to a batch axis), 1:1 with [`SamplePlan::prompts`].
///  * `vae` — the resident Qwen-Image VAE **decoder** (`Arc` as inference holds it); the cache pass
///    loads only the encoder, so the decoder is loaded here for the preview path.
///  * `edge` — the square training-resolution edge (`bucket_resolution(cfg.resolution)`, the same edge
///    the cached latents use) the seeded preview noise is shaped at.
pub struct KreaSampleState {
    contexts: Vec<Tensor>,
    vae: Arc<QwenVae>,
    edge: u32,
}

/// Seeded initial Gaussian latent noise `[1, 16, edge/8, edge/8]` (f32) for a preview render — the
/// training-side twin of [`crate::pipeline`]'s `init_noise`, square at the bucketed training `edge`
/// (sc-8650). Deterministic launch-portable CPU RNG (sc-3673), mirroring the inference path.
fn sample_noise_latent(edge: u32, seed: u64, device: &Device) -> Result<Tensor> {
    let lat = (edge / SPATIAL_SCALE) as usize;
    let n = LATENT_CHANNELS * lat * lat;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Ok(Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat, lat), &Device::Cpu)?.to_device(device)?)
}

/// VAE-decode a final preview latent `[1, 16, H/8, W/8]` → RGB8 [`Image`] — the training-side twin of
/// [`crate::pipeline`]'s `decode` (`QwenVae::decode` de-normalizes internally and returns `[1, 3, H, W]`
/// in `[-1, 1]`; the `(x+1)·127.5` is the reference `clamp(-1,1)·0.5 + 0.5` denormalize) (sc-8650).
fn decode_preview(vae: &QwenVae, lat: &Tensor) -> Result<Image> {
    let decoded = vae.decode(lat)?.to_dtype(DType::F32)?; // [1, 3, H, W] in [-1, 1]
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "krea: preview decode expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Identity + capabilities of the candle Krea trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: KREA_2_RAW_ID,
        family: "krea_2",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// A loaded candle Krea trainer. Loading is **lazy** (no file I/O — mirrors the SDXL/Z-Image trainers):
/// the heavy VAE encoder / text encoder / DiT are built inside [`train`](Trainer::train).
pub struct KreaTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Krea trainer from a [`LoadSpec`] whose `weights` is the Krea-2-Raw
/// snapshot directory (`tokenizer/ text_encoder/ transformer/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(
                "krea_2_raw trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    Ok(Box::new(KreaTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

// Link-time self-registration into gen-core's trainer registry (parallel to the generator's
// registration in `lib.rs`). Kept linked by `crate::force_link`. `register_trainer!` bridges the
// crate's rich `Result` into the registry's `gen_core::Result` via `Into::into`.
candle_gen::register_trainer! { trainer_descriptor => load_trainer }

impl Trainer for KreaTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_flow_match_request(req, LABEL).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        validate_flow_match_request(req, LABEL)?;
        run_flow_match_training(self, req, on_progress).map_err(Into::into)
    }
}

impl FlowMatchTrainer for KreaTrainer {
    type Dit = KreaTrainDit;
    /// `(x0 latent [1,16,h,w], caption stack (L, num_text_layers, text_hidden))`, both f32.
    type Cached = (Tensor, Tensor);
    type Aux = ();
    /// Preview-sample render state: per-prompt pre-encoded conditioning + resident VAE decoder + the
    /// training-resolution edge (sc-8650).
    type SampleState = KreaSampleState;
    const LABEL: &'static str = LABEL;

    fn device(&self) -> &Device {
        &self.device
    }

    fn default_targets(&self) -> &'static [&'static str] {
        &KREA_ATTN_TARGETS
    }

    fn cache(
        &self,
        req: &TrainingRequest,
        device: &Device,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<(Tensor, Tensor)>, (), SamplePlan<KreaSampleState>)> {
        let edge = bucket_resolution(req.config.resolution);
        let vae_encoder = QwenVaeEncoder::new(flow_match::component_vb(
            &self.root,
            "vae",
            device,
            DType::F32,
            LABEL,
        )?)?;
        let tokenizer = KreaTokenizer::from_snapshot(&self.root, device)?;
        let te_cfg = KreaTeConfig::from_snapshot(&self.root)?;
        let te_w = Weights::from_dir(&self.root.join("text_encoder"), device, DType::F32)?;
        let text_encoder =
            KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;

        let total = req.items.len() as u32;
        let mut cache: Vec<(Tensor, Tensor)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = load_image_tensor(&item.image_path, edge, device)?;
            let x0 = vae_encoder.encode(&img)?; // (1, 16, edge/8, edge/8), already normalized
            let cap = encode_caption(&tokenizer, &text_encoder, &item.caption)?;
            cache.push((x0, cap));
        }

        // Preview samples (sc-8650) — while the text encoder is STILL resident, pre-encode up to
        // `SAMPLE_PROMPT_CAP` of the configured prompts with the same `encode_caption` the cache loop
        // uses (train/infer conditioning parity), and load a resident VAE *decoder* (the cache pass
        // loaded only the encoder). The driver renders these from the in-progress adapter each cadence.
        let sample_plan = if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
            let prompts: Vec<String> = req
                .config
                .sample_prompts
                .iter()
                .take(SAMPLE_PROMPT_CAP)
                .cloned()
                .collect();
            let contexts = prompts
                .iter()
                .map(|p| encode_caption(&tokenizer, &text_encoder, p))
                .collect::<Result<Vec<Tensor>>>()?;
            let vae = Arc::new(crate::vae::load_vae(&self.root, device)?);
            SamplePlan {
                prompts,
                state: Some(KreaSampleState {
                    contexts,
                    vae,
                    edge,
                }),
            }
        } else {
            SamplePlan::disabled()
        };

        // Encoders are dead weight once cached + previews pre-encoded — drop them before the DiT
        // (working set) loads. The resident VAE *decoder* lives on in the sample plan's state.
        drop(text_encoder);
        drop(vae_encoder);
        Ok((cache, (), sample_plan))
    }

    fn build_dit(&self, req: &TrainingRequest, device: &Device) -> Result<KreaTrainDit> {
        let compute_dtype = flow_match::parse_compute_dtype(&req.config.train_dtype);
        let dit_cfg = Krea2Config::from_snapshot(&self.root)?;
        let dit_w = Weights::from_dir(&self.root.join("transformer"), device, compute_dtype)?;
        Ok(KreaTrainDit::load(&dit_w, &dit_cfg)?)
    }

    fn micro_step(
        &self,
        dit: &KreaTrainDit,
        vars: &[Var],
        cached: &(Tensor, Tensor),
        _aux: &(),
        cfg: &TrainingConfig,
        step: u32,
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        let (x0, cap) = cached;
        let sigma = flow_match::sample_unit_timestep(
            &cfg.timestep_type,
            &cfg.timestep_bias,
            flow_match::timestep_seed(cfg.seed, step),
        );
        let noise =
            flow_match::sample_noise(x0.dims(), flow_match::noise_seed(cfg.seed, step), device)?;
        compute_loss_grads(
            dit,
            vars,
            x0,
            cap,
            sigma,
            &noise,
            flow_match::is_mae(cfg),
            flow_match::parse_compute_dtype(&cfg.train_dtype),
            cfg.gradient_checkpointing,
        )
    }

    /// Render preview prompt `index` from the **in-progress** trainable DiT (sc-8650) — the training-side
    /// mirror of [`crate::pipeline`]'s CFG-free Turbo `render`. Krea consumes the **raw** velocity at
    /// timestep `σ` ([`TimestepConvention::Sigma`]), so the denoise closure feeds `dit.forward` the bare
    /// σ and returns the raw velocity (no negation, no CFG — the TDM-distilled student is guidance-free,
    /// so `sample_guidance_scale` is ignored). `TrainingConfig` carries no per-run sampler/scheduler
    /// knob, so the native exponential-`mu` Turbo schedule is used (the byte-exact inference default).
    /// Best-effort: any error here is logged + skipped by the driver, never aborting the run.
    fn render_sample(
        &self,
        dit: &KreaTrainDit,
        state: &KreaSampleState,
        index: usize,
        cfg: &TrainingConfig,
        seed: u64,
    ) -> Result<Image> {
        let device = &self.device;
        let steps = (cfg.sample_steps.max(1)) as usize;
        // Native exponential-mu Turbo sigmas — the byte-exact default; `None` scheduler (no per-run knob
        // on `TrainingConfig`) resolves to the native schedule.
        let sigmas =
            candle_gen::resolve_flow_schedule(None, TURBO_MU as f32, steps, &turbo_sigmas(steps));

        let noise = sample_noise_latent(state.edge, seed, device)?;
        // The DiT's `text_fusion` consumes a batched `(1, L, n, d)` context — the cached
        // `encode_caption` output is `(L, n, d)`, so unsqueeze the batch axis exactly as
        // `compute_loss_grads` does before its own forward.
        let context = state.contexts[index].unsqueeze(0)?;

        // A preview need not honor cancel mid-denoise — a fresh never-cancel flag (the trainer's
        // `req.cancel` is only available in `cache`, not here).
        let cancel = CancelFlag::new();
        let mut on_progress = |_: Progress| {};
        let lat = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &cancel,
            &mut on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let v = dit.forward(x, &t, &context)?;
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;
        decode_preview(&state.vae, &lat)
    }

    /// Stamp `baseModel`/`family` provenance into the adapter header so the Turbo cross-apply policy
    /// (family-match) can validate it.
    fn save(&self, set: &LoraSet, path: &Path) -> Result<()> {
        let mut meta = HashMap::new();
        meta.insert("baseModel".to_string(), KREA_2_RAW_ID.to_string());
        meta.insert("family".to_string(), "krea_2".to_string());
        flow_match::save_adapter(set, &meta, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::optim::{clip_grad_norm, TrainOptimizer};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The smallest valid Krea DiT config: 1 single-stream block, 1 layerwise + 1 refiner text block,
    /// head_dim 16 (= sum [4,6,6]), hidden 32, GQA 2/1.
    fn tiny_cfg() -> Krea2Config {
        Krea2Config {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 32,
            num_attention_heads: 2,
            num_kv_heads: 1,
            attention_head_dim: 16,
            num_layers: 1,
            intermediate_size: 16,
            norm_eps: 1e-5,
            axes_dims_rope: [4, 6, 6],
            rope_theta: 1000.0,
            timestep_embed_dim: 8,
            num_text_layers: 2,
            num_layerwise_text_blocks: 1,
            num_refiner_text_blocks: 1,
            text_hidden_dim: 32,
            text_intermediate_size: 16,
            text_num_attention_heads: 2,
            text_num_kv_heads: 2,
        }
    }

    fn rnd(shape: &[usize]) -> Tensor {
        Tensor::randn(0f32, 0.05f32, shape, &Device::Cpu).unwrap()
    }

    fn lin(t: &mut HashMap<String, Tensor>, name: &str, out: usize, inn: usize, bias: bool) {
        t.insert(format!("{name}.weight"), rnd(&[out, inn]));
        if bias {
            t.insert(format!("{name}.bias"), rnd(&[out]));
        }
    }

    /// Push one gated-attention + SwiGLU block's tensors under `prefix` (shared shape between the text
    /// fusion and single-stream blocks, parameterized by widths).
    #[allow(clippy::too_many_arguments)]
    fn attn_ffn(
        t: &mut HashMap<String, Tensor>,
        prefix: &str,
        hidden: usize,
        heads: usize,
        kv: usize,
        hd: usize,
        inter: usize,
    ) {
        t.insert(format!("{prefix}.norm1.weight"), rnd(&[hidden]));
        t.insert(format!("{prefix}.norm2.weight"), rnd(&[hidden]));
        lin(t, &format!("{prefix}.attn.to_q"), heads * hd, hidden, false);
        lin(t, &format!("{prefix}.attn.to_k"), kv * hd, hidden, false);
        lin(t, &format!("{prefix}.attn.to_v"), kv * hd, hidden, false);
        lin(t, &format!("{prefix}.attn.to_gate"), hidden, hidden, false);
        lin(t, &format!("{prefix}.attn.to_out.0"), hidden, hidden, false);
        t.insert(format!("{prefix}.attn.norm_q.weight"), rnd(&[hd]));
        t.insert(format!("{prefix}.attn.norm_k.weight"), rnd(&[hd]));
        lin(t, &format!("{prefix}.ff.gate"), inter, hidden, false);
        lin(t, &format!("{prefix}.ff.up"), inter, hidden, false);
        lin(t, &format!("{prefix}.ff.down"), hidden, inter, false);
    }

    /// Serialize a tiny Krea transformer to a temp `.safetensors` and load it as a [`KreaTrainDit`].
    /// Returns `(dit, cfg, temp_path)` — the caller drops the file when done.
    fn tiny_dit() -> (KreaTrainDit, Krea2Config, PathBuf) {
        let c = tiny_cfg();
        let (hidden, heads, kv, hd) = (
            c.hidden_size,
            c.num_attention_heads,
            c.num_kv_heads,
            c.attention_head_dim,
        );
        let (th, theads, tkv) = (
            c.text_hidden_dim,
            c.text_num_attention_heads,
            c.text_num_kv_heads,
        );
        let mut t: HashMap<String, Tensor> = HashMap::new();

        lin(&mut t, "img_in", hidden, c.in_channels, true);
        lin(
            &mut t,
            "time_embed.linear_1",
            hidden,
            c.timestep_embed_dim,
            true,
        );
        lin(&mut t, "time_embed.linear_2", hidden, hidden, true);
        lin(&mut t, "time_mod_proj", 6 * hidden, hidden, true);
        t.insert("txt_in.norm.weight".into(), rnd(&[th]));
        lin(&mut t, "txt_in.linear_1", hidden, th, true);
        lin(&mut t, "txt_in.linear_2", hidden, hidden, true);
        for i in 0..c.num_layerwise_text_blocks {
            attn_ffn(
                &mut t,
                &format!("text_fusion.layerwise_blocks.{i}"),
                th,
                theads,
                tkv,
                hd,
                c.text_intermediate_size,
            );
        }
        for i in 0..c.num_refiner_text_blocks {
            attn_ffn(
                &mut t,
                &format!("text_fusion.refiner_blocks.{i}"),
                th,
                theads,
                tkv,
                hd,
                c.text_intermediate_size,
            );
        }
        lin(&mut t, "text_fusion.projector", 1, c.num_text_layers, false);
        for i in 0..c.num_layers {
            let p = format!("transformer_blocks.{i}");
            t.insert(format!("{p}.scale_shift_table"), rnd(&[6, hidden]));
            attn_ffn(&mut t, &p, hidden, heads, kv, hd, c.intermediate_size);
        }
        t.insert("final_layer.scale_shift_table".into(), rnd(&[2, hidden]));
        t.insert("final_layer.norm.weight".into(), rnd(&[hidden]));
        lin(&mut t, "final_layer.linear", c.in_channels, hidden, true);

        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "krea_tiny_{}_{}.safetensors",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        candle_gen::candle_core::safetensors::save(&t, &path).unwrap();
        let w = Weights::from_file(&path, &Device::Cpu, DType::F32).unwrap();
        let dit = KreaTrainDit::load(&w, &c).unwrap();
        (dit, c, path)
    }

    /// `(x0, cap, noise)` for the tiny DiT: a `[1, latent_ch, 4, 4]` latent + matching noise, and a
    /// `[3, num_text_layers, text_hidden]` caption stack.
    fn tiny_batch(c: &Krea2Config) -> (Tensor, Tensor, Tensor) {
        let latent_ch = c.in_channels / (c.patch_size * c.patch_size);
        let x0 = rnd(&[1, latent_ch, 4, 4]);
        let cap = rnd(&[3, c.num_text_layers, c.text_hidden_dim]);
        let noise = rnd(&[1, latent_ch, 4, 4]);
        (x0, cap, noise)
    }

    /// `build_batch`: `x_t = (1−σ)x0 + σ·noise`, `target = noise − x0`, `timestep = σ` (the Krea raw-σ
    /// convention — NOT the Z-Image `1−σ` — layered over the shared `flow_match::build_batch`).
    #[test]
    fn build_batch_math() {
        let dev = Device::Cpu;
        let x0 = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target, timestep) = build_batch(&x0, &noise, 0.25).unwrap();
        // x_t = 0.75·[2,4] + 0.25·[1,0] = [1.75, 3.0]; target = [1-2, 0-4] = [-1, -4]; t = σ = 0.25.
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
        assert!((timestep - 0.25).abs() < 1e-6);
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var` (backprop reaches the
    /// LoRA seam through the composable softmax/RMSNorm of the single-stream blocks + final layer).
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let (mut dit, c, path) = tiny_dit();
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off zero so both A and B grads are nonzero (a no-op-init adapter zeros A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, cap, noise) = tiny_batch(&c);
        let (loss, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        for (i, v) in set.vars.iter().enumerate() {
            let g = grads
                .get(v.as_tensor())
                .unwrap_or_else(|| panic!("adapter var {i} has no gradient"));
            assert!(
                g.flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|x| x.is_finite()),
                "var {i} gradient has non-finite entries"
            );
        }
        // The LoRA set covers every attention projection (4) in the single block: 4 targets × 2 factors.
        assert_eq!(
            set.vars.len(),
            4 * c.num_layers * 2,
            "two factors per attention target"
        );
        let _ = std::fs::remove_file(path);
    }

    /// The correctness gate for the `gradient_checkpointing` lever: the checkpointed backward over the
    /// single-stream `blocks` must reproduce the dense `loss.backward()` grads (mod float reassociation).
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let (mut dit, c, path) = tiny_dit();
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, cap, noise) = tiny_batch(&c);

        let (loss_d, g_d) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        let (loss_c, g_c) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            DType::F32,
            true,
        )
        .unwrap();

        assert!(
            (loss_d - loss_c).abs() < 1e-4,
            "loss: dense {loss_d} vs checkpoint {loss_c}"
        );
        let grad_vec = |g: &GradStore, v: &Var| {
            g.get(v.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let mut saw_nonzero = false;
        for (idx, v) in set.vars.iter().enumerate() {
            assert!(
                g_d.get(v.as_tensor()).is_some() && g_c.get(v.as_tensor()).is_some(),
                "var {idx} missing a gradient (dense or checkpoint)"
            );
            let a = grad_vec(&g_d, v);
            let b = grad_vec(&g_c, v);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "grad mismatch for var {idx} (dense {x} vs checkpoint {y})"
                );
                if x.abs() > 1e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected nonzero adapter grads to compare");
        let _ = std::fs::remove_file(path);
    }

    /// A few optimizer steps over the tiny DiT lower the loss on the same fixed batch — the step
    /// actually descends the flow-match objective, end to end through the harness.
    #[test]
    fn optimizer_steps_descend() {
        let dev = Device::Cpu;
        let (mut dit, c, path) = tiny_dit();
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, cap, noise) = tiny_batch(&c);
        let mut opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        let (loss0, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        let mut grads = grads;
        for _ in 0..6 {
            clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            let (_l, g) = compute_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                DType::F32,
                false,
            )
            .unwrap();
            grads = g;
        }
        let (loss1, _) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        assert!(
            loss1 < loss0,
            "steps on a fixed batch should lower the loss: {loss0} -> {loss1}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// The trainer self-registers and resolves through gen-core's trainer registry as the candle Krea
    /// trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(KREA_2_RAW_ID, &spec)
            .expect("candle krea trainer is registered");
        assert_eq!(t.descriptor().id, KREA_2_RAW_ID);
        assert_eq!(t.descriptor().family, "krea_2");
        assert_eq!(t.descriptor().backend, "candle");
        assert!(t.descriptor().supports_lora);
        assert!(t.descriptor().supports_lokr);
    }

    /// `validate` rejects an empty dataset, zero rank/steps, an unsupported optimizer, and an
    /// unrecognized timestep/loss knob — before any load (now via the shared
    /// `flow_match::validate_flow_match_request`).
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        use candle_gen::gen_core::train::TrainingItem;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(KREA_2_RAW_ID, &spec).unwrap();

        let item = TrainingItem {
            image_path: "/img.png".into(),
            caption: "x".into(),
        };
        let base = TrainingRequest {
            items: vec![item],
            config: TrainingConfig::default(),
            output_dir: "/out".into(),
            file_name: "a.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        assert!(t.validate(&base).is_ok());

        let bad = |mutate: &dyn Fn(&mut TrainingRequest)| {
            let mut r = base.clone();
            mutate(&mut r);
            assert!(t.validate(&r).is_err());
        };
        bad(&|r| r.items.clear());
        bad(&|r| r.config.rank = 0);
        bad(&|r| r.config.steps = 0);
        bad(&|r| r.config.optimizer = "lion".into());
        bad(&|r| r.config.timestep_type = "bogus".into());
        bad(&|r| r.config.timestep_bias = "bogus".into());
        bad(&|r| r.config.loss_type = "huber".into());
        // A `_`/case-normalized spelling of a recognized value is accepted.
        let mut ok = base.clone();
        ok.config.timestep_type = "Weighted".into();
        ok.config.timestep_bias = "high-noise".into();
        assert!(t.validate(&ok).is_ok());
    }
}
