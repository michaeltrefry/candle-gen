//! The candle **Krea 2 LoRA/LoKr trainer** (sc-7577) — the candle twin of `mlx-gen-krea`'s
//! `KreaRawTrainer`, implementing the backend-neutral [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer)
//! with `backend = "candle"` and reusing the shared [`candle_gen::train`] harness the SDXL/Z-Image
//! stories established. It trains on the **Krea-2-Raw** 12B base (the undistilled checkpoint; sc-7576)
//! and the adapter cross-applies to **Krea-2-Turbo** at inference by the family-match policy
//! (`baseModel: krea_2_raw`, `family: krea_2`).
//!
//! ## Cache → loop → save, on the flow-match objective
//!
//!  1. **Cache** — for each captioned image: decode/crop/resize to a VAE-input tensor
//!     ([`load_image_tensor`]), encode the **deterministic latent mean** through the Qwen-Image
//!     [`QwenVaeEncoder`] (the `(mean − latents_mean)/latents_std` the DiT consumes — `encode` already
//!     skips the `DiagonalGaussian` draw), and encode the caption through the Qwen3-VL-4B text encoder
//!     with the *exact* tokenizer + select-layer stack inference uses → `(L, num_text_layers,
//!     text_hidden)`. The VAE encoder + text encoder are dropped after caching.
//!  2. **Loop** — sample a flow-match `σ ∈ [1e-3, 1−1e-3]` ([`sample_sigma`]), form
//!     `x_t = (1−σ)·x0 + σ·noise`, predict the velocity through the vendored trainable DiT
//!     ([`KreaTrainDit`]) at timestep `σ` (the raw flow time the DiT's `temb` scales ×1000 — the
//!     [`TimestepConvention::Sigma`](candle_gen::gen_core::sampling) inference uses), and regress it
//!     toward `noise − x0`. Gradient accumulation, the LR schedule, and grad-norm clipping reuse the
//!     harness.
//!  3. **Save** — a PEFT `.safetensors` (`save_lora_peft` with the DiT's **bare** key prefix /
//!     `save_lokr`), the on-disk format the eventual Turbo inference-side merge (sc-7578) reads back.
//!
//! **Velocity sign.** Krea's inference pipeline consumes the **raw** DiT velocity (`x + v·Δσ`,
//! [`crate::pipeline`]) — unlike Z-Image it does not negate — so [`KreaTrainDit::forward`] returns the
//! raw velocity and the trainer regresses it toward `noise − x0` directly (the Lens convention). The
//! timestep fed to the DiT is the raw `σ` (NOT `1−σ`), matching the inference `TimestepConvention::Sigma`.
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

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use candle_gen::gen_core::train::{
    NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, WeightsSource};
use candle_gen::train::checkpoint::{checkpoint_filename, file_stem};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::train::lora::{
    build_lokr_targets, build_lora_targets, save_lokr, save_lora_peft, AdapterKind, LoraSet,
};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen::train::schedule::{lr_multiplier, schedule_updates};
use candle_gen::{CandleError, Result};

use candle_gen_qwen_image::vae::QwenVaeEncoder;

use crate::config::Krea2Config;
use crate::loader::Weights;
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::tokenizer::KreaTokenizer;
use crate::train_dit::{KreaTrainDit, KREA_ATTN_TARGETS};

/// Registry id for the trainable Krea 2 **Raw** base (the undistilled 12B checkpoint LoRAs train on),
/// distinct from the `krea_2_turbo` inference id — mirrors the MLX trainer (sc-7577).
pub const KREA_2_RAW_ID: &str = "krea_2_raw";

/// Max prompt tokens the Qwen3-VL RoPE table is sized for during caption caching (matches the pipeline).
const MAX_TEXT_TOKENS: usize = 1024;

/// Recognized `timestep_type` values — the noise-schedule samplers [`sample_sigma`] branches on, plus
/// the `sigmoid` default. Validation rejects anything else (the MLX F-041 guard).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values — the high/low-noise tilts plus the neutral default.
const TIMESTEP_BIASES: [&str; 9] = [
    "balanced",
    "none",
    "neutral",
    "high",
    "high_noise",
    "favor_high_noise",
    "low",
    "low_noise",
    "favor_low_noise",
];
/// Recognized `loss_type` values — `mae`/`l1` select MAE, `mse`/`l2` the MSE default.
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// `"bf16"`/`"bfloat16"` → [`DType::BF16`] (the Krea DiT's native compute dtype); anything else →
/// [`DType::F32`] (the gen-core contract: unrecognized = f32). Adapter factors / loss / grads stay f32.
fn parse_compute_dtype(s: &str) -> DType {
    let t = s.trim();
    if t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16") {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Normalize a free-form config string (trim, lowercase, `-`/space → `_`) so validation accepts exactly
/// the spellings [`sample_sigma`] would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Sample a normalized flow-match timestep (interpolation coefficient) `σ ∈ [1e-3, 1−1e-3]` — the same
/// `sample_training_timestep` port the Z-Image trainer uses: `sigmoid(randn)` by default, `uniform` for
/// linear, `(uniform + sigmoid(randn))/2` for weighted; bias `high` → `√σ`, `low` → `σ²`. Deterministic
/// in `seed` via the sc-3673 CPU `StdRng` discipline. Cross-framework numeric parity with MLX is a
/// non-goal (different RNG); per-seed determinism is what the worker relies on.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> f32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let t = match normalize_cfg(timestep_type).as_str() {
        "linear" | "uniform" => rng.random::<f32>(),
        "weighted" => {
            let base = rng.random::<f32>();
            let z: f32 = StandardNormal.sample(&mut rng);
            (base + sigmoid(z)) / 2.0
        }
        // "sigmoid" + any unrecognized value (validation rejects the latter up front).
        _ => {
            let z: f32 = StandardNormal.sample(&mut rng);
            sigmoid(z)
        }
    };
    let t = match normalize_cfg(timestep_bias).as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    t.clamp(1e-3, 1.0 - 1e-3)
}

/// `(x_t, target, timestep)` for one sample at flow-match `σ`: `x_t = (1−σ)·x0 + σ·noise`,
/// `target = noise − x0` (the velocity the **raw** DiT output trains toward — no negation, the Krea/Lens
/// convention), `timestep = σ` (the raw flow time the DiT's `temb` consumes; the inference
/// `TimestepConvention::Sigma`). All in f32.
fn build_batch(x0: &Tensor, noise: &Tensor, sigma: f32) -> Result<(Tensor, Tensor, f32)> {
    let x_t = ((x0 * (1.0 - sigma) as f64)? + (noise * sigma as f64)?)?;
    let target = (noise - x0)?;
    Ok((x_t, target, sigma))
}

/// Flow-match velocity loss in f32: `mean((v − target)²)` (MSE) or `mean|v − target|` (MAE). `v` (the
/// raw DiT velocity, in the compute dtype) is promoted to f32 so the loss/grads stay f32.
fn velocity_loss(
    v: &Tensor,
    target_f32: &Tensor,
    mae: bool,
) -> candle_gen::candle_core::Result<Tensor> {
    let diff = (v.to_dtype(DType::F32)? - target_f32)?;
    if mae {
        diff.abs()?.mean_all()
    } else {
        diff.sqr()?.mean_all()
    }
}

/// Deterministic `N(0, 1)` noise of the given shape, drawn from a seeded CPU `StdRng` then moved to
/// `device` (sc-3673 launch-portable discipline). The flow-match prior + the regression target.
fn sample_noise(shape: &[usize], seed: u64, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?.to_device(device)?)
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
    lora_vars: &[candle_gen::candle_core::Var],
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

/// Resolve the sorted `.safetensors` files in the snapshot component subdir `sub`.
fn component_files(root: &Path, sub: &str) -> Result<Vec<PathBuf>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "krea trainer: snapshot missing the {sub}/ component directory (at {})",
            root.display()
        )));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CandleError::Msg(format!("krea trainer: read {sub}/: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "krea trainer: no .safetensors in {sub}/ (at {})",
            dir.display()
        )));
    }
    Ok(files)
}

/// Build a [`VarBuilder`] over the snapshot component subdir `sub` at `dtype` (the VAE-encoder load).
fn component_vb(
    root: &Path,
    sub: &str,
    device: &Device,
    dtype: DType,
) -> Result<VarBuilder<'static>> {
    let files = component_files(root, sub)?;
    // SAFETY: mmap of read-only weight files; standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? })
}

/// Tokenize `caption` + encode it through the Qwen3-VL text encoder to the cached conditioning stack
/// `(L, num_text_layers, text_hidden)` at f32 — the exact tokenizer + select-layer stack the inference
/// [`crate::pipeline`] uses (parity), minus the device-dtype cast (caching keeps f32).
fn encode_caption(tok: &KreaTokenizer, te: &KreaTextEncoder, caption: &str) -> Result<Tensor> {
    let ids = tok.encode_prompt(caption)?;
    let enc = te.forward(&ids)?; // (1, L, num_text_layers, text_hidden)
    Ok(enc.squeeze(0)?.to_dtype(DType::F32)?)
}

/// The config's target-module suffixes (default [`KREA_ATTN_TARGETS`]).
fn resolve_target_suffixes(cfg: &TrainingConfig) -> Vec<String> {
    if cfg.lora_target_modules.is_empty() {
        KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.lora_target_modules.clone()
    }
}

/// Write the adapter as a `.safetensors`: LoRA with the DiT's **bare** dotted keys (empty prefix — the
/// SDXL `base_model.model.unet.` prefix is SDXL-specific), LoKr with bare keys + metadata. Records the
/// base-model id so the Turbo cross-apply policy (family-match) can validate provenance.
fn save_adapter(set: &LoraSet, path: &Path) -> Result<()> {
    let mut meta = HashMap::new();
    meta.insert("baseModel".to_string(), KREA_2_RAW_ID.to_string());
    meta.insert("family".to_string(), "krea_2".to_string());
    match set.kind {
        AdapterKind::Lora => save_lora_peft(set, "", &meta, path),
        AdapterKind::Lokr => save_lokr(set, &meta, path),
    }
}

/// Create the output directory, mapping the `io::Error` into the crate error.
fn create_output_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CandleError::Msg(format!("create output dir {}: {e}", dir.display())))
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

/// Registry adapter: bridge the crate's rich-`Result` [`load_trainer`] into the registry's
/// `gen_core::Result` slot.
fn load_trainer_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    load_trainer(spec).map_err(Into::into)
}

// Link-time self-registration into gen-core's trainer registry (parallel to the generator's
// `ModelRegistration` in `lib.rs`). Kept linked by `crate::force_link`.
inventory::submit! {
    gen_core::registry::TrainerRegistration { descriptor: trainer_descriptor, load: load_trainer_registered }
}

impl Trainer for KreaTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl KreaTrainer {
    /// Reject a request before any expensive load: empty dataset, zero rank/steps, unsupported
    /// optimizer, and an unrecognized `timestep_type`/`timestep_bias`/`loss_type` (the MLX F-041 guard).
    fn validate_impl(&self, req: &TrainingRequest) -> Result<()> {
        let cfg = &req.config;
        if req.items.is_empty() {
            return Err(CandleError::Msg("krea trainer: dataset is empty".into()));
        }
        if cfg.rank == 0 {
            return Err(CandleError::Msg("krea trainer: rank must be > 0".into()));
        }
        if cfg.steps == 0 {
            return Err(CandleError::Msg("krea trainer: steps must be > 0".into()));
        }
        if !TrainOptimizer::is_supported(&cfg.optimizer) {
            return Err(CandleError::Msg(format!(
                "krea trainer: optimizer '{}' is not available (supported: adamw, adam, rose, prodigy)",
                cfg.optimizer
            )));
        }
        if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
            return Err(CandleError::Msg(format!(
                "krea trainer: timestep_type '{}' is not recognized (supported: {})",
                cfg.timestep_type,
                TIMESTEP_TYPES.join(", ")
            )));
        }
        if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
            return Err(CandleError::Msg(format!(
                "krea trainer: timestep_bias '{}' is not recognized (supported: {})",
                cfg.timestep_bias,
                TIMESTEP_BIASES.join(", ")
            )));
        }
        if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
            return Err(CandleError::Msg(format!(
                "krea trainer: loss_type '{}' is not recognized (supported: {})",
                cfg.loss_type,
                LOSS_TYPES.join(", ")
            )));
        }
        Ok(())
    }

    /// The rich-`Result` body behind [`Trainer::train`].
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate_impl(req)?;
        let cfg = &req.config;
        let device = &self.device;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);
        let compute_dtype = parse_compute_dtype(&cfg.train_dtype);

        // --- load + cache: VAE latent means + Qwen3-VL caption stacks (both f32) ---
        on_progress(TrainingProgress::LoadingModel);
        let vae_encoder =
            QwenVaeEncoder::new(component_vb(&self.root, "vae", device, DType::F32)?)?;
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
        // Encoders are dead weight once cached — drop them before the DiT (working set) loads.
        drop(text_encoder);
        drop(vae_encoder);
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            return Err(CandleError::Msg(
                "krea trainer: no usable dataset items".into(),
            ));
        }

        // --- build the trainable DiT + install adapters ---
        let dit_cfg = Krea2Config::from_snapshot(&self.root)?;
        let dit_w = Weights::from_dir(&self.root.join("transformer"), device, compute_dtype)?;
        let mut dit = KreaTrainDit::load(&dit_w, &dit_cfg)?;
        let suffixes = resolve_target_suffixes(cfg);
        let lora_set = match cfg.network_type {
            NetworkType::Lora => {
                build_lora_targets(&mut dit, &suffixes, cfg.rank, cfg.alpha, cfg.seed, device)?
            }
            NetworkType::Lokr => build_lokr_targets(
                &mut dit,
                &suffixes,
                cfg.rank,
                cfg.alpha,
                cfg.decompose_factor,
                cfg.seed,
                device,
            )?,
        };
        let use_checkpoint = cfg.gradient_checkpointing;

        // --- optimizer + schedule ---
        let mae = matches!(normalize_cfg(&cfg.loss_type).as_str(), "mae" | "l1");
        // AdamW with wd=0 ≡ Adam, so the one optimizer covers both choices.
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut opt = TrainOptimizer::from_config(
            &cfg.optimizer,
            lora_set.vars.clone(),
            cfg.learning_rate,
            weight_decay,
        )?;
        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let stem = file_stem(&req.file_name).to_string();

        // --- train loop ---
        let mut accumulated: Option<GradStore> = None;
        let mut update_idx = 0u32;
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, cap) = &cache[((step - 1) as usize) % cache.len()];
            let sigma = sample_sigma(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
            );
            let noise = sample_noise(
                x0.dims(),
                cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                device,
            )?;
            let (loss, grads) = compute_loss_grads(
                &dit,
                &lora_set.vars,
                x0,
                cap,
                sigma,
                &noise,
                mae,
                compute_dtype,
                use_checkpoint,
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads, &lora_set.vars)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                let mut avg = accumulated
                    .take()
                    .expect("an update fires only after accumulation");
                scale_grads(&mut avg, &lora_set.vars, 1.0 / accum as f64)?;
                clip_grad_norm(&mut avg, &lora_set.vars, 1.0)?;
                opt.step(&avg)?;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                create_output_dir(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                save_adapter(&lora_set, &ckpt)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // Cancelled before a single step completed: the factors are still the no-op init (`B = 0`), so
        // surface the typed cancellation rather than shipping an identity adapter (F-040).
        if steps_run == 0 {
            return Err(CandleError::Canceled);
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        create_output_dir(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_adapter(&lora_set, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Var;
    use candle_gen::gen_core::registry;
    use candle_gen::train::lora::build_lora_targets;
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

    /// `sample_sigma` is deterministic in its seed, lands in `[1e-3, 1−1e-3]`, and the bias tilts shift
    /// the mass the documented way (`low` ⇒ smaller σ than neutral than `high`, on average).
    #[test]
    fn sigma_is_deterministic_and_in_range() {
        for seed in [0u64, 1, 42, 9999] {
            let a = sample_sigma("sigmoid", "balanced", seed);
            let b = sample_sigma("sigmoid", "balanced", seed);
            assert_eq!(a, b, "same seed must reproduce");
            assert!((1e-3..=1.0 - 1e-3).contains(&a), "σ out of range: {a}");
        }
        let mean = |bias: &str| {
            let s: f32 = (0..256).map(|i| sample_sigma("sigmoid", bias, i)).sum();
            s / 256.0
        };
        let (lo, mid, hi) = (mean("low"), mean("balanced"), mean("high"));
        assert!(
            lo < mid && mid < hi,
            "bias order low {lo} < mid {mid} < high {hi}"
        );
    }

    /// `build_batch`: `x_t = (1−σ)x0 + σ·noise`, `target = noise − x0`, `timestep = σ` (the Krea raw-σ
    /// convention — NOT the Z-Image `1−σ`).
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
    /// unrecognized timestep/loss knob — before any load.
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
