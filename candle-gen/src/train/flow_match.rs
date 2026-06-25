//! The shared **flow-matching trainer** harness (sc-7787) — the common spine of the candle
//! rectified-flow LoRA/LoKr trainers (Z-Image, Lens, Wan, Krea), hoisted out of the four
//! near-identical `*/src/training.rs` clones.
//!
//! It comes in two tiers, both consumed off the same module:
//!
//!  * **Tier 1 — pure helpers** (this module's free functions + the recognized-knob tables): the
//!    flow-match math (`build_batch`/`velocity_loss`), the seeded `StdRng` samplers
//!    (`sample_unit_timestep`/`sample_noise`, `timestep_seed`/`noise_seed`), the snapshot/IO plumbing
//!    (`component_files`/`component_vb`/`save_adapter`/`create_output_dir`), the config plumbing
//!    (`parse_compute_dtype`/`normalize_cfg`/`is_mae`/`effective_weight_decay`/`resolve_target_suffixes`/
//!    `validate_flow_match_request`), and the optimizer step (`install_adapters`/`apply_update`). These
//!    were copy-pasted verbatim across all four trainers (~150 lines each); every adopter now calls the
//!    one copy. **All four flow-match trainers** consume Tier 1 (Wan included — its dual-expert loop
//!    stays bespoke but is built from these helpers).
//!
//!  * **Tier 2 — the single-model driver** ([`FlowMatchTrainer`] + [`run_flow_match_training`]): the
//!    cache → loop → save scaffolding (optimizer/schedule setup, gradient accumulation, periodic
//!    checkpoint save, cooperative cancel, the `steps_run == 0 ⇒ Canceled` guard, final flush + save).
//!    A trainer implements the small [`FlowMatchTrainer`] hook trait (cache the dataset, build the
//!    trainable DiT, run one micro-step) and the driver owns the loop. **Z-Image, Lens, and Krea**
//!    adopt the driver; each keeps its own `compute_loss_grads` (the parity-critical part — velocity
//!    sign, timestep convention, and the gradient-checkpoint split genuinely differ between them) as
//!    the [`FlowMatchTrainer::micro_step`] body.
//!
//! **Why Wan is driver-exempt.** The Wan A14B is a dual-expert MoE: it alternates a high-noise and a
//! low-noise expert, each with its own adapter set, optimizer, LR schedule, timestep band, and
//! accumulation buffer, and saves an expert-suffixed pair. That loop does not fit the single-model,
//! single-optimizer driver cleanly, so Wan keeps its bespoke loop and consumes only Tier 1 — exactly
//! the split sc-7787 sanctions ("hoist what's genuinely shared; don't force a 3-way abstraction").
//!
//! **Why `compute_loss_grads` stays per-crate.** A single shared loss/grad body would not be faithful:
//! Z-Image negates the DiT velocity, feeds timestep `1−σ`, and stitches grads through trainable
//! pre-main refiner/embedder adapters (`checkpointed_backward_with_input_grad`); Lens/Krea/Wan use the
//! raw velocity with timestep `t` / `σ` / `t·1000` respectively and a plain `checkpointed_backward`
//! over an adapter-free, detached pre-main. So each trainer supplies its own `compute_loss_grads` and
//! the driver only orchestrates around it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::gen_core::train::{
    NetworkType, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use crate::train::checkpoint::{checkpoint_filename, file_stem};
use crate::train::lora::{
    build_lokr_targets, build_lora_targets, save_lokr, save_lora_peft, AdapterKind, LoraHost,
    LoraSet,
};
use crate::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use crate::train::schedule::{lr_multiplier, schedule_updates};
use crate::{CandleError, Result};

/// Recognized `timestep_type` values — the noise-schedule samplers [`sample_unit_timestep`] branches
/// on (`linear`/`uniform`/`weighted`) plus the `sigmoid` default it falls back to. Validation rejects
/// anything else rather than silently sampling sigmoid (the MLX F-041 guard).
pub const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values — the high/low-noise tilts plus the neutral default.
pub const TIMESTEP_BIASES: [&str; 9] = [
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
pub const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// `"bf16"`/`"bfloat16"` → [`DType::BF16`]; anything else → [`DType::F32`] (the gen-core contract:
/// unrecognized = f32). The flow-match DiTs are bf16 models, but the adapter factors / loss / grads
/// stay f32 regardless (master weights).
pub fn parse_compute_dtype(s: &str) -> DType {
    let t = s.trim();
    if t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16") {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Normalize a free-form config string (trim, lowercase, `-`/space → `_`) so validation accepts
/// exactly the spellings [`sample_unit_timestep`] would.
pub fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// `true` iff `loss_type` selects MAE (`mae`/`l1`), else MSE (the default). Normalized first so any
/// recognized spelling/casing resolves identically.
pub fn is_mae(cfg: &TrainingConfig) -> bool {
    matches!(normalize_cfg(&cfg.loss_type).as_str(), "mae" | "l1")
}

/// The effective weight decay: `0` for the `adam` choice (AdamW with `wd = 0` ≡ Adam, so one optimizer
/// covers both), else the config's value.
pub fn effective_weight_decay(cfg: &TrainingConfig) -> f32 {
    if cfg.optimizer.eq_ignore_ascii_case("adam") {
        0.0
    } else {
        cfg.weight_decay
    }
}

/// Sample a **unit** flow-match timestep `t ∈ [1e-3, 1−1e-3]` — a faithful port of the MLX
/// `sample_training_timestep` / SceneWorks sampler: `sigmoid(randn)` by default, `uniform` for
/// `linear`/`uniform`, `(uniform + sigmoid(randn))/2` for `weighted`; bias `high` → `√t`, `low` → `t²`.
/// Deterministic in `seed` via the sc-3673 CPU `StdRng` discipline (NOT candle's device RNG).
/// Cross-framework numeric parity with MLX is a non-goal (different RNG algorithms); per-seed
/// determinism is what the worker relies on.
///
/// Each trainer adapts this unit value to its own convention: Z-Image consumes it as `σ` directly
/// (timestep `1−σ`), Krea as `σ` (timestep `σ`), Lens as `t` (cast to f64, fed directly), Wan affine-maps
/// it into the active expert's noise band.
pub fn sample_unit_timestep(timestep_type: &str, timestep_bias: &str, seed: u64) -> f32 {
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

/// The per-step timestep RNG seed: mixes the config `seed` with `step` via the golden-ratio constant —
/// the derivation every flow-match trainer uses so per-seed runs reproduce.
pub fn timestep_seed(seed: u64, step: u32) -> u64 {
    seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64)
}

/// The per-step noise RNG seed (distinct from [`timestep_seed`] so the timestep and the prior draw
/// don't correlate) — the derivation every flow-match trainer uses.
pub fn noise_seed(seed: u64, step: u32) -> u64 {
    seed.wrapping_add(step as u64).wrapping_mul(2) + 1
}

/// `(x_t, target)` for one sample at flow-match `t`: `x_t = (1−t)·x0 + t·noise`, `target = noise − x0`
/// (the velocity the DiT output regresses toward). All in f32. The per-trainer velocity **sign** (raw vs
/// negated) and **timestep convention** (`t` vs `1−t` vs `t·1000`) are applied at the call site, not here.
pub fn build_batch(x0: &Tensor, noise: &Tensor, t: f64) -> Result<(Tensor, Tensor)> {
    let x_t = ((x0 * (1.0 - t))? + (noise * t)?)?;
    let target = (noise - x0)?;
    Ok((x_t, target))
}

/// Flow-match velocity loss in f32: `mean((v − target)²)` (MSE) or `mean|v − target|` (MAE). `v` (the
/// DiT velocity, in the compute dtype) is promoted to f32 so the loss/grads stay f32.
pub fn velocity_loss(v: &Tensor, target: &Tensor, mae: bool) -> candle_core::Result<Tensor> {
    let diff = (v.to_dtype(DType::F32)? - target)?;
    if mae {
        diff.abs()?.mean_all()
    } else {
        diff.sqr()?.mean_all()
    }
}

/// Deterministic `N(0, 1)` noise of the given shape, drawn from a seeded CPU `StdRng` then moved to
/// `device` (sc-3673 launch-portable discipline). The flow-match prior + the regression target.
pub fn sample_noise(shape: &[usize], seed: u64, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?.to_device(device)?)
}

/// Reject a [`TrainingRequest`] before any expensive load: empty dataset, zero rank/steps, unsupported
/// optimizer, and — rather than silently falling back to a default sampler/loss — an unrecognized
/// `timestep_type`/`timestep_bias`/`loss_type` (the MLX F-041 guard). `label` prefixes every message
/// (e.g. `"z_image trainer"`) so the per-crate error text is unchanged.
pub fn validate_flow_match_request(req: &TrainingRequest, label: &str) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err(CandleError::Msg(format!("{label}: dataset is empty")));
    }
    if cfg.rank == 0 {
        return Err(CandleError::Msg(format!("{label}: rank must be > 0")));
    }
    if cfg.steps == 0 {
        return Err(CandleError::Msg(format!("{label}: steps must be > 0")));
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(CandleError::Msg(format!(
            "{label}: optimizer '{}' is not available (supported: adamw, adam, rose, prodigy)",
            cfg.optimizer
        )));
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
        return Err(CandleError::Msg(format!(
            "{label}: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )));
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(CandleError::Msg(format!(
            "{label}: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )));
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(CandleError::Msg(format!(
            "{label}: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )));
    }
    Ok(())
}

/// Resolve the sorted `.safetensors` files in the snapshot component subdir `sub`. `label` prefixes the
/// error text (e.g. `"lens trainer"`).
pub fn component_files(root: &Path, sub: &str, label: &str) -> Result<Vec<PathBuf>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "{label}: snapshot missing the {sub}/ component directory (at {})",
            root.display()
        )));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CandleError::Msg(format!("{label}: read {sub}/: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "{label}: no .safetensors in {sub}/ (at {})",
            dir.display()
        )));
    }
    Ok(files)
}

/// Build a [`VarBuilder`] over the snapshot component subdir `sub` at `dtype`. `label` prefixes the
/// error text.
pub fn component_vb(
    root: &Path,
    sub: &str,
    device: &Device,
    dtype: DType,
    label: &str,
) -> Result<VarBuilder<'static>> {
    let files = component_files(root, sub, label)?;
    // SAFETY: mmap of read-only weight files; standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? })
}

/// Write the adapter as a `.safetensors`: LoRA with the DiT's **bare** dotted keys (empty prefix — the
/// SDXL `base_model.model.unet.` prefix is SDXL-specific), LoKr with bare keys + metadata. `extra_meta`
/// is merged into the header (e.g. Krea's `baseModel`/`family` provenance; the empty map for the others).
pub fn save_adapter(
    set: &LoraSet,
    extra_meta: &HashMap<String, String>,
    path: &Path,
) -> Result<()> {
    match set.kind {
        AdapterKind::Lora => save_lora_peft(set, "", extra_meta, path),
        AdapterKind::Lokr => save_lokr(set, extra_meta, path),
    }
}

/// Create the output directory, mapping the `io::Error` into the crate error.
pub fn create_output_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CandleError::Msg(format!("create output dir {}: {e}", dir.display())))
}

/// The config's target-module suffixes, falling back to `default` (the family's attention surface) when
/// the request leaves `lora_target_modules` empty.
pub fn resolve_target_suffixes(cfg: &TrainingConfig, default: &[&str]) -> Vec<String> {
    if cfg.lora_target_modules.is_empty() {
        default.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.lora_target_modules.clone()
    }
}

/// Install LoRA/LoKr adapters on `host` for the resolved `suffixes` (dispatching on
/// `cfg.network_type`). `seed` is taken explicitly (not from `cfg.seed`) so the Wan trainer can offset
/// it per expert; the single-model driver passes `cfg.seed`.
pub fn install_adapters(
    host: &mut dyn LoraHost,
    cfg: &TrainingConfig,
    suffixes: &[String],
    seed: u64,
    device: &Device,
) -> Result<LoraSet> {
    match cfg.network_type {
        NetworkType::Lora => build_lora_targets(host, suffixes, cfg.rank, cfg.alpha, seed, device),
        NetworkType::Lokr => build_lokr_targets(
            host,
            suffixes,
            cfg.rank,
            cfg.alpha,
            cfg.decompose_factor,
            seed,
            device,
        ),
    }
}

/// Fire one optimizer update: LR-schedule, average the accumulated grads by `1/accum`, grad-norm clip,
/// step. Panics if called with no pending accumulation (the caller fires it only on an accumulation
/// boundary or the final flush).
#[allow(clippy::too_many_arguments)]
pub fn apply_update(
    opt: &mut TrainOptimizer,
    accumulated: &mut Option<GradStore>,
    set: &LoraSet,
    accum: u32,
    cfg: &TrainingConfig,
    update_idx: u32,
    total_updates: u32,
    warmup_updates: u32,
) -> Result<()> {
    let mult = lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
    opt.set_lr_scaled(mult);
    let mut avg = accumulated
        .take()
        .expect("apply_update called with a pending accumulation");
    scale_grads(&mut avg, &set.vars, 1.0 / accum as f64)?;
    clip_grad_norm(&mut avg, &set.vars, 1.0)?;
    opt.step(&avg)?;
    Ok(())
}

/// The per-model hooks the single-model [`run_flow_match_training`] driver calls. A flow-match trainer
/// with one DiT, one optimizer, and one adapter set (Z-Image, Lens, Krea) implements this; the driver
/// owns the cache → loop → save scaffolding around it. (Wan's dual-expert loop does not use this — it
/// consumes only the Tier-1 helpers.)
pub trait FlowMatchTrainer {
    /// The trainable DiT — must expose its adaptable projections to the harness.
    type Dit: LoraHost;
    /// One dataset sample's cached latent + conditioning (e.g. `(x0, caption_embed)`).
    type Cached;
    /// Run-derived state shared across steps (e.g. Lens's latent grid `(h, w)`; `()` when unused).
    type Aux;

    /// Error-message prefix + the `no usable dataset items` label (e.g. `"z_image trainer"`).
    const LABEL: &'static str;

    /// The compute device the trainer loads onto.
    fn device(&self) -> &Device;

    /// The family's default LoRA target suffixes (used when the request leaves the target list empty).
    fn default_targets(&self) -> &'static [&'static str];

    /// Cache the dataset: encode each item's latent + conditioning (reporting
    /// [`TrainingProgress::Caching`]) and return the per-sample cache plus any run-derived [`Aux`].
    /// Honors `req.cancel` (a cancel mid-cache yields a short/empty cache; the driver maps an empty
    /// cache to `Canceled`). The heavy encoders are loaded and dropped inside this call.
    fn cache(
        &self,
        req: &TrainingRequest,
        device: &Device,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<Self::Cached>, Self::Aux)>;

    /// Build the trainable DiT (no adapters installed — the driver installs them onto the returned
    /// host).
    fn build_dit(&self, req: &TrainingRequest, device: &Device) -> Result<Self::Dit>;

    /// One micro-step's forward+backward: build the noised latent for `cached` at the sampled timestep,
    /// predict + regress the velocity through `dit` (the per-model sign / timestep / checkpoint
    /// convention lives here), and return `(loss, grads)` keyed by `vars`.
    #[allow(clippy::too_many_arguments)]
    fn micro_step(
        &self,
        dit: &Self::Dit,
        vars: &[Var],
        cached: &Self::Cached,
        aux: &Self::Aux,
        cfg: &TrainingConfig,
        step: u32,
        device: &Device,
    ) -> Result<(f32, GradStore)>;

    /// Persist the adapter set to `path`. Defaults to the bare-key [`save_adapter`] with no extra
    /// metadata; override to inject provenance (e.g. Krea's `baseModel`/`family`).
    fn save(&self, set: &LoraSet, path: &Path) -> Result<()> {
        save_adapter(set, &HashMap::new(), path)
    }
}

/// Drive a single-model flow-match trainer end to end: cache → install adapters → train loop → save.
///
/// Owns the loop scaffolding every single-model trainer shared verbatim — optimizer + LR-schedule
/// setup, per-step gradient accumulation with the `1/accum` average applied on each accumulation
/// boundary (plus a final flush of any sub-`accum` remainder), periodic checkpoint save
/// (`save_every`), cooperative cancellation, the `steps_run == 0 ⇒ Canceled` guard (so a cancel before
/// any step ships no identity adapter, F-040), and the final adapter save. The per-model specifics —
/// caching, DiT construction, the loss/grad body, and adapter provenance — are the [`FlowMatchTrainer`]
/// hooks.
pub fn run_flow_match_training<T: FlowMatchTrainer>(
    model: &T,
    req: &TrainingRequest,
    on_progress: &mut dyn FnMut(TrainingProgress),
) -> Result<TrainingOutput> {
    let cfg = &req.config;
    let device = model.device();
    on_progress(TrainingProgress::Preparing);

    // --- cache (latents + conditioning); the encoders load and drop inside the hook ---
    on_progress(TrainingProgress::LoadingModel);
    let (cache, aux) = model.cache(req, device, on_progress)?;
    if cache.is_empty() {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        return Err(CandleError::Msg(format!(
            "{}: no usable dataset items",
            T::LABEL
        )));
    }

    // --- build the trainable DiT + install adapters ---
    let mut dit = model.build_dit(req, device)?;
    let suffixes = resolve_target_suffixes(cfg, model.default_targets());
    let set = install_adapters(&mut dit, cfg, &suffixes, cfg.seed, device)?;

    // --- optimizer + schedule ---
    let mut opt = TrainOptimizer::from_config(
        &cfg.optimizer,
        set.vars.clone(),
        cfg.learning_rate,
        effective_weight_decay(cfg),
    )?;
    let accum = cfg.gradient_accumulation.max(1);
    let (total_updates, warmup_updates) = schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
    let stem = file_stem(&req.file_name).to_string();

    // --- train loop ---
    let mut accumulated: Option<GradStore> = None;
    let mut micro = 0u32;
    let mut update_idx = 0u32;
    let mut last_loss = 0.0f32;
    let mut steps_run = 0u32;
    for step in 1..=cfg.steps {
        if req.cancel.is_cancelled() {
            break;
        }
        let cached = &cache[((step - 1) as usize) % cache.len()];
        let (loss, grads) = model.micro_step(&dit, &set.vars, cached, &aux, cfg, step, device)?;
        last_loss = loss;
        steps_run = step;
        accumulate_grads(&mut accumulated, grads, &set.vars)?;
        micro += 1;
        if micro.is_multiple_of(accum) {
            apply_update(
                &mut opt,
                &mut accumulated,
                &set,
                accum,
                cfg,
                update_idx,
                total_updates,
                warmup_updates,
            )?;
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
            model.save(&set, &ckpt)?;
            on_progress(TrainingProgress::Checkpoint { step });
        }
    }

    // Cancelled before a single step completed: the factors are still the no-op init (`B = 0`), so
    // surface the typed cancellation rather than shipping an identity adapter (F-040).
    if steps_run == 0 {
        return Err(CandleError::Canceled);
    }
    // Flush any pending (sub-`accum`) accumulation so the final partial step is applied.
    if accumulated.is_some() {
        apply_update(
            &mut opt,
            &mut accumulated,
            &set,
            accum,
            cfg,
            update_idx,
            total_updates,
            warmup_updates,
        )?;
    }

    // --- save final adapter ---
    on_progress(TrainingProgress::Saving);
    create_output_dir(&req.output_dir)?;
    let adapter_path = req.output_dir.join(&req.file_name);
    model.save(&set, &adapter_path)?;
    Ok(TrainingOutput {
        adapter_path,
        steps: steps_run,
        final_loss: last_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen_core::runtime::CancelFlag;
    use crate::gen_core::train::{TrainingItem, TrainingRequest};
    use crate::train::lora::{LoraHost, LoraLinear};
    use candle_nn::Linear;
    use std::cell::Cell;

    /// `sample_unit_timestep` is deterministic in its seed, lands in `[1e-3, 1−1e-3]`, and the bias
    /// tilts shift the mass the documented way (`low` ⇒ smaller t than neutral than `high`, on
    /// average) across all sampler types. This is the single home for the sampler test the four
    /// trainers previously each duplicated.
    #[test]
    fn sample_unit_timestep_deterministic_in_range_and_biased() {
        for seed in [0u64, 1, 42, 9999] {
            let a = sample_unit_timestep("sigmoid", "balanced", seed);
            let b = sample_unit_timestep("sigmoid", "balanced", seed);
            assert_eq!(a, b, "same seed must reproduce");
            assert!((1e-3..=1.0 - 1e-3).contains(&a), "t out of range: {a}");
        }
        for ttype in ["uniform", "linear", "weighted", "sigmoid"] {
            let s = sample_unit_timestep(ttype, "neutral", 7);
            assert!(
                (1e-3..=1.0 - 1e-3).contains(&s),
                "{ttype} t out of range: {s}"
            );
        }
        let mean = |bias: &str| {
            let s: f32 = (0..256)
                .map(|i| sample_unit_timestep("sigmoid", bias, i))
                .sum();
            s / 256.0
        };
        let (lo, mid, hi) = (mean("low"), mean("balanced"), mean("high"));
        assert!(
            lo < mid && mid < hi,
            "bias order low {lo} < mid {mid} < high {hi}"
        );
    }

    /// `build_batch`: `x_t = (1−t)x0 + t·noise`, `target = noise − x0`.
    #[test]
    fn build_batch_math() {
        let dev = Device::Cpu;
        let x0 = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target) = build_batch(&x0, &noise, 0.25).unwrap();
        // x_t = 0.75·[2,4] + 0.25·[1,0] = [1.75, 3.0]; target = [1-2, 0-4] = [-1, -4].
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
    }

    /// `velocity_loss`: MSE vs MAE of `v − target`, promoted to f32.
    #[test]
    fn velocity_loss_mse_and_mae() {
        let dev = Device::Cpu;
        let v = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &dev).unwrap();
        let target = Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &dev).unwrap();
        let mse = velocity_loss(&v, &target, false)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let mae = velocity_loss(&v, &target, true)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((mse - 2.5).abs() < 1e-6, "mse {mse}"); // (1+4)/2
        assert!((mae - 1.5).abs() < 1e-6, "mae {mae}"); // (1+2)/2
    }

    /// `sample_noise` is deterministic in its seed and shaped as requested.
    #[test]
    fn sample_noise_deterministic() {
        let dev = Device::Cpu;
        let a = sample_noise(&[2, 3], 7, &dev).unwrap();
        let b = sample_noise(&[2, 3], 7, &dev).unwrap();
        assert_eq!(a.dims(), &[2, 3]);
        assert_eq!(
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            b.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    /// The seed derivations are distinct per step and don't collide between the timestep and noise
    /// draws (so the sampled `t` and prior aren't correlated through a shared seed).
    #[test]
    fn seed_derivations_are_distinct() {
        for step in 1..8u32 {
            assert_ne!(timestep_seed(42, step), noise_seed(42, step));
            assert_ne!(timestep_seed(42, step), timestep_seed(42, step + 1));
        }
    }

    // --- A mock single-model trainer exercising the Tier-2 driver (the loop scaffolding that had no
    //     unit coverage before sc-7787). The DiT is a single adaptable Linear; micro_step makes a loss
    //     out of the adapter factors directly so backprop reaches them with no real model. ---

    struct MockDit(LoraLinear);
    impl LoraHost for MockDit {
        fn visit_lora_mut(
            &mut self,
            f: &mut dyn FnMut(&mut LoraLinear) -> Result<()>,
        ) -> Result<()> {
            f(&mut self.0)
        }
    }

    struct MockTrainer {
        device: Device,
        steps_seen: Cell<u32>,
        saves: Cell<u32>,
        cache_len: usize,
    }

    impl FlowMatchTrainer for MockTrainer {
        type Dit = MockDit;
        type Cached = ();
        type Aux = ();
        const LABEL: &'static str = "mock trainer";

        fn device(&self) -> &Device {
            &self.device
        }
        fn default_targets(&self) -> &'static [&'static str] {
            &["to_q"]
        }
        fn cache(
            &self,
            _req: &TrainingRequest,
            _device: &Device,
            on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> Result<(Vec<()>, ())> {
            for i in 0..self.cache_len {
                on_progress(TrainingProgress::Caching {
                    current: i as u32 + 1,
                    total: self.cache_len as u32,
                });
            }
            Ok((vec![(); self.cache_len], ()))
        }
        fn build_dit(&self, _req: &TrainingRequest, device: &Device) -> Result<MockDit> {
            let w = Tensor::zeros((4, 4), DType::F32, device)?;
            Ok(MockDit(LoraLinear::from_linear(
                Linear::new(w, None),
                4,
                4,
                "to_q".into(),
            )))
        }
        fn micro_step(
            &self,
            _dit: &MockDit,
            vars: &[Var],
            _cached: &(),
            _aux: &(),
            _cfg: &TrainingConfig,
            step: u32,
            _device: &Device,
        ) -> Result<(f32, GradStore)> {
            self.steps_seen.set(step);
            // A loss built straight from the factor Vars: `Σ vᵢ²` → nonzero grad `2vᵢ` on each.
            let mut loss = vars[0].as_tensor().sqr()?.sum_all()?;
            for v in &vars[1..] {
                loss = (loss + v.as_tensor().sqr()?.sum_all()?)?;
            }
            let val = loss.to_scalar::<f32>()?;
            let grads = loss.backward()?;
            Ok((val, grads))
        }
        fn save(&self, _set: &LoraSet, _path: &Path) -> Result<()> {
            self.saves.set(self.saves.get() + 1);
            Ok(())
        }
    }

    fn mock_request(
        items: usize,
        steps: u32,
        accum: u32,
        save_every: u32,
        cancel: CancelFlag,
    ) -> TrainingRequest {
        let config = TrainingConfig {
            steps,
            gradient_accumulation: accum,
            save_every,
            ..TrainingConfig::default()
        };
        TrainingRequest {
            items: (0..items)
                .map(|i| TrainingItem {
                    image_path: format!("/img{i}.png").into(),
                    caption: "x".into(),
                })
                .collect(),
            config,
            output_dir: std::env::temp_dir().join("candle_flow_match_driver_test"),
            file_name: "a.safetensors".into(),
            trigger_words: vec![],
            cancel,
        }
    }

    /// The driver runs all steps, reports the right `steps`, and saves exactly once (the final adapter)
    /// when `save_every == 0`.
    #[test]
    fn driver_runs_all_steps_and_saves_final() {
        let model = MockTrainer {
            device: Device::Cpu,
            steps_seen: Cell::new(0),
            saves: Cell::new(0),
            cache_len: 3,
        };
        let req = mock_request(3, 5, 1, 0, CancelFlag::new());
        let out = run_flow_match_training(&model, &req, &mut |_| {}).unwrap();
        assert_eq!(out.steps, 5);
        assert_eq!(model.steps_seen.get(), 5);
        assert_eq!(model.saves.get(), 1, "only the final save");
        assert!(out.final_loss.is_finite());
    }

    /// `save_every` writes intermediate checkpoints (at steps that are multiples below the last) plus
    /// the final adapter.
    #[test]
    fn driver_writes_periodic_checkpoints() {
        let model = MockTrainer {
            device: Device::Cpu,
            steps_seen: Cell::new(0),
            saves: Cell::new(0),
            cache_len: 2,
        };
        // steps 1..=6, save_every 2 → checkpoints at 2 and 4 (6 is the final step, excluded) + 1 final.
        let req = mock_request(2, 6, 1, 2, CancelFlag::new());
        let out = run_flow_match_training(&model, &req, &mut |_| {}).unwrap();
        assert_eq!(out.steps, 6);
        assert_eq!(model.saves.get(), 3, "2 checkpoints + 1 final");
    }

    /// A cancel tripped before the first step yields the typed `Canceled` (no identity adapter shipped).
    #[test]
    fn driver_cancel_before_first_step_is_canceled() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let model = MockTrainer {
            device: Device::Cpu,
            steps_seen: Cell::new(0),
            saves: Cell::new(0),
            cache_len: 2,
        };
        let req = mock_request(2, 5, 1, 0, cancel);
        let err = run_flow_match_training(&model, &req, &mut |_| {}).unwrap_err();
        assert!(
            matches!(err, CandleError::Canceled),
            "expected Canceled, got {err:?}"
        );
        assert_eq!(model.saves.get(), 0, "nothing saved");
    }

    /// An empty cache (no usable items, not cancelled) is a typed error, not a panic or a save.
    #[test]
    fn driver_empty_cache_errors() {
        let model = MockTrainer {
            device: Device::Cpu,
            steps_seen: Cell::new(0),
            saves: Cell::new(0),
            cache_len: 0,
        };
        let req = mock_request(0, 5, 1, 0, CancelFlag::new());
        let err = run_flow_match_training(&model, &req, &mut |_| {}).unwrap_err();
        match err {
            CandleError::Msg(m) => assert!(m.contains("no usable dataset items"), "got {m}"),
            other => panic!("expected Msg, got {other:?}"),
        }
    }
}
