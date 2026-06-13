//! The candle **Z-Image LoRA/LoKr trainer** (sc-5166) — the candle twin of `mlx-gen-z-image`'s
//! trainer, implementing the backend-neutral [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer)
//! with `backend = "candle"`. It retires the worker's Python torch `ZImageLoraTrainer` — the **base
//! class** of the whole LoRA-trainer hierarchy (epic 5164) — and reuses the shared
//! [`candle_gen::train`] harness the SDXL story established.
//!
//! ## What it does, and the Z-Image realities that shape it
//!
//! Cache → loop → save, mirroring the SDXL trainer's lifecycle bands but on the **flow-match**
//! objective (Z-Image is a rectified-flow model, not DDPM ε-prediction):
//!
//!  1. **Cache** — for each captioned image: decode/crop/resize to a VAE-input tensor
//!     ([`load_image_tensor`]), encode the **deterministic latent mean** (the stock z_image
//!     [`Encoder`] → `(mean − shift)·scale`; the `reg` sampling is skipped so caching is
//!     reproducible), and encode the caption through the Qwen3 text encoder with the *exact* gen-core
//!     [`TokenizerConfig`] inference uses → `(L, 2560)`. The VAE + text encoder are dropped after
//!     caching (idle for the rest of the run).
//!  2. **Loop** — sample a flow-match `σ ∈ [1e-3, 1−1e-3]` ([`sample_sigma`], the timestep
//!     distribution/bias knobs), form `x_t = (1−σ)·x0 + σ·noise`, predict the velocity through the
//!     vendored trainable DiT, regress it toward `noise − x0`, and step the adapter factors.
//!     Gradient accumulation, LR schedule, and grad-norm clipping all reuse the harness.
//!  3. **Save** — a PEFT `.safetensors` (`save_lora_peft` with the DiT's **bare** key prefix /
//!     `save_lokr`), the exact on-disk format [`crate::adapters::merge_adapters`] reads back.
//!
//! **Velocity sign.** The Z-Image inference pipeline negates the DiT's raw output before the
//! flow-match Euler step (`noise_pred.neg()` — the Z-Image sign convention). The vendored
//! [`ZImageTransformer2DModel`] returns the **raw** velocity, so the trainer negates it and regresses
//! the negated velocity toward `noise − x0` — exactly optimizing the adapter against the function
//! inference will run (the MLX twin folds the negation into its `forward`; same target either way).
//!
//! **The eager-`Var` simplification** (inherited from the SDXL harness): the adapter factors are
//! storage-sharing `Var`s installed once ([`build_lora_targets`]/[`build_lokr_targets`]); each forward
//! re-reads the current factor storage and `loss.backward()` attributes grads straight to the `Var`s,
//! so the per-step body is forward → backward → clip → step, with no re-install.
//!
//! **Gradient checkpointing** (`config.gradient_checkpointing`, sc-5246 — the candle twin of the MLX
//! trainer's sc-4874 split) routes the backward through [`checkpointed_backward_with_input_grad`]
//! over the DiT's main `layers` stack ([`ZImageTransformer2DModel::main_layer_segments`]), recomputing
//! each layer instead of retaining its activations — numerically the dense grads (the
//! `dense_and_checkpoint_grads_match` gate asserts this), at the cost of one extra forward over the
//! main stack. The pre-main refiner/embedder forward is run *retained*, so those adapters train via
//! ordinary autograd and their grads are stitched in through the recovered `unified`-boundary
//! cotangent (see [`compute_loss_grads`]). On CUDA an OOM is a catchable error (not the uncatchable
//! Mac-unified-memory SIGKILL that forced the MLX work), so this is the lever for higher training
//! resolutions / smaller cards rather than a correctness requirement.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::train::{
    NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, WeightsSource};
use candle_gen::train::checkpoint::{checkpoint_filename, file_stem};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::gradient_checkpoint::checkpointed_backward_with_input_grad;
use candle_gen::train::lora::{
    build_lokr_targets, build_lora_targets, save_lokr, save_lora_peft, AdapterKind, LoraSet,
};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen::train::schedule::{lr_multiplier, schedule_updates};
use candle_gen::{CandleError, Result};

use candle_transformers::models::z_image::preprocess::prepare_inputs;
use candle_transformers::models::z_image::text_encoder::{TextEncoderConfig, ZImageTextEncoder};
use candle_transformers::models::z_image::transformer::Config as DitConfig;
use candle_transformers::models::z_image::vae::{Encoder as VaeEncoder, VaeConfig};

use crate::dit::{ZImageTransformer2DModel, Z_IMAGE_ATTN_TARGETS};
use crate::pipeline::{QWEN_PAD_TOKEN_ID, TOKENIZER_MAX_LEN};
use crate::MODEL_ID;

/// Recognized `timestep_type` values — the noise-schedule samplers [`sample_sigma`] branches on
/// (`linear`/`uniform`/`weighted`) plus the `sigmoid` default it falls back to. Validation rejects
/// anything else rather than silently sampling sigmoid (matching the MLX trainer's F-041 guard).
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

/// `"bf16"`/`"bfloat16"` → [`DType::BF16`]; anything else → [`DType::F32`] (the gen-core contract:
/// unrecognized = f32). Z-Image is a bf16 model (the default), but the adapter factors / loss / grads
/// stay f32 regardless (master weights).
fn parse_compute_dtype(s: &str) -> DType {
    let t = s.trim();
    if t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16") {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Normalize a free-form config string (trim, lowercase, `-`/space → `_`) so validation accepts
/// exactly the spellings [`sample_sigma`] would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Sample a normalized flow-match timestep (interpolation coefficient) `σ ∈ [1e-3, 1−1e-3]` — a
/// faithful port of the MLX `sample_sigma` / SceneWorks `sample_training_timestep`: `sigmoid(randn)`
/// by default, `uniform` for linear, `(uniform + sigmoid(randn))/2` for weighted; bias `high` → `√σ`,
/// `low` → `σ²`. Deterministic in `seed` via the sc-3673 CPU `StdRng` discipline (NOT candle's device
/// RNG). Cross-framework numeric parity with MLX is a non-goal (different RNG algorithms); per-seed
/// determinism is what the worker relies on.
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
/// `target = noise − x0` (the velocity the *negated* DiT output trains toward), `timestep = 1−σ` (the
/// `current_timestep_normalized` convention the DiT's `t_embedder` consumes). All in f32.
fn build_batch(x0: &Tensor, noise: &Tensor, sigma: f32) -> Result<(Tensor, Tensor, f32)> {
    let x_t = ((x0 * (1.0 - sigma) as f64)? + (noise * sigma as f64)?)?;
    let target = (noise - x0)?;
    Ok((x_t, target, 1.0 - sigma))
}

/// Flow-match velocity loss in f32: `mean((v − target)²)` (MSE) or `mean|v − target|` (MAE). `v` (the
/// negated DiT output, in the compute dtype) is promoted to f32 so the loss/grads stay f32.
fn velocity_loss(v: &Tensor, target_f32: &Tensor, mae: bool) -> candle_core::Result<Tensor> {
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
/// `sigma`, predict the velocity through the (raw-output) DiT, **negate** it (the Z-Image inference
/// sign convention), regress it toward `noise − x0`, and return `(loss, grads)` keyed by the adapter
/// `Var`s. A free function (not a method) so the parity test can drive it against a tiny DiT.
///
/// `x0`/`noise` are the cached `[1, 16, h, w]` clean latent + the per-step noise (f32); `cap` is the
/// cached `(L, cap_feat_dim)` caption embedding. `prepare_inputs` adds the DiT's singleton frame axis
/// and pads the caption to `SEQ_MULTI_OF` (the same call inference makes), so train and infer feed the
/// DiT the identical tensor surface.
///
/// `use_checkpoint` selects the gradient-checkpointed backward over the dense `loss.backward()`. The
/// Z-Image split (mirroring the MLX trainer): the **pre-main** forward
/// ([`forward_pre_main`](ZImageTransformer2DModel::forward_pre_main) — embed + refiners) is run
/// retained, so the refiner/embedder adapters train via ordinary autograd; only the main `layers`
/// stack (the activation-memory bulk) is checkpointed via
/// [`main_layer_segments`](ZImageTransformer2DModel::main_layer_segments) +
/// [`checkpointed_backward_with_input_grad`], recomputing each layer in the backward. The returned
/// boundary cotangent `dL/d unified` is then stitched back through the retained pre-main forward to
/// fold in those adapters' grads. Both paths yield the same grads — the `tests` parity gate
/// (`dense_and_checkpoint_grads_match`) pins this, exactly as the SDXL trainer does.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &ZImageTransformer2DModel,
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
    let prepared = prepare_inputs(&x_t, std::slice::from_ref(cap), device)?;
    let cap_feats = prepared.cap_feats.to_dtype(compute_dtype)?;
    let t = Tensor::from_vec(vec![timestep], (1,), device)?;

    if use_checkpoint {
        // Retain the pre-main forward (refiner/embedder adapters train via ordinary autograd); only
        // the main `layers` stack is checkpointed (each layer recomputed in the backward).
        let (unified, ctx) =
            dit.forward_pre_main(&prepared.latents, &t, &cap_feats, &prepared.cap_mask)?;
        let mut segs = dit.main_layer_segments(&ctx);
        // Final segment: the post-main head + the negated-velocity flow-match regression -> [loss].
        // (`squeeze(2)` drops the singleton frame axis: (1, 16, 1, h, w) -> (1, 16, h, w).)
        let target_owned = target.clone();
        let ctx_ref = &ctx;
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], ctx_ref)?.squeeze(2)?.neg()?;
            Ok(vec![velocity_loss(&v, &target_owned, mae)?])
        }));
        // Seed the checkpointed chain with the detached `unified` boundary, recovering its cotangent.
        let unified_d = unified.detach();
        let (loss_val, mut grads, input_cot) = checkpointed_backward_with_input_grad(
            &segs,
            std::slice::from_ref(&unified_d),
            lora_vars,
        )?;
        drop(segs); // release the closures' borrows of `dit`/`ctx` before the stitch backward

        // Continue the chain rule into the retained pre-main: `s = ⟨unified, dL/d unified⟩` then
        // `s.backward()` delivers the refiner/embedder adapter grads (cotangent is a detached
        // constant). The main-layer grads are already accumulated in `grads`; the two `Var` sets are
        // disjoint (refiner blocks ≠ main blocks), so this is a union, not a sum — but accumulate
        // defensively in case a future layout shares a factor across the boundary.
        let cot = input_cot[0].to_dtype(DType::F32)?;
        let surrogate = (unified.to_dtype(DType::F32)? * cot)?.sum_all()?;
        let pre_grads = surrogate.backward()?;
        for v in lora_vars {
            if let Some(g) = pre_grads.get(v.as_tensor()) {
                let merged = match grads.get(v.as_tensor()) {
                    Some(prev) => (prev + g)?,
                    None => g.clone(),
                };
                grads.insert(v.as_tensor(), merged);
            }
        }
        Ok((loss_val, grads))
    } else {
        // Dense backward: one monolithic `loss.backward()` retains every layer's activations.
        // Raw DiT velocity, then negated to match the inference pipeline's `noise_pred.neg()`.
        let v = dit
            .forward(&prepared.latents, &t, &cap_feats, &prepared.cap_mask)?
            .squeeze(2)? // drop the singleton frame axis: (1, 16, 1, h, w) -> (1, 16, h, w)
            .neg()?;
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
            "z_image trainer: snapshot missing the {sub}/ component directory (at {})",
            root.display()
        )));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CandleError::Msg(format!("z_image trainer: read {sub}/: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "z_image trainer: no .safetensors in {sub}/ (at {})",
            dir.display()
        )));
    }
    Ok(files)
}

/// Build a [`VarBuilder`] over the snapshot component subdir `sub` at `dtype`.
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

/// Encode the **deterministic latent mean** of a VAE-input image `[1, 3, edge, edge]` (range
/// `[-1, 1]`) to a clean latent `[1, 16, edge/8, edge/8]`: run the encoder, take the distribution
/// **mean** (the first channel half, skipping `DiagonalGaussian`'s sampling), then `(mean − shift)·
/// scale` — the same affine `AutoEncoderKL::encode` applies, minus the stochastic draw.
fn vae_encode_mean(encoder: &VaeEncoder, img: &Tensor, shift: f64, scale: f64) -> Result<Tensor> {
    let moments = img.to_dtype(DType::F32)?.apply(encoder)?; // (1, 32, h, w) = [mean; logvar]
    let mean = moments.chunk(2, 1)?[0].clone(); // (1, 16, h, w)
    Ok(((mean - shift)? * scale)?)
}

/// Tokenize `caption` with the Qwen chat template + encode it through the Qwen3 text encoder to the
/// cached conditioning `(L, cap_feat_dim)` at f32 — the exact [`TokenizerConfig`] / encoder the
/// inference [`crate::pipeline`] uses (parity), minus the device-dtype cast (caching keeps f32).
fn encode_caption(
    tok: &TextTokenizer,
    te: &ZImageTextEncoder,
    caption: &str,
    device: &Device,
) -> Result<Tensor> {
    let out = tok
        .tokenize(caption)
        .map_err(|e| CandleError::Msg(format!("z_image trainer: tokenize: {e}")))?;
    if out.ids.is_empty() {
        return Err(CandleError::Msg(
            "z_image trainer: empty caption tokenization".into(),
        ));
    }
    let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), device)?;
    let enc = te.forward(&input_ids)?; // (1, L, 2560)
    Ok(enc.squeeze(0)?.to_dtype(DType::F32)?) // (L, 2560)
}

/// Load the Qwen tokenizer with the inference-identical config.
fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: TOKENIZER_MAX_LEN,
            pad_token_id: QWEN_PAD_TOKEN_ID,
            chat_template: ChatTemplate::QwenInstruct,
            pad_to_max_length: false,
        },
    )
    .map_err(|e| CandleError::Msg(format!("z_image trainer: load tokenizer: {e}")))
}

/// Build the vendored, trainable DiT from the snapshot `transformer/` safetensors at `dtype`.
fn build_dit(root: &Path, device: &Device, dtype: DType) -> Result<ZImageTransformer2DModel> {
    let vb = component_vb(root, "transformer", device, dtype)?;
    // The vendored DiT always runs the materialized math attention (the flash/SDPA paths have no
    // backward); the stock `use_accelerated_attn` knob is irrelevant to it. `z_image_turbo` config.
    Ok(ZImageTransformer2DModel::new(
        &DitConfig::z_image_turbo(),
        vb,
    )?)
}

/// The config's target-module suffixes (default [`Z_IMAGE_ATTN_TARGETS`]).
fn resolve_target_suffixes(cfg: &TrainingConfig) -> Vec<String> {
    if cfg.lora_target_modules.is_empty() {
        Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.lora_target_modules.clone()
    }
}

/// Write the adapter as a `.safetensors`: LoRA with the DiT's **bare** dotted keys (empty prefix —
/// the SDXL `base_model.model.unet.` prefix is SDXL-specific), LoKr with bare keys + metadata.
fn save_adapter(set: &LoraSet, path: &Path) -> Result<()> {
    let meta = HashMap::new();
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

/// Identity + capabilities of the candle Z-Image trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "z-image",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// A loaded candle Z-Image trainer. Loading is **lazy** (no file I/O — mirrors the candle SDXL
/// trainer): the heavy VAE / text-encoder / DiT are built inside [`train`](Trainer::train) at the
/// request's compute dtype.
pub struct ZImageTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Z-Image trainer from a [`LoadSpec`] whose `weights` is the Z-Image
/// snapshot directory (`tokenizer/ text_encoder/ transformer/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(
                "z_image_turbo trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    Ok(Box::new(ZImageTrainer {
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

impl Trainer for ZImageTrainer {
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

impl ZImageTrainer {
    /// Reject a request before any expensive load: empty dataset, zero rank/steps, unsupported
    /// optimizer, and — rather than silently falling back to a default sampler/loss — an unrecognized
    /// `timestep_type`/`timestep_bias`/`loss_type` (the MLX F-041 guard). The target-module
    /// resolution is checked in [`train_impl`](Self::train_impl), where the loaded DiT is available.
    fn validate_impl(&self, req: &TrainingRequest) -> Result<()> {
        let cfg = &req.config;
        if req.items.is_empty() {
            return Err(CandleError::Msg("z_image trainer: dataset is empty".into()));
        }
        if cfg.rank == 0 {
            return Err(CandleError::Msg("z_image trainer: rank must be > 0".into()));
        }
        if cfg.steps == 0 {
            return Err(CandleError::Msg(
                "z_image trainer: steps must be > 0".into(),
            ));
        }
        if !TrainOptimizer::is_supported(&cfg.optimizer) {
            return Err(CandleError::Msg(format!(
                "z_image trainer: optimizer '{}' is not available (supported: adamw, adam, rose, prodigy)",
                cfg.optimizer
            )));
        }
        if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
            return Err(CandleError::Msg(format!(
                "z_image trainer: timestep_type '{}' is not recognized (supported: {})",
                cfg.timestep_type,
                TIMESTEP_TYPES.join(", ")
            )));
        }
        if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
            return Err(CandleError::Msg(format!(
                "z_image trainer: timestep_bias '{}' is not recognized (supported: {})",
                cfg.timestep_bias,
                TIMESTEP_BIASES.join(", ")
            )));
        }
        if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
            return Err(CandleError::Msg(format!(
                "z_image trainer: loss_type '{}' is not recognized (supported: {})",
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

        // --- load + cache: VAE latent means + Qwen caption embeddings (both f32) ---
        on_progress(TrainingProgress::LoadingModel);
        let vae_cfg = VaeConfig::z_image();
        let vae_encoder = VaeEncoder::new(
            &vae_cfg,
            component_vb(&self.root, "vae", device, DType::F32)?.pp("encoder"),
        )?;
        let tokenizer = load_tokenizer(&self.root)?;
        let text_encoder = ZImageTextEncoder::new(
            &TextEncoderConfig::z_image(),
            component_vb(&self.root, "text_encoder", device, DType::F32)?,
        )?;

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
            let x0 = vae_encode_mean(
                &vae_encoder,
                &img,
                vae_cfg.shift_factor,
                vae_cfg.scaling_factor,
            )?;
            let cap = encode_caption(&tokenizer, &text_encoder, &item.caption, device)?;
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
                "z_image trainer: no usable dataset items".into(),
            ));
        }

        // --- build DiT + install adapters ---
        let mut dit = build_dit(&self.root, device, compute_dtype)?;
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
        let mae = matches!(cfg.loss_type.to_ascii_lowercase().as_str(), "mae" | "l1");
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

        // Cancelled before a single step completed: the factors are still the no-op init (`B = 0`),
        // so surface the typed cancellation rather than shipping an identity adapter (F-040).
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
    use candle_gen::gen_core::registry;
    use candle_gen::train::lora::build_lora_targets;
    use candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::z_image::transformer::Config;

    /// The smallest valid Z-Image DiT (head_dim locked to 128 by `axes_dims`), 1 main layer + 1
    /// refiner each, tiny caption dim — exercises the real flow-match forward + backward on CPU.
    fn tiny_dit(vb: VarBuilder) -> (ZImageTransformer2DModel, Config) {
        let mut cfg = Config::z_image_turbo();
        cfg.dim = 128;
        cfg.n_heads = 1;
        cfg.n_kv_heads = 1;
        cfg.n_layers = 1;
        cfg.n_refiner_layers = 1;
        cfg.cap_feat_dim = 64;
        let model = ZImageTransformer2DModel::new(&cfg, vb).unwrap();
        (model, cfg)
    }

    /// `sample_sigma` is deterministic in its seed, lands in `[1e-3, 1−1e-3]`, and the bias tilts
    /// shift the mass the documented way (`low` ⇒ smaller σ than neutral than `high`, on average).
    #[test]
    fn sigma_is_deterministic_and_in_range() {
        for seed in [0u64, 1, 42, 9999] {
            let a = sample_sigma("sigmoid", "balanced", seed);
            let b = sample_sigma("sigmoid", "balanced", seed);
            assert_eq!(a, b, "same seed must reproduce");
            assert!((1e-3..=1.0 - 1e-3).contains(&a), "σ out of range: {a}");
        }
        // uniform/weighted/linear all stay in range too.
        for ttype in ["uniform", "linear", "weighted"] {
            let s = sample_sigma(ttype, "neutral", 7);
            assert!(
                (1e-3..=1.0 - 1e-3).contains(&s),
                "{ttype} σ out of range: {s}"
            );
        }
        // Averaged bias ordering: low (σ²) ≤ neutral ≤ high (√σ).
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

    /// `build_batch`: `x_t = (1−σ)x0 + σ·noise`, `target = noise − x0`, `timestep = 1−σ`.
    #[test]
    fn build_batch_math() {
        let dev = Device::Cpu;
        let x0 = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target, timestep) = build_batch(&x0, &noise, 0.25).unwrap();
        // x_t = 0.75·[2,4] + 0.25·[1,0] = [1.75, 3.0]; target = [1-2, 0-4] = [-1, -4]; t = 0.75.
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
        assert!((timestep - 0.75).abs() < 1e-6);
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var` (backprop reaches
    /// the LoRA seam through the negated-velocity loss).
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let (mut dit, cfg) = tiny_dit(vb);
        let suffixes: Vec<String> = Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off zero so both A and B grads are nonzero (a no-op-init adapter zeros A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }

        let x0 = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();

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
        // Sanity: the LoRA set covers every attention projection in the 1 refiner ×2 + 1 layer.
        let expected_targets = 4 * (cfg.n_refiner_layers * 2 + cfg.n_layers);
        assert_eq!(
            set.vars.len(),
            expected_targets * 2,
            "two factors per target"
        );
    }

    /// The correctness gate for the `gradient_checkpointing` lever: the checkpointed backward must
    /// reproduce the dense `loss.backward()` grads (mod float reassociation) over the tiny DiT with
    /// nonzero LoRA factors. The tiny config adapts BOTH refiner and main-layer blocks, so this
    /// comparison spans the two distinct checkpoint paths — the **retained** pre-main refiner/embedder
    /// adapters (grads stitched in through the recovered `unified`-boundary cotangent) AND the
    /// **checkpointed** main `layers` (grads from the segmented VJP). A break in either path surfaces
    /// here as a missing or mismatched grad. Mirrors the SDXL trainer's `dense_and_checkpoint_grads_match`.
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let (mut dit, cfg) = tiny_dit(vb);
        let suffixes: Vec<String> = Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off zero so both A and B grads are nonzero (a no-op-init adapter zeros A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        // Every projection in both refiner blocks AND the main layer is adapted, so the per-var
        // comparison below necessarily spans the retained (refiner) and checkpointed (main) paths.
        let expected_targets = 4 * (cfg.n_refiner_layers * 2 + cfg.n_layers);
        assert_eq!(
            set.vars.len(),
            expected_targets * 2,
            "two factors per target"
        );

        let x0 = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();

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
    }

    /// One optimizer step over the tiny DiT lowers (or holds) the loss on the same fixed batch — the
    /// step actually descends the flow-match objective, end to end through the harness.
    #[test]
    fn one_optimizer_step_descends() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let (mut dit, cfg) = tiny_dit(vb);
        let suffixes: Vec<String> = Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let x0 = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();

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
        // A few steps on the same batch should reduce the loss (over-fit a single sample).
        let mut grads = grads;
        for _ in 0..5 {
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
            "5 steps on a fixed batch should lower the loss: {loss0} -> {loss1}"
        );
    }

    /// The trainer self-registers and resolves through gen-core's trainer registry as the candle
    /// Z-Image trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer("z_image_turbo", &spec)
            .expect("candle z-image trainer is registered");
        assert_eq!(t.descriptor().id, "z_image_turbo");
        assert_eq!(t.descriptor().backend, "candle");
        assert!(t.descriptor().supports_lora);
        assert!(t.descriptor().supports_lokr);
    }

    /// `validate` rejects an empty dataset, zero rank/steps, an unsupported optimizer, and an
    /// unrecognized timestep/loss knob — before any load.
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer("z_image_turbo", &spec).unwrap();

        let item = candle_gen::gen_core::train::TrainingItem {
            image_path: "/img.png".into(),
            caption: "x".into(),
        };
        let base = TrainingRequest {
            items: vec![item.clone()],
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
