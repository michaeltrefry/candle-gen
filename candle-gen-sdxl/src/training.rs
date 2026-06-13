//! The candle **SDXL LoRA/LoKr trainer** (sc-5165) — the candle twin of `mlx-gen-sdxl`'s trainer,
//! implementing the backend-neutral [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer) with
//! `backend = "candle"`. It retires the worker's Python torch `LoraTrainer` for SDXL (epic 5164, the
//! zero-Python north star).
//!
//! ## What it does, and the candle realities that shape it
//!
//! Cache → loop → save, mirroring the MLX trainer's lifecycle bands:
//!
//!  1. **Cache** — for each captioned image: decode/crop/resize to a VAE-input tensor
//!     ([`dataset::load_image_tensor`]), encode the **deterministic latent mean**
//!     (`VaeMomentsEncoder::encode_mean` → `mean × 0.13025`, the user's `.mean` decision — candle's
//!     stock VAE hides the mean, hence the vendored encoder), and encode the caption through the dual
//!     CLIP into `[1, 77, 2048]`. The CLIP + VAE encoders are dropped after caching.
//!  2. **Loop** — sample a uniform integer DDPM timestep `t`, form `noisy = √ᾱ·x0 + √(1-ᾱ)·noise`
//!     (candle's DDIM `add_noise`), predict ε through the UNet, regress ε→noise (MSE/MAE), and step
//!     the adapter factors. Gradient accumulation, LR schedule, grad-norm clipping, and intermediate
//!     checkpoints all match the MLX twin.
//!  3. **Save** — a PEFT `.safetensors` (`save_lora_peft` / `save_lokr`).
//!
//! **Conditioning matches candle *inference*, not the MLX twin.** The candle SDXL UNet forward is
//! `(noisy, timestep, encoder_hidden_states)` — it wires **no** pooled embed and **no** `time_ids`
//! (the stock candle-transformers SDXL UNet has no `add_embedding`; see [`crate::pipeline`]). So the
//! trainer feeds exactly that 3-tensor surface — optimizing the adapter against the precise function
//! that will load it, not the MLX 5-arg forward. (Wiring true SDXL micro-conditioning into candle is a
//! separate inference-side change.)
//!
//! **The eager-`Var` simplification.** Unlike the MLX trainer (which re-installs the factor map into
//! the model inside every `value_and_grad`), candle's adapters are storage-sharing `Var`s installed
//! **once** ([`build_lora_targets`]/[`build_lokr_targets`]): each forward re-reads the current factor
//! storage and `loss.backward()` attributes grads straight to the `Var`s, so the per-step body is just
//! forward → backward → clip → step, with no re-install.
//!
//! **Gradient checkpointing** (`config.gradient_checkpointing`) routes the backward through
//! [`checkpointed_backward`] over the UNet's block segments ([`UNet2DConditionModel::block_segments`]),
//! recomputing each down/mid/up block instead of retaining its activations — numerically the dense
//! grads (a `tests` parity gate asserts this), at the cost of one extra forward.

use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{Module, VarBuilder};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use tokenizers::Tokenizer;

use candle_gen::gen_core::train::{
    NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, WeightsSource};
use candle_gen::train::checkpoint::{checkpoint_filename, file_stem};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::gradient_checkpoint::{checkpointed_backward, Segment};
use candle_gen::train::lora::{
    build_lokr_targets, build_lora_targets, save_lokr, save_lora_peft, AdapterKind, LoraSet,
    SDXL_ATTN_TARGETS, SDXL_PEFT_PREFIX,
};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen::train::schedule::{lr_multiplier, schedule_updates};
use candle_gen::{CandleError, Result};

use candle_transformers::models::stable_diffusion::ddim::DDIMSchedulerConfig;
use candle_transformers::models::stable_diffusion::schedulers::{Scheduler, SchedulerConfig};
use candle_transformers::models::stable_diffusion::{self, clip, StableDiffusionConfig};

use crate::pipeline::{hf_get, snapshot_file, Clip, VAE_FIX_FILE, VAE_FIX_REPO, VAE_SCALE};
use crate::unet::{
    BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig, VaeMomentsEncoder,
};
use crate::MODEL_ID;

/// DDPM training-noise schedule length (the diffusers `num_train_timesteps`). `t` is sampled uniform
/// over `[0, NUM_TRAIN_TIMESTEPS)`, matching torch's `randint(0, num_train_timesteps)`.
const NUM_TRAIN_TIMESTEPS: usize = 1000;

/// The exact SDXL UNet config (`stabilityai/stable-diffusion-xl-base-1.0/unet/config.json`) — 3 blocks
/// `320/640/1280` with transformer depths `[—, 2, 10]` and 5/10/20 attention heads, `cross_attention_dim
/// 2048`, linear projection. Mirrors candle's `StableDiffusionConfig::sdxl` UNet sub-config so the
/// vendored UNet loads the stock `unet/` safetensors unchanged.
fn sdxl_unet_config() -> UNet2DConditionModelConfig {
    let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    UNet2DConditionModelConfig {
        blocks: vec![
            bc(320, None, 5),
            bc(640, Some(2), 10),
            bc(1280, Some(10), 20),
        ],
        center_input_sample: false,
        cross_attention_dim: 2048,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: true,
    }
}

/// `"bf16"`/`"bfloat16"` → [`DType::BF16`]; anything else → [`DType::F32`] (the gen-core contract:
/// unrecognized = f32). The adapter factors / loss / grads stay f32 regardless (master weights).
fn parse_compute_dtype(s: &str) -> DType {
    let t = s.trim();
    if t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16") {
        DType::BF16
    } else {
        DType::F32
    }
}

/// The two SDXL CLIP text encoders + their tokenizers, loaded f32 for clean cached embeddings.
struct DualClip {
    l: clip::ClipTextTransformer,
    g: clip::ClipTextTransformer,
    tok_l: Tokenizer,
    tok_g: Tokenizer,
    l_max: usize,
    l_pad: u32,
    g_max: usize,
    g_pad: u32,
    device: Device,
}

impl DualClip {
    /// Load CLIP-L (`text_encoder/`) + CLIP-bigG (`text_encoder_2/`) from the snapshot and their
    /// tokenizers from HF — the same sources the inference [`crate::pipeline`] uses, at f32.
    fn load(root: &Path, device: &Device) -> Result<Self> {
        let cfg = StableDiffusionConfig::sdxl(None, None, None);
        let l_cfg = &cfg.clip;
        let g_cfg = cfg
            .clip2
            .as_ref()
            .ok_or_else(|| CandleError::Msg("sdxl config missing clip2".into()))?;
        let (l_tok_repo, l_weights) = Clip::L.sources();
        let (g_tok_repo, g_weights) = Clip::BigG.sources();
        let tok_l = Tokenizer::from_file(hf_get(l_tok_repo, "tokenizer.json")?)
            .map_err(|e| CandleError::Msg(format!("load tokenizer {l_tok_repo}: {e}")))?;
        let tok_g = Tokenizer::from_file(hf_get(g_tok_repo, "tokenizer.json")?)
            .map_err(|e| CandleError::Msg(format!("load tokenizer {g_tok_repo}: {e}")))?;
        let l = stable_diffusion::build_clip_transformer(
            l_cfg,
            snapshot_file(root, l_weights)?,
            device,
            DType::F32,
        )?;
        let g = stable_diffusion::build_clip_transformer(
            g_cfg,
            snapshot_file(root, g_weights)?,
            device,
            DType::F32,
        )?;
        let l_pad = pad_id(&tok_l, l_cfg)?;
        let g_pad = pad_id(&tok_g, g_cfg)?;
        Ok(Self {
            l,
            g,
            tok_l,
            tok_g,
            l_max: l_cfg.max_position_embeddings,
            l_pad,
            g_max: g_cfg.max_position_embeddings,
            g_pad,
            device: device.clone(),
        })
    }

    /// Encode `caption` (no negative — training is CFG-off) to the SDXL dual-CLIP conditioning
    /// `[1, 77, 2048]` = `cat([clip_L_hidden, clip_bigG_hidden], dim=-1)`, exactly the inference
    /// `text_embeddings` concat (minus the `[uncond, cond]` batch stack).
    fn encode(&self, caption: &str) -> Result<Tensor> {
        let lt = tokenize_padded(&self.tok_l, self.l_max, self.l_pad, caption, &self.device)?;
        let gt = tokenize_padded(&self.tok_g, self.g_max, self.g_pad, caption, &self.device)?;
        let l = self.l.forward(&lt)?;
        let g = self.g.forward(&gt)?;
        Ok(Tensor::cat(&[l, g], D::Minus1)?)
    }
}

/// Resolve a CLIP encoder's pad-token id from its tokenizer vocab + config `pad_with` (default
/// `<|endoftext|>`).
fn pad_id(tok: &Tokenizer, cfg: &clip::Config) -> Result<u32> {
    let pad_token = cfg
        .pad_with
        .clone()
        .unwrap_or_else(|| "<|endoftext|>".into());
    tok.get_vocab(true)
        .get(pad_token.as_str())
        .copied()
        .ok_or_else(|| CandleError::Msg(format!("pad token {pad_token:?} not in CLIP vocab")))
}

/// Tokenize `text`, pad to `max` with `pad`, and return a `[1, max]` id tensor (the inference encode
/// closure, factored). Errors if the prompt exceeds the encoder's context.
fn tokenize_padded(
    tok: &Tokenizer,
    max: usize,
    pad: u32,
    text: &str,
    device: &Device,
) -> Result<Tensor> {
    let mut tokens = tok
        .encode(text, true)
        .map_err(|e| CandleError::Msg(format!("tokenize: {e}")))?
        .get_ids()
        .to_vec();
    if tokens.len() > max {
        return Err(CandleError::Msg(format!(
            "caption too long: {} tokens > {max}",
            tokens.len()
        )));
    }
    while tokens.len() < max {
        tokens.push(pad);
    }
    Ok(Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?)
}

/// A uniform integer DDPM timestep in `[0, NUM_TRAIN_TIMESTEPS)`, deterministic in `seed` (the
/// sc-3673 CPU-`StdRng` discipline, NOT candle's device RNG) — the candle analog of the MLX trainer's
/// `sample_timestep` and torch's `randint(0, num_train_timesteps)`.
fn sample_timestep(seed: u64) -> usize {
    StdRng::seed_from_u64(seed).random_range(0..NUM_TRAIN_TIMESTEPS)
}

/// Deterministic `N(0, 1)` noise of the given shape, drawn from a seeded CPU `StdRng` then moved to
/// `device` (sc-3673 launch-portable discipline). Used as both the diffusion noise and the ε target.
fn sample_noise(shape: &[usize], seed: u64, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?.to_device(device)?)
}

/// ε-prediction loss in f32: `mean((eps - noise)²)` (MSE) or `mean|eps - noise|` (MAE). `eps` (the
/// UNet output, in the compute dtype) is promoted to f32 so the loss/grads stay f32 (master weights).
fn eps_loss(eps: &Tensor, target_f32: &Tensor, mae: bool) -> candle_core::Result<Tensor> {
    let diff = (eps.to_dtype(DType::F32)? - target_f32)?;
    if mae {
        diff.abs()?.mean_all()
    } else {
        diff.sqr()?.mean_all()
    }
}

/// One micro-step's forward+backward over the installed adapter `Var`s: build the noised latent at
/// timestep `t`, predict ε, regress it toward `noise`, and return `(loss, grads)` — `grads` keyed by
/// the adapter `Var`s (ready for [`clip_grad_norm`] + [`TrainOptimizer::step`]).
///
/// `use_checkpoint` selects the gradient-checkpointed backward (recompute each block) over the dense
/// `loss.backward()`; both yield the same grads (the `tests` parity gate). A free function (not a
/// method) so the parity test can drive it against a tiny synthetic UNet without real weights.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    unet: &UNet2DConditionModel,
    scheduler: &dyn Scheduler,
    lora_vars: &[Var],
    x0: &Tensor,
    cond: &Tensor,
    t: usize,
    noise: &Tensor,
    mae: bool,
    compute_dtype: DType,
    use_checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let noisy = scheduler
        .add_noise(x0, noise.clone(), t)?
        .to_dtype(compute_dtype)?;
    let target = noise.to_dtype(DType::F32)?;
    let cond_c = cond.to_dtype(compute_dtype)?;
    let bsize = x0.dim(0)?;
    let device = x0.device();
    let t_f64 = t as f64;

    if use_checkpoint {
        // Frozen, constant time embedding shared across every block segment (detached so it is not
        // recomputed per segment-backward).
        let emb = unet
            .time_embed(t_f64, bsize, compute_dtype, device)?
            .detach();
        let h0 = unet.conv_in_forward(&noisy)?;
        let mut segs: Vec<Segment> = unet.block_segments(&emb, &cond_c);
        // Final segment: the frozen head + the ε→noise regression, mapping [final_hidden] → [loss].
        let target_owned = target.clone();
        segs.push(Box::new(move |st: &[Tensor]| {
            let eps = unet.head_out(&st[0])?;
            Ok(vec![eps_loss(&eps, &target_owned, mae)?])
        }));
        // State seed: hidden = conv_in_out, res₀ = conv_in_out (the UNet's `down_block_res_xs[0]`).
        let inputs = [h0.clone(), h0];
        checkpointed_backward(&segs, &inputs, lora_vars)
    } else {
        let eps = unet.forward(&noisy, t_f64, &cond_c)?;
        let loss = eps_loss(&eps, &target, mae)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        Ok((loss_val, grads))
    }
}

/// Resolve the config's target-module suffixes (default [`SDXL_ATTN_TARGETS`]) to full UNet attention
/// paths, then prune for the adapter kind's inference surface: LoRA keeps the **complete** down/mid/up
/// surface; LoKr excludes `mid_block` (the SDXL LoKr inference loader skips it — sc-2640 — so a
/// mid_block LoKr factor would never load; keeping train/inference in lock-step). Returned paths are
/// full dotted paths, which [`build_lora_targets`] matches exactly.
fn resolve_target_paths(
    unet: &mut UNet2DConditionModel,
    cfg: &TrainingConfig,
) -> Result<Vec<String>> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
        SDXL_ATTN_TARGETS.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    let paths: Vec<String> = unet
        .lora_target_paths()?
        .into_iter()
        .filter(|p| {
            suffixes
                .iter()
                .any(|s| p == s || p.ends_with(&format!(".{s}")))
        })
        .filter(|p| match cfg.network_type {
            NetworkType::Lokr => !p.contains("mid_block"),
            NetworkType::Lora => true,
        })
        .collect();
    if paths.is_empty() {
        return Err(CandleError::Msg(format!(
            "sdxl trainer: no adapter targets matched {suffixes:?}"
        )));
    }
    Ok(paths)
}

/// Build the vendored, trainable SDXL UNet from the snapshot `unet/` safetensors at `dtype`. Flash
/// attention is off (non-differentiable; training uses the materialized math attention).
fn build_unet(root: &Path, device: &Device, dtype: DType) -> Result<UNet2DConditionModel> {
    let path = snapshot_file(root, "unet/diffusion_pytorch_model.fp16.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    Ok(UNet2DConditionModel::new(
        vb,
        4,
        4,
        false,
        sdxl_unet_config(),
    )?)
}

/// Create the output directory, mapping the `io::Error` into the crate error (candle's `CandleError`
/// has no `From<io::Error>`).
fn create_output_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CandleError::Msg(format!("create output dir {}: {e}", dir.display())))
}

/// Write the adapter as a PEFT `.safetensors`: LoRA under the SDXL key prefix, LoKr with bare keys.
fn save_adapter(set: &LoraSet, cfg: &TrainingConfig, path: &Path) -> Result<()> {
    let meta = std::collections::HashMap::new();
    match set.kind {
        AdapterKind::Lora => save_lora_peft(set, SDXL_PEFT_PREFIX, &meta, path),
        AdapterKind::Lokr => {
            let _ = cfg; // decompose factor already carried on the set
            save_lokr(set, &meta, path)
        }
    }
}

/// Identity + capabilities of the candle SDXL trainer: LoRA + LoKr, `backend = "candle"` (the two
/// backend-correct deviations from `mlx-gen-sdxl`'s descriptor are `backend` and the host platform).
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// A loaded candle SDXL trainer. Loading is **lazy** (no file I/O — mirrors [`crate::SdxlGenerator`]):
/// the heavy VAE/CLIP/UNet are built inside [`train`](Trainer::train) at the request's compute dtype.
pub struct SdxlTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) candle SDXL trainer from a [`LoadSpec`] whose `weights` is the SDXL snapshot
/// directory (the diffusers multi-component tree: `text_encoder/ text_encoder_2/ unet/ …`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(CandleError::Msg(
            "sdxl trainer expects a snapshot directory (text_encoder/ text_encoder_2/ unet/ …), \
                 not a single .safetensors file"
                .into(),
        )),
    };
    Ok(Box::new(SdxlTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

/// Registry adapter: the trainer registry's `load` slot is typed on [`gen_core::Result`]; bridge the
/// crate's rich-`Result` [`load_trainer`] into it.
fn load_trainer_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    load_trainer(spec).map_err(Into::into)
}

// Link-time self-registration into gen-core's trainer registry (parallel to the generator's
// `ModelRegistration` in `lib.rs`). Kept linked by `crate::force_link`.
inventory::submit! {
    gen_core::registry::TrainerRegistration { descriptor: trainer_descriptor, load: load_trainer_registered }
}

impl Trainer for SdxlTrainer {
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

impl SdxlTrainer {
    /// Reject a request before any expensive load (empty dataset, zero rank, unsupported optimizer).
    fn validate_impl(&self, req: &TrainingRequest) -> Result<()> {
        if req.items.is_empty() {
            return Err(CandleError::Msg("sdxl trainer: dataset is empty".into()));
        }
        if req.config.rank == 0 {
            return Err(CandleError::Msg("sdxl trainer: rank must be > 0".into()));
        }
        if !TrainOptimizer::is_supported(&req.config.optimizer) {
            return Err(CandleError::Msg(format!(
                "sdxl trainer: optimizer '{}' is not available (supported: adamw, adam, rose, prodigy)",
                req.config.optimizer
            )));
        }
        Ok(())
    }

    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`], keeping `?` on candle/family helpers transparent here.
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

        // --- load + cache: VAE latents (.mean × scale) + dual-CLIP conditioning ---
        on_progress(TrainingProgress::LoadingModel);
        let vae = {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[hf_get(VAE_FIX_REPO, VAE_FIX_FILE)?],
                    DType::F32,
                    device,
                )?
            };
            VaeMomentsEncoder::new(vb, VAE_SCALE)?
        };
        let clip = DualClip::load(&self.root, device)?;

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
            let x0 = vae.encode_mean(&img)?;
            let cond = clip.encode(&item.caption)?;
            cache.push((x0, cond));
        }
        // Encoders are dead weight once the dataset is cached — drop them before the UNet loads.
        drop(clip);
        drop(vae);
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            return Err(CandleError::Msg(
                "sdxl trainer: no usable dataset items".into(),
            ));
        }

        // --- build UNet + install adapters ---
        let mut unet = build_unet(&self.root, device, compute_dtype)?;
        let target_paths = resolve_target_paths(&mut unet, cfg)?;
        let lora_set = match cfg.network_type {
            NetworkType::Lora => build_lora_targets(
                &mut unet,
                &target_paths,
                cfg.rank,
                cfg.alpha,
                cfg.seed,
                device,
            )?,
            NetworkType::Lokr => build_lokr_targets(
                &mut unet,
                &target_paths,
                cfg.rank,
                cfg.alpha,
                cfg.decompose_factor,
                cfg.seed,
                device,
            )?,
        };
        let use_checkpoint = cfg.gradient_checkpointing;

        // --- scheduler + optimizer + schedule ---
        let scheduler = DDIMSchedulerConfig::default().build(NUM_TRAIN_TIMESTEPS)?;
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
            let (x0, cond) = &cache[((step - 1) as usize) % cache.len()];
            let t = sample_timestep(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64));
            let noise = sample_noise(
                x0.dims(),
                cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                device,
            )?;
            let (loss, grads) = compute_loss_grads(
                &unet,
                scheduler.as_ref(),
                &lora_set.vars,
                x0,
                cond,
                t,
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
                save_adapter(&lora_set, cfg, &ckpt)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // Cancelled before a single step completed: the factors are still the no-op init (`B = 0`),
        // so surface the typed cancellation rather than shipping an identity adapter as a result.
        if steps_run == 0 {
            return Err(CandleError::Canceled);
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        create_output_dir(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_adapter(&lora_set, cfg, &adapter_path)?;
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
    use candle_core::{DType, Device};
    use candle_gen::gen_core::registry;
    use candle_nn::{VarBuilder, VarMap};

    /// A tiny SDXL-shaped UNet (one cross-attn down block + one basic block + cross-attn mid/up) that
    /// exercises the LoRA seam + the block-segment checkpointing cheaply on CPU, with random weights.
    fn tiny_unet(vb: VarBuilder) -> UNet2DConditionModel {
        let cfg = UNet2DConditionModelConfig {
            blocks: vec![
                BlockConfig {
                    out_channels: 32,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 64,
                    use_cross_attn: None,
                    attention_head_dim: 8,
                },
            ],
            center_input_sample: false,
            cross_attention_dim: 64,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            layers_per_block: 1,
            mid_block_scale_factor: 1.,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size: None,
            use_linear_projection: false,
        };
        UNet2DConditionModel::new(vb, 4, 4, false, cfg).unwrap()
    }

    fn grad_vec(grads: &GradStore, v: &Var) -> Vec<f32> {
        grads
            .get(v.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The correctness gate for the `gradient_checkpointing` lever: the checkpointed backward
    /// (block-recompute) must reproduce the dense `loss.backward()` grads (mod float reassociation),
    /// over a real SDXL-shaped UNet with nonzero LoRA factors so every adapter `Var` carries signal.
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut unet = tiny_unet(vb);
        let paths = unet.lora_target_paths().unwrap();
        let set = build_lora_targets(&mut unet, &paths, 4, 8.0, 7, &dev).unwrap();
        // Move B off zero so both A and B grads are nonzero (a no-op-init adapter would zero A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }

        let scheduler = DDIMSchedulerConfig::default()
            .build(NUM_TRAIN_TIMESTEPS)
            .unwrap();
        let x0 = Tensor::randn(0f32, 1f32, (1, 4, 16, 16), &dev).unwrap();
        let cond = Tensor::randn(0f32, 1f32, (1, 7, 64), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, 4, 16, 16), &dev).unwrap();

        let (loss_d, g_d) = compute_loss_grads(
            &unet,
            scheduler.as_ref(),
            &set.vars,
            &x0,
            &cond,
            500,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        let (loss_c, g_c) = compute_loss_grads(
            &unet,
            scheduler.as_ref(),
            &set.vars,
            &x0,
            &cond,
            500,
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
                    "grad mismatch (dense {x} vs checkpoint {y})"
                );
                if x.abs() > 1e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected nonzero adapter grads to compare");
    }

    /// `sample_timestep` is deterministic in its seed and lands in `[0, NUM_TRAIN_TIMESTEPS)`.
    #[test]
    fn timestep_is_deterministic_and_in_range() {
        for seed in [0u64, 1, 42, 9999] {
            let a = sample_timestep(seed);
            let b = sample_timestep(seed);
            assert_eq!(a, b);
            assert!(a < NUM_TRAIN_TIMESTEPS);
        }
        // Distinct seeds generally differ (not a hard guarantee, but a sanity check on the mapping).
        assert_ne!(sample_timestep(1), sample_timestep(2));
    }

    /// The trainer self-registers and resolves through gen-core's trainer registry as the candle
    /// SDXL trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer("sdxl", &spec).expect("candle sdxl trainer is registered");
        assert_eq!(t.descriptor().id, "sdxl");
        assert_eq!(t.descriptor().backend, "candle");
        assert!(t.descriptor().supports_lora);
        assert!(t.descriptor().supports_lokr);
    }

    /// `validate` rejects an empty dataset, zero rank, and an unsupported optimizer before any load.
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer("sdxl", &spec).unwrap();

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

        let empty = TrainingRequest {
            items: vec![],
            ..base.clone()
        };
        assert!(t.validate(&empty).is_err());

        let zero_rank = TrainingRequest {
            config: TrainingConfig {
                rank: 0,
                ..TrainingConfig::default()
            },
            ..base.clone()
        };
        assert!(t.validate(&zero_rank).is_err());

        let bad_opt = TrainingRequest {
            config: TrainingConfig {
                optimizer: "lion".into(),
                ..TrainingConfig::default()
            },
            ..base
        };
        assert!(t.validate(&bad_opt).is_err());
    }
}
