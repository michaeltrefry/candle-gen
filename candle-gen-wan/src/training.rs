//! The candle **Wan2.2 A14B MoE LoRA/LoKr trainer** (sc-5167) — the candle twin of `mlx-gen-wan`'s
//! `WanMoeTrainer`, implementing the backend-neutral [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer)
//! with `backend = "candle"`. It retires the worker's Python torch `WanMoeLoraTrainer` (the dual-expert
//! path of epic 5164) and reuses the shared [`candle_gen::train`] harness the SDXL/Z-Image stories
//! established, building on [`crate::dit_train`]'s vendored trainable DiT.
//!
//! Registered under `"wan2_2_t2v_14b"` — the **T2V** A14B. (The I2V channel-concat conditioning and the
//! dense 5B path — the latter blocked on a z48 VAE *encoder* port — are sc-5167 follow-ups.)
//!
//! ## Why Wan keeps a bespoke loop (sc-7787)
//!
//! The single-model flow-match driver ([`candle_gen::train::flow_match::run_flow_match_training`]) the
//! Z-Image/Lens/Krea trainers adopted assumes **one** DiT, optimizer, adapter set, and timestep range.
//! The Wan A14B is a **dual-expert MoE**: it alternates a high-noise expert (`transformer/`) and a
//! low-noise expert (`transformer_2/`), each with its own LoRA/LoKr, optimizer, LR schedule, timestep
//! band, and gradient-accumulation buffer, and emits an expert-suffixed save pair. That does not fit the
//! single-model driver cleanly, so Wan keeps the alternating loop below and consumes only the **Tier-1
//! shared helpers** from [`flow_match`] (sampling, batch math, validation, snapshot IO, adapter install
//! + optimizer step). Exactly the split sc-7787 sanctions.
//!
//! ## The Wan realities that shape it
//!
//! Cache → loop → save, on the **flow-match** objective. The two Wan-specific twists vs the Z-Image
//! trainer:
//!  1. **No velocity negation.** Wan feeds the transformer output to the flow-match step *without*
//!     negation (opposite of Z-Image's `noise_pred.neg()`), so the trainer regresses the **raw** DiT
//!     velocity toward `noise − x0` (`target = noise − x0`). The timestep fed to the DiT is `t · 1000`
//!     (the `[0, NUM_TRAIN_TIMESTEPS]` integer convention), not `1 − σ` — see [`compute_loss_grads`].
//!  2. **MoE dual-expert.** The A14B denoises with a **high-noise** expert (`transformer/`, timestep
//!     ≥ `boundary·1000`) and a **low-noise** expert (`transformer_2/`, below it). Each gets its **own**
//!     LoRA/LoKr (separate factor map + optimizer + LR schedule + timestep band). Training **alternates**
//!     per step (odd → high, even → low), sampling that expert's band, and emits a `{stem}.high_noise` /
//!     `{stem}.low_noise` pair — what the inference loader ([`crate::adapters`] via [`crate::wan14b`])
//!     merges back onto the matching expert ([`MoeExpert`](candle_gen::gen_core::MoeExpert)).
//!
//! Caching: each still image is z16-VAE-encoded to a single-frame latent `[1, 16, 1, h, w]` (the
//! deterministic posterior **mean**, normalized) and its caption UMT5-encoded to `[1, 512, 4096]`
//! (zero-padded to 512, the same context surface inference feeds). The VAE + text encoder are dropped
//! after caching; the two experts are the working set.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor, Var};

use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, WeightsSource};
use candle_gen::train::checkpoint::file_stem;
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{self, validate_flow_match_request, velocity_loss};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::train::lora::LoraSet;
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen::train::schedule::schedule_updates;
use candle_gen::{CandleError, Result};

use crate::config::{
    TextEncoderConfig, TransformerConfig, Vae16Config, MODEL_ID_T2V_14B, NUM_TRAIN_TIMESTEPS,
    T2V_14B_BOUNDARY,
};
use crate::dit_train::{WanTransformerTrain, WAN_ATTN_TARGETS};
use crate::rope::WanRope;
use crate::text_encoder::Umt5Encoder;
use crate::vae16::WanVae16;

/// Error-message prefix shared by [`validate_flow_match_request`] and the component-IO helpers.
const LABEL: &str = "wan trainer";

/// Sample a timestep `t ∈ [lo, hi)` inside an expert's noise band: draw a unit `t_unit`
/// ([`flow_match::sample_unit_timestep`]) then affine-map it into `(lo, hi)`. The high-noise expert
/// samples `(boundary, 1)`, the low-noise `(0, boundary)` — the per-expert split the A14B trains. This
/// band map is Wan-specific (the other flow-match trainers train one model over the full `(0, 1)`), so
/// it stays local; only the unit draw is shared.
fn sample_band_timestep(
    timestep_type: &str,
    timestep_bias: &str,
    band: (f64, f64),
    seed: u64,
) -> f64 {
    let t_unit = flow_match::sample_unit_timestep(timestep_type, timestep_bias, seed) as f64;
    let (lo, hi) = band;
    (lo + t_unit * (hi - lo)).clamp(1e-3, 1.0 - 1e-3)
}

/// One micro-step's forward+backward over one expert's installed adapter `Var`s: build the noised
/// latent at `t`, predict the **raw** velocity through the (LoRA-adapted) DiT, regress it toward
/// `noise − x0`, return `(loss, grads)` keyed by `lora_vars`. `cos`/`sin` are the (constant,
/// per-resolution) RoPE tables. A free function so the tests can drive it against a tiny DiT.
///
/// `use_checkpoint` selects the **gradient-checkpointed** backward. This is not just a memory lever for
/// the 14B experts — it is **required**: candle's matmul backward materializes a gradient for the
/// *frozen* base weight as well as the activation, so a dense 40-block backward holds ~40 layers of f32
/// base-weight grads at once (tens of GB), OOMing even a 96 GB card with both experts resident. The
/// checkpointed path runs the (adapter-free) pre-main forward detached, then segments the per-block
/// stack so only one block's transient weight-grads are live at a time (see
/// [`WanTransformerTrain::main_block_segments`]). Both paths yield the same adapter grads (the
/// `dense_and_checkpoint_grads_match` test pins this on a tiny DiT).
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &WanTransformerTrain,
    lora_vars: &[Var],
    x0: &Tensor,
    umt5: &Tensor,
    t: f64,
    noise: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mae: bool,
    compute_dtype: DType,
    use_checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let (x_t, target) = flow_match::build_batch(x0, noise, t)?;
    let x_t = x_t.to_dtype(compute_dtype)?;
    // Text context + timestep are adapter-free constants the blocks consume.
    let ctx = dit.embed_text(umt5)?;
    let timestep = t * NUM_TRAIN_TIMESTEPS as f64;

    if use_checkpoint {
        // Pre-main (patch + time embed) has no adapters → run it detached; no upstream grads to stitch.
        let (hidden, mctx) = dit.forward_pre_main(&x_t, timestep)?;
        let hidden_d = hidden.detach();
        let mut segs = dit.main_block_segments(&mctx, &ctx, cos, sin);
        // Final segment: head → raw velocity (NO negation) → flow-match regression → [loss].
        let target_owned = target.clone();
        let mctx_ref = &mctx;
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], mctx_ref)?;
            Ok(vec![velocity_loss(&v, &target_owned, mae)?])
        }));
        checkpointed_backward(&segs, std::slice::from_ref(&hidden_d), lora_vars)
    } else {
        // Dense backward (tiny models / tests only — see the `use_checkpoint` note re: OOM at scale).
        let v = dit.forward(&x_t, &ctx, timestep, cos, sin)?;
        let loss = velocity_loss(&v, &target, mae)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        Ok((loss_val, grads))
    }
}

/// Tokenize + UMT5-encode `caption` → `[1, 512, 4096]` (f32, zero-padded to 512 — the same context
/// surface inference feeds; see [`crate::wan14b`]'s `encode`).
fn encode_caption(
    root: &Path,
    te_cfg: &TextEncoderConfig,
    te: &Umt5Encoder,
    caption: &str,
    device: &Device,
) -> Result<Tensor> {
    let tok = TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: te_cfg.max_length,
            pad_token_id: te_cfg.pad_token_id,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(|e| CandleError::Msg(format!("wan trainer: load tokenizer: {e}")))?;
    let out = tok
        .tokenize(caption)
        .map_err(|e| CandleError::Msg(format!("wan trainer: tokenize: {e}")))?;
    let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    if ids.is_empty() {
        // Empty caption → zero ids → a degenerate `(1,1)` tensor (the old `.max(1)` padded the
        // shape, not the data) whose 0-element f32 embedding gather reads out of bounds on CUDA
        // (`CUDA_ERROR_ILLEGAL_ADDRESS`, surfacing as a misleading cublas failure). Emit one pad
        // token so a 0-length sequence never reaches the gather. (sc-7078)
        ids.push(te_cfg.pad_token_id as u32);
    }
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), device)?;
    let embeds = te.encode(&input_ids)?.to_dtype(DType::F32)?; // [1, L, 4096]
    let max_len = te_cfg.max_length;
    let dim = embeds.dim(2)?;
    match len.cmp(&max_len) {
        std::cmp::Ordering::Less => {
            let pad = Tensor::zeros((1, max_len - len, dim), DType::F32, device)?;
            Ok(Tensor::cat(&[&embeds, &pad], 1)?)
        }
        std::cmp::Ordering::Greater => Ok(embeds.narrow(1, 0, max_len)?),
        std::cmp::Ordering::Equal => Ok(embeds),
    }
}

/// Insert `.{suffix}` before the extension of `file_name` (`a.safetensors` → `a.high_noise.safetensors`).
fn with_expert_suffix(file_name: &str, suffix: &str) -> String {
    match file_name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.{suffix}.{ext}"),
        None => format!("{file_name}.{suffix}"),
    }
}

/// One MoE expert's full trainable state: the (vendored) DiT with adapters installed, its optimizer +
/// LR schedule, its timestep band, and its own gradient-accumulation buffer + step counters.
struct ExpertState {
    dit: WanTransformerTrain,
    set: LoraSet,
    opt: TrainOptimizer,
    band: (f64, f64),
    accumulated: Option<GradStore>,
    micro: u32,
    update_idx: u32,
    total_updates: u32,
    warmup_updates: u32,
    /// `"high_noise"` / `"low_noise"` — the saved-file suffix + the [`MoeExpert`] the inference loader
    /// merges this onto.
    suffix: &'static str,
}

/// Identity + capabilities of the candle Wan A14B trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID_T2V_14B,
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// A loaded candle Wan A14B (T2V) MoE trainer. Loading is **lazy** — the heavy VAE / text-encoder / two
/// experts are built inside [`train`](Trainer::train) at the request's compute dtype.
pub struct WanMoeTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Wan A14B trainer from a [`LoadSpec`] whose `weights` is the A14B snapshot
/// directory (`tokenizer/ text_encoder/ transformer/ transformer_2/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(CandleError::Msg(
                "wan2_2_t2v_14b trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ transformer_2/ vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    Ok(Box::new(WanMoeTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

// Link-time self-registration into gen-core's trainer registry (kept linked by `crate::force_link`).
// `register_trainer!` bridges the crate's rich `Result` into `gen_core::Result` via `Into::into`.
candle_gen::register_trainer! { trainer_descriptor => load_trainer }

impl Trainer for WanMoeTrainer {
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
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl WanMoeTrainer {
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        validate_flow_match_request(req, LABEL)?;
        let cfg = &req.config;
        let device = &self.device;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);
        let compute_dtype = flow_match::parse_compute_dtype(&cfg.train_dtype);
        let dit_cfg = TransformerConfig::t2v_14b();

        // --- load + cache: z16 VAE latent means + UMT5 caption embeds (both f32) ---
        on_progress(TrainingProgress::LoadingModel);
        let vae_cfg = Vae16Config::wan21();
        let vae = WanVae16::new_with_encoder(
            &vae_cfg,
            flow_match::component_vb(&self.root, "vae", device, DType::F32, LABEL)?,
        )?;
        let te_cfg = TextEncoderConfig::umt5_xxl();
        // Load the UMT5 encoder at bf16 for caching (11 GB vs 22 GB at f32) — it only produces the
        // cached caption embeds (kept f32 in the cache), and is dropped before the experts load.
        let text_encoder = Umt5Encoder::new(
            &te_cfg,
            flow_match::component_vb(&self.root, "text_encoder", device, DType::BF16, LABEL)?,
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
            let img = load_image_tensor(&item.image_path, edge, device)?; // [1,3,edge,edge] in [-1,1]
            let video = img.unsqueeze(2)?; // [1,3,1,edge,edge] (T=1 still frame)
            let x0 = vae.encode(&video)?.to_dtype(DType::F32)?; // [1,16,1,h,w] normalized mean
            let cap = encode_caption(&self.root, &te_cfg, &text_encoder, &item.caption, device)?;
            cache.push((x0, cap));
        }
        drop(text_encoder);
        drop(vae);
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            return Err(CandleError::Msg(
                "wan trainer: no usable dataset items".into(),
            ));
        }

        // RoPE tables for the (constant) token geometry — every cached latent shares one resolution.
        let (_, _, _, hl, wl) = cache[0].0.dims5()?;
        let (pt, ph, pw) = dit_cfg.patch;
        let (ppf, pph, ppw) = (1 / pt, hl / ph, wl / pw);
        let (cos, sin) = WanRope::new(&dit_cfg).cos_sin(ppf, pph, ppw, device)?;

        // --- build the two experts (transformer/ = high-noise, transformer_2/ = low-noise) ---
        let suffixes = flow_match::resolve_target_suffixes(cfg, &WAN_ATTN_TARGETS);
        let accum = cfg.gradient_accumulation.max(1);
        let weight_decay = flow_match::effective_weight_decay(cfg);
        let mae = flow_match::is_mae(cfg);
        let boundary = T2V_14B_BOUNDARY;

        // odd steps → high (≈ ceil(steps/2) micro-steps), even → low (≈ floor(steps/2)).
        let high_micro = cfg.steps.div_ceil(2);
        let low_micro = cfg.steps / 2;
        let mut experts: Vec<ExpertState> = Vec::with_capacity(2);
        for (idx, (sub, suffix, band, micro)) in [
            ("transformer", "high_noise", (boundary, 1.0), high_micro),
            ("transformer_2", "low_noise", (0.0, boundary), low_micro),
        ]
        .into_iter()
        .enumerate()
        {
            let mut dit = WanTransformerTrain::new(
                &dit_cfg,
                flow_match::component_vb(&self.root, sub, device, compute_dtype, LABEL)?,
            )?;
            // Distinct per-expert seed (so the two adapters don't init identically), reproducible.
            let seed = cfg
                .seed
                .wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9));
            let set = flow_match::install_adapters(&mut dit, cfg, &suffixes, seed, device)?;
            let opt = TrainOptimizer::from_config(
                &cfg.optimizer,
                set.vars.clone(),
                cfg.learning_rate,
                weight_decay,
            )?;
            let (total_updates, warmup_updates) =
                schedule_updates(micro.max(1), accum, cfg.lr_warmup_steps / 2);
            experts.push(ExpertState {
                dit,
                set,
                opt,
                band,
                accumulated: None,
                micro: 0,
                update_idx: 0,
                total_updates,
                warmup_updates,
                suffix,
            });
        }

        // --- train loop (alternating experts) ---
        // The 14B experts' dense backward is infeasible (candle materializes a grad for every frozen
        // base weight too), so training always uses the gradient-checkpointed backward.
        let use_checkpoint = true;
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let ei = if step % 2 == 1 { 0 } else { 1 }; // odd → high (experts[0]), even → low
            let (x0, cap) = &cache[((step - 1) as usize) % cache.len()];
            let band = experts[ei].band;
            let t = sample_band_timestep(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                band,
                flow_match::timestep_seed(cfg.seed, step),
            );
            let noise = flow_match::sample_noise(
                x0.dims(),
                flow_match::noise_seed(cfg.seed, step),
                device,
            )?;
            let (loss, grads) = compute_loss_grads(
                &experts[ei].dit,
                &experts[ei].set.vars,
                x0,
                cap,
                t,
                &noise,
                &cos,
                &sin,
                mae,
                compute_dtype,
                use_checkpoint,
            )?;
            last_loss = loss;
            steps_run = step;

            let ex = &mut experts[ei];
            accumulate_grads(&mut ex.accumulated, grads, &ex.set.vars)?;
            ex.micro += 1;
            if ex.micro.is_multiple_of(accum) {
                apply_update(ex, accum, cfg)?;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                flow_match::create_output_dir(&req.output_dir)?;
                for ex in &experts {
                    let name = with_expert_suffix(
                        &format!("{}-step{step:06}.safetensors", file_stem(&req.file_name)),
                        ex.suffix,
                    );
                    flow_match::save_adapter(&ex.set, &HashMap::new(), &req.output_dir.join(name))?;
                }
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        if steps_run == 0 {
            return Err(CandleError::Canceled);
        }
        // Flush any expert's pending (sub-`accum`) accumulation so the final partial step is applied.
        for ex in &mut experts {
            if ex.accumulated.is_some() {
                apply_update(ex, accum, cfg)?;
            }
        }

        // --- save the high/low adapter pair; report the high-noise file as the primary path ---
        on_progress(TrainingProgress::Saving);
        flow_match::create_output_dir(&req.output_dir)?;
        let mut primary: Option<PathBuf> = None;
        for ex in &experts {
            let path = req
                .output_dir
                .join(with_expert_suffix(&req.file_name, ex.suffix));
            flow_match::save_adapter(&ex.set, &HashMap::new(), &path)?;
            if ex.suffix == "high_noise" {
                primary = Some(path);
            }
        }
        Ok(TrainingOutput {
            adapter_path: primary.expect("the high-noise expert is always present"),
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Fire one optimizer update for `ex`: delegates the average-clip-step to the shared
/// [`flow_match::apply_update`] (over `ex`'s own optimizer/accumulation/schedule), then advances the
/// expert's update counter.
fn apply_update(ex: &mut ExpertState, accum: u32, cfg: &TrainingConfig) -> Result<()> {
    flow_match::apply_update(
        &mut ex.opt,
        &mut ex.accumulated,
        &ex.set,
        accum,
        cfg,
        ex.update_idx,
        ex.total_updates,
        ex.warmup_updates,
    )?;
    ex.update_idx += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::gen_core::registry;
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::optim::clip_grad_norm;

    /// A tiny Wan-shaped DiT (z16, head_dim 128, 1 head, 1 layer) — exercises the real flow-match
    /// forward+backward on CPU.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 128,
            dim: 128,
            ffn_dim: 256,
            freq_dim: 256,
            text_dim: 64,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }

    /// Randomize every var in a fresh `VarMap` — `vb.get` raw tensors (notably the `patch_embedding`
    /// conv weight) default to ZERO-init, and a zero patch kernel makes `hidden ≡ 0`, which makes the
    /// LoRA adapters' inputs zero and their grads vacuously zero. Real training loads nonzero weights;
    /// the tiny tests must do the same to exercise the gradient path.
    fn randomize_base(vm: &VarMap, dev: &Device) {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), dev).unwrap())
                .unwrap();
        }
    }

    fn tiny_inputs(
        cfg: &TransformerConfig,
        dev: &Device,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let x0 = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 1, 4, 4), dev).unwrap();
        let umt5 = Tensor::randn(0f32, 1f32, (1, 3, cfg.text_dim), dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 1, 4, 4), dev).unwrap();
        let (cos, sin) = WanRope::new(cfg).cos_sin(1, 2, 2, dev).unwrap();
        (x0, umt5, noise, cos, sin)
    }

    /// ISOLATION: does `build_lora_targets` + `set.vars[i].set(..)` propagate to the installed
    /// LoraLinear's forward, with NO Wan dit involved? (Pins the harness mechanism in this crate.)
    #[test]
    fn harness_factor_set_propagates_to_forward() {
        use candle_gen::candle_nn::{Linear, Module};
        use candle_gen::train::lora::{LoraHost, LoraLinear};
        struct H(LoraLinear);
        impl LoraHost for H {
            fn visit_lora_mut(
                &mut self,
                f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
            ) -> candle_gen::Result<()> {
                f(&mut self.0)
            }
        }
        let dev = Device::Cpu;
        let w = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        let mut h = H(LoraLinear::from_linear(
            Linear::new(w, None),
            4,
            4,
            "to_q".into(),
        ));
        let set = build_lora_targets(&mut h, &["to_q".to_string()], 2, 4.0, 7, &dev).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 4), &dev).unwrap();
        let y0 =
            h.0.forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.5f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let y1 =
            h.0.forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        assert_ne!(
            y0, y1,
            "setting set.vars must change the installed LoraLinear forward"
        );
    }

    /// Band sampling is deterministic, in-band, and the high band lies above the low band.
    #[test]
    #[allow(clippy::manual_range_contains)] // the `±1e-6` tolerance reads clearer as explicit bounds
    fn band_timestep_is_in_band_and_ordered() {
        let hi = (T2V_14B_BOUNDARY, 1.0);
        let lo = (0.0, T2V_14B_BOUNDARY);
        for seed in [0u64, 1, 42, 9999] {
            let a = sample_band_timestep("sigmoid", "balanced", hi, seed);
            let b = sample_band_timestep("sigmoid", "balanced", hi, seed);
            assert_eq!(a, b, "same seed reproduces");
            assert!(
                a >= T2V_14B_BOUNDARY - 1e-6 && a < 1.0,
                "high band t out of range: {a}"
            );
            let l = sample_band_timestep("sigmoid", "balanced", lo, seed);
            assert!(
                l > 0.0 && l <= T2V_14B_BOUNDARY + 1e-6,
                "low band t out of range: {l}"
            );
        }
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var`.
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off its zero-init so both A and B grads are nonzero (a no-op-init adapter zeros A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, umt5, noise, cos, sin) = tiny_inputs(&cfg, &dev);
        let (loss, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        let mut saw_nonzero = false;
        for (i, v) in set.vars.iter().enumerate() {
            let g = grads
                .get(v.as_tensor())
                .unwrap_or_else(|| panic!("adapter var {i} has no gradient"));
            let gv = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                gv.iter().all(|x| x.is_finite()),
                "var {i} gradient has non-finite entries"
            );
            if gv.iter().any(|x| x.abs() > 1e-9) {
                saw_nonzero = true;
            }
        }
        assert!(
            saw_nonzero,
            "every adapter gradient was zero — backprop is not reaching the factors"
        );
        // 4 projections × 2 attentions (attn1 self + attn2 cross) × num_layers, ×2 factors.
        assert_eq!(set.vars.len(), 4 * 2 * cfg.num_layers * 2);
    }

    /// The correctness gate for the gradient-checkpointed backward (the path real training always uses):
    /// it must reproduce the dense `loss.backward()` grads (mod float reassociation) on the tiny DiT.
    /// All Wan adapters live in the checkpointed block stack (no retained pre-main adapters), so this
    /// spans every adapter target.
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, umt5, noise, cos, sin) = tiny_inputs(&cfg, &dev);
        let (loss_d, g_d) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        let (loss_c, g_c) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
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
        for (i, v) in set.vars.iter().enumerate() {
            let a = g_d
                .get(v.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b = g_c
                .get(v.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "grad mismatch for var {i} (dense {x} vs checkpoint {y})"
                );
                if x.abs() > 1e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected nonzero adapter grads to compare");
    }

    /// A few optimizer steps on a fixed batch lower the loss — the step descends the flow-match
    /// objective end to end through the harness.
    #[test]
    fn one_optimizer_step_descends() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, umt5, noise, cos, sin) = tiny_inputs(&cfg, &dev);
        let mut opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        let (loss0, mut grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        for _ in 0..5 {
            clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            let (_l, g) = compute_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &umt5,
                0.5,
                &noise,
                &cos,
                &sin,
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
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
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

    /// The trainer self-registers and resolves through gen-core's trainer registry as the candle Wan
    /// A14B trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(MODEL_ID_T2V_14B, &spec)
            .expect("candle wan a14b trainer is registered");
        assert_eq!(t.descriptor().id, MODEL_ID_T2V_14B);
        assert_eq!(t.descriptor().backend, "candle");
        assert_eq!(t.descriptor().modality, Modality::Video);
        assert!(t.descriptor().supports_lora && t.descriptor().supports_lokr);
    }

    /// `validate` rejects an empty dataset, zero rank/steps, an unsupported optimizer, and unrecognized
    /// timestep/loss knobs — before any load (now via the shared `flow_match::validate_flow_match_request`).
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        use candle_gen::gen_core::train::TrainingItem;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(MODEL_ID_T2V_14B, &spec).unwrap();
        let base = TrainingRequest {
            items: vec![TrainingItem {
                image_path: "/img.png".into(),
                caption: "x".into(),
            }],
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
        bad(&|r| r.config.loss_type = "huber".into());
    }

    /// The expert-suffix filename insertion lands before the extension.
    #[test]
    fn expert_suffix_naming() {
        assert_eq!(
            with_expert_suffix("mylora.safetensors", "high_noise"),
            "mylora.high_noise.safetensors"
        );
        assert_eq!(
            with_expert_suffix("mylora.safetensors", "low_noise"),
            "mylora.low_noise.safetensors"
        );
        assert_eq!(
            with_expert_suffix("noext", "high_noise"),
            "noext.high_noise"
        );
    }
}
