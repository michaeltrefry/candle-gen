//! The candle **Lens LoRA/LoKr trainer** (sc-5147) — the candle twin of the worker's Python torch
//! `lens_train_runner.py`, implementing the backend-neutral
//! [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer) with `backend = "candle"`. Together with
//! the inference cutover it retires `/opt/lens-venv` + `INCLUDE_LENS` — the last Python holdout for Lens
//! (epic 3482 / 5164). It reuses the shared [`candle_gen::train`] harness the SDXL/Z-Image/Wan stories
//! established, building on [`crate::dit_train`]'s vendored trainable DiT and [`crate::vae`]'s encode
//! shim.
//!
//! Since sc-7787 the cache → loop → save scaffolding lives in the shared single-model flow-match
//! driver ([`candle_gen::train::flow_match`]); this module supplies the Lens-specific hooks via
//! [`FlowMatchTrainer`] — caching, DiT construction, and the parity-critical [`compute_loss_grads`].
//!
//! Registered under `"lens"` — the **non-distilled** `microsoft/Lens` base (the de-distill lesson,
//! sc-1583; a LoRA trained here applies cleanly to `lens_turbo`, same architecture).
//!
//! ## The Lens recipe (from `lens_train_runner.py`)
//!
//! Cache → loop → save, on the **flow-match** objective:
//!  - **Flow-match, no negation.** `x_t = (1−t)·x0 + t·noise`, `target = noise − x0`; the DiT's **raw**
//!    velocity is regressed toward it (Lens feeds the transformer output to the scheduler *without*
//!    negation — opposite of Z-Image). The timestep `t ∈ (0, 1)` is fed to the DiT **directly** (no
//!    `1 − σ`, no `·1000`) — which (with the gradient-checkpoint split below) is why
//!    [`compute_loss_grads`] stays per-crate rather than collapsing into the shared driver.
//!  - **gpt-oss text front-end, cached + frozen.** Each caption is gpt-oss-encoded and its 4 selected
//!    layers ([`DEFAULT_SELECTED_LAYERS`] = 5/11/17/23) captured + cropped at [`TXT_OFFSET`] (the
//!    harmony-preamble offset) — exactly the inference `encode_one`. Cached once; the encoder is dropped
//!    before the DiT loads.
//!  - **Latents from a neural VAE encode.** Each image is `Flux2Vae`-encoded to the packed DiT latent
//!    `[1, S, 128]` ([`crate::vae::encode`], posterior mean) and cached.
//!  - **Targets:** the fused dual-stream attention projections [`LENS_ATTN_TARGETS`]
//!    (`img_qkv`/`txt_qkv`/`to_out.0`/`to_add_out`); train only the adapter, freeze the gpt-oss encoder
//!    + VAE + DiT base.
//!  - **Save** a diffusers-format `.safetensors` (bare dotted PEFT keys for LoRA / `lokr_w*` + metadata
//!    for LoKr) that the inference merge ([`crate::adapters`]) loads unchanged.
//!
//! The 48-block backward always runs **gradient-checkpointed** (candle's matmul backward materializes a
//! grad for the frozen base weight too, so a dense 48-block backward holds ~48 layers of weight-grads at
//! once — the Wan lesson). Adapter factors / loss / grads / optimizer state stay f32 (master weights);
//! the frozen base + activation stream follow `train_dtype` (bf16 default).

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor, Var};

use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, WeightsSource};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{
    self, run_flow_match_training, validate_flow_match_request, velocity_loss, FlowMatchTrainer,
};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::{CandleError, Result};

use crate::dit_train::{LensTransformerTrain, LENS_ATTN_TARGETS};
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::{Config as EncoderConfig, GptOssTextEncoder, DEFAULT_SELECTED_LAYERS};
use crate::transformer::LensDitConfig;
use crate::vae::{encode as vae_encode, Flux2Vae};
use crate::{DEFAULT_DATE, MODEL_ID_BASE};

/// Error-message prefix shared by [`validate_flow_match_request`] and the driver's `no usable dataset
/// items` guard.
const LABEL: &str = "lens trainer";

/// gpt-oss is encoded at bf16 for caching (it only produces the cached, frozen features; kept f32 in
/// the cache and dropped before the DiT loads).
const ENC_DTYPE: DType = DType::BF16;

/// One micro-step's forward+backward over the installed adapter `Var`s: build the noised latent at `t`,
/// predict the **raw** velocity through the (LoRA-adapted) DiT, regress it toward `noise − x0`, and
/// return `(loss, grads)` keyed by `lora_vars`. `(h, w)` is the (constant, per-resolution) latent grid;
/// `text_feats` are the cached, frozen gpt-oss features (any dtype — cast to `compute_dtype` here). A
/// free function so the tests can drive it against a tiny DiT.
///
/// `use_checkpoint` selects the **gradient-checkpointed** backward — required at scale, not just a memory
/// lever: candle's matmul backward materializes a gradient for the *frozen* base weight too, so a dense
/// 48-block backward holds ~48 layers of base-weight grads at once. The checkpointed path runs the
/// adapter-free pre-main forward detached, then segments the per-block stack so only one block's
/// transient weight-grads are live at a time (see [`LensTransformerTrain::main_block_segments`]). Both
/// paths yield the same adapter grads (the `dense_and_checkpoint_grads_match` test pins this).
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &LensTransformerTrain,
    lora_vars: &[Var],
    x0: &Tensor,
    text_feats: &[Tensor],
    h: usize,
    w: usize,
    t: f64,
    noise: &Tensor,
    mae: bool,
    compute_dtype: DType,
    use_checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let (x_t, target) = flow_match::build_batch(x0, noise, t)?;
    let x_t = x_t.to_dtype(compute_dtype)?;
    let feats: Vec<Tensor> = text_feats
        .iter()
        .map(|f| f.to_dtype(compute_dtype))
        .collect::<candle_gen::candle_core::Result<_>>()?;
    let timestep = t as f32; // fed to the DiT directly (no 1−σ, no ·1000)

    if use_checkpoint {
        // Pre-main (img/txt embeds, frozen) has no adapters → its `(hidden, encoder)` boundary is a
        // detached constant; the input cotangent is discarded.
        let (hidden, encoder, ctx) = dit.forward_pre_main(&x_t, &feats, None, timestep, 1, h, w)?;
        let hidden_d = hidden.detach();
        let encoder_d = encoder.detach();
        let mut segs = dit.main_block_segments(&ctx);
        // Final segment: head → raw velocity (NO negation) → flow-match regression → [loss].
        let target_owned = target.clone();
        let ctx_ref = &ctx;
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], ctx_ref)?;
            Ok(vec![velocity_loss(&v, &target_owned, mae)?])
        }));
        checkpointed_backward(&segs, &[hidden_d, encoder_d], lora_vars)
    } else {
        // Dense backward (tiny models / tests only — see the `use_checkpoint` note re: OOM at scale).
        let v = dit.forward(&x_t, &feats, None, timestep, 1, h, w)?;
        let loss = velocity_loss(&v, &target, mae)?;
        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        Ok((loss_val, grads))
    }
}

/// gpt-oss-encode `caption` → its 4 captured layers cropped at [`TXT_OFFSET`], each `[1, s, 2880]`
/// (f32, cached). Mirrors the inference `encode_one` (single prompt, unpadded). A caption whose token
/// length is `≤ TXT_OFFSET` (the harmony preamble alone) yields length-0 features — surfaced as an error
/// (an empty caption is a dataset bug, not silently trained on zero text).
fn encode_caption(
    tokenizer: &LensTokenizer,
    encoder: &GptOssTextEncoder,
    caption: &str,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let ids = tokenizer
        .encode(caption, DEFAULT_DATE)
        .map_err(|e| CandleError::Msg(format!("lens trainer: tokenize caption: {e}")))?;
    let l = ids.len();
    if l <= TXT_OFFSET {
        return Err(CandleError::Msg(format!(
            "lens trainer: caption {caption:?} tokenizes to {l} tokens (≤ the {TXT_OFFSET}-token \
             harmony preamble) — it carries no text features"
        )));
    }
    let input_ids = Tensor::from_vec(ids, (1, l), device)?;
    let layers = encoder.capture(&input_ids, &DEFAULT_SELECTED_LAYERS)?;
    let s = l - TXT_OFFSET;
    layers
        .iter()
        .map(|f| {
            Ok(f.narrow(1, TXT_OFFSET, s)?
                .to_dtype(DType::F32)?
                .contiguous()?)
        })
        .collect()
}

/// Identity + capabilities of the candle Lens trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID_BASE,
        family: "lens",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// A loaded candle Lens trainer. Loading is **lazy** — the gpt-oss encoder / VAE / DiT are built inside
/// [`train`](Trainer::train) at the request's compute dtype.
pub struct LensTrainer {
    descriptor: TrainerDescriptor,
    root: std::path::PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Lens trainer from a [`LoadSpec`] whose `weights` is the `microsoft/Lens`
/// snapshot directory (`tokenizer/ text_encoder/ transformer/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(CandleError::Msg(
                "lens trainer expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    Ok(Box::new(LensTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

// Link-time self-registration into gen-core's trainer registry (kept linked by `crate::force_link`).
// `register_trainer!` bridges the crate's rich `Result` into `gen_core::Result` via `Into::into`.
candle_gen::register_trainer! { trainer_descriptor => load_trainer }

impl Trainer for LensTrainer {
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

impl FlowMatchTrainer for LensTrainer {
    type Dit = LensTransformerTrain;
    /// `(x0 packed latent [1, S, 128], the 4 cached gpt-oss feature layers)`, both f32.
    type Cached = (Tensor, Vec<Tensor>);
    /// The (constant, per-resolution) latent grid `(lat_h, lat_w)`.
    type Aux = (usize, usize);
    const LABEL: &'static str = LABEL;

    fn device(&self) -> &Device {
        &self.device
    }

    fn default_targets(&self) -> &'static [&'static str] {
        &LENS_ATTN_TARGETS
    }

    fn cache(
        &self,
        req: &TrainingRequest,
        device: &Device,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<(Tensor, Vec<Tensor>)>, (usize, usize))> {
        let edge = bucket_resolution(req.config.resolution);
        let tokenizer =
            LensTokenizer::from_file(self.root.join("tokenizer").join("tokenizer.json"))?;
        // gpt-oss is the caching workhorse (dense bf16, ~40 GB transient) — built then dropped.
        let encoder = GptOssTextEncoder::new(
            &EncoderConfig::gpt_oss_20b(),
            flow_match::component_vb(&self.root, "text_encoder", device, ENC_DTYPE, LABEL)?,
        )?;
        let vae = Flux2Vae::new_with_encoder(flow_match::component_vb(
            &self.root,
            "vae",
            device,
            DType::F32,
            LABEL,
        )?)?;

        let total = req.items.len() as u32;
        let mut cache: Vec<(Tensor, Vec<Tensor>)> = Vec::with_capacity(req.items.len());
        let mut grid: Option<(usize, usize)> = None;
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = load_image_tensor(&item.image_path, edge, device)?; // [1,3,edge,edge] in [-1,1]
            let (x0, lh, lw) = vae_encode(&vae, &img)?; // [1, S, 128] packed latent (mean), f32
            let feats = encode_caption(&tokenizer, &encoder, &item.caption, device)?;
            grid.get_or_insert((lh, lw));
            cache.push((x0, feats));
        }
        drop(encoder);
        drop(vae);
        // The grid is set on the first cached item; `(0, 0)` is a placeholder for an empty cache (the
        // driver maps that to `Canceled`/error before any step reads the aux).
        Ok((cache, grid.unwrap_or((0, 0))))
    }

    fn build_dit(&self, req: &TrainingRequest, device: &Device) -> Result<LensTransformerTrain> {
        let compute_dtype = flow_match::parse_compute_dtype(&req.config.train_dtype);
        Ok(LensTransformerTrain::new(
            &LensDitConfig::lens(),
            flow_match::component_vb(&self.root, "transformer", device, compute_dtype, LABEL)?,
        )?)
    }

    fn micro_step(
        &self,
        dit: &LensTransformerTrain,
        vars: &[Var],
        cached: &(Tensor, Vec<Tensor>),
        aux: &(usize, usize),
        cfg: &TrainingConfig,
        step: u32,
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        let (x0, feats) = cached;
        let (lat_h, lat_w) = *aux;
        // Lens feeds `t` to the DiT directly (cast to f64), and the 48-block backward always uses the
        // gradient-checkpointed path.
        let t = flow_match::sample_unit_timestep(
            &cfg.timestep_type,
            &cfg.timestep_bias,
            flow_match::timestep_seed(cfg.seed, step),
        ) as f64;
        let noise =
            flow_match::sample_noise(x0.dims(), flow_match::noise_seed(cfg.seed, step), device)?;
        compute_loss_grads(
            dit,
            vars,
            x0,
            feats,
            lat_h,
            lat_w,
            t,
            &noise,
            flow_match::is_mae(cfg),
            flow_match::parse_compute_dtype(&cfg.train_dtype),
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::gen_core::registry;
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::optim::{clip_grad_norm, TrainOptimizer};

    /// A tiny Lens-shaped DiT config (2 layers, 2 heads × 8, 1 text layer) — exercises the real
    /// flow-match forward+backward on CPU. Mirrors `dit_train`'s tiny cfg (Σ axes = head_dim).
    fn tiny_cfg() -> LensDitConfig {
        LensDitConfig {
            patch_size: 2,
            in_channels: 32,
            out_channels: 8,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            inner_dim: 16,
            enc_hidden_dim: 12,
            num_text_layers: 1,
            timestep_channels: 16,
            axes_dims_rope: [2, 2, 4],
            rope_theta: 10_000.0,
        }
    }

    /// Randomize every var in a fresh `VarMap` — a zero patch/img_in weight makes `hidden ≡ 0` and the
    /// adapter grads vacuously zero; real training loads nonzero weights, so the tiny tests must too.
    fn randomize_base(vm: &VarMap, dev: &Device) {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), dev).unwrap())
                .unwrap();
        }
    }

    /// Tiny synthetic inputs: a packed latent `[1, h·w, in_channels]`, one text-feature layer, noise,
    /// and the latent grid `(h, w)`.
    fn tiny_inputs(
        cfg: &LensDitConfig,
        dev: &Device,
    ) -> (Tensor, Vec<Tensor>, Tensor, usize, usize) {
        let (h, w) = (2usize, 2usize);
        let img_len = h * w;
        let x0 = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), dev).unwrap();
        let feat = Tensor::randn(0f32, 1f32, (1, 3, cfg.enc_hidden_dim), dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), dev).unwrap();
        (x0, vec![feat], noise, h, w)
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var` (save the last block's
    /// `to_add_out`, whose text-stream output the image-velocity head discards — see `dit_train`).
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off its zero-init so both A and B grads are nonzero.
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let (loss, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &feats,
            h,
            w,
            0.5,
            &noise,
            false,
            DType::F32,
            false,
        )
        .unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        let mut saw_nonzero = false;
        for v in &set.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let gv = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert!(gv.iter().all(|x| x.is_finite()), "non-finite gradient");
                if gv.iter().any(|x| x.abs() > 1e-9) {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "backprop is not reaching the adapter factors");
        assert_eq!(set.vars.len(), 4 * 2 * cfg.num_layers); // 4 projections × 2 factors × layers
    }

    /// The correctness gate for the gradient-checkpointed backward (the path real training always uses):
    /// it must reproduce the dense `loss.backward()` grads (mod float reassociation) on the tiny DiT.
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let (loss_d, g_d) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &feats,
            h,
            w,
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
            &feats,
            h,
            w,
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
        let mut saw_nonzero = false;
        for (i, v) in set.vars.iter().enumerate() {
            // A var with no dense grad (the discarded last-block to_add_out) is skipped in both paths.
            let (Some(a), Some(b)) = (g_d.get(v.as_tensor()), g_c.get(v.as_tensor())) else {
                continue;
            };
            let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
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
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let mut opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        let loss_at = |dit: &LensTransformerTrain| {
            compute_loss_grads(
                dit,
                &set.vars,
                &x0,
                &feats,
                h,
                w,
                0.5,
                &noise,
                false,
                DType::F32,
                false,
            )
            .unwrap()
        };
        let (loss0, mut grads) = loss_at(&dit);
        for _ in 0..5 {
            clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            grads = loss_at(&dit).1;
        }
        let (loss1, _) = loss_at(&dit);
        assert!(
            loss1 < loss0,
            "5 steps on a fixed batch should lower the loss: {loss0} -> {loss1}"
        );
    }

    /// The trainer self-registers and resolves through gen-core's trainer registry as the candle Lens
    /// trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(MODEL_ID_BASE, &spec)
            .expect("candle lens trainer is registered");
        assert_eq!(t.descriptor().id, MODEL_ID_BASE);
        assert_eq!(t.descriptor().backend, "candle");
        assert_eq!(t.descriptor().modality, Modality::Image);
        assert!(t.descriptor().supports_lora && t.descriptor().supports_lokr);
    }

    /// `validate` rejects an empty dataset, zero rank/steps, an unsupported optimizer, and unrecognized
    /// timestep/loss knobs — before any load (now via the shared `flow_match::validate_flow_match_request`).
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        use candle_gen::gen_core::train::TrainingItem;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_trainer(MODEL_ID_BASE, &spec).unwrap();
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
}
