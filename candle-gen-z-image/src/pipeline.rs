//! The candle Z-Image **txt2img** pipeline (sc-3693) — the `candle-transformers` `z_image`
//! reference model (Qwen3 text encoder → DiT transformer → AutoencoderKL VAE, flow-match Euler)
//! driven through the backend-neutral [`gen_core::Generator`] contract, parity-matched to the
//! macOS `mlx-gen-z-image` provider.
//!
//! What this wires, and the deliberate parity choices (all grounded in the mlx provider's
//! `model.rs`/`pipeline.rs` and Z-Image's `scheduler_config.json`):
//!
//! - **Components**: the three `candle-transformers::models::z_image` modules — `ZImageTextEncoder`
//!   (Qwen3, hidden 2560, 36 layers; returns the second-to-last hidden state, no final norm),
//!   `ZImageTransformer2DModel` (the DiT, 16-channel latent, patch 2), and `AutoEncoderKL`
//!   (diffusers VAE, /8 spatial, scaling 0.3611 / shift 0.1159 applied **inside** `decode`). Loaded
//!   at **bf16** — Z-Image is a bf16 model (unlike the fp16 SDXL family), and candle's CUDA backend
//!   runs bf16 natively.
//! - **Prompt → cap_feats**: the Qwen chat-template wrapping + host-vec tokenization come from
//!   gen-core's [`TextTokenizer`] with [`ChatTemplate::QwenInstruct`] — the *exact* template the mlx
//!   provider uses ([`crate`] docs). This is the epic-3692 "carries over via gen-core" reuse: the
//!   parity-critical template is written once in gen-core, not re-derived here. The encoder output is
//!   padded to the DiT's `SEQ_MULTI_OF` with an attention mask by the reference `prepare_inputs`.
//! - **Distilled schedule (no CFG)**: Z-Image-Turbo is guidance-distilled — a fixed **4-step**
//!   flow-match Euler schedule, no classifier-free guidance and no negative prompt. The DiT is fed
//!   the **1−σ** timestep convention and its predicted velocity is **negated** before the Euler step
//!   (Z-Image sign convention). The scheduler is driven exactly as candle's own `z_image` example —
//!   `set_timesteps(steps, Some(mu))` — which under the `z_image_turbo` config keeps the σ schedule
//!   consistent with the DiT timestep (the `None`/static-shift path desyncs them and speckles; see
//!   [`Pipeline::render`]).
//! - **Deterministic seeding (sc-3673 parity)**: initial latent noise is drawn from a
//!   fixed-algorithm CPU RNG (`StdRng`, ChaCha) seeded by `seed` and moved to the device — NOT
//!   candle's CUDA `Tensor::randn`, whose seed→noise mapping is not launch-portable. The flow-match
//!   Euler step is non-stochastic, so the whole generation is a pure function of `(seed, request)` —
//!   which is what the gen-core-testkit seed-determinism check (sc-4481) requires.
//! - **CLI/PNG/sidecar removed**: progress is `on_progress(Progress::Step/Decoding)`, cancellation is
//!   `req.cancel` → typed [`gen_core::Error::Canceled`], and each image is returned as a
//!   `gen_core::Image` (RGB8) — the worker owns asset writes.
//!
//! **First-slice surface (sc-3693), matching the SDXL slice (sc-3675):** txt2img only. img2img
//! (the mlx provider's `Reference` conditioning), LoRA/LoKr, and whole-model Q4/Q8 quantization are
//! NOT wired here — they are rejected loudly (the worker routes them to the Python fallback) rather
//! than silently dropped. Component caching across calls is a follow-up (the mlx provider holds all
//! components resident too); peak-VRAM staging is the Z-Image analogue of SDXL's sc-4987 and is left
//! to a later slice.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{self, AdapterSpec, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::z_image::preprocess::prepare_inputs;
use candle_transformers::models::z_image::sampling::postprocess_image;
use candle_transformers::models::z_image::scheduler::{
    calculate_shift, FlowMatchEulerDiscreteScheduler, SchedulerConfig, BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT,
};
use candle_transformers::models::z_image::text_encoder::{TextEncoderConfig, ZImageTextEncoder};
use candle_transformers::models::z_image::transformer::{
    Config as DitConfig, ZImageTransformer2DModel,
};
use candle_transformers::models::z_image::vae::{AutoEncoderKL, VaeConfig};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// Z-Image-Turbo is guidance-distilled to a fixed 4-step schedule; used when a request omits
/// `steps`. Matches `mlx-gen-z-image`'s `DEFAULT_STEPS`.
pub(crate) const DEFAULT_STEPS: usize = 4;

/// VAE spatial downscale — the latent is image/8 per side (the 4-stage `block_out_channels`
/// `[128,256,512,512]` AutoencoderKL has 3 downsamplers). Matches `mlx-gen-z-image`'s `SPATIAL_SCALE`.
/// `pub(crate)` so the trainer's preview-sample path (sc-8650) shapes its seeded noise at the identical
/// /8 latent geometry inference uses (single source of truth).
pub(crate) const SPATIAL_SCALE: u32 = 8;

/// DiT patch size on each spatial axis (`Config::z_image_turbo().all_patch_size[0]`). The flow-match
/// `mu` shift is computed from the post-patchify image sequence length, so it is needed here.
/// `pub(crate)` so the trainer's preview `mu` (sc-8650) is derived identically to inference.
pub(crate) const PATCH_SIZE: u32 = 2;

/// Z-Image latent channel count (the VAE's `latent_channels` and the DiT's `in_channels`).
/// `pub(crate)` so the trainer's preview noise (sc-8650) is the identical 16-channel prior.
pub(crate) const LATENT_CHANNELS: usize = 16;

/// Qwen3 pad token id (`<|endoftext|>`). Only consulted when padding to a fixed length, which the
/// txt2img path does not do (`pad_to_max_length: false`); the DiT's `prepare_inputs` does the
/// SEQ_MULTI_OF padding + mask. Carried for correctness/parity with the mlx loader. `pub(crate)` so
/// the trainer's caption caching uses the exact same tokenizer config (single source of truth).
pub(crate) const QWEN_PAD_TOKEN_ID: i32 = 151643;

/// Right-truncation cap for prompt tokenization (HF single-sequence truncation). Z-Image prompts are
/// short; 512 is generous and never engages in practice.
pub(crate) const TOKENIZER_MAX_LEN: usize = 512;

/// The per-image seed within a batch: image `index` of a `count`-image request renders at
/// `base_seed + index` (wrapping). Mirrors `mlx-gen-z-image`'s `seed + i` convention, so the *n*-th
/// image of a batch reproduces in isolation as a single `count: 1` render at that derived seed. A
/// pure function so the law is unit-testable without a GPU.
pub(crate) fn image_seed(base_seed: u64, index: u32) -> u64 {
    base_seed.wrapping_add(index as u64)
}

/// A txt2img pipeline handle: the snapshot `root` + the compute device/dtype (bf16) + any LoRA/LoKr
/// adapters to merge into the DiT at component-load time (sc-5166). Loading the heavy components is
/// done by [`load_components`](Self::load_components) and owned/cached by the generator, mirroring
/// the SDXL provider's lazy split.
pub(crate) struct Pipeline {
    root: PathBuf,
    device: Device,
    dtype: DType,
    /// Adapters merged into the DiT weights at load. Empty ⇒ the stock mmap build (zero regression).
    adapters: Vec<AdapterSpec>,
}

/// The loaded Z-Image components, `Arc`-shared so the generator can cache them across `generate`
/// calls and cheaply clone them out for a render. All three are resolution-agnostic (the DiT/VAE
/// read fixed configs; latent dims come from the request), so one set serves every request size.
#[derive(Clone)]
pub(crate) struct Components {
    text_encoder: Arc<ZImageTextEncoder>,
    transformer: Arc<ZImageTransformer2DModel>,
    vae: Arc<AutoEncoderKL>,
}

impl Pipeline {
    /// Build the (light) pipeline handle for the Z-Image snapshot `root` at the given device/dtype,
    /// with `adapters` to merge into the DiT. Does **no** weight I/O — components load lazily via
    /// [`load_components`](Self::load_components).
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        dtype: DType,
        adapters: &[AdapterSpec],
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
            adapters: adapters.to_vec(),
        }
    }

    /// Load the three heavy components from the snapshot's diffusers component subdirs
    /// (`text_encoder/`, `transformer/`, `vae/`). `use_accelerated_attn` enables the DiT's fused
    /// attention dispatch (CUDA flash-attn / Metal SDPA); on a build without those features the
    /// reference falls back to the backend-agnostic manual path, so this is inert there.
    pub(crate) fn load_components(&self, use_accelerated_attn: bool) -> Result<Components> {
        let te_vb = self.component_vb("text_encoder")?;
        let text_encoder = ZImageTextEncoder::new(&TextEncoderConfig::z_image(), te_vb)?;

        let mut dit_cfg = DitConfig::z_image_turbo();
        dit_cfg.set_use_accelerated_attn(use_accelerated_attn);
        let dit_vb = if self.adapters.is_empty() {
            // No adapters: the stock mmap build — byte-identical to the pre-sc-5166 path.
            self.component_vb("transformer")?
        } else {
            self.transformer_vb_with_adapters()?
        };
        let transformer = ZImageTransformer2DModel::new(&dit_cfg, dit_vb)?;

        let vae_vb = self.component_vb("vae")?;
        let vae = AutoEncoderKL::new(&VaeConfig::z_image(), vae_vb)?;

        Ok(Components {
            text_encoder: Arc::new(text_encoder),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
        })
    }

    /// Resolve the sorted list of `.safetensors` files in the snapshot component subdir `sub`
    /// (single-file or sharded — diffusers ships both layouts), erroring if the dir or files are
    /// missing.
    fn component_files(&self, sub: &str) -> Result<Vec<PathBuf>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "z-image snapshot is missing the {sub}/ component directory (expected a diffusers \
                 multi-component snapshot at {})",
                self.root.display()
            )));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("z-image: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "z-image: no .safetensors found in {sub}/ (at {})",
                dir.display()
            )));
        }
        Ok(files)
    }

    /// Build a [`VarBuilder`] over every `.safetensors` in the snapshot component subdir `sub`, at
    /// this pipeline's dtype/device (the stock mmap path; no adapters).
    fn component_vb(&self, sub: &str) -> Result<VarBuilder<'static>> {
        let files = self.component_files(sub)?;
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, self.dtype, &self.device)? };
        Ok(vb)
    }

    /// Build the DiT [`VarBuilder`] with the LoRA/LoKr [`AdapterSpec`]s merged into its weights
    /// (sc-5166). The base `transformer/` tensors are loaded into a CPU map, each adapter's delta is
    /// folded in ([`crate::adapters::merge_adapters`], f32 math), then the stock candle DiT is built
    /// from the merged map — **merge, not residual** (Z-Image's flow-match sampler is chaos-sensitive;
    /// `(W+δ)·x` ≠ `W·x + δ·x` to ~1 ULP). Only reached when adapters are present.
    fn transformer_vb_with_adapters(&self) -> Result<VarBuilder<'static>> {
        let files = self.component_files("transformer")?;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for f in &files {
            let part = candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?;
            tensors.extend(part);
        }
        crate::adapters::merge_adapters(&mut tensors, &self.adapters)?;
        Ok(VarBuilder::from_tensors(tensors, self.dtype, &self.device))
    }

    /// Prompt → `cap_feats` `(seq, 2560)` at the compute dtype. Tokenizes with the Qwen chat
    /// template (gen-core's [`TextTokenizer`]), runs the Qwen3 encoder, and squeezes the batch axis.
    /// The reference `prepare_inputs` does the SEQ_MULTI_OF padding + attention mask downstream, so
    /// every returned token is valid (no padding here).
    pub(crate) fn text_embeddings(&self, te: &ZImageTextEncoder, prompt: &str) -> Result<Tensor> {
        let tok = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: TOKENIZER_MAX_LEN,
                pad_token_id: QWEN_PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenInstruct,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("z-image: load tokenizer: {e}")))?;
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("z-image: tokenize: {e}")))?;
        if out.ids.is_empty() {
            // Defense-in-depth: `validate` already rejects an empty prompt; guard before the
            // (1, 0) tensor would reach the encoder.
            return Err(CandleError::Msg("z-image: empty prompt".into()));
        }
        // candle embeddings index with u32; the chat-template ids are small non-negative Qwen ids.
        let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let enc = te.forward(&input_ids)?; // (1, L, 2560)
        Ok(enc.squeeze(0)?.to_dtype(self.dtype)?) // (L, 2560)
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. Returns one `gen_core::Image` per `req.count` (each with seed `base_seed + index`).
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let lat_h = (req.height / SPATIAL_SCALE) as usize;
        let lat_w = (req.width / SPATIAL_SCALE) as usize;

        // Text embeddings are seed- and image-independent: encode once for the whole batch.
        let cap = self.text_embeddings(&components.text_encoder, &req.prompt)?;

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = image_seed(base_seed, index);

            // sc-3673 parity — deterministic, launch-portable initial noise: N(0,1) from a
            // fixed-algorithm CPU RNG seeded by `seed`, built on CPU then moved to the device. The
            // flow-match Euler step injects no per-step noise, so generation is a pure function of
            // `(seed, request)`.
            let n = LATENT_CHANNELS * lat_h * lat_w;
            let mut rng = StdRng::seed_from_u64(seed);
            let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
            let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
                .to_device(&self.device)?
                .to_dtype(self.dtype)?;

            // Flow-match Euler schedule. Match the candle `z_image` reference: pass `Some(mu)` (the
            // resolution-dependent shift parameter from `calculate_shift`). Under
            // `use_dynamic_shifting=false` (the `z_image_turbo` config) the `Some(mu)` arm applies NO
            // sigma shift, so the sigmas stay LINEAR and consistent with `current_timestep_normalized`
            // (which is derived from the un-shifted `timesteps`). This is correctness-critical, NOT a
            // style knob: passing `None` takes the scheduler's static-shift branch, which shifts
            // `sigmas` WITHOUT updating `timesteps` — desyncing the t fed to the DiT from the σ used in
            // the Euler step, which leaves residual high-frequency noise (visible speckle) in the
            // decode. The unit-normal noise is the flow-match txt2img prior as-is (max σ = 1.0).
            let image_seq_len =
                ((lat_h as u32 / PATCH_SIZE) * (lat_w as u32 / PATCH_SIZE)) as usize;
            let mu = calculate_shift(
                image_seq_len,
                BASE_IMAGE_SEQ_LEN,
                MAX_IMAGE_SEQ_LEN,
                BASE_SHIFT,
                MAX_SHIFT,
            );
            let mut scheduler =
                FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
            scheduler.set_timesteps(steps, Some(mu));

            // Unified curated sampler/scheduler routing (epic 7114 P4, sc-7123). The NATIVE schedule is
            // the scheduler's σ table verbatim (linear / un-shifted for the turbo config — see the
            // comment above), so `resolve_flow_schedule(None, …)` returns it byte-for-byte and the
            // default `euler` is the N1 no-op = the legacy `scheduler.step` loop
            // `x + v·(σ_{i+1} − σ_i)`. The schedule is unshifted (`mu = 0.0` for the curated axis).
            // Z-Image feeds the DiT the 1−σ conditioning (`OneMinusSigma`) and the predicted velocity
            // is NEGATED before the step — both Z-Image-specific quirks live inside the `predict`
            // closure, so a multi-eval solver re-applies them each eval.
            let native: Vec<f32> = scheduler.sigmas.iter().map(|&s| s as f32).collect();
            let sigmas =
                candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), 0.0, steps, &native);

            // `prepare_inputs` pads cap_feats to SEQ_MULTI_OF (+ attention mask) and adds the
            // singleton frame axis to the latents → (1, 16, 1, lat_h, lat_w).
            let prepared = prepare_inputs(&noise, std::slice::from_ref(&cap), &self.device)?;
            let cap_feats = prepared.cap_feats;
            let cap_mask = prepared.cap_mask;

            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::OneMinusSigma,
                &sigmas,
                prepared.latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, t| -> Result<Tensor> {
                    // `t` is the 1−σ conditioning (OneMinusSigma) the DiT embeds — the same value the
                    // reference scheduler's `current_timestep_normalized` returns. The embedder upcasts
                    // to f32 internally, so f32 here is correct regardless of the model dtype.
                    let t_tensor = Tensor::from_vec(vec![t], (1,), &self.device)?;
                    let velocity = components
                        .transformer
                        .forward(latents, &t_tensor, &cap_feats, &cap_mask)?
                        .neg()?;
                    Ok(velocity)
                },
            )?;

            on_progress(Progress::Decoding);
            images.push(self.decode(&components.vae, &latents)?);
        }
        Ok(images)
    }

    /// VAE-decode the final latents `(1, 16, 1, h, w)` to an RGB8 [`Image`]. The VAE applies its own
    /// `/scaling_factor + shift_factor` un-scale inside `decode`; `postprocess_image` maps the
    /// `[-1, 1]` output to `[0, 255]` u8.
    fn decode(&self, vae: &AutoEncoderKL, latents: &Tensor) -> Result<Image> {
        // Drop the singleton frame axis: (1, 16, 1, h, w) -> (1, 16, h, w).
        let latents = latents.squeeze(2)?;
        let decoded = vae.decode(&latents)?.to_dtype(DType::F32)?; // (1, 3, H, W) in [-1, 1]
        let img = postprocess_image(&decoded)? // u8 (1, 3, H, W)
            .i(0)?
            .to_device(&Device::Cpu)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity anchors against `mlx-gen-z-image`: the distilled 4-step default and the /8 16-channel
    /// latent geometry. GPU-free (asserts constants directly).
    #[test]
    fn parity_defaults_match_mlx_provider() {
        assert_eq!(DEFAULT_STEPS, 4);
        assert_eq!(SPATIAL_SCALE, 8);
        assert_eq!(LATENT_CHANNELS, 16);
        assert_eq!(PATCH_SIZE, 2);
    }

    /// Per-image seed in a `count`-batch is `base + index` (wrapping), so image *n* reproduces in
    /// isolation at that derived seed — the mlx `seed + i` convention. Pure function, no GPU.
    #[test]
    fn image_seed_is_base_plus_index() {
        assert_eq!(image_seed(42, 0), 42);
        assert_eq!(image_seed(42, 1), 43);
        assert_eq!(image_seed(42, 7), 49);
        assert_eq!(image_seed(u64::MAX, 1), 0);
    }

    /// The flow-match Euler schedule the pipeline drives (`set_timesteps(steps, Some(mu))`) must, for
    /// the distilled 4-step config: have `num_steps + 1` sigmas, start at max-σ **1.0**, be strictly
    /// decreasing, and terminate at 0.
    ///
    /// **Regression guard for the speckle bug:** at every step the timestep fed to the DiT
    /// (`(1000 − timesteps[i]) / 1000`, i.e. `current_timestep_normalized`) must equal `1 − σᵢ` (the σ
    /// the Euler step actually uses). The `Some(mu)` call keeps `timesteps` and `sigmas` consistent;
    /// the `None` call would shift `sigmas` without updating `timesteps`, breaking this identity and
    /// leaving residual high-frequency noise in the decode. GPU-free.
    #[test]
    fn flow_match_schedule_keeps_timestep_and_sigma_consistent() {
        // mu for a representative 1024² render: latent 128² → seq (128/2)² = 4096.
        let mu = calculate_shift(
            4096,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        let mut s = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        s.set_timesteps(DEFAULT_STEPS, Some(mu));
        assert_eq!(s.sigmas.len(), DEFAULT_STEPS + 1);
        assert!(
            (s.sigmas[0] - 1.0).abs() < 1e-6,
            "max sigma: {}",
            s.sigmas[0]
        );
        assert!(
            (s.sigmas[DEFAULT_STEPS]).abs() < 1e-6,
            "terminal sigma must be 0"
        );
        for w in s.sigmas.windows(2) {
            assert!(w[0] > w[1], "sigmas must strictly decrease: {:?}", s.sigmas);
        }
        // The correctness-critical identity: t fed to the DiT == 1 − σ at every step.
        for i in 0..DEFAULT_STEPS {
            let t = (1000.0 - s.timesteps[i]) / 1000.0;
            assert!(
                (t - (1.0 - s.sigmas[i])).abs() < 1e-9,
                "t/σ desync at step {i}: t={t}, 1-σ={}",
                1.0 - s.sigmas[i]
            );
        }
    }
}
