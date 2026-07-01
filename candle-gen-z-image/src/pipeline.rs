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
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{self, AdapterSpec, Conditioning, GenerationRequest, Image, Progress};
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
use candle_transformers::models::z_image::vae::{AutoEncoderKL, Encoder as VaeEncoder, VaeConfig};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// Z-Image-Turbo is guidance-distilled to a fixed 4-step schedule; used when a request omits
/// `steps`. Matches `mlx-gen-z-image`'s `DEFAULT_STEPS`.
pub(crate) const DEFAULT_STEPS: usize = 4;

/// Base (non-Turbo) Z-Image default steps — undistilled foundation model. The card recommends 28–50;
/// 50 matches the reference `ZImagePipeline` example (`num_inference_steps=50`) and the mlx base
/// provider (`mlx-gen-z-image::model_base::DEFAULT_STEPS`, sc-8320). Used when a base request omits
/// `steps`.
pub(crate) const BASE_DEFAULT_STEPS: usize = 50;

/// Flow-match time-shift for the **base** Z-Image: `scheduler/scheduler_config.json`
/// (`FlowMatchEulerDiscreteScheduler`, `shift=6.0`, `use_dynamic_shifting=false`) — static,
/// resolution-independent. **This is the sole scheduler delta vs Turbo (3.0).** Mirrors
/// `mlx-gen-z-image::model_base::SCHEDULE_SHIFT`.
pub(crate) const BASE_SCHEDULE_SHIFT: f64 = 6.0;

/// Default CFG scale for the base — the card recommends 3.0–5.0; 4.0 matches the reference
/// `ZImagePipeline` example (`guidance_scale=4`) and the mlx base provider. Used when a base request
/// omits `guidance`.
pub(crate) const BASE_DEFAULT_GUIDANCE: f32 = 4.0;

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

/// The VAE **encoder** runs f32 (the encode path's dtype, matching [`crate::edit`]); its distribution
/// mean is cast to the compute dtype (bf16) for the img2img init latent. Only the base img2img /
/// `Reference` path (sc-8646) builds/uses an encoder — txt2img never touches it.
const ENC_DTYPE: DType = DType::F32;

/// img2img start step — the Z-Image "structure-preservation" convention (the fork's `init_time_step`,
/// mirrored from `mlx-gen`'s shared `img2img::init_time_step`): for a reference with `strength` in
/// `(0, 1]`, `max(1, floor(num_steps · strength))`; otherwise `0` (pure txt2img, no reference blend).
/// **Higher strength → later start → fewer denoise steps → output stays CLOSER to the reference** — the
/// inverse of the SDXL knob, matched here so the strength knob behaves identically on the Mac (MLX) and
/// Windows (candle) base lanes. `floor` because Python `int(steps · strength)` truncates toward zero for
/// `s ≥ 0`. Pure function so the cross-backend-parity law is unit-testable without a GPU.
pub(crate) fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => {
            let s = s.clamp(0.0, 1.0);
            ((num_steps as f32 * s) as usize).max(1)
        }
        _ => 0,
    }
}

/// Resolve the single img2img init image + its effective strength from the request's conditioning
/// (sc-8646), mirroring `mlx-gen-z-image::pipeline::resolve_reference`. A per-reference `strength`
/// overrides `req.strength`. The base Z-Image conditions on exactly one init image, so more than one
/// [`Conditioning::Reference`] is an error (multi-image would be `MultiReference`, unadvertised here);
/// non-`Reference` conditioning kinds are already rejected by the capability floor in `validate`.
pub(crate) fn resolve_reference(req: &GenerationRequest) -> Result<Option<(&Image, Option<f32>)>> {
    let mut reference = None;
    for c in &req.conditioning {
        if let Conditioning::Reference { image, strength } = c {
            if reference.is_some() {
                return Err(CandleError::Msg(
                    "z_image: multiple reference images are not supported (single img2img init only)"
                        .into(),
                ));
            }
            reference = Some((image, strength.or(req.strength)));
        }
    }
    Ok(reference)
}

/// The **base** (non-Turbo) Z-Image flow-match scheduler config: `shift = 6.0`,
/// `use_dynamic_shifting = false` (the base model's `scheduler/scheduler_config.json`). Distinct from
/// `SchedulerConfig::z_image_turbo()` (shift 3.0) — the sole scheduler delta the base introduces
/// (sc-8414). Built explicitly because candle-transformers only ships a `z_image_turbo()` constructor.
pub(crate) fn base_scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        num_train_timesteps: 1000,
        shift: BASE_SCHEDULE_SHIFT,
        use_dynamic_shifting: false,
    }
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

    /// Build the standalone f32 VAE **encoder** for the base img2img / `Reference` path (sc-8646). The
    /// decode `AutoEncoderKL` holds an encoder too, but (a) it is private and (b) its `encode` samples
    /// the diagonal-gaussian via the *device* RNG (not launch-portable — breaks sc-3673), so — exactly
    /// like [`crate::edit`] and [`crate::control`] — the raw `Encoder` is run here to take the
    /// distribution **mean** deterministically. Only built on the first img2img request (cached by the
    /// generator), so the txt2img / Turbo path never pays for it.
    pub(crate) fn load_vae_encoder(&self) -> Result<VaeEncoder> {
        let files = self.component_files("vae")?;
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, ENC_DTYPE, &self.device)? };
        Ok(VaeEncoder::new(&VaeConfig::z_image(), vb.pp("encoder"))?)
    }

    /// VAE-encode `source` (LANCZOS-resized to the render size, normalized to `[-1, 1]` NCHW) to the
    /// deterministic clean init latent `(1, 16, H/8, W/8)` at the compute dtype (bf16): the distribution
    /// **mean** (not a sampled draw), mapped to latent space as `(mean − shift) · scale` — the same
    /// deterministic encode [`crate::edit::ZImageEdit::encode_source`] uses. `encoder` is the f32 encoder
    /// from [`load_vae_encoder`](Self::load_vae_encoder).
    pub(crate) fn encode_reference(
        &self,
        encoder: &VaeEncoder,
        source: &Image,
        width: u32,
        height: u32,
    ) -> Result<Tensor> {
        let vae_cfg = VaeConfig::z_image();
        let img = preprocess_source(source, width, height, &self.device)?; // f32 (1,3,H,W) [-1,1]
        let moments = img.apply(encoder)?; // (1, 32, H/8, W/8) — [mean | logvar]
        let mean = moments.chunk(2, 1)?[0].clone(); // (1, 16, H/8, W/8)
        let latents = ((mean - vae_cfg.shift_factor)? * vae_cfg.scaling_factor)?;
        Ok(latents.to_dtype(self.dtype)?)
    }

    /// Build the Z-Image Qwen tokenizer (chat template + max-length policy). Shared by the conditional
    /// ([`text_embeddings`](Self::text_embeddings)) and unconditional
    /// ([`uncond_embeddings`](Self::uncond_embeddings)) encode paths so their tokenization policy can
    /// never drift.
    fn tokenizer(&self) -> Result<TextTokenizer> {
        TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: TOKENIZER_MAX_LEN,
                pad_token_id: QWEN_PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenInstruct,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("z-image: load tokenizer: {e}")))
    }

    /// Token `ids` → `cap_feats` `(seq, 2560)` at the compute dtype: run the Qwen3 encoder and squeeze
    /// the batch axis. The reference `prepare_inputs` does the SEQ_MULTI_OF padding + attention mask
    /// downstream, so every id here is a valid token (no padding at this seam).
    fn encode_cap(&self, te: &ZImageTextEncoder, ids: &[i32]) -> Result<Tensor> {
        // candle embeddings index with u32; the chat-template ids are small non-negative Qwen ids.
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let enc = te.forward(&input_ids)?; // (1, L, 2560)
        Ok(enc.squeeze(0)?.to_dtype(self.dtype)?) // (L, 2560)
    }

    /// Prompt → `cap_feats` `(seq, 2560)`. Tokenizes with the Qwen chat template (gen-core's
    /// [`TextTokenizer`]) and runs the Qwen3 encoder.
    pub(crate) fn text_embeddings(&self, te: &ZImageTextEncoder, prompt: &str) -> Result<Tensor> {
        let out = self
            .tokenizer()?
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("z-image: tokenize: {e}")))?;
        if out.ids.is_empty() {
            // Defense-in-depth: `validate` already rejects an empty prompt; guard before the
            // (1, 0) tensor would reach the encoder.
            return Err(CandleError::Msg("z-image: empty prompt".into()));
        }
        self.encode_cap(te, &out.ids)
    }

    /// Negative prompt → `cap_feats` for the **unconditional** CFG branch of the base path (sc-8414).
    /// Identical encoding to [`text_embeddings`](Self::text_embeddings) — same Qwen chat template, same
    /// encoder — but the negative prompt may be the **empty string** (the unconditional embedding).
    ///
    /// The empty-string case must NOT route through `text_embeddings`: gen-core's
    /// [`TextTokenizer::tokenize`] short-circuits an empty prompt to a `(1, 0)` sequence **before** the
    /// chat template is applied (our config has `pad_to_max_length = false`), so an empty negative
    /// prompt would trip the empty-`ids` guard and error `z-image: empty prompt` (sc-8646, observed on
    /// real weights: base CFG with an unset negative prompt). Instead we render the QwenInstruct
    /// scaffolding around `""` via [`encode_chat_ids`] — `<|im_start|>user\n<|im_end|>\n<|im_start|>
    /// assistant\n` — which tokenizes to the non-empty role-marker sequence the reference
    /// `mlx-gen-z-image::model_base` feeds its uncond branch. A non-empty negative prompt takes the
    /// ordinary `text_embeddings` path.
    pub(crate) fn uncond_embeddings(
        &self,
        te: &ZImageTextEncoder,
        negative_prompt: &str,
    ) -> Result<Tensor> {
        if !negative_prompt.is_empty() {
            return self.text_embeddings(te, negative_prompt);
        }
        // `add_special_tokens = true` mirrors `tokenize`'s `encode(text, true)`. For Qwen this only
        // governs the auto-added BOS/EOS (Qwen adds none), so the ids equal the templated tokens.
        let ids = self
            .tokenizer()?
            .encode_chat_ids("", true)
            .map_err(|e| CandleError::Msg(format!("z-image: tokenize uncond: {e}")))?;
        if ids.is_empty() {
            // Only reachable if a degenerate template rendered "" to nothing — surface as a typed
            // error rather than letting a (1, 0) tensor reach the encoder.
            return Err(CandleError::Msg(
                "z-image: unconditional embedding tokenized to an empty sequence".into(),
            ));
        }
        self.encode_cap(te, &ids)
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

    /// Render `req` against pre-loaded `components` on the **base** (non-Turbo) path: real
    /// classifier-free guidance over the static **shift=6.0** flow-match schedule (sc-8414, the candle
    /// sibling of `mlx-gen-z-image::model_base`). Emits per-step progress and honors `req.cancel`.
    ///
    /// Differences from [`render`](Self::render) (the Turbo path), all from the base model card /
    /// `scheduler/scheduler_config.json`:
    ///
    /// - **Static shift = 6.0** (Turbo's effective inference schedule is linear/un-shifted because its
    ///   `set_timesteps(steps, Some(mu))` call no-ops under `use_dynamic_shifting=false`). The base
    ///   builds its σ table with `set_timesteps(steps, None)` against a `shift=6.0` config, so the
    ///   static-shift branch actually fires. We feed that σ table to [`run_flow_sampler`] with
    ///   [`TimestepConvention::OneMinusSigma`], which derives the DiT timestep `t = 1−σ` from the σ
    ///   schedule **itself** — so the Turbo `None`-path "timesteps desync" speckle bug is structurally
    ///   absent here (we never read the scheduler's `timesteps`/`current_timestep_normalized`).
    /// - **Real CFG**: each step runs the DiT twice (cond + uncond) and combines
    ///   `v = v_uncond + guidance·(v_cond − v_uncond)`. `guidance == 1.0` collapses to a single cond
    ///   forward (Turbo-equivalent cost). The uncond branch encodes the negative prompt (empty string
    ///   when unset — the unconditional embedding).
    /// - **Default 50 steps** when `req.steps` is unset ([`BASE_DEFAULT_STEPS`]).
    ///
    /// **img2img / `Reference` (sc-8646).** When `clean` is `Some` (the caller VAE-encoded a reference
    /// image via [`encode_reference`](Self::encode_reference)) and `start_step > 0`, each image blends
    /// the pre-encoded clean latent with the seeded noise at `σ_start` (the flow-match interpolation
    /// `x_t = (1 − σ)·clean + σ·noise`) and denoises the **reduced** `start_step..` tail of the σ
    /// schedule — real CFG applies to the img2img tail exactly as to txt2img. `start_step == 0` (`clean`
    /// is `None`) is pure txt2img: `x_t = noise`, full schedule — byte-identical to the pre-sc-8646 path.
    /// Mirrors `mlx-gen-z-image::model_base` + `pipeline::render_batch`.
    pub(crate) fn render_base(
        &self,
        req: &GenerationRequest,
        components: &Components,
        clean: Option<&Tensor>,
        start_step: usize,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req.steps.map(|s| s as usize).unwrap_or(BASE_DEFAULT_STEPS);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let lat_h = (req.height / SPATIAL_SCALE) as usize;
        let lat_w = (req.width / SPATIAL_SCALE) as usize;

        // Real CFG: `req.guidance` is the classifier-free guidance scale (default 4.0). A value of 1.0
        // turns CFG off (single cond forward, Turbo-equivalent cost).
        let guidance = req.guidance.unwrap_or(BASE_DEFAULT_GUIDANCE);
        let cfg_on = guidance != 1.0;

        // Text embeddings are seed- and image-independent: encode once for the whole batch. The uncond
        // branch (negative prompt, empty when unset) is only encoded when CFG is active.
        let cap = self.text_embeddings(&components.text_encoder, &req.prompt)?;
        let neg_cap = if cfg_on {
            let neg = req.negative_prompt.as_deref().unwrap_or("");
            Some(self.uncond_embeddings(&components.text_encoder, neg)?)
        } else {
            None
        };

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = image_seed(base_seed, index);

            // sc-3673 parity — deterministic, launch-portable initial noise (see `render`).
            let n = LATENT_CHANNELS * lat_h * lat_w;
            let mut rng = StdRng::seed_from_u64(seed);
            let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
            let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
                .to_device(&self.device)?
                .to_dtype(self.dtype)?;

            // Static shift=6.0 schedule (the base model's scheduler_config.json). Unlike the Turbo
            // path's `Some(mu)` no-op, the base passes `None` so the static-shift branch actually
            // shifts the σ table; `run_flow_sampler`'s `OneMinusSigma` derives the DiT timestep from
            // these σ directly, so there is no timesteps desync to guard against.
            let mut scheduler = FlowMatchEulerDiscreteScheduler::new(base_scheduler_config());
            scheduler.set_timesteps(steps, None);
            let native: Vec<f32> = scheduler.sigmas.iter().map(|&s| s as f32).collect();

            // Curated scheduler axis (epic 7114): an unset `req.scheduler` returns `native` verbatim
            // (the byte-exact shift=6.0 default); a curated name re-shapes σ over the same shift
            // (`mu = ln(shift)`), exactly as `mlx-gen-z-image::model_base`.
            let sigmas = candle_gen::resolve_flow_schedule(
                req.scheduler.as_deref(),
                (BASE_SCHEDULE_SHIFT as f32).ln(),
                steps,
                &native,
            );

            // img2img / `Reference` (sc-8646): blend the pre-encoded clean latent with the seeded noise
            // at `σ_start = sigmas[start]` and denoise the reduced `start..` schedule tail. `start` is
            // clamped to the schedule because a curated scheduler may return a length ≠ `steps + 1`.
            // For txt2img (`clean` is `None`, `start_step == 0`) this is `x_t = noise` over the full
            // schedule — byte-identical to the pre-sc-8646 path. Mirrors `render_batch`'s
            // `add_noise_by_interpolation` (`x_t = (1 − σ)·clean + σ·noise`).
            let start = start_step.min(sigmas.len().saturating_sub(1));
            let x_t = match clean {
                Some(clean) => {
                    let sigma_start = sigmas[start] as f64;
                    (clean.affine(1.0 - sigma_start, 0.0)? + noise.affine(sigma_start, 0.0)?)?
                }
                None => noise,
            };
            let run_sigmas = &sigmas[start..];

            // `prepare_inputs` pads cap_feats to SEQ_MULTI_OF (+ attention mask) for both the cond and
            // (when CFG is on) the uncond branch, and adds the singleton frame axis to the latents. The
            // uncond branch only uses cap_feats/cap_mask (its `latents` are discarded), so passing `x_t`
            // there is fine.
            let prepared = prepare_inputs(&x_t, std::slice::from_ref(&cap), &self.device)?;
            let cap_feats = prepared.cap_feats;
            let cap_mask = prepared.cap_mask;
            let uncond = match neg_cap.as_ref() {
                Some(neg) => {
                    let p = prepare_inputs(&x_t, std::slice::from_ref(neg), &self.device)?;
                    Some((p.cap_feats, p.cap_mask))
                }
                None => None,
            };

            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::OneMinusSigma,
                run_sigmas,
                prepared.latents,
                seed,
                &req.cancel,
                on_progress,
                |latents, t| -> Result<Tensor> {
                    let t_tensor = Tensor::from_vec(vec![t], (1,), &self.device)?;
                    // Conditional velocity (Z-Image sign convention: the DiT output is negated before
                    // the flow-match step). The CFG combine is done on the negated velocities, which is
                    // linear so the result is identical to combining-then-negating.
                    let v_cond = components
                        .transformer
                        .forward(latents, &t_tensor, &cap_feats, &cap_mask)?
                        .neg()?;
                    let velocity = match uncond.as_ref() {
                        Some((neg_feats, neg_mask)) => {
                            let v_uncond = components
                                .transformer
                                .forward(latents, &t_tensor, neg_feats, neg_mask)?
                                .neg()?;
                            // v = v_uncond + guidance·(v_cond − v_uncond)
                            let delta = (&v_cond - &v_uncond)?;
                            (v_uncond + (delta * guidance as f64)?)?
                        }
                        None => v_cond,
                    };
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

/// An RGB8 img2img reference → `[1, 3, H, W]` f32 in `[-1, 1]` (the VAE encoder's input range),
/// LANCZOS-resized to the render `width × height` (the worker pre-fits, but resizing here keeps the
/// base provider robust to an off-size reference — the same normalization [`crate::edit`] uses,
/// sc-8646). A no-op resize when already at the render size.
fn preprocess_source(image: &Image, width: u32, height: u32, device: &Device) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "z_image img2img: reference buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    let resized: Vec<f32> = if (ih, iw) == (rh, rw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, rh, rw) // HWC f32 [0,255]
    };
    // [0,255] → [-1,1], HWC → CHW.
    let mut data = vec![0f32; 3 * rh * rw];
    for y in 0..rh {
        for x in 0..rw {
            for c in 0..3 {
                data[c * rh * rw + y * rw + x] = resized[(y * rw + x) * 3 + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, rh, rw), device)?.to_dtype(ENC_DTYPE)?)
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

    /// Base (non-Turbo) parity constants vs Turbo + the mlx base provider (sc-8414 / mlx sc-8320):
    /// shift 6.0 (Turbo's config is 3.0), default 50 steps (Turbo 4), default CFG 4.0. These are the
    /// load-bearing port values from the base `scheduler_config.json` + the model card. GPU-free.
    #[test]
    fn base_constants_match_the_model_card() {
        assert_eq!(BASE_SCHEDULE_SHIFT, 6.0);
        assert_eq!(BASE_DEFAULT_STEPS, 50);
        assert_eq!(BASE_DEFAULT_GUIDANCE, 4.0);
        // The base scheduler config differs from Turbo only in the static shift.
        let base = base_scheduler_config();
        let turbo = SchedulerConfig::z_image_turbo();
        assert_eq!(base.shift, 6.0);
        assert_eq!(turbo.shift, 3.0);
        assert!(!base.use_dynamic_shifting && !turbo.use_dynamic_shifting);
        assert_eq!(base.num_train_timesteps, turbo.num_train_timesteps);
    }

    /// sc-8646 root-cause guard at the tokenizer seam (no GPU / no model weights — only the snapshot's
    /// `tokenizer/tokenizer.json`): base CFG with an **unset** negative prompt must be able to build an
    /// unconditional embedding. gen-core's [`TextTokenizer::tokenize`] short-circuits an empty prompt to
    /// a `(1, 0)` sequence **before** the chat template is applied (`pad_to_max_length = false`) — which
    /// is why routing the empty uncond through `text_embeddings` errored `z-image: empty prompt`. The
    /// fix ([`Pipeline::uncond_embeddings`]) encodes it via `encode_chat_ids("", true)`, which renders
    /// the QwenInstruct scaffolding around `""` and yields a **non-empty** role-marker token sequence.
    /// Set `Z_IMAGE_SNAPSHOT` or `Z_IMAGE_BASE_SNAPSHOT` (both ship the same Qwen tokenizer).
    #[test]
    #[ignore = "needs Z_IMAGE_SNAPSHOT/Z_IMAGE_BASE_SNAPSHOT for tokenizer.json (no GPU); run with --ignored"]
    fn empty_uncond_tokenizes_via_chat_template() {
        let snap = std::env::var("Z_IMAGE_BASE_SNAPSHOT")
            .or_else(|_| std::env::var("Z_IMAGE_SNAPSHOT"))
            .expect("set Z_IMAGE_SNAPSHOT or Z_IMAGE_BASE_SNAPSHOT to a Z-Image snapshot dir");
        let tok = TextTokenizer::from_file(
            Path::new(&snap).join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: TOKENIZER_MAX_LEN,
                pad_token_id: QWEN_PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenInstruct,
                pad_to_max_length: false,
            },
        )
        .expect("load tokenizer.json");

        // The trap: an empty prompt short-circuits to (1, 0) BEFORE the chat template is applied.
        assert!(
            tok.tokenize("").unwrap().ids.is_empty(),
            "empty prompt must short-circuit before the chat template (the sc-8646 trap)"
        );
        // The fix: the QwenInstruct scaffolding around "" tokenizes to a non-empty sequence, distinct
        // from a real prompt's encoding.
        let uncond_ids = tok.encode_chat_ids("", true).expect("encode empty uncond");
        assert!(
            !uncond_ids.is_empty(),
            "empty uncond must tokenize via the chat template to a non-empty sequence"
        );
        let real_ids = tok
            .encode_chat_ids("a red fox", true)
            .expect("encode real prompt");
        assert_ne!(uncond_ids, real_ids, "uncond scaffolding != a real prompt");
    }

    /// The base static **shift=6.0** schedule (built `set_timesteps(steps, None)`) must: have
    /// `num_steps + 1` sigmas, start at max-σ **1.0**, strictly decrease, terminate at 0 — and, the
    /// load-bearing delta vs Turbo, actually apply the shift so its σ table is NOT the linear ramp.
    /// The shift biases the schedule toward high-noise steps (σ at a given fraction is ≥ the linear
    /// value), which is what an undistilled CFG model needs. GPU-free.
    #[test]
    fn base_schedule_applies_shift_six() {
        let steps = 50usize;
        let mut s = FlowMatchEulerDiscreteScheduler::new(base_scheduler_config());
        s.set_timesteps(steps, None);
        assert_eq!(s.sigmas.len(), steps + 1);
        assert!(
            (s.sigmas[0] - 1.0).abs() < 1e-9,
            "max sigma: {}",
            s.sigmas[0]
        );
        assert!(s.sigmas[steps].abs() < 1e-9, "terminal sigma must be 0");
        for w in s.sigmas.windows(2) {
            assert!(w[0] > w[1], "sigmas must strictly decrease: {:?}", s.sigmas);
        }
        // Shift actually applied: shift*x/(1+(shift-1)*x) > x for x in (0,1), so the shifted σ table
        // is strictly above the linear ramp at every interior node (and differs from Turbo's table).
        let mut turbo = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        turbo.set_timesteps(steps, None);
        for i in 1..steps {
            let linear = 1.0 - (i as f64) / (steps as f64);
            assert!(
                s.sigmas[i] > linear + 1e-9,
                "shift=6.0 must lift σ[{i}]={} above the linear ramp {linear}",
                s.sigmas[i]
            );
            assert!(
                s.sigmas[i] > turbo.sigmas[i] + 1e-9,
                "shift=6.0 σ[{i}]={} must exceed Turbo shift=3.0 σ={}",
                s.sigmas[i],
                turbo.sigmas[i]
            );
        }
        // The DiT timestep the base render feeds (1 − σ, OneMinusSigma) is derived from THIS σ table,
        // so it is consistent by construction — no `timesteps` desync (the Turbo `None`-path speckle
        // bug cannot occur on the base path).
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

    /// The base img2img start-step law (sc-8646, the fork's `init_time_step` over `Option<f32>`):
    /// `max(1, floor(steps·strength))` for a strength in `(0, 1]`, else `0` (pure txt2img). Higher
    /// strength → later start → fewer denoise steps (Z-Image structure preservation). Pure, no GPU —
    /// the cross-backend-parity contract with `mlx-gen`'s shared `img2img::init_time_step`.
    #[test]
    fn init_time_step_is_the_fork_convention() {
        // None / non-positive strength ⇒ pure txt2img (start 0, reference ignored).
        assert_eq!(init_time_step(50, None), 0);
        assert_eq!(init_time_step(50, Some(0.0)), 0);
        assert_eq!(init_time_step(50, Some(-1.0)), 0);
        // floor(steps·strength), min 1.
        assert_eq!(init_time_step(50, Some(0.6)), 30); // floor(30.0)
        assert_eq!(init_time_step(50, Some(0.01)), 1); // floor(0.5)=0 → max(1,0)=1
        assert_eq!(init_time_step(4, Some(0.6)), 2); // floor(2.4)
        assert_eq!(init_time_step(50, Some(1.0)), 50); // == steps ⇒ empty loop, source round-trip
        assert_eq!(init_time_step(50, Some(2.0)), 50); // clamped above 1
                                                       // Monotone: higher strength ⇒ later (or equal) start.
        let starts: Vec<usize> = [0.1, 0.3, 0.5, 0.7, 0.9]
            .iter()
            .map(|&s| init_time_step(50, Some(s)))
            .collect();
        assert!(starts.windows(2).all(|w| w[0] <= w[1]), "{starts:?}");
    }

    /// `resolve_reference` pulls the single img2img init image + its effective strength from the
    /// request's conditioning (sc-8646): the per-reference strength wins over `req.strength`, a bare
    /// `Reference` falls back to `req.strength`, no `Reference` is `None`, and >1 `Reference` errors.
    /// Pure, no GPU.
    #[test]
    fn resolve_reference_picks_single_ref_and_strength() {
        use candle_gen::gen_core::{Conditioning, Image};
        let img = || Image {
            width: 8,
            height: 8,
            pixels: vec![0u8; 8 * 8 * 3],
        };

        // No conditioning ⇒ txt2img.
        let none = GenerationRequest::default();
        assert!(resolve_reference(&none).unwrap().is_none());

        // Per-reference strength wins over req.strength.
        let per_ref = GenerationRequest {
            strength: Some(0.2),
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: Some(0.75),
            }],
            ..Default::default()
        };
        assert_eq!(resolve_reference(&per_ref).unwrap().unwrap().1, Some(0.75));

        // A bare Reference falls back to req.strength.
        let fallback = GenerationRequest {
            strength: Some(0.3),
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: None,
            }],
            ..Default::default()
        };
        assert_eq!(resolve_reference(&fallback).unwrap().unwrap().1, Some(0.3));

        // More than one Reference is an error (single img2img init only).
        let two = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img(),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img(),
                    strength: None,
                },
            ],
            ..Default::default()
        };
        assert!(resolve_reference(&two).is_err());
    }

    /// The img2img blend + reduced-schedule indices the base render loop reads (sc-8646), asserted on
    /// the static shift=6.0 σ table. At start `k`: the loop runs the `sigmas[k..]` tail (so
    /// `steps − k + 1` σ nodes / `steps − k` steps), σ_start = sigmas[k] ∈ (0,1) for interior `k`, and
    /// the flow-match interpolation `x_t = (1−σ)·clean + σ·noise` seeds the loop. Max strength (k=steps)
    /// ⇒ σ_start = 0 ⇒ x_t = clean and a single-node (0-step) tail: the source VAE round-trip. GPU-free.
    #[test]
    fn img2img_reduced_schedule_indices() {
        let steps = 50usize;
        let mut s = FlowMatchEulerDiscreteScheduler::new(base_scheduler_config());
        s.set_timesteps(steps, None);
        let sigmas: Vec<f32> = s.sigmas.iter().map(|&x| x as f32).collect();
        assert_eq!(sigmas.len(), steps + 1);

        // Default strength 0.6 → start 30; the tail runs sigmas[30..] (21 nodes, 20 steps).
        let start = init_time_step(steps, Some(0.6));
        assert_eq!(start, 30);
        let tail = &sigmas[start..];
        assert_eq!(tail.len(), steps - start + 1);
        assert!(
            tail[0] > 0.0 && tail[0] < 1.0,
            "σ_start in (0,1): {}",
            tail[0]
        );
        assert!(tail[tail.len() - 1].abs() < 1e-6, "tail ends at 0");

        // Max strength → start == steps → single-node tail (0 steps), σ_start == 0 ⇒ x_t == clean.
        let full = init_time_step(steps, Some(1.0));
        assert_eq!(full, steps);
        assert_eq!(sigmas[full..].len(), 1);
        assert!(sigmas[full].abs() < 1e-6);
    }
}
