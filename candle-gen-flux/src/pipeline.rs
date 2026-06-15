//! The candle FLUX.1 **txt2img** pipeline (sc-3694) — the `candle-transformers` `flux` reference
//! model (dual CLIP-L + T5-XXL text encoders → FLUX DiT, flow-match Euler → FLUX AutoEncoder VAE)
//! driven through the backend-neutral [`gen_core::Generator`] contract, parity-matched to the macOS
//! `mlx-gen-flux` provider for both the `flux1_schnell` (distilled, 4-step, no guidance) and
//! `flux1_dev` (guidance-distilled, 25-step, guidance ~3.5) variants.
//!
//! What this wires, and the deliberate parity choices (grounded in the candle `flux` example and the
//! mlx provider's `config.rs`/`loader.rs`/`model.rs`):
//!
//! - **Weight layout — the clean split**: a black-forest-labs FLUX snapshot ships *both* the original
//!   single-file checkpoints at the root (`flux1-{schnell,dev}.safetensors`, `ae.safetensors`) *and*
//!   the diffusers component subdirs. candle's [`flux::model::Flux`] / [`flux::autoencoder::AutoEncoder`]
//!   are written against the **original BFL key layout**, so the DiT + VAE load directly from the root
//!   files (no diffusers→BFL key remap needed — the part mlx had to hand-write). The two text encoders
//!   come from the diffusers subdirs: CLIP-L from `text_encoder/` and T5-XXL from `text_encoder_2/`.
//! - **Dual text encoders**: candle's [`clip::text_model::ClipTextTransformer`] returns the **pooled**
//!   `(1, 768)` vector (argmax-at-EOT over a causal stack — FLUX's `vec`/`y` conditioning), and
//!   [`t5::T5EncoderModel`] returns the `(1, L, 4096)` **sequence** (FLUX's `txt`). T5 is padded to the
//!   variant's max length (**256** schnell / **512** dev, matching the diffusers FluxPipeline default)
//!   with the T5 pad id 0; every padded token is attended (FLUX applies no T5 attention mask), so the
//!   length is parity-critical.
//! - **CLIP tokenizer is vendored** (sc-2787 parity): the FLUX snapshot ships CLIP only as
//!   `vocab.json` + `merges.txt` (no `tokenizer.json`), and a byte-level BPE built from those
//!   mis-tokenizes CLIP's lowercased word-BPE — silently corrupting the pooled conditioning. So the
//!   HF-faithful `clip_tokenizer.json` is **compiled into the crate** (`assets/`, the same asset the
//!   mlx provider vendors) and never reconstructed from the snapshot. T5 ships a real
//!   `tokenizer_2/tokenizer.json`, which is used directly.
//! - **Flow-match schedule**: schnell uses the linear `get_schedule(steps, None)`; dev uses the
//!   resolution-dependent time-shifted `get_schedule(steps, Some((seq_len, 0.5, 1.15)))`. The denoise
//!   is candle's own additive Euler update `img = img + pred·(t_prev − t_curr)` over **descending**
//!   timesteps (1→0) — the FLUX sign convention is baked into the descending step, so unlike Z-Image
//!   there is **no velocity negation** and no separate `mu` scheduler gotcha (the shift lives inside
//!   `get_schedule`). Guidance is passed as a per-batch tensor and only *used* when the DiT config has
//!   `guidance_embed` (dev); schnell's DiT ignores it.
//! - **Deterministic seeding (sc-3673 parity)**: initial latent noise is drawn from a fixed-algorithm
//!   CPU RNG (`StdRng`, ChaCha) seeded by `seed` and moved to the device — NOT candle's CUDA
//!   `flux::sampling::get_noise` (`Tensor::randn`), whose seed→noise mapping is not launch-portable.
//!   The flow-match Euler step injects no per-step noise, so generation is a pure function of
//!   `(seed, request)` — what the gen-core-testkit seed-determinism check (sc-4481) requires.
//! - **Contract surface**: progress is `on_progress(Progress::Step/Decoding)`, cancellation is
//!   `req.cancel` → typed [`gen_core::Error::Canceled`], and each image is returned as a
//!   `gen_core::Image` (RGB8) — the worker owns asset writes.
//!
//! **First-slice surface (sc-3694), matching the SDXL/Z-Image slices:** txt2img only. img2img
//! (mlx's `Reference`/IP-adapter), LoRA/LoKr, and Q4/Q8 quantization are NOT wired here — they are
//! rejected loudly (the worker routes them to the Python fallback) rather than silently dropped.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{Module, VarBuilder};
use candle_gen::gen_core::{self, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::clip::text_model::{
    Activation as ClipActivation, ClipTextConfig, ClipTextTransformer,
};
use candle_transformers::models::flux::autoencoder::{AutoEncoder, Config as AeConfig};
use candle_transformers::models::flux::model::{Config as FluxConfig, Flux};
use candle_transformers::models::flux::sampling::{get_schedule, unpack, State};
use candle_transformers::models::flux::WithForward;
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use tokenizers::Tokenizer;

use crate::Variant;

/// FLUX latent channel count (the VAE's `z_channels` and the DiT's pre-pack channel count). The DiT
/// works on the 2×2-packed form (16·4 = 64 channels), but the raw noise / VAE latent is 16-channel.
const LATENT_CHANNELS: usize = 16;

/// FLUX dev's resolution-dependent flow-match time-shift endpoints (`base_shift`, `max_shift`),
/// matching the candle `flux` example's `get_schedule(.., Some((seq_len, 0.5, 1.15)))` and the
/// diffusers FluxPipeline. schnell uses no shift (`None`).
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// T5 pad token id (`<pad>`) — FLUX pads the T5 sequence to the variant max length with this id, and
/// attends every padded position (no attention mask), so it is parity-relevant.
const T5_PAD_TOKEN_ID: u32 = 0;

/// A txt2img pipeline handle: the snapshot `root`, the variant, and the compute device/dtype (bf16).
/// Loading the heavy components is done by [`load_components`](Self::load_components) and owned/cached
/// by the generator, mirroring the SDXL/Z-Image providers' lazy split.
pub(crate) struct Pipeline {
    variant: Variant,
    root: PathBuf,
    device: Device,
    dtype: DType,
}

/// The loaded FLUX components, `Arc`-shared so the generator can cache them across `generate` calls
/// and cheaply clone them out for a render. The T5 encoder is behind a `Mutex` because its
/// `forward` takes `&mut self` (relative-position-bias cache) while `Generator::generate` is `&self`;
/// it is locked only for the once-per-request text encode, never across the denoise.
#[derive(Clone)]
pub(crate) struct Components {
    clip: Arc<ClipTextTransformer>,
    t5: Arc<Mutex<T5EncoderModel>>,
    transformer: Arc<Flux>,
    vae: Arc<AutoEncoder>,
}

impl Pipeline {
    /// Build the (light) pipeline handle for the FLUX snapshot `root` at the given device/dtype. Does
    /// **no** weight I/O — components load lazily via [`load_components`](Self::load_components).
    pub(crate) fn load(variant: Variant, root: &Path, device: &Device, dtype: DType) -> Self {
        Self {
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            dtype,
        }
    }

    /// Load the four heavy components from the snapshot. The DiT (`flux1-*.safetensors`) and VAE
    /// (`ae.safetensors`) come from the root BFL single-file checkpoints; CLIP-L from `text_encoder/`
    /// and T5-XXL from `text_encoder_2/` (diffusers subdirs).
    pub(crate) fn load_components(&self) -> Result<Components> {
        // CLIP-L (openai/clip-vit-large-patch14 layout) under the diffusers `text_encoder/` subdir;
        // the candle transformer pools under the `text_model.` prefix. Config is fixed for FLUX.
        let clip_vb = self.mmap_vb(&[self.root.join("text_encoder/model.safetensors")])?;
        let clip = ClipTextTransformer::new(clip_vb.pp("text_model"), &clip_config())?;

        // T5-XXL under `text_encoder_2/` (sharded; config.json alongside).
        let t5_dir = self.root.join("text_encoder_2");
        let t5_cfg: T5Config = {
            let cfg = std::fs::read_to_string(t5_dir.join("config.json")).map_err(|e| {
                CandleError::Msg(format!("flux: read text_encoder_2/config.json: {e}"))
            })?;
            serde_json::from_str(&cfg)
                .map_err(|e| CandleError::Msg(format!("flux: parse T5 config.json: {e}")))?
        };
        let t5_vb = self.mmap_vb(&self.safetensors_in(&t5_dir)?)?;
        let t5 = T5EncoderModel::load(t5_vb, &t5_cfg)?;

        // FLUX DiT (original BFL checkpoint) at the snapshot root; config differs only by the
        // guidance embedding (dev embeds the guidance scale, schnell does not).
        let dit_vb = self.mmap_vb(&[self.root.join(self.variant.transformer_file())])?;
        let transformer = Flux::new(&flux_config(self.variant), dit_vb)?;

        // FLUX AutoEncoder (`ae.safetensors`) at the root.
        let vae_vb = self.mmap_vb(&[self.root.join("ae.safetensors")])?;
        let vae = AutoEncoder::new(&ae_config(self.variant), vae_vb)?;

        Ok(Components {
            clip: Arc::new(clip),
            t5: Arc::new(Mutex::new(t5)),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
        })
    }

    /// mmap a [`VarBuilder`] over `files` at this pipeline's dtype/device, erroring if any is missing.
    fn mmap_vb(&self, files: &[PathBuf]) -> Result<VarBuilder<'static>> {
        for f in files {
            if !f.is_file() {
                return Err(CandleError::Msg(format!(
                    "flux snapshot is missing {} (expected a black-forest-labs FLUX.1 snapshot at {})",
                    f.display(),
                    self.root.display()
                )));
            }
        }
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(files, self.dtype, &self.device)? };
        Ok(vb)
    }

    /// Sorted list of every `.safetensors` in `dir` (sharded T5 checkpoints ship as
    /// `model-0000n-of-0000m.safetensors`). Errors if none are found.
    fn safetensors_in(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| CandleError::Msg(format!("flux: read {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "flux: no .safetensors found in {}",
                dir.display()
            )));
        }
        Ok(files)
    }

    /// Encode `prompt` into FLUX's two conditioning tensors: the T5 sequence `(1, L, 4096)` and the
    /// CLIP pooled vector `(1, 768)`, both at the compute dtype. T5 is tokenized with the snapshot's
    /// `tokenizer_2/tokenizer.json` (padded to the variant max length with id 0); CLIP with the
    /// vendored `clip_tokenizer.json` (natural length — the pooled vector is the EOT hidden state, so
    /// trailing pad would not change it under CLIP's causal attention, and is omitted to match the
    /// candle reference exactly).
    pub(crate) fn text_embeddings(
        &self,
        comps: &Components,
        prompt: &str,
    ) -> Result<(Tensor, Tensor)> {
        encode_text(
            self.variant,
            &self.root,
            &self.device,
            self.dtype,
            &comps.clip,
            &comps.t5,
            prompt,
        )
    }

    /// Render `req` against pre-loaded `components`, emitting per-step progress and honoring
    /// `req.cancel`. Returns one `gen_core::Image` per `req.count` (each with seed `base_seed + index`).
    pub(crate) fn render(
        &self,
        req: &GenerationRequest,
        components: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(self.variant.default_steps() as usize);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        // Guidance is only consumed by the dev DiT (`guidance_embed`); schnell's DiT ignores the
        // tensor, so 0.0 there is inert. Validation rejects a guidance request on schnell already.
        let guidance: f64 = if self.variant.supports_guidance() {
            req.guidance.unwrap_or(self.variant.default_guidance()) as f64
        } else {
            0.0
        };

        // candle's get_noise geometry: the latent is padded to `div_ceil(16)*2` per side (== /8 for a
        // multiple-of-16 request) — i.e. the VAE's /8 latent. We enforce the /16 alignment in `validate`.
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;

        // Text embeddings are seed- and image-independent: encode once for the whole batch.
        let (t5_emb, clip_emb) = self.text_embeddings(components, &req.prompt)?;

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = image_seed(base_seed, index);

            // sc-3673 parity — deterministic, launch-portable initial noise in candle's get_noise
            // shape (1, 16, h/8, w/8): N(0,1) from a fixed-algorithm CPU RNG seeded by `seed`, built
            // on CPU then moved to the device.
            let n = LATENT_CHANNELS * lat_h * lat_w;
            let mut rng = StdRng::seed_from_u64(seed);
            let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
            let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
                .to_device(&self.device)?
                .to_dtype(self.dtype)?;

            // Pack noise + build the conditioning state (img/img_ids/txt/txt_ids/vec) exactly as the
            // candle reference. The packed token count drives dev's resolution-dependent time-shift.
            let state = State::new(&t5_emb, &clip_emb, &noise)?;
            let timesteps = if self.variant.is_dev() {
                get_schedule(steps, Some((state.img.dim(1)?, BASE_SHIFT, MAX_SHIFT)))
            } else {
                get_schedule(steps, None)
            };

            let latents = self.denoise(
                components.transformer.as_ref(),
                &state,
                &timesteps,
                guidance,
                req,
                on_progress,
            )?;

            on_progress(Progress::Decoding);
            images.push(self.decode(
                &components.vae,
                &latents,
                req.height as usize,
                req.width as usize,
            )?);
        }
        Ok(images)
    }

    /// The flow-match Euler denoise — candle's `flux::sampling::denoise` re-implemented inline so it
    /// can emit per-step `Progress` and honor `req.cancel`. The update is `img += pred·(t_prev−t_curr)`
    /// over the **descending** schedule (1→0); the FLUX sign convention lives in the descending step,
    /// so there is no velocity negation (contrast Z-Image). `guidance` is passed as a per-batch tensor
    /// and only embedded by the dev DiT.
    fn denoise(
        &self,
        model: &Flux,
        state: &State,
        timesteps: &[f64],
        guidance: f64,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let dev = &self.device;
        let guidance_t = Tensor::full(guidance as f32, b_sz, dev)?;
        let total = timesteps.len().saturating_sub(1) as u32;
        let mut img = state.img.clone();
        for (i, window) in timesteps.windows(2).enumerate() {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let (t_curr, t_prev) = (window[0], window[1]);
            let t_vec = Tensor::full(t_curr as f32, b_sz, dev)?;
            let pred = model.forward(
                &img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &t_vec,
                &state.vec,
                Some(&guidance_t),
            )?;
            img = (img + (pred * (t_prev - t_curr))?)?;
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }
        Ok(img)
    }

    /// Unpack the denoised latents `(1, h·w, 64)` back to `(1, 16, H/8, W/8)`, VAE-decode to an RGB8
    /// [`Image`]. The AutoEncoder applies its own `(z / scale) + shift` un-scale inside `decode`; the
    /// `[-1, 1]` output is mapped to `[0, 255]` u8.
    fn decode(
        &self,
        vae: &AutoEncoder,
        latents: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Image> {
        decode_latents(vae, latents, height, width)
    }
}

/// Encode `prompt` into FLUX's two conditioning tensors for `variant`: the T5 sequence `(1, L, 4096)`
/// and the CLIP pooled vector `(1, 768)`, both at `dtype`. Shared by the txt2img
/// [`Pipeline::text_embeddings`] and the IP-Adapter provider ([`crate::ip_provider`]) so the two never
/// drift on the parity-critical tokenization (T5 padded to the variant length; the vendored CLIP
/// tokenizer). `t5` is locked only for the once-per-request encode.
pub(crate) fn encode_text(
    variant: Variant,
    root: &Path,
    device: &Device,
    dtype: DType,
    clip: &ClipTextTransformer,
    t5: &Mutex<T5EncoderModel>,
    prompt: &str,
) -> Result<(Tensor, Tensor)> {
    // T5 sequence.
    let t5_tok = Tokenizer::from_file(root.join("tokenizer_2/tokenizer.json"))
        .map_err(|e| CandleError::Msg(format!("flux: load T5 tokenizer: {e}")))?;
    let mut t5_ids: Vec<u32> = t5_tok
        .encode(prompt, true)
        .map_err(|e| CandleError::Msg(format!("flux: T5 tokenize: {e}")))?
        .get_ids()
        .to_vec();
    // Pad/truncate to the variant's fixed T5 length (256 schnell / 512 dev). FLUX attends every
    // position (no T5 mask), so the padded length is parity-critical, not a perf knob.
    t5_ids.resize(variant.t5_max_len(), T5_PAD_TOKEN_ID);
    let t5_input = Tensor::new(t5_ids.as_slice(), device)?.unsqueeze(0)?;
    let t5_emb = {
        let mut t5 = t5.lock().expect("flux T5 mutex poisoned");
        t5.forward(&t5_input)?
    }
    .to_dtype(dtype)?;

    // CLIP pooled vector.
    const CLIP_TOKENIZER_JSON: &[u8] = include_bytes!("../assets/clip_tokenizer.json");
    let clip_tok = Tokenizer::from_bytes(CLIP_TOKENIZER_JSON)
        .map_err(|e| CandleError::Msg(format!("flux: load vendored CLIP tokenizer: {e}")))?;
    let clip_ids: Vec<u32> = clip_tok
        .encode(prompt, true)
        .map_err(|e| CandleError::Msg(format!("flux: CLIP tokenize: {e}")))?
        .get_ids()
        .to_vec();
    if clip_ids.is_empty() {
        return Err(CandleError::Msg("flux: empty CLIP tokenization".into()));
    }
    let clip_input = Tensor::new(clip_ids.as_slice(), device)?.unsqueeze(0)?;
    let clip_emb = clip.forward(&clip_input)?.to_dtype(dtype)?;

    Ok((t5_emb, clip_emb))
}

/// Unpack the denoised latents `(1, h·w, 64)` back to `(1, 16, H/8, W/8)`, VAE-decode to an RGB8
/// [`Image`]. Shared by the txt2img [`Pipeline::decode`] and the IP-Adapter provider. The AutoEncoder
/// applies its own `(z / scale) + shift` un-scale inside `decode`; the `[-1, 1]` output is mapped to
/// `[0, 255]` u8.
pub(crate) fn decode_latents(
    vae: &AutoEncoder,
    latents: &Tensor,
    height: usize,
    width: usize,
) -> Result<Image> {
    let latents = unpack(latents, height, width)?;
    let decoded = vae.decode(&latents)?.to_dtype(DType::F32)?; // (1, 3, H, W) in [-1, 1]
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

/// The per-image seed within a batch: image `index` of a `count`-image request renders at
/// `base_seed + index` (wrapping). Mirrors `mlx-gen-flux`'s `seed + i` convention, so the *n*-th
/// image of a batch reproduces in isolation as a single `count: 1` render at that derived seed. A
/// pure function so the law is unit-testable without a GPU.
pub(crate) fn image_seed(base_seed: u64, index: u32) -> u64 {
    base_seed.wrapping_add(index as u64)
}

/// The fixed CLIP-L (openai/clip-vit-large-patch14) text config FLUX uses — identical across
/// schnell/dev. Mirrors the candle `flux` example's hardcoded `ClipTextConfig`.
pub(crate) fn clip_config() -> ClipTextConfig {
    ClipTextConfig {
        vocab_size: 49408,
        projection_dim: 768,
        activation: ClipActivation::QuickGelu,
        intermediate_size: 3072,
        embed_dim: 768,
        max_position_embeddings: 77,
        pad_with: None,
        num_hidden_layers: 12,
        num_attention_heads: 12,
    }
}

/// The FLUX DiT config for `variant` — schnell and dev differ only in `guidance_embed`.
pub(crate) fn flux_config(variant: Variant) -> FluxConfig {
    if variant.is_dev() {
        FluxConfig::dev()
    } else {
        FluxConfig::schnell()
    }
}

/// The FLUX AutoEncoder config for `variant` (the scale/shift factors are identical across variants;
/// the variant arm mirrors the candle example's per-model selection).
pub(crate) fn ae_config(variant: Variant) -> AeConfig {
    if variant.is_dev() {
        AeConfig::dev()
    } else {
        AeConfig::schnell()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity anchors against `mlx-gen-flux`: distilled step defaults (4 schnell / 25 dev), guidance
    /// support (dev only) + the 3.5 dev default, and the T5 max lengths (256 / 512). GPU-free.
    #[test]
    fn variant_defaults_match_mlx_provider() {
        assert_eq!(Variant::Schnell.default_steps(), 4);
        assert_eq!(Variant::Dev.default_steps(), 25);
        assert!(!Variant::Schnell.supports_guidance());
        assert!(Variant::Dev.supports_guidance());
        assert_eq!(Variant::Dev.default_guidance(), 3.5);
        assert_eq!(Variant::Schnell.t5_max_len(), 256);
        assert_eq!(Variant::Dev.t5_max_len(), 512);
        assert_eq!(LATENT_CHANNELS, 16);
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

    /// The DiT config tracks the variant only through `guidance_embed`: dev embeds the guidance scale,
    /// schnell does not. The rest of the FLUX config is shared. GPU-free.
    #[test]
    fn flux_config_guidance_embed_tracks_variant() {
        assert!(flux_config(Variant::Dev).guidance_embed);
        assert!(!flux_config(Variant::Schnell).guidance_embed);
    }

    /// schnell uses an unshifted linear schedule; dev applies the resolution-dependent time-shift.
    /// Both produce `num_steps + 1` timesteps descending from 1 to 0 (the flow-match prior). The
    /// descending order is what makes the additive Euler update walk noise→data without a negation.
    #[test]
    fn schedule_is_descending_and_shift_tracks_variant() {
        let schnell = get_schedule(4, None);
        assert_eq!(schnell.len(), 5);
        assert!((schnell[0] - 1.0).abs() < 1e-9, "starts at 1: {schnell:?}");
        assert!(schnell[4].abs() < 1e-9, "ends at 0: {schnell:?}");
        for w in schnell.windows(2) {
            assert!(w[0] > w[1], "must descend: {schnell:?}");
        }
        // dev's time-shift moves the interior timesteps but keeps the 1→0 endpoints and monotonicity.
        let dev = get_schedule(25, Some((4096, BASE_SHIFT, MAX_SHIFT)));
        assert_eq!(dev.len(), 26);
        assert!((dev[0] - 1.0).abs() < 1e-9);
        assert!(dev[25].abs() < 1e-9);
        for w in dev.windows(2) {
            assert!(w[0] > w[1], "dev schedule must descend: {dev:?}");
        }
        // The shift actually changes the schedule (interior points differ from linear).
        let dev_linear = get_schedule(25, None);
        assert!(
            dev.iter()
                .zip(&dev_linear)
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "dev time-shift should differ from the linear schedule"
        );
    }
}
