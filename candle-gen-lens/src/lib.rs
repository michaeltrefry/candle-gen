//! # candle-gen-lens
//!
//! The **Lens / Lens-Turbo** text-to-image provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of the `mlx-gen` Lens port (epic 3164). Lens is a three-component model:
//!
//! 1. a **gpt-oss-20b** MoE LLM used **encoder-only** ([`text_encoder`]) — 24-layer / 32-expert /
//!    top-4, attention sinks, alternating sliding/full attention, YaRN RoPE, clamped-SwiGLU experts,
//!    MXFP4-native expert weights; run forward capturing hidden states at `[5, 11, 17, 23]`;
//! 2. a **48-layer dual-stream MMDiT** ([`transformer`], `LensTransformer2DModel`, sc-5112) —
//!    fused-QKV joint attention over `[img, txt]`, complex axial RoPE ([`rope`]), AdaLN dual
//!    modulation, SwiGLU MLPs, multi-layer text front-end;
//! 3. the **Flux.2 VAE** ([`vae`], `AutoencoderKLFlux2`, sc-5113) — reused from `candle-gen-flux2`
//!    via a thin decode shim (reshape the DiT output into the packed NCHW grid → `decode_packed`).
//!
//! This crate is being built story-by-story under epic **5107**. The first landed piece is the
//! gpt-oss encoder decoder block ([`text_encoder`], sc-5108): a from-scratch port — candle-transformers
//! ships no `gpt_oss` model (the Gate-0 spike found upstream PRs #3129/#3581/#3391 all unmerged), so
//! the decoder is adapted from the verified-parity reference in candle PR #3581 onto `candle_nn`.
//!
//! **Dtype:** the encoder runs **bf16** (the checkpoint's native non-expert dtype); the MXFP4 expert
//! weights are dequantized to bf16 at load (sc-5108 bring-up). The eventual MXFP4 → GGUF Q4 `QMatMul`
//! transcode that keeps the ~12 GB footprint is sc-5111.

pub mod adapters;
pub mod quant;
pub mod resolution;
pub mod rope;
pub mod schedule;
pub mod text;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use adapters::{merge_adapters, MergeReport};
pub use quant::QLinear;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::ModelRegistration;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use schedule::{
    cfg_rescale, euler_step, lens_sigmas, timesteps, LensSamplingDefaults, BASE, TURBO,
};
use text::{LensTokenizer, TXT_OFFSET};
use text_encoder::{Config as EncoderConfig, GptOssTextEncoder, DEFAULT_SELECTED_LAYERS};
use transformer::{LensDitConfig, LensTransformer};
use vae::Flux2Vae;

/// Registry id — the distilled turbo variant (4-step / guidance 1.0).
pub const MODEL_ID_TURBO: &str = "lens_turbo";
/// Registry id — the base variant (20-step / CFG 5.0).
pub const MODEL_ID_BASE: &str = "lens";

/// The VAE downsample factor (`vae_scale_factor`): a Lens latent cell maps to a 16×16 pixel tile
/// (Flux.2's 8× conv VAE composed with the 2× DiT patchify). Image dims must be multiples of this.
pub const VAE_SCALE_FACTOR: u32 = 16;

/// Fixed harmony-preamble `Current date:`. The preamble is the first [`TXT_OFFSET`] tokens, which are
/// **sliced off** before the DiT conditioning, so the date never reaches the image path — a fixed
/// constant keeps generation deterministic regardless of wall-clock.
pub const DEFAULT_DATE: &str = "2025-01-01";

/// The encoder + DiT run **bf16** (the checkpoint dtype). By default the MXFP4 experts dequantize to
/// bf16 at load; with `spec.quantize` they transcode to GGUF Q4/Q8 instead (sc-5111, the quantized
/// experts then compute in f32). The VAE always runs **f32** (the shared Flux.2 decoder).
const ENC_DTYPE: DType = DType::BF16;
const DIT_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;

/// The loaded four components, shared by both variants (cloneable `Arc` handles).
#[derive(Clone)]
struct Components {
    tokenizer: Arc<LensTokenizer>,
    encoder: Arc<GptOssTextEncoder>,
    transformer: Arc<LensTransformer>,
    vae: Arc<Flux2Vae>,
}

/// A loadable Lens pipeline (the snapshot root + device + any DiT LoRA/LoKr adapters + optional DiT
/// quant level); components are loaded lazily on first use.
struct Pipeline {
    root: PathBuf,
    device: Device,
    /// LoRA/LoKr adapters merged into the `transformer/` weights on load (sc-5116). Empty = the stock
    /// mmap path.
    adapters: Vec<AdapterSpec>,
    /// Q4/Q8 quantization requested at load (`None` = dense bf16). When set it transcodes **both** the
    /// gpt-oss encoder MoE experts to GGUF (sc-5111, the ~12 GB encoder footprint) and the DiT's
    /// compute-heavy linears (sc-5117) — the encoder is the memory hog, the DiT the compute. The VAE
    /// stays f32. One `Quant` drives both; each consumer maps it to the GGUF block dtype it needs.
    quant: Option<Quant>,
}

impl Pipeline {
    fn load(
        root: &Path,
        device: &Device,
        adapters: Vec<AdapterSpec>,
        quant: Option<Quant>,
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
            adapters,
            quant,
        }
    }

    /// The sorted `.safetensors` files of a snapshot sub-dir (errors if the dir or its weights are
    /// missing).
    fn component_files(&self, sub: &str) -> CResult<Vec<PathBuf>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "lens snapshot is missing the {sub}/ dir (expected a Lens diffusers snapshot at {})",
                self.root.display()
            )));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("lens: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "lens: no .safetensors in {sub}/ (at {})",
                dir.display()
            )));
        }
        Ok(files)
    }

    /// A `VarBuilder` over the `.safetensors` of a snapshot sub-dir, mmapped at `dtype`.
    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        let files = self.component_files(sub)?;
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &self.device)? };
        Ok(vb)
    }

    /// The DiT `VarBuilder` with any [`AdapterSpec`]s merged into its weights (sc-5116). With no
    /// adapters this is the stock mmap path; with adapters the `transformer/` shards load into a CPU
    /// map, each LoRA/LoKr delta is folded in ([`adapters::merge_adapters`], f32 math), then the DiT
    /// is built from the merged map — **merge, not residual** (the Lens flow-match sampler is
    /// chaos-sensitive; `(W+δ)·x` ≠ `W·x + δ·x` to ~1 ULP, a visibly different image).
    fn transformer_vb(&self) -> CResult<VarBuilder<'static>> {
        if self.adapters.is_empty() {
            return self.component_vb("transformer", DIT_DTYPE);
        }
        let files = self.component_files("transformer")?;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for f in &files {
            let part = candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?;
            tensors.extend(part);
        }
        adapters::merge_adapters(&mut tensors, &self.adapters)?;
        Ok(VarBuilder::from_tensors(tensors, DIT_DTYPE, &self.device))
    }

    fn load_components(&self) -> CResult<Components> {
        let tokenizer =
            LensTokenizer::from_file(self.root.join("tokenizer").join("tokenizer.json"))?;
        let encoder = GptOssTextEncoder::new_quant(
            &EncoderConfig::gpt_oss_20b(),
            self.component_vb("text_encoder", ENC_DTYPE)?,
            self.quant.map(quant::ggml_dtype),
        )?;
        let mut transformer = LensTransformer::new(&LensDitConfig::lens(), self.transformer_vb()?)?;
        // Q4/Q8 transcode the DiT's compute-heavy linears after the dense weights (and any merged
        // adapter delta) have loaded — `apply_adapters → quantize` ordering (sc-5117).
        if let Some(quant) = self.quant {
            transformer.quantize(quant)?;
        }
        let vae = Flux2Vae::new(self.component_vb("vae", VAE_DTYPE)?)?;
        Ok(Components {
            tokenizer: Arc::new(tokenizer),
            encoder: Arc::new(encoder),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
        })
    }

    /// Encode one prompt → its `num_text_layers` captured gpt-oss layers (sliced at [`TXT_OFFSET`]) +
    /// the valid mask `[1, S]` (all-1; a single prompt is unpadded). A prompt shorter than the offset
    /// (never, for real prompts) collapses to length-0 features.
    fn encode_one(
        &self,
        comps: &Components,
        prompt: &str,
        date: &str,
    ) -> CResult<(Vec<Tensor>, Tensor)> {
        let ids = comps.tokenizer.encode(prompt, date)?;
        let l = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, l), &self.device)?;
        let layers = comps
            .encoder
            .capture(&input_ids, &DEFAULT_SELECTED_LAYERS)?;
        if l > TXT_OFFSET {
            let s = l - TXT_OFFSET;
            let features = layers
                .iter()
                .map(|f| f.narrow(1, TXT_OFFSET, s))
                .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
            let mask = Tensor::ones((1, s), DType::F32, &self.device)?;
            Ok((features, mask))
        } else {
            let dim = layers[0].dim(2)?;
            let features = (0..DEFAULT_SELECTED_LAYERS.len())
                .map(|_| Tensor::zeros((1, 0, dim), ENC_DTYPE, &self.device))
                .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
            let mask = Tensor::zeros((1, 0), DType::F32, &self.device)?;
            Ok((features, mask))
        }
    }

    /// Encode positives + negatives and assemble the joint CFG batch: each feature layer is
    /// `[2, S_txt, 2880]` (`[pos; neg]`) and the mask is `[2, S_txt]` (`1` = valid). An empty negative
    /// is the **unconditional branch**: zero text features + an all-zero mask (no text tokens), not a
    /// second encode.
    fn encode_prompt(
        &self,
        comps: &Components,
        prompt: &str,
        negative: &str,
        date: &str,
    ) -> CResult<(Vec<Tensor>, Tensor)> {
        let (pos_feats, pos_mask) = self.encode_one(comps, prompt, date)?;
        let s_pos = pos_feats[0].dim(1)?;
        let (neg_feats, neg_mask) = if negative.trim().is_empty() {
            let zeros = pos_feats
                .iter()
                .map(|f| f.zeros_like())
                .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
            (zeros, pos_mask.zeros_like()?)
        } else {
            self.encode_one(comps, negative, date)?
        };
        let s_neg = neg_feats[0].dim(1)?;

        let target = s_pos.max(s_neg);
        let pos_feats = pad_features(&pos_feats, s_pos, target, &self.device)?;
        let neg_feats = pad_features(&neg_feats, s_neg, target, &self.device)?;
        let pos_mask = pad_mask(&pos_mask, s_pos, target, &self.device)?;
        let neg_mask = pad_mask(&neg_mask, s_neg, target, &self.device)?;

        let mut features = Vec::with_capacity(pos_feats.len());
        for (pf, nf) in pos_feats.iter().zip(neg_feats.iter()) {
            features.push(Tensor::cat(&[pf, nf], 0)?.to_dtype(DIT_DTYPE)?);
        }
        let mask = Tensor::cat(&[&pos_mask, &neg_mask], 0)?;
        Ok((features, mask))
    }

    /// The denoising loop over the joint CFG conditioning + an initial latent
    /// (`[1, latent_h·latent_w, 128]`). Returns the final patch-space latents (feed to [`vae::decode`]).
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        comps: &Components,
        features: &[Tensor],
        mask: &Tensor,
        init_latents: &Tensor,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance: f32,
        is_cancelled: &dyn Fn() -> bool,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> CResult<Tensor> {
        let sigmas = lens_sigmas(num_steps, latent_h, latent_w);
        let ts = timesteps(&sigmas);
        let mut latents = init_latents.to_dtype(DIT_DTYPE)?;
        for (i, &sigma) in ts.iter().enumerate() {
            if is_cancelled() {
                return Err(CandleError::Msg("lens: generation cancelled".into()));
            }
            // Joint CFG batch: duplicate the latent (cond/uncond share x_t), one DiT call.
            let hidden = Tensor::cat(&[&latents, &latents], 0)?; // [2, seq, 128]
            let noise = comps.transformer.forward(
                &hidden,
                features,
                Some(mask),
                sigma,
                1,
                latent_h,
                latent_w,
            )?;
            let cond = noise.narrow(0, 0, 1)?;
            let uncond = noise.narrow(0, 1, 1)?;
            let noise_pred = cfg_rescale(&cond, &uncond, guidance)?;
            latents = euler_step(&latents, &noise_pred, &sigmas, i)?;
            on_step(i + 1, num_steps);
        }
        Ok(latents)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        defaults: Defaults,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(defaults.steps as usize);
        let guidance = req.guidance.unwrap_or(defaults.guidance);
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let total = steps as u32;
        let latent_h = (req.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (req.width / VAE_SCALE_FACTOR) as usize;

        let (features, mask) = self.encode_prompt(comps, &req.prompt, negative, DEFAULT_DATE)?;

        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let seed = base_seed.wrapping_add(index as u64);
            let init = create_noise(seed, latent_h, latent_w, &self.device)?;
            let latents = self.denoise(
                comps,
                &features,
                &mask,
                &init,
                latent_h,
                latent_w,
                steps,
                guidance,
                &|| req.cancel.is_cancelled(),
                &mut |cur, _| {
                    on_progress(Progress::Step {
                        current: cur as u32,
                        total,
                    })
                },
            )?;
            on_progress(Progress::Decoding);
            let decoded = vae::decode(&comps.vae, &latents, latent_h, latent_w)?;
            images.push(to_image(&decoded)?);
        }
        Ok(images)
    }
}

/// Zero-pad each `[B, cur, C]` feature layer along the sequence axis to length `target`.
fn pad_features(
    features: &[Tensor],
    cur: usize,
    target: usize,
    device: &Device,
) -> candle_gen::candle_core::Result<Vec<Tensor>> {
    if cur == target {
        return Ok(features.to_vec());
    }
    let pad = target - cur;
    features
        .iter()
        .map(|f| {
            let (b, _, c) = f.dims3()?;
            let z = Tensor::zeros((b, pad, c), f.dtype(), device)?;
            Tensor::cat(&[f, &z], 1)
        })
        .collect()
}

/// Zero-pad a `[B, cur]` mask along the sequence axis to length `target`.
fn pad_mask(
    mask: &Tensor,
    cur: usize,
    target: usize,
    device: &Device,
) -> candle_gen::candle_core::Result<Tensor> {
    if cur == target {
        return Ok(mask.clone());
    }
    let pad = target - cur;
    let b = mask.dim(0)?;
    let z = Tensor::zeros((b, pad), DType::F32, device)?;
    Tensor::cat(&[mask, &z], 1)
}

/// Deterministic packed initial noise `[1, latent_h·latent_w, 128]` (sc-3673 pattern): N(0,1) from a
/// fixed CPU RNG (NOT candle's CUDA `randn`), then moved to `device`.
fn create_noise(
    seed: u64,
    latent_h: usize,
    latent_w: usize,
    device: &Device,
) -> candle_gen::candle_core::Result<Tensor> {
    let seq = latent_h * latent_w;
    let n = seq * 128;
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Tensor::from_vec(data, (1, seq, 128), &Device::Cpu)?.to_device(device)
}

/// Convert a decoded image `[1, 3, H, W]` (NCHW) in `[-1, 1]` to an RGB8 [`Image`].
fn to_image(decoded: &Tensor) -> CResult<Image> {
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "lens: expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`) baked into the loaded generator.
#[derive(Clone, Copy)]
struct Defaults {
    id: &'static str,
    steps: u32,
    guidance: f32,
}

impl Defaults {
    const fn from(id: &'static str, d: LensSamplingDefaults) -> Self {
        Self {
            id,
            steps: d.num_steps as u32,
            guidance: d.guidance_scale,
        }
    }
}

const TURBO_DEFAULTS: Defaults = Defaults::from(MODEL_ID_TURBO, TURBO);
const BASE_DEFAULTS: Defaults = Defaults::from(MODEL_ID_BASE, BASE);

/// A loaded, dispatchable Lens generator: the pipeline + the variant's descriptor & sampling defaults.
/// Components are cached after the first `generate`.
pub struct LensGenerator {
    descriptor: ModelDescriptor,
    defaults: Defaults,
    pipeline: Pipeline,
    components: Mutex<Option<Components>>,
}

impl LensGenerator {
    /// Test/parity constructor: a generator over a snapshot dir with the turbo defaults (lazy
    /// components). The sampling defaults are irrelevant to [`denoise_for_parity`] (which takes
    /// explicit `steps`/`guidance`); this just gives the e2e gate a concrete generator to drive.
    pub fn for_parity(root: impl AsRef<Path>) -> CResult<Self> {
        let device = candle_gen::default_device()?;
        Ok(Self {
            descriptor: descriptor_turbo(),
            defaults: TURBO_DEFAULTS,
            pipeline: Pipeline::load(root.as_ref(), &device, Vec::new(), None),
            components: Mutex::new(None),
        })
    }

    fn components(&self) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("lens components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = self.pipeline.load_components()?;
        *guard = Some(c.clone());
        Ok(c)
    }

    /// e2e-parity hook (sc-5115): encode → denoise from **injected** latents → decode, factoring out
    /// the RNG so a cross-build comparison isolates the wiring. Returns the final patch latents
    /// `[1, seq, 128]` and the decoded image `[1, 3, H, W]` in `[-1, 1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_for_parity(
        &self,
        prompt: &str,
        negative: &str,
        date: &str,
        init_latents: &Tensor,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance: f32,
    ) -> CResult<(Tensor, Tensor)> {
        let comps = self
            .components()
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        let (features, mask) = self
            .pipeline
            .encode_prompt(&comps, prompt, negative, date)?;
        let latents = self.pipeline.denoise(
            &comps,
            &features,
            &mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance,
            &|| false,
            &mut |_, _| {},
        )?;
        let decoded = vae::decode(&comps.vae, &latents, latent_h, latent_w)?;
        Ok((latents, decoded))
    }
}

impl Generator for LensGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(self.defaults.id, &self.descriptor.capabilities, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let comps = self.components()?;
        let images = self
            .pipeline
            .render(req, &comps, self.defaults, on_progress)?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Lens' identity + capabilities for `id` — constructible without loading weights. The norm-rescaled
/// CFG path is always present; turbo simply defaults guidance to 1.0. **Standard guidance, not
/// true-CFG.** LoRA/LoKr are wired (sc-5116, merged into the DiT on load); Q4/Q8 quant is wired for
/// **both** the gpt-oss encoder experts (sc-5111) and the DiT (sc-5117, GGUF `QMatMul` folded in after
/// the merge).
fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "lens",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![], // pure T2I — no img2img / control / IP in the Lens port
            supports_lora: true,
            supports_lokr: true,
            samplers: vec!["flow_match_euler"],
            schedulers: vec!["flow_match"],
            // Buckets span 736..2080 (all ÷16); allow any ÷16 size in a sane range.
            min_size: 256,
            max_size: 2080,
            max_count: 8,
            mac_only: false,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            // The Lens schedule computes its own empirical-μ shift internally (not a loader hint).
            requires_sigma_shift: false,
        },
    }
}

/// Public descriptor accessors (used by the registry submits + tests).
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(MODEL_ID_TURBO)
}
pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(MODEL_ID_BASE)
}

/// Capability-driven request validation (unit-testable without loaded weights).
fn validate_request(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    caps.validate_request(id, req)?;
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
    }
    if !req.width.is_multiple_of(VAE_SCALE_FACTOR) || !req.height.is_multiple_of(VAE_SCALE_FACTOR) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: width/height must be multiples of {VAE_SCALE_FACTOR} (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

/// Construct a lazy candle Lens generator with the given per-variant defaults. `spec.weights` must be
/// a `microsoft/Lens` / `microsoft/Lens-Turbo` diffusers snapshot dir (`tokenizer/`, `text_encoder/`,
/// `transformer/`, `vae/`). DiT LoRA/LoKr adapters (`spec.adapters`) are merged into the transformer
/// weights on first use (sc-5116). `spec.quantize` (Q4/Q8) transcodes **both** the gpt-oss encoder
/// experts to GGUF `Q4_0`/`Q8_0` (sc-5111; ~13 GB at Q4 vs ~40 GB bf16, the encoder is the memory hog)
/// and the DiT's compute-heavy linears (sc-5117, folded in after the adapter merge). ControlNet /
/// IP-Adapter are not part of the Lens port and are rejected here.
fn load_with(spec: &LoadSpec, defaults: Defaults) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{}: expects a Lens snapshot directory (tokenizer/ text_encoder/ transformer/ vae/), \
                 not a single .safetensors file",
                defaults.id
            )));
        }
    };
    // `spec.quantize` (encoder + DiT) and `spec.adapters` (DiT merge, sc-5116) are both applied
    // downstream in `load_components`/`transformer_vb`, so neither is rejected here.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: ControlNet / IP-Adapter conditioning is not part of the Lens port",
            defaults.id
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(LensGenerator {
        descriptor: descriptor_for(defaults.id),
        defaults,
        pipeline: Pipeline::load(&root, &device, spec.adapters.clone(), spec.quantize),
        components: Mutex::new(None),
    }))
}

fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, TURBO_DEFAULTS)
}
fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, BASE_DEFAULTS)
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_turbo, load: load_turbo }
}
inventory::submit! {
    ModelRegistration { descriptor: descriptor_base, load: load_base }
}

/// Force-link hook (keeps the `inventory::submit!` registrations from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn descriptors_are_lens() {
        for (d, id, steps, g) in [
            (descriptor_turbo(), MODEL_ID_TURBO, 4u32, 1.0f32),
            (descriptor_base(), MODEL_ID_BASE, 20, 5.0),
        ] {
            assert_eq!(d.id, id);
            assert_eq!(d.family, "lens");
            assert_eq!(d.backend, "candle");
            assert_eq!(d.modality, Modality::Image);
            assert!(d.capabilities.supports_guidance);
            assert!(d.capabilities.supports_negative_prompt);
            assert!(!d.capabilities.supports_true_cfg);
            assert!(d.capabilities.conditioning.is_empty());
            assert!(!d.capabilities.mac_only);
            let def = if id == MODEL_ID_TURBO {
                TURBO_DEFAULTS
            } else {
                BASE_DEFAULTS
            };
            assert_eq!((def.steps, def.guidance), (steps, g));
        }
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        // The two `inventory::submit!`s link into this test binary, so the registry resolves both
        // ids. Loading is **lazy** (weights are read on first `generate`), so construction succeeds
        // even with a bogus dir — proving registration without needing the ~50 GB snapshot.
        for id in [MODEL_ID_TURBO, MODEL_ID_BASE] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/lens".into()));
            assert!(
                candle_gen::gen_core::registry::load(id, &spec).is_ok(),
                "{id} should resolve + lazily construct in the registry"
            );
        }
    }

    #[test]
    fn validate_rejects_bad_inputs() {
        let caps = descriptor_turbo().capabilities;
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &ok).is_ok());
        let empty = GenerationRequest {
            prompt: "".into(),
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &empty).is_err());
        let bad_dims = GenerationRequest {
            width: 1000,
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &bad_dims).is_err());
    }
}
