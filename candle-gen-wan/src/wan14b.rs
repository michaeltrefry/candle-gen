//! The Wan2.2 **A14B dual-expert MoE** video providers (sc-5174) — the candle (Windows/CUDA) siblings
//! of `mlx-gen-wan`'s `wan2_2_t2v_14b` / `wan2_2_i2v_14b`. Both register as `backend = "candle"`,
//! [`Modality::Video`].
//!
//! Wan2.2's "MoE" is **two complete `WanTransformer3DModel` checkpoints**, not token routing: a
//! **high-noise** expert (`transformer/`) and a **low-noise** expert (`transformer_2/`). A single
//! flow-match scheduler drives the denoise; each step picks the high expert while the integer timestep
//! is `≥ boundary·1000` (T2V `0.875`, I2V `0.900`) and the low expert below it, switching the
//! transformer, its (per-expert) text context, and its guidance scale together (T2V 3.0/4.0, I2V
//! 3.5/3.5). The experts share the dimension-parametric [`WanTransformer`] (loaded with
//! [`TransformerConfig::t2v_14b`]/[`i2v_14b`](TransformerConfig::i2v_14b)) and the [`crate::vae16`] z16
//! VAE — *not* the 5B's z48 VAE (the 14B emits 16-channel latents).
//!
//! **T2V** (`wan2_2_t2v_14b`): pure text→video. **I2V** (`wan2_2_i2v_14b`): channel-concat conditioning
//! — the reference image's first-frame z16 latent + a temporal mask form a 20-channel `y` appended to
//! the 16-channel noise latent (in_dim 36) every forward (the image enters via the channels, not noise).
//!
//! **Dtypes:** UMT5 + VAE run **f32**; the experts run **bf16** (norms/modulation upcast to f32),
//! mirroring the 5B. The VAE decode **streams one latent frame at a time** (sc-5176) to bound the
//! decode-stage peak — the heavier-than-5B fix the story (sc-5174) requires.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, MoeExpert, Progress,
    Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use crate::config::{
    TextEncoderConfig, TransformerConfig, Vae16Config, DEFAULT_FPS_14B, DEFAULT_FRAMES_14B,
    DEFAULT_STEPS_14B, I2V_14B_BOUNDARY, I2V_14B_FLOW_SHIFT, I2V_14B_GUIDANCE_HIGH,
    I2V_14B_GUIDANCE_LOW, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B, NEGATIVE_FALLBACK,
    NUM_TRAIN_TIMESTEPS, SIZE_MULTIPLE_14B, T2V_14B_BOUNDARY, T2V_14B_FLOW_SHIFT,
    T2V_14B_GUIDANCE_HIGH, T2V_14B_GUIDANCE_LOW, VAE16_STRIDE_SPATIAL, VAE16_STRIDE_TEMPORAL,
};
use crate::pipeline::{cfg, create_noise, frames_to_images};
use crate::rope::WanRope;
use crate::scheduler::{FlowScheduler, Sampler};
use crate::text_encoder::Umt5Encoder;
use crate::transformer::WanTransformer;
use crate::vae16::WanVae16;

/// The experts run bf16 (the diffusers fp32 weights load as bf16, the 5B regime); UMT5 + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;
const VAE_DTYPE: DType = DType::F32;
const Z_DIM: usize = 16;

/// Which A14B model this generator serves — selects in_dim (16 vs 36), the MoE knobs, and whether the
/// VAE carries an encoder (I2V conditioning).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    T2v,
    I2v,
}

impl Variant {
    fn id(self) -> &'static str {
        match self {
            Variant::T2v => MODEL_ID_T2V_14B,
            Variant::I2v => MODEL_ID_I2V_14B,
        }
    }

    fn dit_cfg(self) -> TransformerConfig {
        match self {
            Variant::T2v => TransformerConfig::t2v_14b(),
            Variant::I2v => TransformerConfig::i2v_14b(),
        }
    }

    /// `(boundary, default flow-shift, guidance_low, guidance_high)`.
    fn moe_knobs(self) -> (f64, f64, f32, f32) {
        match self {
            Variant::T2v => (
                T2V_14B_BOUNDARY,
                T2V_14B_FLOW_SHIFT,
                T2V_14B_GUIDANCE_LOW,
                T2V_14B_GUIDANCE_HIGH,
            ),
            Variant::I2v => (
                I2V_14B_BOUNDARY,
                I2V_14B_FLOW_SHIFT,
                I2V_14B_GUIDANCE_LOW,
                I2V_14B_GUIDANCE_HIGH,
            ),
        }
    }
}

#[derive(Clone)]
struct Components {
    te: Arc<Umt5Encoder>,
    /// `transformer/` — the **high-noise** expert (timestep ≥ boundary).
    high: Arc<WanTransformer>,
    /// `transformer_2/` — the **low-noise** expert (timestep < boundary).
    low: Arc<WanTransformer>,
    vae: Arc<WanVae16>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    dit_cfg: TransformerConfig,
    vae_cfg: Vae16Config,
    variant: Variant,
    root: PathBuf,
    device: Device,
    /// Trained LoRA/LoKr adapters to merge into the experts at load (sc-5167). Each is routed to the
    /// high and/or low expert by its [`AdapterSpec::moe_expert`].
    adapters: Vec<AdapterSpec>,
}

impl Pipeline {
    fn load(root: &Path, device: &Device, variant: Variant, adapters: Vec<AdapterSpec>) -> Self {
        Self {
            te_cfg: TextEncoderConfig::umt5_xxl(),
            dit_cfg: variant.dit_cfg(),
            vae_cfg: Vae16Config::wan21(),
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            adapters,
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "wan-14b snapshot is missing the {sub}/ dir (expected a Wan2.2-{}-A14B diffusers \
                 snapshot at {})",
                match self.variant {
                    Variant::T2v => "T2V",
                    Variant::I2v => "I2V",
                },
                self.root.display()
            )));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("wan-14b: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "wan-14b: no .safetensors in {sub}/ (at {})",
                dir.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &self.device)? };
        Ok(vb)
    }

    /// Build one expert from its `sub` dir, folding in any adapter whose [`AdapterSpec::moe_expert`]
    /// targets it (`Some(expert)` or `None` = shared). With no adapter for this expert, the fast
    /// mmap path is used; otherwise the weights are loaded to CPU, the delta is merged
    /// ([`crate::adapters::merge_adapters`], f32 math), and the expert is built from the merged map
    /// (`VarBuilder::from_tensors` casts/moves per-tensor on `get`, so peak GPU is unchanged) — the
    /// merge-not-residual pattern the SDXL/Z-Image ports established.
    fn build_expert(&self, sub: &str, expert: MoeExpert) -> CResult<WanTransformer> {
        let specs: Vec<AdapterSpec> = self
            .adapters
            .iter()
            .filter(|s| s.moe_expert.is_none_or(|e| e == expert))
            .cloned()
            .collect();
        if specs.is_empty() {
            return Ok(WanTransformer::new(
                &self.dit_cfg,
                self.component_vb(sub, DIT_DTYPE)?,
            )?);
        }
        let mut map = self.load_component_map(sub)?;
        crate::adapters::merge_adapters(&mut map, &specs)?;
        let vb = VarBuilder::from_tensors(map, DIT_DTYPE, &self.device);
        Ok(WanTransformer::new(&self.dit_cfg, vb)?)
    }

    /// Load every `.safetensors` in the component subdir `sub` into one CPU tensor map (native dtype) —
    /// the merge-ready form the adapter fold needs (vs the mmap `component_vb` fast path).
    fn load_component_map(&self, sub: &str) -> CResult<HashMap<String, Tensor>> {
        let dir = self.root.join(sub);
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| CandleError::Msg(format!("wan-14b: read {sub}/: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "wan-14b: no .safetensors in {sub}/ (at {})",
                dir.display()
            )));
        }
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for f in &files {
            map.extend(cst::load(f, &Device::Cpu)?);
        }
        Ok(map)
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Umt5Encoder::new(&self.te_cfg, self.component_vb("text_encoder", ENC_DTYPE)?)?;
        // transformer/ = high-noise expert, transformer_2/ = low-noise expert (diffusers WanPipeline).
        let high = self.build_expert("transformer", MoeExpert::High)?;
        let low = self.build_expert("transformer_2", MoeExpert::Low)?;
        let vae_vb = self.component_vb("vae", VAE_DTYPE)?;
        let vae = match self.variant {
            // I2V needs the VAE encoder (the conditioning image's first-frame latent).
            Variant::I2v => WanVae16::new_with_encoder(&self.vae_cfg, vae_vb)?,
            Variant::T2v => WanVae16::new(&self.vae_cfg, vae_vb)?,
        };
        Ok(Components {
            te: Arc::new(te),
            high: Arc::new(high),
            low: Arc::new(low),
            vae: Arc::new(vae),
        })
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, 512, 4096]` (f32), zero-padded to `max_length` (the DiT
    /// cross-attends over the 512-padded context — the same rule as the 5B, sc-3697).
    fn encode(&self, te: &Umt5Encoder, prompt: &str) -> CResult<Tensor> {
        let tok = TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.te_cfg.max_length,
                pad_token_id: self.te_cfg.pad_token_id,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("wan-14b: load tokenizer: {e}")))?;
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("wan-14b: tokenize: {e}")))?;
        let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        if ids.is_empty() {
            // Empty prompt → zero ids → a degenerate `(1,1)` tensor (the old `.max(1)` padded the
            // shape, not the data) whose 0-element f32 embedding gather reads out of bounds on CUDA
            // (`CUDA_ERROR_ILLEGAL_ADDRESS`, surfacing as a misleading cublas failure). Emit one pad
            // token so a 0-length sequence never reaches the gather. (sc-7078)
            ids.push(self.te_cfg.pad_token_id as u32);
        }
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let embeds = te.encode(&input_ids)?;
        let max_len = self.te_cfg.max_length;
        let dim = embeds.dim(2)?;
        match len.cmp(&max_len) {
            std::cmp::Ordering::Less => {
                let pad = Tensor::zeros((1, max_len - len, dim), embeds.dtype(), &self.device)?;
                Ok(Tensor::cat(&[&embeds, &pad], 1)?)
            }
            std::cmp::Ordering::Greater => Ok(embeds.narrow(1, 0, max_len)?),
            std::cmp::Ordering::Equal => Ok(embeds),
        }
    }

    /// Build the I2V channel-concat conditioning `y` `[1, 20, t_lat, h_lat, w_lat]` =
    /// `[mask(4), z_video(16)]`: a conditioning video (frame 0 = the preprocessed image, the rest zero)
    /// is z16-VAE-encoded, and a temporal mask (1.0 at latent frame 0, else 0.0) is prepended. Mirrors
    /// `generate_wan.py`'s `is_i2v_channel_concat` setup. Constant across denoise steps + both experts.
    fn build_i2v_y(
        &self,
        vae: &WanVae16,
        image: &Image,
        frames: u32,
        width: u32,
        height: u32,
    ) -> CResult<Tensor> {
        // Conditioning video [1, 3, F, H, W]: frame 0 = image (in [-1,1]), rest zeros.
        let first = preprocess_i2v_image(image, width, height, &self.device)?; // [1,3,1,H,W]
        let video = if frames > 1 {
            let rest = Tensor::zeros(
                (1, 3, frames as usize - 1, height as usize, width as usize),
                DType::F32,
                &self.device,
            )?;
            Tensor::cat(&[&first, &rest], 2)?
        } else {
            first
        };
        let z_video = vae.encode(&video)?; // [1, 16, t_lat, h_lat, w_lat]

        // Mask dims follow the encoder's actual output, so they always match `z_video`.
        let (_, _, t_lat, h_lat, w_lat) = z_video.dims5()?;
        // 4-channel temporal mask: 1.0 at latent frame 0 (all channels/spatial), 0.0 elsewhere.
        let plane = h_lat * w_lat;
        let mut mask = vec![0f32; 4 * t_lat * plane];
        for c in 0..4 {
            let base = c * t_lat * plane; // temporal index 0 of channel c
            for v in mask.iter_mut().skip(base).take(plane) {
                *v = 1.0;
            }
        }
        let mask = Tensor::from_vec(mask, (1, 4, t_lat, h_lat, w_lat), &self.device)?;
        Ok(Tensor::cat(&[&mask, &z_video], 1)?) // [1, 20, t_lat, h_lat, w_lat]
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS_14B as usize);
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES_14B);
        let fps = req.fps.unwrap_or(DEFAULT_FPS_14B);
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let sampler = Sampler::parse(req.sampler.as_deref());
        let (boundary, default_shift, gl, gh) = self.variant.moe_knobs();
        let shift = req
            .scheduler_shift
            .map(|s| s as f64)
            .unwrap_or(default_shift);
        // A scalar request guidance overrides both experts; else the per-expert (low, high) defaults.
        let (g_low, g_high) = match req.guidance {
            Some(g) => (g as f64, g as f64),
            None => (gl as f64, gh as f64),
        };

        // Text encode (pos + neg) once; project to each expert's context (per-expert text_embedder).
        let pos = self.encode(&comps.te, &req.prompt)?;
        let neg_prompt = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
        let neg = self.encode(&comps.te, neg_prompt)?;
        let high_pos = comps.high.embed_text(&pos)?;
        let high_neg = comps.high.embed_text(&neg)?;
        let low_pos = comps.low.embed_text(&pos)?;
        let low_neg = comps.low.embed_text(&neg)?;

        // Latent geometry (z16 strides) + RoPE for the shared token grid.
        let t_lat = ((frames - 1) / VAE16_STRIDE_TEMPORAL + 1) as usize;
        let h_lat = (req.height / VAE16_STRIDE_SPATIAL) as usize;
        let w_lat = (req.width / VAE16_STRIDE_SPATIAL) as usize;
        let (pt, ph, pw) = self.dit_cfg.patch;
        let (ppf, pph, ppw) = (t_lat / pt, h_lat / ph, w_lat / pw);
        let (cos, sin) = WanRope::new(&self.dit_cfg).cos_sin(ppf, pph, ppw, &self.device)?;

        // I2V: build the constant channel-concat conditioning `y` (needs the VAE encoder).
        let y = match self.variant {
            Variant::I2v => {
                let image = i2v_reference(req).ok_or_else(|| {
                    CandleError::Msg(format!(
                        "{}: image-to-video requires a Reference conditioning image",
                        self.variant.id()
                    ))
                })?;
                Some(self.build_i2v_y(&comps.vae, image, frames, req.width, req.height)?)
            }
            Variant::T2v => None,
        };

        let mut latents = create_noise(seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;
        let mut sched = FlowScheduler::new(sampler, steps, shift);
        let boundary_ts = boundary * NUM_TRAIN_TIMESTEPS as f64;
        let total = steps as u32;

        for i in 0..steps {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let t = sched.timestep(i);
            // MoE: high-noise expert at/above the boundary timestep, low-noise below — switching the
            // transformer, its context, and its guidance together.
            let (expert, ctx_pos, ctx_neg, guidance) = if t >= boundary_ts {
                (&comps.high, &high_pos, &high_neg, g_high)
            } else {
                (&comps.low, &low_pos, &low_neg, g_low)
            };
            // I2V: concat the conditioning `y` onto the noise latent (→ in_dim 36) before the forward.
            let x = match &y {
                Some(y) => Tensor::cat(&[&latents, y], 1)?,
                None => latents.clone(),
            };
            let v_pos = expert.forward(&x, ctx_pos, t, &cos, &sin)?;
            let v = if guidance > 1.0 {
                let v_neg = expert.forward(&x, ctx_neg, t, &cos, &sin)?;
                cfg(&v_pos, &v_neg, guidance)?
            } else {
                v_pos
            };
            latents = sched.step(&v, &latents)?; // 16-channel latent (out_dim 16)
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }

        on_progress(Progress::Decoding);
        let decoded = comps.vae.decode(&latents)?;
        let images = frames_to_images(&decoded)?;
        Ok((images, fps))
    }
}

/// The single conditioning reference image for I2V (the first video frame), if present.
fn i2v_reference(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

/// Preprocess an I2V conditioning [`Image`] to `[1, 3, 1, height, width]` f32 in `[-1, 1]`: a cover-fit
/// resize (`scale = max(W/iw, H/ih)`) + center-crop to the target, then `px/255·2 − 1`. Uses **bilinear**
/// resampling (the reference's PIL-exact LANCZOS, for bit-exact MLX parity, is a follow-up — sc-5174).
pub(crate) fn preprocess_i2v_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> CResult<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "wan-14b i2v image buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // Cover-fit scale + integer resize dims (≥ target so the center-crop is fully covered).
    let scale = (tw as f64 / iw as f64).max(th as f64 / ih as f64);
    let nw = ((iw as f64 * scale).round() as usize).max(tw);
    let nh = ((ih as f64 * scale).round() as usize).max(th);
    let resized = bilinear_rgb(&image.pixels, iw, ih, nw, nh);
    // Center-crop to (th, tw), normalize → CHW [-1,1].
    let (x1, y1) = ((nw - tw) / 2, (nh - th) / 2);
    let plane = th * tw;
    let mut chw = vec![0f32; 3 * plane];
    for yy in 0..th {
        for xx in 0..tw {
            let src = ((y1 + yy) * nw + (x1 + xx)) * 3;
            for c in 0..3 {
                chw[c * plane + yy * tw + xx] = 2.0 * (resized[src + c] / 255.0) - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, 1, th, tw), device)?)
}

/// Bilinear resize of an `iw×ih` RGB8 (HWC) buffer to `nw×nh`, returning HWC f32 pixel values in
/// `[0, 255]` (not normalized).
fn bilinear_rgb(px: &[u8], iw: usize, ih: usize, nw: usize, nh: usize) -> Vec<f32> {
    let mut out = vec![0f32; nw * nh * 3];
    let sx = iw as f64 / nw as f64;
    let sy = ih as f64 / nh as f64;
    for oy in 0..nh {
        // Pixel-center mapping (align_corners=False), clamped to the source extent.
        let fy = ((oy as f64 + 0.5) * sy - 0.5).clamp(0.0, (ih - 1) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(ih - 1);
        let wy = fy - y0 as f64;
        for ox in 0..nw {
            let fx = ((ox as f64 + 0.5) * sx - 0.5).clamp(0.0, (iw - 1) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(iw - 1);
            let wx = fx - x0 as f64;
            for c in 0..3 {
                let p = |y: usize, x: usize| px[(y * iw + x) * 3 + c] as f64;
                let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                out[(oy * nw + ox) * 3 + c] = (top * (1.0 - wy) + bot * wy) as f32;
            }
        }
    }
    out
}

/// A loaded Wan2.2 A14B generator (T2V or I2V). Heavy components (UMT5, the two 14B experts, the z16
/// VAE) are loaded lazily on the first `generate` and cached.
pub struct Wan14bGenerator {
    descriptor: ModelDescriptor,
    variant: Variant,
    root: PathBuf,
    device: Device,
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Components>>,
}

impl Wan14bGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        let mut guard = self
            .components
            .lock()
            .expect("wan-14b components cache mutex poisoned");
        if let Some(c) = guard.as_ref() {
            return Ok(c.clone());
        }
        let c = pipe.load_components()?;
        *guard = Some(c.clone());
        Ok(c)
    }
}

impl Generator for Wan14bGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.variant.id();
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE_14B)
            || !req.height.is_multiple_of(SIZE_MULTIPLE_14B)
        {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE_14B} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % 4 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: frames must satisfy frames % 4 == 1 (got {f})"
                )));
            }
        }
        if self.variant == Variant::I2v && i2v_reference(req).is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: image-to-video requires a Reference conditioning image"
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = Pipeline::load(
            &self.root,
            &self.device,
            self.variant,
            self.adapters.clone(),
        );
        let components = self.components(&pipe)?;
        let (frames, fps) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }
}

/// Shared descriptor surface for both A14B variants — CFG (per-expert guidance) + negative prompt,
/// UniPC/Euler samplers; H/W multiple of 16; **LoRA/LoKr supported** (sc-5167 — merged per-expert at
/// load; quant still deferred). `conditioning` differs per variant.
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    ModelDescriptor {
        id: variant.id(),
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: match variant {
                Variant::T2v => vec![],
                Variant::I2v => vec![ConditioningKind::Reference],
            },
            supports_lora: true,
            supports_lokr: true,
            // Curated `uni_pc` (sc-7296) → Wan's native UniPC; `euler` flow Euler. Legacy `unipc` alias.
            samplers: vec!["uni_pc", "euler", "unipc"],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Wan2.2 T2V-A14B dual-expert MoE text→video descriptor.
pub fn descriptor_t2v_14b() -> ModelDescriptor {
    descriptor_for(Variant::T2v)
}

/// Wan2.2 I2V-A14B dual-expert MoE channel-concat image→video descriptor.
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    descriptor_for(Variant::I2v)
}

fn load_variant(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (text_encoder/ transformer/ transformer_2/ vae/ \
                 tokenizer/), not a single .safetensors file"
            )));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} does not support on-the-fly Q4/Q8 quantization yet"
        )));
    }
    // I2V's conditioning image arrives per-request (`Conditioning::Reference`), not via `spec.control`;
    // the diffusers control/VACE overlays are not wired here.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} does not support control / VACE / IP-adapter overlays"
        )));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(Wan14bGenerator {
        descriptor: descriptor_for(variant),
        variant,
        root,
        device,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Wan2.2 T2V-A14B generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `Wan-AI/Wan2.2-T2V-A14B-Diffusers` snapshot (`text_encoder/`, `transformer/`,
/// `transformer_2/`, `vae/`, `tokenizer/`).
pub fn load_t2v_14b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::T2v)
}

/// Construct a lazy candle Wan2.2 I2V-A14B generator (channel-concat image→video). Same snapshot layout
/// as the T2V variant; the conditioning image arrives per-request as a `Conditioning::Reference`.
pub fn load_i2v_14b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::I2v)
}

candle_gen::register_generators! {
    descriptor_t2v_14b => load_t2v_14b,
    descriptor_i2v_14b => load_i2v_14b,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn registers_both_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for (id, conditioning_len) in [(MODEL_ID_T2V_14B, 0usize), (MODEL_ID_I2V_14B, 1)] {
            let g = registry::load(id, &spec).expect("14b model is registered");
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "wan");
            assert_eq!(g.descriptor().backend, "candle");
            assert_eq!(g.descriptor().modality, Modality::Video);
            assert!(!g.descriptor().capabilities.mac_only);
            assert_eq!(
                g.descriptor().capabilities.conditioning.len(),
                conditioning_len
            );
        }
    }

    #[test]
    fn descriptor_surface() {
        let t2v = descriptor_t2v_14b();
        assert!(t2v.capabilities.supports_guidance);
        assert!(t2v.capabilities.supports_negative_prompt);
        assert!(!t2v.capabilities.supports_true_cfg);
        assert!(t2v.capabilities.conditioning.is_empty());
        assert!(t2v.capabilities.samplers.contains(&"uni_pc")); // curated spelling (sc-7296)
        assert!(t2v.capabilities.samplers.contains(&"unipc")); // legacy alias retained

        let i2v = descriptor_i2v_14b();
        assert!(i2v.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn validate_enforces_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t2v = registry::load(MODEL_ID_T2V_14B, &spec).unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 256,
            height: 256,
            guidance: Some(4.0),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(t2v.validate(&ok).is_ok());
        // Legacy `unipc` spelling stays accepted (sc-7296 alias).
        assert!(t2v
            .validate(&GenerationRequest {
                sampler: Some("unipc".into()),
                ..ok.clone()
            })
            .is_ok());
        for bad in [
            // empty prompt
            GenerationRequest::default(),
            // frames not ≡ 1 (mod 4)
            GenerationRequest {
                prompt: "x".into(),
                frames: Some(16),
                ..Default::default()
            },
            // size not a multiple of 16
            GenerationRequest {
                prompt: "x".into(),
                width: 300,
                ..Default::default()
            },
            // unadvertised sampler
            GenerationRequest {
                prompt: "x".into(),
                sampler: Some("dpmpp2m".into()),
                ..Default::default()
            },
        ] {
            assert!(t2v.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // I2V rejects a request with no Reference image.
        let i2v = registry::load(MODEL_ID_I2V_14B, &spec).unwrap();
        assert!(i2v.validate(&ok).is_err(), "i2v needs a reference image");
    }

    #[test]
    fn load_accepts_adapters_rejects_quant() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // LoRA/LoKr are now supported (sc-5167) — load is lazy, so attaching adapters resolves OK
        // (the merge happens at the first `generate`).
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load_t2v_14b(&lora).is_ok());
        // Quant is still deferred.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load_i2v_14b(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
