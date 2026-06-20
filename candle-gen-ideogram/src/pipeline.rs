//! Ideogram 4 text-to-image pipeline: Qwen3-VL text encode → flow-matching denoise → latent
//! de-normalize + (ph,pw,c) unpatchify + VAE decode. Port of `mlx-gen-ideogram`'s
//! `Ideogram4Pipeline` (T2I path; img2img/edit is sc-6598).
//!
//! Two denoise modes share the loop, selected by whether the **unconditional** DiT is present:
//! * **Quality (asymmetric CFG, default)** — the conditional DiT runs over the full `[text ; image]`
//!   sequence; the unconditional DiT runs over the **image-only** slice with zeroed conditioning.
//!   Per step `v = g·pos_v + (1−g)·neg_v` (guidance drops to 3.0 for the final 3 polish steps).
//! * **Turbo (CFG-free single DiT)** — `uncond` is `None` and the conditional DiT carries the ostris
//!   TurboTime LoRA (merged at load via [`load_components_turbo`]); per step `v = pos_v`, few-step.
//!
//! The VAE is reused from `candle-gen-flux2` (`AutoencoderKLFlux2`), but Ideogram packs the 128
//! transformer channels as `(ph,pw,c)` (c innermost) vs FLUX.2's `(c,ph,pw)`, so the bn-denorm +
//! unpatchify are done here (via [`Flux2Vae::bn_stats`] / [`Flux2Vae::decode_latent`]) rather than
//! flux2's `decode_packed`.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{self, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_flux2::vae::Flux2Vae;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::config::{
    Ideogram4DitConfig, Ideogram4TextEncoderConfig, DEFAULT_GUIDANCE, DEFAULT_STEPS,
    DEFAULT_TURBO_STEPS, EXTRACTED_LAYERS, MAX_TEXT_TOKENS, PAD_TOKEN_ID, TURBO_LORA_FILE,
    TURBO_LORA_SCALE,
};
use crate::scheduler::{make_step_intervals, preset_mu_std, LogitNormalSchedule};
use crate::text_encoder::Ideogram4TextEncoder;
use crate::transformer::Ideogram4Transformer;

/// The conditional DiT is the bottleneck — bf16. Encoder + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;

/// patch (2) × VAE scale (8) — height/width must be a multiple of this (=16).
const PATCH_AE: u32 = 16;
const IMAGE_POSITION_OFFSET: i64 = 65536;
const LLM_TOKEN_INDICATOR: i64 = 3;
const OUTPUT_IMAGE_INDICATOR: i64 = 2;
/// The final `POLISH_STEPS` low-noise steps use the reduced `POLISH_GUIDANCE` (the reference
/// `DEFAULT_GUIDANCE_SCHEDULE`); a constant high guidance over-cooks the detail steps.
const POLISH_STEPS: usize = 3;
const POLISH_GUIDANCE: f32 = 3.0;

/// The loaded Ideogram 4 components.
pub struct Components {
    cond: Ideogram4Transformer,
    /// The unconditional DiT (asymmetric-CFG negative branch). `None` for turbo.
    uncond: Option<Ideogram4Transformer>,
    te: Ideogram4TextEncoder,
    vae: Flux2Vae,
    tokenizer_path: PathBuf,
    dit: Ideogram4DitConfig,
}

fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "ideogram snapshot missing the {sub}/ dir (at {})",
            root.display()
        )));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CandleError::Msg(format!("ideogram: read {sub}/: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "ideogram: no .safetensors in {sub}/ (at {})",
            dir.display()
        )));
    }
    // SAFETY: read-only mmap of weight files; standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? })
}

/// Load all components from a candle-readable (bf16) Ideogram 4 snapshot dir (`transformer/`,
/// `unconditional_transformer/`, `text_encoder/`, `vae/`, `tokenizer/`) — the quality (asymmetric
/// CFG) mode.
pub fn load_components(root: &Path, device: &Device) -> CResult<Components> {
    let dit = Ideogram4DitConfig::v4();
    let te_cfg = Ideogram4TextEncoderConfig::qwen3_vl_8b();

    let cond = crate::loader::Weights::from_dir(&root.join("transformer"), device, DIT_DTYPE)?;
    let cond = Ideogram4Transformer::load(&cond, &dit)?;

    let uncond = crate::loader::Weights::from_dir(
        &root.join("unconditional_transformer"),
        device,
        DIT_DTYPE,
    )?;
    let uncond = Some(Ideogram4Transformer::load(&uncond, &dit)?);

    let te = Ideogram4TextEncoder::new(
        &te_cfg,
        &EXTRACTED_LAYERS,
        MAX_TEXT_TOKENS,
        component_vb(root, "text_encoder", ENC_DTYPE, device)?,
    )?;
    let vae = Flux2Vae::new(component_vb(root, "vae", ENC_DTYPE, device)?)?;

    Ok(Components {
        cond,
        uncond,
        te,
        vae,
        tokenizer_path: root.join("tokenizer/tokenizer.json"),
        dit,
    })
}

/// Load the **turbo** components: the conditional DiT with the bundled TurboTime LoRA merged in (no
/// unconditional DiT). CFG-free single-DiT few-step path.
pub fn load_components_turbo(root: &Path, device: &Device) -> CResult<Components> {
    let dit = Ideogram4DitConfig::v4();
    let te_cfg = Ideogram4TextEncoderConfig::qwen3_vl_8b();

    let mut cond_w =
        crate::loader::Weights::from_dir(&root.join("transformer"), device, DIT_DTYPE)?;
    let n = crate::adapters::merge_turbo_lora(
        &mut cond_w,
        &root.join(TURBO_LORA_FILE),
        TURBO_LORA_SCALE,
    )?;
    eprintln!("ideogram turbo: merged {n} TurboTime LoRA target module(s)");
    let cond = Ideogram4Transformer::load(&cond_w, &dit)?;

    let te = Ideogram4TextEncoder::new(
        &te_cfg,
        &EXTRACTED_LAYERS,
        MAX_TEXT_TOKENS,
        component_vb(root, "text_encoder", ENC_DTYPE, device)?,
    )?;
    let vae = Flux2Vae::new(component_vb(root, "vae", ENC_DTYPE, device)?)?;

    Ok(Components {
        cond,
        uncond: None,
        te,
        vae,
        tokenizer_path: root.join("tokenizer/tokenizer.json"),
        dit,
    })
}

impl Components {
    /// Tokenize a prompt to `input_ids` exactly as the reference `_tokenize`: the Qwen3-VL single-user
    /// chat template, `add_special_tokens=false`. Rejects > `MAX_TEXT_TOKENS`.
    fn tokenize(&self, prompt: &str) -> CResult<Vec<i32>> {
        let tok = TextTokenizer::from_file(
            &self.tokenizer_path,
            TokenizerConfig {
                max_length: MAX_TEXT_TOKENS,
                pad_token_id: PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenInstruct,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("ideogram: load tokenizer: {e}")))?;
        let ids = tok
            .encode_chat_ids(prompt, false)
            .map_err(|e| CandleError::Msg(format!("ideogram: tokenize: {e}")))?;
        if ids.len() > MAX_TEXT_TOKENS {
            return Err(CandleError::Msg(format!(
                "ideogram: prompt has {} tokens, exceeds max_text_tokens={MAX_TEXT_TOKENS}",
                ids.len()
            )));
        }
        Ok(ids)
    }
}

/// Render `req.count` images for `req`.
pub fn render(
    comps: &Components,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> CResult<Vec<Image>> {
    // Turbo (no unconditional DiT) defaults to the few-step count; quality to 48.
    let default_steps = if comps.uncond.is_none() {
        DEFAULT_TURBO_STEPS
    } else {
        DEFAULT_STEPS
    };
    let steps = req
        .steps
        .map(|s| s as usize)
        .unwrap_or(default_steps as usize);
    let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

    let ids = comps.tokenize(&req.prompt)?;
    let num_text = ids.len();
    if num_text == 0 {
        return Err(CandleError::Msg(
            "ideogram: empty prompt token sequence".into(),
        ));
    }

    let mut images = Vec::with_capacity(req.count as usize);
    for index in 0..req.count {
        let seed = base_seed.wrapping_add(index as u64);
        let z = denoise(comps, &ids, req, steps, guidance, seed, device, on_progress)?;
        on_progress(Progress::Decoding);
        images.push(decode(comps, &z, req.width, req.height)?);
    }
    Ok(images)
}

/// One flow-matching denoise → packed image latent `[1, num_img, 128]` (f32).
#[allow(clippy::too_many_arguments)]
fn denoise(
    comps: &Components,
    ids: &[i32],
    req: &GenerationRequest,
    steps: usize,
    guidance: f32,
    seed: u64,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> CResult<Tensor> {
    let (width, height) = (req.width, req.height);
    let grid_h = (height / PATCH_AE) as usize;
    let grid_w = (width / PATCH_AE) as usize;
    let num_img = grid_h * grid_w;
    let num_text = ids.len();
    let seq = num_text + num_img;
    let llm_dim = comps.dit.llm_features_dim;
    let ch = comps.dit.in_channels;

    // ── Text encode (single prompt, no padding → positions 0..num_text) ──
    let ids_u32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let id_tensor = Tensor::from_vec(ids_u32, (1, num_text), device)?;
    let attn = Tensor::ones((1, num_text), DType::U32, device)?;
    let te_out = comps.te.prompt_embeds(&id_tensor, &attn)?; // [1, num_text, llm_dim] f32
    let llm_zeros = Tensor::zeros((1, num_img, llm_dim), ENC_DTYPE, device)?;
    let llm_features = Tensor::cat(&[&te_out, &llm_zeros], 1)?; // [1, seq, llm_dim]

    // ── Packed positions / segments / role indicators (host-built) ──
    let pack = Packing::build(num_text, grid_h, grid_w);
    let position_ids = Tensor::from_vec(pack.position_ids, (1, seq, 3), device)?;
    let segment_ids = Tensor::from_vec(pack.segment_ids, (1, seq), device)?;
    let indicator = Tensor::from_vec(pack.indicator, (1, seq), device)?;
    let neg = match &comps.uncond {
        Some(uncond) => Some((
            uncond,
            Tensor::from_vec(pack.neg_position_ids, (1, num_img, 3), device)?,
            Tensor::from_vec(pack.neg_segment_ids, (1, num_img), device)?,
            Tensor::from_vec(pack.neg_indicator, (1, num_img), device)?,
            Tensor::zeros((1, num_img, llm_dim), ENC_DTYPE, device)?,
        )),
        None => None,
    };

    // ── Flow-matching schedule (mu/std from the V4 preset for this step count) ──
    let (mu_eff, std_eff) = preset_mu_std(steps);
    let schedule = LogitNormalSchedule::for_resolution(height, width, mu_eff, std_eff);
    let si = make_step_intervals(steps);

    // ── Init from pure noise; image-token velocity slice; text padding for the cond sequence ──
    let mut z = create_noise(seed, num_img, ch, device)?; // [1, num_img, ch] f32
    let text_z_padding = Tensor::zeros((1, num_text, ch), DType::F32, device)?;

    for i in (0..steps).rev() {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let t_val = schedule.eval(si[i + 1]) as f32;
        let s_val = schedule.eval(si[i]) as f32;
        let t = Tensor::from_vec(vec![t_val], 1, device)?;

        let pos_z = Tensor::cat(&[&text_z_padding, &z], 1)?; // [1, seq, ch]
        let pos_out = comps.cond.forward(
            &llm_features,
            &pos_z,
            &t,
            &position_ids,
            &segment_ids,
            &indicator,
        )?;
        let pos_v = pos_out.narrow(1, num_text, num_img)?; // image-token velocities [1, num_img, ch]

        let v = match &neg {
            Some((uncond, neg_pos, neg_seg, neg_ind, neg_llm)) => {
                let neg_v = uncond.forward(neg_llm, &z, &t, neg_pos, neg_seg, neg_ind)?;
                // Per-step asymmetric CFG: the loop runs i = steps-1 → 0, so the final POLISH_STEPS
                // are i ∈ {0,1,2}.
                let gw = if i < POLISH_STEPS {
                    POLISH_GUIDANCE
                } else {
                    guidance
                };
                ((pos_v * gw as f64)? + (neg_v * (1.0 - gw) as f64)?)?
            }
            // Turbo: CFG-free single DiT (TurboTime LoRA distilled the guided velocity into `cond`).
            None => pos_v,
        };
        z = (&z + (v * (s_val - t_val) as f64)?)?;
        on_progress(Progress::Step {
            current: (steps - i) as u32,
            total: steps as u32,
        });
    }
    Ok(z)
}

/// De-normalize (bn) → (ph,pw,c) unpatchify → VAE decode → RGB image.
fn decode(comps: &Components, z: &Tensor, width: u32, height: u32) -> CResult<Image> {
    let grid_h = (height / PATCH_AE) as usize;
    let grid_w = (width / PATCH_AE) as usize;

    // bn de-normalize in the packed [1, L, 128] space (Ideogram's (ph,pw,c) channel order).
    let (bn_std, bn_mean) = comps.vae.bn_stats();
    let bn_std = bn_std.reshape((1, 1, 128))?;
    let bn_mean = bn_mean.reshape((1, 1, 128))?;
    let denorm = z.broadcast_mul(&bn_std)?.broadcast_add(&bn_mean)?; // [1, L, 128]

    // Unpatchify (ph,pw,c) → NCHW [1, 32, gh·2, gw·2]: split 128 = (ph=2, pw=2, c=32),
    // bring c to the channel axis and interleave (gh,ph)/(gw,pw).
    let latent = denorm
        .reshape((1, grid_h, grid_w, 2, 2, 32))?
        .permute((0, 5, 1, 3, 2, 4))? // [1, c, gh, ph, gw, pw]
        .contiguous()?
        .reshape((1, 32, grid_h * 2, grid_w * 2))?;

    let decoded = comps.vae.decode_latent(&latent)?; // [1, 3, H, W] f32 ~[-1,1]
    to_image(&decoded)
}

/// `[1, 3, H, W]` f32 ~[-1,1] → RGB8 [`Image`].
fn to_image(decoded: &Tensor) -> CResult<Image> {
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "ideogram: expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Deterministic CPU-seeded standard-normal noise `[1, num_img, ch]` (f32). (RNG differs from MLX's,
/// so renders are deterministic but not bit-identical to the MLX reference — functional parity.)
fn create_noise(seed: u64, num_img: usize, ch: usize, device: &Device) -> CResult<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = num_img * ch;
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Ok(Tensor::from_vec(data, (1, num_img, ch), device)?)
}

/// Host-built packed sequence metadata: text tokens (`LLM`) then image tokens (`IMAGE`).
struct Packing {
    position_ids: Vec<i64>,
    segment_ids: Vec<i64>,
    indicator: Vec<i64>,
    neg_position_ids: Vec<i64>,
    neg_segment_ids: Vec<i64>,
    neg_indicator: Vec<i64>,
}

impl Packing {
    fn build(num_text: usize, grid_h: usize, grid_w: usize) -> Self {
        let num_img = grid_h * grid_w;
        let mut position_ids = Vec::with_capacity((num_text + num_img) * 3);
        let mut indicator = Vec::with_capacity(num_text + num_img);
        for i in 0..num_text {
            let i = i as i64;
            position_ids.extend_from_slice(&[i, i, i]);
            indicator.push(LLM_TOKEN_INDICATOR);
        }
        let mut neg_position_ids = Vec::with_capacity(num_img * 3);
        for j in 0..num_img {
            let (h, w) = ((j / grid_w) as i64, (j % grid_w) as i64);
            // Reference `_prepare_ids`: image positions are `[t,h,w] + OFFSET` on ALL THREE axes
            // (t_idx is 0, so t = OFFSET) — keeping image positions disjoint from text (0..num_text);
            // leaving t=0 collides with the text t-axis and corrupts the text→image MRoPE.
            let p = [
                IMAGE_POSITION_OFFSET,
                h + IMAGE_POSITION_OFFSET,
                w + IMAGE_POSITION_OFFSET,
            ];
            position_ids.extend_from_slice(&p);
            neg_position_ids.extend_from_slice(&p);
            indicator.push(OUTPUT_IMAGE_INDICATOR);
        }
        let seq = num_text + num_img;
        Self {
            position_ids,
            segment_ids: vec![1; seq],
            indicator,
            neg_position_ids,
            neg_segment_ids: vec![1; num_img],
            neg_indicator: vec![OUTPUT_IMAGE_INDICATOR; num_img],
        }
    }
}
