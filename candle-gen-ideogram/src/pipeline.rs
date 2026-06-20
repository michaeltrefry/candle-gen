//! Ideogram 4 text-to-image **and edit** pipeline: Qwen3-VL text encode → flow-matching denoise →
//! latent de-normalize + (ph,pw,c) unpatchify + VAE decode. Port of `mlx-gen-ideogram`'s
//! `Ideogram4Pipeline` (T2I + the sc-6303/6330 img2img/Remix + mask inpaint edit, sc-6598).
//!
//! **Edit (img2img / mask inpaint).** With an [`EditInit`] (a source `Reference` + optional `Mask`),
//! the denoise starts from the VAE-encoded source latent noised to a strength-derived step instead of
//! pure noise (img2img / Remix); an optional latent-grid mask additionally pins the keep region
//! (mask 0) to the source re-noised to each step's σ while regenerating the white region (mask 1) —
//! masked img2img inpaint on the same flow-match loop. With no [`EditInit`] the path is identical to
//! the original text-to-image render.
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
use candle_gen::gen_core::imageops::{resize_lanczos_u8, resize_nearest_u8};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{self, Conditioning, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_flux2::vae::Flux2Vae;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::config::{
    Ideogram4DitConfig, Ideogram4TextEncoderConfig, DEFAULT_GUIDANCE, DEFAULT_IMG2IMG_STRENGTH,
    DEFAULT_INPAINT_STRENGTH, DEFAULT_STEPS, DEFAULT_TURBO_STEPS, EXTRACTED_LAYERS,
    MAX_TEXT_TOKENS, PAD_TOKEN_ID, TURBO_LORA_FILE, TURBO_LORA_SCALE,
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
    // With the encoder (img2img/edit needs `vae.encode`); the encoder adds only ~the decoder's worth
    // of weights on top of the multi-GB DiTs, so it is always loaded — a Generator serves both T2I and
    // edit requests and does not know which at load time.
    let vae = Flux2Vae::new_with_encoder(component_vb(root, "vae", ENC_DTYPE, device)?)?;

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
    // With the encoder (img2img/edit needs `vae.encode`); the encoder adds only ~the decoder's worth
    // of weights on top of the multi-GB DiTs, so it is always loaded — a Generator serves both T2I and
    // edit requests and does not know which at load time.
    let vae = Flux2Vae::new_with_encoder(component_vb(root, "vae", ENC_DTYPE, device)?)?;

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

    /// VAE-encode a source image into the bn-normalized packed latent `[1, num_img, 128]` the denoise
    /// operates on — the exact inverse of [`decode`]'s de-normalize + (ph,pw,c) unpatchify: resize →
    /// `vae.encode` (posterior mean) → 2×2 patchify (Ideogram's `(ph,pw,c)` c-innermost order) →
    /// bn-normalize `(x − mean)/std`. Seed-independent; encode once per request.
    fn encode_init_latents(
        &self,
        image: &Image,
        height: u32,
        width: u32,
        device: &Device,
    ) -> CResult<Tensor> {
        let grid_h = (height / PATCH_AE) as usize;
        let grid_w = (width / PATCH_AE) as usize;
        let pre = preprocess_source_image(image, width, height, device)?; // [1, 3, H, W]
        let enc = self.vae.encode(&pre)?; // [1, 32, H/8, W/8] = [1, 32, gh·2, gw·2]
                                          // Patchify to packed [1, L, 128] (ph,pw,c) — inverse of decode's unpatchify permute.
        let packed = enc
            .reshape((1, 32, grid_h, 2, grid_w, 2))? // [B, c, gh, ph, gw, pw]
            .permute((0, 2, 4, 3, 5, 1))? // [B, gh, gw, ph, pw, c]
            .contiguous()?
            .reshape((1, grid_h * grid_w, 128))?;
        let (bn_std, bn_mean) = self.vae.bn_stats();
        let bn_std = bn_std.reshape((1, 1, 128))?;
        let bn_mean = bn_mean.reshape((1, 1, 128))?;
        Ok(packed.broadcast_sub(&bn_mean)?.broadcast_div(&bn_std)?)
    }

    /// Prepare the per-request [`EditInit`] (img2img / inpaint): VAE-encode the source once and build
    /// the optional latent-grid mask. Reused across the per-seed count loop (seed-independent).
    fn prepare_edit(
        &self,
        source: &Image,
        mask: Option<&Image>,
        strength: f32,
        height: u32,
        width: u32,
        device: &Device,
    ) -> CResult<EditInit> {
        let z0 = self.encode_init_latents(source, height, width, device)?;
        let mask = match mask {
            Some(m) => Some(preprocess_mask_packed(m, width, height, device)?),
            None => None,
        };
        Ok(EditInit { z0, mask, strength })
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

    // Edit (img2img / inpaint): resolve a source `Reference` (+ optional `Mask`) and VAE-encode the
    // source once (seed-independent). `None` → the text-to-image path (identical to before).
    let edit = match resolve_edit(req)? {
        Some((source, mask, strength)) => {
            Some(comps.prepare_edit(source, mask, strength, req.height, req.width, device)?)
        }
        None => None,
    };

    let mut images = Vec::with_capacity(req.count as usize);
    for index in 0..req.count {
        let seed = base_seed.wrapping_add(index as u64);
        let z = denoise(
            comps,
            &ids,
            req,
            steps,
            guidance,
            seed,
            edit.as_ref(),
            device,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        images.push(decode(comps, &z, req.width, req.height)?);
    }
    Ok(images)
}

/// Resolve the optional edit conditioning: a single img2img/inpaint source [`Conditioning::Reference`]
/// plus an optional [`Conditioning::Mask`]. Returns `(source, mask, strength)`; `None` for pure
/// text-to-image. A per-reference strength wins over `req.strength`, else the img2img/inpaint default.
/// More than one `Reference`/`Mask`, or a `Mask` without a `Reference`, is an error.
fn resolve_edit(req: &GenerationRequest) -> CResult<Option<(&Image, Option<&Image>, f32)>> {
    let mut source: Option<(&Image, Option<f32>)> = None;
    let mut mask: Option<&Image> = None;
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, strength } => {
                if source.is_some() {
                    return Err(CandleError::Msg(
                        "ideogram: only one reference (source) image is supported for edit".into(),
                    ));
                }
                source = Some((image, strength.or(req.strength)));
            }
            Conditioning::Mask { image } => {
                if mask.is_some() {
                    return Err(CandleError::Msg(
                        "ideogram: only one inpaint mask is supported".into(),
                    ));
                }
                mask = Some(image);
            }
            // Other conditioning kinds are rejected by the capability floor in `validate`.
            _ => {}
        }
    }
    match source {
        Some((image, strength)) => {
            let default = if mask.is_some() {
                DEFAULT_INPAINT_STRENGTH
            } else {
                DEFAULT_IMG2IMG_STRENGTH
            };
            Ok(Some((image, mask, strength.unwrap_or(default))))
        }
        None if mask.is_some() => Err(CandleError::Msg(
            "ideogram: an inpaint mask requires a reference (source) image".into(),
        )),
        None => Ok(None),
    }
}

/// One flow-matching denoise → packed image latent `[1, num_img, 128]` (f32). With `edit = Some`
/// (img2img / inpaint) the denoise starts from the source latent noised to a strength-derived step
/// and (with a mask) pins the keep region per step; `edit = None` is the original text-to-image path.
#[allow(clippy::too_many_arguments)]
fn denoise(
    comps: &Components,
    ids: &[i32],
    req: &GenerationRequest,
    steps: usize,
    guidance: f32,
    seed: u64,
    edit: Option<&EditInit>,
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

    // Edit: run only `num_run = floor(steps·strength)` of the reversed loop (skip the noisiest leading
    // steps) and start from the source noised to σ = schedule.eval(si[num_run]). T2I runs the full
    // range from pure noise. (Ideogram's schedule is inverted — larger σ = cleaner, so a larger
    // num_run/strength → a smaller start σ → more change.) `init_time_step` floors strength to ≥1 step.
    let num_run = match edit {
        Some(e) => init_time_step(steps, e.strength),
        None => steps,
    };

    // ── Init: always draw the noise (identical RNG stream); blend with the source for an edit ──
    let noise = create_noise(seed, num_img, ch, device)?; // [1, num_img, ch] f32
    let mut z = match edit {
        Some(e) => add_noise_by_interpolation(&e.z0, &noise, schedule.eval(si[num_run]) as f32)?,
        None => noise.clone(),
    };
    let text_z_padding = Tensor::zeros((1, num_text, ch), DType::F32, device)?;

    for i in (0..num_run).rev() {
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
        // Inpaint: pin the keep region (mask 0) to the source re-noised to this step's σ (= s_val, the
        // post-step time) and regenerate the white region (mask 1). At the final step s≈0 the keep
        // region is the clean source. Mask is `[1, num_img, 1]`, broadcast over the channel axis;
        // draws no RNG, so an all-white mask reduces to plain img2img.
        if let Some(e) = edit {
            if let Some(mask) = &e.mask {
                let init_noised = add_noise_by_interpolation(&e.z0, &noise, s_val)?;
                let keep = mask.affine(-1.0, 1.0)?; // 1 − mask
                z = (z.broadcast_mul(mask)? + init_noised.broadcast_mul(&keep)?)?;
            }
        }
        on_progress(Progress::Step {
            current: (num_run - i) as u32,
            total: num_run as u32,
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

/// Edit (img2img / mask inpaint) conditioning prepared **once per request** (seed-independent) by
/// [`Components::prepare_edit`]: the bn-normalized packed source latent `[1, num_img, 128]`, an
/// optional latent-grid inpaint mask `[1, num_img, 1]` (1 = repaint, 0 = keep), and the strength.
pub struct EditInit {
    /// bn-normalized packed source latent `[1, num_img, 128]` (same space as the running `z`).
    pub z0: Tensor,
    /// Latent-grid inpaint mask `[1, num_img, 1]` (1.0 = repaint/white, 0.0 = keep/black). `None` for
    /// plain img2img (regenerate everywhere from the noised source).
    pub mask: Option<Tensor>,
    /// img2img strength in `(0, 1]` — fraction of the denoise executed from the noised source.
    pub strength: f32,
}

/// img2img start step (the flux2/fork `init_time_step`): `max(1, floor(num_steps·strength))` for a
/// positive strength clamped to `[0,1]`, else `0`. The denoise executes the lowest `num_run` steps
/// over the source noised to `schedule.eval(si[num_run])`.
fn init_time_step(num_steps: usize, strength: f32) -> usize {
    if strength > 0.0 {
        let s = strength.clamp(0.0, 1.0);
        // `int(num_steps * strength)` truncates toward zero == floor for s >= 0.
        ((num_steps as f32 * s) as usize).max(1)
    } else {
        0
    }
}

/// Flow-matching interpolation `z = σ·clean + (1−σ)·noise`. Ideogram's [`LogitNormalSchedule`] is
/// inverted from the usual flow-match σ (`eval(0) ≈ clean`, `eval(1) ≈ noise`), so a larger σ weights
/// the clean source more — the mirror of the fork's `add_noise_by_interpolation`.
fn add_noise_by_interpolation(clean: &Tensor, noise: &Tensor, sigma: f32) -> CResult<Tensor> {
    Ok(((clean * sigma as f64)? + (noise * (1.0 - sigma) as f64)?)?)
}

/// Preprocess a source image onto the model's input grid: Lanczos-resize to `width×height`, normalize
/// to `[-1,1]`, NCHW `[1,3,H,W]` f32 (candle VAE layout). Mirrors the MLX `preprocess_source_image`.
fn preprocess_source_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> CResult<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "ideogram edit: source pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let hwc = Tensor::from_vec(norm, (th, tw, 3), device)?; // [H, W, 3]
    Ok(hwc
        .permute((2, 0, 1))?
        .contiguous()?
        .reshape((1, 3, th, tw))?)
}

/// Build the latent-grid inpaint mask `[1, num_img, 1]` (f32; 1.0 = repaint/white, 0.0 = keep/black)
/// from a mask image: luma → nearest `patch·ae = 16×` downsample (top-left of each block, torch
/// `nearest`'s `floor(dst·scale)`) → binarize at 0.5, row-major to match the image-token order
/// (`j = h·grid_w + w`). Mirrors the MLX `preprocess_mask_packed` (downsample factor 16, not 8).
fn preprocess_mask_packed(
    mask: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> CResult<Tensor> {
    let (w, h) = (width as usize, height as usize);
    let patch = PATCH_AE as usize; // 16
                                   // Nearest (not bicubic): a mask must not gain interpolated grays that flip the 0.5 binarize.
    let luma: Vec<u8> = if (mask.width as usize, mask.height as usize) == (w, h) {
        rgb_to_luma(&mask.pixels)
    } else {
        let resized = resize_nearest_u8(
            &mask.pixels,
            mask.height as usize,
            mask.width as usize,
            h,
            w,
        );
        let u8s: Vec<u8> = resized
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        rgb_to_luma(&u8s)
    };
    let (gh, gw) = (h / patch, w / patch);
    let mut packed = Vec::with_capacity(gh * gw);
    for ly in 0..gh {
        for lx in 0..gw {
            let v = luma[(ly * patch) * w + (lx * patch)]; // top-left of the block
            packed.push(if v as f32 / 255.0 >= 0.5 { 1.0f32 } else { 0.0 });
        }
    }
    Ok(Tensor::from_vec(packed, (1, gh * gw, 1), device)?)
}

/// PIL "L" grayscale luma: `round(R·299/1000 + G·587/1000 + B·114/1000)` per RGB pixel.
fn rgb_to_luma(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| {
            let l = (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000;
            l.min(255) as u8
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, pixels: Vec<u8>) -> Image {
        Image {
            width: w,
            height: h,
            pixels,
        }
    }

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> Image {
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            px.extend_from_slice(&rgb);
        }
        img(w, h, px)
    }

    #[test]
    fn init_time_step_floors_and_clamps() {
        assert_eq!(init_time_step(48, 0.0), 0); // no strength → no edit steps
        assert_eq!(init_time_step(48, 1.0), 48); // full strength → full range
        assert_eq!(init_time_step(48, 0.5), 24); // floor(48·0.5)
        assert_eq!(init_time_step(48, 0.6), 28); // floor(28.8)
        assert_eq!(init_time_step(48, 0.001), 1); // tiny positive floors to ≥1
        assert_eq!(init_time_step(8, 2.0), 8); // strength clamps to 1.0
    }

    #[test]
    fn add_noise_by_interpolation_blends() {
        let dev = Device::Cpu;
        let clean = Tensor::from_vec(vec![2.0f32, 2.0], (1, 2, 1), &dev).unwrap();
        let noise = Tensor::from_vec(vec![10.0f32, 10.0], (1, 2, 1), &dev).unwrap();
        let at = |s: f32| {
            add_noise_by_interpolation(&clean, &noise, s)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(at(1.0), vec![2.0, 2.0]); // σ=1 → clean
        assert_eq!(at(0.0), vec![10.0, 10.0]); // σ=0 → noise
        assert_eq!(at(0.5), vec![6.0, 6.0]); // halfway
    }

    #[test]
    fn rgb_to_luma_matches_pil() {
        assert_eq!(rgb_to_luma(&[255, 255, 255]), vec![255]); // white
        assert_eq!(rgb_to_luma(&[0, 0, 0]), vec![0]); // black
                                                      // round(255·0.587) = round(149.685) = 150 for pure green.
        assert_eq!(rgb_to_luma(&[0, 255, 0]), vec![150]);
    }

    #[test]
    fn preprocess_mask_packed_binarizes_and_downsamples() {
        // 32×16, patch 16 → grid 2×1 (gw=2, gh=1) → 2 tokens. Left 16 cols white, right 16 black.
        let (w, h) = (32u32, 16u32);
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..h {
            for x in 0..w {
                let v = if x < 16 { 255 } else { 0 };
                px.extend_from_slice(&[v, v, v]);
            }
        }
        let m = preprocess_mask_packed(&img(w, h, px), w, h, &Device::Cpu).unwrap();
        assert_eq!(m.dims(), &[1, 2, 1]); // [1, num_img, 1]
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 0.0]); // left block repaint, right block keep
    }

    fn req_with(conditioning: Vec<Conditioning>, strength: Option<f32>) -> GenerationRequest {
        GenerationRequest {
            prompt: "a fox".into(),
            width: 512,
            height: 512,
            conditioning,
            strength,
            ..Default::default()
        }
    }

    #[test]
    fn resolve_edit_defaults_and_pairing() {
        // No conditioning → no edit.
        assert!(resolve_edit(&req_with(vec![], None)).unwrap().is_none());

        // Reference only → img2img default strength.
        let r = req_with(
            vec![Conditioning::Reference {
                image: solid(8, 8, [10, 20, 30]),
                strength: None,
            }],
            None,
        );
        let (_, mask, strength) = resolve_edit(&r).unwrap().expect("edit");
        assert!(mask.is_none());
        assert_eq!(strength, DEFAULT_IMG2IMG_STRENGTH);

        // Reference + Mask → inpaint default strength.
        let r = req_with(
            vec![
                Conditioning::Reference {
                    image: solid(8, 8, [10, 20, 30]),
                    strength: None,
                },
                Conditioning::Mask {
                    image: solid(8, 8, [255, 255, 255]),
                },
            ],
            None,
        );
        let (_, mask, strength) = resolve_edit(&r).unwrap().expect("edit");
        assert!(mask.is_some());
        assert_eq!(strength, DEFAULT_INPAINT_STRENGTH);

        // Per-reference strength wins over the default and over req.strength.
        let r = req_with(
            vec![Conditioning::Reference {
                image: solid(8, 8, [10, 20, 30]),
                strength: Some(0.42),
            }],
            Some(0.9),
        );
        assert_eq!(resolve_edit(&r).unwrap().unwrap().2, 0.42);
        // req.strength is used when the reference carries none.
        let r = req_with(
            vec![Conditioning::Reference {
                image: solid(8, 8, [10, 20, 30]),
                strength: None,
            }],
            Some(0.33),
        );
        assert_eq!(resolve_edit(&r).unwrap().unwrap().2, 0.33);

        // A second Reference is an error.
        let r = req_with(
            vec![
                Conditioning::Reference {
                    image: solid(8, 8, [1, 2, 3]),
                    strength: None,
                },
                Conditioning::Reference {
                    image: solid(8, 8, [4, 5, 6]),
                    strength: None,
                },
            ],
            None,
        );
        assert!(resolve_edit(&r).is_err());

        // A Mask without a Reference is an error.
        let r = req_with(
            vec![Conditioning::Mask {
                image: solid(8, 8, [255, 255, 255]),
            }],
            None,
        );
        assert!(resolve_edit(&r).is_err());
    }
}
