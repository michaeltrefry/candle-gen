//! FLUX.2-klein **reference-image edit** provider (sc-5487, epic 5480) — Kontext-style edit / identity
//! conditioning on FLUX.2-klein-9B off-Mac (Windows/CUDA), the candle sibling of the `mlx-gen-flux2`
//! edit variant (`flux2_klein_9b_edit`) and the **provider half** that unblocks the worker wiring.
//! FLUX.2-klein has no torch path (it is diffusers/MLX-only), so this lane retires the worker's
//! `edit_image` → torch deferral for `flux2_klein_9b`.
//!
//! **How it conditions (no transformer change):** each reference image is VAE-encoded into the packed,
//! bn-normalized transformer latent ([`Flux2Vae::encode_packed`]) and packed to tokens, then
//! concatenated AFTER the noised target tokens on the sequence axis — the joint image stream
//! `[target, ref0, ref1, …]`. The reference grid ids are offset at `t = 10 + 10·i` (the mlx fork's
//! per-reference temporal coordinate) so the 4-axis RoPE keeps the references positionally distinct
//! from the `t = 0` target grid. The existing [`Flux2Transformer::forward`] already accepts arbitrary
//! `img_ids`, so it runs the full joint sequence unchanged; the provider keeps the leading `target_seq`
//! velocity tokens and steps only the target. The reference tokens are clean and constant across the
//! denoise (re-concatenated each step, never noised).
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — mirroring the SDXL edit /
//! IP-Adapter / InstantID / PuLID providers. Determinism is the candle-lane contract (sc-3673): the
//! seeded CPU init noise reuses [`pipeline::create_noise`]. Distilled klein runs CFG-free (guidance
//! 1.0); guidance > 1 adds a classifier-free negative pass (the same convention as txt2img). No
//! `strength`: FLUX.2 edit conditions via reference token concat (a full denoise from noise), not an
//! img2img noise blend.

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};

use crate::config::{Flux2Variant, DEFAULT_GUIDANCE, DEFAULT_STEPS, SIZE_MULTIPLE};
use crate::text_encoder::Qwen3TextEncoder;
use crate::transformer::Flux2Transformer;
use crate::vae::Flux2Vae;
use crate::{pipeline, to_image, Pipeline};

/// Paths to the FLUX.2-klein edit checkpoints — just the klein snapshot dir (`text_encoder/`,
/// `transformer/`, `vae/`, `tokenizer/`), the same snapshot the txt2img path loads.
pub struct Flux2EditPaths {
    /// FLUX.2-klein-9B diffusers snapshot dir.
    pub root: PathBuf,
}

/// One FLUX.2-klein edit request.
#[derive(Clone)]
pub struct Flux2EditRequest {
    pub prompt: String,
    /// Classifier-free negative prompt — used only when `guidance > 1` (distilled klein runs CFG-free).
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Guidance scale. 1.0 (klein default) = a single CFG-free forward; > 1.0 adds a negative pass.
    pub guidance: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Flux2EditRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: DEFAULT_STEPS as usize,
            guidance: DEFAULT_GUIDANCE,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// Loaded FLUX.2-klein edit model: the Qwen3 text encoder + the MMDiT + the VAE **with the encoder**
/// (the reference encode), plus the txt2img [`Pipeline`] handle (snapshot mmap + prompt encode + the
/// latent geometry/dtype). `generate` takes `&self` (no per-call mutation), so one load serves many
/// edits.
pub struct Flux2Edit {
    pipe: Pipeline,
    te: Qwen3TextEncoder,
    transformer: Flux2Transformer,
    vae: Flux2Vae,
}

impl Flux2Edit {
    /// Load the klein backbone with the VAE encoder enabled (the reference encode); everything else is
    /// the txt2img load (f32 compute, parity-sensitive).
    pub fn load(paths: &Flux2EditPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        // The klein edit path loads the klein snapshot dense (dev edit is epic 6564 story 4).
        let pipe = Pipeline::load(Flux2Variant::Klein9b, None, &paths.root, &device);
        let te = Qwen3TextEncoder::new(&pipe.cfg, pipe.component_vb("text_encoder")?)?;
        let transformer = Flux2Transformer::new(&pipe.cfg, pipe.component_vb("transformer")?)?;
        let vae = Flux2Vae::new_with_encoder(pipe.component_vb("vae")?)?;
        Ok(Self {
            pipe,
            te,
            transformer,
            vae,
        })
    }

    /// Generate one edited image. `references` (≥ 1) condition the denoise via reference token concat;
    /// the worker pre-fits them to the render size, but this re-resizes defensively.
    pub fn generate(
        &self,
        req: &Flux2EditRequest,
        references: &[Image],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if references.is_empty() {
            return Err(CandleError::Msg(
                "flux2 edit: at least one reference image is required".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(CandleError::Msg(format!(
                "flux2 edit: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if req.steps == 0 {
            return Err(CandleError::Msg("flux2 edit: steps must be >= 1".into()));
        }

        let device = &self.pipe.device;
        let cfg = &self.pipe.cfg;
        let guidance = req.guidance;
        let cfg_on = guidance > 1.0;

        // Prompt embeds are seed-independent: encode once. Negative only under CFG.
        let prompt_embeds = self.pipe.encode(&self.te, &req.prompt)?;
        let negative = if cfg_on {
            let neg = if req.negative.trim().is_empty() {
                " "
            } else {
                req.negative.as_str()
            };
            Some(self.pipe.encode(&self.te, neg)?)
        } else {
            None
        };

        // Reference conditioning: VAE-encode each ref → packed tokens [1, seq_ref, 128] + grid ids at
        // t = 10 + 10·i, all concatenated on the sequence axis. Clean + constant across the denoise.
        let (ref_tokens, ref_ids) = self.encode_references(references, req.width, req.height)?;

        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);
        let target_seq = lat_h * lat_w;
        // The joint image-stream ids: the t=0 target grid followed by the reference grids.
        let mut img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        img_ids.extend_from_slice(&ref_ids);
        let txt_ids = pipeline::prepare_text_ids(cfg.max_sequence_length);

        // Curated sampler/scheduler routing (epic 7114 P4, sc-7123) — the same driver the txt2img path
        // uses. The bespoke edit request carries no per-generation sampler/scheduler knob, so this runs
        // the default (`None`) euler over the native empirical-mu schedule: the N1 no-op that reproduces
        // the legacy `euler_step` flow-match loop within tolerance.
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), req.steps);
        let (native, _timesteps) = pipeline::schedule(req.steps, req.width, req.height);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);

        let latents = pipeline::create_noise(cfg, req.seed, req.width, req.height, device)?;
        // The driver does cancel + progress + the integrator step. The joint `[target, refs]` concat,
        // the transformer forward, the target-slice, and the guidance>1 CFG blend all live inside the
        // predict closure so a multi-eval solver re-runs them. FLUX.2 uses the Sigma convention but the
        // model embeds σ×1000, so feed `sigma * 1000.0` to the transformer.
        let latents = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            |latents, sigma| -> Result<Tensor> {
                let ts = sigma * 1000.0;
                // Joint image stream [target, refs] — references re-concatenated with the current target.
                let hidden = Tensor::cat(&[latents, &ref_tokens], 1)?;
                let v =
                    self.velocity(&hidden, &prompt_embeds, &img_ids, &txt_ids, ts, target_seq)?;
                match &negative {
                    Some(neg) => {
                        let vn = self.velocity(&hidden, neg, &img_ids, &txt_ids, ts, target_seq)?;
                        // vn + guidance·(v − vn)
                        Ok((&vn + ((&v - &vn)? * guidance as f64)?)?)
                    }
                    None => Ok(v),
                }
            },
        )?;

        on_progress(Progress::Decoding);
        let packed = pipeline::unpack_latents(&latents, req.width, req.height)?;
        let decoded = self.vae.decode_packed(&packed)?; // [1,3,H,W] in [-1,1]
        to_image(&decoded)
    }

    /// Run the transformer on the joint `[target, refs]` image stream and keep the leading
    /// `target_seq` velocity tokens (the target image stream; `proj_out` is per-token, so the slice is
    /// exact).
    fn velocity(
        &self,
        hidden: &Tensor,
        embeds: &Tensor,
        img_ids: &[[i64; 4]],
        txt_ids: &[[i64; 4]],
        ts: f32,
        target_seq: usize,
    ) -> Result<Tensor> {
        // klein edit is distilled (CFG-free / true-CFG via the negative pass) — no embedded guidance.
        let out = self
            .transformer
            .forward(hidden, embeds, img_ids, txt_ids, ts, None)?;
        Ok(out.narrow(1, 0, target_seq)?)
    }

    /// Encode N reference images into packed transformer tokens + their grid ids. Each: Lanczos-resize
    /// to the render size → normalize to `[-1,1]` NCHW → [`Flux2Vae::encode_packed`] (the mean encode +
    /// 2×2 patchify + bn-normalize the transformer space expects) → pack to `[1, seq, 128]`, tagged
    /// with grid ids at `t = 10 + 10·i`. Returns the concatenated `([1, Σseq, 128], Σ grid ids)`.
    fn encode_references(
        &self,
        references: &[Image],
        width: u32,
        height: u32,
    ) -> Result<(Tensor, Vec<[i64; 4]>)> {
        let (lat_h, lat_w) = pipeline::latent_dims(width, height);
        let mut tokens: Vec<Tensor> = Vec::with_capacity(references.len());
        let mut ids: Vec<[i64; 4]> = Vec::with_capacity(references.len() * lat_h * lat_w);
        for (i, image) in references.iter().enumerate() {
            let nchw = preprocess_ref(image, width, height, &self.pipe.device, self.pipe.dtype)?;
            let packed = self.vae.encode_packed(&nchw)?; // [1, 128, H/16, W/16]
            tokens.push(pipeline::pack_nchw(&packed)?); // [1, seq, 128]
            ids.extend(pipeline::prepare_grid_ids_t(
                lat_h,
                lat_w,
                10 + 10 * i as i64,
            ));
        }
        Ok((Tensor::cat(&tokens, 1)?, ids))
    }
}

/// Lanczos-resize a reference [`Image`] (RGB8) to the render size, normalize `[0,255] → [-1,1]`, lay
/// out as NCHW `[1, 3, H, W]` — the input [`Flux2Vae::encode_packed`] expects. Mirrors the mlx
/// `preprocess_ref_image` (`2·x − 1`). A no-op resize when the source is already the render size.
fn preprocess_ref(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(CandleError::Msg(format!(
            "flux2 edit: reference pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    let resized: Vec<f32> = if (ih, iw) == (rh, rw) {
        image.pixels.iter().map(|&v| v as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, rh, rw) // HWC f32 [0,255]
    };
    // [0,255] → [-1,1], then HWC → NCHW.
    let data: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let hwc = Tensor::from_vec(data, (rh, rw, 3), device)?;
    let nchw = hwc.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
    Ok(nchw.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request defaults match the klein edit production knobs (1024², 4 distilled steps, CFG-free).
    #[test]
    fn request_defaults() {
        let r = Flux2EditRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, DEFAULT_STEPS as usize);
        assert_eq!(r.guidance, DEFAULT_GUIDANCE);
        assert!(!r.cancel.is_cancelled());
    }

    /// `preprocess_ref` lays a same-size RGB8 reference out as NCHW `[1,3,H,W]` in `[-1,1]`: white → 1,
    /// black → −1 (the `2·x − 1` normalization), with the channel axis moved to front.
    #[test]
    fn preprocess_ref_normalizes_and_lays_out_nchw() {
        let dev = Device::Cpu;
        // 2×2 image: top-left white, the rest black.
        let pixels = vec![
            255, 255, 255, 0, 0, 0, // row 0: white, black
            0, 0, 0, 0, 0, 0, // row 1: black, black
        ];
        let img = Image {
            width: 2,
            height: 2,
            pixels,
        };
        let t = preprocess_ref(&img, 2, 2, &dev, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 3, 2, 2]);
        // Channel 0 (R), row-major after the HWC→NCHW move: [1, −1, −1, −1].
        let r = t
            .narrow(1, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(r, vec![1.0, -1.0, -1.0, -1.0]);
    }
}
