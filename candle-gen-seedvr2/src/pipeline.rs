//! SeedVR2 image-mode pipeline — candle port of `mlx-gen-seedvr2/src/pipeline.rs` (sc-5157).
//!
//! One-step super-resolution: preprocess the LR image (bicubic upscale to target, optional `softness`
//! pre-blur, [-1,1]) → VAE encode → conditioning latent (encoded latent + ones-mask) → concat fresh
//! noise → DiT (one step) → `latents = noise − DiT_out` → VAE decode → crop → LAB+wavelet color
//! correction → RGB8.
//!
//! **Video mode (sc-5926):** the 5-D temporal pass. `generate_video` sizes a temporal chunk against
//! the device VRAM budget ([`crate::video`]), runs each chunk through the same `T`-aware
//! encode/condition/denoise/decode path, per-frame color-corrects, and cross-fades chunk overlaps to
//! close the causal-VAE seam. It falls back to a per-frame (`T=1`) path under tight memory, and to
//! per-frame **spatial tiling** when even one full-resolution frame exceeds the budget (HD).
//!
//! The negative-prompt conditioning is a precomputed embedding (bundled `data/neg_embed.safetensors`,
//! no runtime text encoder), loaded at construction.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::gen_core::{imageops, Image, Quant};
use candle_gen::{CandleError, Result as CResult};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::config::{DitConfig, TIMESTEP};
use crate::dit::Seedvr2Transformer;
use crate::vae::Seedvr2Vae;
use crate::video::{self, Chunk, ChunkPlan};
use crate::weights::Weights;
use crate::{color, convert};

/// Post-decode color-correction luminance weight (the reference `apply_color_correction` default).
const LUMINANCE_WEIGHT: f32 = 0.8;

pub struct Seedvr2Pipeline {
    pub vae: Seedvr2Vae,
    pub transformer: Seedvr2Transformer,
    neg_embed: Tensor,
    dtype: DType,
    device: Device,
    /// Resident weight bytes (VAE + DiT at `dtype`) — drives the video memory-budget chunk sizer.
    weights_bytes: usize,
}

/// Estimate resident weight bytes for the video memory budget: the raw `fp16` checkpoint file sizes
/// scaled by the load `dtype` (`BF16` keeps the 2 B/param footprint, `F32` doubles it). File sizes
/// (vs summing per-tensor) match the wan `dit_resident_bytes` convention; the safetensors header
/// overhead is negligible.
fn resident_weight_bytes(files: &[&std::path::Path], dt: DType) -> usize {
    let raw: u64 = files
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let ratio = match dt {
        DType::F32 => 2.0, // fp16-on-disk → f32 resident
        _ => 1.0,          // bf16/fp16 resident
    };
    (raw as f64 * ratio) as usize
}

/// The bundled precomputed negative-prompt embedding → `(1, Lt, 5120)` at `dt`.
fn load_neg_embed(dt: DType, dev: &Device) -> CResult<Tensor> {
    const BYTES: &[u8] = include_bytes!("../data/neg_embed.safetensors");
    let map = candle_gen::candle_core::safetensors::load_buffer(BYTES, dev)?;
    let emb = map
        .get("embedding")
        .ok_or_else(|| CandleError::Msg("seedvr2 neg-embed: missing `embedding`".into()))?;
    Ok(emb.to_dtype(dt)?.unsqueeze(0)?)
}

/// Deterministic N(0,1) noise of a 5-D shape (CPU `StdRng`/ChaCha, launch-portable per seed).
fn seeded_normal5(
    seed: u64,
    shape: (usize, usize, usize, usize, usize),
    dt: DType,
    dev: &Device,
) -> CResult<Tensor> {
    let (a, b, c, d, e) = shape;
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..a * b * c * d * e)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    Ok(Tensor::from_vec(data, shape, dev)?.to_dtype(dt)?)
}

/// `(1,3,H,W)` in [-1,1] → RGB8 [`Image`].
fn decoded_to_image(decoded: &Tensor) -> Result<Image> {
    let (_b, _c, h, w) = decoded.dims4()?;
    let u8s = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .to_device(&Device::Cpu)?;
    let chw = u8s.squeeze(0)?; // (3,H,W)
    let pixels = chw.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

impl Seedvr2Pipeline {
    /// Build from already-converted candle-layout VAE + DiT weights + a neg-embed (parity tests).
    pub fn from_parts(
        vae: Seedvr2Vae,
        transformer: Seedvr2Transformer,
        neg_embed: Tensor,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            vae,
            transformer,
            neg_embed,
            dtype,
            device,
            weights_bytes: 0,
        }
    }

    /// Load from a raw `numz/SeedVR2_comfyUI` checkpoint dir: convert in-memory (no Python), cast to
    /// `dt`, attach the bundled neg-embed. `dit_file` selects 3B/7B.
    pub fn load(
        raw_dir: impl AsRef<std::path::Path>,
        dit_file: &str,
        cfg: &DitConfig,
        dt: DType,
        device: &Device,
    ) -> CResult<Self> {
        let dir = raw_dir.as_ref();
        let vae_path = dir.join("ema_vae_fp16.safetensors");
        let dit_path = dir.join(dit_file);
        let weights_bytes = resident_weight_bytes(&[vae_path.as_path(), dit_path.as_path()], dt);
        let vae_raw = Weights::from_file(&vae_path, device)?;
        let dit_raw = Weights::from_file(&dit_path, device)?;
        let vae_w = convert::convert_vae(&vae_raw)?.cast(dt)?;
        let dit_w = convert::convert_dit(&dit_raw)?.cast(dt)?;
        let vae = Seedvr2Vae::from_weights(&vae_w)?;
        let transformer = Seedvr2Transformer::from_weights(&dit_w, cfg)?;
        let neg_embed = load_neg_embed(dt, device)?;
        let mut p = Self::from_parts(vae, transformer, neg_embed, dt, device.clone());
        p.weights_bytes = weights_bytes;
        Ok(p)
    }

    /// Quantize the DiT Linears to `quant` (`Q4_0`/`Q8_0`) — Linear-only (sc-5927); the VAE stays
    /// dense. Call once after [`Self::load`], before the pipeline is shared. `weights_bytes` is
    /// intentionally **not** reduced — keeping the dense estimate makes the video chunk-size budget
    /// conservative (quant shrinks the weights, not the activations, so the headroom stays safe).
    pub fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.transformer.quantize(quant)
    }

    /// Build the static condition `[latent, ones-mask]` → `(B, 17, T', h, w)`.
    fn condition(latent: &Tensor) -> Result<Tensor> {
        let (b, _c, t, h, w) = latent.dims5()?;
        let mask = Tensor::ones((b, 1, t, h, w), latent.dtype(), latent.device())?;
        Tensor::cat(&[latent, &mask], 1)
    }

    /// One denoise step: `vid = [noise, condition]` → DiT → `noise − DiT_out`.
    fn denoise(&self, noise: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let model_input = Tensor::cat(&[noise, condition], 1)?; // (B,33,T',h,w)
        let dit_out = self
            .transformer
            .forward(&model_input, &self.neg_embed, TIMESTEP)?;
        noise - dit_out
    }

    /// Decode latents and crop to `(true_h, true_w)` (first frame) → `(1,3,true_h,true_w)`.
    fn decode_crop(&self, latents: &Tensor, true_h: usize, true_w: usize) -> Result<Tensor> {
        let decoded = self.vae.decode(latents)?; // (1,3,T,H,W)
        decoded
            .narrow(2, 0, 1)?
            .squeeze(2)? // (1,3,H,W)
            .narrow(2, 0, true_h)?
            .narrow(3, 0, true_w)?
            .contiguous()
    }

    /// Full model path (no color correction): preprocessed image + injected noise → decoded crop.
    /// Public for the golden parity harness (the engine-level parity check).
    pub fn run_model(
        &self,
        processed: &Tensor,
        noise: &Tensor,
        true_h: usize,
        true_w: usize,
    ) -> Result<Tensor> {
        let latent = self.vae.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond)?;
        self.decode_crop(&latents, true_h, true_w)
    }

    /// End-to-end upscale: LR `image` → `(width, height)` super-resolved RGB8 image.
    ///
    /// Spatial-tiles when a single full-resolution pass would exceed the memory budget (sc-6225) — the
    /// image analog of the video path's HD-tiling fallback (sc-5926). See [`Self::generate_budgeted`].
    pub fn generate(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
    ) -> CResult<Image> {
        self.generate_budgeted(
            image,
            width,
            height,
            seed,
            softness,
            video::safe_budget_gib(),
        )
    }

    /// [`Self::generate`] with the safe peak-GB ceiling injected (so the spatial-tiling path is
    /// unit-testable without a multi-GB target — mirrors [`crate::video::plan_chunk_size_with`]).
    /// When even a single full-resolution pass would exceed `safe_gib`, the image is upscaled by
    /// feather-blended spatial tiling ([`Self::run_frame_tiled`], the parity-gated sc-5926 tiler)
    /// rather than one allocation that would blow past the device's free VRAM and OOM the worker
    /// (sc-6225); otherwise the one-pass still path runs (numerically unchanged from before).
    pub fn generate_budgeted(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        safe_gib: f64,
    ) -> CResult<Image> {
        if matches!(
            video::plan_chunk_size_with(self.weights_bytes, height as i32, width as i32, safe_gib),
            ChunkPlan::OverBudget { .. }
        ) {
            return self.generate_tiled(image, width, height, seed, softness, safe_gib);
        }

        let processed = self.preprocess(image, width, height, softness)?; // (1,3,H,W)
        let latent = self.vae.encode(&processed)?;
        let (_b, _c, lt, lh, lw) = latent.dims5()?;
        let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(&noise, &cond)?;
        let decoded = self.decode_crop(&latents, height, width)?; // (1,3,H,W)
        let corrected = color::apply_color_correction(
            &decoded.to_dtype(DType::F32)?,
            &processed.to_dtype(DType::F32)?,
            LUMINANCE_WEIGHT,
        )?;
        Ok(decoded_to_image(&corrected)?)
    }

    /// Spatial-tiling still-image path (sc-6225): upscale one LR image by feather-blended spatial
    /// tiling — the image analog of the [`Self::generate_video_tiled`] per-frame branch. Reuses the
    /// budget tile sizer + parity-gated [`Self::run_frame_tiled`] + per-frame color correction, so peak
    /// stays bounded at any resolution (no single allocation exceeds the budget-sized tile). `safe_gib`
    /// sizes the tile.
    fn generate_tiled(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        safe_gib: f64,
    ) -> CResult<Image> {
        let tile = video::plan_spatial_tile_px(self.weights_bytes, safe_gib);
        let overlap = video::SPATIAL_OVERLAP.min(tile / 2);
        let processed = self
            .preprocess(image, width, height, softness)?
            .unsqueeze(2)?; // (1,3,1,H,W)
        let decoded = self.run_frame_tiled(&processed, seed, tile, overlap)?;
        Ok(self
            .frames_from_decoded(&decoded, &processed, 1)?
            .into_iter()
            .next()
            .expect("one tiled frame"))
    }

    /// LR `Image` → `(1,3,height,width)` in [-1,1] at the model dtype. Bicubic resize to target;
    /// optional `softness` pre-blur via a smaller round-trip.
    pub fn preprocess(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        softness: f32,
    ) -> CResult<Tensor> {
        let (ih, iw) = (image.height as usize, image.width as usize);
        let resized: Vec<f32> = if softness > 0.0 {
            let factor = 1.0 + softness.clamp(0.0, 1.0) * 7.0;
            let dw = ((width as f32 / factor) as usize).max(2);
            let dh = ((height as f32 / factor) as usize).max(2);
            let down = imageops::resize_bicubic_u8(&image.pixels, ih, iw, dh, dw);
            let down_u8: Vec<u8> = down
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect();
            imageops::resize_bicubic_u8(&down_u8, dh, dw, height, width)
        } else {
            imageops::resize_bicubic_u8(&image.pixels, ih, iw, height, width)
        };
        // HWC [0,255] → [-1,1] → (1,3,H,W)
        let arr = Tensor::from_vec(resized, (height, width, 3), &self.device)?;
        let arr = (arr.affine(2.0 / 255.0, -1.0))?;
        Ok(arr
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?
            .contiguous()?)
    }

    // -----------------------------------------------------------------------
    // video mode (sc-5926): multi-frame 5-D pass + temporal chunking/overlap + HD tiling
    // -----------------------------------------------------------------------

    /// Decode latents and crop spatially to `(true_h, true_w)`, **keeping all `T` frames** →
    /// `(B,3,T,true_h,true_w)`. The 5-D analog of [`Self::decode_crop`] (which keeps only frame 0).
    pub fn decode_crop_5d(&self, latents: &Tensor, true_h: usize, true_w: usize) -> Result<Tensor> {
        let decoded = self.vae.decode(latents)?; // (B,3,T,H,W)
        decoded
            .narrow(3, 0, true_h)?
            .narrow(4, 0, true_w)?
            .contiguous()
    }

    /// Multi-frame model path (no color correction): a preprocessed clip `(1,3,T,H,W)` + injected
    /// noise `(1,16,T',h,w)` → decoded crop `(1,3,T,true_h,true_w)`. The video analog of
    /// [`Self::run_model`]; `encode`/`condition`/`denoise` are already `T`-aware. Public for the harness.
    pub fn run_model_5d(
        &self,
        processed: &Tensor,
        noise: &Tensor,
        true_h: usize,
        true_w: usize,
    ) -> Result<Tensor> {
        let latent = self.vae.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond)?;
        self.decode_crop_5d(&latents, true_h, true_w)
    }

    /// Per-frame color-correct a decoded clip `(1,3,T,Hc,Wc)` against its preprocessed `style`
    /// `(1,3,Ts,Hc,Wc)` → `count` RGB8 frames. Frame `t` matches style frame `min(t, Ts-1)`.
    fn frames_from_decoded(
        &self,
        decoded: &Tensor,
        style: &Tensor,
        count: usize,
    ) -> CResult<Vec<Image>> {
        let style_t = style.dim(2)?;
        let mut out = Vec::with_capacity(count);
        for t in 0..count {
            let d = decoded.narrow(2, t, 1)?.squeeze(2)?; // (1,3,Hc,Wc)
            let s = style.narrow(2, t.min(style_t - 1), 1)?.squeeze(2)?;
            let corrected = color::apply_color_correction(
                &d.to_dtype(DType::F32)?,
                &s.to_dtype(DType::F32)?,
                LUMINANCE_WEIGHT,
            )?;
            out.push(decoded_to_image(&corrected)?);
        }
        Ok(out)
    }

    /// Preprocess one temporal chunk: pixel-frames `[start, start+len)` of `frames`, clamping past the
    /// end to the last real frame (last-frame padding) → `(1,3,len,H,W)` in `[-1,1]`.
    fn preprocess_chunk(
        &self,
        frames: &[Image],
        start: i32,
        len: i32,
        width: usize,
        height: usize,
        softness: f32,
    ) -> CResult<Tensor> {
        let n = frames.len() as i32;
        let mut per = Vec::with_capacity(len as usize);
        for j in 0..len {
            let idx = (start + j).clamp(0, n - 1) as usize;
            per.push(
                self.preprocess(&frames[idx], width, height, softness)?
                    .unsqueeze(2)?, // (1,3,1,H,W)
            );
        }
        let refs: Vec<&Tensor> = per.iter().collect();
        Ok(Tensor::cat(&refs, 2)?)
    }

    /// End-to-end **video** upscale: a sequence of LR `frames` → upscaled `(width, height)` RGB8
    /// frames (same count). Sizes the temporal chunk against the memory budget (or `chunk_override`),
    /// processes each chunk through the 5-D model path with one-step Euler, per-frame color-corrects,
    /// and cross-fades chunk overlaps to close the causal-VAE seam ([`crate::video`]). Falls back to
    /// the per-frame (`T=1`) path under tight memory, and to per-frame **spatial tiling** when even one
    /// full-resolution frame exceeds the budget (HD), so peak stays bounded at any resolution.
    pub fn generate_video(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        chunk_override: Option<i32>,
    ) -> CResult<Vec<Image>> {
        let n = frames.len() as i32;
        if n == 0 {
            return Ok(Vec::new());
        }
        let chunk = match (
            chunk_override,
            video::plan_chunk_size(self.weights_bytes, height as i32, width as i32),
        ) {
            // Clamp an override DOWN to the budget-safe chunk so it can't bypass the planner and OOM;
            // a smaller-than-safe override is honored as-is. PerFrame/OverBudget route to the safe
            // paths regardless of the override, since forcing a chunk there would OOM.
            (Some(c), ChunkPlan::Chunked(safe)) => video::pad_to_valid_chunk(c).min(safe),
            (_, ChunkPlan::Chunked(c)) => c,
            (_, ChunkPlan::PerFrame) => {
                return self.generate_video_per_frame(frames, width, height, seed, softness)
            }
            // Even one full-resolution frame exceeds the budget → spatially tile each frame
            // (per-frame T=1 + overlap feather blend). Bounds peak at any resolution (sc-5201).
            (_, ChunkPlan::OverBudget { .. }) => {
                return self.generate_video_tiled(frames, width, height, seed, softness)
            }
        };

        let plan = video::plan_chunks(n, chunk, video::DEFAULT_OVERLAP);
        let mut chunk_frames: Vec<Vec<Image>> = Vec::with_capacity(plan.len());
        for Chunk { start, len } in &plan {
            let clip = self.preprocess_chunk(frames, *start, *len, width, height, softness)?;
            let latent = self.vae.encode(&clip)?;
            let (_b, _c, lt, lh, lw) = latent.dims5()?;
            // Same noise per chunk (deterministic) → a clean overlap blend (the reference behavior).
            let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
            let cond = Self::condition(&latent)?;
            let latents = self.denoise(&noise, &cond)?;
            let decoded = self.decode_crop_5d(&latents, height, width)?;
            chunk_frames.push(self.frames_from_decoded(&decoded, &clip, *len as usize)?);
        }
        Ok(video::assemble_overlap(
            &plan,
            &chunk_frames,
            n,
            video::DEFAULT_OVERLAP,
        ))
    }

    /// Per-frame (`T=1`) video fallback: each frame through the still path with a fixed (anchored)
    /// seed — intrinsically temporally stable (spike sc-4812). Used when even an 8-frame chunk
    /// exceeds the memory budget.
    fn generate_video_per_frame(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
    ) -> CResult<Vec<Image>> {
        frames
            .iter()
            .map(|f| self.generate(f, width, height, seed, softness))
            .collect()
    }

    /// HD spatial-tiling video path (sc-5201): each frame is upscaled per-frame (`T=1`) but **spatially
    /// tiled** — the budget sizer picks the largest square tile that fits, and the decoded tiles are
    /// feather-blended. Used when even one full-resolution frame exceeds the memory budget; bounds peak
    /// at any resolution.
    fn generate_video_tiled(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
    ) -> CResult<Vec<Image>> {
        let tile = video::plan_spatial_tile_px(self.weights_bytes, video::safe_budget_gib());
        let overlap = video::SPATIAL_OVERLAP.min(tile / 2);
        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            let processed = self.preprocess(f, width, height, softness)?.unsqueeze(2)?; // (1,3,1,H,W)
            let decoded = self.run_frame_tiled(&processed, seed, tile, overlap)?;
            let imgs = self.frames_from_decoded(&decoded, &processed, 1)?;
            out.push(imgs.into_iter().next().expect("one frame"));
        }
        Ok(out)
    }

    /// Upscale one preprocessed frame `(1,3,1,H,W)` by spatial tiling: run the full encode → DiT →
    /// decode path on each overlapping `tile`-px tile (one-step Euler, same-seed noise) and feather-
    /// blend the decoded tiles into a full `(1,3,1,H,W)` frame. candle is eager, so only one tile's
    /// activations are resident at a time (the memory bound); the two full-frame accumulators are the
    /// only persistent allocations. Public for the harness.
    pub fn run_frame_tiled(
        &self,
        processed: &Tensor,
        seed: u64,
        tile: i32,
        overlap: i32,
    ) -> CResult<Tensor> {
        let (_b, _c, _t, height, width) = processed.dims5()?; // (1,3,1,H,W)
        let (h_i, w_i) = (height as i32, width as i32);
        let plan = video::plan_spatial_tiles(h_i, w_i, tile, overlap);
        let mut acc: Option<Tensor> = None; // (1,3,1,H,W)
        let mut wsum: Option<Tensor> = None; // (1,1,1,H,W)
        for t in &plan {
            let (th, tw) = ((t.y1 - t.y0) as usize, (t.x1 - t.x0) as usize);
            let (y0, x0) = (t.y0 as usize, t.x0 as usize);
            let tile_clip = processed.narrow(3, y0, th)?.narrow(4, x0, tw)?; // (1,3,1,th,tw)

            // full model path on the tile (one-step Euler), same noise per tile.
            let latent = self.vae.encode(&tile_clip)?;
            let (_b, _c, lt, lh, lw) = latent.dims5()?;
            let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
            let cond = Self::condition(&latent)?;
            let latents = self.denoise(&noise, &cond)?;
            let decoded = self.decode_crop_5d(&latents, th, tw)?; // (1,3,1,th,tw)

            // feather weight tapering on edges that abut a neighbor; placed at (y0,x0).
            let wvec = video::feather_weight(
                th as i32,
                tw as i32,
                t.y0 > 0,
                t.y1 < h_i,
                t.x0 > 0,
                t.x1 < w_i,
                overlap,
            );
            let weight =
                Tensor::from_vec(wvec, (1, 1, 1, th, tw), &self.device)?.to_dtype(self.dtype)?;
            let (pad_r, pad_b) = (width - x0 - tw, height - y0 - th);
            let wdec = decoded
                .broadcast_mul(&weight)?
                .pad_with_zeros(3, y0, pad_b)?
                .pad_with_zeros(4, x0, pad_r)?;
            let wpad = weight
                .pad_with_zeros(3, y0, pad_b)?
                .pad_with_zeros(4, x0, pad_r)?;
            acc = Some(match acc {
                Some(a) => a.add(&wdec)?,
                None => wdec,
            });
            wsum = Some(match wsum {
                Some(a) => a.add(&wpad)?,
                None => wpad,
            });
        }
        Ok(acc
            .expect("≥1 tile")
            .broadcast_div(&wsum.expect("≥1 tile"))?)
    }
}
