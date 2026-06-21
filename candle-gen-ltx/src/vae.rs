//! LTX-2.3 **video VAE decoder** (`CausalVideoAutoencoder`, latent 128-ch, patch 4, 8× temporal /
//! 32× spatial) — port of mlx-gen-ltx `vae.rs` (`LTX2VideoDecoder`). T2V needs only `decode`; the
//! encoder (I2V) is deferred.
//!
//! Decode: denormalize `latent·std + mean` → `conv_in 128→1024` → 9 up_blocks (`Res` groups +
//! `DepthToSpace` upsamplers) → pixel-norm (eps 1e-8) → SiLU → `conv_out 128→48` → unpatchify(×4).
//! All convs are non-causal (frame-replication temporal pad). pixel_norm = `x/√(mean(x² over C)+eps)`
//! (no √C, no γ). Runs **f32**.
//!
//! Block execution order (the config `decoder_blocks` list is encoder-order; the decoder reverses
//! it): `Res(2), Up(2,2,2), Res(2), Up(2,2,2), Res(4), Up(2,1,1), Res(6), Up(1,2,2), Res(4)`. Each
//! `Up` with temporal stride 2 doubles then drops the first frame, so latent T=7 → 49 pixel frames;
//! spatial 15 → 480 px (×2×2×2 then unpatchify ×4).

use candle_gen::candle_core::{Error, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::{
    budgeted_plan, TileCandidates, TilingBudgetError, TilingConfig, VaeTiling,
};

use crate::conv3d::CausalConv3d;

const DEC_NORM_EPS: f64 = 1e-8;

/// `x / sqrt(mean(x² over C, keepdims) + eps)` — LTX PixelNorm (channel axis = 1, no √C, no γ).
fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)?;
    let sumsq = x.sqr()?.sum_keepdim(1)?;
    let mean = (sumsq / c as f64)?;
    let denom = (mean + DEC_NORM_EPS)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// Decoder residual block (`ResnetBlock3DSimple`): pixel-norm → SiLU → conv → pixel-norm → SiLU →
/// conv → residual add. Channels constant (no shortcut).
struct DecResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl DecResBlock {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::load(vb.clone(), &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::load(vb, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm(x)?)?;
        let h = self.conv1.forward(&h, false)?;
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm(&h)?)?;
        let h = self.conv2.forward(&h, false)?;
        h + x
    }
}

/// `DepthToSpaceUpsample` (residual=false): conv → depth-to-space → (st>1) drop first temporal frame.
struct DepthToSpace {
    conv: CausalConv3d,
    st: usize,
    sh: usize,
    sw: usize,
}

impl DepthToSpace {
    fn load(vb: VarBuilder, prefix: &str, stride: (usize, usize, usize)) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::load(vb, &format!("{prefix}.conv.conv"))?,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
        })
    }

    /// `(B, C·st·sh·sw, D, H, W) -> (B, C, D·st, H·sh, W·sw)`.
    fn depth_to_space(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_packed, d, h, w) = x.dims5()?;
        let (st, sh, sw) = (self.st, self.sh, self.sw);
        let c = c_packed / (st * sh * sw);
        let x = x.reshape([b, c, st, sh, sw, d, h, w].as_slice())?;
        // transpose to (B, C, D, st, H, sh, W, sw) = axes [0,1,5,2,6,3,7,4].
        let x = x.permute([0usize, 1, 5, 2, 6, 3, 7, 4].as_slice())?;
        x.reshape((b, c, d * st, h * sh, w * sw))?.contiguous()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x, false)?;
        let x = self.depth_to_space(&x)?;
        if self.st > 1 {
            let t = x.dim(2)?;
            x.narrow(2, 1, t - 1)
        } else {
            Ok(x)
        }
    }
}

enum UpLayer {
    Res(Vec<DecResBlock>),
    Up(DepthToSpace),
}

/// One decoder block in execution order: a res group of `n` blocks, or an upsampler with `stride`.
enum DBlock {
    Res(usize),
    Up((usize, usize, usize)),
}

/// The fixed LTX-2.3 decoder block order (config `decoder_blocks` already reversed to execution order).
const DECODER_BLOCKS: [DBlock; 9] = [
    DBlock::Res(2),
    DBlock::Up((2, 2, 2)),
    DBlock::Res(2),
    DBlock::Up((2, 2, 2)),
    DBlock::Res(4),
    DBlock::Up((2, 1, 1)),
    DBlock::Res(6),
    DBlock::Up((1, 2, 2)),
    DBlock::Res(4),
];

/// `(B, C·p², F, H, W) -> (B, C, F, H·p, W·p)` (spatial-only unpatchify, patch_size_t = 1).
fn unpatchify(x: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c_packed, f, h, w) = x.dims5()?;
    let c = c_packed / (p * p);
    // (B, C, 1, p, p, F, H, W) -> transpose (0,1,5,2,6,4,7,3) -> (B, C, F, H·p, W·p).
    let x = x.reshape([b, c, 1, p, p, f, h, w].as_slice())?;
    let x = x.permute([0usize, 1, 5, 2, 6, 4, 7, 3].as_slice())?;
    x.reshape((b, c, f, h * p, w * p))?.contiguous()
}

/// The LTX-2.3 video VAE (decoder only, T2V).
pub struct LtxVideoVae {
    conv_in: CausalConv3d,
    up_blocks: Vec<UpLayer>,
    conv_out: CausalConv3d,
    mean: Tensor, // [1, 128, 1, 1, 1]
    std: Tensor,  // [1, 128, 1, 1, 1]
    patch_size: usize,
}

impl LtxVideoVae {
    /// Build from a VarBuilder rooted at the `vae.` prefix of the checkpoint.
    pub fn new(vb: VarBuilder, latent_channels: usize, patch_size: usize) -> Result<Self> {
        let dec = vb.pp("decoder");
        let mut up_blocks = Vec::with_capacity(DECODER_BLOCKS.len());
        for (idx, block) in DECODER_BLOCKS.iter().enumerate() {
            let prefix = format!("up_blocks.{idx}");
            up_blocks.push(match block {
                DBlock::Res(n) => {
                    let mut blocks = Vec::with_capacity(*n);
                    for j in 0..*n {
                        blocks.push(DecResBlock::load(
                            dec.clone(),
                            &format!("{prefix}.res_blocks.{j}"),
                        )?);
                    }
                    UpLayer::Res(blocks)
                }
                DBlock::Up(stride) => {
                    UpLayer::Up(DepthToSpace::load(dec.clone(), &prefix, *stride)?)
                }
            });
        }
        let stats = vb.pp("per_channel_statistics");
        let mean = stats
            .get_unchecked("mean-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        let std = stats
            .get_unchecked("std-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        Ok(Self {
            conv_in: CausalConv3d::load(dec.clone(), "conv_in.conv")?,
            up_blocks,
            conv_out: CausalConv3d::load(dec, "conv_out.conv")?,
            mean,
            std,
            patch_size,
        })
    }

    /// Decode a normalized latent `[B, 128, F', H', W']` → video `[B, 3, F, 32·H', 32·W']` in ~[-1,1].
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        // Denormalize: x · std + mean.
        let x =
            (latent.broadcast_mul(&self.std)? + self.mean.broadcast_as(latent.shape())?.clone())?;
        let mut x = self.conv_in.forward(&x, false)?;
        for layer in &self.up_blocks {
            x = match layer {
                UpLayer::Res(blocks) => {
                    let mut h = x;
                    for b in blocks {
                        h = b.forward(&h)?;
                    }
                    h
                }
                UpLayer::Up(u) => u.forward(&x)?,
            };
        }
        let x = pixel_norm(&x)?;
        let x = candle_gen::candle_nn::ops::silu(&x)?;
        let x = self.conv_out.forward(&x, false)?;
        unpatchify(&x, self.patch_size)
    }

    /// Decode with **tiling** for memory-bounded large/long-video decode (`cfg`) — the candle port of
    /// mlx-gen-ltx `LtxVideoVae::decode_tiled` (sc-7076 / sc-6894). Splits the latent into overlapping
    /// spatial/temporal tiles via the shared pure `gen_core::tiling` geometry (`VaeTiling::LTX`: ×32
    /// spatial, ×8 **causal** temporal), decodes each tile through [`decode`](Self::decode), and
    /// trapezoidally blends them into the full video by pad-and-accumulate (bounded peak = one tile's
    /// decode + the full-output `output`/`weights` buffers). Falls back to single-pass [`decode`] when
    /// `cfg` does not fire for these dims.
    ///
    /// Numerically mirrors the parity-validated mlx version op-for-op; candle's eager evaluation makes
    /// the reference's per-tile `mx.eval` (the peak-bounding barrier) unnecessary. **NOTE (CUDA-gated):**
    /// the spatial-tiling path is straightforward, but temporal tiling crosses the causal-Conv3d frame
    /// boundary (each tile's leading edge replicates the *tile's* first frame) — the gen_core causal
    /// temporal mapping handles the geometry, but byte-parity vs. a full single-pass decode must be
    /// confirmed on real weights + CUDA (the Mac dev host can only compile-check this).
    pub fn decode_tiled(&self, latent: &Tensor, cfg: &TilingConfig) -> Result<Tensor> {
        let (_b, _c, f, h, w) = latent.dims5()?;
        if !cfg.needs_tiling(VaeTiling::LTX, f as i32, h as i32, w as i32) {
            return self.decode(latent);
        }
        let plan = cfg.plan(VaeTiling::LTX, f as i32, h as i32, w as i32);
        let dev = latent.device();

        // Full-size accumulators (the reference allocates these too); pad-and-add each tile in turn.
        // `output` carries the batch; `weights` stays `b=1` and broadcasts on the final divide.
        let mut output: Option<Tensor> = None; // [B, 3, out_f, out_h, out_w]
        let mut weights: Option<Tensor> = None; // [1, 1, out_f, out_h, out_w]

        for t in &plan.t {
            for hh in &plan.h {
                for ww in &plan.w {
                    let tile = latent
                        .narrow(2, t.start as usize, (t.end - t.start) as usize)?
                        .narrow(3, hh.start as usize, (hh.end - hh.start) as usize)?
                        .narrow(4, ww.start as usize, (ww.end - ww.start) as usize)?;
                    let dec = self.decode(&tile)?; // [B, 3, td, hd, wd]
                    let (_, _, td, hd, wd) = dec.dims5()?;
                    let at = td.min((t.out_stop - t.out_start) as usize);
                    let ah = hd.min((hh.out_stop - hh.out_start) as usize);
                    let aw = wd.min((ww.out_stop - ww.out_start) as usize);

                    // 1-D trapezoidal masks → outer product [1, 1, at, ah, aw].
                    let tm = Tensor::from_slice(&t.mask[..at], (1, 1, at, 1, 1), dev)?;
                    let hm = Tensor::from_slice(&hh.mask[..ah], (1, 1, 1, ah, 1), dev)?;
                    let wm = Tensor::from_slice(&ww.mask[..aw], (1, 1, 1, 1, aw), dev)?;
                    let blend = tm.broadcast_mul(&hm)?.broadcast_mul(&wm)?;

                    let dec = dec.narrow(2, 0, at)?.narrow(3, 0, ah)?.narrow(4, 0, aw)?;
                    let weighted = dec.broadcast_mul(&blend)?;

                    // Place each tile at its output offset by zero-padding to the full output shape.
                    // `out_start + a* <= out_stop <= out_*`, so the right pad never underflows.
                    let (pt0, pt1) = (
                        t.out_start as usize,
                        plan.out_f as usize - (t.out_start as usize + at),
                    );
                    let (ph0, ph1) = (
                        hh.out_start as usize,
                        plan.out_h as usize - (hh.out_start as usize + ah),
                    );
                    let (pw0, pw1) = (
                        ww.out_start as usize,
                        plan.out_w as usize - (ww.out_start as usize + aw),
                    );
                    let pad5 = |x: &Tensor| -> Result<Tensor> {
                        x.pad_with_zeros(2, pt0, pt1)?
                            .pad_with_zeros(3, ph0, ph1)?
                            .pad_with_zeros(4, pw0, pw1)
                    };
                    let weighted_full = pad5(&weighted)?;
                    let blend_full = pad5(&blend)?;

                    output = Some(match output {
                        None => weighted_full,
                        Some(acc) => acc.add(&weighted_full)?,
                    });
                    weights = Some(match weights {
                        None => blend_full,
                        Some(acc) => acc.add(&blend_full)?,
                    });
                }
            }
        }

        let output =
            output.ok_or_else(|| Error::Msg("ltx vae: tile-decode plan had no tiles".into()))?;
        let weights =
            weights.ok_or_else(|| Error::Msg("ltx vae: tile-decode plan had no tiles".into()))?;
        // Normalize by the summed blend weight (clamped away from 0), broadcasting [1,1,F,H,W] over C.
        output.broadcast_div(&weights.maximum(1e-8f64)?)
    }

    /// **Memory-bounded** decode (sc-7076): derive the decoded output dims from the latent geometry
    /// (LTX VAE: ×32 spatial, ×8 **causal** temporal ⇒ `out_f = 1 + (T_lat−1)·8`), pick a budgeted
    /// tiling via [`auto_tiling_budgeted_ltx`], and run [`decode_tiled`](Self::decode_tiled) — or the
    /// single-pass [`decode`](Self::decode) when the whole decode already fits the VRAM budget. An
    /// over-budget decode returns a **catchable** error here instead of OOM-ing the worker. The candle
    /// analogue of mlx-gen-ltx `decode_to_frames`'s internal budgeting.
    pub fn decode_budgeted(&self, latent: &Tensor) -> Result<Tensor> {
        let (_b, _c, f, h, w) = latent.dims5()?;
        let out_f = 1 + (f as i32 - 1) * VaeTiling::LTX.temporal_scale; // causal ×8
        let out_h = h as i32 * VaeTiling::LTX.spatial_scale; // ×32
        let out_w = w as i32 * VaeTiling::LTX.spatial_scale;
        match auto_tiling_budgeted_ltx(out_h, out_w, out_f)? {
            Some(cfg) => self.decode_tiled(latent, &cfg),
            None => self.decode(latent),
        }
    }
}

// --- sc-7076 / sc-6894: budgeted LTX VAE decode (candle) ------------------------------------------
//
// Mirrors mlx-gen-ltx's budgeted decode: the shared `gen_core::tiling::budgeted_plan` selector + an
// LTX cost model + a CUDA-VRAM budget source. The selector and geometry are byte-identical to the mlx
// side (pure gen-core); only the cost CONSTANTS and the budget source are backend-specific.

const GIB_F64: f64 = 1024.0 * 1024.0 * 1024.0;
/// Fraction of total VRAM treated as safe (matches the mlx 0.85 + candle-gen-seedvr2 convention).
const LTX_VAE_BUDGET_SAFE_FRAC: f64 = 0.85;
/// Fallback budget when neither the env override nor `nvidia-smi` yields a value.
const LTX_VAE_DEFAULT_BUDGET_GIB: f64 = 16.0;

// Cost-model constants. **PLACEHOLDER — seeded from the mlx-Metal real-weight anchors (sc-6894):
// LTX is fixed-floor-dominated (resident decoder) + light per-voxel. The candle/CUDA peak profile
// (discrete VRAM, different allocator) differs, so these MUST be re-measured on a CUDA box (a
// `vae_decode_sweep`-style harness) before the budget is trusted in prod — they are directionally
// right (conservative on Metal), not transferable. Until then the selector tiles *roughly* correctly.
const LTX_VAE_FIXED_BYTES: f64 = 3.3e9;
const LTX_VAE_ACCUM_BYTES_PER_VOXEL: f64 = 40.0;
const LTX_VAE_TILE_BYTES_PER_OUT_VOXEL: f64 = 300.0;

/// Candidate spatial tile sizes (output px, multiples of the LTX ×32 scale, overlap 64).
const LTX_VAE_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames.
const LTX_VAE_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 16), (48, 16), (24, 8)];

/// Estimated concurrent peak (GiB) of an LTX decode whose largest tile spans `tile_*` output voxels
/// while assembling an `out_*` video. `FIXED + ACCUM·out_vox + TILE·tile_vox`. Single-pass is
/// `tile_* == out_*`; a zero tile is the accumulator+fixed floor.
fn estimated_ltx_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    tile_f: i64,
    tile_h: i64,
    tile_w: i64,
) -> f64 {
    let out_voxels = (out_f * out_h * out_w) as f64;
    let tile_voxels = (tile_f * tile_h * tile_w) as f64;
    (LTX_VAE_FIXED_BYTES
        + LTX_VAE_ACCUM_BYTES_PER_VOXEL * out_voxels
        + LTX_VAE_TILE_BYTES_PER_OUT_VOXEL * tile_voxels)
        / GIB_F64
}

/// Total VRAM (GiB) read from `nvidia-smi` (min across GPUs) — the SceneWorks worker convention.
/// `None` off-CUDA (e.g. the Mac dev host, where the budget falls back to the env override / default).
/// Mirrors `candle-gen-seedvr2::video`'s reader (de-dupe into candle-gen core is a follow-up).
fn nvidia_smi_total_gib() -> Option<f64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let min_mb = text
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .filter(|&mb| mb > 0.0)
        .fold(f64::INFINITY, f64::min);
    min_mb.is_finite().then_some(min_mb / 1024.0)
}

/// The safe peak-GiB budget for the LTX decode tiler. Resolved in order: `LTX_VAE_BUDGET_GIB` env
/// override (positive float — the deterministic injection point for the worker/tests) → total VRAM ×
/// [`LTX_VAE_BUDGET_SAFE_FRAC`] (via `nvidia-smi`) → [`LTX_VAE_DEFAULT_BUDGET_GIB`].
pub fn ltx_vae_safe_budget_gib() -> f64 {
    if let Ok(raw) = std::env::var("LTX_VAE_BUDGET_GIB") {
        if let Ok(gib) = raw.trim().parse::<f64>() {
            if gib > 0.0 {
                return gib;
            }
        }
    }
    match nvidia_smi_total_gib() {
        Some(total) => total * LTX_VAE_BUDGET_SAFE_FRAC,
        None => LTX_VAE_DEFAULT_BUDGET_GIB,
    }
}

/// **Memory-budgeted** tiling for the LTX VAE decode — routes the shared [`budgeted_plan`] selector
/// through the LTX cost model. Caller passes the **output** dims. `Ok(None)` → single-pass already
/// fits; `Err` → a catchable over-budget signal returned before the decode (not an OOM).
pub fn auto_tiling_budgeted_ltx(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    plan_ltx_tiling(height, width, out_frames, ltx_vae_safe_budget_gib())
}

/// Pure LTX tile selector (the `safe_gib` ceiling injected so it is unit-testable without a GPU).
fn plan_ltx_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
) -> Result<Option<TilingConfig>> {
    let candidates = TileCandidates {
        spatial_px: &LTX_VAE_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &LTX_VAE_TEMPORAL_FR,
    };
    budgeted_plan(
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        estimated_ltx_decode_peak_gib,
    )
    .map_err(|e| match e {
        TilingBudgetError::AccumulatorsExceedBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "ltx vae decode: assembling a {width}×{height}×{out_frames} video needs ~{projected_gib:.0} \
             GB just for the output buffers, over the ~{safe_gib:.0} GB safe VRAM budget. Reduce the \
             resolution or frame count."
        )),
        TilingBudgetError::SmallestTileExceedsBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "ltx vae decode: a {width}×{height}×{out_frames} video peaks at ~{projected_gib:.0} GB even \
             with the smallest tile, over the ~{safe_gib:.0} GB safe VRAM budget. Reduce the resolution \
             or frame count."
        )),
    })
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn ltx_tiling_single_pass_when_small() {
        // A short, low-res clip fits a single-pass decode → no tiling.
        assert!(plan_ltx_tiling(256, 256, 25, 60.0).unwrap().is_none());
    }

    #[test]
    fn ltx_tiling_bounds_moderate_res_peak() {
        // 1280×1280×121 single-pass would peak ~66 GB; on a 48 GiB-class budget it must tile and keep
        // the recomputed peak under the safe ceiling (bounded/catchable). Cost constants are the
        // mlx-seeded placeholders — this checks the SELECTOR logic, not the (CUDA-pending) calibration.
        let safe = 48.0 * 0.85;
        let cfg = plan_ltx_tiling(1280, 1280, 121, safe)
            .unwrap()
            .expect("moderate-res LTX must tile");
        let th = cfg
            .spatial
            .map(|s| (s.tile_px as i64).min(1280))
            .unwrap_or(1280);
        let tw = th;
        let tf = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(121))
            .unwrap_or(121);
        let peak = estimated_ltx_decode_peak_gib(121, 1280, 1280, tf, th, tw);
        assert!(peak <= safe, "chosen peak {peak:.1} over safe {safe:.1}");
    }

    #[test]
    fn ltx_tiling_errors_when_unfittable() {
        // 4K × 257 frames under 8 GiB: output accumulators (+ fixed floor) alone blow it → catchable.
        assert!(plan_ltx_tiling(2160, 3840, 257, 8.0).is_err());
    }

    #[test]
    fn ltx_budget_env_override_wins() {
        // The deterministic injection point the worker/tests use. (Set/clear in-process.)
        std::env::set_var("LTX_VAE_BUDGET_GIB", "42.5");
        assert_eq!(ltx_vae_safe_budget_gib(), 42.5);
        std::env::remove_var("LTX_VAE_BUDGET_GIB");
    }
}
