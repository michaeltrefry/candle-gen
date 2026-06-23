//! Qwen3-VL image preprocessing — packed patch pixels + `grid_thw` for [`super::VisionTower`]. Port
//! of `mlx-gen-boogu`'s `vision/preprocess.rs` (single-image Edit path).
//!
//! [`smart_resize`] snaps the image to `(h, w)` divisible by `factor = patch·merge` (32) with the
//! area clamped to `[min_pixels, max_pixels]`; [`preprocess_image`] then BICUBIC-resizes, normalizes
//! with the Qwen3 default `mean = std = 0.5` (→ `[-1, 1]`), and lays the patches out in the
//! **merge-grouped** order the tower (and `get_vision_position_ids`) consume.
//!
//! The resampler is gen-core's PIL-exact [`resize_bicubic_u8`] (the Qwen3-VL processor config's
//! `resample: bicubic`), the same one the candle Qwen-Image-Edit port uses — closer to the real
//! reference than the mlx twin's Catmull-Rom approximation. `grid_thw` is exact integer math either
//! way; only the resampled pixels differ, which a semantic vision tower is insensitive to.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::imageops::resize_bicubic_u8;
use candle_gen::{CandleError, Result};

/// Qwen3-VL image norm (`mllm/preprocessor_config.json`: `image_mean = image_std = 0.5`).
const IMAGE_MEAN: f32 = 0.5;
const IMAGE_STD: f32 = 0.5;

/// Patch geometry (Qwen3-VL vision processor): patch **16** (vs Qwen2.5's 14), temporal 2, merge 2.
pub const PATCH_SIZE: usize = 16;
pub const TEMPORAL_PATCH_SIZE: usize = 2;
pub const MERGE_SIZE: usize = 2;
/// `patch · merge` — the dimension-divisibility factor.
pub const FACTOR: usize = PATCH_SIZE * MERGE_SIZE;

/// Pixel-area bounds (`size = {shortest_edge, longest_edge}` in the processor config).
pub const MIN_PIXELS: usize = 65_536;
pub const MAX_PIXELS: usize = 16_777_216;

/// Qwen-VL `smart_resize`: snap `(height, width)` to multiples of `factor`, keeping aspect ratio
/// while clamping total pixels into `[min_pixels, max_pixels]`. Python round-half-to-even.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let (hf, wf, ff) = (height as f64, width as f64, factor as f64);
    let (minp, maxp) = (min_pixels as f64, max_pixels as f64);
    let mut h_bar = (hf / ff).round_ties_even() * ff;
    let mut w_bar = (wf / ff).round_ties_even() * ff;
    if h_bar * w_bar > maxp {
        let beta = ((hf * wf) / maxp).sqrt();
        h_bar = ff.max((hf / beta / ff).floor() * ff);
        w_bar = ff.max((wf / beta / ff).floor() * ff);
    } else if h_bar * w_bar < minp {
        let beta = (minp / (hf * wf)).sqrt();
        h_bar = (hf * beta / ff).ceil() * ff;
        w_bar = (wf * beta / ff).ceil() * ff;
    }
    (h_bar as usize, w_bar as usize)
}

/// Full Qwen3-VL preprocessing of one RGB8 image (HWC, `[0, 255]`) → `pixel_values [seq, 1536]`
/// (on `device`, f32) + `grid_thw [grid_t, grid_h, grid_w]` (patch units, `grid_t == 1`).
///
/// Layout mirrors the reference's `(grid_t, gh, gw, m, m, C, T, ph, pw)` flatten: the **row** order
/// groups each `merge²` patch block contiguously (the tower's merger relies on it) and the
/// **feature** order is `(channel, temporal, patch_y, patch_x)` (matching the folded Conv3d weight).
pub fn preprocess_image(
    pixels_hwc: &[u8],
    height: usize,
    width: usize,
    device: &Device,
) -> Result<(Tensor, [i32; 3])> {
    if height == 0 || width == 0 {
        return Err(CandleError::Msg(format!(
            "boogu vision: zero dimension ({width}x{height})"
        )));
    }
    if pixels_hwc.len() != height * width * 3 {
        return Err(CandleError::Msg(format!(
            "boogu vision: pixel buffer {} != {width}x{height}x3",
            pixels_hwc.len()
        )));
    }
    let (rh, rw) = smart_resize(height, width, FACTOR, MIN_PIXELS, MAX_PIXELS);

    // Resize on the uint8 image (PIL-exact bicubic) → f32 HWC in [0, 255].
    let resized: Vec<f32> = if (height, width) == (rh, rw) {
        pixels_hwc.iter().map(|&p| p as f32).collect()
    } else {
        resize_bicubic_u8(pixels_hwc, height, width, rh, rw)
    };

    // /255, normalize (x - 0.5)/0.5, laid out as the single frame repeated across the temporal axis.
    let (c, t) = (3usize, TEMPORAL_PATCH_SIZE);
    let plane = rh * rw;
    let mut chw = vec![0f32; t * c * plane];
    for ch in 0..c {
        for y in 0..rh {
            for x in 0..rw {
                let v = (resized[(y * rw + x) * c + ch] / 255.0 - IMAGE_MEAN) / IMAGE_STD;
                let chw_idx = ch * plane + y * rw + x;
                for frame in 0..t {
                    chw[frame * c * plane + chw_idx] = v;
                }
            }
        }
    }

    // Patchify to (gh·gw, c·T·patch²) in the merge-grouped row order.
    let p = PATCH_SIZE;
    let m = MERGE_SIZE;
    let (gh, gw) = (rh / p, rw / p);
    let feat = c * t * p * p; // 1536
    let mut pixel_values = vec![0f32; gh * gw * feat];
    let mut row = 0usize;
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for mr in 0..m {
                for mc in 0..m {
                    let gy = bh * m + mr;
                    let gx = bw * m + mc;
                    let mut f = row * feat;
                    for ch in 0..c {
                        for ft in 0..t {
                            for py in 0..p {
                                for px in 0..p {
                                    let sy = gy * p + py;
                                    let sx = gx * p + px;
                                    pixel_values[f] =
                                        chw[ft * c * plane + ch * plane + sy * rw + sx];
                                    f += 1;
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }

    let pixel_values = Tensor::from_vec(pixel_values, (gh * gw, feat), device)?;
    Ok((pixel_values, [1, gh as i32, gw as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_factor_32() {
        // 512×512 is already a multiple of 32 and within [min,max] → unchanged.
        assert_eq!(
            smart_resize(512, 512, 32, MIN_PIXELS, MAX_PIXELS),
            (512, 512)
        );
    }

    #[test]
    fn preprocess_shapes_qwen3() {
        // 512×512 → grid [1, 32, 32]; seq = 1·32·32 = 1024; feat = 3·2·16·16 = 1536.
        let img = vec![128u8; 512 * 512 * 3];
        let (pv, grid) = preprocess_image(&img, 512, 512, &Device::Cpu).unwrap();
        assert_eq!(grid, [1, 32, 32]);
        assert_eq!(pv.dims(), &[1024, 1536]);
        let v = pv.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn preprocess_rejects_bad_buffer() {
        assert!(preprocess_image(&[0u8; 10], 8, 8, &Device::Cpu).is_err());
    }
}
