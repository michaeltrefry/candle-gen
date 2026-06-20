//! `F.interpolate`-faithful resize for SCAIL-2 image preprocessing.
//!
//! SCAIL-2's conditioning path uses **`torch.nn.functional.interpolate(..., align_corners=False)`**:
//! the CLIP image encode resizes to 224² with `mode='bicubic'` (upstream `CLIPModel.visual`), and the
//! pose video + driving mask are 0.5×-downsampled with `mode='bilinear'`. These are PyTorch's own
//! kernels (bicubic Keys **a = −0.75**, 4-tap, *no* antialias; bilinear 2-tap). Separable, accumulated
//! in f64, host-side (the conditioning tensors are small and this is precompute-friendly).

use candle_gen::candle_core::{DType, Result, Tensor};

/// CLIP / open-CLIP image normalization (matches the EVA constants).
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// `F.interpolate` mode.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    Bicubic,
    Bilinear,
}

fn cubic1(x: f64, a: f64) -> f64 {
    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}
fn cubic2(x: f64, a: f64) -> f64 {
    ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
}

/// Per-output-pixel `(clamped source index, weight)` taps for one axis, matching PyTorch
/// `area_pixel_compute_source_index(align_corners=false)` + the bicubic/bilinear tap weights.
fn axis_taps(in_size: usize, out_size: usize, mode: Interp) -> Vec<Vec<(usize, f64)>> {
    let scale = in_size as f64 / out_size as f64;
    let clamp = |i: i64| -> usize { i.clamp(0, in_size as i64 - 1) as usize };
    (0..out_size)
        .map(|ox| {
            match mode {
                Interp::Bicubic => {
                    // cubic=true → source index is NOT clamped to ≥0.
                    let real = scale * (ox as f64 + 0.5) - 0.5;
                    let base = real.floor();
                    let t = real - base;
                    let a = -0.75;
                    let w = [
                        cubic2(t + 1.0, a),
                        cubic1(t, a),
                        cubic1(1.0 - t, a),
                        cubic2(2.0 - t, a),
                    ];
                    (0..4)
                        .map(|k| (clamp(base as i64 - 1 + k as i64), w[k]))
                        .collect()
                }
                Interp::Bilinear => {
                    // cubic=false → source index clamped to ≥0.
                    let real = (scale * (ox as f64 + 0.5) - 0.5).max(0.0);
                    let x0 = real.floor();
                    let t = real - x0;
                    vec![(clamp(x0 as i64), 1.0 - t), (clamp(x0 as i64 + 1), t)]
                }
            }
        })
        .collect()
}

/// Separable resize of a contiguous `[N, C, H, W]` f32 buffer → `[N, C, out_h, out_w]`.
fn resize_nchw(
    src: &[f32],
    dims: (usize, usize, usize, usize),
    out_dims: (usize, usize),
    mode: Interp,
) -> Vec<f32> {
    let (n, c, ih, iw) = dims;
    let (oh, ow) = out_dims;
    let htaps = axis_taps(iw, ow, mode); // horizontal: iw → ow
    let vtaps = axis_taps(ih, oh, mode); // vertical: ih → oh
    let mut out = vec![0f32; n * c * oh * ow];
    for plane in 0..(n * c) {
        let s = &src[plane * ih * iw..(plane + 1) * ih * iw];
        // Horizontal pass: (ih, iw) → (ih, ow).
        let mut horiz = vec![0f32; ih * ow];
        for y in 0..ih {
            for (ox, taps) in htaps.iter().enumerate() {
                let mut acc = 0f64;
                for &(xi, w) in taps {
                    acc += s[y * iw + xi] as f64 * w;
                }
                horiz[y * ow + ox] = acc as f32;
            }
        }
        // Vertical pass: (ih, ow) → (oh, ow).
        let dst = &mut out[plane * oh * ow..(plane + 1) * oh * ow];
        for (oy, taps) in vtaps.iter().enumerate() {
            for x in 0..ow {
                let mut acc = 0f64;
                for &(yi, w) in taps {
                    acc += horiz[yi * ow + x] as f64 * w;
                }
                dst[oy * ow + x] = acc as f32;
            }
        }
    }
    out
}

/// `F.interpolate(x, size=(out_h, out_w), mode, align_corners=False)` for an `[N, C, H, W]` image.
pub fn interpolate(x: &Tensor, out_h: usize, out_w: usize, mode: Interp) -> Result<Tensor> {
    let (n, c, ih, iw) = x.dims4()?;
    let src: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let out = resize_nchw(&src, (n, c, ih, iw), (out_h, out_w), mode);
    Tensor::from_vec(out, (n, c, out_h, out_w), x.device())
}

/// `F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)` (pose video / driving
/// mask half-resolution), `[N, C, H, W] → [N, C, H/2, W/2]`.
pub fn downsample_half(x: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    interpolate(x, h / 2, w / 2, Interp::Bilinear)
}

/// CLIP image preprocessing (upstream `CLIPModel.visual`): bicubic-resize an `[B, 3, H, W]` image in
/// `[-1, 1]` to `size²`, map to `[0, 1]` (`·0.5 + 0.5`), and CLIP mean/std normalize → `[B, 3, size,
/// size]` ready for [`crate::clip::ScailClip::encode`].
pub fn clip_preprocess(image: &Tensor, size: usize) -> Result<Tensor> {
    let resized = interpolate(image, size, size, Interp::Bicubic)?;
    // [-1,1] → [0,1]
    let zero_one = resized.affine(0.5, 0.5)?;
    let dev = image.device();
    let mean = Tensor::from_vec(CLIP_MEAN.to_vec(), (1, 3, 1, 1), dev)?;
    let std = Tensor::from_vec(CLIP_STD.to_vec(), (1, 3, 1, 1), dev)?;
    zero_one.broadcast_sub(&mean)?.broadcast_div(&std)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn bilinear_half_downsample_averages_2x2() {
        // A 1×1×2×2 block [[1,3],[5,7]] half-downsampled (align_corners=False) samples the single
        // output pixel at source (0.5,0.5) → the 2×2 mean = 4.
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 3.0, 5.0, 7.0], (1, 1, 2, 2), &dev).unwrap();
        let y = downsample_half(&x).unwrap();
        assert_eq!(y.dims(), &[1, 1, 1, 1]);
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((v - 4.0).abs() < 1e-4, "got {v}");
    }

    #[test]
    fn bicubic_identity_size_is_near_identity() {
        // Resizing to the same size leaves a smooth ramp essentially unchanged at interior pixels.
        let dev = Device::Cpu;
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = Tensor::from_vec(data.clone(), (1, 1, 4, 4), &dev).unwrap();
        let y = interpolate(&x, 4, 4, Interp::Bicubic).unwrap();
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[5] - data[5]).abs() < 1e-3, "interior pixel changed: {}", v[5]);
    }
}
