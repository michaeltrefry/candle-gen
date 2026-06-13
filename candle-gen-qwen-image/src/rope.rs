//! Qwen-Image's **3-axis (frame, height, width) interleaved RoPE** for the MMDiT. Port of
//! `mlx-gen-qwen-image`'s `transformer/rope.rs`. Each axis (`axes_dim = [16, 56, 56]`) contributes
//! `dim/2` frequencies → `8 + 28 + 28 = 64` per token (= `head_dim/2`). θ = 10000, `scale_rope` (the
//! height/width positions are centered). Frequencies and angles are computed **host-side** in f32.
//!
//! - **Image tokens** at grid `(h, w)`: the frame axis uses position 0 (single image), height/width
//!   use **centered** positions `h - (latent_h - latent_h/2)` / `w - (latent_w - latent_w/2)`.
//! - **Text tokens** at index `t`: a single scalar position `txt_base + t`
//!   (`txt_base = max(latent_h/2, latent_w/2)`) applied across **all 64** frequencies.
//!
//! Application is **interleaved** (lanes `2i`/`2i+1` are the real/imag pair), via candle's `rope_i`.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::rotary_emb::rope_i;

use crate::config::TransformerConfig;

pub struct QwenRope {
    theta: f32,
    axes_dim: [usize; 3],
    half: usize,
}

impl QwenRope {
    pub fn new(cfg: &TransformerConfig) -> Self {
        Self {
            theta: cfg.rope_theta,
            axes_dim: cfg.axes_dim,
            half: cfg.axes_dim.iter().sum::<usize>() / 2,
        }
    }

    /// The 64-wide concatenated frequency vector `[ω_frame(8), ω_h(28), ω_w(28)]`,
    /// `ω_d[k] = theta^{-(2k)/d}`.
    fn omega(&self) -> Vec<f32> {
        let mut all = Vec::with_capacity(self.half);
        for &dim in &self.axes_dim {
            for k in 0..dim / 2 {
                all.push(1.0f32 / self.theta.powf((2 * k) as f32 / dim as f32));
            }
        }
        all
    }

    /// Image-token `(cos, sin)` `[lat_h·lat_w, 64]` (row-major over the grid).
    pub fn img_cos_sin(
        &self,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let omega = self.omega();
        let (n_f, n_h) = (self.axes_dim[0] / 2, self.axes_dim[1] / 2); // 8, 28
        let h_center = (lat_h - lat_h / 2) as i64;
        let w_center = (lat_w - lat_w / 2) as i64;
        let seq = lat_h * lat_w;
        let mut cos = vec![0f32; seq * self.half];
        let mut sin = vec![0f32; seq * self.half];
        for h in 0..lat_h {
            for w in 0..lat_w {
                let row = h * lat_w + w;
                let hp = h as i64 - h_center;
                let wp = w as i64 - w_center;
                for (j, &om) in omega.iter().enumerate() {
                    // axis position by frequency band: frame [0,8) → 0, height [8,36) → hp, width [36,64) → wp.
                    let pos = if j < n_f {
                        0i64
                    } else if j < n_f + n_h {
                        hp
                    } else {
                        wp
                    } as f32;
                    let a = pos * om;
                    cos[row * self.half + j] = a.cos();
                    sin[row * self.half + j] = a.sin();
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, (seq, self.half), device)?,
            Tensor::from_vec(sin, (seq, self.half), device)?,
        ))
    }

    /// Text-token `(cos, sin)` `[txt_seq, 64]`: scalar position `txt_base + t` across all freqs.
    pub fn txt_cos_sin(
        &self,
        txt_seq: usize,
        lat_h: usize,
        lat_w: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let omega = self.omega();
        let txt_base = (lat_h / 2).max(lat_w / 2) as i64;
        let mut cos = vec![0f32; txt_seq * self.half];
        let mut sin = vec![0f32; txt_seq * self.half];
        for t in 0..txt_seq {
            let pos = (txt_base + t as i64) as f32;
            for (j, &om) in omega.iter().enumerate() {
                let a = pos * om;
                cos[t * self.half + j] = a.cos();
                sin[t * self.half + j] = a.sin();
            }
        }
        Ok((
            Tensor::from_vec(cos, (txt_seq, self.half), device)?,
            Tensor::from_vec(sin, (txt_seq, self.half), device)?,
        ))
    }
}

/// Apply interleaved RoPE to `x` `[B, H, S, head_dim]` with `cos`/`sin` `[S, head_dim/2]`.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    rope_i(&xf, cos, sin)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omega_is_64_wide_with_band_layout() {
        let r = QwenRope::new(&TransformerConfig::qwen_image());
        assert_eq!(r.half, 64);
        let om = r.omega();
        assert_eq!(om.len(), 64);
        // First freq of each band is theta^0 = 1.
        assert!((om[0] - 1.0).abs() < 1e-6, "frame band base");
        assert!((om[8] - 1.0).abs() < 1e-6, "height band base");
        assert!((om[36] - 1.0).abs() < 1e-6, "width band base");
    }

    #[test]
    fn img_frame_axis_is_zero_angle() {
        let r = QwenRope::new(&TransformerConfig::qwen_image());
        let (cos, sin) = r.img_cos_sin(4, 4, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[16, 64]);
        // Frame band (first 8 lanes) has position 0 → cos 1, sin 0 for every token.
        let cv = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sv = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for tok in 0..16 {
            for j in 0..8 {
                assert!((cv[tok * 64 + j] - 1.0).abs() < 1e-6);
                assert!(sv[tok * 64 + j].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn apply_rope_at_zero_is_identity() {
        let r = QwenRope::new(&TransformerConfig::qwen_image());
        // txt position 0 with lat 2x2 → txt_base = max(1,1)=1, so not zero; build an explicit zero table.
        let cos = Tensor::ones((3, 64), DType::F32, &Device::Cpu).unwrap();
        let sin = Tensor::zeros((3, 64), DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::arange(0f32, 3.0 * 128.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 3, 128))
            .unwrap();
        let y = apply_rope(&x, &cos, &sin).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-5);
        }
        let _ = &r;
    }
}
