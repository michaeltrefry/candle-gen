//! Flow-matching head, timestep/noise-scale embedders, the FM schedule math, and
//! patchify/unpatchify — the candle port of `mlx-gen-sensenova`'s `fm.rs`.
//!
//! For the 8B-MoT checkpoint (`use_pixel_head=false`, `fm_head_layers=2`) the FM head is a plain
//! `Linear → erf-GELU → Linear`. [`TimestepEmbedder`] (GLIDE sinusoidal → SiLU MLP) backs both the
//! `timestep_embedder` and the `noise_scale_embedder`. The effective time schedule is always the
//! "standard" branch (`σ = shift·σ / (1 + (shift−1)·σ)` with `σ = 1−t`).

use candle_gen::candle_core::{Result as CResult, Tensor};
use candle_gen::candle_nn::{ops, Linear, Module, VarBuilder};
use candle_gen::Result;

use crate::distill::DistillLora;

/// Load a biased Linear from `{prefix}.weight` + `{prefix}.bias` (both present for the FM/timestep
/// modules). Shapeless via `get_unchecked` (the f32 VarBuilder fixes the dtype).
pub(crate) fn load_linear_biased(vb: &VarBuilder, prefix: &str) -> Result<Linear> {
    let w = vb.get_unchecked(&format!("{prefix}.weight"))?;
    let b = vb.get_unchecked(&format!("{prefix}.bias"))?;
    Ok(Linear::new(w, Some(b)))
}

/// The shallow flow-matching head: `Linear → erf-GELU → Linear`. Maps a generation-path hidden
/// state `[…, llm_hidden]` to a patch latent `[…, 3·(patch·merge)²]`.
pub struct FmHead {
    l0: Linear,
    l2: Linear,
}

impl FmHead {
    /// `prefix` = e.g. `"fm_modules.fm_head"` (Sequential indices 0 = first Linear, 2 = second).
    pub fn from_weights(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            l0: load_linear_biased(vb, &format!("{prefix}.0"))?,
            l2: load_linear_biased(vb, &format!("{prefix}.2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let h = self.l0.forward(x)?.gelu_erf()?;
        self.l2.forward(&h)
    }

    /// Merge the distill LoRA (the 8-step `fast` variant) into the two FM-head Linears (`{prefix}.0`
    /// and `{prefix}.2`). Returns the number merged (≤ 2; absent targets are skipped).
    pub fn merge_distill_lora(&mut self, lora: &DistillLora, prefix: &str) -> Result<usize> {
        let mut n = 0;
        if let Some(m) = lora.merge_linear(&self.l0, &format!("{prefix}.0"))? {
            self.l0 = m;
            n += 1;
        }
        if let Some(m) = lora.merge_linear(&self.l2, &format!("{prefix}.2"))? {
            self.l2 = m;
            n += 1;
        }
        Ok(n)
    }
}

/// GLIDE-style sinusoidal timestep embedding → 2-layer SiLU MLP. Backs both the timestep and the
/// noise-scale conditioning (`fm_modules.{timestep_embedder,noise_scale_embedder}`).
pub struct TimestepEmbedder {
    mlp0: Linear,
    mlp2: Linear,
    freq_size: usize,
}

impl TimestepEmbedder {
    /// `prefix` = e.g. `"fm_modules.timestep_embedder"`. `frequency_embedding_size` is 256.
    pub fn from_weights(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            mlp0: load_linear_biased(vb, &format!("{prefix}.mlp.0"))?,
            mlp2: load_linear_biased(vb, &format!("{prefix}.mlp.2"))?,
            freq_size: 256,
        })
    }

    /// Embed scalar timesteps `t` `[N]` → `[N, hidden]`.
    pub fn forward(&self, t: &Tensor) -> CResult<Tensor> {
        let freq = timestep_embedding(t, self.freq_size)?;
        let h = ops::silu(&self.mlp0.forward(&freq)?)?;
        self.mlp2.forward(&h)
    }
}

/// GLIDE sinusoidal embedding: `freqs = exp(-ln(max_period)·arange(half)/half)`, then
/// `cat(cos(t·freqs), sin(t·freqs))`. `dim` is even (256), so no zero-pad branch.
fn timestep_embedding(t: &Tensor, dim: usize) -> CResult<Tensor> {
    const MAX_PERIOD: f64 = 10000.0;
    let half = dim / 2;
    let log_max = MAX_PERIOD.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-log_max * i as f64 / half as f64).exp() as f32)
        .collect();
    let n = t.dim(0)?;
    let t2 = t.reshape((n, 1))?;
    let freqs = Tensor::from_vec(freqs, (1, half), t.device())?;
    let args = t2.matmul(&freqs)?; // [N, half]
    Tensor::cat(&[&args.cos()?, &args.sin()?], 1)
}

/// The flow-matching time schedule (always the standard branch): `σ = 1−t`,
/// `σ ← shift·σ / (1 + (shift−1)·σ)`, return `1−σ`. Elementwise over `t`.
pub fn apply_time_schedule(t: &Tensor, shift: f32) -> CResult<Tensor> {
    let sigma = t.affine(-1.0, 1.0)?; // 1 - t
    let num = (&sigma * shift as f64)?;
    let denom = sigma.affine((shift - 1.0) as f64, 1.0)?; // 1 + (shift-1)·σ
    let sigma = (num / denom)?;
    sigma.affine(-1.0, 1.0) // 1 - σ
}

/// One forward-Euler step: `z + (t_next − t)·v_pred`.
pub fn euler_step(v_pred: &Tensor, z: &Tensor, t: f32, t_next: f32) -> CResult<Tensor> {
    z + (v_pred * (t_next - t) as f64)?
}

/// Flow-matching velocity: `(x_pred − z) / max(1 − t, t_eps)`.
pub fn velocity(x_pred: &Tensor, z: &Tensor, t: f32, t_eps: f32) -> CResult<Tensor> {
    let denom = (1.0 - t).max(t_eps) as f64;
    (x_pred - z)? / denom
}

/// `images` `[N,3,H,W]` → patches `[N, (H/ps)·(W/ps), ps²·3]` (channel-last patch layout, matching
/// the reference `patchify(..., channel_first=False)`: `nchpwq → nhwpqc`).
pub fn patchify(images: &Tensor, patch_size: usize) -> CResult<Tensor> {
    let (n, _c, h_pix, w_pix) = images.dims4()?;
    let (h, w) = (h_pix / patch_size, w_pix / patch_size);
    images
        .reshape((n, 3, h, patch_size, w, patch_size))?
        .permute((0, 2, 4, 3, 5, 1))? // nchpwq -> nhwpqc
        .contiguous()?
        .reshape((n, h * w, patch_size * patch_size * 3))
}

/// Channel-**first** patchify: `nchpwq → nhwcpq` — the layout the gen-path vision embedder expects
/// (its `patch_embedding` conv weight flattens to `[embed, ch·ps·ps]` in the same `c,ph,pw` order).
pub fn patchify_channel_first(images: &Tensor, patch_size: usize) -> CResult<Tensor> {
    let (n, _c, h_pix, w_pix) = images.dims4()?;
    let (h, w) = (h_pix / patch_size, w_pix / patch_size);
    images
        .reshape((n, 3, h, patch_size, w, patch_size))?
        .permute((0, 2, 4, 1, 3, 5))? // nchpwq -> nhwcpq
        .contiguous()?
        .reshape((n, h * w, 3 * patch_size * patch_size))
}

/// Inverse of [`patchify`]: patches `[N,L,ps²·3]` → `[N,3,H,W]` (`nhwpqc → nchpwq`), with `h`/`w`
/// the token-grid dims.
pub fn unpatchify(x: &Tensor, patch_size: usize, h: usize, w: usize) -> CResult<Tensor> {
    let n = x.dim(0)?;
    x.reshape((n, h, w, patch_size, patch_size, 3))?
        .permute((0, 5, 1, 3, 2, 4))? // nhwpqc -> nchpwq
        .contiguous()?
        .reshape((n, 3, h * patch_size, w * patch_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn patchify_unpatchify_roundtrip() {
        // A [1,3,4,4] image with ps=2 → 4 patches of 12 → back to [1,3,4,4].
        let dev = Device::Cpu;
        let data: Vec<f32> = (0..(3 * 4 * 4)).map(|i| i as f32).collect();
        let img = Tensor::from_vec(data, (1, 3, 4, 4), &dev).unwrap();
        let patches = patchify(&img, 2).unwrap();
        assert_eq!(patches.dims(), &[1, 4, 12]); // (4/2)·(4/2)=4 patches, 2·2·3=12
        let back = unpatchify(&patches, 2, 2, 2).unwrap();
        assert_eq!(back.dims(), img.dims());
        let a = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "patchify∘unpatchify is identity");
    }

    #[test]
    fn time_schedule_matches_formula() {
        let dev = Device::Cpu;
        let t = Tensor::from_vec(vec![0.0f32, 0.25, 0.5, 0.75, 1.0], (5,), &dev).unwrap();
        let shift = 3.0f32;
        let got = apply_time_schedule(&t, shift)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Reference: σ=1−t; σ←shift·σ/(1+(shift−1)·σ); return 1−σ.
        let expected: Vec<f32> = [0.0f32, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|&t| {
                let sigma = 1.0 - t;
                let sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma);
                1.0 - sigma
            })
            .collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "got {g} expected {e}");
        }
        // Endpoints are fixed (t=0 → 0, t=1 → 1).
        assert!((got[0] - 0.0).abs() < 1e-6);
        assert!((got[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn euler_step_is_linear_extrapolation() {
        let dev = Device::Cpu;
        let v = Tensor::from_vec(vec![2.0f32, 4.0], (2,), &dev).unwrap();
        let z = Tensor::from_vec(vec![1.0f32, 1.0], (2,), &dev).unwrap();
        // z + (0.5)·v = [2, 3]
        let out = euler_step(&v, &z, 0.0, 0.5)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, vec![2.0, 3.0]);
    }
}
