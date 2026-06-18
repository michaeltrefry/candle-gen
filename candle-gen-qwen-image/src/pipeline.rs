//! Qwen-Image latent geometry, the dynamic-μ flow-match-Euler schedule, and the true-CFG combine.
//! Port of `mlx-gen-qwen-image`'s `pipeline.rs` + `sampler.rs` (txt2img path). All weight-free and
//! unit-tested on CPU.
//!
//! Geometry: an `W×H` image → VAE latent `[1, 16, H/8, W/8]` → 2×2 patchify → packed token sequence
//! `[1, (H/16)·(W/16), 64]`. txt2img samples noise directly in the packed 64-ch space.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::config::{LATENT_CHANNELS, PATCH};

// Dynamic-μ shift endpoints (image-area driven), from the fork's `qwen_scheduler`.
const SIGMA_BASE_SHIFT: f32 = 0.5;
const SIGMA_MAX_SHIFT: f32 = 0.9;
const SIGMA_BASE_SEQ_LEN: f32 = 256.0;
const SIGMA_MAX_SEQ_LEN: f32 = 8192.0;
const SIGMA_SHIFT_TERMINAL: f32 = 0.02;

/// Packed token grid `(lat_h, lat_w) = (H/16, W/16)`.
pub fn latent_dims(width: u32, height: u32) -> (usize, usize) {
    ((height / 16) as usize, (width / 16) as usize)
}

/// Deterministic packed initial noise `[1, seq, 64]` (sc-3673 parity): N(0,1) from a fixed CPU RNG.
pub fn create_noise(seed: u64, width: u32, height: u32, device: &Device) -> Result<Tensor> {
    let (lat_h, lat_w) = latent_dims(width, height);
    let seq = lat_h * lat_w;
    let feat = LATENT_CHANNELS * PATCH * PATCH; // 64
    let n = seq * feat;
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Tensor::from_vec(data, (1, seq, feat), &Device::Cpu)?.to_device(device)
}

/// Unpack packed latents `[1, seq, 64]` → VAE latent `[1, 16, H/8, W/8]` (the 2×2 patchify inverse).
pub fn unpack_latents(packed: &Tensor, width: u32, height: u32) -> Result<Tensor> {
    let (lat_h, lat_w) = latent_dims(width, height);
    let c = LATENT_CHANNELS;
    let p = PATCH;
    // [1, h/16, w/16, 16, 2, 2] -> [1, 16, h/16, 2, w/16, 2] -> [1, 16, h/8, w/8]
    packed
        .reshape((1, lat_h, lat_w, c, p, p))?
        .permute((0, 3, 1, 4, 2, 5))?
        .reshape((1, c, lat_h * p, lat_w * p))?
        .contiguous()
}

/// Pack a VAE latent `[1, 16, H/8, W/8]` → the packed token sequence `[1, seq, 64]` (the 2×2 patchify;
/// the inverse of [`unpack_latents`]). Used to feed a VAE-encoded control image to the ControlNet
/// branch (sc-5489) in the same packed space the noise lives in.
pub fn pack_latents(latent: &Tensor, width: u32, height: u32) -> Result<Tensor> {
    let (lat_h, lat_w) = latent_dims(width, height);
    let c = LATENT_CHANNELS;
    let p = PATCH;
    // [1, 16, h/8, w/8] -> [1, 16, lat_h, 2, lat_w, 2] -> [1, lat_h, lat_w, 16, 2, 2] -> [1, seq, 64]
    latent
        .reshape((1, c, lat_h, p, lat_w, p))?
        .permute((0, 2, 4, 1, 3, 5))?
        .contiguous()?
        .reshape((1, lat_h * lat_w, c * p * p))
}

/// The Qwen-Image sigma schedule (length `steps + 1`, descending to 0): a linspace `1 → 1/n` warped
/// by an image-area-driven exponential μ shift, then rescaled so the terminal one-minus-σ hits
/// `1 − 0.02`, then a trailing `0.0`.
pub fn qwen_sigmas(num_steps: usize, width: u32, height: u32) -> Vec<f32> {
    let n = num_steps.max(1);
    let nf = n as f32;
    let (start, end) = (1.0f32, 1.0f32 / nf);
    let linspace: Vec<f32> = (0..n)
        .map(|i| {
            if n == 1 {
                start
            } else {
                start + (end - start) * (i as f32) / (nf - 1.0)
            }
        })
        .collect();
    let m = (SIGMA_MAX_SHIFT - SIGMA_BASE_SHIFT) / (SIGMA_MAX_SEQ_LEN - SIGMA_BASE_SEQ_LEN);
    let b = SIGMA_BASE_SHIFT - m * SIGMA_BASE_SEQ_LEN;
    let mu = m * (width as f32 * height as f32 / 256.0) + b;
    let e = mu.exp();
    let mut shifted: Vec<f32> = linspace
        .iter()
        .map(|&s| e / (e + (1.0 / s - 1.0)))
        .collect();
    // Terminal-sigma rescale: map the last 1−σ to 1 − 0.02.
    let one_minus: Vec<f32> = shifted.iter().map(|&s| 1.0 - s).collect();
    let scale = one_minus[n - 1] / (1.0 - SIGMA_SHIFT_TERMINAL);
    for (s, om) in shifted.iter_mut().zip(one_minus) {
        *s = 1.0 - om / scale;
    }
    shifted.push(0.0);
    shifted
}

/// The official lightx2v Qwen-Image-Lightning flow-match shift (`exp(μ)`, μ = `ln 3`): the model card
/// sets `base_shift = max_shift = ln 3`, so the per-resolution dynamic shift collapses to this constant.
const LIGHTNING_SHIFT: f32 = 3.0;
/// Flow-match training timesteps (diffusers `num_train_timesteps`): the Lightning span runs down to
/// `1/1000`, the full diffusers minimum (NOT the production schedule's `1/n` floor).
const LIGHTNING_NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// The few-step **Lightning** sigma schedule (length `steps + 1`, descending to 0) for the
/// Qwen-Image-Edit-2511-Lightning distill (sc-6220) — the candle twin of `mlx-gen-qwen-image`'s
/// `sampler::lightning`. Reproduces diffusers' `FlowMatchEulerDiscreteScheduler` under the lightx2v
/// Lightning config: a **static** flow-match shift of 3.0 (`base_shift == max_shift`) over
/// `linspace(1, 1/1000, n)`, with **no terminal rescale** (unlike [`qwen_sigmas`]'s 0.02). Resolution-
/// independent. 4-step → `[1.0, 0.857.., 0.601.., 0.00299.., 0.0]`. Drives the same [`euler_step`].
pub fn lightning_sigmas(num_steps: usize) -> Vec<f32> {
    let n = num_steps.max(1);
    let e = LIGHTNING_SHIFT; // exp(μ)
    let sigma_min = 1.0 / LIGHTNING_NUM_TRAIN_TIMESTEPS;
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            // linspace(1.0, sigma_min, n)
            let s = if n == 1 {
                1.0
            } else {
                1.0 + (sigma_min - 1.0) * (i as f32) / ((n - 1) as f32)
            };
            e / (e + (1.0 / s - 1.0))
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// One flow-match Euler step: `x_{i+1} = x_i + (σ_{i+1} − σ_i)·v` (descending sigmas, negative dt, no
/// velocity negation). The model timestep is the **raw sigma** `σ_i`.
pub fn euler_step(latents: &Tensor, velocity: &Tensor, sigmas: &[f32], i: usize) -> Result<Tensor> {
    let dt = (sigmas[i + 1] - sigmas[i]) as f64;
    latents + (velocity * dt)?
}

/// True-CFG combine with norm correction: `combined = neg + g·(pos − neg)`, then rescale `combined`
/// to the per-token channel L2 norm of `pos`. Shapes `[1, seq, 64]`.
pub fn compute_guided_noise(pos: &Tensor, neg: &Tensor, guidance: f32) -> Result<Tensor> {
    let combined = (neg + ((pos - neg)? * guidance as f64)?)?;
    let cond_norm = l2_over_channels(pos)?;
    let comb_norm = l2_over_channels(&combined)?;
    combined.broadcast_mul(&cond_norm.broadcast_div(&comb_norm)?)
}

/// Per-token L2 norm over the last (channel) axis, keepdim: `sqrt(sum(x², -1) + 1e-12)`. Computed in
/// f32 for stability, cast back to `x`'s dtype so it composes with a bf16 denoise loop.
fn l2_over_channels(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    ((xf.sqr()?.sum_keepdim(D::Minus1)? + 1e-12)?.sqrt()?).to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry() {
        assert_eq!(latent_dims(1024, 1024), (64, 64));
        let noise = create_noise(7, 256, 256, &Device::Cpu).unwrap();
        assert_eq!(noise.dims(), &[1, 256, 64]); // (256/16)^2 = 256
    }

    #[test]
    fn noise_is_deterministic() {
        let a = create_noise(7, 256, 256, &Device::Cpu).unwrap();
        let b = create_noise(7, 256, 256, &Device::Cpu).unwrap();
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(av, bv);
    }

    #[test]
    fn unpack_shape() {
        let packed = create_noise(1, 256, 256, &Device::Cpu).unwrap();
        let un = unpack_latents(&packed, 256, 256).unwrap();
        assert_eq!(un.dims(), &[1, 16, 32, 32]); // H/8 = 256/8
    }

    #[test]
    fn pack_is_unpack_inverse() {
        // pack(unpack(x)) == x: the control encode-path packing must round-trip the noise packing.
        let packed = create_noise(3, 256, 384, &Device::Cpu).unwrap();
        let latent = unpack_latents(&packed, 256, 384).unwrap();
        assert_eq!(latent.dims(), &[1, 16, 48, 32]); // (H/8, W/8) = (384/8, 256/8)
        let repacked = pack_latents(&latent, 256, 384).unwrap();
        assert_eq!(repacked.dims(), packed.dims());
        let a = packed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = repacked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "pack∘unpack must be identity");
    }

    #[test]
    fn sigmas_descend_to_zero_with_terminal() {
        let s = qwen_sigmas(20, 1024, 1024);
        assert_eq!(s.len(), 21);
        assert!((s[0] - 1.0).abs() < 1e-4, "start ~1: {}", s[0]);
        assert!(s[20].abs() < 1e-9, "trailing 0");
        // terminal (pre-0) one-minus-sigma ~ 1 - 0.02 = 0.98 → sigma ~0.02.
        assert!(
            (s[19] - 0.02).abs() < 1e-3,
            "terminal sigma ~0.02: {}",
            s[19]
        );
        for w in &s[..20].windows(2).collect::<Vec<_>>() {
            assert!(w[0] > w[1], "descending: {s:?}");
        }
    }

    #[test]
    fn lightning_sigmas_match_diffusers_golden() {
        // The official lightx2v 4-step recipe (shift 3.0, no terminal rescale) over linspace(1, 1/1000,
        // 4) — the bit-exact diffusers values pinned in mlx-gen's `sampler::lightning` test.
        let s = lightning_sigmas(4);
        assert_eq!(s.len(), 5);
        let expected = [1.0_f32, 0.857_326_5, 0.600_719_4, 0.002_994_012, 0.0];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (s[i] - want).abs() < 1e-5,
                "lightning sigma[{i}] = {} want {want}",
                s[i]
            );
        }
        // No terminal rescale: the last pre-0 sigma runs to the diffusers 1/1000 floor (≈0.003), NOT
        // the production schedule's 0.02 — the whole difference from `qwen_sigmas`.
        assert!(s[3] < 0.01);
        // Resolution-independent: identical regardless of size, unlike the production schedule.
        assert_eq!(lightning_sigmas(4), s);
    }

    #[test]
    fn euler_step_descending() {
        let dev = Device::Cpu;
        let x = Tensor::ones((1, 4, 64), DType::F32, &dev).unwrap();
        let v = Tensor::ones((1, 4, 64), DType::F32, &dev).unwrap();
        let out = euler_step(&x, &v, &[1.0, 0.7, 0.0], 0).unwrap();
        let ov = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for z in ov {
            assert!((z - 0.7).abs() < 1e-6); // 1 + (0.7-1.0)*1
        }
    }

    #[test]
    fn cfg_combine_rescales_to_pos_norm() {
        let dev = Device::Cpu;
        // pos has channel-L2 norm sqrt(64*4)=16 per token; combined gets rescaled to match.
        let pos = (Tensor::ones((1, 2, 64), DType::F32, &dev).unwrap() * 2.0).unwrap();
        let neg = Tensor::zeros((1, 2, 64), DType::F32, &dev).unwrap();
        let out = compute_guided_noise(&pos, &neg, 4.0).unwrap();
        // combined = 0 + 4*(2-0) = 8 everywhere; rescaled to pos norm → back to 2 everywhere.
        let ov = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for z in ov {
            assert!((z - 2.0).abs() < 1e-4, "got {z}");
        }
    }
}
