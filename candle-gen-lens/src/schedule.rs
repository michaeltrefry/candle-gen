//! Lens sampling schedule + CFG (sc-5114). The schedule is the core flow-match Euler verbatim: the
//! Lens `compute_empirical_mu` is **byte-identical** to gen-core's [`compute_mu`] (same calibrated
//! constants + `>4300` branch), and the Lens `linspace(1, 1/n, n)` → dynamic-shift `set_timesteps`
//! is exactly [`build_flow_sigmas`]. Only two pieces are Lens-specific:
//!
//! 1. **Timestep convention** — Lens feeds the transformer the *shifted sigma* directly (the
//!    reference `timestep / 1000`, where `scheduler.timesteps = shifted_sigma · 1000`), **not** the
//!    `1 − sigma` other DiT families use. [`timesteps`] returns those shifted sigmas.
//! 2. **Norm-rescaled CFG** — [`cfg_rescale`]: `comb = uncond + g·(cond − uncond)`, then rescale
//!    `comb` to carry `cond`'s per-token (channel-axis) L2 norm.
//!
//! The denoise step itself is the core flow-match Euler step ([`euler_step`]). **Lens is the
//! standard-guidance family, NOT true-CFG**: Turbo = 4-step / guidance 1.0 (≈ no CFG), base =
//! 20-step / guidance 5.0.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::gen_core::sampling::{build_flow_sigmas, compute_mu};

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensSamplingDefaults {
    pub num_steps: usize,
    pub guidance_scale: f32,
}

/// `microsoft/Lens-Turbo`: distilled **4 steps, guidance 1.0** (≈ no CFG).
pub const TURBO: LensSamplingDefaults = LensSamplingDefaults {
    num_steps: 4,
    guidance_scale: 1.0,
};
/// `microsoft/Lens` (base): **20 steps, guidance 5.0**.
pub const BASE: LensSamplingDefaults = LensSamplingDefaults {
    num_steps: 20,
    guidance_scale: 5.0,
};

/// Build the Lens flow-match sigma schedule for `num_steps` at the given latent grid (length
/// `num_steps + 1`, descending, trailing `0.0`). The empirical time-shift `mu` is fit from the latent
/// token count `latent_h · latent_w` (== the reference `compute_empirical_mu(seq_len, num_steps)`).
pub fn lens_sigmas(num_steps: usize, latent_h: usize, latent_w: usize) -> Vec<f32> {
    let mu = compute_mu(latent_h * latent_w, num_steps);
    build_flow_sigmas(num_steps, mu)
}

/// The per-step transformer timesteps: the **shifted sigmas** `sigmas[0..num_steps]` (Lens feeds the
/// sigma directly; the reference's `scheduler.timesteps` is these `· 1000`).
pub fn timesteps(sigmas: &[f32]) -> &[f32] {
    &sigmas[..sigmas.len() - 1]
}

/// One flow-match Euler step: `x_{i+1} = x_i + (σ_{i+1} − σ_i)·v` (descending sigmas → negative dt, no
/// velocity negation). The model timestep is the **raw sigma** `σ_i` (see [`timesteps`]).
pub fn euler_step(latents: &Tensor, velocity: &Tensor, sigmas: &[f32], i: usize) -> Result<Tensor> {
    let dt = (sigmas[i + 1] - sigmas[i]) as f64;
    latents + (velocity * dt)?
}

/// Norm-rescaled classifier-free guidance (the reference per-step CFG).
///
/// `cond`/`uncond`: `[B, seq, C]` predictions. Returns `comb · (‖cond‖ / ‖comb‖)` per token
/// (channel-axis L2 norm), with `comb = uncond + g·(cond − uncond)`; where `‖comb‖ = 0` the scale is
/// `1` (matching the reference `torch.where(comb_norm > 0, cond_norm / comb_norm.clamp_min(1e-12), 1)`).
pub fn cfg_rescale(cond: &Tensor, uncond: &Tensor, guidance: f32) -> Result<Tensor> {
    let comb = (uncond + ((cond - uncond)? * guidance as f64)?)?;
    let cond_norm = l2_over_channels(cond)?; // [B, seq, 1]
    let comb_norm = l2_over_channels(&comb)?;
    let ratio = cond_norm.broadcast_div(&comb_norm.maximum(1e-12)?)?;
    let scale = comb_norm
        .gt(0f64)?
        .where_cond(&ratio, &Tensor::ones_like(&comb_norm)?)?;
    comb.broadcast_mul(&scale)
}

/// Per-token L2 norm over the last (channel) axis, keepdim: `sqrt(sum(x², -1))`. Computed in f32 for
/// stability, cast back to `x`'s dtype so it composes with a bf16 denoise loop. No epsilon inside the
/// `sqrt` — the reference uses `torch.norm` and guards the divide separately (see [`cfg_rescale`]).
fn l2_over_channels(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    xf.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn variant_defaults() {
        assert_eq!(TURBO.num_steps, 4);
        assert_eq!(TURBO.guidance_scale, 1.0);
        assert_eq!(BASE.num_steps, 20);
        assert_eq!(BASE.guidance_scale, 5.0);
    }

    #[test]
    fn sigmas_descend_to_zero() {
        for n in [4usize, 20] {
            let s = lens_sigmas(n, 64, 64);
            assert_eq!(s.len(), n + 1, "n={n} length");
            assert_eq!(*s.last().unwrap(), 0.0, "n={n} trailing 0");
            assert!((s[0] - 1.0).abs() < 1e-4, "n={n} start ~1: {}", s[0]);
            assert!(s[..n].windows(2).all(|w| w[0] > w[1]), "n={n} descending");
            // timesteps drop the trailing 0 (the model sees the shifted sigma directly).
            assert_eq!(timesteps(&s).len(), n);
            assert_eq!(timesteps(&s), &s[..n]);
        }
    }

    #[test]
    fn euler_step_is_pure_flow_match() {
        let dev = Device::Cpu;
        let x = Tensor::ones((1, 4, 8), DType::F32, &dev).unwrap();
        let v = Tensor::ones((1, 4, 8), DType::F32, &dev).unwrap();
        let out = euler_step(&x, &v, &[1.0, 0.7, 0.0], 0).unwrap();
        let ov = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for z in ov {
            assert!((z - 0.7).abs() < 1e-6); // 1 + (0.7 - 1.0)·1
        }
    }

    #[test]
    fn cfg_rescale_carries_cond_norm() {
        let dev = Device::Cpu;
        let cond = Tensor::from_vec(
            vec![3.0f32, 4.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0],
            (1, 2, 4),
            &dev,
        )
        .unwrap();
        let uncond = Tensor::from_vec(
            vec![0.5f32, -0.5, 1.0, 0.0, -1.0, 0.0, 0.5, 0.5],
            (1, 2, 4),
            &dev,
        )
        .unwrap();
        let out = cfg_rescale(&cond, &uncond, 2.0).unwrap();
        // Per-token output L2 norm must equal cond's per-token L2 norm (token0: 5, token1: 3).
        let on = l2_over_channels(&out)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((on[0] - 5.0).abs() < 1e-4, "token0 norm {}", on[0]);
        assert!((on[1] - 3.0).abs() < 1e-4, "token1 norm {}", on[1]);
    }
}
