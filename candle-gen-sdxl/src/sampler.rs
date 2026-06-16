//! Euler / Euler-ancestral sampler for the candle SDXL denoise (sc-5491, epic 5480) — the candle
//! port of `mlx-gen-sdxl::sampler::EulerSampler` (the vendored `SimpleEulerAncestralSampler`).
//!
//! **Why a new sampler here.** candle-gen-sdxl's txt2img path (`pipeline`) runs candle-transformers'
//! DDIM — deterministic, *non-ancestral*. InstantID's reference pipeline (diffusers SDXL) denoises
//! with **Euler-ancestral**, and its identity/control conditioning was tuned against that solver, so
//! the InstantID denoise loop (2e) needs it. This is also the SDXL building block the broader
//! IP-Adapter / ControlNet slices (sc-5488 / sc-5489) reuse.
//!
//! **Two deliberate divergences from the MLX version**, both correct for the candle lane:
//!  1. **Determinism is a seeded CPU `StdRng`, not MLX's global RNG.** The mlx version draws the
//!     prior and each ancestral step's noise from MLX's process-global RNG (seeded once) to reproduce
//!     the vendored Python noise *stream*. The candle lane's contract (sc-3673) is instead "generation
//!     is a pure function of `(seed, request)`, launch-portable" — its txt2img already draws noise
//!     from a fixed-algorithm `StdRng` on CPU. So this sampler does **not** own an RNG or draw noise:
//!     [`EulerAncestralSampler::step`] takes the ancestral noise as an injected tensor, and the
//!     denoise loop (2e) owns the one `StdRng` it draws the prior + every step's noise from. That
//!     keeps `step` a pure function (CPU-testable against a host reference) and the RNG ownership in
//!     one place.
//!  2. **Scalar schedule math runs in host `f64`, not MLX ops + f16 rounding.** The mlx version routes
//!     `σ(t)`, `σ_up`, `σ_down` through MLX ops and rounds timesteps to f16 *solely* to be bit-exact
//!     to the vendored Python `mlx_sd` (a 1-ULP drift seeds the chaotic ancestral trajectory and
//!     breaks parity). candle has no Python reference to match ULP-for-ULP — its standard is a
//!     correct, deterministic trajectory — so the schedule scalars are computed in clean host f64
//!     (strictly *more* accurate) and applied to the latents via scalar ops in the latent dtype,
//!     exactly as the reference's per-step tensor math runs in the eps dtype.
//!
//! The math itself is a faithful port: the `scaled_linear` β schedule → `alphas_cumprod` → the
//! 1001-entry σ table (a leading 0 + the 1000 training-step σ), the linear σ(t) interp, the prior
//! `noise·σ_last·rsqrt(σ_last²+1)`, and the ancestral step `σ_up = sqrt(σ_prev²·(σ²−σ_prev²)/σ²)`,
//! `σ_down = sqrt(σ_prev² − σ_up²)`, `x' = (sqrt(σ²+1)·x_t + eps·(σ_down−σ) + noise·σ_up)·rsqrt(σ_prev²+1)`.

use candle_core::Tensor;

use candle_gen::{CandleError, Result};

/// SDXL's `scaled_linear` β endpoints + train-step count (the diffusers SDXL scheduler defaults,
/// also what candle-transformers' `DDIMSchedulerConfig::default()` carries). The σ table is built
/// from these.
const SDXL_BETA_START: f64 = 0.00085;
const SDXL_BETA_END: f64 = 0.012;
const SDXL_TRAIN_STEPS: usize = 1000;

/// A discrete Euler / Euler-ancestral sampler over a precomputed σ table. SDXL uses the **ancestral**
/// variant; the plain Euler step is kept for completeness (and matches the mlx `ancestral=false` path).
pub struct EulerAncestralSampler {
    /// `[0, σ_1, …, σ_T]` (length `train_steps + 1`), host f64.
    sigmas: Vec<f64>,
    /// Ancestral (SDXL default, injects per-step noise) vs plain Euler (deterministic).
    ancestral: bool,
}

impl EulerAncestralSampler {
    /// The production SDXL ancestral sampler (`scaled_linear` β 0.00085→0.012, 1000 train steps).
    pub fn sdxl() -> Self {
        // The table is built from constants that cannot fail; `new` only errors on an empty schedule.
        Self::new(SDXL_TRAIN_STEPS, SDXL_BETA_START, SDXL_BETA_END, true)
            .expect("sdxl sampler: nonzero train steps")
    }

    /// Build a sampler from the `scaled_linear` β schedule. `ancestral` selects the
    /// Euler-ancestral step (SDXL) vs the plain Euler step. The σ table is computed in host f64.
    ///
    /// `_linspace(a, b, n) = arange(n)/(n−1)·(b−a) + a`, with the `scaled_linear` `**0.5`/`**2`
    /// taken around it (matching the vendored reference), then `acp = cumprod(1−β)` and
    /// `σ = concat([0], sqrt((1−acp)/acp))`.
    pub fn new(
        train_steps: usize,
        beta_start: f64,
        beta_end: f64,
        ancestral: bool,
    ) -> Result<Self> {
        if train_steps == 0 {
            return Err(CandleError::Msg(
                "euler sampler: train_steps must be >= 1".into(),
            ));
        }
        let n = train_steps;
        let (a, b) = (beta_start.sqrt(), beta_end.sqrt());
        let mut acp = 1.0f64;
        let mut sigmas = Vec::with_capacity(n + 1);
        sigmas.push(0.0); // the leading 0 (σ at t=0)
        for i in 0..n {
            let x = i as f64 / (n - 1) as f64;
            let beta = {
                let lin = a + x * (b - a);
                lin * lin
            };
            acp *= 1.0 - beta;
            sigmas.push(((1.0 - acp) / acp).sqrt());
        }
        Ok(Self { sigmas, ancestral })
    }

    /// Whether this sampler injects ancestral noise (SDXL: true).
    pub fn is_ancestral(&self) -> bool {
        self.ancestral
    }

    /// The maximum (start) time index: `len(sigmas) − 1` = `train_steps`. txt2img starts here;
    /// img2img would start at `max_time · strength`.
    pub fn max_time(&self) -> f64 {
        (self.sigmas.len() - 1) as f64
    }

    /// `σ_last` — the largest σ, the prior's noise scale.
    pub fn sigma_last(&self) -> f64 {
        *self.sigmas.last().expect("σ table is never empty")
    }

    /// Linearly interpolate the σ table at the (float) time `t` (the vendored `_interp`), host f64.
    /// At integer `t` this is exactly `sigmas[t]`.
    pub fn sigma(&self, t: f64) -> f64 {
        let last = self.sigmas.len() - 1;
        let lo = (t as usize).min(last);
        let hi = (lo + 1).min(last);
        let delta = t - lo as f64;
        self.sigmas[lo] * (1.0 - delta) + delta * self.sigmas[hi]
    }

    /// The `(t, t_prev)` step pairs: `_linspace(start_time, 0, num_steps + 1)` zipped with its tail.
    ///
    /// Op order matches the vendored `_linspace`: `x = arange/num_steps`, then `(0−start)·x + start`
    /// (arange-over-N first), so the timesteps land on the same floats the reference produces. An
    /// `img2img` run would pass `start_time = max_time·strength` and the reduced step count; txt2img
    /// passes `max_time` and the full count.
    pub fn timesteps(&self, num_steps: usize, start_time: f64) -> Vec<(f64, f64)> {
        if num_steps == 0 {
            return Vec::new();
        }
        let n = num_steps as f64;
        let steps: Vec<f64> = (0..=num_steps)
            .map(|i| {
                let x = i as f64 / n; // arange/num_steps first, matching _linspace
                (0.0 - start_time) * x + start_time
            })
            .collect();
        steps.windows(2).map(|w| (w[0], w[1])).collect()
    }

    /// The prior-scaling factor `σ_last · rsqrt(σ_last² + 1)` — the scalar [`scale_prior_noise`]
    /// multiplies unit-normal noise by. Exposed so the denoise loop / tests can compute the prior
    /// without a tensor.
    ///
    /// [`scale_prior_noise`]: Self::scale_prior_noise
    pub fn prior_scale(&self) -> f64 {
        let s = self.sigma_last();
        s / (s * s + 1.0).sqrt()
    }

    /// Scale already-drawn unit-normal `noise` into the prior latent space: `noise · σ_last ·
    /// rsqrt(σ_last²+1)`. The denoise loop draws the unit-normal `noise` from its seeded `StdRng`
    /// (the launch-portable determinism, sc-3673) and calls this — mirroring the mlx
    /// `scale_prior_noise`, minus the f32-op-order dance (no Python parity to hold).
    pub fn scale_prior_noise(&self, noise: &Tensor) -> Result<Tensor> {
        Ok(noise.affine(self.prior_scale(), 0.0)?)
    }

    /// Re-noise a clean latent `x` to the schedule time `t`: `(x + noise·σ(t)) · rsqrt(σ(t)²+1)` — the
    /// img2img / inpaint init-noising (the mlx `add_noise`). For img2img the caller noises the
    /// VAE-encoded source to `start_time = max_time·strength`; the inpaint blend re-noises the *clean*
    /// init to each step's `t_prev`. `noise` is unit-normal (the caller's seeded stream) and is cast to
    /// `x`'s dtype so an f16 latent stays f16. At `t = 0`, `σ = 0`, so this returns `x` unchanged (the
    /// clean init) — the final-step inpaint blend then pins the kept region to the source exactly.
    pub fn add_noise(&self, x: &Tensor, noise: &Tensor, t: f64) -> Result<Tensor> {
        let sigma = self.sigma(t);
        let renorm = 1.0 / (sigma * sigma + 1.0).sqrt();
        let noised = (x + noise.affine(sigma, 0.0)?.to_dtype(x.dtype())?)?;
        Ok(noised.affine(renorm, 0.0)?)
    }

    /// One denoise step from `x_t` (at time `t`) to `x_{t_prev}`. Euler-ancestral when
    /// `self.ancestral` — the caller-supplied unit-normal `noise` (drawn from the loop's seeded RNG)
    /// is scaled by `σ_up`; plain Euler otherwise (`noise` unused). All scalar schedule math is host
    /// f64; the latents are combined via scalar ops in their own dtype.
    ///
    /// `σ_up = sqrt(σ_prev²·(σ²−σ_prev²)/σ²)`, `σ_down = sqrt(σ_prev² − σ_up²)`, then
    /// `x' = (sqrt(σ²+1)·x_t + eps·(σ_down−σ) + noise·σ_up) · rsqrt(σ_prev²+1)`. At the final step
    /// `t_prev = 0 ⇒ σ_prev = 0 ⇒ σ_up = σ_down = 0`, so no noise is added and the renorm is 1.
    pub fn step(
        &self,
        eps_pred: &Tensor,
        x_t: &Tensor,
        t: f64,
        t_prev: f64,
        noise: &Tensor,
    ) -> Result<Tensor> {
        let sigma = self.sigma(t);
        let sigma_prev = self.sigma(t_prev);
        let sigma2 = sigma * sigma;
        let sigma_prev2 = sigma_prev * sigma_prev;

        // x' = sqrt(σ²+1)·x_t + eps·dt, dt = σ_down−σ (ancestral) | σ_prev−σ (euler), then ·renorm.
        let c1 = (sigma2 + 1.0).sqrt();
        let renorm = 1.0 / (sigma_prev2 + 1.0).sqrt();

        if self.ancestral {
            // σ_up²·(positive) only when σ²>0 (true for any t>0; the schedule's t is always >0,
            // t_prev may be 0). Guard the σ²=0 corner to a no-noise step rather than dividing by 0.
            let sigma_up = if sigma2 > 0.0 {
                (sigma_prev2 * (sigma2 - sigma_prev2) / sigma2).sqrt()
            } else {
                0.0
            };
            // σ_down² = σ_prev² − σ_up² (the reference's `σ_prev² − σ_up**2`); clamp away a tiny
            // negative from f64 round-off before the sqrt.
            let sigma_down = (sigma_prev2 - sigma_up * sigma_up).max(0.0).sqrt();
            let dt = sigma_down - sigma;
            let mut x = (x_t.affine(c1, 0.0)? + eps_pred.affine(dt, 0.0)?)?;
            if sigma_up != 0.0 {
                // The denoise loop draws the ancestral noise on CPU in f32 (the seeded `StdRng`
                // stream); cast to the running latent dtype so an f16 denoise doesn't error / get
                // promoted to f32. A no-op when `noise` is already the latent dtype (e.g. the f32 tests).
                let noise_term = noise.affine(sigma_up, 0.0)?.to_dtype(x.dtype())?;
                x = (x + noise_term)?;
            }
            Ok(x.affine(renorm, 0.0)?)
        } else {
            let dt = sigma_prev - sigma;
            let x = (x_t.affine(c1, 0.0)? + eps_pred.affine(dt, 0.0)?)?;
            Ok(x.affine(renorm, 0.0)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Host-f64 reference for one ancestral/euler step, element-wise — the ground truth the tensor
    /// path is checked against (run at f32 so the comparison is tight, not f16-lossy).
    fn ref_step(
        eps: &[f32],
        x_t: &[f32],
        sigma: f64,
        sigma_prev: f64,
        noise: &[f32],
        ancestral: bool,
    ) -> Vec<f32> {
        let sigma2 = sigma * sigma;
        let sigma_prev2 = sigma_prev * sigma_prev;
        let c1 = (sigma2 + 1.0).sqrt();
        let renorm = 1.0 / (sigma_prev2 + 1.0).sqrt();
        let (dt, sigma_up) = if ancestral {
            let sigma_up = if sigma2 > 0.0 {
                (sigma_prev2 * (sigma2 - sigma_prev2) / sigma2).sqrt()
            } else {
                0.0
            };
            let sigma_down = (sigma_prev2 - sigma_up * sigma_up).max(0.0).sqrt();
            (sigma_down - sigma, sigma_up)
        } else {
            (sigma_prev - sigma, 0.0)
        };
        eps.iter()
            .zip(x_t.iter())
            .zip(noise.iter())
            .map(|((&e, &x), &nz)| {
                let v = c1 * x as f64 + dt * e as f64 + sigma_up * nz as f64;
                (v * renorm) as f32
            })
            .collect()
    }

    /// The σ table: a leading 0, length `train_steps + 1`, monotonically increasing, `max_time` =
    /// `train_steps`, and the σ(t) interp is linear at a half index (mirrors the mlx sampler test).
    #[test]
    fn sigma_table_endpoints_and_interp() {
        let s = EulerAncestralSampler::sdxl();
        assert_eq!(s.sigmas.len(), 1001);
        assert_eq!(s.sigmas[0], 0.0);
        assert_eq!(s.max_time(), 1000.0);
        assert!(s.sigmas.windows(2).all(|w| w[1] >= w[0]), "σ not monotonic");
        // σ_last is the largest entry.
        assert!((s.sigma_last() - s.sigmas[1000]).abs() < 1e-12);
        // Linear interp at a half index.
        let mid = s.sigma(10.5);
        assert!((mid - 0.5 * (s.sigmas[10] + s.sigmas[11])).abs() < 1e-9);
    }

    /// `timesteps` spans `start_time` down to 0 over `num_steps` pairs; 0 steps yields no pairs
    /// (the img2img-at-tiny-strength no-op — never invokes the σ=0 step).
    #[test]
    fn timesteps_span_and_zero_steps() {
        let s = EulerAncestralSampler::sdxl();
        assert!(s.timesteps(0, 1000.0).is_empty());
        let ts = s.timesteps(4, 1000.0);
        assert_eq!(ts.len(), 4);
        assert_eq!(ts[0].0, 1000.0);
        assert!((ts.last().unwrap().1 - 0.0).abs() < 1e-9);
        // Adjacent pairs chain: each t_prev is the next t.
        for w in ts.windows(2) {
            assert!((w[0].1 - w[1].0).abs() < 1e-12);
        }
    }

    /// `scale_prior_noise` = `noise · σ_last · rsqrt(σ_last²+1)`, matching the host scalar.
    #[test]
    fn prior_scale_matches_host() {
        let s = EulerAncestralSampler::sdxl();
        let dev = Device::Cpu;
        let noise = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap();
        let scaled = s.scale_prior_noise(&noise).unwrap();
        let factor = s.prior_scale() as f32;
        let n = noise.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let g = scaled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (gi, ni) in g.iter().zip(n.iter()) {
            assert!((gi - ni * factor).abs() < 1e-6, "{gi} vs {}", ni * factor);
        }
        // σ_last/sqrt(σ_last²+1) is just under 1 (σ_last ≫ 1).
        assert!(s.prior_scale() < 1.0 && s.prior_scale() > 0.99);
    }

    /// `add_noise(x, noise, t)` = `(x + noise·σ(t))·rsqrt(σ(t)²+1)`: at `t = 0` (σ = 0) it returns `x`
    /// unchanged (the img2img-at-strength-0 / inpaint final-step pin), and at a mid-schedule `t` it
    /// matches the host scalar and actually changes `x`.
    #[test]
    fn add_noise_zero_t_identity_and_mid_t_noises() {
        let s = EulerAncestralSampler::sdxl();
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let nv = noise.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // t = 0 ⇒ σ = 0 ⇒ identity (the kept-region pin at the final inpaint step).
        let a0 = s
            .add_noise(&x, &noise, 0.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in a0.iter().zip(xv.iter()) {
            assert!(
                (g - w).abs() < 1e-6,
                "add_noise at t=0 must be identity: {g} vs {w}"
            );
        }

        // Mid-schedule: matches the host scalar and differs from x.
        let t = 600.0;
        let sigma = s.sigma(t);
        let renorm = 1.0 / (sigma * sigma + 1.0).sqrt();
        let got = s
            .add_noise(&x, &noise, t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for ((g, &xi), &ni) in got.iter().zip(xv.iter()).zip(nv.iter()) {
            let want = ((xi as f64 + ni as f64 * sigma) * renorm) as f32;
            assert!((g - want).abs() < 1e-4, "add_noise off: {g} vs {want}");
        }
        assert!(
            got.iter().zip(xv.iter()).any(|(g, x)| (g - x).abs() > 1e-3),
            "a mid-schedule add_noise must change x"
        );
    }

    /// The ancestral `step` matches the host-f64 reference for a mid-schedule step (σ_up > 0, so the
    /// injected noise contributes) — validating the σ_up/σ_down/renorm math through the tensor path.
    /// Run at f32 for a tight comparison.
    #[test]
    fn ancestral_step_matches_reference() {
        let s = EulerAncestralSampler::sdxl();
        let dev = Device::Cpu;
        let (t, t_prev) = (800.0, 600.0);
        let sigma = s.sigma(t);
        let sigma_prev = s.sigma(t_prev);
        let shape = (1usize, 4usize, 4usize, 4usize);
        let eps = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let x_t = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();

        let out = s.step(&eps, &x_t, t, t_prev, &noise).unwrap();
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = ref_step(
            &eps.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &x_t.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            sigma,
            sigma_prev,
            &noise.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            true,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "ancestral step off: {g} vs {w}");
        }
        // Mid-schedule σ_up is strictly positive (noise actually injected here).
        let sigma2 = sigma * sigma;
        let sigma_prev2 = sigma_prev * sigma_prev;
        assert!((sigma_prev2 * (sigma2 - sigma_prev2) / sigma2).sqrt() > 0.0);
    }

    /// At the final step (`t_prev = 0`) σ_up = 0, so the injected noise is multiplied away — the step
    /// is deterministic regardless of the noise tensor. Two different noise draws must give the same
    /// output (the diffusers "last step is noiseless" property).
    #[test]
    fn final_step_ignores_noise() {
        let s = EulerAncestralSampler::sdxl();
        let dev = Device::Cpu;
        let (t, t_prev) = (250.0, 0.0);
        let shape = (1usize, 4usize, 4usize, 4usize);
        let eps = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let x_t = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let noise_a = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let noise_b = Tensor::randn(0f32, 5f32, shape, &dev).unwrap();

        let a = s
            .step(&eps, &x_t, t, t_prev, &noise_a)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = s
            .step(&eps, &x_t, t, t_prev, &noise_b)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "final step is noise-dependent: {x} vs {y}"
            );
        }
        // And a mid-schedule step IS noise-dependent (guards against a degenerate σ_up≡0 bug).
        let mid_a = s.step(&eps, &x_t, 800.0, 600.0, &noise_a).unwrap();
        let mid_b = s.step(&eps, &x_t, 800.0, 600.0, &noise_b).unwrap();
        let da = mid_a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let db = mid_b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            da.iter().zip(db.iter()).any(|(x, y)| (x - y).abs() > 1e-4),
            "mid-schedule step should depend on the injected noise"
        );
    }

    /// The plain Euler (non-ancestral) step matches its host reference and is noise-independent at
    /// every step (no σ_up term) — pinning the `ancestral=false` branch.
    #[test]
    fn euler_step_matches_reference_and_ignores_noise() {
        let s = EulerAncestralSampler::new(1000, SDXL_BETA_START, SDXL_BETA_END, false).unwrap();
        assert!(!s.is_ancestral());
        let dev = Device::Cpu;
        let (t, t_prev) = (700.0, 500.0);
        let shape = (1usize, 4usize, 4usize, 4usize);
        let eps = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let x_t = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let out = s.step(&eps, &x_t, t, t_prev, &noise).unwrap();
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = ref_step(
            &eps.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &x_t.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            s.sigma(t),
            s.sigma(t_prev),
            &noise.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            false,
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "euler step off: {g} vs {w}");
        }
    }

    /// `new(0, …)` is rejected (an empty σ table would make `sigma_last` panic and the denoise NaN).
    #[test]
    fn zero_train_steps_is_rejected() {
        assert!(EulerAncestralSampler::new(0, SDXL_BETA_START, SDXL_BETA_END, true).is_err());
    }
}
