//! Ideogram 4's logit-normal flow-matching schedule + Euler sampler timing — a verbatim port of
//! `mlx-gen-ideogram`'s `scheduler.rs` (pure `f64` math, no backend dependency). The published
//! `model_index.json` names `FlowMatchEulerDiscreteScheduler`, but the real `Ideogram4Pipeline`
//! samples with this resolution-aware `LogitNormalSchedule`.
//!
//! `schedule(t) = clamp(1 − σ(mean + std·Φ⁻¹(t)), t_min, t_max)` where `mean` grows with the image
//! pixel count, `Φ⁻¹` is the probit (inverse normal CDF), `σ` is the logistic, and the clamp bounds
//! come from the logSNR range. The denoise loop walks [`make_step_intervals`] (linear `[0,1]`) from
//! high noise to low, Euler-stepping `z += v·(s−t)` per step.

/// Logit-normal noise schedule for one resolution. `mean`/`std` are the logit-normal params; the
/// clamp bounds come from `logsnr_min=-15`, `logsnr_max=18`.
#[derive(Clone, Copy, Debug)]
pub struct LogitNormalSchedule {
    mean: f64,
    std: f64,
    t_min: f64,
    t_max: f64,
}

impl LogitNormalSchedule {
    /// Eval-time resolution-aware schedule: `mean = known_mean + 0.5·ln(pixels / known_pixels)`,
    /// `known_resolution = 512×512`.
    pub fn for_resolution(height: u32, width: u32, known_mean: f64, std: f64) -> Self {
        let num_pixels = (height as f64) * (width as f64);
        let known_pixels = 512.0 * 512.0;
        let mean = known_mean + 0.5 * (num_pixels / known_pixels).ln();
        Self {
            mean,
            std,
            t_min: 1.0 / (1.0 + (0.5 * 18.0_f64).exp()),
            t_max: 1.0 / (1.0 + (0.5 * -15.0_f64).exp()),
        }
    }

    /// Map a step interval `t ∈ [0,1]` to a flow-matching time (σ, the noise level).
    pub fn eval(&self, t: f64) -> f64 {
        let z = ndtri(t);
        let y = self.mean + self.std * z;
        let mapped = 1.0 - logistic(y);
        mapped.clamp(self.t_min, self.t_max)
    }
}

/// Linear step grid `[0,1]` with `num_steps + 1` points (upstream `make_step_intervals`).
pub fn make_step_intervals(num_steps: usize) -> Vec<f64> {
    if num_steps == 0 {
        return vec![0.0];
    }
    (0..=num_steps)
        .map(|i| i as f64 / num_steps as f64)
        .collect()
}

/// The `(mu, std)` preset the reference pipeline selects by step count (upstream `preset_mu_std`):
/// `V4_TURBO_12` (≤15), `V4_DEFAULT_20` (≤33), else `V4_QUALITY_48`.
pub fn preset_mu_std(num_steps: usize) -> (f64, f64) {
    match num_steps {
        s if s <= 15 => (0.5, 1.75),
        s if s <= 33 => (0.0, 1.75),
        _ => (0.0, 1.5),
    }
}

fn logistic(y: f64) -> f64 {
    1.0 / (1.0 + (-y).exp())
}

/// Inverse normal CDF (probit), Acklam's rational approximation (|err| ≲ 1.15e-9). Endpoints map to
/// ±∞ so the schedule clamp produces the boundary `t_min`/`t_max` (matching torch `ndtri`).
fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    let p_high = 1.0 - P_LOW;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matches the upstream `get_schedule_for_resolution((256,256), known_mean=0.5)` over
    /// `make_step_intervals(4)` (values from the reference torch scheduler).
    #[test]
    fn schedule_matches_reference() {
        let s = LogitNormalSchedule::for_resolution(256, 256, 0.5, 1.0);
        let si = make_step_intervals(4);
        assert_eq!(si, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let want = [0.99944723, 0.70425373, 0.54813725, 0.38193515, 0.00012339];
        for (&t, &w) in si.iter().zip(want.iter()) {
            let g = s.eval(t);
            assert!((g - w).abs() < 1e-5, "schedule({t}) = {g}, want {w}");
        }
    }

    #[test]
    fn presets_match_step_buckets() {
        assert_eq!(preset_mu_std(8), (0.5, 1.75));
        assert_eq!(preset_mu_std(20), (0.0, 1.75));
        assert_eq!(preset_mu_std(48), (0.0, 1.5));
    }
}
