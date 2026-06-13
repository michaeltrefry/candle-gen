//! Rectified-flow distilled scheduler (single-stage) — port of mlx-gen-ltx `pipeline.rs`
//! `to_denoised` + `euler_step`. The distilled model bakes guidance in (no CFG); the schedule is the
//! fixed [`STAGE1_SIGMAS`](crate::config::STAGE1_SIGMAS) (σ 1.0 → 0.0 in 8 steps).

use candle_gen::candle_core::{Result, Tensor};

/// `denoised = latent − σ·velocity` (velocity → x₀).
pub fn to_denoised(latent: &Tensor, velocity: &Tensor, sigma: f64) -> Result<Tensor> {
    latent - velocity.affine(sigma, 0.0)?
}

/// Legacy dtype-preserving Euler: for `σ_next > 0`, `x' = denoised + σ_next·(x − denoised)/σ`; at the
/// final step (`σ_next = 0`), `x' = denoised`.
pub fn euler_step(x: &Tensor, denoised: &Tensor, sigma: f64, sigma_next: f64) -> Result<Tensor> {
    if sigma_next <= 0.0 {
        return Ok(denoised.clone());
    }
    let step = (x - denoised)?.affine(sigma_next / sigma, 0.0)?;
    denoised + step
}
