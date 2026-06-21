//! The candle backend impl of the unified `gen_core::sampling::LatentOps` (epic 7114 P2, sc-7119) —
//! the tensor primitives the unified curated samplers (Euler / Heun / DPM++ 2M·SDE / UniPC /
//! ancestral / LCM / DDIM, sc-7117) are written against, over `candle_core::Tensor`. The candle twin
//! of mlx-gen's `MlxLatentOps`.
//!
//! Carries the same byte-parity convention so a migrated engine's DEFAULT sampler stays bit-identical
//! (the N1 gate): `scale(x, 1.0)` and `axpy(1.0, x, b, y) = x + y·b` elide the multiply-by-one.
//! `randn_like` uses the same per-step subkey derivation as mlx-gen's `StepRng` (D6 determinism),
//! drawn with a seeded CPU `StdRng` + `StandardNormal` so the noise is launch-portable and matched to
//! the latent's dtype/shape/device (the sc-3673 / sc-5179 initial-noise lineage). Cross-backend bitwise
//! equality of the draw is NOT a goal (the RNG differs from mlx).

use candle_core::Tensor;
use gen_core::sampling::LatentOps;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::CandleError;

/// Lift a `candle_core::Result` into the backend-neutral `gen_core::Result` (the `LatentOps` trait is
/// declared in gen-core, so its methods return `gen_core::Result`). Routes through the existing
/// candle-gen bridge: `candle_core::Error -> CandleError -> gen_core::Error::Backend`.
#[inline]
fn ge<T>(r: candle_core::Result<T>) -> gen_core::Result<T> {
    r.map_err(|e| CandleError::from(e).into())
}

/// The candle backend impl of [`gen_core::sampling::LatentOps`]. See the module docs.
#[derive(Clone, Copy, Debug, Default)]
pub struct CandleLatentOps;

impl LatentOps for CandleLatentOps {
    type Latent = Tensor;

    fn scale(&self, x: &Tensor, scale: f32) -> gen_core::Result<Tensor> {
        if scale == 1.0 {
            return Ok(x.clone());
        }
        ge(x.affine(scale as f64, 0.0))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> gen_core::Result<Tensor> {
        ge(a.add(b))
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> gen_core::Result<Tensor> {
        ge(a.sub(b))
    }

    fn axpy(&self, a: f32, x: &Tensor, b: f32, y: &Tensor) -> gen_core::Result<Tensor> {
        // Byte-parity with mlx-gen's apply_step a_x==1 branch: emit `x + y·b` (multiply-by-one elided).
        let yb = ge(y.affine(b as f64, 0.0))?;
        if a == 1.0 {
            return ge(x.add(&yb));
        }
        let ax = ge(x.affine(a as f64, 0.0))?;
        ge(ax.add(&yb))
    }

    fn randn_like(&self, x: &Tensor, seed: u64, step: usize) -> gen_core::Result<Tensor> {
        // Same per-step subkey derivation as mlx-gen's StepRng (de-correlate steps; `+1` keeps step 0
        // off the raw init-noise seed). Seeded CPU StdRng draw -> launch-portable, then matched to the
        // latent's shape/device/dtype.
        let sub = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(step as u64 + 1));
        let mut rng = StdRng::seed_from_u64(sub);
        let data: Vec<f32> = (0..x.elem_count())
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let noise = ge(Tensor::from_vec(data, x.shape().clone(), x.device()))?;
        ge(noise.to_dtype(x.dtype()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use gen_core::sampling::{
        build_flow_sigmas, compute_mu, denoise, image_seq_len, sampler_by_name, Euler,
        FlowModelSampling, Sampler, TimestepConvention,
    };

    fn t(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), v.len(), &Device::Cpu).unwrap()
    }
    fn vec1(x: &Tensor) -> Vec<f32> {
        x.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }
    /// A constant flow velocity (independent of x) for the exactness check.
    fn const_v() -> Tensor {
        t(&[0.37, -0.12, 0.8, -0.5])
    }

    #[test]
    fn candle_scale_add_sub() {
        let ops = CandleLatentOps;
        let a = t(&[1.0, 2.0, 3.0]);
        let b = t(&[0.5, -1.0, 4.0]);
        assert_eq!(vec1(&ops.scale(&a, 2.0).unwrap()), vec![2.0, 4.0, 6.0]);
        assert_eq!(vec1(&ops.scale(&a, 1.0).unwrap()), vec1(&a)); // 1.0 -> clone
        assert_eq!(vec1(&ops.add(&a, &b).unwrap()), vec![1.5, 1.0, 7.0]);
        assert_eq!(vec1(&ops.sub(&a, &b).unwrap()), vec![0.5, 3.0, -1.0]);
    }

    #[test]
    fn candle_axpy_a1_byte_parity() {
        let ops = CandleLatentOps;
        let x = t(&[0.3, -1.2, 2.5]);
        let y = t(&[0.7, 0.1, -0.4]);
        let got = ops.axpy(1.0, &x, 0.25, &y).unwrap();
        let want = x.add(&y.affine(0.25, 0.0).unwrap()).unwrap();
        assert_eq!(vec1(&got), vec1(&want), "axpy a=1 not byte-identical");
        // General a: 2·x + (−3)·y.
        let got2 = ops.axpy(2.0, &x, -3.0, &y).unwrap();
        let want2 = x
            .affine(2.0, 0.0)
            .unwrap()
            .add(&y.affine(-3.0, 0.0).unwrap())
            .unwrap();
        assert_eq!(vec1(&got2), vec1(&want2));
    }

    #[test]
    fn candle_randn_like_deterministic_shaped() {
        let ops = CandleLatentOps;
        let x = t(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let a = ops.randn_like(&x, 42, 0).unwrap();
        assert_eq!(a.dims(), x.dims());
        assert_eq!(a.dtype(), x.dtype());
        assert_eq!(vec1(&a), vec1(&ops.randn_like(&x, 42, 0).unwrap()));
        assert_ne!(vec1(&a), vec1(&ops.randn_like(&x, 42, 1).unwrap()));
        assert_ne!(vec1(&a), vec1(&ops.randn_like(&x, 43, 0).unwrap()));
    }

    #[test]
    fn candle_euler_integrates_constant_velocity_exactly() {
        // The rectified-flow ODE dx/dσ = v (constant) is linear, so the unified Euler over
        // CandleLatentOps must land EXACTLY on x_init − v·σ_0. Proves scale/sub/axpy compose right.
        let ops = CandleLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = build_flow_sigmas(10, compute_mu(image_seq_len(1024, 1024), 10));
        let v = const_v();
        let x_init = t(&[0.3, -1.1, 2.0, 0.05]);
        let mut dn = |x: &Tensor, s: f32| denoise(&ops, &ms, x, s, |_xin, _t| Ok(v.clone()));
        let out = Euler
            .sample(&ops, &mut dn, x_init.clone(), &sigmas, 0)
            .unwrap();
        let want = x_init
            .sub(&v.affine(sigmas[0] as f64, 0.0).unwrap())
            .unwrap();
        assert!(max_abs(&vec1(&out), &vec1(&want)) < 1e-4);
    }

    #[test]
    fn candle_drives_every_curated_solver_to_finite_output() {
        // The P2 deliverable: every gen-core curated sampler runs end-to-end over candle Tensor.
        let ops = CandleLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = build_flow_sigmas(6, compute_mu(image_seq_len(512, 512), 6));
        let x_init = t(&[0.2, -0.5, 1.0, 0.3]);
        for name in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "lcm",
            "ddim",
        ] {
            let sampler = sampler_by_name::<CandleLatentOps>(name).expect("known solver");
            // Smooth velocity field v = 0.25·x + 0.1.
            let mut dn =
                |x: &Tensor, s: f32| denoise(&ops, &ms, x, s, |xin, _t| ge(xin.affine(0.25, 0.1)));
            let out = sampler
                .sample(&ops, &mut dn, x_init.clone(), &sigmas, 7)
                .unwrap();
            assert!(
                vec1(&out).iter().all(|v| v.is_finite()),
                "{name} produced non-finite output"
            );
        }
    }
}
