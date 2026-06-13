//! Training optimizers (sc-5165) — the candle twin of `mlx_gen::train::optim`. Unlike candle's stock
//! `Optimizer` trait (which only ships AdamW/SGD), this exposes the full SceneWorks optimizer set
//! (`adamw`/`adam`/`rose`/`prodigy`), stepping the adapter-factor `Var`s directly from a
//! [`GradStore`]. Adam/AdamW delegate to candle's `AdamW`; Rose and Prodigy are faithful ports of the
//! MLX implementations (decoupled weight decay, f32 compute, no bias correction / safeguard warmup).
//!
//! The LR-schedule multiplier is applied via [`TrainOptimizer::set_lr_scaled`] each optimizer update;
//! grad-norm clipping ([`clip_grad_norm`]) — which candle has no built-in for — runs on the
//! `GradStore` before the step.

use std::collections::HashMap;

use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::{CandleError, Result};

/// The optimizer names the worker may request (mirrors the MLX `SUPPORTED_OPTIMIZERS`).
pub const SUPPORTED_OPTIMIZERS: [&str; 4] = ["adamw", "adam", "rose", "prodigy"];

/// Collapse an optimizer name to its canonical form (`"AdamW"`/`"adamw8bit"` → `"adamw"`,
/// `"prodigy-opt"` → `"prodigy"`, `"rose_opt"` → `"rose"`). Unknown names pass through (and then fail
/// [`TrainOptimizer::from_config`]).
pub fn normalize(name: &str) -> String {
    let s: String = name
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    if s.contains("prodigy") {
        "prodigy".into()
    } else if s.contains("rose") {
        "rose".into()
    } else if s.contains("adamw") {
        "adamw".into()
    } else if s.contains("adam") {
        "adam".into()
    } else {
        s
    }
}

/// Global L2-norm gradient clipping over `vars`' gradients in `grads` (candle ships no built-in). If
/// the total norm exceeds `max_norm`, every gradient is scaled by `max_norm / norm` in place. Returns
/// the pre-clip total norm. Mirrors the MLX `clip_grad_norm(&grads, 1.0)` the trainer applies to the
/// averaged gradient before the optimizer step.
pub fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<f64> {
    let mut total_sq = 0f64;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let s = g
                .sqr()?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            total_sq += s;
        }
    }
    let norm = total_sq.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for v in vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let scaled = (g * scale)?;
                grads.insert(v.as_tensor(), scaled);
            }
        }
    }
    Ok(norm)
}

/// Accumulate one micro-step's `grads` into `acc` (`+=` per `Var`); the first step seeds it. The
/// gradient-accumulation companion to [`scale_grads`] (the `1/accum` averaging that follows) — the
/// generic `GradStore` half of every family trainer's step loop, so it lives in the shared harness
/// rather than each `training.rs`.
pub fn accumulate_grads(acc: &mut Option<GradStore>, grads: GradStore, vars: &[Var]) -> Result<()> {
    match acc {
        None => *acc = Some(grads),
        Some(a) => {
            for v in vars {
                if let Some(g) = grads.get(v.as_tensor()) {
                    let summed = match a.get(v.as_tensor()) {
                        Some(prev) => (prev + g)?,
                        None => g.clone(),
                    };
                    a.insert(v.as_tensor(), summed);
                }
            }
        }
    }
    Ok(())
}

/// Scale every `Var`'s gradient in `grads` by `factor` in place — the `1/accumulation` averaging
/// applied before [`clip_grad_norm`] + [`TrainOptimizer::step`].
pub fn scale_grads(grads: &mut GradStore, vars: &[Var], factor: f64) -> Result<()> {
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let scaled = (g * factor)?;
            grads.insert(v.as_tensor(), scaled);
        }
    }
    Ok(())
}

/// One of the supported training optimizers, owning the factor `Var`s it steps.
pub enum TrainOptimizer {
    /// candle AdamW (also serves plain `adam` with `weight_decay = 0`).
    Adam {
        inner: AdamW,
        base_lr: f64,
    },
    Rose(Rose),
    Prodigy(Prodigy),
}

impl TrainOptimizer {
    pub fn is_supported(name: &str) -> bool {
        SUPPORTED_OPTIMIZERS.contains(&normalize(name).as_str())
    }

    /// Construct the optimizer named `name` over `vars` at learning rate `lr` and weight decay
    /// `weight_decay`. Betas/eps follow the torch/diffusers defaults (0.9, 0.999, 1e-8).
    pub fn from_config(name: &str, vars: Vec<Var>, lr: f32, weight_decay: f32) -> Result<Self> {
        match normalize(name).as_str() {
            "adamw" => Ok(Self::adam(vars, lr, weight_decay)?),
            "adam" => Ok(Self::adam(vars, lr, 0.0)?),
            "rose" => Ok(Self::Rose(Rose::new(vars, lr, weight_decay))),
            "prodigy" => Ok(Self::Prodigy(Prodigy::new(vars, lr, weight_decay))),
            other => Err(CandleError::Msg(format!(
                "unsupported optimizer {other:?}; supported: {}",
                SUPPORTED_OPTIMIZERS.join(", ")
            ))),
        }
    }

    fn adam(vars: Vec<Var>, lr: f32, weight_decay: f32) -> Result<Self> {
        let params = ParamsAdamW {
            lr: lr as f64,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: weight_decay as f64,
        };
        let inner = AdamW::new(vars, params)?;
        Ok(Self::Adam {
            inner,
            base_lr: lr as f64,
        })
    }

    /// Scale the base learning rate by the schedule multiplier for the next update.
    pub fn set_lr_scaled(&mut self, mult: f32) {
        match self {
            Self::Adam { inner, base_lr } => inner.set_learning_rate(*base_lr * mult as f64),
            Self::Rose(r) => r.set_lr_scaled(mult),
            Self::Prodigy(p) => p.set_lr_scaled(mult),
        }
    }

    /// Apply one optimizer step from the (already clipped) gradients.
    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        match self {
            Self::Adam { inner, .. } => Ok(inner.step(grads)?),
            Self::Rose(r) => r.step(grads),
            Self::Prodigy(p) => p.step(grads),
        }
    }
}

/// Replace exact zeros in `t` with 1 (avoid div-by-zero in a range/denominator). `where t==0 → 1`.
fn zeros_to_one(t: &Tensor) -> candle_core::Result<Tensor> {
    let is_zero = t.eq(&t.zeros_like()?)?.to_dtype(t.dtype())?;
    t + is_zero
}

/// Stateless Range-Of-Slice Equilibration optimizer (rose-opt). The only mutable state is the
/// (schedule-scaled) learning rate. Faithful port of the MLX `Rose` per-parameter update: decoupled
/// weight decay, then range-normalization over the trailing axes with optional centralization +
/// a coefficient-of-variation trust gate (`centralize = stabilize = true`, the SceneWorks default).
pub struct Rose {
    vars: Vec<Var>,
    base_lr: f64,
    lr: f64,
    weight_decay: f64,
    centralize: bool,
    stabilize: bool,
}

impl Rose {
    pub fn new(vars: Vec<Var>, lr: f32, weight_decay: f32) -> Self {
        Self {
            vars,
            base_lr: lr as f64,
            lr: lr as f64,
            weight_decay: weight_decay as f64,
            centralize: true,
            stabilize: true,
        }
    }

    fn set_lr_scaled(&mut self, mult: f32) {
        self.lr = self.base_lr * mult as f64;
    }

    fn step(&self, grads: &GradStore) -> Result<()> {
        for v in &self.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let updated = self.update_one(v.as_tensor(), g)?;
                v.set(&updated)?;
            }
        }
        Ok(())
    }

    /// One Rose update for a single parameter (ports `Rose.step`'s per-`p` body).
    fn update_one(&self, param: &Tensor, grad: &Tensor) -> Result<Tensor> {
        let lr = self.lr;
        // Decoupled multiplicative weight decay: θ *= max(0, 1 − lr·wd).
        let mut param = if self.weight_decay != 0.0 {
            (param * (1.0 - lr * self.weight_decay).max(0.0))?
        } else {
            param.clone()
        };
        match grad.rank() {
            0 => {
                return Err(CandleError::Msg(
                    "Rose: 0-D parameters are unsupported (adapter factors are matrices)".into(),
                ))
            }
            1 => {
                // Global range over the whole vector.
                let g_max = grad.max(0)?;
                let g_min = grad.min(0)?;
                let denom = zeros_to_one(&(g_max.abs()? - g_min)?)?;
                let upd = (grad.broadcast_div(&denom)? * (-lr))?;
                param = (param + upd)?;
            }
            _ => {
                // Active axes = every axis except the leading one. Adapter factors are 2-D, so the
                // trailing axes flatten to one and the reduction is per leading slice (per row).
                let dims = grad.dims().to_vec();
                let leading = dims[0];
                let rest: usize = dims[1..].iter().product();
                let g2 = grad.reshape((leading, rest))?;
                let g2 = if self.centralize {
                    let mean = g2.mean_keepdim(D::Minus1)?;
                    g2.broadcast_sub(&mean)?
                } else {
                    g2
                };
                let raw_scale = (g2.max_keepdim(D::Minus1)?.abs()? - g2.min_keepdim(D::Minus1)?)?; // [leading,1]
                let denom = if self.stabilize {
                    // Population mean/std over the per-row range tensor; trust = mean/(std+mean).
                    let mean = raw_scale.mean_all()?;
                    let var = raw_scale.broadcast_sub(&mean)?.sqr()?.mean_all()?;
                    let std = var.sqrt()?;
                    let trust = mean.broadcast_div(&zeros_to_one(&(std + &mean)?)?)?; // scalar
                                                                                      // denom = mean + trust·(raw_scale − mean).
                    let centered = raw_scale.broadcast_sub(&mean)?;
                    centered.broadcast_mul(&trust)?.broadcast_add(&mean)?
                } else {
                    raw_scale
                };
                let denom = zeros_to_one(&denom)?; // [leading,1]
                let upd = (g2.broadcast_div(&denom)? * (-lr))?;
                let upd = upd.reshape(dims)?;
                param = (param + upd)?;
            }
        }
        Ok(param)
    }
}

/// Per-parameter Prodigy state: the Adam EMAs, the `s` accumulator, and the initial parameter `p0`.
struct ProdigyState {
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    s: Tensor,
    p0: Tensor,
}

/// Prodigy (prodigyopt): Adam with a learning-rate-free, globally-adapted step size `d`. Faithful
/// port of the MLX `Prodigy.step` (`slice_p = 1`, `beta1 > 0`, decoupled weight decay, no bias
/// correction / safeguard warmup).
pub struct Prodigy {
    vars: Vec<Var>,
    base_lr: f64,
    lr: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    beta3: f64,
    eps: f64,
    d: f64,
    d0: f64,
    d_max: f64,
    d_numerator: f64,
    d_coef: f64,
    growth_rate: f64,
    state: HashMap<usize, ProdigyState>,
}

impl Prodigy {
    /// `lr = lr ≥ 0.1 ? lr : 1.0` (LoRA LRs ≪ 0.1 ⇒ the knob is the Prodigy-convention 1.0), eps 1e-6,
    /// betas (0.9, 0.999), beta3 = √beta2, d0 = 1e-6, d_coef = 1, growth ∞.
    pub fn new(vars: Vec<Var>, lr: f32, weight_decay: f32) -> Self {
        let use_lr = if lr >= 0.1 { lr as f64 } else { 1.0 };
        let beta2 = 0.999;
        Self {
            vars,
            base_lr: use_lr,
            lr: use_lr,
            weight_decay: weight_decay as f64,
            beta1: 0.9,
            beta2,
            beta3: beta2.sqrt(),
            eps: 1e-6,
            d: 1e-6,
            d0: 1e-6,
            d_max: 1e-6,
            d_numerator: 0.0,
            d_coef: 1.0,
            growth_rate: f64::INFINITY,
            state: HashMap::new(),
        }
    }

    fn set_lr_scaled(&mut self, mult: f32) {
        self.lr = self.base_lr * mult as f64;
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let (beta1, beta2, beta3) = (self.beta1, self.beta2, self.beta3);
        let (d, d0, lr, eps) = (self.d, self.d0, self.lr, self.eps);
        let dlr = d * lr; // bias_correction = 1
        let d_numerator = self.d_numerator * beta3;

        // --- Pass 1: EMAs + s; accumulate the global numerator/denominator ---
        let mut delta_numerator = 0f64;
        let mut d_denom = 0f64;
        for (i, v) in self.vars.iter().enumerate() {
            let Some(g) = grads.get(v.as_tensor()) else {
                continue;
            };
            let p = v.as_tensor();
            // `entry`/`?` rather than `or_insert_with` — the state init is fallible (allocation).
            if let std::collections::hash_map::Entry::Vacant(e) = self.state.entry(i) {
                e.insert(ProdigyState {
                    exp_avg: p.zeros_like()?,
                    exp_avg_sq: p.zeros_like()?,
                    s: p.zeros_like()?,
                    p0: p.detach(),
                });
            }
            let st = self.state.get(&i).unwrap();
            // delta_numerator += (d/d0)·dlr·⟨g, p0 − p⟩
            let dot = (g * (&st.p0 - p)?)?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            delta_numerator += (d / d0) * dlr * dot;
            let exp_avg = ((&st.exp_avg * beta1)? + (g * (d * (1.0 - beta1)))?)?;
            let exp_avg_sq = ((&st.exp_avg_sq * beta2)? + (g.sqr()? * (d * d * (1.0 - beta2)))?)?;
            let s = ((&st.s * beta3)? + (g * ((d / d0) * dlr))?)?;
            let s_abs_sum = s
                .abs()?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            d_denom += s_abs_sum;
            let st = self.state.get_mut(&i).unwrap();
            st.exp_avg = exp_avg;
            st.exp_avg_sq = exp_avg_sq;
            st.s = s;
        }

        // No usable gradient signal this step — leave d/params unchanged.
        if d_denom == 0.0 {
            return Ok(());
        }

        // --- Re-estimate the adapted step d ---
        let global_d_numerator = d_numerator + delta_numerator;
        let d_hat = self.d_coef * global_d_numerator / d_denom;
        let mut d_new = d;
        if d == d0 {
            d_new = d.max(d_hat);
        }
        let d_max = self.d_max.max(d_hat);
        d_new = d_max.min(d_new * self.growth_rate);
        self.d_numerator = global_d_numerator;
        self.d = d_new;
        self.d_max = d_max;

        // --- Pass 2: Adam step. denom uses the NEW d; dlr/weight-decay use the OLD d ---
        for (i, v) in self.vars.iter().enumerate() {
            if grads.get(v.as_tensor()).is_none() {
                continue;
            }
            let p = v.as_tensor();
            let st = self.state.get(&i).expect("state created in pass 1");
            let denom = (st.exp_avg_sq.sqrt()? + (d_new * eps))?;
            let mut np = p.clone();
            if self.weight_decay != 0.0 {
                np = (np * (1.0 - self.weight_decay * dlr))?;
            }
            np = (np - (st.exp_avg.broadcast_div(&denom)? * dlr)?)?;
            v.set(&np)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn var(data: &[f32], shape: (usize, usize)) -> Var {
        Var::from_tensor(&Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()).unwrap()
    }

    #[test]
    fn normalize_collapses_aliases() {
        assert_eq!(normalize("AdamW"), "adamw");
        assert_eq!(normalize("prodigy-opt"), "prodigy");
        assert_eq!(normalize("rose_opt"), "rose");
        assert_eq!(normalize("adamw8bit"), "adamw");
        assert_eq!(normalize("adam"), "adam");
        assert!(TrainOptimizer::is_supported("Prodigy"));
        assert!(TrainOptimizer::is_supported("rose"));
        assert!(!TrainOptimizer::is_supported("lion"));
    }

    #[test]
    fn from_config_rejects_unsupported() {
        let v = vec![var(&[1.0], (1, 1))];
        assert!(TrainOptimizer::from_config("lion", v, 1e-4, 0.0).is_err());
    }

    /// Rose 2-D, centralize + stabilize OFF: θ -= lr · g / (|max_row| − min_row), per row.
    #[test]
    fn rose_2d_update_matches_closed_form_no_stabilize() {
        let p = var(&[0.0, 0.0, 0.0, 0.0], (2, 2));
        let mut rose = Rose::new(vec![p.clone()], 0.1, 0.0);
        rose.centralize = false;
        rose.stabilize = false;
        // grad rows: [1, 3] → denom = |3| − 1 = 2; [-4, -2] → denom = |-2| − (-4) = 2 + 4 = 6.
        let g = Tensor::from_vec(vec![1.0f32, 3.0, -4.0, -2.0], (2, 2), &Device::Cpu).unwrap();
        let mut grads = GradStore::default();
        grads.insert(p.as_tensor(), g);
        rose.step(&grads).unwrap();
        let out = p.as_tensor().to_vec2::<f32>().unwrap();
        // row0: -0.1 · [1,3]/2 = [-0.05, -0.15]; row1: -0.1 · [-4,-2]/6 = [0.0666.., 0.0333..].
        assert!((out[0][0] - -0.05).abs() < 1e-6);
        assert!((out[0][1] - -0.15).abs() < 1e-6);
        assert!((out[1][0] - 0.066_666_67).abs() < 1e-5);
        assert!((out[1][1] - 0.033_333_34).abs() < 1e-5);
    }

    /// AdamW takes one well-formed step that moves the parameter opposite the gradient sign.
    #[test]
    fn adamw_steps_downhill() {
        let p = var(&[1.0, -1.0], (1, 2));
        let mut opt = TrainOptimizer::from_config("adamw", vec![p.clone()], 1e-2, 0.0).unwrap();
        let g = Tensor::from_vec(vec![1.0f32, -1.0], (1, 2), &Device::Cpu).unwrap();
        let mut grads = GradStore::default();
        grads.insert(p.as_tensor(), g);
        opt.step(&grads).unwrap();
        let out = p.as_tensor().to_vec2::<f32>().unwrap();
        assert!(
            out[0][0] < 1.0 && out[0][1] > -1.0,
            "should move opposite the grad"
        );
    }

    /// clip_grad_norm scales an over-large gradient down to exactly `max_norm` and reports the
    /// pre-clip norm.
    #[test]
    fn clip_scales_to_max_norm() {
        let p = var(&[0.0, 0.0], (1, 2));
        let g = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &Device::Cpu).unwrap(); // norm 5
        let mut grads = GradStore::default();
        grads.insert(p.as_tensor(), g);
        let pre = clip_grad_norm(&mut grads, std::slice::from_ref(&p), 1.0).unwrap();
        assert!((pre - 5.0).abs() < 1e-6);
        let clipped = grads.get(p.as_tensor()).unwrap().to_vec2::<f32>().unwrap();
        let n = (clipped[0][0].powi(2) + clipped[0][1].powi(2)).sqrt();
        assert!(
            (n - 1.0).abs() < 1e-6,
            "clipped norm should be 1.0, got {n}"
        );
    }

    /// Prodigy takes a finite step and adapts `d` upward on the first step (d starts at d0=1e-6).
    #[test]
    fn prodigy_first_step_adapts_d() {
        let p = var(&[0.5, -0.5], (1, 2));
        let mut prod = Prodigy::new(vec![p.clone()], 1e-4, 0.0);
        let g = Tensor::from_vec(vec![0.2f32, -0.3], (1, 2), &Device::Cpu).unwrap();
        let mut grads = GradStore::default();
        grads.insert(p.as_tensor(), g);
        prod.step(&grads).unwrap();
        assert!(
            prod.d >= prod.d0,
            "d must not shrink below d0 on the first step"
        );
        let out = p.as_tensor().to_vec2::<f32>().unwrap();
        assert!(out[0][0].is_finite() && out[0][1].is_finite());
    }
}
