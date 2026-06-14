//! The trainable LoRA/LoKr adapter seam (sc-5165).
//!
//! [`LoraLinear`] is the candle analog of a PEFT-wrapped `nn::Linear`: a **frozen** base `Linear`
//! plus an optional low-rank (LoRA) or Kronecker (LoKr) residual added *in the forward*. The residual
//! factors are held as `Var`-backed [`Tensor`]s — storage-sharing clones of the trainer's `Var`s — so:
//!
//!  * the optimizer's `Var::set` mutates that storage in place, and the **next forward reads the new
//!    values** with no re-install and no model rebuild (candle is eager: each forward re-reads the
//!    factor storage at matmul time); and
//!  * the clones keep the `Var`'s tensor-id and variable flag, so `loss.backward()` records them as
//!    leaves and `GradStore::get(var)` returns the factor gradient.
//!
//! Factors are **f32** regardless of the train dtype (master-weights pattern, per the gen-core
//! `TrainingConfig` contract); the forward casts them to the activation dtype for the matmul (a
//! differentiable cast, so grads flow back to the f32 `Var`s). The LoKr residual is reconstructed the
//! same way the inference loader does — `ΔW = (alpha/rank)·kron(w1, w2)` at f32 — so a trained adapter
//! round-trips exactly (mirrors mlx-gen's `reconstruct_lokr_delta`, SDXL f32 path).
//!
//! A model exposes its adaptable projections by implementing [`LoraHost`]; [`build_lora_targets`] /
//! [`build_lokr_targets`] then walk the host, size + initialize the factors per target, install them,
//! and return a [`LoraSet`] (the flat `Var` list for the optimizer + the per-target metadata for
//! checkpoint save). This keeps the harness model-agnostic — SDXL, Z-Image, and Wan reuse it.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Linear, Module, VarBuilder};
use rand::distr::Uniform;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::{CandleError, Result};

/// PEFT gaussian-init standard deviation for the **LoRA** `A` factor (diffusers/PEFT
/// `init_lora_weights="gaussian"` uses 0.02); the LoRA `B` leg starts at zero, so the LoRA is the
/// identity at step 0. (LoKr no longer uses this — sc-5179 moved its init to PEFT's zero-`w1` /
/// kaiming-`w2` `reset_adapter_parameters`; see [`build_lokr_targets`].)
const INIT_STD: f32 = 0.02;

/// The SDXL default LoRA target suffixes — the attention projections (matches the torch
/// `DEFAULT_LORA_TARGET_MODULES` and the MLX trainer). `to_out.0` is the first element of diffusers'
/// `to_out` `ModuleList`, so its path segment literally contains the `.0`.
pub const SDXL_ATTN_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// The PEFT key prefix SDXL writes (what `peft.save_pretrained()` emits and the SDXL loader's PEFT
/// classifier expects). The DiT families use `""` (bare dotted paths).
pub const SDXL_PEFT_PREFIX: &str = "base_model.model.unet.";

/// Which adapter parameterization a [`LoraSet`] carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    /// Standard low-rank `B·A` residual.
    Lora,
    /// LyCORIS Kronecker-product residual.
    Lokr,
}

impl AdapterKind {
    /// The `networkType` metadata string written into the adapter `.safetensors`.
    pub fn network_type(self) -> &'static str {
        match self {
            AdapterKind::Lora => "lora",
            AdapterKind::Lokr => "lokr",
        }
    }
}

/// The trainable second Kronecker leg of a LoKr residual: either a full `[out_b, in_b]` matrix or a
/// low-rank product `w2_a[out_b, rank] · w2_b[rank, in_b]`.
#[derive(Debug, Clone)]
enum LokrW2 {
    Full(Tensor),
    LowRank { a: Tensor, b: Tensor },
}

/// The trainable residual spliced into a [`LoraLinear`]'s forward. Holds storage-sharing clones of
/// the trainer's `Var`s (see the module docs) — never owned weight copies.
#[derive(Debug, Clone)]
enum Adapter {
    /// `down`: `A` `[rank, in]`; `up`: `B` `[out, rank]`; residual = `scale · (x·Aᵀ)·Bᵀ`.
    Lora {
        down: Tensor,
        up: Tensor,
        scale: f64,
    },
    /// `w1` `[out_a, in_a]`, `w2` (full/low-rank) reconstructing `[out_b, in_b]`; residual =
    /// `x · ΔWᵀ` with `ΔW = scale · kron(w1, w2)` reshaped to `[out, in]`.
    Lokr {
        w1: Tensor,
        w2: LokrW2,
        out_f: usize,
        in_f: usize,
        scale: f64,
    },
}

/// 2-D Kronecker product `kron(a[m,n], b[p,q]) = [m·p, n·q]` via broadcast — differentiable, so grads
/// flow to `a`/`b`. `out[i·p+k, j·q+l] = a[i,j]·b[k,l]`.
fn kron2d(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let (m, n) = a.dims2()?;
    let (p, q) = b.dims2()?;
    let a4 = a.reshape((m, 1, n, 1))?;
    let b4 = b.reshape((1, p, 1, q))?;
    a4.broadcast_mul(&b4)?.reshape((m * p, n * q))
}

/// A frozen base `Linear` with an optional trainable LoRA/LoKr residual. Implements
/// [`Module`](candle_nn::Module) so it drops into a vendored model exactly where an `nn::Linear` was,
/// and carries its own PEFT-style `path` (captured from the `VarBuilder` prefix at construction) so a
/// [`LoraHost`] visitor can route adapters without threading prefixes through the module tree.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    base: Linear,
    in_features: usize,
    out_features: usize,
    path: String,
    adapter: Option<Adapter>,
}

impl LoraLinear {
    /// Wrap an already-built frozen base `Linear` known to map `in_features -> out_features`, at the
    /// given PEFT module `path`.
    pub fn from_linear(
        base: Linear,
        in_features: usize,
        out_features: usize,
        path: String,
    ) -> Self {
        Self {
            base,
            in_features,
            out_features,
            path,
            adapter: None,
        }
    }

    /// The PEFT module path (e.g. `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q`),
    /// captured from the `VarBuilder` prefix at construction. Drives target matching + save keys.
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Whether a trainable residual is currently installed.
    pub fn is_adapted(&self) -> bool {
        self.adapter.is_some()
    }

    /// Install a LoRA residual. `down`/`up` are expected to be `Var`-backed (storage-sharing) f32
    /// tensors of shape `[rank, in]` / `[out, rank]`; `scale = alpha / rank`.
    pub fn install_lora(&mut self, down: Tensor, up: Tensor, scale: f64) {
        self.adapter = Some(Adapter::Lora { down, up, scale });
    }

    /// Drop any installed residual (back to the frozen base — the inference path with no adapter).
    pub fn clear(&mut self) {
        self.adapter = None;
    }

    fn install_lokr(&mut self, w1: Tensor, w2: LokrW2, scale: f64) {
        self.adapter = Some(Adapter::Lokr {
            w1,
            w2,
            out_f: self.out_features,
            in_f: self.in_features,
            scale,
        });
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.base.forward(x)?;
        match &self.adapter {
            None => Ok(y),
            // Factors are f32; cast to the activation dtype for the matmul. The cast is
            // differentiable, so grads flow back to the f32 `Var`s (master-weights). The factor
            // tensors share storage with those `Var`s, so this reads the current optimizer-updated value.
            Some(Adapter::Lora { down, up, scale }) => {
                let xd = x.dtype();
                let down = down.to_dtype(xd)?;
                let up = up.to_dtype(xd)?;
                let lora = x.broadcast_matmul(&down.t()?)?.broadcast_matmul(&up.t()?)?;
                y + (lora * *scale)?
            }
            Some(Adapter::Lokr {
                w1,
                w2,
                out_f,
                in_f,
                scale,
            }) => {
                let factor2 = match w2 {
                    LokrW2::Full(w) => w.clone(),
                    LokrW2::LowRank { a, b } => a.matmul(b)?, // [out_b, rank] · [rank, in_b]
                };
                // ΔW = scale · kron(w1, w2) at f32, reshaped to [out, in] (kron already yields that
                // shape for the linear case; the reshape is a safety no-op). Cast to the activation
                // dtype for the residual matmul x·ΔWᵀ.
                let delta = kron2d(w1, &factor2)?.reshape((*out_f, *in_f))?;
                let delta = (delta * *scale)?.to_dtype(x.dtype())?;
                y + x.broadcast_matmul(&delta.t()?)?
            }
        }
    }
}

/// Build a frozen base `Linear` (no bias) and wrap it as a [`LoraLinear`], recording the
/// `VarBuilder`'s current prefix as the PEFT path. Drop-in replacement for `candle_nn::linear_no_bias`
/// inside a vendored, trainable model.
pub fn lora_linear_no_bias(
    in_f: usize,
    out_f: usize,
    vs: VarBuilder,
) -> candle_core::Result<LoraLinear> {
    let path = vs.prefix();
    let base = candle_nn::linear_no_bias(in_f, out_f, vs)?;
    Ok(LoraLinear::from_linear(base, in_f, out_f, path))
}

/// Build a frozen base `Linear` (with bias) and wrap it as a [`LoraLinear`]. Drop-in replacement for
/// `candle_nn::linear`. The adapter residual adapts only the weight; the base bias is frozen.
pub fn lora_linear(in_f: usize, out_f: usize, vs: VarBuilder) -> candle_core::Result<LoraLinear> {
    let path = vs.prefix();
    let base = candle_nn::linear(in_f, out_f, vs)?;
    Ok(LoraLinear::from_linear(base, in_f, out_f, path))
}

/// A model that exposes its adaptable [`LoraLinear`]s for the harness to install adapters into. The
/// candle analog of the MLX `AdaptableHost`. Implementors recurse their module tree, invoking `f` once
/// per adaptable projection; each `LoraLinear` already carries its PEFT `path`, so no prefix threading
/// is needed.
pub trait LoraHost {
    fn visit_lora_mut(&mut self, f: &mut dyn FnMut(&mut LoraLinear) -> Result<()>) -> Result<()>;
}

/// One installed target: its PEFT path plus the trainer-owned factor `Var`s keyed by their save-key
/// suffix (e.g. `("lora_A.weight", a)` / `("lokr_w1", w1)`). The same `Var`s are flattened into
/// [`LoraSet::vars`] for the optimizer; here they carry the suffix the checkpoint writer needs.
#[derive(Debug, Clone)]
pub struct AdapterTarget {
    pub path: String,
    factors: Vec<(&'static str, Var)>,
}

/// The result of installing adapters onto a host: the flat `Var` list to optimize, the per-target
/// metadata to save, and the network descriptors echoed into the adapter metadata.
#[derive(Debug, Clone)]
pub struct LoraSet {
    pub kind: AdapterKind,
    pub rank: u32,
    pub alpha: f32,
    /// LoKr block-split factor (`-1` = auto); unused for plain LoRA.
    pub decompose_factor: i32,
    /// Every trainable factor, for the optimizer.
    pub vars: Vec<Var>,
    targets: Vec<AdapterTarget>,
}

impl LoraSet {
    /// `scale = alpha / rank` (the residual multiplier).
    pub fn scale(&self) -> f64 {
        self.alpha as f64 / self.rank.max(1) as f64
    }

    /// Number of adapted projections.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

/// PEFT suffix match: `suffix` matches a module `path` iff the path equals it or ends with `.{suffix}`
/// (so `to_q` matches `…attn1.to_q` but not `…attn1.to_qx`, and `to_out.0` matches `…attn1.to_out.0`).
fn path_matches(path: &str, suffix: &str) -> bool {
    path == suffix || path.ends_with(&format!(".{suffix}"))
}

/// LyCORIS dimension factorization: split `dimension` into `(a, b)` with `a·b == dimension` and
/// `a ≤ b`. `factor > 0` requests a block size (the pair containing `factor`, smaller-first); `-1`
/// (auto) picks the most balanced divisor pair. Faithful port of the MLX/LyCORIS `factorization`.
pub fn factorization(dimension: usize, factor: i32) -> (usize, usize) {
    if factor > 0 {
        let f = factor as usize;
        if dimension.is_multiple_of(f) {
            let n = dimension / f;
            return if f > n { (n, f) } else { (f, n) };
        }
    }
    // auto (or a `factor` that doesn't divide): climb to the most balanced divisor pair, bounded by
    // `factor` (= dimension when auto).
    let cap = if factor < 0 {
        dimension
    } else {
        factor as usize
    };
    let (mut m, mut n) = (1usize, dimension);
    let mut length = m + n;
    while m < n {
        let mut new_m = m + 1;
        while !dimension.is_multiple_of(new_m) {
            new_m += 1;
        }
        let new_n = dimension / new_m;
        if new_m + new_n > length || new_m > cap {
            break;
        }
        m = new_m;
        n = new_n;
        length = m + n;
    }
    if m > n {
        (n, m)
    } else {
        (m, n)
    }
}

/// Deterministic, launch-portable factor init: draw `rows·cols` `N(0, std²)` values from a seeded CPU
/// `StdRng` (NOT candle's device RNG — same reasoning as the sc-3673 initial-noise path), build the
/// tensor on CPU, and move it to `device`. Returned as a trainable f32 `Var`.
fn gaussian_var(
    rows: usize,
    cols: usize,
    std: f32,
    rng: &mut StdRng,
    device: &Device,
) -> Result<Var> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            let z: f32 = StandardNormal.sample(rng);
            std * z
        })
        .collect();
    let t = Tensor::from_vec(data, (rows, cols), &Device::Cpu)?.to_device(device)?;
    Ok(Var::from_tensor(&t)?)
}

fn zero_var(rows: usize, cols: usize, device: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::zeros(
        (rows, cols),
        DType::F32,
        device,
    )?)?)
}

/// Deterministic kaiming-uniform factor init matching `torch.nn.init.kaiming_uniform_(a=√5)`:
/// `U(±1/√fan_in)` with `fan_in = cols` (gain `√(2/(1+5)) = 1/√3`, bound `√3·gain/√fan_in = 1/√fan_in`).
/// Same seeded-CPU-`StdRng` portability as [`gaussian_var`] (sc-5179, the LoKr second-factor init).
fn kaiming_uniform_var(rows: usize, cols: usize, rng: &mut StdRng, device: &Device) -> Result<Var> {
    let bound = 1.0f32 / (cols as f32).sqrt();
    // `bound > 0` always (cols ≥ 1), so the range is well-formed; `new_inclusive` only errs on an
    // empty/NaN range.
    let dist = Uniform::new_inclusive(-bound, bound).expect("valid kaiming-uniform bounds");
    let data: Vec<f32> = (0..rows * cols).map(|_| dist.sample(rng)).collect();
    let t = Tensor::from_vec(data, (rows, cols), &Device::Cpu)?.to_device(device)?;
    Ok(Var::from_tensor(&t)?)
}

/// Install LoRA adapters on `host` for every adaptable projection whose path matches one of
/// `target_suffixes`. `A ~ N(0, 0.02²)` `[rank, in]`, `B = 0` `[out, rank]` (identity at step 0).
/// Factors are f32 on `device`; init is seeded by `seed` for reproducibility.
pub fn build_lora_targets(
    host: &mut dyn LoraHost,
    target_suffixes: &[String],
    rank: u32,
    alpha: f32,
    seed: u64,
    device: &Device,
) -> Result<LoraSet> {
    if rank == 0 {
        return Err(CandleError::Msg("lora rank must be >= 1".into()));
    }
    let r = rank as usize;
    let scale = alpha as f64 / r as f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vars: Vec<Var> = Vec::new();
    let mut targets: Vec<AdapterTarget> = Vec::new();

    host.visit_lora_mut(&mut |lin: &mut LoraLinear| {
        if !target_suffixes.iter().any(|s| path_matches(lin.path(), s)) {
            return Ok(());
        }
        let (in_f, out_f) = (lin.in_features(), lin.out_features());
        let down = gaussian_var(r, in_f, INIT_STD, &mut rng, device)?; // A [rank, in]
        let up = zero_var(out_f, r, device)?; // B [out, rank]
        lin.install_lora(down.as_tensor().clone(), up.as_tensor().clone(), scale);
        vars.push(down.clone());
        vars.push(up.clone());
        targets.push(AdapterTarget {
            path: lin.path().to_string(),
            factors: vec![("lora_A.weight", down), ("lora_B.weight", up)],
        });
        Ok(())
    })?;

    if targets.is_empty() {
        return Err(CandleError::Msg(format!(
            "no LoRA targets matched suffixes {target_suffixes:?} on the host"
        )));
    }
    Ok(LoraSet {
        kind: AdapterKind::Lora,
        rank,
        alpha,
        decompose_factor: -1,
        vars,
        targets,
    })
}

/// Install LoKr adapters on `host` for every matching projection, matching **PEFT
/// `LoKrConfig(init_weights=True)`'s `reset_adapter_parameters`** (sc-5179) so a Python LoKr learning
/// rate transfers. The weight `[out,in]` factors as `kron(w1[out_a,in_a], w2[out_b,in_b])`; `w2` is
/// low-ranked to `rank` when `rank < max(out_b,in_b)/2` (PEFT's `use_w2`). The **first** factor `w1` is
/// **zero-init**; the **second** factor `w2` (full, or both `w2_a`/`w2_b` low-rank) is **kaiming-uniform
/// `a=√5`** ⇒ `U(±1/√fan_in)`. The zeroed factor is `w1`, so the initial delta `kron(0, w2)·scale = 0`.
/// `decompose_factor` (`-1` = auto) is the block-split knob. Mirrors the MLX `build_lokr_targets`
/// (sc-5179); the residual is reconstructed at f32 (SDXL path). NOTE: replaced the prior `w1 ~ N(0,0.02)`
/// / zeroed-`w2` init (opposite zeroed factor + a fixed ~4-5× smaller scale), which forced a ~10× higher,
/// non-transferable LoKr lr; the save/round-trip format is unchanged so prior adapters still load.
pub fn build_lokr_targets(
    host: &mut dyn LoraHost,
    target_suffixes: &[String],
    rank: u32,
    alpha: f32,
    decompose_factor: i32,
    seed: u64,
    device: &Device,
) -> Result<LoraSet> {
    if rank == 0 {
        return Err(CandleError::Msg("lokr rank must be >= 1".into()));
    }
    let r = rank as usize;
    let scale = alpha as f64 / r as f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vars: Vec<Var> = Vec::new();
    let mut targets: Vec<AdapterTarget> = Vec::new();

    host.visit_lora_mut(&mut |lin: &mut LoraLinear| {
        if !target_suffixes.iter().any(|s| path_matches(lin.path(), s)) {
            return Ok(());
        }
        let (in_f, out_f) = (lin.in_features(), lin.out_features());
        let (out_a, out_b) = factorization(out_f, decompose_factor);
        let (in_a, in_b) = factorization(in_f, decompose_factor);

        // w1 = zeros [out_a, in_a] — the zeroed factor, so the initial delta is exactly 0 (PEFT).
        let w1 = zero_var(out_a, in_a, device)?;
        vars.push(w1.clone());
        let mut factors: Vec<(&'static str, Var)> = vec![("lokr_w1", w1.clone())];

        // PEFT `use_w2 = not(r < max(out_b,in_b)/2)`: low-rank w2 = w2_a @ w2_b only below half the
        // larger factor dim, else a full w2. Both factors are kaiming-init (the delta is held at 0 by w1).
        let runtime_w2 = if (r as f32) < (out_b.max(in_b) as f32) / 2.0 {
            let w2a = kaiming_uniform_var(out_b, r, &mut rng, device)?; // fan_in = r
            let w2b = kaiming_uniform_var(r, in_b, &mut rng, device)?; // fan_in = in_b
            vars.push(w2a.clone());
            vars.push(w2b.clone());
            factors.push(("lokr_w2_a", w2a.clone()));
            factors.push(("lokr_w2_b", w2b.clone()));
            LokrW2::LowRank {
                a: w2a.as_tensor().clone(),
                b: w2b.as_tensor().clone(),
            }
        } else {
            let w2 = kaiming_uniform_var(out_b, in_b, &mut rng, device)?; // fan_in = in_b
            vars.push(w2.clone());
            factors.push(("lokr_w2", w2.clone()));
            LokrW2::Full(w2.as_tensor().clone())
        };
        lin.install_lokr(w1.as_tensor().clone(), runtime_w2, scale);
        targets.push(AdapterTarget {
            path: lin.path().to_string(),
            factors,
        });
        Ok(())
    })?;

    if targets.is_empty() {
        return Err(CandleError::Msg(format!(
            "no LoKr targets matched suffixes {target_suffixes:?} on the host"
        )));
    }
    Ok(LoraSet {
        kind: AdapterKind::Lokr,
        rank,
        alpha,
        decompose_factor,
        vars,
        targets,
    })
}

/// Reconstruct the LoRA weight delta `ΔW = (alpha/rank)·scale·(B·A)` as an `[out, in]` **f32** tensor
/// — the inference-side **merge** counterpart to [`LoraLinear`]'s training **forward**, which adds the
/// mathematically-identical residual `scale·(x·Aᵀ)·Bᵀ` with the install `scale = alpha/rank`. `down`
/// is `A` `[rank, in]`, `up` is `B` `[out, rank]`; `scale` is the caller's per-adapter strength
/// (`gen_core::AdapterSpec::scale`, `1.0` reconstructs the trained delta verbatim). Computed in f32 so
/// a candle-trained adapter round-trips through inference exactly.
///
/// SDXL merges this into the dense UNet weight (`W += ΔW`) rather than adding it live: the ancestral
/// sampler is chaos-sensitive and the merged forward `(W+ΔW)·x` differs from the residual form
/// `W·x + ΔW·x` by ~1 ULP, which cascades to a visibly different image (see `candle-gen-sdxl`'s
/// adapter merge). Holding both forms to the same f32 reconstruction keeps train and infer in lockstep.
pub fn reconstruct_lora_delta(
    down: &Tensor,
    up: &Tensor,
    alpha: f32,
    rank: f32,
    scale: f32,
) -> Result<Tensor> {
    let down = down.to_dtype(DType::F32)?;
    let up = up.to_dtype(DType::F32)?;
    let ba = up.matmul(&down)?; // [out, rank] · [rank, in] → [out, in]
    let eff = (alpha as f64 / rank as f64) * scale as f64;
    Ok((ba * eff)?)
}

/// Reconstruct the LoKr weight delta `ΔW = (alpha/rank)·scale·kron(w1, w2)` reshaped to `base_shape`
/// (`[out, in]`), as **f32** — the inference-side merge counterpart to [`LoraLinear`]'s LoKr forward,
/// using the *same* [`kron2d`] reconstruction. Each Kronecker leg is either a full factor (`w1`/`w2`)
/// or a low-rank product (`w1_a·w1_b` / `w2_a·w2_b`, e.g. the trainer's zero-init `w2_a`/`w2_b` form).
/// Errors if a leg is missing. Linear-only: pass 2-D factors (the SDXL trainer adapts Linears; Tucker /
/// conv reconstruction is not handled here).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta(
    w1: Option<&Tensor>,
    w1_a: Option<&Tensor>,
    w1_b: Option<&Tensor>,
    w2: Option<&Tensor>,
    w2_a: Option<&Tensor>,
    w2_b: Option<&Tensor>,
    alpha: f32,
    rank: f32,
    scale: f32,
    base_shape: (usize, usize),
) -> Result<Tensor> {
    let f32d = |t: &Tensor| t.to_dtype(DType::F32);
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => f32d(w)?,
        (_, Some(a), Some(b)) => f32d(a)?.matmul(&f32d(b)?)?,
        _ => {
            return Err(CandleError::Msg(
                "lokr: w1 missing (need full lokr_w1 or lokr_w1_a·lokr_w1_b)".into(),
            ))
        }
    };
    let factor2 = match (w2, w2_a, w2_b) {
        (Some(w), _, _) => f32d(w)?,
        (_, Some(a), Some(b)) => f32d(a)?.matmul(&f32d(b)?)?,
        _ => {
            return Err(CandleError::Msg(
                "lokr: w2 missing (need full lokr_w2 or lokr_w2_a·lokr_w2_b)".into(),
            ))
        }
    };
    let (out_f, in_f) = base_shape;
    let delta = kron2d(&factor1, &factor2)?.reshape((out_f, in_f))?;
    let eff = (alpha as f64 / rank as f64) * scale as f64;
    Ok((delta * eff)?)
}

/// Fuse a **conv-layer** LoRA pair into a single conv-weight delta in the trained-file **NCHW**
/// `[out, in, kH, kW]` layout (sc-5225) — the conv analog of [`reconstruct_lora_delta`]. Community SDXL
/// LoRAs adapt convs (resnet `conv1`/`conv2`/`conv_shortcut`, the down/up-samplers, `conv_in`/`conv_out`)
/// by decomposing a conv into a spatial `down` (`lora_down`, `[rank, in, kH, kW]`) followed by a 1×1
/// `up` (`lora_up`, `[out, rank, 1, 1]`); the fused weight is the composition of those two convs:
///   `δ[o, i, y, x] = Σ_r up[o, r] · down[r, i, y, x]`,
/// which is exactly `up[out, rank] · down[rank, in·kH·kW]` reshaped back to `[out, in, kH, kW]` —
/// bit-identical to PEFT/diffusers' `Conv2d` LoRA fusion, uniform across 1×1 and k×k kernels — then
/// scaled by `(alpha/rank)·scale`. Computed in **f32** (the candle SDXL merge path is f32-everywhere,
/// matching [`reconstruct_lora_delta`]); the caller folds it into the conv weight (`W += δ`).
///
/// Unlike mlx-gen's `conv_lora_delta` (which returns NCHW and transposes to NHWC at the merge site,
/// because mlx stores conv weights NHWC), candle convs are already NCHW (`candle_nn::Conv2d`), so the
/// returned delta merges into the diffusers `{path}.weight` tensor with no transpose. A non-4-D factor
/// (a malformed conv LoRA) is a typed error rather than a panic on the kernel-dim reshape.
pub fn conv_lora_delta(
    down: &Tensor,
    up: &Tensor,
    alpha: f32,
    rank: f32,
    scale: f32,
) -> Result<Tensor> {
    let (ds, us) = (down.dims(), up.dims());
    if ds.len() != 4 || us.len() != 4 {
        return Err(CandleError::Msg(format!(
            "conv LoRA: expected 4-D factors (down [rank,in,kH,kW], up [out,rank,1,1]), got down \
             {ds:?} up {us:?}"
        )));
    }
    if us[1] != ds[0] {
        return Err(CandleError::Msg(format!(
            "conv LoRA: rank mismatch between factors — down[0]={} but up[1]={} (down {ds:?} up {us:?})",
            ds[0], us[1]
        )));
    }
    let (r, cin, kh, kw) = (ds[0], ds[1], ds[2], ds[3]);
    let out = us[0];
    let down2 = down.to_dtype(DType::F32)?.reshape((r, cin * kh * kw))?; // [rank, in·kH·kW]
    let up2 = up.to_dtype(DType::F32)?.reshape((out, r))?; // [out, rank]
    let ba = up2.matmul(&down2)?; // [out, in·kH·kW]
    let eff = (alpha as f64 / rank as f64) * scale as f64;
    Ok((ba * eff)?.reshape((out, cin, kh, kw))?)
}

/// Reconstruct a **LoHa** (LyCORIS Hadamard-product) weight delta `ΔW = scale · ((w1_a·w1_b) ⊙
/// (w2_a·w2_b))` as an `[out, in]` **f32** tensor (sc-5225). Third-party LoHa decomposes a delta as the
/// elementwise product of TWO low-rank products (vs LoKr's Kronecker). `w1_a`/`w2_a` are `[out, rank]`,
/// `w1_b`/`w2_b` are `[rank, in]`; `scale` is the fully-effective multiplier (the lycoris `alpha/rank`
/// times the caller's per-adapter strength). Mirrors LyCORIS `LohaModule.get_weight` / `HadaWeight`
/// (`w1d=hada_w1_b, w1u=hada_w1_a`, so the product is `(hada_w1_a·hada_w1_b) ⊙ (hada_w2_a·hada_w2_b)`).
///
/// **Linear-only**: pass 2-D factors. The conv/tucker LoHa form (lycoris `use_cp`, `hada_t1`/`hada_t2`)
/// is out of the candle SDXL adapter surface — like third-party LoKr, LoHa merges only into the
/// attention/proj Linears (the conv surface is LoRA-only), so a conv-shaped LoHa is surfaced as skipped.
pub fn reconstruct_loha_delta(
    w1_a: &Tensor,
    w1_b: &Tensor,
    w2_a: &Tensor,
    w2_b: &Tensor,
    scale: f32,
    base_shape: (usize, usize),
) -> Result<Tensor> {
    let f32d = |t: &Tensor| t.to_dtype(DType::F32);
    let m1 = f32d(w1_a)?.matmul(&f32d(w1_b)?)?; // [out, rank]·[rank, in] → [out, in]
    let m2 = f32d(w2_a)?.matmul(&f32d(w2_b)?)?;
    let (out_f, in_f) = base_shape;
    let delta = (m1 * m2)?.reshape((out_f, in_f))?;
    Ok((delta * scale as f64)?)
}

/// Collect a target's factor tensors as CPU/f32 `(key, tensor)` save entries under `prefix`.
fn factor_entries(set: &LoraSet, prefix: &str) -> Result<Vec<(String, Tensor)>> {
    let mut out = Vec::with_capacity(set.targets.len() * 3);
    for t in &set.targets {
        for (suffix, var) in &t.factors {
            let v = var
                .as_tensor()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            out.push((format!("{prefix}{}.{suffix}", t.path), v));
        }
    }
    Ok(out)
}

fn write_safetensors(
    tensors: Vec<(String, Tensor)>,
    meta: HashMap<String, String>,
    path: &Path,
) -> Result<()> {
    safetensors::serialize_to_file(tensors, Some(meta), path)
        .map_err(|e| CandleError::Msg(format!("save adapter {}: {e}", path.display())))?;
    Ok(())
}

/// Write a LoRA [`LoraSet`] as a PEFT-format `.safetensors`: keys `{prefix}{path}.lora_A.weight`
/// (`[rank, in]`), `{prefix}{path}.lora_B.weight` (`[out, rank]`), and a per-target scalar
/// `{prefix}{path}.alpha`, plus `networkType`/`rank`/`alpha` metadata (the candle save path that
/// candle-core's own `save` cannot produce — it passes `None` for metadata). `prefix` is
/// [`SDXL_PEFT_PREFIX`] for SDXL, `""` for the DiT families. Matches the MLX `save_lora_peft`.
pub fn save_lora_peft(
    set: &LoraSet,
    prefix: &str,
    extra_meta: &HashMap<String, String>,
    path: &Path,
) -> Result<()> {
    if set.kind != AdapterKind::Lora {
        return Err(CandleError::Msg(
            "save_lora_peft called on a non-LoRA set".into(),
        ));
    }
    let mut tensors = factor_entries(set, prefix)?;
    // Per-target scalar `.alpha` (PEFT reload contract).
    for t in &set.targets {
        tensors.push((
            format!("{prefix}{}.alpha", t.path),
            Tensor::from_vec(vec![set.alpha], (1,), &Device::Cpu)?,
        ));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".into(), set.kind.network_type().into());
    meta.insert("rank".into(), set.rank.to_string());
    meta.insert("alpha".into(), set.alpha.to_string());
    for (k, v) in extra_meta {
        meta.entry(k.clone()).or_insert_with(|| v.clone());
    }
    write_safetensors(tensors, meta, path)
}

/// Write a LoKr [`LoraSet`] as `.safetensors`: bare keys `{path}.lokr_w1` + (`lokr_w2` |
/// `lokr_w2_a`/`lokr_w2_b`), with `networkType`/`rank`/`alpha`/`decomposeFactor` metadata. No key
/// prefix (the SDXL LoKr loader accepts a `base_model.model.unet.` prefix but bare keys resolve for
/// every family). Matches the MLX `save_lokr` (integer rank/alpha rendering).
pub fn save_lokr(set: &LoraSet, extra_meta: &HashMap<String, String>, path: &Path) -> Result<()> {
    if set.kind != AdapterKind::Lokr {
        return Err(CandleError::Msg(
            "save_lokr called on a non-LoKr set".into(),
        ));
    }
    let tensors = factor_entries(set, "")?;
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".into(), set.kind.network_type().into());
    meta.insert("rank".into(), (set.rank as i64).to_string());
    meta.insert("alpha".into(), (set.alpha as i64).to_string());
    meta.insert("decomposeFactor".into(), set.decompose_factor.to_string());
    for (k, v) in extra_meta {
        meta.entry(k.clone()).or_insert_with(|| v.clone());
    }
    write_safetensors(tensors, meta, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    fn fixed_linear(weight: &[f32], out_f: usize, in_f: usize) -> LoraLinear {
        let w = Tensor::from_vec(weight.to_vec(), (out_f, in_f), &Device::Cpu).unwrap();
        LoraLinear::from_linear(Linear::new(w, None), in_f, out_f, "test.to_q".into())
    }

    #[test]
    fn no_adapter_is_base_linear() {
        let lin = fixed_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![3.0, 5.0]);
    }

    #[test]
    fn zero_b_residual_is_identity() {
        let mut lin = fixed_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let down = Tensor::from_vec(vec![0.5f32, -0.3, 0.1, 0.2], (2, 2), &Device::Cpu).unwrap();
        let up = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        lin.install_lora(down, up, 1.0);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![3.0, 5.0]);
    }

    #[test]
    fn lora_residual_math() {
        let mut lin = fixed_linear(&[0.0, 0.0, 0.0, 0.0], 2, 2);
        let down = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let up = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        lin.install_lora(down, up, 2.0);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![6.0, 10.0]);
    }

    #[test]
    fn backward_reaches_factors() {
        let w = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let mut lin = LoraLinear::from_linear(Linear::new(w, None), 2, 2, "t".into());
        let down = Var::from_tensor(
            &Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (2, 2), &Device::Cpu).unwrap(),
        )
        .unwrap();
        let up = Var::from_tensor(
            &Tensor::from_vec(vec![0.5f32, 0.6, 0.7, 0.8], (2, 2), &Device::Cpu).unwrap(),
        )
        .unwrap();
        lin.install_lora(down.as_tensor().clone(), up.as_tensor().clone(), 1.0);
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let loss = lin.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        assert!(
            grads.get(down.as_tensor()).is_some(),
            "A factor must receive a gradient"
        );
        assert!(
            grads.get(up.as_tensor()).is_some(),
            "B factor must receive a gradient"
        );
    }

    #[test]
    fn optimizer_update_seen_without_reinstall() {
        let w = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
        let mut lin = LoraLinear::from_linear(Linear::new(w, None), 1, 1, "t".into());
        let down = Var::from_tensor(&Tensor::from_vec(vec![1.0f32], (1, 1), &Device::Cpu).unwrap())
            .unwrap();
        let up = Var::from_tensor(&Tensor::from_vec(vec![1.0f32], (1, 1), &Device::Cpu).unwrap())
            .unwrap();
        lin.install_lora(down.as_tensor().clone(), up.as_tensor().clone(), 1.0);
        let x = Tensor::from_vec(vec![2.0f32], (1, 1), &Device::Cpu).unwrap();
        let y0 = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert_eq!(y0, 2.0);
        up.set(&Tensor::from_vec(vec![3.0f32], (1, 1), &Device::Cpu).unwrap())
            .unwrap();
        let y1 = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert_eq!(y1, 6.0);
    }

    #[test]
    fn kron2d_matches_reference() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![0.0f32, 5.0, 6.0, 7.0], (2, 2), &Device::Cpu).unwrap();
        let k = kron2d(&a, &b).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(
            k,
            vec![
                vec![0.0, 5.0, 0.0, 10.0],
                vec![6.0, 7.0, 12.0, 14.0],
                vec![0.0, 15.0, 0.0, 20.0],
                vec![18.0, 21.0, 24.0, 28.0],
            ]
        );
    }

    /// A zero second Kronecker leg ⇒ zero delta ⇒ the LoKr adapter is the identity over the base at
    /// init (the property training relies on), same as LoRA's `B = 0`.
    #[test]
    fn lokr_zero_init_is_identity() {
        let mut lin = fixed_linear(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let w1 = Tensor::from_vec(vec![0.3f32, -0.1, 0.2, 0.4], (2, 2), &Device::Cpu).unwrap();
        let w2 = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
        lin.install_lokr(w1, LokrW2::Full(w2), 1.0);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![3.0, 5.0]);
    }

    /// sc-5179 — the native LoKr init must match PEFT `LoKrConfig(init_weights=True)`'s
    /// `reset_adapter_parameters` (w1 zero-init, w2 kaiming-uniform `a=√5`), mirroring the MLX
    /// `build_lokr_init_matches_peft_reset_adapter_parameters`, so a Python LoKr lr transfers.
    #[test]
    fn build_lokr_init_matches_peft_reset_adapter_parameters() {
        struct OneHost(LoraLinear);
        impl LoraHost for OneHost {
            fn visit_lora_mut(
                &mut self,
                f: &mut dyn FnMut(&mut LoraLinear) -> Result<()>,
            ) -> Result<()> {
                f(&mut self.0)
            }
        }
        // out=64 → fac(-1)=(8,8); in=48 → (6,8). out_b = in_b = 8. rank 3 < max(8,8)/2 = 4 → low-rank.
        let w = Tensor::zeros((64, 48), DType::F32, &Device::Cpu).unwrap();
        let lin = LoraLinear::from_linear(Linear::new(w, None), 48, 64, "w".into());
        let mut host = OneHost(lin);
        let set =
            build_lokr_targets(&mut host, &["w".to_string()], 3, 3.0, -1, 7, &Device::Cpu).unwrap();
        assert_eq!(set.targets.len(), 1);
        let factors = &set.targets[0].factors;
        let get = |name: &str| {
            factors
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.as_tensor().clone())
        };
        let maxabs = |t: &Tensor| {
            t.abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        // w1 is the ZEROED factor (PEFT), shape [out_a=8, in_a=6].
        let w1 = get("lokr_w1").expect("lokr_w1");
        assert_eq!(w1.dims(), &[8, 6]);
        assert_eq!(
            maxabs(&w1),
            0.0,
            "w1 must be zero-init (PEFT), not N(0,0.02)"
        );

        // w2 is low-rank kaiming: no full w2; w2_a [8,3] bound 1/√3, w2_b [3,8] bound 1/√8.
        assert!(
            get("lokr_w2").is_none(),
            "rank 3 < max(8,8)/2 → low-rank w2"
        );
        let w2a = get("lokr_w2_a").expect("lokr_w2_a");
        let w2b = get("lokr_w2_b").expect("lokr_w2_b");
        let bound_a = 1.0f32 / 3f32.sqrt(); // ≈ 0.577
        let bound_b = 1.0f32 / 8f32.sqrt(); // ≈ 0.354
        assert!(
            maxabs(&w2a) <= bound_a + 1e-6 && maxabs(&w2a) > bound_a * 0.3,
            "w2_a must be kaiming U(±1/√rank): max {} vs bound {bound_a}",
            maxabs(&w2a)
        );
        assert!(
            maxabs(&w2b) <= bound_b + 1e-6 && maxabs(&w2b) > bound_b * 0.3,
            "w2_b must be kaiming U(±1/√in_b): max {} vs bound {bound_b}",
            maxabs(&w2b)
        );
    }

    /// LoKr residual on a 1×1 base: out=in=1 factor as (1,1)⊗(1,1); ΔW = scale·(w1·w2). base 0,
    /// w1=2, w2=3, scale=1, x=5 ⇒ y = 5·(2·3) = 30.
    #[test]
    fn lokr_residual_math() {
        let mut lin = fixed_linear(&[0.0], 1, 1);
        let w1 = Tensor::from_vec(vec![2.0f32], (1, 1), &Device::Cpu).unwrap();
        let w2 = Tensor::from_vec(vec![3.0f32], (1, 1), &Device::Cpu).unwrap();
        lin.install_lokr(w1, LokrW2::Full(w2), 1.0);
        let x = Tensor::from_vec(vec![5.0f32], (1, 1), &Device::Cpu).unwrap();
        let y = lin
            .forward(&x)
            .unwrap()
            .i(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert_eq!(y, 30.0);
    }

    /// The train→infer round-trip guarantee for LoRA: the weight delta the inference loader merges
    /// ([`reconstruct_lora_delta`]) must equal the residual the training forward adds. With base `W`,
    /// `forward(x) - W·x` is the residual; reconstructing the delta and applying `x·ΔWᵀ` must match —
    /// proving the merged-weight forward `(W+ΔW)·x` reproduces the trained adapter.
    #[test]
    fn reconstruct_lora_delta_matches_forward_residual() {
        let w = Tensor::from_vec(
            vec![0.5f32, -0.2, 0.1, 0.3, -0.4, 0.6],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap(); // base [out=2, in=3]
        let down = Tensor::from_vec(
            vec![0.1f32, 0.2, -0.3, 0.4, 0.5, -0.6],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap(); // A [rank=2, in=3]
        let up = Tensor::from_vec(vec![0.7f32, -0.8, 0.9, 1.0], (2, 2), &Device::Cpu).unwrap(); // B [out=2, rank=2]
        let (alpha, rank) = (4.0f32, 2.0f32); // train scale = alpha/rank = 2.0
        let mut lin = LoraLinear::from_linear(Linear::new(w.clone(), None), 3, 2, "t".into());
        lin.install_lora(down.clone(), up.clone(), (alpha / rank) as f64);

        let x = Tensor::from_vec(vec![1.0f32, -2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let residual = (lin.forward(&x).unwrap() - x.matmul(&w.t().unwrap()).unwrap()).unwrap();
        // scale 1.0 ⇒ ΔW = (alpha/rank)·B·A, the exact delta the forward applies.
        let delta = reconstruct_lora_delta(&down, &up, alpha, rank, 1.0).unwrap();
        let from_delta = x.matmul(&delta.t().unwrap()).unwrap();
        let diff = (residual - from_delta)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "reconstruct_lora_delta diverged from forward residual: {diff}"
        );
    }

    /// The train→infer round-trip guarantee for LoKr: [`reconstruct_lokr_delta`] must equal the
    /// residual the LoKr forward adds. 2×2 base factored `1×1 ⊗ 2×2`.
    #[test]
    fn reconstruct_lokr_delta_matches_forward_residual() {
        let w = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (2, 2), &Device::Cpu).unwrap();
        let w1 = Tensor::from_vec(vec![2.0f32], (1, 1), &Device::Cpu).unwrap(); // [out_a=1, in_a=1]
        let w2 = Tensor::from_vec(vec![0.5f32, -0.3, 0.2, 0.6], (2, 2), &Device::Cpu).unwrap(); // [out_b=2, in_b=2]
        let (alpha, rank) = (3.0f32, 1.0f32); // train scale = alpha/rank = 3.0
        let mut lin = LoraLinear::from_linear(Linear::new(w.clone(), None), 2, 2, "t".into());
        lin.install_lokr(w1.clone(), LokrW2::Full(w2.clone()), (alpha / rank) as f64);

        let x = Tensor::from_vec(vec![1.5f32, -0.5], (1, 2), &Device::Cpu).unwrap();
        let residual = (lin.forward(&x).unwrap() - x.matmul(&w.t().unwrap()).unwrap()).unwrap();
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            alpha,
            rank,
            1.0,
            (2, 2),
        )
        .unwrap();
        let from_delta = x.matmul(&delta.t().unwrap()).unwrap();
        let diff = (residual - from_delta)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "reconstruct_lokr_delta diverged from forward residual: {diff}"
        );
    }

    /// Low-rank LoKr legs reconstruct via the `a·b` product — the trainer's zero-init `w2_a`/`w2_b`
    /// form (and a community low-rank `w1_a`/`w1_b`). A zero `w2_b` ⇒ zero delta, the init identity.
    #[test]
    fn reconstruct_lokr_delta_low_rank_legs() {
        let w1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let w2a = Tensor::from_vec(vec![0.5f32, 0.7], (2, 1), &Device::Cpu).unwrap(); // [out_b=2, r=1]
        let w2b_zero = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap(); // [r=1, in_b=2]
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            None,
            Some(&w2a),
            Some(&w2b_zero),
            2.0,
            1.0,
            1.0,
            (4, 4),
        )
        .unwrap();
        let max = delta
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(max, 0.0, "zero w2_b must give a zero LoKr delta");
        // A non-zero w2_b yields the kron product scaled by alpha/rank.
        let w2b = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &Device::Cpu).unwrap();
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            None,
            Some(&w2a),
            Some(&w2b),
            2.0,
            1.0,
            1.0,
            (4, 4),
        )
        .unwrap();
        assert_eq!(delta.dims(), &[4, 4]);
        let nonzero = delta
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(nonzero > 0.0, "non-zero w2_b must give a non-zero delta");
    }

    #[test]
    fn factorization_matches_lycoris() {
        assert_eq!(factorization(320, -1), (16, 20));
        assert_eq!(factorization(64, -1), (8, 8));
        // factor>0: the pair containing `factor`, smaller-first (MLX convention).
        assert_eq!(factorization(320, 4), (4, 80));
        assert_eq!(factorization(320, 80), (4, 80));
    }

    #[test]
    fn path_match_rules() {
        assert!(path_matches("a.b.attn1.to_q", "to_q"));
        assert!(path_matches("a.b.attn1.to_out.0", "to_out.0"));
        assert!(!path_matches("a.b.attn1.to_qx", "to_q"));
        assert!(path_matches("to_q", "to_q"));
    }

    /// sc-5225: a 1×1 conv LoRA (rank 2, in 2, out 2). `down`/`up` are `[*, *, 1, 1]`; the fused delta
    /// is `Σ_r up[o,r]·down[r,i]`, scaled by `alpha/rank`. Hand-computed independently (mirrors mlx):
    ///   down2 = [[1,2],[3,4]] (rank,in); up2 = [[5,6],[7,8]] (out,rank)
    ///   δ[0,0]=5·1+6·3=23  δ[0,1]=5·2+6·4=34  δ[1,0]=7·1+8·3=31  δ[1,1]=7·2+8·4=46
    ///   eff = alpha/rank = 4/2 = 2 → [[46,68],[62,92]].
    #[test]
    fn conv_lora_delta_one_by_one_matches_hand_fold() {
        let dev = Device::Cpu;
        let down = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2, 1, 1), &dev).unwrap();
        let up = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2, 1, 1), &dev).unwrap();
        let delta = conv_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        assert_eq!(delta.dims(), &[2, 2, 1, 1]);
        let got = delta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![46.0, 68.0, 62.0, 92.0]);
    }

    /// sc-5225: a k×k (here 2×2) conv LoRA with rank 1 reduces to `δ[o,i,y,x] = up[o]·down[0,i,y,x]` —
    /// proving the spatial kernel is preserved (not collapsed). in=1, out=2.
    ///   down[0,0,:,:] = [[1,2],[3,4]]; up = [10, 20]
    ///   δ[0] = 10·[1,2,3,4] = [10,20,30,40];  δ[1] = 20·[...] = [20,40,60,80].
    /// The user scale composes multiplicatively (scale 0 ⇒ a zero delta ⇒ no-op merge).
    #[test]
    fn conv_lora_delta_kxk_rank1_broadcasts_spatial_kernel() {
        let dev = Device::Cpu;
        let down = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &dev).unwrap();
        let up = Tensor::from_vec(vec![10.0f32, 20.0], (2, 1, 1, 1), &dev).unwrap();
        let delta = conv_lora_delta(&down, &up, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(delta.dims(), &[2, 1, 2, 2]);
        let got = delta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![10.0, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0]);
        let zero = conv_lora_delta(&down, &up, 1.0, 1.0, 0.0).unwrap();
        let zmax = zero
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(zmax, 0.0, "scale 0 must give a zero conv delta");
    }

    /// A malformed conv LoRA with 2-D factors must surface a typed error, not panic on the kernel-dim
    /// reshape; a rank mismatch between the factors is rejected too (mirrors mlx-gen's F-006).
    #[test]
    fn conv_lora_delta_rejects_non_4d_or_mismatched_rank() {
        let dev = Device::Cpu;
        let down2d = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).unwrap();
        let up = Tensor::from_vec(vec![10.0f32, 20.0], (2, 1, 1, 1), &dev).unwrap();
        let err = conv_lora_delta(&down2d, &up, 1.0, 1.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("4-D factors"), "got: {err}");

        let down = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 1, 2), &dev).unwrap(); // rank 1
        let up_bad = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2, 1, 1), &dev).unwrap(); // rank 2
        let err = conv_lora_delta(&down, &up_bad, 1.0, 1.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank mismatch"), "got: {err}");
    }

    /// sc-5225: the LoHa delta is `((w1_a·w1_b) ⊙ (w2_a·w2_b))·scale`. Hand-computed on a 2×2 base,
    /// rank 1: w1_a=[1;2], w1_b=[3,4] ⇒ m1=[[3,4],[6,8]]; w2_a=[1;0], w2_b=[5,6] ⇒ m2=[[5,6],[0,0]];
    /// m1⊙m2=[[15,24],[0,0]]; scale 0.5 ⇒ [[7.5,12],[0,0]].
    #[test]
    fn reconstruct_loha_delta_matches_hand_fold() {
        let dev = Device::Cpu;
        let w1a = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &dev).unwrap();
        let w1b = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &dev).unwrap();
        let w2a = Tensor::from_vec(vec![1.0f32, 0.0], (2, 1), &dev).unwrap();
        let w2b = Tensor::from_vec(vec![5.0f32, 6.0], (1, 2), &dev).unwrap();
        let delta = reconstruct_loha_delta(&w1a, &w1b, &w2a, &w2b, 0.5, (2, 2)).unwrap();
        assert_eq!(delta.dims(), &[2, 2]);
        let got = delta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![7.5, 12.0, 0.0, 0.0]);
    }
}
