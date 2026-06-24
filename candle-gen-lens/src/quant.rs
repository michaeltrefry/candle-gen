//! Lens DiT load-time Q4/Q8 quantization seam (sc-5117) — the candle twin of
//! `mlx-gen-lens`'s `AdaptableLinear` quant path (sc-3175), built on **candle-core's first-class GGUF
//! `QMatMul`** (the epic's stated "reuse the quant path, don't build a bespoke seam"). A [`QLinear`]
//! is a `Linear` that is **either** dense (bf16/f32) **or** GGUF-quantized; the DiT swaps its
//! compute-heavy projections to `QLinear` and [`crate::transformer::LensTransformer::quantize`] folds
//! each one to `Q4_0`/`Q8_0` in place after the (dense) weights — and any adapter merge — have loaded.
//!
//! **The quantized matmul runs in f32.** candle's CPU `QMatMul` and the CUDA dmmv *fallback* both
//! require an f32 activation (only the CUDA fast MMVQ/MMQ path takes bf16 directly), so the quantized
//! branch casts the input to f32, runs `QMatMul`, adds the (f32) bias, and casts back to the input
//! dtype. This is correct on CPU and CUDA for every batch size; the cast cost is transient and dwarfed
//! by the weight-VRAM saving (the stored Q4/Q8 blocks are ~4×/2× smaller than bf16). The surrounding
//! DiT keeps flowing bf16 between layers exactly as the dense path does.
//!
//! **Quantize from CPU, store on the DiT's device.** `QTensor::quantize_onto` requires the source on
//! the CPU, so each weight round-trips device→CPU→`quantize_onto(dev)`; the resulting `QTensor` lives
//! on the original device (CPU or CUDA) and the dense copy is dropped. This mirrors mlx's build-dense-
//! then-`quantize()`-in-place ordering, so the transient load-time peak holds the dense DiT briefly
//! before the quantized blocks replace it (the steady-state resident footprint is the quantized one —
//! the point of the story).

use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::gen_core::Quant;

/// The GGUF block type a [`Quant`] level maps to — `Q4_0` / `Q8_0` (block size 32, the candle-core
/// default GGUF quant). Every Lens DiT projection has an `in_features` divisible by 32
/// (128 / 1536 / 4096 / 11520), so the last-dim block check always passes. Shared with the gpt-oss
/// encoder quant (sc-5111) — its 2880-wide contraction is also ÷32 (but not ÷256, so only these
/// 32-block quants apply), the single source of truth for the family's `Quant → GgmlDType` mapping.
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}

/// A Linear projection that is **dense** (the loaded bf16/f32 weight) or **GGUF-quantized** (a
/// `QMatMul` over `Q4_0`/`Q8_0` blocks + the full-precision bias). Built dense; [`Self::quantize`]
/// transitions it to quantized in place. The dense and quantized forwards are the same `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    Quantized {
        matmul: QMatMul,
        /// Bias kept in f32 (added after the f32 `QMatMul`); `None` for the bias-less SwiGLU MLPs.
        bias: Option<Tensor>,
    },
}

impl QLinear {
    /// A biased dense `[out, in]` projection from `vb` (`{prefix}.weight` + `{prefix}.bias`).
    pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_gen::candle_nn::linear(
            in_dim, out_dim, vb,
        )?))
    }

    /// A bias-less dense `[out, in]` projection from `vb` (`{prefix}.weight`).
    pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Dense(candle_gen::candle_nn::linear_no_bias(
            in_dim, out_dim, vb,
        )?))
    }

    /// `x·Wᵀ + b`. Dense delegates to `candle_nn::Linear`; quantized casts the input to f32, runs the
    /// GGUF `QMatMul`, adds the f32 bias, and casts the result back to the input dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized { matmul, bias } => {
                let in_dtype = x.dtype();
                // `QMatMul` (CPU + CUDA dmmv fallback) needs a contiguous f32 activation.
                let xf = x.to_dtype(DType::F32)?.contiguous()?;
                let out = matmul.forward(&xf)?;
                let out = match bias {
                    Some(b) => out.broadcast_add(b)?,
                    None => out,
                };
                out.to_dtype(in_dtype)
            }
        }
    }

    /// Fold a dense projection to `Q4_0`/`Q8_0` in place (idempotent — a no-op if already quantized).
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.quantize_to(Some(ggml_dtype(quant)))
    }

    /// Fold a dense projection to a **specific** GGUF block type, or keep it dense when `dtype` is
    /// `None` (sc-7702 mixed-precision: the divergence-prone SwiGLU MLP stays at Q8 while the rest of
    /// the DiT is Q4_0). Idempotent. The weight is quantized on the CPU and placed back on its original
    /// device via `QTensor::quantize_onto`; the bias is promoted to f32 for the post-matmul add.
    pub fn quantize_to(&mut self, dtype: Option<GgmlDType>) -> Result<()> {
        let Some(dtype) = dtype else {
            return Ok(());
        };
        let Self::Dense(l) = self else {
            return Ok(());
        };
        let device = l.weight().device().clone();
        let w_cpu = l.weight().to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let qtensor = QTensor::quantize_onto(&w_cpu, dtype, &device)?;
        let matmul = QMatMul::from_qtensor(qtensor)?;
        let bias = match l.bias() {
            Some(b) => Some(b.to_dtype(DType::F32)?),
            None => None,
        };
        *self = Self::Quantized { matmul, bias };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense `[out, in]` `QLinear` straight from explicit weight/bias tensors (no VarBuilder), so a
    /// test can capture the dense output and quantize the *same* weights for a 1:1 comparison.
    fn dense_from(w: &Tensor, b: Option<&Tensor>) -> QLinear {
        QLinear::Dense(Linear::new(w.clone(), b.cloned()))
    }

    /// A `[64, 32]` projection (in=32 = one Q4_0/Q8_0 block per row) quantizes and forwards
    /// near-losslessly at Q8 / coherently at Q4 vs the dense f32 result — the per-linear analog of the
    /// full-DiT `dit_quant_parity` gate, runnable on CPU with no weights.
    fn quant_roundtrip(quant: Quant, min_cos: f32) {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32usize, 64usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();
        let mut lin = dense_from(&w, Some(&b));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev).unwrap();
        let dense = lin.forward(&x).unwrap();

        lin.quantize(quant).unwrap();
        assert!(
            matches!(lin, QLinear::Quantized { .. }),
            "must be quantized"
        );
        let q = lin.forward(&x).unwrap();

        let a = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let c = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nc) = (0f64, 0f64, 0f64);
        for (p, r) in a.iter().zip(c.iter()) {
            dot += (*p as f64) * (*r as f64);
            na += (*p as f64) * (*p as f64);
            nc += (*r as f64) * (*r as f64);
        }
        let cos = (dot / (na.sqrt() * nc.sqrt() + 1e-12)) as f32;
        assert!(cos > min_cos, "{quant:?} cosine {cos:.5} ≤ {min_cos}");
    }

    #[test]
    fn q8_is_near_lossless() {
        quant_roundtrip(Quant::Q8, 0.999);
    }

    #[test]
    fn q4_stays_coherent() {
        quant_roundtrip(Quant::Q4, 0.95);
    }

    /// `quantize` is idempotent — a second call on an already-quantized linear is a no-op, not a panic
    /// (the DiT's quantize pass runs uniformly over every `QLinear`).
    #[test]
    fn quantize_is_idempotent() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev).unwrap();
        let mut lin = dense_from(&w, None);
        lin.quantize(Quant::Q8).unwrap();
        lin.quantize(Quant::Q8).unwrap(); // no-op, must not error
        assert!(matches!(lin, QLinear::Quantized { bias: None, .. }));
    }
}
