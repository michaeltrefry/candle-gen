//! SD3.5 MMDiT load-time Q4/Q8 quantization seam (sc-7879, epic 7982) — the candle twin of the
//! Lens DiT quant path (sc-5117) and the FLUX.2 MMDiT quant (sc-7457), built on candle-core's
//! first-class GGUF quantization (`QTensor`). A [`QLinear`] is a `Linear` that is **either** dense
//! (bf16/f32) **or** GGUF-quantized; the SD3.5 transformer swaps its compute-heavy projections
//! (attention q/k/v/out, the joint text-stream `add_*`, the GELU MLP, the image-only `attn2`) to
//! `QLinear` and [`crate::transformer::Sd3Transformer::quantize`] folds each one to `Q4_0`/`Q8_0`
//! in place after the dense weights load.
//!
//! **The quantized matmul DEQUANTIZES the weight and runs a *dense* matmul — it does NOT take
//! candle's int8 `QMatMul` fast path (sc-7702).** That fast path (CUDA `fast_mmvq`/`fast_mmq`)
//! quantizes the *activation* to per-32-element `q8_1` blocks; a single large activation outlier
//! sets a block's int8 scale and rounds every co-located channel to zero, which made the Lens Q4 DiT
//! render solid black. Dequantizing the weight to a dense matmul keeps the activation full-precision,
//! so **uniform Q4 renders coherently** (GPU-verified on Blackwell for Lens). Each forward
//! dequantizes the stored `Q4_0`/`Q8_0` blocks to the activation dtype on the fly, so the resident
//! weight footprint stays the small quantized one (the point of the story) while the matmul sees a
//! full-precision activation. The surrounding MMDiT keeps flowing f32 between layers exactly as the
//! dense path does.
//!
//! **Quantize from CPU, store on the DiT's device.** `QTensor::quantize_onto` requires the source on
//! the CPU, so each weight round-trips device→CPU→`quantize_onto(dev)`; the resulting `QTensor`
//! lives on the original device (CPU or CUDA) and the dense copy is dropped. SD3.5 Large's DiT (~8 B
//! params) fits the GPU dense transiently, so — like Lens, unlike the 32B FLUX.2-dev — we build dense
//! on the target device and quantize in place (no CPU staging needed).

use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::gen_core::Quant;

/// The GGUF block type a [`Quant`] level maps to — `Q4_0` / `Q8_0` (block size 32, the candle-core
/// default GGUF quant). Every SD3.5 DiT projection contraction is divisible by 32 (`inner_dim`:
/// Large 2432, Medium 1536; `ff_hidden`: 9728 / 6144; `joint_attention_dim` 4096), so the last-dim
/// block check always passes. Shared single source of truth with the Lens/FLUX.2 DiT quant.
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}

/// Bytes-per-parameter of a GGUF block type, **including** the per-32-element block scale overhead.
/// `Q4_0` packs 32 weights into a 18-byte block (16 nibbles + one f16 scale) ⇒ 0.5625 B/param;
/// `Q8_0` packs 32 weights into a 34-byte block (32 int8 + one f16 scale) ⇒ 1.0625 B/param. Used by
/// [`crate::memory`] so the `minMemoryGb` estimate reflects the real on-device quantized footprint
/// (not the idealized 0.5 / 1.0).
pub fn bytes_per_param(quant: Quant) -> f64 {
    match quant {
        // 18 bytes / 32 weights.
        Quant::Q4 => 18.0 / 32.0,
        // 34 bytes / 32 weights.
        Quant::Q8 => 34.0 / 32.0,
    }
}

/// A Linear projection that is **dense** (the loaded bf16/f32 weight) or **GGUF-quantized** (the
/// `Q4_0`/`Q8_0` weight blocks + the bias, dequantized to a dense matmul each forward — sc-7702).
/// Built dense; [`Self::quantize`] transitions it to quantized in place. The dense and quantized
/// forwards are the same `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    Quantized {
        /// The GGUF-quantized weight (`Q4_0`/`Q8_0`); dequantized to the activation dtype per forward.
        weight: QTensor,
        /// The bias kept full-precision (`None` for the bias-less projections, if any).
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

    /// `x·Wᵀ + b`. Both arms run a **dense** matmul: `Dense` delegates to `candle_nn::Linear`;
    /// `Quantized` dequantizes its `Q4_0`/`Q8_0` weight (and bias) to the activation dtype and
    /// delegates likewise. Dequantizing to a dense matmul — rather than candle's int8 `QMatMul` fast
    /// path — is the sc-7702 fix (see the module docs).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized { weight, bias } => {
                let in_dtype = x.dtype();
                let w = weight.dequantize(x.device())?.to_dtype(in_dtype)?;
                let bias = match bias {
                    Some(b) => Some(b.to_dtype(in_dtype)?),
                    None => None,
                };
                Linear::new(w, bias).forward(x)
            }
        }
    }

    /// Fold a dense projection to `Q4_0`/`Q8_0` in place (idempotent — a no-op if already quantized).
    /// The weight is quantized on the CPU and placed back on its original device via
    /// `QTensor::quantize_onto`; the bias is kept full-precision for the (dense) post-matmul add.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        let Self::Dense(l) = self else {
            return Ok(());
        };
        let device = l.weight().device().clone();
        let w_cpu = l.weight().to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let weight = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), &device)?;
        let bias = l.bias().cloned();
        *self = Self::Quantized { weight, bias };
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

    /// Cosine similarity over all elements (f64).
    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// A `[64, 32]` projection (in=32 = one Q4_0/Q8_0 block per row) quantizes and forwards
    /// near-losslessly at Q8 / coherently at Q4 vs the dense f32 result — the per-linear analog of the
    /// full-DiT quant parity, on CPU with no weights.
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

        // Quantized output stays finite and tracks the dense reference.
        for v in q.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!(v.is_finite(), "{quant:?} produced a non-finite output");
        }
        let cos = cosine(&dense, &q);
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

    /// The quantize→dequantize round-trip error is bounded: dequantizing the stored blocks recovers
    /// the dense weight within the block's quant step (Q8 tight, Q4 coarser).
    #[test]
    fn dequant_round_trip_error_is_bounded() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 64usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        for (quant, max_rel) in [(Quant::Q8, 0.05f32), (Quant::Q4, 0.30f32)] {
            let w_cpu = w.to_dtype(DType::F32).unwrap();
            let qt = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), &dev).unwrap();
            let recon = qt.dequantize(&dev).unwrap();
            let num = (&w - &recon)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt();
            let den = w
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt();
            let rel = num / den;
            assert!(
                rel < max_rel,
                "{quant:?} relative recon error {rel:.4} ≥ {max_rel}"
            );
        }
    }

    /// Block-scale overhead is accounted for: Q4 ≈ 0.5625 B/param, Q8 ≈ 1.0625 B/param, and both are
    /// below bf16's 2.0.
    #[test]
    fn bytes_per_param_includes_block_scale() {
        assert!((bytes_per_param(Quant::Q4) - 0.5625).abs() < 1e-9);
        assert!((bytes_per_param(Quant::Q8) - 1.0625).abs() < 1e-9);
        assert!(bytes_per_param(Quant::Q4) < bytes_per_param(Quant::Q8));
        assert!(bytes_per_param(Quant::Q8) < 2.0);
    }
}
