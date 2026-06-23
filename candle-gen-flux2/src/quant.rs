//! FLUX.2 load-time Q4/Q8 quantization seam — the candle twin of `mlx-gen-flux2`'s packed-weight
//! path (sc-5917), built on candle-core's GGUF `QMatMul` (the same seam Lens uses, sc-5117). A
//! [`QLinear`] is a `Linear` that is **either** dense (f32) **or** GGUF-quantized; the dev TE + DiT
//! build their projections as `QLinear` and quantize them after the dense weights load.
//!
//! **Why CPU-staged, unlike Lens.** Lens loads its DiT dense on the GPU and quantizes in place. The
//! dev model is 32B; dense f32 (~128 GB DiT + ~96 GB TE) does not fit the 96 GB GPU even transiently,
//! so the dev quant path loads the dense weights into **system RAM** (the box has 512 GB) and
//! quantizes each projection **onto** the GPU via [`QTensor::quantize_onto`] (which requires a CPU
//! source). The dense CPU copy is dropped as each weight is folded; the GPU only ever holds the
//! quantized footprint (~Q4: ¼ of bf16). The small dense leaves that stay full precision (RMSNorms,
//! the token embedding) are moved to the GPU alongside via [`rms_norm_to`] / [`Tensor::to_device`].
//!
//! **The quantized matmul runs in f32.** candle's CPU `QMatMul` and the CUDA dmmv fallback need an
//! f32 activation; FLUX.2 already flows f32, so the cast is a no-op here. The bias (when present) is
//! kept f32 and added after the matmul.

use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, RmsNorm, VarBuilder};
use candle_gen::gen_core::Quant;

/// The GGUF block type a [`Quant`] level maps to — `Q4_0` / `Q8_0` (block size 32). Every dev TE/DiT
/// projection's `in_features` is divisible by 32 (128 / 256 / 4096 / 5120 / 6144 / 15360 / 24576 /
/// 32768), so the last-dim block check always passes. Shared mapping with the Lens DiT quant (sc-5117).
pub fn ggml_dtype(quant: Quant) -> GgmlDType {
    match quant {
        Quant::Q4 => GgmlDType::Q4_0,
        Quant::Q8 => GgmlDType::Q8_0,
    }
}

/// A Linear projection that is **dense** (the loaded f32 weight) or **GGUF-quantized** (a `QMatMul`
/// over `Q4_0`/`Q8_0` blocks + the full-precision bias). Built dense; [`Self::quantize_onto`]
/// transitions it to quantized on a target device in place. Both forwards compute `x·Wᵀ + b`.
pub enum QLinear {
    Dense(Linear),
    Quantized {
        matmul: QMatMul,
        /// Bias kept in f32 (added after the f32 `QMatMul`); `None` for the bias-less projections.
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
    /// GGUF `QMatMul`, adds the f32 bias, and casts back to the input dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized { matmul, bias } => {
                let in_dtype = x.dtype();
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

    /// Fold a dense projection to `Q4_0`/`Q8_0` **onto `device`** in place (idempotent — a no-op if
    /// already quantized). The weight is round-tripped to the CPU (the `quantize_onto` source
    /// requirement) and the resulting `QTensor` lives on `device`; the dense copy is dropped. The bias
    /// is promoted to f32 on `device`. Unlike Lens's same-device `quantize`, this targets an explicit
    /// device so the dev model can stage dense in system RAM and land quantized on the GPU.
    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        let Self::Dense(l) = self else {
            return Ok(());
        };
        let w_cpu = l.weight().to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let qtensor = QTensor::quantize_onto(&w_cpu, ggml_dtype(quant), device)?;
        let matmul = QMatMul::from_qtensor(qtensor)?;
        let bias = match l.bias() {
            Some(b) => Some(b.to_device(device)?.to_dtype(DType::F32)?),
            None => None,
        };
        *self = Self::Quantized { matmul, bias };
        Ok(())
    }
}

/// Rebuild a dense `RmsNorm` on `device` at `eps` (a no-op-cost move when already there). Used by the
/// CPU-staged dev quant path to carry the full-precision norms onto the GPU alongside the quantized
/// projections.
pub fn rms_norm_to(n: &RmsNorm, eps: f64, device: &Device) -> Result<RmsNorm> {
    Ok(RmsNorm::new(n.weight().to_device(device)?, eps))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_from(w: &Tensor, b: Option<&Tensor>) -> QLinear {
        QLinear::Dense(Linear::new(w.clone(), b.cloned()))
    }

    /// A `[64, 32]` projection quantizes and forwards near-losslessly at Q8 / coherently at Q4 vs the
    /// dense f32 result — the per-linear analog of the full-model quant parity, on CPU with no weights.
    fn quant_roundtrip(quant: Quant, min_cos: f32) {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32usize, 64usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();
        let mut lin = dense_from(&w, Some(&b));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev).unwrap();
        let dense = lin.forward(&x).unwrap();

        lin.quantize_onto(quant, &dev).unwrap();
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

    #[test]
    fn quantize_is_idempotent() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev).unwrap();
        let mut lin = dense_from(&w, None);
        lin.quantize_onto(Quant::Q8, &dev).unwrap();
        lin.quantize_onto(Quant::Q8, &dev).unwrap(); // no-op, must not error
        assert!(matches!(lin, QLinear::Quantized { bias: None, .. }));
    }
}
