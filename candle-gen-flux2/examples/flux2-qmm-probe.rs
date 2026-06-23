//! Minimal candle `QMatMul` correctness probe (sc-7457). The dev TE/DiT black image traces to the
//! CUDA Q4_0 quantized matmul producing NaN/garbage while the CPU path is correct. This isolates the
//! candle primitive — no model weights — sweeping {Q4_0, Q8_0} × batch sizes × {CPU, CUDA} and
//! reporting cosine vs the dense f32 reference + the non-finite count. Pinpoints whether the failure
//! is in `quantize_onto`→CUDA, the CUDA mat-vec (small M) vs mat-mat (large M) kernel, or the dtype.
//!
//! ```text
//! cargo run --release --example flux2-qmm-probe --features cuda
//! ```

use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_gen::candle_core::{Device, Module, Result, Tensor};

fn cos_and_bad(reference: &Tensor, got: &Tensor) -> Result<(f32, usize, f32)> {
    let a = reference.flatten_all()?.to_vec1::<f32>()?;
    let c = got
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (mut dot, mut na, mut nc, mut bad, mut absmax) = (0f64, 0f64, 0f64, 0usize, 0f32);
    for (p, r) in a.iter().zip(c.iter()) {
        if !r.is_finite() {
            bad += 1;
            continue;
        }
        absmax = absmax.max(r.abs());
        dot += (*p as f64) * (*r as f64);
        na += (*p as f64) * (*p as f64);
        nc += (*r as f64) * (*r as f64);
    }
    let cos = (dot / (na.sqrt() * nc.sqrt() + 1e-12)) as f32;
    Ok((cos, bad, absmax))
}

fn run(
    dt: GgmlDType,
    m: usize,
    k: usize,
    n: usize,
    cpu: &Device,
    gpu: Option<&Device>,
) -> Result<()> {
    let w = Tensor::randn(0f32, 1f32, (n, k), cpu)?; // [out, in]
    let x = Tensor::randn(0f32, 1f32, (m, k), cpu)?;
    let reference = x.matmul(&w.t()?)?; // dense f32 ref [m, n]

    // CPU quantized
    let qm_cpu = QMatMul::from_qtensor(QTensor::quantize_onto(&w, dt, cpu)?)?;
    let (cos_c, bad_c, _) = cos_and_bad(&reference, &qm_cpu.forward(&x)?)?;

    if let Some(gpu) = gpu {
        // CUDA quantized via quantize_onto (the render path)
        let qm_g = QMatMul::from_qtensor(QTensor::quantize_onto(&w, dt, gpu)?)?;
        let og = qm_g.forward(&x.to_device(gpu)?)?;
        let (cos_g, bad_g, am_g) = cos_and_bad(&reference, &og)?;
        println!(
            "{dt:?} m={m:<4} k={k} n={n}: CPU cos={cos_c:.4} bad={bad_c} | CUDA cos={cos_g:.4} bad={bad_g} absmax={am_g:.2}"
        );
    } else {
        println!("{dt:?} m={m:<4} k={k} n={n}: CPU cos={cos_c:.4} bad={bad_c} | (no CUDA)");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).ok();
    println!("CUDA available: {}", gpu.is_some());
    // Mistral TE shapes: hidden 5120, head proj 5120->4096, mlp 5120->32768. Sweep M (mat-vec vs mat-mat).
    for dt in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
        for &m in &[1usize, 8, 64, 512] {
            run(dt, m, 5120, 5120, &cpu, gpu.as_ref())?;
        }
        // a non-square TE shape (down_proj: 32768 -> 5120) at the real seq len
        run(dt, 512, 32768, 5120, &cpu, gpu.as_ref())?;
    }
    Ok(())
}
