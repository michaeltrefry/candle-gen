//! Self-contained CUDA quantized-matmul regression smoke (sc-7544).
//!
//! candle-kernels compiles its GGUF `QMatMul` kernels (`mmq_gguf/*`, the Q4_0/Q8_0/k-quant matmuls)
//! into a **static `libmoe.a`** via cudaforge `build_lib()` → `nvcc -c -gencode=arch=…,code=sm_XX`,
//! i.e. **SASS, no PTX**. Built at the old `CUDA_COMPUTE_CAP=80` packaging baseline the archive holds
//! *only* sm_80 cubin; on a Blackwell sm_120 GPU there is no compatible code and nothing to JIT, so
//! the quant matmul **silently no-ops to zeros/garbage** (cos≈0 vs the CPU reference) while dense
//! (PTX) kernels JIT up fine. The packaging fix is a **multi-arch fatbin** that embeds native sm_120
//! SASS alongside the sm_80 baseline + forward-JIT PTX (see README "Packaging"); with it CUDA matches
//! the CPU reference (cos≈1).
//!
//! This test is the canary so that regression can't return silently. It is **weightless** (no
//! checkpoints) and fast, so it runs unconditionally in the local CUDA gate
//! (`scripts/check-cuda.ps1` → `cargo test --workspace --features cuda`). On a CPU/Metal build it is
//! a graceful no-op — the bug only exists on the CUDA backend, so the check is meaningful only when
//! `default_device()` resolves to CUDA.

use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_gen::candle_core::{Device, Module, Tensor};
use candle_gen::default_device;

/// Deterministic, launch-portable pseudo-random f32 in roughly [-1, 1] (splitmix64-style hash of the
/// index). Avoids a device RNG so the CPU reference and the CUDA result quantize byte-identical data.
fn pseudo_random(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // map the top 24 bits to [-1, 1)
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Cosine similarity of two tensors over all elements (flattened, on the CPU).
fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let na = (&a * &a)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let nb = (&b * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

fn all_finite(t: &Tensor) -> bool {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())
}

/// The GGUF Q4_0/Q8_0 `QMatMul` on the CUDA device matches the CPU reference (cos≈1, all-finite).
///
/// On the broken sm_80-SASS-only packaging the CUDA result is all-zeros/garbage (cos≈0) — this fails
/// loudly. With the multi-arch fatbin (native sm_120 cubin) it passes.
#[test]
fn cuda_qmatmul_matches_cpu() {
    let device = default_device().expect("default device");
    if !device.is_cuda() {
        eprintln!("SKIP cuda_qmatmul_matches_cpu: default_device()={device:?} is not CUDA");
        return;
    }
    eprintln!("[quant-smoke] device={device:?}");

    // out=N, in=K, rows=M. K is a multiple of 32 (Q4_0/Q8_0 block) and 256 (k-quant QK_K), so the
    // shapes are valid for every GGUF dtype should we extend the sweep later.
    let (n, k, m) = (512usize, 1024usize, 8usize);
    let w_cpu = Tensor::from_vec(pseudo_random(n * k), (n, k), &Device::Cpu).expect("w");
    let x_cpu = Tensor::from_vec(pseudo_random(m * k), (m, k), &Device::Cpu).expect("x");

    // Q4 quantization-noise floor is wider; Q8 is near-lossless.
    for (dtype, min_cos, label) in [
        (GgmlDType::Q8_0, 0.999f32, "Q8_0"),
        (GgmlDType::Q4_0, 0.99f32, "Q4_0"),
    ] {
        // CPU reference: quantize + matmul entirely on the CPU.
        let mm_cpu = QMatMul::from_qtensor(QTensor::quantize(&w_cpu, dtype).expect("cpu quantize"))
            .expect("cpu qmatmul");
        let y_cpu = mm_cpu.forward(&x_cpu).expect("cpu forward");

        // CUDA: quantize the SAME cpu source straight onto the device, matmul on the device.
        let mm_cuda = QMatMul::from_qtensor(
            QTensor::quantize_onto(&w_cpu, dtype, &device).expect("cuda quantize_onto"),
        )
        .expect("cuda qmatmul");
        let x_cuda = x_cpu.to_device(&device).expect("x->cuda");
        let y_cuda = mm_cuda
            .forward(&x_cuda)
            .expect("cuda forward")
            .to_device(&Device::Cpu)
            .expect("y->cpu");

        let cos = cosine(&y_cpu, &y_cuda);
        let finite = all_finite(&y_cuda);
        eprintln!("[quant-smoke] {label}: cos(CUDA, CPU)={cos:.5} all_finite={finite}");

        assert!(
            finite,
            "{label} CUDA QMatMul produced non-finite values — likely no compatible cubin for this \
             arch (sm_80-SASS-only build on a newer GPU). Rebuild with the multi-arch fatbin."
        );
        assert!(
            cos > min_cos,
            "{label} CUDA QMatMul does not match the CPU reference (cos={cos:.5} ≤ {min_cos}). On \
             Blackwell sm_120 this means candle-kernels' libmoe.a has no native sm_120 cubin (the \
             quant kernels silently no-op). Build the multi-arch fatbin — see README \"Packaging\"."
        );
    }
}
