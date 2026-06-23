use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/mmvq_gguf.cu",
            "src/mmq_gguf/mmq_quantize.cu",
            "src/mmq_gguf/mmq_instance_q4_0.cu",
            "src/mmq_gguf/mmq_instance_q4_1.cu",
            "src/mmq_gguf/mmq_instance_q5_0.cu",
            "src/mmq_gguf/mmq_instance_q5_1.cu",
            "src/mmq_gguf/mmq_instance_q8_0.cu",
            "src/mmq_gguf/mmq_instance_q2_k.cu",
            "src/mmq_gguf/mmq_instance_q3_k.cu",
            "src/mmq_gguf/mmq_instance_q4_k.cu",
            "src/mmq_gguf/mmq_instance_q5_k.cu",
            "src/mmq_gguf/mmq_instance_q6_k.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        // --- sc-7544: multi-arch fatbin for the statically-linked quant/moe kernels ---------------
        // THE ONLY change vs upstream candle-kernels 0.10.2 (see VENDORED.md). These kernels are
        // compiled with `nvcc -c` (a SASS object), NOT `--ptx` like the dense build_ptx() path
        // above, so the archive holds *cubin, no PTX*. cudaforge prepends ONE `-gencode` from
        // CUDA_COMPUTE_CAP (the packaging baseline 80 → `code=sm_80`, an Ampere-only cubin). On a
        // Blackwell sm_120 GPU there is then no compatible code and nothing to JIT, so every Q4/Q8
        // QMatMul silently returns zeros (proven: candle-gen `cuda_quant_smoke` measured cos≈0 at
        // cap=80). nvcc accumulates `-gencode` flags, so these extras turn the single-arch object
        // into a true multi-arch fatbin alongside the cudaforge sm_80 baseline:
        //   sm_80 (baseline, also runs sm_86/sm_89) + sm_90 (Hopper) + sm_120 (Blackwell) SASS,
        //   plus compute_120 PTX so yet-newer archs (sm_121/sm_130…) JIT forward.
        // Mirrors the README "Packaging" arch ladder Ampere → Ada → Hopper → Blackwell. Keep
        // CUDA_COMPUTE_CAP=80 in the build recipes so the cudaforge baseline contributes sm_80; the
        // dense build_ptx() path stays compute_80-PTX-only (it JITs up correctly). Datacenter
        // Blackwell sm_100 (B100/B200) is intentionally out of scope — add `code=sm_100` here if it
        // ever becomes a target.
        .arg("-gencode=arch=compute_90,code=sm_90")
        .arg("-gencode=arch=compute_120,code=sm_120")
        .arg("-gencode=arch=compute_120,code=compute_120");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
