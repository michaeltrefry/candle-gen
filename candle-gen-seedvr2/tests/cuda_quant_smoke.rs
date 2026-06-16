//! Real-checkpoint CUDA validation for the SeedVR2 model options (sc-5927): int8/int4 DiT
//! quantization (near-lossless vs fp16) and the 7B variant (pixel-mode RoPE).
//!
//! `#[ignore]` by default (needs the weights + a GPU build). Run on the Blackwell box with:
//! ```text
//! set SEEDVR2_CKPT=D:\sceneworks-seedvr2-validate\ckpt
//! set SEEDVR2_DTYPE=bf16
//! cargo test -p candle-gen-seedvr2 --features cuda --release --test cuda_quant_smoke -- --ignored --nocapture
//! ```
//! `SEEDVR2_CKPT` is a dir holding `ema_vae_fp16.safetensors` + the 3B/7B DiT files. `SEEDVR2_DTYPE`
//! defaults to bf16 here (the worker's production dtype — the quant path's f32 QMatMul must round-trip
//! a bf16 dense pipeline). Each test skips gracefully if its required checkpoint file is absent.
//!
//! Functional validation (no bit-exact goldens on this box; the mflux reference is Apple-only):
//!   * **quant** — the Q8/Q4 upscale is highly correlated with the fp16 upscale (near-lossless / coherent)
//!     and stays sharper than a bicubic baseline (quant didn't destroy the super-resolution);
//!   * **7B** — loads with zero missing tensors (the pixel-mode-RoPE config + key renames match the
//!     real 7B checkpoint) and produces a faithful, sharper-than-bicubic upscale.

use candle_gen::candle_core::DType;
use candle_gen::gen_core::{imageops, Image, Quant};
use candle_gen_seedvr2::config::DitConfig;
use candle_gen_seedvr2::pipeline::Seedvr2Pipeline;

const DIT_FILE_3B: &str = "seedvr2_ema_3b_fp16.safetensors";
const DIT_FILE_7B: &str = "seedvr2_ema_7b_fp16.safetensors";

/// A deterministic structured LR image (gradients + a checkerboard + circles) so there is real
/// detail for the upscaler to act on.
fn synth_lr(side: usize) -> Image {
    let mut pixels = vec![0u8; side * side * 3];
    for y in 0..side {
        for x in 0..side {
            let i = (y * side + x) * 3;
            let check = (((x / 12) + (y / 12)) % 2) as u8 * 90;
            let cx = side as f32 / 2.0;
            let dr = (((x as f32 - cx).powi(2) + (y as f32 - cx).powi(2)).sqrt() * 0.18).sin();
            pixels[i] = (x * 255 / side) as u8;
            pixels[i + 1] = (40 + check as usize).min(255) as u8;
            pixels[i + 2] = (((dr + 1.0) * 0.5) * 255.0) as u8;
        }
    }
    Image {
        width: side as u32,
        height: side as u32,
        pixels,
    }
}

/// Pearson correlation over two equal-length sequences.
fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt()).max(1e-12)
}

/// Mean |gradient| (a crude sharpness proxy) of an interleaved RGB8-as-f32 buffer.
fn grad_energy(px: &[f32], h: usize, w: usize) -> f64 {
    let mut acc = 0f64;
    let mut cnt = 0u64;
    for y in 0..h {
        for x in 0..w - 1 {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                acc += (px[i + 3] - px[i]).abs() as f64;
                cnt += 1;
            }
        }
    }
    acc / cnt.max(1) as f64
}

fn ckpt_dir() -> Option<String> {
    match std::env::var("SEEDVR2_CKPT") {
        Ok(p) => Some(p),
        Err(_) => {
            eprintln!("SKIP: set SEEDVR2_CKPT to a numz/SeedVR2_comfyUI checkpoint dir");
            None
        }
    }
}

/// bf16 by default here (the worker dtype); `SEEDVR2_DTYPE=f32` forces full precision.
fn dtype() -> DType {
    match std::env::var("SEEDVR2_DTYPE").as_deref() {
        Ok("f32") | Ok("fp32") => DType::F32,
        _ => DType::BF16,
    }
}

/// Run a `src→tgt` upscale and return the RGB8 pixels as f32.
fn upscale(pipe: &Seedvr2Pipeline, lr: &Image, tgt: usize, seed: u64) -> Vec<f32> {
    let out = pipe.generate(lr, tgt, tgt, seed, 0.0).expect("generate");
    assert_eq!((out.width, out.height), (tgt as u32, tgt as u32));
    out.pixels.iter().map(|&v| v as f32).collect()
}

/// Q8/Q4 DiT quantization is near-lossless / coherent vs the fp16 (bf16) upscale, and stays sharper
/// than a bicubic baseline. Uses the 3B checkpoint (always present); the seam is variant-independent.
#[test]
#[ignore = "needs SEEDVR2_CKPT weights + a CUDA build"]
fn cuda_quant_near_lossless() {
    let Some(ckpt) = ckpt_dir() else { return };
    let dt = dtype();
    let device = candle_gen::default_device().expect("device");
    eprintln!("[seedvr2-quant] device={device:?} dtype={dt:?} ckpt={ckpt}");

    let cfg = DitConfig::seedvr2_3b();
    let (src, tgt, seed) = (256usize, 1024usize, 42u64);
    let lr = synth_lr(src);

    // fp16/bf16 dense reference.
    let pipe = Seedvr2Pipeline::load(&ckpt, DIT_FILE_3B, &cfg, dt, &device).expect("load 3B");
    let t = std::time::Instant::now();
    let reference = upscale(&pipe, &lr, tgt, seed);
    eprintln!("[seedvr2-quant] dense {src}->{tgt} in {:?}", t.elapsed());
    drop(pipe);

    let base = imageops::resize_bicubic_u8(&lr.pixels, src, src, tgt, tgt);
    let ge_ref = grad_energy(&reference, tgt, tgt);
    let ge_base = grad_energy(&base, tgt, tgt);

    // (quant level, min corr-vs-dense, label).
    for (quant, min_corr, label) in [(Quant::Q8, 0.98, "Q8"), (Quant::Q4, 0.90, "Q4")] {
        let mut q = Seedvr2Pipeline::load(&ckpt, DIT_FILE_3B, &cfg, dt, &device)
            .unwrap_or_else(|e| panic!("load 3B for {label}: {e}"));
        q.quantize(quant)
            .unwrap_or_else(|e| panic!("quantize {label}: {e}"));
        let t = std::time::Instant::now();
        let out = upscale(&q, &lr, tgt, seed);
        let corr_dense = pearson(&out, &reference);
        let corr_base = pearson(&out, &base);
        let ge_q = grad_energy(&out, tgt, tgt);
        eprintln!(
            "[seedvr2-quant] {label} {src}->{tgt} in {:?}: corr_vs_fp16={corr_dense:.4} \
             corr_vs_bicubic={corr_base:.4} grad_q={ge_q:.3} (dense {ge_ref:.3}, bicubic {ge_base:.3})",
            t.elapsed()
        );
        drop(q);

        assert!(
            corr_dense > min_corr,
            "{label} not {} vs fp16 (corr={corr_dense:.4} ≤ {min_corr})",
            if quant == Quant::Q8 {
                "near-lossless"
            } else {
                "coherent"
            }
        );
        assert!(
            ge_q > ge_base,
            "{label} upscale not sharper than bicubic (grad {ge_q:.3} ≤ {ge_base:.3}) — quant destroyed the SR"
        );
    }
}

/// The 7B variant loads (zero missing tensors → the pixel-mode-RoPE config + key renames match the
/// real 7B checkpoint) and produces a faithful, sharper-than-bicubic upscale.
#[test]
#[ignore = "needs SEEDVR2_CKPT 7B weights + a CUDA build"]
fn cuda_7b_upscale_smoke() {
    let Some(ckpt) = ckpt_dir() else { return };
    let dit_path = std::path::Path::new(&ckpt).join(DIT_FILE_7B);
    if !dit_path.exists() {
        eprintln!("SKIP: {DIT_FILE_7B} not present in {ckpt} (7B checkpoint not downloaded)");
        return;
    }
    let dt = dtype();
    let device = candle_gen::default_device().expect("device");
    eprintln!("[seedvr2-7b] device={device:?} dtype={dt:?} ckpt={ckpt}");

    let cfg = DitConfig::seedvr2_7b();
    let t_load = std::time::Instant::now();
    let pipe = Seedvr2Pipeline::load(&ckpt, DIT_FILE_7B, &cfg, dt, &device).expect("load 7B");
    eprintln!("[seedvr2-7b] loaded in {:?}", t_load.elapsed());

    let (src, tgt) = (256usize, 1024usize);
    let lr = synth_lr(src);
    let t_gen = std::time::Instant::now();
    let out_f = upscale(&pipe, &lr, tgt, 42);
    eprintln!("[seedvr2-7b] {src}->{tgt} in {:?}", t_gen.elapsed());

    let mn = out_f.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = out_f.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        mx > mn,
        "7B output is constant (degenerate): min={mn} max={mx}"
    );

    let base = imageops::resize_bicubic_u8(&lr.pixels, src, src, tgt, tgt);
    let corr = pearson(&out_f, &base);
    let ge_out = grad_energy(&out_f, tgt, tgt);
    let ge_base = grad_energy(&base, tgt, tgt);
    eprintln!(
        "[seedvr2-7b] corr_vs_bicubic={corr:.4} grad_out={ge_out:.3} grad_bicubic={ge_base:.3}"
    );
    if let Ok(p) = std::env::var("SEEDVR2_OUT_7B") {
        let u8s: Vec<u8> = out_f.iter().map(|&v| v as u8).collect();
        if let Some(im) = image::RgbImage::from_raw(tgt as u32, tgt as u32, u8s) {
            let _ = im.save(&p);
        }
    }
    assert!(
        corr > 0.7,
        "7B upscale not structurally faithful to the LR (corr={corr:.4}) — likely a 7B transcription bug"
    );
    assert!(
        ge_out > ge_base,
        "7B upscale not sharper than bicubic (grad {ge_out:.3} ≤ {ge_base:.3})"
    );
}
