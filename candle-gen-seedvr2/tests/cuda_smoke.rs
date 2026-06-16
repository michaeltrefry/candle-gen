//! Real-checkpoint CUDA smoke / functional validation for the SeedVR2 3B engine (sc-5157).
//!
//! `#[ignore]` by default (needs the weights + a GPU build). Run on the Blackwell box with:
//! ```text
//! set SEEDVR2_CKPT=D:\sceneworks-seedvr2-validate\ckpt
//! cargo test -p candle-gen-seedvr2 --features cuda --release --test cuda_smoke -- --ignored --nocapture
//! ```
//! `SEEDVR2_CKPT` is a dir holding `ema_vae_fp16.safetensors` + `seedvr2_ema_3b_fp16.safetensors`.
//! Optional: `SEEDVR2_OUT` (PNG path for the upscaled result), `SEEDVR2_DTYPE=bf16` (default f32).
//!
//! This is a *functional* validation (does the engine run end-to-end on CUDA and produce a faithful,
//! non-degenerate upscale?), not a bit-exact parity check — there are no reference goldens on this
//! box and the mflux reference is Apple-only. It asserts: finite output, correct dims, non-constant,
//! and high structural correlation with a bicubic baseline (a transcription bug in RoPE / window
//! partition / AdaLN / the conv3d decomposition would destroy that correlation).

use candle_gen::candle_core::DType;
use candle_gen::gen_core::{imageops, Image};
use candle_gen_seedvr2::config::DitConfig;
use candle_gen_seedvr2::pipeline::Seedvr2Pipeline;

const DIT_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";

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
            pixels[i] = (x * 255 / side) as u8; // R gradient
            pixels[i + 1] = (40 + check as usize).min(255) as u8; // G checkerboard
            pixels[i + 2] = (((dr + 1.0) * 0.5) * 255.0) as u8; // B rings
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

/// Mean |gradient| (a crude sharpness proxy) of an interleaved RGB8 buffer.
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

#[test]
#[ignore = "needs SEEDVR2_CKPT weights + a CUDA build"]
fn cuda_upscale_smoke() {
    let ckpt = match std::env::var("SEEDVR2_CKPT") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: set SEEDVR2_CKPT to a numz/SeedVR2_comfyUI checkpoint dir");
            return;
        }
    };
    let dtype = match std::env::var("SEEDVR2_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let device = candle_gen::default_device().expect("device");
    eprintln!("[seedvr2-smoke] device={device:?} dtype={dtype:?} ckpt={ckpt}");

    let cfg = DitConfig::seedvr2_3b();
    let t_load = std::time::Instant::now();
    let pipe = Seedvr2Pipeline::load(&ckpt, DIT_FILE, &cfg, dtype, &device).expect("load pipeline");
    eprintln!("[seedvr2-smoke] loaded in {:?}", t_load.elapsed());

    let (src, tgt) = (256usize, 1024usize); // 4× upscale; 1024 is ÷16
    let lr = synth_lr(src);

    let t_gen = std::time::Instant::now();
    let out = pipe.generate(&lr, tgt, tgt, 42, 0.0).expect("generate");
    eprintln!(
        "[seedvr2-smoke] {src}->{tgt} in {:?} -> {}x{}",
        t_gen.elapsed(),
        out.width,
        out.height
    );

    // dims + non-degenerate
    assert_eq!((out.width, out.height), (tgt as u32, tgt as u32));
    assert_eq!(out.pixels.len(), tgt * tgt * 3);
    let mn = *out.pixels.iter().min().unwrap();
    let mx = *out.pixels.iter().max().unwrap();
    assert!(
        mx > mn,
        "output is constant (degenerate): min={mn} max={mx}"
    );

    // structural faithfulness vs a bicubic baseline of the same LR
    let base = imageops::resize_bicubic_u8(&lr.pixels, src, src, tgt, tgt); // f32 [0,255] HWC
    let out_f: Vec<f32> = out.pixels.iter().map(|&v| v as f32).collect();
    let corr = pearson(&out_f, &base);
    let ge_out = grad_energy(&out_f, tgt, tgt);
    let ge_base = grad_energy(&base, tgt, tgt);
    eprintln!(
        "[seedvr2-smoke] min={mn} max={mx} corr_vs_bicubic={corr:.4} grad_out={ge_out:.3} grad_bicubic={ge_base:.3}"
    );

    if let Ok(p) = std::env::var("SEEDVR2_OUT") {
        if let Some(im) = image::RgbImage::from_raw(out.width, out.height, out.pixels.clone()) {
            if im.save(&p).is_ok() {
                eprintln!("[seedvr2-smoke] wrote {p}");
            }
        }
    }

    assert!(
        corr > 0.7,
        "upscale not structurally faithful to the LR (corr={corr:.4}) — likely a transcription bug"
    );
}
