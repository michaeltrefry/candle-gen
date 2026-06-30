//! Qwen-Image **2512-Fun-Controlnet-Union** (VACE) real-weight GPU validation (sc-8350) — an env-driven,
//! `#[ignore]`d integration test that drives the REAL [`QwenFunControl`] stack on the deployed hardware
//! (a `Qwen/Qwen-Image-2512` snapshot + the alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union`
//! checkpoint + a preprocessed pose/canny/depth control image). The candle sibling of the mlx
//! `tests/control_real_weights.rs`, and the analog of the InstantX [`crate::control_validate`].
//!
//! **Gate.** A VACE control should make the generation *follow* the control image, so the metric is a
//! with-control vs no-control ablation at one seed: generate **with** control (`control_scale > 0`) and
//! **without** (`control_scale = 0` → the hints are zeroed, so the forked forward is byte-identical to
//! plain txt2img — see `transformer::tests::fun_control_scale_zero_is_byte_exact_base`) and assert the
//! outputs differ meaningfully. Plus the cancel contract. "Does it match the control" is the eyeball
//! check on the written PPMs.
//!
//! **Note (Phase B, sc-8246).** The `Qwen/Qwen-Image-2512` base is a large (~40 GB) download likely
//! NOT on this box. The overlay is ungated, but without a local 2512 snapshot this test is left ready
//! and deferred — do NOT attempt the multi-tens-of-GB download. Set the env vars below to run it.
//!
//! Run (after deploying weights into a local dir):
//! ```text
//! set QWEN_FUN_BASE=...\Qwen-Image-2512       # diffusers snapshot (text_encoder/ transformer/ vae/ tokenizer/)
//! set QWEN_FUN_NET=...\Qwen-Image-2512-Fun-Controlnet-Union.safetensors   # alibaba-pai (file or dir)
//! set QWEN_FUN_HINT=...\control.ppm           # a preprocessed pose/canny/depth image at the request size
//! set QWEN_FUN_OUT=...\out
//! cargo test -p candle-gen-qwen-image --features cuda --release control_fun_validate::real_weight -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};

use crate::control_fun::{QwenFunControl, QwenFunControlPaths, QwenFunControlRequest};

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
}

/// Minimal P6 PPM reader (tolerant of a single comment line).
fn read_ppm(path: &Path) -> Image {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut i = 0usize;
    let mut tok = || -> String {
        loop {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            } else {
                break;
            }
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        String::from_utf8_lossy(&bytes[start..i]).to_string()
    };
    assert_eq!(tok(), "P6", "not a binary PPM");
    let w: usize = tok().parse().expect("ppm width");
    let h: usize = tok().parse().expect("ppm height");
    let _max: usize = tok().parse().expect("ppm maxval");
    i += 1;
    let pixels = bytes[i..i + w * h * 3].to_vec();
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

fn write_ppm(path: &Path, img: &Image) {
    let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    out.extend_from_slice(&img.pixels);
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Mean absolute per-pixel difference (0..255) between two same-size images.
fn mean_abs_diff(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.pixels.len() as f32
}

#[test]
#[ignore = "real-weight GPU validation; set QWEN_FUN_BASE/QWEN_FUN_NET/QWEN_FUN_HINT/QWEN_FUN_OUT"]
fn real_weight_fun_control() {
    let out_dir = env_path("QWEN_FUN_OUT");
    std::fs::create_dir_all(&out_dir).ok();

    let paths = QwenFunControlPaths {
        qwen_base: env_path("QWEN_FUN_BASE"),
        controlnet: env_path("QWEN_FUN_NET"),
    };
    let hint = read_ppm(&env_path("QWEN_FUN_HINT"));
    println!(
        "control hint {}x{}; loading QwenFunControl …",
        hint.width, hint.height
    );

    let t0 = std::time::Instant::now();
    let model = QwenFunControl::load(&paths).expect("load QwenFunControl");
    println!("loaded in {:?}", t0.elapsed());

    let base = QwenFunControlRequest {
        prompt: "a person standing, full body, photorealistic, studio lighting, sharp focus".into(),
        negative: "blurry, lowres, deformed, extra limbs, watermark".into(),
        width: hint.width,
        height: hint.height,
        steps: 20,
        guidance: 4.0,
        control_scale: 1.0,
        seed: 12345,
        cancel: CancelFlag::new(),
    };

    let mut noop = |_p: Progress| {};

    // With control.
    let t = std::time::Instant::now();
    let out_ctrl = model
        .generate(&base, &hint, &mut noop)
        .expect("generate (control)");
    println!("[control] {:?}", t.elapsed());
    write_ppm(&out_dir.join("qwen_fun_control.ppm"), &out_ctrl);

    // Without control (scale 0 → hints zeroed → plain txt2img at the same seed/prompt).
    let plain_req = QwenFunControlRequest {
        control_scale: 0.0,
        ..base.clone()
    };
    let t = std::time::Instant::now();
    let out_plain = model
        .generate(&plain_req, &hint, &mut noop)
        .expect("generate (no control)");
    println!("[no-control] {:?}", t.elapsed());
    write_ppm(&out_dir.join("qwen_fun_no_control.ppm"), &out_plain);

    let diff = mean_abs_diff(&out_ctrl, &out_plain);
    println!("=== Qwen-Image 2512-Fun-Controlnet-Union validation ===");
    println!("  mean abs pixel diff (control vs no-control): {diff:.2}");
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel.
    let cancelled = QwenFunControlRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let pre = model.generate(&cancelled, &hint, &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel on step 3.
    let mid = CancelFlag::new();
    let mid_req = QwenFunControlRequest {
        cancel: mid.clone(),
        ..base.clone()
    };
    let seen = Arc::new(AtomicUsize::new(0));
    let seen_cb = seen.clone();
    let mut cancel_at_3 = move |p: Progress| {
        if let Progress::Step { current, .. } = p {
            seen_cb.store(current as usize, Ordering::SeqCst);
            if current >= 3 {
                mid.cancel();
            }
        }
    };
    let res = model.generate(&mid_req, &hint, &mut cancel_at_3);
    assert!(
        matches!(res, Err(candle_gen::CandleError::Canceled)),
        "mid-cancel must return Canceled"
    );
    let steps_seen = seen.load(Ordering::SeqCst);
    assert!(
        (3..=4).contains(&steps_seen),
        "mid-cancel should stop right after step 3 (saw {steps_seen})"
    );
    println!("[cancel:mid] Err(Canceled) after {steps_seen} steps ✓");

    // The gate: the control path meaningfully changes the output (it actually conditions the image).
    assert!(
        diff > 5.0,
        "control vs no-control mean diff {diff:.2} too small — control may not be wired"
    );
    println!(
        "Qwen-Image 2512-Fun-Controlnet-Union validation PASS ✅ (eyeball the PPMs for adherence)"
    );
}
