//! Z-Image Fun-ControlNet (strict-pose) real-weight GPU validation (sc-5489, epic 5480) — an
//! env-driven, `#[ignore]`d integration test that drives the REAL [`ZImageControl`] stack on the
//! deployed hardware (a `Tongyi-MAI/Z-Image-Turbo` snapshot + the
//! `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` checkpoint + a rendered pose skeleton). The
//! Z-Image sibling of the Qwen/Kolors strict-pose harnesses.
//!
//! **Gate.** A strict-pose ControlNet should make the generation *follow* the skeleton, so the metric
//! is a with-control vs no-control ablation: generate twice at one seed — **with** control
//! (`control_scale > 0`) and **without** (`control_scale = 0` → the VACE hints contribute zero, so the
//! forward reduces to the base Z-Image txt2img) — and assert the outputs differ meaningfully. Plus the
//! cancel contract. The "does it match the pose" judgement is the eyeball check on the written PPMs.
//!
//! Run (after deploying weights into a local dir):
//! ```text
//! set ZIMG_CTRL_BASE=...\Z-Image-Turbo          # tokenizer/ text_encoder/ transformer/ vae/
//! set ZIMG_CTRL_NET=...\Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors   # file or dir
//! set ZIMG_CTRL_POSE=...\skeleton.ppm           # a rendered OpenPose skeleton at the request size (P6)
//! set ZIMG_CTRL_OUT=...\out
//! cargo test -p candle-gen-z-image --features cuda --release control_validate::real_weight -- --ignored --nocapture
//! ```
//!
//! **Base mode (sc-8680).** A second `#[ignore]`d test, `real_weight_control_base`, drives the
//! **undistilled base** control path (shift-6.0, ~50-step, real CFG) — the candle sibling of the MLX base
//! control variant. Point `ZIMG_CTRL_BASE` at a `Tongyi-MAI/Z-Image` (non-Turbo) snapshot and
//! `ZIMG_CTRL_NET` at the base `Z-Image-Fun-Controlnet-Union-2.1` checkpoint (a **dir** exercises the
//! deterministic overlay resolution — it must pick the Union file, not a Tile-lite sibling), then run
//! `control_validate::real_weight_control_base -- --ignored --nocapture`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};

use crate::control::{ZImageControl, ZImageControlPaths, ZImageControlRequest};

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
#[ignore = "real-weight GPU validation; set ZIMG_CTRL_BASE/ZIMG_CTRL_NET/ZIMG_CTRL_POSE/ZIMG_CTRL_OUT"]
fn real_weight_control() {
    run_control_validation(false, "turbo", 8, None, None);
}

/// Base-mode (sc-8680) real-weight validation: point `ZIMG_CTRL_BASE` at a `Tongyi-MAI/Z-Image`
/// (non-Turbo) snapshot and `ZIMG_CTRL_NET` at the base `Z-Image-Fun-Controlnet-Union-2.1` checkpoint
/// (a **dir** exercises the Union-vs-Tile-lite overlay resolution). Runs the undistilled shift-6.0
/// ~50-step CFG path (guidance 4.0 + a negative prompt) — the ablation gate + cancel contract are
/// identical.
#[test]
#[ignore = "real-weight GPU validation (base mode); set ZIMG_CTRL_BASE/ZIMG_CTRL_NET/ZIMG_CTRL_POSE/ZIMG_CTRL_OUT"]
fn real_weight_control_base() {
    run_control_validation(
        true,
        "base",
        50,
        Some(4.0),
        Some("blurry, low quality, deformed"),
    );
}

/// The shared control-validation harness (sc-8680): loads the model in the requested mode (`base`),
/// runs a with-control vs no-control (scale 0) ablation, checks the pre-/mid-denoise cancel contract,
/// and asserts the control path meaningfully changes the output. `steps`/`guidance`/`negative` tune the
/// per-mode request (Turbo ignores guidance/negative; base uses them).
fn run_control_validation(
    base: bool,
    tag: &str,
    steps: usize,
    guidance: Option<f32>,
    negative: Option<&str>,
) {
    let out_dir = env_path("ZIMG_CTRL_OUT");
    std::fs::create_dir_all(&out_dir).ok();

    let paths = ZImageControlPaths {
        snapshot: env_path("ZIMG_CTRL_BASE"),
        control: env_path("ZIMG_CTRL_NET"),
        base,
    };
    let skeleton = read_ppm(&env_path("ZIMG_CTRL_POSE"));
    println!(
        "[{tag}] pose skeleton {}x{}; loading ZImageControl (base={base}) …",
        skeleton.width, skeleton.height
    );

    let t0 = std::time::Instant::now();
    let model = ZImageControl::load(&paths).expect("load ZImageControl");
    println!("[{tag}] loaded in {:?}", t0.elapsed());

    let req = ZImageControlRequest {
        prompt: "a person standing, full body, photorealistic, studio lighting, sharp focus".into(),
        width: skeleton.width,
        height: skeleton.height,
        steps,
        control_scale: 1.0,
        guidance,
        negative_prompt: negative.map(str::to_string),
        seed: 12345,
        cancel: CancelFlag::new(),
    };

    let mut noop = |_p: Progress| {};

    // With control.
    let t = std::time::Instant::now();
    let out_ctrl = model
        .generate(&req, &skeleton, &mut noop)
        .expect("generate (control)");
    println!("[{tag}][control] {:?}", t.elapsed());
    write_ppm(
        &out_dir.join(format!("zimage_control_{tag}.ppm")),
        &out_ctrl,
    );

    // Without control (scale 0 → the VACE hints contribute zero → plain Z-Image at the same seed/prompt).
    let plain_req = ZImageControlRequest {
        control_scale: 0.0,
        ..req.clone()
    };
    let t = std::time::Instant::now();
    let out_plain = model
        .generate(&plain_req, &skeleton, &mut noop)
        .expect("generate (no control)");
    println!("[{tag}][no-control] {:?}", t.elapsed());
    write_ppm(
        &out_dir.join(format!("zimage_no_control_{tag}.ppm")),
        &out_plain,
    );

    let diff = mean_abs_diff(&out_ctrl, &out_plain);
    println!("=== Z-Image Fun-ControlNet validation ({tag}) ===");
    println!("  mean abs pixel diff (control vs no-control): {diff:.2}");
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel.
    let cancelled = ZImageControlRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..req.clone()
    };
    let pre = model.generate(&cancelled, &skeleton, &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[{tag}][cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel on step 3.
    let mid = CancelFlag::new();
    let mid_req = ZImageControlRequest {
        cancel: mid.clone(),
        ..req.clone()
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
    let res = model.generate(&mid_req, &skeleton, &mut cancel_at_3);
    assert!(
        matches!(res, Err(candle_gen::CandleError::Canceled)),
        "mid-cancel must return Canceled"
    );
    let steps_seen = seen.load(Ordering::SeqCst);
    assert!(
        (3..=4).contains(&steps_seen),
        "mid-cancel should stop right after step 3 (saw {steps_seen})"
    );
    println!("[{tag}][cancel:mid] Err(Canceled) after {steps_seen} steps ✓");

    // The gate: the control path meaningfully changes the output (it actually conditions the image).
    assert!(
        diff > 5.0,
        "control vs no-control mean diff {diff:.2} too small — control may not be wired"
    );
    println!(
        "[{tag}] Z-Image Fun-ControlNet validation PASS ✅ (eyeball the PPMs for pose adherence)"
    );
}
