//! Qwen-Image-Edit full-provider real-weight GPU validation (sc-5487, epic 5480) — an env-driven,
//! `#[ignore]`d integration test that drives the REAL [`QwenEdit`] stack (VL encoder + MMDiT + VAE)
//! on the deployed hardware from a `Qwen/Qwen-Image-Edit` snapshot + a reference image.
//!
//! **Gate.** A reference edit should (1) be coherent (finite, non-degenerate), and (2) follow the
//! edit prompt — so the metric is a **prompt-sensitivity ablation**: edit the same reference at the
//! same seed with two different prompts and assert the outputs differ meaningfully (the VL/dual-latent
//! conditioning actually steers the result). Plus the cancel contract. The "does it look like the
//! reference edited per the prompt" judgement is the eyeball check on the written PPMs.
//!
//! Run (after deploying a Qwen-Image-Edit snapshot + a reference PPM; keep output ≤ ~1536²):
//! ```text
//! set QWEN_EDIT_BASE=...\Qwen-Image-Edit   # diffusers snapshot (text_encoder/ transformer/ vae/ tokenizer/)
//! set QWEN_EDIT_REF=...\reference.ppm       # an RGB P6 PPM
//! set QWEN_EDIT_OUT=...\out
//! cargo test -p candle-gen-qwen-image --features cuda --release edit_validate::real_weight -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{AdapterKind, AdapterSpec, Image, Progress};

use crate::edit::{QwenEdit, QwenEditPaths, QwenEditRequest};

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

/// Per-channel std of an image (a degeneracy/coherence sanity check).
fn pixel_std(img: &Image) -> f32 {
    let n = img.pixels.len() as f32;
    let mean = img.pixels.iter().map(|&p| p as f32).sum::<f32>() / n;
    (img.pixels
        .iter()
        .map(|&p| (p as f32 - mean).powi(2))
        .sum::<f32>()
        / n)
        .sqrt()
}

#[test]
#[ignore = "real-weight GPU validation; set QWEN_EDIT_BASE/QWEN_EDIT_REF/QWEN_EDIT_OUT"]
fn real_weight_edit() {
    let out_dir = env_path("QWEN_EDIT_OUT");
    std::fs::create_dir_all(&out_dir).ok();
    let reference = read_ppm(&env_path("QWEN_EDIT_REF"));
    println!(
        "reference {}x{}; loading QwenEdit …",
        reference.width, reference.height
    );

    let t0 = std::time::Instant::now();
    let model = QwenEdit::load(&QwenEditPaths {
        root: env_path("QWEN_EDIT_BASE"),
        adapters: vec![],
    })
    .expect("load QwenEdit");
    println!(
        "loaded in {:?} (zero_cond_t handled internally)",
        t0.elapsed()
    );

    let base = QwenEditRequest {
        prompt: "turn it into a snowy winter landscape, heavy snowfall, cold blue tones".into(),
        negative: "blurry, lowres, artifacts, watermark".into(),
        width: 1024,
        height: 1024,
        steps: 20,
        guidance: 4.0,
        seed: 12345,
        lightning: false,
        cancel: CancelFlag::new(),
    };

    let mut noop = |_p: Progress| {};

    // Edit A.
    let t = std::time::Instant::now();
    let out_a = model
        .generate(&base, std::slice::from_ref(&reference), &mut noop)
        .expect("generate (prompt A)");
    println!("[edit A] {:?}", t.elapsed());
    write_ppm(&out_dir.join("qwen_edit_a.ppm"), &out_a);

    // Edit B — same reference + seed, a different prompt (prompt-sensitivity ablation).
    let req_b = QwenEditRequest {
        prompt: "turn it into a sunny tropical beach at noon, palm trees, warm golden light".into(),
        ..base.clone()
    };
    let t = std::time::Instant::now();
    let out_b = model
        .generate(&req_b, std::slice::from_ref(&reference), &mut noop)
        .expect("generate (prompt B)");
    println!("[edit B] {:?}", t.elapsed());
    write_ppm(&out_dir.join("qwen_edit_b.ppm"), &out_b);

    let diff = mean_abs_diff(&out_a, &out_b);
    let (sa, sb) = (pixel_std(&out_a), pixel_std(&out_b));
    println!("=== Qwen-Image-Edit validation ===");
    println!("  edit A std {sa:.2}, edit B std {sb:.2}");
    println!("  mean abs pixel diff (prompt A vs prompt B): {diff:.2}");
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel.
    let cancelled = QwenEditRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let pre = model.generate(&cancelled, std::slice::from_ref(&reference), &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel on step 3.
    let mid = CancelFlag::new();
    let mid_req = QwenEditRequest {
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
    let res = model.generate(&mid_req, std::slice::from_ref(&reference), &mut cancel_at_3);
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

    // Coherence + prompt-sensitivity gates.
    assert_eq!(out_a.width, 1024);
    assert!(
        sa > 5.0 && sb > 5.0,
        "edits are degenerate (flat) — std {sa:.2}/{sb:.2}"
    );
    assert!(
        diff > 5.0,
        "prompt A vs B mean diff {diff:.2} too small — VL/dual-latent conditioning may not steer"
    );
    println!(
        "Qwen-Image-Edit validation PASS ✅ (eyeball the PPMs for reference-respecting edits)"
    );
}

/// sc-6217: a >1024² edit makes the joint `[txt, noise, ref]` sequence (24 heads) blow past candle's
/// i32 attention-index limit in a single pass (at 1536² the scores tensor is ~8.2B > i32::MAX ~2.147B),
/// which silently corrupts the trailing query rows → noise in the lower part of the image. The
/// query-row chunking in `JointAttention` must keep the output coherent. This run would produce a
/// corrupted/noisy tail WITHOUT the fix; with it the whole frame stays a coherent reference-respecting
/// edit. Eyeball `qwen_edit_highres_1536.ppm` to confirm (the automated gates only catch NaN / flat).
#[test]
#[ignore = "real-weight GPU validation; set QWEN_EDIT_BASE/QWEN_EDIT_REF/QWEN_EDIT_OUT"]
fn high_res_edit_avoids_i32_overflow() {
    let out_dir = env_path("QWEN_EDIT_OUT");
    std::fs::create_dir_all(&out_dir).ok();
    let reference = read_ppm(&env_path("QWEN_EDIT_REF"));
    let model = QwenEdit::load(&QwenEditPaths {
        root: env_path("QWEN_EDIT_BASE"),
        adapters: vec![],
    })
    .expect("load QwenEdit");

    let req = QwenEditRequest {
        prompt: "turn it into a snowy winter landscape, heavy snowfall, cold blue tones".into(),
        negative: "blurry, lowres, artifacts, watermark".into(),
        width: 1536,
        height: 1536,
        steps: 20,
        guidance: 4.0,
        seed: 12345,
        lightning: false,
        cancel: CancelFlag::new(),
    };
    let mut noop = |_p: Progress| {};

    let t = std::time::Instant::now();
    let out = model
        .generate(&req, std::slice::from_ref(&reference), &mut noop)
        .expect("generate 1536² edit");
    let std = pixel_std(&out);
    println!("[highres 1536²] {:?}, std {std:.2}", t.elapsed());
    write_ppm(&out_dir.join("qwen_edit_highres_1536.ppm"), &out);

    assert_eq!((out.width, out.height), (1536, 1536), "wrong output size");
    assert!(
        std.is_finite() && std > 5.0,
        "1536² edit is NaN/degenerate (std {std:.2}) — i32 attention overflow not contained"
    );
    println!(
        "sc-6217 high-res edit PASS ✅ (eyeball qwen_edit_highres_1536.ppm — no noisy bottom band)"
    );
}

/// sc-6220: the **Qwen-Image-Edit-2511-Lightning** few-step distill — load the `-2511` base with the
/// lightx2v 4-step LoRA folded into the MMDiT ([`QwenEditPaths::adapters`]), then run the CFG-off
/// lightning schedule at 4 steps ([`QwenEditRequest::lightning`]). Gates: the 4-step edit is coherent
/// (finite, non-flat) AND prompt-sensitive (A vs B differ), proving the merged distill produces a clean
/// image in 4 steps rather than the 20+ the production schedule needs — i.e. both the adapter merge and
/// the lightning sampler are wired correctly. Run with `QWEN_EDIT_BASE` = a `-2511` snapshot and
/// `QWEN_EDIT_LIGHTNING_LORA` = the 4-step bf16 distill `.safetensors`.
#[test]
#[ignore = "real-weight GPU validation; set QWEN_EDIT_BASE/QWEN_EDIT_REF/QWEN_EDIT_OUT/QWEN_EDIT_LIGHTNING_LORA"]
fn lightning_edit_4steps() {
    let out_dir = env_path("QWEN_EDIT_OUT");
    std::fs::create_dir_all(&out_dir).ok();
    let reference = read_ppm(&env_path("QWEN_EDIT_REF"));
    let lora = env_path("QWEN_EDIT_LIGHTNING_LORA");
    println!("loading QwenEdit + lightning distill {} …", lora.display());

    let t0 = std::time::Instant::now();
    let model = QwenEdit::load(&QwenEditPaths {
        root: env_path("QWEN_EDIT_BASE"),
        adapters: vec![AdapterSpec::new(lora, 1.0, AdapterKind::Lora)],
    })
    .expect("load QwenEdit + lightning LoRA");
    println!(
        "loaded in {:?} (distill merged into the MMDiT)",
        t0.elapsed()
    );

    let base = QwenEditRequest {
        prompt: "turn it into a snowy winter landscape, heavy snowfall, cold blue tones".into(),
        negative: String::new(), // CFG-off → unused
        width: 1024,
        height: 1024,
        steps: 4,
        guidance: 1.0,
        seed: 12345,
        lightning: true,
        cancel: CancelFlag::new(),
    };
    let mut noop = |_p: Progress| {};

    // Edit A (4 steps).
    let t = std::time::Instant::now();
    let out_a = model
        .generate(&base, std::slice::from_ref(&reference), &mut noop)
        .expect("lightning edit A");
    println!("[lightning A · 4 steps] {:?}", t.elapsed());
    write_ppm(&out_dir.join("qwen_edit_lightning_a.ppm"), &out_a);

    // Edit B — same reference + seed, a different prompt (prompt-sensitivity ablation).
    let req_b = QwenEditRequest {
        prompt: "turn it into a sunny tropical beach at noon, palm trees, warm golden light".into(),
        ..base.clone()
    };
    let out_b = model
        .generate(&req_b, std::slice::from_ref(&reference), &mut noop)
        .expect("lightning edit B");
    write_ppm(&out_dir.join("qwen_edit_lightning_b.ppm"), &out_b);

    let diff = mean_abs_diff(&out_a, &out_b);
    let (sa, sb) = (pixel_std(&out_a), pixel_std(&out_b));
    println!("=== Qwen-Image-Edit-2511 Lightning (4-step) validation ===");
    println!("  edit A std {sa:.2}, edit B std {sb:.2}; prompt A-vs-B diff {diff:.2}");
    println!("  outputs: {}", out_dir.display());

    assert_eq!(out_a.width, 1024);
    assert!(
        sa > 5.0 && sb > 5.0,
        "lightning edits are degenerate (flat) — std {sa:.2}/{sb:.2}"
    );
    assert!(
        diff > 5.0,
        "lightning prompt A-vs-B diff {diff:.2} too small — distill/sampler not steering"
    );
    println!("sc-6220 lightning 4-step edit PASS ✅ (eyeball the PPMs)");
}
