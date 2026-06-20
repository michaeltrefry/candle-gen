//! Z-Image **img2img / edit** real-weight GPU validation (sc-6595, epic 5480) — an env-driven,
//! `#[ignore]`d integration test that drives the REAL [`ZImageEdit`] stack on the deployed hardware (a
//! `Tongyi-MAI/Z-Image-Turbo` snapshot + a source image). The Z-Image sibling of the SDXL edit harness.
//!
//! **Gate.** img2img should (a) actually edit toward the prompt, and (b) honor the Z-Image
//! structure-preservation strength convention — **higher strength ⇒ closer to the source** (the fork's
//! `init_time_step`; the inverse of the SDXL knob). So:
//!  - **low** strength (heavy regeneration) departs from the source's VAE round-trip the MOST (> 4),
//!  - two distinct **prompts** at the same seed/strength diverge (> 4) — the edit follows the text,
//!  - the diff-vs-source is monotone in strength the INVERSE of SDXL (s=0.25 > s=0.6 > s=0.9),
//!  - the default-strength edit is a real change (> 1) — a deliberately light touch, hence modest.
//!
//! Plus the cancel contract. The "is it a good edit" judgement is the eyeball check on the written PPMs.
//!
//! Run (after deploying weights + a source into local dirs):
//! ```text
//! set ZIMG_EDIT_BASE=...\Z-Image-Turbo          # tokenizer/ text_encoder/ transformer/ vae/
//! set ZIMG_EDIT_SRC=...\source.ppm              # a source image (P6 binary PPM)
//! set ZIMG_EDIT_OUT=...\out
//! cargo test -p candle-gen-z-image --features cuda --release edit_validate::real_weight -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};

use crate::edit::{ZImageEdit, ZImageEditPaths, ZImageEditRequest};

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
#[ignore = "real-weight GPU validation; set ZIMG_EDIT_BASE/ZIMG_EDIT_SRC/ZIMG_EDIT_OUT"]
fn real_weight_edit() {
    let out_dir = env_path("ZIMG_EDIT_OUT");
    std::fs::create_dir_all(&out_dir).ok();

    let paths = ZImageEditPaths {
        base: env_path("ZIMG_EDIT_BASE"),
    };
    let source = read_ppm(&env_path("ZIMG_EDIT_SRC"));
    println!(
        "source {}x{}; loading ZImageEdit …",
        source.width, source.height
    );

    let t0 = std::time::Instant::now();
    let model = ZImageEdit::load(&paths).expect("load ZImageEdit");
    println!("loaded in {:?}", t0.elapsed());

    // One seed/prompt; the source is fit to a clean multiple-of-16 render size.
    let width = source.width - (source.width % 16);
    let height = source.height - (source.height % 16);
    let base = ZImageEditRequest {
        prompt: "a watercolor painting, soft pastel colors, dreamy, artistic".into(),
        width,
        height,
        steps: 8,
        strength: 0.6,
        seed: 12345,
        cancel: CancelFlag::new(),
    };

    // A second, stylistically distinct prompt at the same seed/strength — proves the edit follows the
    // *prompt*, not just the injected noise (the strongest "it actually edits toward the text" signal).
    let prompt_b =
        "an oil painting, dark dramatic chiaroscuro lighting, heavy impasto brushstrokes";

    let mut noop = |_p: Progress| {};
    let gen = |strength: f32, prompt: &str| -> Image {
        let req = ZImageEditRequest {
            strength,
            prompt: prompt.to_owned(),
            ..base.clone()
        };
        let mut noop = |_p: Progress| {};
        model
            .generate(&req, &source, &mut noop)
            .unwrap_or_else(|e| panic!("generate (s={strength} \"{prompt}\"): {e}"))
    };

    // Default-strength edit (the product default).
    let t = std::time::Instant::now();
    let out_edit = model
        .generate(&base, &source, &mut noop)
        .expect("generate edit");
    println!("[edit s=0.6] {:?}", t.elapsed());
    write_ppm(&out_dir.join("zimage_edit_s06.ppm"), &out_edit);

    // Low strength = heavy regeneration (farthest from source); high strength = light touch (closest).
    let out_regen = gen(0.25, &base.prompt);
    write_ppm(&out_dir.join("zimage_edit_s025.ppm"), &out_regen);
    let out_regen_b = gen(0.25, prompt_b);
    write_ppm(&out_dir.join("zimage_edit_s025_b.ppm"), &out_regen_b);
    let out_preserve = gen(0.9, &base.prompt);
    write_ppm(&out_dir.join("zimage_edit_s09.ppm"), &out_preserve);

    // strength 1.0 ⇒ init_time_step == steps ⇒ empty loop ⇒ the source's VAE round-trip at the render
    // size: the reference "no-edit" image to measure structure preservation against (the provider resizes
    // the source internally, so this is the apples-to-apples baseline).
    let src_resized = gen(1.0, &base.prompt);
    write_ppm(&out_dir.join("zimage_edit_roundtrip.ppm"), &src_resized);

    let d_edit = mean_abs_diff(&out_edit, &src_resized);
    let d_regen = mean_abs_diff(&out_regen, &src_resized);
    let d_preserve = mean_abs_diff(&out_preserve, &src_resized);
    let d_prompt = mean_abs_diff(&out_regen, &out_regen_b);
    println!("=== Z-Image img2img validation ===");
    println!("  diff vs source round-trip: s=0.25 {d_regen:.2}  s=0.6 {d_edit:.2}  s=0.9 {d_preserve:.2}");
    println!("  prompt A-vs-B diff @ s=0.25: {d_prompt:.2}");
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel.
    let cancelled = ZImageEditRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let pre = model.generate(&cancelled, &source, &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel on step 3 (strength 0.25 → start=2, 6 steps run, so step 3 exists).
    let mid = CancelFlag::new();
    let mid_req = ZImageEditRequest {
        strength: 0.25,
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
    let res = model.generate(&mid_req, &source, &mut cancel_at_3);
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

    // Gate 1: heavy regeneration (low strength) clearly departs from the source — img2img is wired.
    // (Z-Image's strength is structure-preservation, so the *low* strength is the heavy-edit case; the
    // default 0.6 is a deliberately light touch, hence a smaller diff — not a weak provider.)
    assert!(
        d_regen > 4.0,
        "low-strength regen diff {d_regen:.2} too small — img2img may not be wired"
    );
    // Gate 2: the edit follows the PROMPT — two distinct prompts at the same seed/strength diverge.
    assert!(
        d_prompt > 4.0,
        "prompt A-vs-B diff {d_prompt:.2} too small — the edit may ignore the prompt"
    );
    // Gate 3: the Z-Image structure-preservation convention — strength is monotone the INVERSE of SDXL,
    // so a lower strength diverges from the source MORE than a higher one.
    assert!(
        d_regen > d_edit && d_edit > d_preserve,
        "strength monotonicity broken (expected s=0.25 > s=0.6 > s=0.9): {d_regen:.2} / {d_edit:.2} / {d_preserve:.2}"
    );
    // Gate 4: the default-strength edit is a real (non-trivial) change from the source.
    assert!(
        d_edit > 1.0,
        "default-strength edit diff {d_edit:.2} too small"
    );
    println!("Z-Image img2img validation PASS ✅ (eyeball the PPMs for edit quality)");
}
