//! SDXL edit (img2img / inpaint / outpaint) real-weight GPU validation (sc-6037, epic 5480) — an
//! env-driven, `#[ignore]`d integration test that drives the REAL [`SdxlEdit`] stack on the deployed
//! hardware (RealVisXL/SDXL diffusers tree + a source image). The analog of the IP-Adapter Phase-5
//! harness, with a **dep-free** quantitative gate built from pixel differences:
//!
//!  - **strength=0 round-trip** (`recon`): an empty schedule decodes the VAE-encoded source, so `recon`
//!    is the source's VAE round-trip at the render size — the reference both gates compare against.
//!  - **img2img ablation**: a high-strength edit must diverge from `recon` more than a low-strength one
//!    (`mean|img_hi − recon| > mean|img_lo − recon|`) — strength actually controls how much is redrawn.
//!  - **inpaint mask**: with a left-half-white (repaint) / right-half-black (keep) mask, the kept region
//!    must stay close to `recon` while the repaint region diverges (`repaint_diff ≫ kept_diff`) — the
//!    per-step blend pins the kept region to the source and frees the masked region.
//!  - the cancel contract (pre + mid-denoise).
//!
//! Run (after deploying weights into the HF cache / a local dir):
//! ```text
//! set EDIT_SDXL_BASE=...\RealVisXL_V5.0   # diffusers tree (unet/, text_encoder{,_2}/, vae omitted — f16-fix VAE via hf-hub)
//! set EDIT_SRC=...\src.ppm                # a source image (P6 PPM)
//! set EDIT_OUT=...\out                    # output dir
//! cargo test -p candle-gen-sdxl --features cuda --release edit_validate::real_weight -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};

use crate::edit_provider::{SdxlEdit, SdxlEditPaths, SdxlEditRequest};

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
}

/// Minimal P6 PPM reader (binary `P6\n<w> <h>\n<max>\n<rgb bytes>`), tolerant of a single comment line
/// and arbitrary header whitespace — enough for hand-prepared source images.
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
    i += 1; // single whitespace after maxval, before the pixel block
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

/// A left-half-white (repaint) / right-half-black (keep) RGB8 mask at `w × h` — the synthetic inpaint
/// mask the region gate keys off (so the kept/repaint split is exactly the image halves).
fn split_mask(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for _y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 { 255u8 } else { 0u8 };
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Mean absolute per-channel pixel difference between two equal-size RGB8 images over columns `[x0, x1)`.
fn region_diff(a: &Image, b: &Image, x0: usize, x1: usize) -> f32 {
    assert_eq!((a.width, a.height), (b.width, b.height), "size mismatch");
    let (w, h) = (a.width as usize, a.height as usize);
    let mut sum = 0f64;
    let mut n = 0u64;
    for y in 0..h {
        for x in x0..x1 {
            for c in 0..3 {
                let idx = (y * w + x) * 3 + c;
                sum += (a.pixels[idx] as f64 - b.pixels[idx] as f64).abs();
                n += 1;
            }
        }
    }
    (sum / n as f64) as f32
}

/// Drive the real SDXL edit stack: the strength-0 round-trip, the img2img strength ablation, the
/// inpaint kept-vs-repaint region gate, and the cancel contract. Visually-inspectable PPMs land in
/// `EDIT_OUT`.
#[test]
#[ignore = "real-weight GPU validation; set EDIT_SDXL_BASE/EDIT_SRC/EDIT_OUT"]
fn real_weight_edit() {
    let out_dir = env_path("EDIT_OUT");
    std::fs::create_dir_all(&out_dir).ok();

    let paths = SdxlEditPaths {
        sdxl_base: env_path("EDIT_SDXL_BASE"),
    };
    let source = read_ppm(&env_path("EDIT_SRC"));
    println!(
        "source {}x{}; loading SdxlEdit …",
        source.width, source.height
    );

    let t0 = std::time::Instant::now();
    let model = SdxlEdit::load(&paths).expect("load SdxlEdit");
    println!("loaded in {:?}", t0.elapsed());

    let (w, h) = (1024u32, 1024u32);
    let prompt = "a vibrant oil painting, rich brush strokes, warm palette, masterpiece";
    let negative = "blurry, lowres, deformed, watermark, text";
    let base = SdxlEditRequest {
        prompt: prompt.into(),
        negative: negative.into(),
        width: w,
        height: h,
        steps: 30,
        guidance: 5.0,
        strength: 0.8,
        seed: 12345,
        cancel: CancelFlag::new(),
    };
    let mut noop = |_p: Progress| {};

    // strength = 0 ⇒ empty schedule ⇒ the VAE round-trip of the source at the render size.
    let recon = model
        .generate(
            &SdxlEditRequest {
                strength: 0.0,
                ..base.clone()
            },
            &source,
            &mut noop,
        )
        .expect("generate (strength 0 round-trip)");
    write_ppm(&out_dir.join("recon.ppm"), &recon);

    // img2img: high vs low strength. More strength ⇒ more divergence from the source round-trip.
    let t = std::time::Instant::now();
    let img_hi = model
        .generate(
            &SdxlEditRequest {
                strength: 0.8,
                ..base.clone()
            },
            &source,
            &mut noop,
        )
        .expect("generate (img2img hi)");
    println!("[img2img s=0.8] {:?}", t.elapsed());
    write_ppm(&out_dir.join("img2img_hi.ppm"), &img_hi);

    let img_lo = model
        .generate(
            &SdxlEditRequest {
                strength: 0.2,
                ..base.clone()
            },
            &source,
            &mut noop,
        )
        .expect("generate (img2img lo)");
    write_ppm(&out_dir.join("img2img_lo.ppm"), &img_lo);

    let diff_hi = region_diff(&img_hi, &recon, 0, w as usize);
    let diff_lo = region_diff(&img_lo, &recon, 0, w as usize);

    // Inpaint: left half white (repaint), right half black (keep). The kept region stays near recon;
    // the repaint region diverges.
    let mask = split_mask(w, h);
    let t = std::time::Instant::now();
    let inpaint = model
        .generate_masked(
            &SdxlEditRequest {
                strength: 0.85,
                ..base.clone()
            },
            &source,
            &mask,
            &mut noop,
        )
        .expect("generate_masked (inpaint)");
    println!("[inpaint s=0.85] {:?}", t.elapsed());
    write_ppm(&out_dir.join("inpaint.ppm"), &inpaint);

    let half = (w / 2) as usize;
    let repaint_diff = region_diff(&inpaint, &recon, 0, half); // left half = repaint
    let kept_diff = region_diff(&inpaint, &recon, half, w as usize); // right half = keep

    println!("=== SDXL edit validation ===");
    println!("  img2img mean|·−recon|  hi(0.8)={diff_hi:.2}  lo(0.2)={diff_lo:.2}");
    println!(
        "  inpaint mean|·−recon|  repaint(left)={repaint_diff:.2}  kept(right)={kept_diff:.2}"
    );
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel.
    let pre = model.generate(
        &SdxlEditRequest {
            cancel: {
                let c = CancelFlag::new();
                c.cancel();
                c
            },
            ..base.clone()
        },
        &source,
        &mut noop,
    );
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel: flip the flag from the progress callback on the 3rd step.
    let mid = CancelFlag::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let seen_cb = seen.clone();
    let mid_flag = mid.clone();
    let mut cancel_at_3 = move |p: Progress| {
        if let Progress::Step { current, .. } = p {
            seen_cb.store(current as usize, Ordering::SeqCst);
            if current >= 3 {
                mid_flag.cancel();
            }
        }
    };
    let res = model.generate(
        &SdxlEditRequest {
            cancel: mid,
            ..base.clone()
        },
        &source,
        &mut cancel_at_3,
    );
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

    // The gates.
    assert!(
        diff_hi > diff_lo,
        "img2img strength 0.8 ({diff_hi:.2}) must diverge from the source more than 0.2 ({diff_lo:.2})"
    );
    assert!(
        repaint_diff > kept_diff * 2.0,
        "inpaint repaint diff ({repaint_diff:.2}) must dominate the kept-region diff ({kept_diff:.2})"
    );
    assert!(
        kept_diff < 12.0,
        "inpaint kept region ({kept_diff:.2}) drifted from the source — the blend should pin it"
    );
    assert!(inpaint.width == w && inpaint.height == h);
    println!("SDXL edit validation PASS ✅");
}
