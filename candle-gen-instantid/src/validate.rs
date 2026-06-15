//! Phase 5 real-weight GPU validation (sc-5491, epic 5480) — `#[ignore]`d by default.
//!
//! The unit tests in `model.rs` only prove shape/determinism with tiny random tensors. This drives the
//! REAL InstantID stack (RealVisXL_V5.0 + the InstantX IdentityNet + the `SceneWorks/instantid-mlx`
//! bundle + xinsir OpenPose) on the GPU and asserts **identity recovery** — the ArcFace cosine between
//! the reference face and the generated face — across the three production modes (Identity / AngleSet /
//! PoseSet) plus pre- and mid-denoise cancellation. It re-embeds each output through the model's own
//! face stack, so a broken inference path (≈0 cosine) is caught quantitatively, not just visually.
//!
//! Env-driven so no real weights live in the repo. Run (PowerShell, MSVC 14.44 vcvars +
//! `CUDA_COMPUTE_CAP=120`):
//! ```text
//! $env:IID_SDXL_BASE   = "<RealVisXL snapshot dir>"
//! $env:IID_IDENTITYNET = "<InstantX snapshot>/ControlNetModel"
//! $env:IID_IP_ADAPTER  = "<instantid-mlx snapshot>/ip-adapter.safetensors"
//! $env:IID_FACE_DIR    = "<instantid-mlx snapshot>"           # scrfd_10g + arcface_iresnet100
//! $env:IID_OPENPOSE    = "<xinsir snapshot dir>"              # optional → enables PoseSet
//! $env:IID_REF         = "<reference face .ppm (P6)>"
//! $env:IID_OUT         = "<output dir for the .ppm renders>"
//! cargo test -p candle-gen-instantid --features cuda --release validate::real_weight -- --ignored --nocapture
//! ```

#![cfg(test)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress, WeightsSource};
use candle_gen::CandleError;

use crate::model::{InstantId, InstantIdPaths, InstantIdRequest};
use crate::openpose::BodyPoint;

/// A required env path (panics with a clear message if unset — these tests are opt-in).
fn env_path(key: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(key).unwrap_or_else(|_| panic!("set ${key} (see validate.rs module docs)")),
    )
}

/// An optional env path.
fn env_opt(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

/// Minimal P6 (binary) PPM reader — the harness owns image IO since the `image` dep here is built
/// without codecs. Parses `P6 <w> <h> <maxval>` (whitespace-separated) then `w*h*3` raw RGB bytes.
fn read_ppm(path: &Path) -> Image {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..2], b"P6", "{} is not a P6 PPM", path.display());
    // Tokenize the three ASCII header numbers starting after the magic.
    let mut nums = [0usize; 3];
    let mut n = 0;
    let mut i = 2usize;
    while n < 3 {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        nums[n] = std::str::from_utf8(&bytes[start..i])
            .unwrap()
            .parse()
            .unwrap();
        n += 1;
    }
    let (w, h, maxval) = (nums[0], nums[1], nums[2]);
    assert_eq!(maxval, 255, "only 8-bit PPM supported");
    let pixels = bytes[i + 1..].to_vec(); // single whitespace byte after maxval, then raw RGB
    assert_eq!(pixels.len(), w * h * 3, "PPM body size mismatch");
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// Write a P6 PPM (converted to PNG out-of-band for viewing).
fn write_ppm(path: &Path, img: &Image) {
    let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    out.extend_from_slice(&img.pixels);
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Cosine similarity of two equal-length embeddings (ArcFace embeddings are not pre-normalized).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Re-embed the largest face in `img` and report its cosine to `ref_emb`. Returns `-1.0` (with a
/// warning) when no face is detected, so framing-dependent modes don't hard-panic.
fn output_cosine(model: &InstantId, ref_emb: &[f32], img: &Image, tag: &str) -> f32 {
    match model.largest_face(img) {
        Ok(face) => {
            let c = cosine(ref_emb, &face.embedding);
            eprintln!(
                "[{tag}] face detected (det={:.3}) cosine={c:.4}",
                face.det_score
            );
            c
        }
        Err(e) => {
            eprintln!("[{tag}] WARNING: no face detected in output ({e}) — cosine n/a");
            -1.0
        }
    }
}

/// A simple front-facing standing skeleton (COCO-18, normalized `[0,1]`) with the head visible, so
/// `face_box_from_keypoints` yields a head box → IdentityNet + the face IP tokens engage in pose mode.
fn front_pose() -> Vec<BodyPoint> {
    let p = |x: f64, y: f64| Some((x, y));
    vec![
        p(0.50, 0.18), // 0 nose
        p(0.50, 0.28), // 1 neck
        p(0.42, 0.30), // 2 r_sho
        p(0.38, 0.45), // 3 r_elb
        p(0.36, 0.58), // 4 r_wri
        p(0.58, 0.30), // 5 l_sho
        p(0.62, 0.45), // 6 l_elb
        p(0.64, 0.58), // 7 l_wri
        p(0.45, 0.55), // 8 r_hip
        p(0.44, 0.72), // 9 r_kne
        p(0.44, 0.90), // 10 r_ank
        p(0.55, 0.55), // 11 l_hip
        p(0.56, 0.72), // 12 l_kne
        p(0.56, 0.90), // 13 l_ank
        p(0.47, 0.16), // 14 r_eye
        p(0.53, 0.16), // 15 l_eye
        p(0.44, 0.17), // 16 r_ear
        p(0.56, 0.17), // 17 l_ear
    ]
}

/// A coarse step-progress printer (every 5th step) so a slow GPU run shows life under `--nocapture`.
fn make_progress() -> impl FnMut(Progress) {
    move |p: Progress| {
        if let Progress::Step { current, total } = p {
            if current == 1 || current % 5 == 0 || current == total {
                eprintln!("    step {current}/{total}");
            }
        }
    }
}

#[test]
#[ignore = "real-weight GPU validation; set IID_* env + run with --features cuda --release"]
fn real_weight_instantid() {
    let out_dir = env_path("IID_OUT");
    std::fs::create_dir_all(&out_dir).unwrap();

    let paths = InstantIdPaths {
        sdxl_base: env_path("IID_SDXL_BASE"),
        identitynet: WeightsSource::Dir(env_path("IID_IDENTITYNET")),
        ip_adapter: env_path("IID_IP_ADAPTER"),
    };

    eprintln!("loading InstantId (RealVisXL + IdentityNet + IP-Adapter + VAE) ...");
    let t0 = Instant::now();
    let mut model = InstantId::load(&paths).expect("InstantId::load");
    model = model
        .with_face(&env_path("IID_FACE_DIR"))
        .expect("with_face (SCRFD + ArcFace)");
    let pose_enabled = env_opt("IID_OPENPOSE").is_some();
    if let Some(op) = env_opt("IID_OPENPOSE") {
        model = model
            .with_openpose(&WeightsSource::Dir(op))
            .expect("with_openpose");
    }
    eprintln!("loaded in {:?} (pose mode: {pose_enabled})", t0.elapsed());

    let reference = read_ppm(&env_path("IID_REF"));
    let ref_emb = model
        .largest_face(&reference)
        .expect("detect a face in the reference image")
        .embedding;
    eprintln!(
        "reference {}x{} — ArcFace embedding {}-d",
        reference.width,
        reference.height,
        ref_emb.len()
    );

    let steps: usize = std::env::var("IID_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    // The worker's `instantid_realvisxl` production defaults (RealVisXL is tuned for a low CFG).
    let base = InstantIdRequest {
        prompt: "a professional studio portrait photograph of a person, looking at the camera, \
                 natural soft lighting, sharp focus, high detail"
            .to_owned(),
        negative: "lowres, blurry, out of focus, deformed, disfigured, extra limbs, bad anatomy, \
                   watermark, text, cartoon, 3d render"
            .to_owned(),
        width: 1024,
        height: 1024,
        steps,
        guidance: 3.0,
        ip_adapter_scale: 0.8,
        controlnet_scale: 0.8,
        openpose_scale: 0.7,
        seed: 12345,
        cancel: CancelFlag::new(),
    };

    // --- 1) Identity (engine `generate`, W×H letterboxed) ---
    let t = Instant::now();
    let img = model
        .generate(&base, &reference, &mut make_progress())
        .expect("identity generate");
    eprintln!("[identity] {:?}", t.elapsed());
    write_ppm(&out_dir.join("identity.ppm"), &img);
    let id_cos = output_cosine(&model, &ref_emb, &img, "identity");

    // --- 2) AngleSet (engine `generate_angle`, front view, square) ---
    let t = Instant::now();
    let img = model
        .generate_angle(&base, &reference, "front", &mut make_progress())
        .expect("angle generate");
    eprintln!("[angle:front] {:?}", t.elapsed());
    write_ppm(&out_dir.join("angle_front.ppm"), &img);
    let ang_cos = output_cosine(&model, &ref_emb, &img, "angle:front");

    // --- 3) PoseSet (engine `generate_pose`, MultiControlNet IdentityNet + OpenPose, square) ---
    let mut pose_cos = None;
    if pose_enabled {
        let t = Instant::now();
        let img = model
            .generate_pose(&base, &reference, &front_pose(), &mut make_progress())
            .expect("pose generate");
        eprintln!("[pose:front] {:?}", t.elapsed());
        write_ppm(&out_dir.join("pose_front.ppm"), &img);
        pose_cos = Some(output_cosine(&model, &ref_emb, &img, "pose:front"));
    } else {
        eprintln!("[pose] skipped (IID_OPENPOSE unset)");
    }

    // --- 4) Cancel — pre-denoise (flag set before the call) ---
    let pre = InstantIdRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let r = model.generate(&pre, &reference, &mut make_progress());
    assert!(
        matches!(r, Err(CandleError::Canceled)),
        "pre-cancel must return Err(Canceled), got {r:?}"
    );
    eprintln!("[cancel:pre] Err(Canceled) ✓");

    // --- 5) Cancel — mid-denoise (flip the flag from the progress callback on the 3rd step) ---
    let cancel = CancelFlag::new();
    let seen = Arc::new(AtomicUsize::new(0));
    let mut prog = {
        let cancel = cancel.clone();
        let seen = seen.clone();
        move |p: Progress| {
            if let Progress::Step { .. } = p {
                if seen.fetch_add(1, Ordering::Relaxed) >= 2 {
                    cancel.cancel();
                }
            }
        }
    };
    let mid = InstantIdRequest {
        cancel: cancel.clone(),
        ..base.clone()
    };
    let r = model.generate(&mid, &reference, &mut prog);
    assert!(
        matches!(r, Err(CandleError::Canceled)),
        "mid-denoise cancel must return Err(Canceled), got {r:?}"
    );
    eprintln!(
        "[cancel:mid] Err(Canceled) after {} steps ✓",
        seen.load(Ordering::Relaxed)
    );

    // --- Identity-recovery gate ---
    // InstantID's validated envelope is ArcFace-cosine ≈0.82 @1024²; >0.45 is a conservative pass bar
    // that still catches a broken inference path (random/mismatched faces score ≈0).
    eprintln!(
        "\n=== Phase 5 summary ===\n  identity cosine: {id_cos:.4}\n  angle    cosine: {ang_cos:.4}\n  pose     cosine: {}\n  outputs: {}",
        pose_cos.map(|c| format!("{c:.4}")).unwrap_or_else(|| "n/a".into()),
        out_dir.display()
    );
    assert!(
        id_cos > 0.45,
        "identity cosine too low ({id_cos}) — inference likely broken"
    );
    assert!(
        ang_cos > 0.45,
        "angle cosine too low ({ang_cos}) — inference likely broken"
    );
    if let Some(c) = pose_cos {
        assert!(
            c > 0.30,
            "pose cosine too low ({c}) — face control likely broken"
        );
    }
    eprintln!("Phase 5 PASS ✅");
}
