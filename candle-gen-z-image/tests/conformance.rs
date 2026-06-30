//! gen-core contract conformance for the candle Z-Image provider (sc-4481, epic 3720 / sc-3693).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator.
//! The **seed-determinism** check is the regression guard for the deterministic CPU-seeded noise
//! (sc-3673 parity) the pipeline relies on.
//!
//! It drives a real `generate`, so it needs the CUDA backend + a local Z-Image-Turbo snapshot and is
//! `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set Z_IMAGE_SNAPSHOT=C:\Users\…\models--Tongyi-MAI--Z-Image-Turbo\snapshots\<hash>
//! cargo test -p candle-gen-z-image --features cuda --release --test conformance -- --ignored
//! ```
//!
//! Passing the suite is the parity evidence sc-3693 asks for: it locks **output dims** (request WxH
//! is the emitted image size), **seed semantics** (same request+seed ⇒ byte-identical output),
//! **distilled step count** (`Step.total` == resolved steps), **contract/sidecar field shape**
//! (validate-honesty + `GenerationOutput::Images`), and **cancellation/progress** (typed `Canceled`,
//! monotone `Step`). The accepted difference vs the macOS `mlx-gen-z-image` provider — same-backend
//! determinism only; **cross-backend pixel equality is NOT a goal** (RNG algorithms differ) — is the
//! same stance the SDXL slice documents.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use gen_core_testkit::{conformance, Profile};

#[test]
#[ignore = "needs Z_IMAGE_SNAPSHOT (a Z-Image-Turbo diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn z_image_conformance() {
    let snap = std::env::var("Z_IMAGE_SNAPSHOT")
        .expect("set Z_IMAGE_SNAPSHOT to a Tongyi-MAI/Z-Image-Turbo snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 256² (== the descriptor's min_size) keeps the suite's ~4 generate() calls cheap — it verifies
    // contract behavior, not image quality. The distilled default is 4 steps; the pipeline uses
    // req.steps verbatim, so `steps: 4` → Step.total == 4 (check_progress). Both dims are multiples
    // of the /16 alignment the provider requires.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 4,
        ..Profile::cheap()
    };

    // Resolve through THIS crate's `load` (its inventory registration is linked into the test binary,
    // so the suite's registry round-trip check also passes). Panics with aggregated failures.
    conformance(|| candle_gen_z_image::load(&spec).unwrap(), &profile);
}

/// Base (non-Turbo) `z_image` real-weight smoke (sc-8414): load a `Tongyi-MAI/Z-Image` snapshot and
/// render a small base-CFG t2i image, asserting a **coherent, non-degenerate** output (the candle
/// sibling of mlx sc-8320). Unlike the Turbo conformance above, this exercises the real-CFG +
/// shift-6.0 [`render_base`] path (guidance + a negative prompt). `#[ignore]`d — needs the base
/// weights + a CUDA GPU. Run on the Windows/Blackwell box (GPU 1 only, GPU 0 is busy):
///
/// ```text
/// set Z_IMAGE_BASE_SNAPSHOT=C:\Users\…\models--Tongyi-MAI--Z-Image\snapshots\<hash>
/// set CUDA_VISIBLE_DEVICES=1
/// cargo test -p candle-gen-z-image --features cuda --release --test conformance base_z_image_smoke -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs Z_IMAGE_BASE_SNAPSHOT (a Tongyi-MAI/Z-Image diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn base_z_image_smoke() {
    use candle_gen::gen_core::{GenerationOutput, GenerationRequest, Progress};

    let snap = std::env::var("Z_IMAGE_BASE_SNAPSHOT")
        .expect("set Z_IMAGE_BASE_SNAPSHOT to a Tongyi-MAI/Z-Image (base, non-Turbo) snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // Resolve the BASE engine id `z_image` through the registry (this crate's base inventory entry is
    // linked into the test binary). A small render with real CFG (guidance 4.0 + a negative prompt)
    // over the shift=6.0 schedule; few steps keep it cheap while still exercising the CFG loop.
    let gen = candle_gen::gen_core::registry::load("z_image", &spec)
        .expect("candle base z_image is registered");
    assert_eq!(gen.descriptor().id, "z_image");
    assert!(gen.descriptor().capabilities.supports_guidance);

    let req = GenerationRequest {
        prompt: "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting".into(),
        negative_prompt: Some("blurry, low quality, watermark".into()),
        guidance: Some(4.0),
        width: 512,
        height: 512,
        steps: Some(12),
        seed: Some(42),
        count: 1,
        ..Default::default()
    };

    let mut steps_seen = 0u32;
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            steps_seen = steps_seen.max(current);
            assert!(current <= total, "progress step {current} > total {total}");
        }
    };
    let out = gen.generate(&req, &mut on_progress).expect("base generate");
    let images = match out {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => panic!("expected images, got video"),
    };
    assert_eq!(images.len(), 1);
    let img = &images[0];
    assert_eq!(
        (img.width, img.height),
        (512, 512),
        "output dims == request"
    );
    assert_eq!(img.pixels.len(), 512 * 512 * 3, "RGB8 buffer size");
    assert!(steps_seen >= 1, "at least one denoise step reported");

    // Non-degenerate output: a coherent CFG render is neither a flat color nor pure noise. Assert the
    // pixel mean is in a sane mid-range and there is real spatial variance (std-dev well above 0).
    let n = img.pixels.len() as f64;
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let std = var.sqrt();
    assert!(
        (8.0..248.0).contains(&mean),
        "degenerate (saturated/black) render: mean={mean:.1}"
    );
    assert!(std > 8.0, "near-flat render (no structure): std={std:.1}");
    eprintln!("[base smoke] coherent 512x512 render: mean={mean:.1} std={std:.1}");
}
