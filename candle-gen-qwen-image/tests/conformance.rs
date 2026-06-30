//! gen-core contract conformance for the candle Qwen-Image provider (sc-4481, epic 3692 / sc-3696).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator.
//! Drives a real `generate`, so it needs the CUDA backend + a local Qwen-Image snapshot (with a
//! built `tokenizer/tokenizer.json`) and is `#[ignore]`d by default:
//!
//! ```text
//! set QWEN_IMAGE_SNAPSHOT=C:\Users\…\models--Qwen--Qwen-Image\snapshots\<hash>
//! cargo test -p candle-gen-qwen-image --features cuda --release --test conformance -- --ignored
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use gen_core_testkit::{conformance, Profile};

/// Resolve the base snapshot dir from env, preferring the sc-8647 / mlx-gen sc-8271 base refresh
/// `Qwen/Qwen-Image-2512` (`QWEN_2512_SNAPSHOT`, else the legacy `QWEN_BASE`) and falling back to the
/// original `Qwen/Qwen-Image` (`QWEN_IMAGE_SNAPSHOT`). The 2512 base is a structural drop-in (same
/// diffusers layout, same 60-layer MMDiT, same tokenizer), so this single smoke validates both.
fn base_snapshot() -> PathBuf {
    for var in ["QWEN_2512_SNAPSHOT", "QWEN_BASE", "QWEN_IMAGE_SNAPSHOT"] {
        if let Ok(p) = std::env::var(var) {
            return PathBuf::from(p);
        }
    }
    panic!(
        "set QWEN_2512_SNAPSHOT (Qwen/Qwen-Image-2512) or QWEN_IMAGE_SNAPSHOT (Qwen/Qwen-Image) \
         to a snapshot dir w/ a built tokenizer/tokenizer.json"
    );
}

#[test]
#[ignore = "needs QWEN_2512_SNAPSHOT/QWEN_IMAGE_SNAPSHOT (a Qwen-Image snapshot dir w/ tokenizer.json) + a CUDA GPU"]
fn qwen_image_conformance() {
    let spec = LoadSpec::new(WeightsSource::Dir(base_snapshot()));

    // 256² / 4 steps keeps the suite's ~4 generate() calls affordable on the 20B DiT (each step is
    // two CFG forwards). Verifies contract behavior, not image quality.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 4,
        ..Profile::cheap()
    };
    conformance(|| candle_gen_qwen_image::load(&spec).unwrap(), &profile);
}

/// sc-8647 (parity with mlx-gen sc-8271): real-weight t2i smoke for the `Qwen/Qwen-Image-2512` base
/// refresh. Loads the 2512 snapshot (or the legacy base — both are structural drop-ins) and renders
/// a small image, asserting the output is coherent (right shape, not a degenerate flat/black frame).
///
/// The 2512 snapshot is a multi-tens-of-GB download and is NOT present in CI; run on-device:
/// ```text
/// set QWEN_2512_SNAPSHOT=…\models--Qwen--Qwen-Image-2512\snapshots\<hash>
/// cargo test -p candle-gen-qwen-image --features cuda --release --test conformance \
///     qwen_image_2512_t2i_smoke -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs the Qwen/Qwen-Image-2512 base snapshot (~40 GB) + a CUDA GPU (Phase-B on-device, sc-8246)"]
fn qwen_image_2512_t2i_smoke() {
    use candle_gen::gen_core::{GenerationOutput, GenerationRequest, Progress};

    let spec = LoadSpec::new(WeightsSource::Dir(base_snapshot()));
    let gen = candle_gen_qwen_image::load(&spec).expect("load Qwen-Image-2512 base");

    let req = GenerationRequest {
        prompt: "a rusty robot holding a lit candle, studio photo".into(),
        width: 512,
        height: 512,
        steps: Some(20),
        guidance: Some(4.0),
        seed: Some(0),
        count: 1,
        ..Default::default()
    };

    let mut last = None;
    let out = gen
        .generate(&req, &mut |p: Progress| last = Some(p))
        .expect("2512 t2i render");
    let GenerationOutput::Images(imgs) = out else {
        panic!("expected image output");
    };
    assert_eq!(imgs.len(), 1, "count=1 → one image");
    let img = &imgs[0];
    assert_eq!(
        (img.width, img.height),
        (512, 512),
        "rendered at requested size"
    );
    assert_eq!(
        img.pixels.len(),
        512 * 512 * 3,
        "RGB8 pixel buffer of the right size"
    );

    // Coherence guard: a healthy render has real tonal variation. A degenerate (all-black / flat /
    // NaN-wash) frame collapses to ~zero spread. This catches a broken 2512 load (wrong config,
    // missing tokenizer overlay, quant no-op) without pinning exact pixels.
    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() as f64 / img.pixels.len() as f64;
    assert!(
        max - min > 32 && (8.0..248.0).contains(&mean),
        "render looks degenerate (min={min} max={max} mean={mean:.1}) — 2512 base may not have \
         loaded coherently"
    );
}
