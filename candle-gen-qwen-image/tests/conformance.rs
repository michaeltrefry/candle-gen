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

#[test]
#[ignore = "needs QWEN_IMAGE_SNAPSHOT (a Qwen-Image snapshot dir w/ tokenizer.json) + a CUDA GPU"]
fn qwen_image_conformance() {
    let snap = std::env::var("QWEN_IMAGE_SNAPSHOT")
        .expect("set QWEN_IMAGE_SNAPSHOT to a Qwen/Qwen-Image snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

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
