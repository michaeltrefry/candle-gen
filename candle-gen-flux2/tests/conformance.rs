//! gen-core contract conformance for the candle FLUX.2-klein provider (sc-4481, epic 3692 / sc-3695).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator.
//! The **seed-determinism** check is the regression guard for the deterministic CPU-seeded noise
//! (sc-3673 parity).
//!
//! It drives a real `generate`, so it needs the CUDA backend + a local FLUX.2-klein-9B snapshot and
//! is `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set FLUX2_SNAPSHOT=C:\Users\…\models--black-forest-labs--FLUX.2-klein-9B\snapshots\<hash>
//! cargo test -p candle-gen-flux2 --features cuda --release --test conformance -- --ignored
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use gen_core_testkit::{conformance, Profile};

#[test]
#[ignore = "needs FLUX2_SNAPSHOT (a FLUX.2-klein-9B snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn flux2_klein_9b_conformance() {
    let snap = std::env::var("FLUX2_SNAPSHOT")
        .expect("set FLUX2_SNAPSHOT to a black-forest-labs/FLUX.2-klein-9B snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 256² (== the descriptor's min_size) keeps the suite's ~4 generate() calls cheap. The distilled
    // default is 4 steps; the pipeline uses req.steps verbatim, so `steps: 4` → Step.total == 4.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 4,
        ..Profile::cheap()
    };
    conformance(|| candle_gen_flux2::load_klein(&spec).unwrap(), &profile);
}
