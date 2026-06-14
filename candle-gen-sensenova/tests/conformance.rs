//! gen-core contract conformance for the candle SenseNova-U1 provider (sc-4481, epic 3692 / sc-5486).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator. The
//! **seed-determinism** check is the regression guard for the deterministic CPU-seeded noise (sc-3673
//! parity) the denoise loop relies on.
//!
//! It drives a real `generate`, so it needs the CUDA backend + a local SenseNova-U1-8B-MoT snapshot
//! and is `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set SENSENOVA_SNAPSHOT=C:\Users\…\models--sensenova--SenseNova-U1-8B-MoT\snapshots\<hash>
//! cargo test -p candle-gen-sensenova --features cuda --release --test conformance -- --ignored
//! ```
//!
//! The snapshot must carry a materialized `tokenizer.json` (the checkpoint ships only vocab.json +
//! merges.txt — run SenseNova's `tools/build_sensenova_tokenizer.py` once). As with the other candle
//! slices: same-backend determinism only; cross-backend pixel equality vs `mlx-gen-sensenova` is NOT a
//! goal (RNG algorithms differ).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use gen_core_testkit::{conformance, Profile};

#[test]
#[ignore = "needs SENSENOVA_SNAPSHOT (a SenseNova-U1-8B-MoT snapshot dir with a materialized tokenizer.json) + a CUDA GPU; run with --features cuda --ignored"]
fn sensenova_conformance() {
    let snap = std::env::var("SENSENOVA_SNAPSHOT")
        .expect("set SENSENOVA_SNAPSHOT to a sensenova/SenseNova-U1-8B-MoT snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 256² (== the descriptor's min_size, 32-aligned) and a tiny step count keep the suite's handful
    // of generate() calls cheap — it verifies contract behavior, not image quality.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 2,
        ..Profile::cheap()
    };
    conformance(|| candle_gen_sensenova::load(&spec).unwrap(), &profile);
}
