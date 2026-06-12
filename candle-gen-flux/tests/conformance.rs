//! gen-core contract conformance for the candle FLUX.1 provider (sc-4481, epic 3692 / sc-3694).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator for
//! BOTH variants. The **seed-determinism** check is the regression guard for the deterministic
//! CPU-seeded noise (sc-3673 parity) the pipeline relies on.
//!
//! Each drives a real `generate`, so it needs the CUDA backend + a local FLUX.1 snapshot and is
//! `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set FLUX_SCHNELL_SNAPSHOT=C:\Users\…\models--black-forest-labs--FLUX.1-schnell\snapshots\<hash>
//! set FLUX_DEV_SNAPSHOT=C:\Users\…\models--black-forest-labs--FLUX.1-dev\snapshots\<hash>
//! cargo test -p candle-gen-flux --features cuda --release --test conformance -- --ignored
//! ```
//!
//! Passing the suite is the parity evidence sc-3694 asks for: it locks **output dims** (request WxH
//! is the emitted image size), **seed semantics** (same request+seed ⇒ byte-identical output),
//! **distilled step count** (`Step.total` == resolved steps), **contract field shape** (validate-
//! honesty + `GenerationOutput::Images`), and **cancellation/progress** (typed `Canceled`, monotone
//! `Step`). As with the SDXL/Z-Image slices: same-backend determinism only; cross-backend pixel
//! equality vs `mlx-gen-flux` is NOT a goal (RNG algorithms differ).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use gen_core_testkit::{conformance, Profile};

#[test]
#[ignore = "needs FLUX_SCHNELL_SNAPSHOT (a FLUX.1-schnell snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn flux_schnell_conformance() {
    let snap = std::env::var("FLUX_SCHNELL_SNAPSHOT")
        .expect("set FLUX_SCHNELL_SNAPSHOT to a black-forest-labs/FLUX.1-schnell snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 256² (== the descriptor's min_size) keeps the suite's ~4 generate() calls cheap — it verifies
    // contract behavior, not image quality. schnell's distilled default is 4 steps; the pipeline uses
    // req.steps verbatim, so `steps: 4` → Step.total == 4 (check_progress). Both dims are multiples
    // of the /16 alignment the provider requires.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 4,
        ..Profile::cheap()
    };
    conformance(|| candle_gen_flux::load_schnell(&spec).unwrap(), &profile);
}

#[test]
#[ignore = "needs FLUX_DEV_SNAPSHOT (a FLUX.1-dev snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn flux_dev_conformance() {
    let snap = std::env::var("FLUX_DEV_SNAPSHOT")
        .expect("set FLUX_DEV_SNAPSHOT to a black-forest-labs/FLUX.1-dev snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // dev is heavier (guidance + 25-step default); keep the conformance run at 256²/6 steps so the
    // ~4 generate() calls stay affordable while still exercising the guidance/time-shift path.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 6,
        ..Profile::cheap()
    };
    conformance(|| candle_gen_flux::load_dev(&spec).unwrap(), &profile);
}
