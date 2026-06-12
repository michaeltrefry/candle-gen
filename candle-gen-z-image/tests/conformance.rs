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
