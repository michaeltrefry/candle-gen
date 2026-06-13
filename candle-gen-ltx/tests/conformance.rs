//! gen-core contract conformance for the candle LTX-2.3 provider (epic 3692 / sc-3698).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator.
//! Drives a real `generate`, so it needs the CUDA backend + a local LTX-2.3 snapshot + a Gemma-3-12B
//! encoder snapshot (`LTX_GEMMA_DIR`) and is `#[ignore]`d by default:
//!
//! ```text
//! set LTX_SNAPSHOT=C:\Users\…\models--Lightricks--LTX-2.3\snapshots\<hash>
//! set LTX_GEMMA_DIR=C:\Users\…\models--google--gemma-3-12b-it\snapshots\<hash>
//! cargo test -p candle-gen-ltx --features cuda --release --test conformance -- --ignored
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor, Progress,
    WeightsSource,
};
use gen_core_testkit::{conformance, Profile};

/// A test-only wrapper pinning `frames` to a tiny count so the suite's ~4 `generate()` calls stay
/// affordable on the 22B DiT (the `Profile` can't set `frames`). 9 frames → 2 latent frames.
struct TinyClip(Box<dyn Generator>);

impl Generator for TinyClip {
    fn descriptor(&self) -> &ModelDescriptor {
        self.0.descriptor()
    }
    fn validate(&self, req: &GenerationRequest) -> candle_gen::gen_core::Result<()> {
        self.0.validate(req)
    }
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<GenerationOutput> {
        let mut r = req.clone();
        r.frames = Some(9);
        self.0.generate(&r, on_progress)
    }
}

#[test]
#[ignore = "needs LTX_SNAPSHOT + LTX_GEMMA_DIR + a CUDA GPU"]
fn ltx_conformance() {
    let snap = std::env::var("LTX_SNAPSHOT").expect("set LTX_SNAPSHOT to an LTX-2.3 snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 256² / 9 frames keeps the suite affordable. Verifies contract behavior, not quality. The
    // distilled sigma schedule is fixed (8 steps), so `steps` is informational here.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 8,
        ..Profile::cheap()
    };
    conformance(
        || Box::new(TinyClip(candle_gen_ltx::load(&spec).unwrap())),
        &profile,
    );
}
