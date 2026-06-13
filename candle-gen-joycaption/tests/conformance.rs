//! gen-core captioner-contract conformance for the candle JoyCaption provider (epic 3692 / sc-3699).
//!
//! Runs the backend-neutral [`gen_core_testkit`] captioner suite — validate-honesty, `Progress`
//! monotonicity, typed pre-cancellation, registry round-trip — against the real candle captioner.
//! The progress check drives a real `caption`, so it needs the CUDA backend + a local JoyCaption
//! snapshot (`JOYCAPTION_SNAPSHOT`) and is `#[ignore]`d by default:
//!
//! ```text
//! set JOYCAPTION_SNAPSHOT=C:\Users\…\models--fancyfeast--llama-joycaption-beta-one-hf-llava\snapshots\<hash>
//! cargo test -p candle-gen-joycaption --features cuda --release --test conformance -- --ignored
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{CaptionSampling, LoadSpec, WeightsSource};
use gen_core_testkit::{captioner_conformance, CaptionerProfile};

#[test]
#[ignore = "needs JOYCAPTION_SNAPSHOT + a CUDA GPU"]
fn joycaption_conformance() {
    let snap =
        std::env::var("JOYCAPTION_SNAPSHOT").expect("set JOYCAPTION_SNAPSHOT to a snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // The cheap profile keeps the real-lane caption short (greedy, 12 new tokens) — it verifies
    // contract behavior, not caption quality.
    let profile = CaptionerProfile {
        sampling: CaptionSampling {
            temperature: 0.0,
            max_new_tokens: 12,
            ..Default::default()
        },
        ..CaptionerProfile::cheap()
    };
    captioner_conformance(|| candle_gen_joycaption::load(&spec).unwrap(), &profile);
}
