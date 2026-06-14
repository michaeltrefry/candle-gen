//! Real-weights TextLlm conformance for the candle prompt-refine provider (sc-5500).
//!
//! `#[ignore]`d so the default suite stays weight-free: it runs a real `generate`, so it needs a
//! Llama-3.2-3B-Instruct snapshot dir (`config.json`, `tokenizer.json`, `model-*.safetensors`) in
//! `PROMPT_REFINE_MODEL_DIR` and a working device (CPU f32 works for a smoke; CUDA for the real run).
//! Drives the provider through the public `gen_core::TextLlm` contract — validate honesty, `Progress`
//! monotonicity, pre-inference typed cancellation, registry round-trip.
//!
//! Run: `set PROMPT_REFINE_MODEL_DIR=...; cargo test -p candle-gen-prompt-refine --features cuda -- --ignored`

use candle_gen::gen_core::{registry, LoadSpec, WeightsSource};
use candle_gen_prompt_refine::prompt::PROMPT_REFINE_ID;
use gen_core_testkit::{textllm_conformance, TextLlmProfile};

#[test]
#[ignore = "needs a Llama-3.2-3B-Instruct snapshot in PROMPT_REFINE_MODEL_DIR + a device"]
fn prompt_refine_textllm_conformance() {
    let dir = std::env::var("PROMPT_REFINE_MODEL_DIR").expect(
        "set PROMPT_REFINE_MODEL_DIR to a Llama-3.2-3B-Instruct snapshot directory \
         (config.json, tokenizer.json, model-*.safetensors)",
    );
    let spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
    textllm_conformance(
        || registry::load_textllm(PROMPT_REFINE_ID, &spec).expect("load prompt_refine provider"),
        &TextLlmProfile::cheap(),
    );
}
