//! SAM3 text-encoder parity (sc-6241): load the real `facebook/sam3` weights, run the CLIP-H text
//! tower + `text_projection`, and check the projected text features against the SAME torch oracle
//! fixture the MLX port uses (`mlx-gen/scripts/spikes/sam3_oracle/dump_text_fixture.py`). Optionally
//! checks the CLIP tokenizer reproduces the reference ids. The candle output is the
//! reimplementation-of-record against `mlx-gen-sam3` (valid-token cosine > 0.999, the MLX bar).
//! #[ignore] until `facebook/sam3` (gated) + the fixture are staged on the box (sc-6248).
//!
//! Run (CUDA build on the Blackwell box):
//!   SAM3_WEIGHTS=<facebook/sam3 snapshot dir OR model.safetensors> \
//!   SAM3_TEXT_FIXTURE=<.../sam3_oracle/text_fixture.safetensors> \
//!   SAM3_TOKENIZER=<.../tokenizer.json> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test text_parity -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::Tensor;
use candle_gen::default_device;
use candle_gen_sam3::{Sam3TextConfig, Sam3TextEncoder, Sam3Tokenizer, Weights};

/// Cosine similarity between two tensors (flattened).
fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let na = (&a * &a)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let nb = (&b * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_TEXT_FIXTURE — sc-6248"]
fn text_encoder_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_TEXT_FIXTURE").expect("set SAM3_TEXT_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let cfg = Sam3TextConfig::sam3();

    let wp = Path::new(&weights_path);
    let w = if wp.is_dir() {
        Weights::from_dir(wp, &device)
    } else {
        Weights::from_file(wp, &device)
    }
    .expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path, &device).expect("load text fixture");

    let enc = Sam3TextEncoder::from_weights(
        &w,
        "detector_model.text_encoder.text_model",
        "detector_model.text_projection",
        &cfg,
    )
    .expect("build text encoder");

    // Optional: verify the tokenizer reproduces the reference ids for "person".
    if let Ok(tok_path) = std::env::var("SAM3_TOKENIZER") {
        let tok = Sam3Tokenizer::from_file(&tok_path, &cfg).expect("load tokenizer");
        let (ids, mask) = tok.encode("person", &device).expect("tokenize");
        let ids_host: Vec<u32> = ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(&ids_host[..3], &[49406, 2533, 49407], "person ids[:3]");
        assert_eq!(ids_host.len(), 32, "padded to 32");
        assert_eq!(&mask[..4], &[1, 1, 1, 0], "attention mask");
        println!("tokenizer: 'person' → ids[:3]={:?} ok", &ids_host[..3]);
    }

    let mut worst = 1.0f32;
    for concept in ["person", "car"] {
        let input_ids = fx.require(&format!("{concept}.input_ids")).unwrap();
        let mask: Vec<i32> = fx
            .require(&format!("{concept}.attention_mask"))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|&m| m as i32)
            .collect();
        let n_valid = mask.iter().filter(|&&m| m == 1).count();

        let got = enc.forward(&input_ids, &mask).expect("text forward"); // [1,N,256]
        let want = fx.require(&format!("{concept}.text_features")).unwrap();
        assert_eq!(got.dims(), want.dims(), "{concept} text_features shape");

        // Slice the valid (non-padding) tokens — what the detector consumes.
        let got_v = got.narrow(1, 0, n_valid).unwrap();
        let want_v = want.narrow(1, 0, n_valid).unwrap();

        let cos_full = cosine(&got, &want);
        let cos_valid = cosine(&got_v, &want_v);
        worst = worst.min(cos_valid);
        println!(
            "{concept}: cosine_valid({n_valid} tok)={cos_valid:.6}  cosine_full={cos_full:.6}"
        );
    }
    assert!(
        worst > 0.999,
        "worst valid-token cosine {worst:.6} below 0.999"
    );
}
