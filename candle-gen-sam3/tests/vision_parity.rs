//! SAM3-A vision-encoder parity (sc-6240): load the real `facebook/sam3` weights, run the PE
//! backbone + FPN neck on candle, and check the four feature maps against the SAME torch oracle
//! fixture the MLX port uses (`mlx-gen/scripts/spikes/sam3_oracle/dump_vision_fixture.py`). The
//! candle output is the reimplementation-of-record against `mlx-gen-sam3` (cosine > 0.99, the MLX
//! bar). #[ignore] until `facebook/sam3` (gated) + the fixture are staged on the box (sc-6248).
//!
//! Run (CUDA build on the Blackwell box):
//!   SAM3_WEIGHTS=<facebook/sam3 snapshot dir OR model.safetensors> \
//!   SAM3_VISION_FIXTURE=<.../sam3_oracle/vision_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test vision_parity -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::Tensor;
use candle_gen_sam3::{Sam3VisionConfig, Sam3VisionEncoder, Weights};

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
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_VISION_FIXTURE — sc-6248"]
fn vision_encoder_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_VISION_FIXTURE").expect("set SAM3_VISION_FIXTURE to the oracle dump");

    let device = candle_gen::default_device().expect("default device");

    let wp = Path::new(&weights_path);
    let w = if wp.is_dir() {
        Weights::from_dir(wp, &device)
    } else {
        Weights::from_file(wp, &device)
    }
    .expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path, &device).expect("load vision fixture");

    let enc = Sam3VisionEncoder::from_weights(
        &w,
        "detector_model.vision_encoder",
        &Sam3VisionConfig::sam3(),
    )
    .expect("build vision encoder");

    let pixel_values = fx.require("pixel_values").expect("fixture pixel_values");
    let fpn = enc.forward(&pixel_values).expect("vision forward");

    assert_eq!(fpn.len(), 4, "expected 4 FPN levels");
    let mut worst_cos = 1.0f32;
    for (i, got_nhwc) in fpn.iter().enumerate() {
        // ours NHWC [1,H,W,256] → NCHW to match the fixture
        let got = got_nhwc
            .permute([0, 3, 1, 2])
            .unwrap()
            .contiguous()
            .unwrap();
        let want = fx.require(&format!("fpn_{i}")).expect("fixture fpn");
        assert_eq!(got.dims(), want.dims(), "fpn_{i} shape");
        let cos = cosine(&got, &want);
        worst_cos = worst_cos.min(cos);
        println!("fpn_{i} {:?}: cosine={cos:.6}", want.dims());
    }
    assert!(
        worst_cos > 0.99,
        "worst FPN cosine {worst_cos:.6} below 0.99"
    );
}
