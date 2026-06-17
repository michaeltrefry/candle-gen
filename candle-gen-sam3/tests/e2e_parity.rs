//! SAM3 end-to-end / PCS parity (sc-6243): run the full still-image segmenter (PE vision, CLIP text,
//! DETR, mask head) from the reference pixel_values and check the produced instance masks against the
//! SAME torch oracle fixture the MLX port uses (`mlx-gen/scripts/spikes/sam3_oracle/dump_e2e_fixture.py`).
//! The candle output is the reimplementation-of-record against `mlx-gen-sam3` (instance count exact +
//! per-instance mask IoU > 0.95, the MLX bar). Stays `#[ignore]` until `facebook/sam3` (gated) plus
//! the fixture are staged on the box (sc-6248).
//!
//! Run (CUDA build on the Blackwell box):
//!   SAM3_WEIGHTS=<facebook/sam3 snapshot dir OR model.safetensors> \
//!   SAM3_E2E_FIXTURE=<.../sam3_oracle/e2e_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test e2e_parity -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::Tensor;
use candle_gen::default_device;
use candle_gen_sam3::{Sam3ImageSegmenter, Weights};

/// IoU of two binary `[h, w]` masks (any dtype with 0/1 values).
fn iou(a: &Tensor, b: &Tensor) -> f32 {
    let af = a.to_dtype(candle_gen::candle_core::DType::F32).unwrap();
    let bf = b.to_dtype(candle_gen::candle_core::DType::F32).unwrap();
    let scalar = |t: Tensor| t.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let inter = scalar((&af * &bf).unwrap());
    let sa = scalar(af);
    let sb = scalar(bf);
    let union = sa + sb - inter;
    if union <= 0.0 {
        1.0
    } else {
        inter / union
    }
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_E2E_FIXTURE — sc-6248"]
fn full_segmenter_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_E2E_FIXTURE").expect("set SAM3_E2E_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");

    let wp = Path::new(&weights_path);
    let w = if wp.is_dir() {
        Weights::from_dir(wp, &device)
    } else {
        Weights::from_file(wp, &device)
    }
    .expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path, &device).expect("load e2e fixture");

    let seg = Sam3ImageSegmenter::from_weights(&w).expect("build segmenter");

    let pixel_values = fx.require("pixel_values").unwrap();
    let input_ids = fx.require("input_ids").unwrap();
    let mask: Vec<i32> = fx
        .require("attention_mask")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&m| m as i32)
        .collect();

    // target_wh (1,1) keeps boxes in [0,1]; masks come back at native 288².
    let got = seg
        .segment(&pixel_values, &input_ids, &mask, (1.0, 1.0), 0.5, 0.5)
        .expect("segment");

    let want_masks = fx.require("instance_masks").unwrap(); // [n,288,288]
    let want_n = want_masks.dim(0).unwrap();
    println!("instances: got {} want {}", got.len(), want_n);
    assert_eq!(got.len(), want_n, "instance count mismatch");

    // Reference instances and ours are both in query order → compare index-aligned.
    let mut worst_iou = 1.0f32;
    for (i, inst) in got.iter().enumerate() {
        let want = want_masks
            .narrow(0, i, 1)
            .unwrap()
            .reshape((288, 288))
            .unwrap();
        let m = iou(&inst.mask, &want);
        worst_iou = worst_iou.min(m);
        println!("  instance {i}: score={:.3} mask IoU={:.4}", inst.score, m);
    }
    assert!(
        worst_iou > 0.95,
        "worst instance mask IoU {worst_iou:.4} below 0.95"
    );
}
