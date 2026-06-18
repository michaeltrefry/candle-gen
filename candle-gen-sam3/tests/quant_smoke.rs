//! SAM3 quantization smoke (sc-6246): build the image segmenter + tracker dense / Q8 / Q4 from the
//! real `facebook/sam3` weights and check that Q8 stays near-lossless vs the dense baseline while Q4
//! stays coherent (the candle twin of `mlx-gen-sam3`'s `quant_smoke`). `#[ignore]` until weights +
//! fixtures are staged on the Blackwell box (sc-6248). Run:
//!   SAM3_WEIGHTS=<snapshot> SAM3_E2E_FIXTURE=<e2e_fixture.safetensors> \
//!   SAM3_TRACKER_FIXTURE=<tracker_fixture.safetensors> \
//!     cargo test -p candle-gen-sam3 --release --features cuda --test quant_smoke -- --ignored --nocapture

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::default_device;
use candle_gen::gen_core::Quant;
use candle_gen_sam3::{Instance, Sam3ImageSegmenter, Sam3Tracker, Weights};

fn sum_scalar(t: Tensor) -> f32 {
    t.sum_all().unwrap().to_scalar::<f32>().unwrap()
}

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = sum_scalar((&a * &b).unwrap());
    let na = sum_scalar((&a * &a).unwrap()).sqrt();
    let nb = sum_scalar((&b * &b).unwrap()).sqrt();
    dot / (na * nb)
}

/// IoU of two binary (or sign-thresholded) masks.
fn iou(a: &Tensor, b: &Tensor) -> f32 {
    let af = a.to_dtype(DType::F32).unwrap();
    let bf = b.to_dtype(DType::F32).unwrap();
    let inter = sum_scalar((&af * &bf).unwrap());
    let sa = sum_scalar(af);
    let sb = sum_scalar(bf);
    let union = sa + sb - inter;
    if union <= 0.0 {
        1.0
    } else {
        inter / union
    }
}

/// Sign-threshold mask logits at 0 → {0,1} f32 (the binary mask depends only on the logit sign).
fn binarize(logits: &Tensor) -> Tensor {
    logits.gt(0f64).unwrap().to_dtype(DType::F32).unwrap()
}

/// Worst index-aligned mask IoU between two instance lists (must be equal length).
fn worst_iou(a: &[Instance], b: &[Instance]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| iou(&x.mask, &y.mask))
        .fold(1.0f32, f32::min)
}

fn load_weights(path: &str, device: &Device) -> Weights {
    let wp = Path::new(path);
    if wp.is_dir() {
        Weights::from_dir(wp, device)
    } else {
        Weights::from_file(wp, device)
    }
    .expect("load sam3 weights")
}

fn fx_i32(fx: &Weights, key: &str) -> Vec<i32> {
    fx.require(key)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&v| v as i32)
        .collect()
}

fn segment(seg: &Sam3ImageSegmenter, fx: &Weights) -> Vec<Instance> {
    let pixel_values = fx.require("pixel_values").unwrap();
    let input_ids = fx.require("input_ids").unwrap();
    let mask = fx_i32(fx, "attention_mask");
    seg.segment(&pixel_values, &input_ids, &mask, (1.0, 1.0), 0.5, 0.5)
        .expect("segment")
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_E2E_FIXTURE — sc-6248"]
fn quantized_segmenter_stays_close_to_dense() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_E2E_FIXTURE").expect("set SAM3_E2E_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load e2e fixture");

    // Dense baseline (a fresh model per precision — quantize mutates in place).
    let dense = segment(&Sam3ImageSegmenter::from_weights(&w).unwrap(), &fx);

    let mut q8 = Sam3ImageSegmenter::from_weights(&w).unwrap();
    q8.quantize(Quant::Q8).expect("quantize q8");
    let q8_inst = segment(&q8, &fx);

    let mut q4 = Sam3ImageSegmenter::from_weights(&w).unwrap();
    q4.quantize(Quant::Q4).expect("quantize q4");
    let q4_inst = segment(&q4, &fx);

    println!(
        "instances: dense {} | Q8 {} | Q4 {}",
        dense.len(),
        q8_inst.len(),
        q4_inst.len()
    );

    // Q8 is near-lossless: same instance set as dense, near-identical masks.
    assert_eq!(q8_inst.len(), dense.len(), "Q8 instance count != dense");
    let q8_iou = worst_iou(&dense, &q8_inst);
    println!("Q8 vs dense: worst mask IoU = {q8_iou:.4}");
    assert!(q8_iou > 0.95, "Q8 worst mask IoU {q8_iou:.4} below 0.95");

    // Q4 stays coherent: it still finds the people.
    assert!(!q4_inst.is_empty(), "Q4 found no instances");
    if q4_inst.len() == dense.len() {
        let q4_iou = worst_iou(&dense, &q4_inst);
        println!("Q4 vs dense: worst mask IoU = {q4_iou:.4}");
        assert!(q4_iou > 0.80, "Q4 worst mask IoU {q4_iou:.4} below 0.80");
    } else {
        println!(
            "Q4 instance count {} differs from dense {} (coarser quant; coherence-only check)",
            q4_inst.len(),
            dense.len()
        );
    }
}

#[test]
#[ignore = "needs staged facebook/sam3 weights (SAM3_WEIGHTS) + SAM3_TRACKER_FIXTURE — sc-6248"]
fn quantized_tracker_stays_close_to_dense() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to the facebook/sam3 snapshot");
    let fixture_path =
        std::env::var("SAM3_TRACKER_FIXTURE").expect("set SAM3_TRACKER_FIXTURE to the oracle dump");

    let device = default_device().expect("default device");
    let w = load_weights(&weights_path, &device);
    let fx = Weights::from_file(&fixture_path, &device).expect("load tracker fixture");

    let pixel_values = fx.require("pixel_values").unwrap();
    let box_v = fx
        .require("box_1008")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let box_xyxy = [box_v[0], box_v[1], box_v[2], box_v[3]];

    let dense = Sam3Tracker::from_weights(&w)
        .unwrap()
        .segment(&pixel_values, box_xyxy)
        .expect("dense segment");

    let mut q8 = Sam3Tracker::from_weights(&w).unwrap();
    q8.quantize(Quant::Q8).expect("quantize tracker q8");
    let q8 = q8.segment(&pixel_values, box_xyxy).expect("q8 segment");

    let cos = cosine(&dense.low_res, &q8.low_res);
    let mask_iou = iou(&binarize(&dense.low_res), &binarize(&q8.low_res));
    println!(
        "tracker Q8 vs dense: mask logit cosine={cos:.5} binary IoU={mask_iou:.4} (iou dense={:.3} q8={:.3})",
        dense.iou, q8.iou
    );
    // Primary near-lossless gate: the mask logits are essentially identical (cosine ~1).
    assert!(
        cos > 0.999,
        "tracker Q8 mask logit cosine {cos:.5} below 0.999"
    );
    assert!(
        mask_iou > 0.85,
        "tracker Q8 binary IoU {mask_iou:.4} below 0.85"
    );
}
