//! Numerical parity of the candle face sub-models vs the antelopev2 onnx reference (sc-5490). The
//! goldens — produced by mlx-gen's `tools/convert_glintr100.py` / `convert_scrfd.py` and shared with
//! the MLX parity tests — carry the deterministic normalized inputs and the **onnx** outputs, so this
//! validates candle directly against onnx (no MLX / Mac needed). Cosine / max-abs gates are identical
//! to the MLX `arcface_parity.rs` / `scrfd_parity.rs`.
//!
//! `#[ignore]`d: the goldens are large and local-only. Point [`golden_dir`] at them and run:
//!   set FACE_GOLDEN_DIR=D:\repos\mlx-gen\tools\golden
//!   cargo test -p candle-gen-face -- --ignored --nocapture

use std::path::PathBuf;

use candle_gen::candle_core::{safetensors, Device, Tensor};

use crate::common::Weights;
use crate::iresnet::ArcFace;
use crate::scrfd::Scrfd;

/// The directory holding the converted weights + goldens (`FACE_GOLDEN_DIR`, default the mlx-gen
/// checkout's `tools/golden`).
fn golden_dir() -> PathBuf {
    std::env::var("FACE_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"D:\repos\mlx-gen\tools\golden"))
}

/// Validation device: CPU by default (proves the port math), or `FACE_DEVICE=cuda` to confirm parity
/// on the actual GPU the worker runs (requires a `--features cuda` build).
fn device() -> Device {
    match std::env::var("FACE_DEVICE").as_deref() {
        Ok("cuda") => Device::new_cuda(0).expect("cuda device (build with --features cuda)"),
        _ => Device::Cpu,
    }
}

fn load_goldens(name: &str, device: &Device) -> std::collections::HashMap<String, Tensor> {
    let path = golden_dir().join(name);
    safetensors::load(&path, device).unwrap_or_else(|e| {
        panic!(
            "missing golden {}: {e}\nRun the convert_*.py tool first.",
            path.display()
        )
    })
}

/// Per-row cosine of `[K, dim]` candle output vs `[K, dim]` onnx golden.
fn row_cosines(got: &Tensor, want: &Tensor) -> Vec<f32> {
    let (k, dim) = (want.dim(0).unwrap(), want.dim(1).unwrap());
    assert_eq!(got.dims(), &[k, dim], "embedding shape mismatch");
    let gv = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let wv = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    (0..k)
        .map(|i| {
            let (a, b) = (&gv[i * dim..(i + 1) * dim], &wv[i * dim..(i + 1) * dim]);
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..dim {
                dot += a[j] as f64 * b[j] as f64;
                na += (a[j] as f64).powi(2);
                nb += (b[j] as f64).powi(2);
            }
            (dot / (na.sqrt() * nb.sqrt())) as f32
        })
        .collect()
}

/// ArcFace (iresnet100 / glintr100) embedding cosine ≥ 0.9999 vs onnx — the fidelity gate the whole
/// identity stack rides on (InstantID / PuLID consume this exact embedding).
#[test]
#[ignore = "needs local goldens from tools/convert_glintr100.py"]
fn arcface_cosine_parity() {
    let device = device();
    let w = Weights::from_file(
        &golden_dir().join("arcface_iresnet100.safetensors"),
        &device,
    )
    .unwrap();
    let net = ArcFace::from_weights(&w).unwrap();

    let g = load_goldens("arcface_goldens.safetensors", &device);
    let inputs = g.get("inputs").expect("goldens.inputs"); // [K,112,112,3] NHWC, normalized
    let want = g.get("embeddings").expect("goldens.embeddings"); // [K,512] raw onnx
    let nchw = inputs.permute((0, 3, 1, 2)).unwrap().contiguous().unwrap();

    let got = net.forward(&nchw).unwrap();
    let cosines = row_cosines(&got, want);
    let min_cos = cosines.iter().cloned().fold(f32::INFINITY, f32::min);
    for (i, c) in cosines.iter().enumerate() {
        println!("face {i}: cosine = {c:.8}");
    }
    println!("ArcFace min cosine = {min_cos:.8}");
    assert!(
        min_cos >= 0.9999,
        "ArcFace embedding cosine {min_cos:.8} < 0.9999 vs glintr100 onnx"
    );
}

/// SCRFD raw per-stride network outputs (score / bbox / kps) vs onnx — the detector-network gate.
/// (The full detect-vs-insightface gate lives in the MLX test; it needs insightface + the t1.jpg
/// fixture, which this candle-side check intentionally does not pull. The raw-output match proves the
/// backbone + neck + heads port faithfully; the decode/NMS is plain host math already unit-tested.)
#[test]
#[ignore = "needs local goldens from tools/convert_scrfd.py"]
fn scrfd_raw_output_parity() {
    let device = device();
    let w = Weights::from_file(&golden_dir().join("scrfd_10g.safetensors"), &device).unwrap();
    let net = Scrfd::from_weights(&w).unwrap();

    let g = load_goldens("scrfd_goldens.safetensors", &device);
    let input = g.get("input").expect("goldens.input"); // [1,640,640,3] NHWC f32 blob
    let nchw = input.permute((0, 3, 1, 2)).unwrap().contiguous().unwrap();

    let raw = net.raw_outputs(&nchw).unwrap();
    let mut worst = 0.0f32;
    for (stride, scores, bbox, kps) in &raw {
        for (label, got) in [("score", scores), ("bbox", bbox), ("kps", kps)] {
            let want = g
                .get(&format!("{label}.{stride}"))
                .unwrap_or_else(|| panic!("golden {label}.{stride}"));
            assert_eq!(got.dims(), want.dims(), "{label}.{stride} shape");
            let gv = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let wv = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let m = gv
                .iter()
                .zip(&wv)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("  raw {label}.{stride}: max abs diff = {m:.6}");
            worst = worst.max(m);
        }
    }
    println!("SCRFD network parity worst max abs diff = {worst:.6}");
    assert!(worst < 5e-3, "SCRFD network diverged from onnx: {worst}");
}
