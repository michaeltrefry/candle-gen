//! Depth Anything V2 port — non-weight shape/contract tests (build a tiny synthetic checkpoint and
//! assert the backbone → neck → head graph wires up to the right shapes) plus an `#[ignore]`
//! real-weight CUDA/CPU smoke (the maintainer's on-device gate). The candle twin of
//! `mlx-gen-depth`'s `tests/depth_shapes.rs`.
//!
//! The synthetic tests use a **downscaled** config (small embed dim, a 4-patch grid) so they run on
//! CPU in milliseconds while still exercising every module: the DINOv2 layers, the four-level DPT
//! reassemble (each factor branch: transposed-conv up, identity, strided-conv down), the RefineNet
//! fusion stage, and the head's upsample-to-input. The architecture (key names, factor logic, fusion
//! ordering) is what these lock; numeric fidelity is the real-weight smoke's job.

use std::collections::HashMap;

use candle_gen::candle_core::{Device, Tensor};

use candle_gen_depth::common::Weights;
use candle_gen_depth::config::DepthAnythingConfig;
use candle_gen_depth::{preprocess, DepthAnythingV2};

/// A tiny DA-V2-shaped config: 8-dim embed, 2 heads, 4 layers, grid 4 (image 8 / patch 2). Keeps the
/// real factor set [4,2,1,0.5] and channel ladder so the neck/head graph is identical to ViT-S, just
/// smaller.
fn tiny_cfg() -> DepthAnythingConfig {
    DepthAnythingConfig {
        hidden_size: 8,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        mlp_ratio: 2,
        num_channels: 3,
        image_size: 8,
        patch_size: 2,
        layer_norm_eps: 1e-6,
        out_indices: [1, 2, 3, 4],
        neck_hidden_sizes: [3, 4, 5, 6],
        reassemble_factors: [4.0, 2.0, 1.0, 0.5],
        fusion_hidden_size: 6,
        head_hidden_size: 4,
    }
}

fn dev() -> Device {
    Device::Cpu
}

/// A small constant tensor (a low non-zero value keeps the synthetic forward finite and non-degenerate).
fn ones(map: &mut HashMap<String, Tensor>, key: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    let t = Tensor::from_vec(vec![0.02f32; n], shape, &dev()).unwrap();
    map.insert(key.to_string(), t);
}

/// Build a complete synthetic checkpoint for `cfg` (every key the loader requires, OIHW/IOHW conv
/// weights as the real checkpoint ships them — candle convs consume them natively, no permute).
fn synth_weights(cfg: &DepthAnythingConfig) -> Weights {
    let mut w: HashMap<String, Tensor> = HashMap::new();
    let h = cfg.hidden_size;
    let grid = cfg.grid();
    let inter = cfg.intermediate_size();

    // --- backbone ---
    ones(
        &mut w,
        "backbone.embeddings.patch_embeddings.projection.weight",
        &[h, cfg.num_channels, cfg.patch_size, cfg.patch_size],
    );
    ones(
        &mut w,
        "backbone.embeddings.patch_embeddings.projection.bias",
        &[h],
    );
    ones(&mut w, "backbone.embeddings.cls_token", &[1, 1, h]);
    ones(
        &mut w,
        "backbone.embeddings.position_embeddings",
        &[1, grid * grid + 1, h],
    );
    for i in 0..cfg.num_hidden_layers {
        let p = format!("backbone.encoder.layer.{i}");
        for leaf in ["norm1", "norm2"] {
            ones(&mut w, &format!("{p}.{leaf}.weight"), &[h]);
            ones(&mut w, &format!("{p}.{leaf}.bias"), &[h]);
        }
        for leaf in ["query", "key", "value"] {
            ones(
                &mut w,
                &format!("{p}.attention.attention.{leaf}.weight"),
                &[h, h],
            );
            ones(
                &mut w,
                &format!("{p}.attention.attention.{leaf}.bias"),
                &[h],
            );
        }
        ones(
            &mut w,
            &format!("{p}.attention.output.dense.weight"),
            &[h, h],
        );
        ones(&mut w, &format!("{p}.attention.output.dense.bias"), &[h]);
        ones(&mut w, &format!("{p}.layer_scale1.lambda1"), &[h]);
        ones(&mut w, &format!("{p}.layer_scale2.lambda1"), &[h]);
        ones(&mut w, &format!("{p}.mlp.fc1.weight"), &[inter, h]);
        ones(&mut w, &format!("{p}.mlp.fc1.bias"), &[inter]);
        ones(&mut w, &format!("{p}.mlp.fc2.weight"), &[h, inter]);
        ones(&mut w, &format!("{p}.mlp.fc2.bias"), &[h]);
    }
    ones(&mut w, "backbone.layernorm.weight", &[h]);
    ones(&mut w, "backbone.layernorm.bias", &[h]);

    // --- neck: reassemble + convs + fusion ---
    let fh = cfg.fusion_hidden_size;
    for i in 0..4 {
        let nh = cfg.neck_hidden_sizes[i];
        let p = format!("neck.reassemble_stage.layers.{i}");
        ones(&mut w, &format!("{p}.projection.weight"), &[nh, h, 1, 1]);
        ones(&mut w, &format!("{p}.projection.bias"), &[nh]);
        let factor = cfg.reassemble_factors[i];
        if factor > 1.0 {
            let k = factor as usize;
            // ConvTranspose2d weight is IOHW: [in=nh, out=nh, k, k].
            ones(&mut w, &format!("{p}.resize.weight"), &[nh, nh, k, k]);
            ones(&mut w, &format!("{p}.resize.bias"), &[nh]);
        } else if factor < 1.0 {
            // Conv2d 3×3 downsample: OIHW [out=nh, in=nh, 3, 3].
            ones(&mut w, &format!("{p}.resize.weight"), &[nh, nh, 3, 3]);
            ones(&mut w, &format!("{p}.resize.bias"), &[nh]);
        }
        // neck.convs.{i}: 3×3 no-bias OIHW [out=fh, in=nh, 3, 3].
        ones(&mut w, &format!("neck.convs.{i}.weight"), &[fh, nh, 3, 3]);
        // fusion layer i.
        let fp = format!("neck.fusion_stage.layers.{i}");
        for res in ["residual_layer1", "residual_layer2"] {
            for c in ["convolution1", "convolution2"] {
                ones(&mut w, &format!("{fp}.{res}.{c}.weight"), &[fh, fh, 3, 3]);
                ones(&mut w, &format!("{fp}.{res}.{c}.bias"), &[fh]);
            }
        }
        ones(&mut w, &format!("{fp}.projection.weight"), &[fh, fh, 1, 1]);
        ones(&mut w, &format!("{fp}.projection.bias"), &[fh]);
    }

    // --- head ---
    let hh = cfg.head_hidden_size;
    let half = fh / 2;
    ones(&mut w, "head.conv1.weight", &[half, fh, 3, 3]);
    ones(&mut w, "head.conv1.bias", &[half]);
    ones(&mut w, "head.conv2.weight", &[hh, half, 3, 3]);
    ones(&mut w, "head.conv2.bias", &[hh]);
    ones(&mut w, "head.conv3.weight", &[1, hh, 1, 1]);
    ones(&mut w, "head.conv3.bias", &[1]);
    Weights::from_map(w)
}

#[test]
fn backbone_captures_four_states_of_expected_shape() {
    let cfg = tiny_cfg();
    let w = synth_weights(&cfg);
    let backbone =
        candle_gen_depth::backbone::Dinov2Backbone::from_weights(&w, "backbone", cfg.clone())
            .unwrap();
    let grid = cfg.grid();
    let input = Tensor::from_vec(
        vec![0.02f32; cfg.image_size * cfg.image_size * 3],
        (1, cfg.image_size, cfg.image_size, 3),
        &dev(),
    )
    .unwrap();
    let states = backbone.forward(&input).unwrap();
    assert_eq!(states.len(), 4, "out_indices [1,2,3,4] → 4 captured states");
    for s in &states {
        assert_eq!(
            s.dims(),
            &[1, grid * grid + 1, cfg.hidden_size],
            "each captured hidden is [B, grid²+1, hidden] (CLS included)"
        );
    }
}

#[test]
fn full_forward_produces_input_resolution_depth() {
    let cfg = tiny_cfg();
    let w = synth_weights(&cfg);
    let model = DepthAnythingV2::from_weights(&w, cfg.clone(), &dev()).unwrap();
    let input = Tensor::from_vec(
        vec![0.02f32; cfg.image_size * cfg.image_size * 3],
        (1, cfg.image_size, cfg.image_size, 3),
        &dev(),
    )
    .unwrap();
    let depth = model.forward(&input).unwrap();
    // Head upsamples to grid · patch_size = image_size.
    assert_eq!(
        depth.dims(),
        &[cfg.image_size, cfg.image_size],
        "depth map is [H, W] at the input resolution"
    );
    let vals: Vec<f32> = depth.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all depth values finite"
    );
    assert!(
        vals.iter().all(|&v| v >= 0.0),
        "relative-depth head is ReLU → non-negative"
    );
}

#[test]
fn estimate_control_returns_input_dims_grayscale() {
    let cfg = tiny_cfg();
    let w = synth_weights(&cfg);
    let model = DepthAnythingV2::from_weights(&w, cfg, &dev()).unwrap();
    // Arbitrary 6×10 RGB image → a 6×10 control map.
    let (wd, ht) = (6u32, 10u32);
    let rgb: Vec<u8> = (0..(wd * ht * 3)).map(|i| (i % 256) as u8).collect();
    let out = model.estimate_control_rgb8(&rgb, wd, ht).unwrap();
    assert_eq!(
        out.len(),
        (wd * ht * 3) as usize,
        "control image matches input dims"
    );
    assert!(
        out.chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]),
        "depth control image is grayscale broadcast to RGB"
    );
}

#[test]
fn rejects_wrong_buffer_size() {
    let cfg = tiny_cfg();
    let w = synth_weights(&cfg);
    let model = DepthAnythingV2::from_weights(&w, cfg, &dev()).unwrap();
    let err = model.estimate_control_rgb8(&[0u8; 10], 4, 4).unwrap_err();
    assert!(
        err.to_string().contains("expected"),
        "size-mismatch error names the expected length: {err}"
    );
}

/// Real-weight CUDA/CPU smoke (the maintainer's on-device gate). Set `DEPTH_ANYTHING_V2_DIR` to a
/// local `depth-anything/Depth-Anything-V2-Small-hf` snapshot dir (containing `model.safetensors`).
///
/// ```bash
/// DEPTH_ANYTHING_V2_DIR=$HOME/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/<rev> \
///   cargo test -p candle-gen-depth --test depth_shapes -- --ignored real_weight_smoke
/// ```
#[test]
#[ignore = "needs DEPTH_ANYTHING_V2_DIR (a Depth-Anything-V2-Small-hf snapshot)"]
fn real_weight_smoke() {
    let dir = std::env::var("DEPTH_ANYTHING_V2_DIR")
        .expect("set DEPTH_ANYTHING_V2_DIR to a Depth-Anything-V2-Small-hf snapshot dir");
    let model = DepthAnythingV2::from_dir(&dir).expect("load DA-V2 Small");

    // A synthetic but non-degenerate image: a bright disc on a dark field (a clear near/far cue).
    let (wd, ht) = (256u32, 192u32);
    let mut rgb = vec![20u8; (wd * ht * 3) as usize];
    let (cx, cy, r) = (wd as i32 / 2, ht as i32 / 2, 60i32);
    for y in 0..ht as i32 {
        for x in 0..wd as i32 {
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r {
                let idx = ((y * wd as i32 + x) * 3) as usize;
                rgb[idx] = 230;
                rgb[idx + 1] = 230;
                rgb[idx + 2] = 230;
            }
        }
    }

    let control = model
        .estimate_control_rgb8(&rgb, wd, ht)
        .expect("estimate depth");
    assert_eq!(control.len(), (wd * ht * 3) as usize);

    // Plausibility: the normalized control map must span a real range (non-degenerate) and be
    // grayscale; a flat map would mean the forward collapsed.
    let lumas: Vec<u8> = control.iter().step_by(3).copied().collect();
    let lo = *lumas.iter().min().unwrap();
    let hi = *lumas.iter().max().unwrap();
    assert!(
        hi as i32 - lo as i32 > 20,
        "depth control must be non-degenerate (lo={lo}, hi={hi})"
    );
    assert!(
        control.chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]),
        "depth control image is grayscale"
    );

    // The full raw forward must also be finite over a real resize-to-518 input.
    let input = model.preprocess_rgb8(&rgb, wd, ht).unwrap();
    let depth = model.forward(&input).unwrap();
    let vals: Vec<f32> = depth.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "raw depth must be finite"
    );
    assert_eq!(depth.dims(), &[518, 518], "native depth resolution is 518²");

    let _ = preprocess::INPUT_SIZE; // keep the preprocess import meaningful on all cfgs
}
