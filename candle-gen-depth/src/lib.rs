//! `candle-gen-depth` — native-candle **Depth Anything V2** monocular depth estimator for candle-gen
//! (epic 8236, sc-8413). The Windows/CUDA sibling of `mlx-gen-depth`.
//!
//! A plain utility *preprocessor* (like `candle-gen-face` / `candle-gen-sam3`, NOT a
//! generation-registry provider): an arbitrary RGB image → a normalized single-channel depth-control
//! image, with **no Python / torch**. It is the auto depth source for the Fun-Controlnet-Union depth
//! tier — the off-Mac sibling of the host-side canny / pose preprocessors — but, unlike those pure-
//! raster ones, depth needs real neural inference, so it runs on candle (CUDA / CPU). The worker
//! `depth.rs` non-macOS wiring is a separate later step (sc-8304 / sc-8246).
//!
//! ## Architecture (port of the HF `transformers` `DepthAnythingForDepthEstimation`)
//! * [`backbone::Dinov2Backbone`] — DINOv2 ViT-S/14 encoder; returns the four `out_indices`
//!   ([3,6,9,12], captured at layer-output [2,5,8,11]) hidden states.
//! * [`neck::DptNeck`] — DPT reassemble (per-level 1×1 projection + factor resize) + 3×3 projection
//!   (`convs`) + RefineNet feature-fusion stage.
//! * [`head::DepthHead`] — `conv1` → bilinear upsample → `conv2`+ReLU → `conv3`+ReLU → `[B,H,W]`.
//!
//! ## Variant / weights
//! Default is **Small** (ViT-S/14): `depth-anything/Depth-Anything-V2-Small-hf` (apache-2.0,
//! **ungated**, ships standard `model.safetensors`). The Base/Large `-hf` checkpoints share the
//! module graph and plug in via [`config::DepthAnythingConfig`].
//!
//! ## Public API (mirrors `mlx-gen-depth::DepthAnythingV2`)
//! [`DepthAnythingV2::from_dir`] / [`DepthAnythingV2::from_weights`] load the model;
//! [`DepthAnythingV2::estimate_control_rgb8`] takes an arbitrary RGB8 image and returns a
//! min/max-normalized grayscale-broadcast RGB depth-control image (same `width`·`height`).

pub mod backbone;
pub mod common;
pub mod config;
pub mod head;
pub mod neck;
pub mod preprocess;

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::{default_device, CandleError, Result};

pub use common::Weights;
pub use config::DepthAnythingConfig;

/// The loaded Depth Anything V2 estimator (backbone + neck + head).
pub struct DepthAnythingV2 {
    backbone: backbone::Dinov2Backbone,
    neck: neck::DptNeck,
    head: head::DepthHead,
    cfg: DepthAnythingConfig,
    device: Device,
}

impl DepthAnythingV2 {
    /// Load from a directory holding the transformers checkpoint (`model.safetensors` + `config.json`)
    /// at the **Small** default config, on the default candle device (CUDA if built+available, else
    /// CPU).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let device = default_device()?;
        let w = Weights::from_dir(dir, &device)?;
        Self::from_weights(&w, DepthAnythingConfig::small(), &device)
    }

    /// Load from a directory onto an explicit device.
    pub fn from_dir_on(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let w = Weights::from_dir(dir, device)?;
        Self::from_weights(&w, DepthAnythingConfig::small(), device)
    }

    /// Load from already-read [`Weights`] with an explicit config + device (for Base/Large or testing).
    pub fn from_weights(w: &Weights, cfg: DepthAnythingConfig, device: &Device) -> Result<Self> {
        let backbone = backbone::Dinov2Backbone::from_weights(w, "backbone", cfg.clone())?;
        let neck = neck::DptNeck::from_weights(w, "neck", &cfg)?;
        let head = head::DepthHead::from_weights(w, "head", &cfg)?;
        Ok(Self {
            backbone,
            neck,
            head,
            cfg,
            device: device.clone(),
        })
    }

    /// The loaded configuration.
    pub fn config(&self) -> &DepthAnythingConfig {
        &self.cfg
    }

    /// The device the model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Build the model's NHWC input tensor `[1, size, size, 3]` from an arbitrary RGB8 HWC image,
    /// resized + ImageNet-normalized. Exposed for parity/testing.
    pub fn preprocess_rgb8(&self, rgb: &[u8], width: u32, height: u32) -> Result<Tensor> {
        let size = self.cfg.image_size;
        let buf = preprocess::rgb8_to_input_buf(rgb, width, height, size);
        Ok(Tensor::from_vec(buf, (1, size, size, 3), &self.device)?)
    }

    /// Run the model on a normalized NHWC input `[1, image_size, image_size, 3]` → a depth map
    /// `[H, W]` (f32, model units; relative depth). Exposed for parity/testing; most callers want
    /// [`estimate_control_rgb8`](Self::estimate_control_rgb8).
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let grid = self.cfg.grid();
        let hidden = self.backbone.forward(pixel_values)?;
        if hidden.len() != 4 {
            return Err(CandleError::Msg(format!(
                "depth backbone produced {} captured states (expected 4)",
                hidden.len()
            )));
        }
        let fused = self.neck.forward(&hidden, grid, self.cfg.hidden_size)?;
        let depth = self.head.forward(&fused, grid)?; // [1, H, W]
        let (_, h, wd) = depth.dims3()?;
        Ok(depth.reshape((h, wd))?)
    }

    /// Arbitrary RGB8 HWC image (`width`·`height`·3 bytes) → a depth-control RGB8 image of the SAME
    /// `width`·`height` (min/max-normalized, grayscale broadcast; near = bright). The model runs at
    /// its native 518² and the result is bilinearly resized back to the input dimensions on the host.
    pub fn estimate_control_rgb8(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let expected = width as usize * height as usize * 3;
        if rgb.len() != expected {
            return Err(CandleError::Msg(format!(
                "depth input buffer is {} bytes, expected {expected} ({width}×{height}×3)",
                rgb.len()
            )));
        }
        let input = self.preprocess_rgb8(rgb, width, height)?;
        let depth = self.forward(&input)?; // [image_size, image_size]
        let (dh, dw) = (depth.dim(0)?, depth.dim(1)?);
        let depth_vals: Vec<f32> = depth.flatten_all()?.to_vec1::<f32>()?;
        // Normalize at native resolution, then resize the control image back to input dims.
        let native = preprocess::depth_to_control_rgb8(&depth_vals, dh, dw);
        Ok(preprocess::resize_control_rgb8(
            &native,
            dh,
            dw,
            height as usize,
            width as usize,
        ))
    }
}
