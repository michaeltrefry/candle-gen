//! `candle-gen-sam3` — native-candle SAM3 (Segment Anything 3) concept segmenter for candle-gen,
//! the Windows/CUDA sibling of [`mlx-gen-sam3`](https://github.com/michaeltrefry/mlx-gen) (epic
//! 5482, sc-5062). It ports the model directly from the public Apache-2.0 `transformers` reference
//! — the same source `mlx-gen-sam3` ported against, so that crate (parity-tested on MLX) is the
//! reimplementation oracle.
//!
//! SAM3 adds open-vocabulary **Promptable Concept Segmentation** (PCS): segment *all* instances of a
//! text concept ("person") with no geometric prompt, plus the **PVS** box/point prompt path and a
//! memory-based video tracker. This is what replaces the off-Mac SAM2 box-prompt in the SceneWorks
//! person-track (sc-5062), bringing the Windows/Candle lane to mask-quality parity with the Mac MLX
//! lane (sc-4926).
//!
//! ## Public API (a plain utility segmenter — not a generation-registry provider)
//! Mirrors `mlx-gen-sam3`'s surface. Loaded incrementally as the slices land:
//! * [`Sam3VisionEncoder`] — the shared PE ViT backbone + FPN neck (slice sc-6240; **this slice**).
//! * `Sam3TextEncoder` / `Sam3Detector` / `Sam3MaskHead` / `Sam3ImageSegmenter` / `Sam3VideoModel` —
//!   later slices (sc-6241…sc-6246).
//!
//! ## Layout note
//! The MLX port runs NHWC and permutes the torch OIHW/IOHW conv kernels to MLX OHWI at load. candle's
//! `conv2d`/`conv_transpose2d` are NCHW with torch-native OIHW/IOHW kernels — and SAM3 loads the RAW
//! `facebook/sam3` checkpoint (no pre-conversion), so the kernels are ALREADY candle-native: we load
//! them as-is (no permute) and transpose only the *activations* NHWC↔NCHW around each conv, keeping
//! the transformer body channels-last so it mirrors the MLX module line-by-line.

mod common;
pub mod config;
pub mod vision;

pub use common::Weights;
pub use config::{Sam3DetrConfig, Sam3GeometryConfig, Sam3TextConfig, Sam3VisionConfig};
pub use vision::Sam3VisionEncoder;
