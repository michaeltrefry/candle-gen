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
//! * [`Sam3VisionEncoder`] — the shared PE ViT backbone + FPN neck (slice sc-6240).
//! * [`Sam3TextEncoder`] / [`Sam3Tokenizer`] — the CLIP-H text tower + `text_projection`
//!   (1024→256) and the CLIP BPE tokenizer that produce the concept conditioning the DETR stack
//!   consumes (slice sc-6241).
//! * [`Sam3Detector`] — the DETR encoder/decoder + presence + dot-product scoring that turns the
//!   72² FPN feature + text conditioning into concept logits, boxes, and presence (slice sc-6242).
//! * [`Sam3MaskHead`] + [`Sam3ImageSegmenter`] — the MaskFormer-style mask head and the end-to-end
//!   still-image segmenter (`pixel_values + "person" → per-instance masks`) that assembles vision +
//!   text + DETR + mask head (slice sc-6243).
//! * [`Sam3GeometryEncoder`] — the box/point **PVS** prompt encoder (`roi_align` + box sine-PE + 3
//!   cross-attending layers) that feeds `Sam3ImageSegmenter::forward_with_boxes` (slice sc-6244).
//! * [`Sam3Tracker`] — the SAM2.1 single-frame box-prompt tracker (tracker neck + prompt encoder +
//!   two-way mask decoder) plus the video memory primitives (memory encoder, RoPE memory attention,
//!   per-object bank conditioning), and [`Sam3VideoModel`] — the multi-object video PCS pipeline that
//!   orchestrates the detector + tracker frame-by-frame (slice sc-6245).
//!
//! ## Quantization (sc-6246)
//! Each model loads dense and can be affine-quantized in place to Q8 (near-lossless) or Q4 (coherent)
//! with `quantize(Quant)` — built on candle-core's GGUF `QMatMul`, the same seam `candle-gen-lens` /
//! `candle-gen-seedvr2` use. The attention/FFN/projection linears fold to `Q8_0`/`Q4_0`; convs,
//! GroupNorms, embeddings, and the few sub-block-width projections (the BoxRPB `2→256`, the geometry
//! `4→256` / `258→256`) stay dense (`in_features` must divide the 32-wide block). The video model
//! quantizes its one shared PE backbone (F-028) exactly once.
//!
//! ## Layout note
//! The MLX port runs NHWC and permutes the torch OIHW/IOHW conv kernels to MLX OHWI at load. candle's
//! `conv2d`/`conv_transpose2d` are NCHW with torch-native OIHW/IOHW kernels — and SAM3 loads the RAW
//! `facebook/sam3` checkpoint (no pre-conversion), so the kernels are ALREADY candle-native: we load
//! them as-is (no permute) and transpose only the *activations* NHWC↔NCHW around each conv, keeping
//! the transformer body channels-last so it mirrors the MLX module line-by-line.

mod common;
pub mod config;
pub mod detr;
pub mod geometry;
pub mod mask;
pub mod model;
pub mod text;
pub mod tracker;
pub mod video;
pub mod vision;

pub use common::Weights;
pub use config::{Sam3DetrConfig, Sam3GeometryConfig, Sam3TextConfig, Sam3VisionConfig};
pub use detr::{DetectorOutput, Sam3Detector};
pub use geometry::Sam3GeometryEncoder;
pub use mask::{post_process_instances, Instance, MaskOutput, Sam3MaskHead};
pub use model::{Sam3ImageSegmenter, SegmentationOutput};
pub use text::{Sam3TextEncoder, Sam3Tokenizer};
pub use tracker::{MemoryFeatures, Sam3Tracker, TrackerFrameOutput, TrackerMask};
pub use video::{Sam3VideoModel, VideoFrameOutput};
pub use vision::Sam3VisionEncoder;
