//! Ideogram 4's single-stream flow-matching DiT (`Ideogram4Transformer2DModel`): 34 layers over one
//! concatenated `[text ; image]` token sequence, AdaLN-modulated per block by the flow-matching
//! timestep, with interleaved 3D MRoPE and full (bidirectional, segment-masked) attention. Port of
//! `mlx-gen-ideogram`'s `transformer/` (upstream `modeling_ideogram4.py`). Instantiated twice
//! (conditional + unconditional) for the quality variant's asymmetric CFG.

pub mod block;
pub mod model;
pub mod mrope;

pub use block::Ideogram4Block;
pub use model::Ideogram4Transformer;
pub use mrope::Ideogram4MRoPE;

use candle_gen::candle_core::{Result, Tensor};

/// RMSNorm over the last dim with weight `w` (candle's fused op; eps as f32).
pub(crate) fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    candle_gen::candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)
}
