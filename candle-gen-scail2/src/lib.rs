//! # candle-gen-scail2
//!
//! zai-org **SCAIL-2** — the candle (Windows/CUDA + Linux/NVIDIA) sibling of `mlx-gen-scail2` (epic
//! 6563, the CUDA port of the MLX product epic 5439).
//!
//! SCAIL-2 is an end-to-end controlled **character-animation / motion-transfer** model: a reference
//! image + driving video (+ color-coded segmentation masks) → an animated or identity-replaced video.
//! The backbone is **Wan2.1-14B I2V** (dense), so it reuses the [`candle_gen_wan`] foundation (z16 VAE,
//! UMT5, the flow/UniPC scheduler, the base 3-axis RoPE apply) with three SCAIL-2-specific deltas:
//!
//!   1. **packed-token conditioning** — reference + driving (pose) + 28-channel color-coded masks are
//!      patch-embedded (three Conv3d stems; the mask/pose embeds are *added* to the latent embeds) and
//!      concatenated with the noisy target on the token axis (Bernini-family packed conditioning, not
//!      VACE). Only the target tokens are kept from the prediction.
//!   2. **per-source RoPE shifts** ([`rope::ScailRope`]) — the base 3-axis Wan RoPE with integer
//!      (T,H,W) position shifts per chunk; `replace_flag` flips the reference H-shift (animation vs.
//!      cross-identity replacement), and the pose chunk is spatially frequency-downsampled.
//!   3. **CLIP image cross-attention** — the reference image is encoded by an open-CLIP XLM-RoBERTa
//!      ViT-H/14 visual tower ([`clip::ScailClip`]) and injected via Wan-I2V image cross-attention
//!      (`k_img`/`v_img`).
//!
//! Plain single-scale CFG; f32 DiT compute (bf16 overflows to NaN at high token length); temporal-tiled
//! VAE decode for high-res clips. `backend = "candle"`, `mac_only = false`.
//!
//! ## Status (sc-6836, in progress)
//! Landed: the per-chunk [`rope::ScailRope`], the open-CLIP [`clip::ScailClip`] image encoder, the
//! 28-channel [`preprocess::extract_and_compress_mask_to_latent`] mask build, the PyTorch-faithful
//! [`resize`] kernels, and the [`config::Scail2Config`] dims — each CPU-unit-tested. The [`model`]
//! DiT forward, the [`generate`] denoise pipeline, and the provider registration are the remaining
//! slices of sc-6836.

mod common;

pub mod clip;
pub mod config;
pub mod model;
pub mod preprocess;
pub mod resize;
pub mod rope;

pub use clip::{ClipVisionConfig, ScailClip};
pub use config::Scail2Config;
pub use model::{Scail2Dit, Scail2Inputs};
pub use preprocess::extract_and_compress_mask_to_latent;
pub use resize::{clip_preprocess, downsample_half, interpolate, Interp};
pub use rope::ScailRope;
