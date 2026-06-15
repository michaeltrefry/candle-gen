//! # candle-gen-instantid
//!
//! InstantID (epic 5480, sc-5491) — identity-preserving SDXL, the candle (Windows/CUDA) sibling of
//! [`mlx-gen-instantid`](https://github.com/michaeltrefry/mlx-gen). macOS keeps the MLX crate.
//!
//! **Unlike the MLX crate, this is not pure glue.** In mlx-gen the InstantID provider merely composes
//! mlx-gen-sdxl's IP-Adapter `Resampler` + decoupled cross-attention + `ControlNet` + the
//! `denoise_ip_control` loops. candle has none of that machinery yet (candle-gen-sdxl inference is
//! txt2img-only on the stock candle-transformers UNet), so the InstantID port builds those reusable
//! SDXL building blocks into [`candle-gen-sdxl`](candle_gen_sdxl) — front-loading the SDXL slices of
//! sc-5488 (IP-Adapter) / sc-5489 (ControlNet) — and this crate stays the InstantID-specific glue.
//!
//! What lives here:
//! - [`kps`] — the 5-point facial-landmark control-image renderer (a bit-exact OpenCV-4.13 port of the
//!   vendored `draw_kps`) + [`letterbox`](kps::letterbox) (the sc-2009 aspect rule) + the canonical
//!   multi-view [`VIEW_ANGLE_KPS`](kps::VIEW_ANGLE_KPS) landmark sets.
//! - [`openpose`] — the COCO-18 body-skeleton control-image renderer for pose mode (a bit-exact port of
//!   the worker's `draw_bodypose`), sharing the OpenCV primitives in [`kps`].
//! - [`restore`] — the face-restoration compositing primitives (feathered elliptical mask + alpha
//!   paste-back) for the ADetailer-style identity-recovery pass.
//!
//! The InstantID model itself — [`InstantId::load`] + the `generate*` / [`InstantId::restore_face`]
//! entry points the worker drives — lives in [`model`], composing the candle-gen-sdxl IP-Adapter /
//! ControlNet / sampler / denoise stack and the candle-gen-face embedder.

mod resample;

pub mod kps;
pub mod model;
pub mod openpose;
pub mod restore;

/// Phase 5 real-weight GPU validation (sc-5491) — env-driven, `#[ignore]`d integration test.
#[cfg(test)]
mod validate;

pub use kps::{draw_kps, letterbox, view_angle_kps, ANGLE_SET_ORDER, VIEW_ANGLE_KPS};
pub use model::{
    InstantId, InstantIdPaths, InstantIdRequest, DEFAULT_CONTROLNET_SCALE, DEFAULT_IP_SCALE,
    DEFAULT_OPENPOSE_SCALE, FACE_RESTORE_PROMPT,
};
pub use openpose::{
    draw_bodypose, face_box_from_keypoints, normalize_keypoints, square_fit, BodyPoint,
    NUM_BODY_KEYPOINTS, STICKWIDTH,
};
pub use restore::{feather_mask, paste_alpha};
