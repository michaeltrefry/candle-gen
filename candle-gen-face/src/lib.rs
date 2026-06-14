//! # candle-gen-face
//!
//! Native **candle** face-analysis stack — the Windows/CUDA sibling of
//! [`mlx-gen-face`](https://github.com/michaeltrefry/mlx-gen). A SCRFD-10g 5-point detector
//! ([`scrfd`]) + an ArcFace `glintr100` (iresnet100) 512-d embedder ([`iresnet`]) + the
//! insightface-faithful 5-point alignment ([`align`]), orchestrated by [`face::FaceAnalysis`] and
//! exposed through the backend-neutral [`gen_core::FaceEmbedder`] contract (epic 5480, sc-5490).
//!
//! Consumers — the candle InstantID (sc-5491) and PuLID-FLUX (sc-5492) identity providers, and the
//! Phase-5 keypoint-extract surface (epic 5482) — depend on the `gen_core` trait, not this crate's
//! concrete types, so the worker stays backend-neutral. macOS keeps the MLX implementation.
//!
//! Unlike the Generator providers this crate is **not** an `inventory`-registered model: a face
//! embedder is a composed utility, so consumers construct it directly via [`load`] / [`load_on`].

pub mod align;
mod common;
pub mod face;
pub mod iresnet;
#[cfg(test)]
mod parity;
pub mod scrfd;

use std::path::Path;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{DetectedFace, FaceEmbedder, FaceEmbedderDescriptor, Image};
use candle_gen::{CandleError, Result};

use crate::common::Weights;
use crate::face::{Face, FaceAnalysis};
use crate::iresnet::ArcFace;
use crate::scrfd::Scrfd;

/// Stable id for the antelopev2 SCRFD + ArcFace pair.
const MODEL_ID: &str = "antelopev2";
/// ArcFace `glintr100` embedding dimensionality.
const EMBEDDING_DIM: usize = 512;
/// Detector checkpoint filename (shared with the MLX `convert_scrfd.py` output).
const SCRFD_FILE: &str = "scrfd_10g.safetensors";
/// Recognizer checkpoint filename (shared with the MLX `convert_glintr100.py` output).
const ARCFACE_FILE: &str = "arcface_iresnet100.safetensors";

/// The candle face embedder: a thin [`gen_core::FaceEmbedder`] adapter over [`FaceAnalysis`].
pub struct CandleFaceAnalysis {
    inner: FaceAnalysis,
    descriptor: FaceEmbedderDescriptor,
}

impl CandleFaceAnalysis {
    fn descriptor() -> FaceEmbedderDescriptor {
        FaceEmbedderDescriptor {
            id: MODEL_ID,
            family: "face",
            backend: "candle",
            embedding_dim: EMBEDDING_DIM,
            mac_only: false,
        }
    }

    /// Access the underlying [`FaceAnalysis`] for the concrete-typed helpers the identity providers
    /// need beyond the neutral contract (e.g. embed-on-demand of a chosen detection).
    pub fn inner(&self) -> &FaceAnalysis {
        &self.inner
    }
}

/// Load the SCRFD + ArcFace pair from a directory holding `scrfd_10g.safetensors` +
/// `arcface_iresnet100.safetensors`, onto `device`. (The face stack runs f32 regardless of the
/// build's default dtype.)
pub fn load_on(dir: &Path, device: &Device) -> Result<CandleFaceAnalysis> {
    let scrfd_w = Weights::from_file(&dir.join(SCRFD_FILE), device)?;
    let arcface_w = Weights::from_file(&dir.join(ARCFACE_FILE), device)?;
    let scrfd = Scrfd::from_weights(&scrfd_w)?;
    let arcface = ArcFace::from_weights(&arcface_w)?;
    Ok(CandleFaceAnalysis {
        inner: FaceAnalysis::new(scrfd, arcface, device.clone()),
        descriptor: CandleFaceAnalysis::descriptor(),
    })
}

/// Load the face stack onto the build's default compute device (CUDA on Windows, CPU/Metal on Mac).
pub fn load(dir: &Path) -> Result<CandleFaceAnalysis> {
    let device = candle_gen::default_device()?;
    load_on(dir, &device)
}

/// `[x1,y1,x2,y2]` image-space dims of an [`Image`], rejecting a zero/oversized buffer.
fn image_dims(image: &Image) -> Result<(usize, usize)> {
    let (w, h) = (image.width as usize, image.height as usize);
    if image.pixels.len() < w * h * 3 {
        return Err(CandleError::Msg(format!(
            "face: image buffer of {} bytes too small for {w}×{h}×3",
            image.pixels.len()
        )));
    }
    Ok((h, w))
}

fn to_detected(face: &Face) -> DetectedFace {
    DetectedFace {
        bbox: face.bbox,
        kps: face.kps,
        det_score: face.det_score,
        embedding: face.embedding.clone(),
    }
}

impl FaceEmbedder for CandleFaceAnalysis {
    fn descriptor(&self) -> &FaceEmbedderDescriptor {
        &self.descriptor
    }

    fn detect(&self, image: &Image) -> candle_gen::gen_core::Result<Vec<DetectedFace>> {
        let (h, w) = image_dims(image)?;
        Ok(self
            .inner
            .detect(&image.pixels, h, w)?
            .iter()
            .map(|d| DetectedFace {
                bbox: d.bbox,
                kps: d.kps,
                det_score: d.score,
                embedding: Vec::new(),
            })
            .collect())
    }

    fn analyze(&self, image: &Image) -> candle_gen::gen_core::Result<Vec<DetectedFace>> {
        let (h, w) = image_dims(image)?;
        Ok(self
            .inner
            .analyze(&image.pixels, h, w)?
            .iter()
            .map(to_detected)
            .collect())
    }

    /// Detect + embed only the largest face (F-090): one detect sweep + a single recognition forward
    /// on the area-largest detection, instead of embedding every face the default would.
    fn largest_face(&self, image: &Image) -> candle_gen::gen_core::Result<DetectedFace> {
        let (h, w) = image_dims(image)?;
        let dets = self.inner.detect(&image.pixels, h, w)?;
        let Some(largest) = dets.first() else {
            return Err(candle_gen::gen_core::Error::Msg(format!(
                "{MODEL_ID}: no face detected"
            )));
        };
        Ok(to_detected(&self.inner.embed(
            &image.pixels,
            h,
            w,
            largest,
        )?))
    }
}
