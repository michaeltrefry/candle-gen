//! Unified `FaceAnalysis` — the one entry point that orchestrates SCRFD + ArcFace, the candle twin
//! of mlx-gen-face's `face.rs` (mirroring insightface `app.get()`). BiSeNet face-parsing (PuLID's
//! `face_features_image`) is intentionally NOT ported here: it is PuLID-only and lands with the PuLID
//! provider (sc-5492), keeping this sc-5490 slice the shared detect/embed core that InstantID and the
//! Phase-5 kps_extract surface both need.
//!
//! Pipeline (zero Python): detector blob (cv2-faithful resize-to-fit 640 + pad + normalize) → SCRFD
//! detect → 5-pt `norm_crop` 112² → glintr100 embedding → `Vec<Face>` sorted largest-first.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::{CandleError, Result};

use crate::align;
use crate::iresnet::ArcFace;
use crate::scrfd::{Detection, Scrfd, DET_SIZE};

/// One detected face — mirrors insightface's `Face` fields the consumers use.
#[derive(Clone, Debug)]
pub struct Face {
    /// `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
    /// 5 landmarks (L-eye, R-eye, nose, L-mouth, R-mouth) in original-image pixels.
    pub kps: [[f32; 2]; 5],
    /// SCRFD detection confidence.
    pub det_score: f32,
    /// Raw 512-d glintr100 recognition embedding (un-normalized; L2-normalize for cosine).
    pub embedding: Vec<f32>,
}

/// Bounding-box area of a detection (for largest-first ordering).
fn det_area(d: &Detection) -> f32 {
    (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
}

/// cv2 `resize` `INTER_LINEAR` for an RGB `u8` HWC image — the SCRFD detector preprocessing. Faithful
/// fixed-point bilinear (half-pixel coords, 11-bit weights, two integer passes, `>>22` with rounding),
/// identical to the MLX sibling.
pub fn resize_bilinear_cv2(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<u8> {
    const C: usize = 3;
    assert!(
        src.len() >= in_h * in_w * C,
        "resize_bilinear_cv2: src buffer of {} bytes too small for {in_h}×{in_w}×3",
        src.len()
    );
    const BITS: i64 = 11;
    const SCALE: f64 = (1i64 << BITS) as f64; // 2048

    let coeffs = |in_n: usize, out_n: usize| {
        let scale = in_n as f64 / out_n as f64;
        let mut ofs = Vec::with_capacity(out_n);
        let mut a = Vec::with_capacity(out_n);
        for d in 0..out_n {
            let f = (d as f64 + 0.5) * scale - 0.5;
            let mut s = f.floor() as i64;
            let mut fr = f - s as f64;
            if s < 0 {
                s = 0;
                fr = 0.0;
            }
            if s >= in_n as i64 - 1 {
                s = in_n as i64 - 1;
                fr = 0.0;
            }
            let s1 = (s + 1).min(in_n as i64 - 1);
            let w1 = (fr * SCALE).round_ties_even() as i64;
            let w0 = ((1.0 - fr) * SCALE).round_ties_even() as i64;
            ofs.push((s as usize, s1 as usize));
            a.push((w0, w1));
        }
        (ofs, a)
    };

    let (xofs, xa) = coeffs(in_w, out_w);
    let (yofs, ya) = coeffs(in_h, out_h);

    // The vertical pass only reads the source rows named in `yofs`, so resample just those rows.
    let mut needed: Vec<usize> = yofs.iter().flat_map(|&(s0, s1)| [s0, s1]).collect();
    needed.sort_unstable();
    needed.dedup();
    let mut row_of = vec![usize::MAX; in_h];
    for (hi, &sy) in needed.iter().enumerate() {
        row_of[sy] = hi;
    }

    // Horizontal pass over the needed source rows → int (value·2048).
    let mut hbuf = vec![0i64; needed.len() * out_w * C];
    for (hi, &sy) in needed.iter().enumerate() {
        for (dx, (&(sx, sx1), &(w0, w1))) in xofs.iter().zip(&xa).enumerate() {
            for ch in 0..C {
                hbuf[(hi * out_w + dx) * C + ch] = src[(sy * in_w + sx) * C + ch] as i64 * w0
                    + src[(sy * in_w + sx1) * C + ch] as i64 * w1;
            }
        }
    }

    // Vertical pass → uint8, (acc + 2^21) >> 22.
    let mut out = vec![0u8; out_h * out_w * C];
    for (dy, (&(sy0, sy1), &(v0, v1))) in yofs.iter().zip(&ya).enumerate() {
        let (r0, r1) = (row_of[sy0], row_of[sy1]);
        for dx in 0..out_w {
            for ch in 0..C {
                let acc =
                    hbuf[(r0 * out_w + dx) * C + ch] * v0 + hbuf[(r1 * out_w + dx) * C + ch] * v1;
                out[(dy * out_w + dx) * C + ch] = (((acc + (1 << 21)) >> 22).clamp(0, 255)) as u8;
            }
        }
    }
    out
}

/// Build the SCRFD detector blob from an RGB `u8` image: insightface-faithful resize-to-fit 640
/// (aspect-preserving) → top-left pad to 640² → `(rgb − 127.5) / 128`. Returns the **NCHW**
/// `[1,3,640,640]` f32 blob (MLX returns NHWC) and `det_scale` (= `new_h / h`).
pub fn detector_blob(img: &[u8], h: usize, w: usize, device: &Device) -> Result<(Tensor, f32)> {
    assert!(
        img.len() >= h * w * 3,
        "detector_blob: img buffer of {} bytes too small for {h}×{w}×3",
        img.len()
    );
    let det = DET_SIZE;
    let im_ratio = h as f64 / w as f64;
    let (new_w, new_h) = if im_ratio > 1.0 {
        ((det as f64 / im_ratio) as usize, det)
    } else {
        (det, (det as f64 * im_ratio) as usize)
    };
    let det_scale = new_h as f32 / h as f32;
    let resized = resize_bilinear_cv2(img, h, w, new_h, new_w);

    // top-left into a 640² canvas; normalize (rgb-127.5)/128.
    let norm = |v: u8| (v as f32 - 127.5) / 128.0;
    let mut blob = vec![norm(0); det * det * 3]; // padded region = normalized 0
    for y in 0..new_h {
        for x in 0..new_w {
            for ch in 0..3 {
                blob[(y * det + x) * 3 + ch] = norm(resized[(y * new_w + x) * 3 + ch]);
            }
        }
    }
    let nhwc = Tensor::from_vec(blob, (1, det, det, 3), device)?;
    Ok((nhwc.permute((0, 3, 1, 2))?.contiguous()?, det_scale))
}

/// The native face-analysis stack: SCRFD + ArcFace.
pub struct FaceAnalysis {
    scrfd: Scrfd,
    arcface: ArcFace,
    device: Device,
    /// Detection score / NMS thresholds (insightface defaults: 0.5 / 0.4).
    pub det_thresh: f32,
    pub nms_thresh: f32,
}

impl FaceAnalysis {
    /// Build the detection + recognition stack from already-loaded sub-models on `device`.
    pub fn new(scrfd: Scrfd, arcface: ArcFace, device: Device) -> Self {
        Self {
            scrfd,
            arcface,
            device,
            det_thresh: 0.5,
            nms_thresh: 0.4,
        }
    }

    /// Detect every face in an RGB `u8` image, sorted **largest-first** (insightface `app.get()`
    /// order). No ArcFace forward is run — consumers that need only the box/landmarks use this.
    pub fn detect(&self, img: &[u8], h: usize, w: usize) -> Result<Vec<Detection>> {
        // A zero dimension makes `detector_blob` compute `det_scale = new_h / 0 = NaN`; reject first.
        if h == 0 || w == 0 {
            return Err(CandleError::Msg(format!(
                "face detect: image has a zero dimension ({h}×{w})"
            )));
        }
        if img.len() < h * w * 3 {
            return Err(CandleError::Msg(format!(
                "face detect: img buffer of {} bytes too small for {h}×{w}×3",
                img.len()
            )));
        }
        let (blob, det_scale) = detector_blob(img, h, w, &self.device)?;
        let mut dets = self
            .scrfd
            .detect(&blob, det_scale, self.det_thresh, self.nms_thresh)?;
        dets.sort_by(|a, b| det_area(b).total_cmp(&det_area(a)));
        Ok(dets)
    }

    /// Align + ArcFace-embed a single [`detect`](Self::detect) result into a [`Face`] — one
    /// `[1,3,112,112]` recognition forward (embed-on-demand for the largest face).
    pub fn embed(&self, img: &[u8], h: usize, w: usize, det: &Detection) -> Result<Face> {
        let crop = align::norm_crop(img, h, w, &det.kps);
        let emb = self
            .arcface
            .forward(&align::to_arcface_input(&[crop], &self.device)?)?;
        Ok(Face {
            bbox: det.bbox,
            kps: det.kps,
            det_score: det.score,
            embedding: emb.flatten_all()?.to_vec1::<f32>()?,
        })
    }

    /// Detect → align → embed every face, sorted **largest-first**. Runs ONE batched
    /// `[N,3,112,112]` ArcFace forward (iresnet100 has no cross-batch ops, so each row is identical
    /// to the per-face forward).
    pub fn analyze(&self, img: &[u8], h: usize, w: usize) -> Result<Vec<Face>> {
        let dets = self.detect(img, h, w)?;
        if dets.is_empty() {
            return Ok(Vec::new());
        }
        let crops: Vec<Vec<u8>> = dets
            .iter()
            .map(|d| align::norm_crop(img, h, w, &d.kps))
            .collect();
        let emb = self
            .arcface
            .forward(&align::to_arcface_input(&crops, &self.device)?)?;
        let flat = emb.flatten_all()?.to_vec1::<f32>()?;
        let dim = flat.len() / dets.len(); // [N, 512] → 512 per row
        Ok(dets
            .iter()
            .enumerate()
            .map(|(i, d)| Face {
                bbox: d.bbox,
                kps: d.kps,
                det_score: d.score,
                embedding: flat[i * dim..(i + 1) * dim].to_vec(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "too small for 4×4×3")]
    fn resize_rejects_undersized_buffer() {
        let src = vec![0u8; 4 * 4 * 3 - 1];
        let _ = resize_bilinear_cv2(&src, 4, 4, 8, 8);
    }

    #[test]
    fn resize_same_size_is_identity() {
        let src: Vec<u8> = (0..5 * 3 * 3).map(|i| (i * 37 % 256) as u8).collect();
        let out = resize_bilinear_cv2(&src, 5, 3, 5, 3);
        assert_eq!(out, src);
    }

    #[test]
    fn resize_constant_preserved_on_tall_downscale() {
        let (in_h, w) = (200usize, 4usize);
        let src = vec![123u8; in_h * w * 3];
        let out = resize_bilinear_cv2(&src, in_h, w, 8, w);
        assert_eq!(out.len(), 8 * w * 3);
        assert!(out.iter().all(|&v| v == 123), "constant must be preserved");
    }
}
