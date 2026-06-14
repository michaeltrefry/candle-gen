//! SCRFD-10g face detector — the candle port of antelopev2 `scrfd_10g_bnkps`, sibling of
//! mlx-gen-face's `scrfd.rs`. Produces, per face, a bounding box + 5-point landmarks (the kps that
//! drive the ArcFace `norm_crop` alignment and InstantID's `draw_kps`). ResNet-style backbone +
//! PAFPN neck + per-stride heads; the decode (anchor centres + `distance2bbox` / `distance2kps` +
//! NMS) is plain host math, identical to MLX.
//!
//! **Layout note:** MLX runs NHWC; candle's `conv2d` is NCHW (weights transposed OHWI→OIHW at load,
//! see [`crate::common`]). The two spots where layout is load-bearing are handled explicitly: the
//! 2×2 pool reshapes the NCHW size-2 axes (3,5), and each head output is permuted back to NHWC
//! before the `[-1, K]` reshape so the `(h, w, anchor)` row order matches the onnx graph (and the
//! anchor-centre decode below).

use candle_gen::candle_core::Tensor;
use candle_gen::Result;
use candle_nn::ops::sigmoid;

use crate::common::{Conv, Weights};

/// Backbone residual-block counts per stage.
const STAGE_BLOCKS: [usize; 4] = [3, 4, 2, 3];
/// Fixed detector input side.
pub const DET_SIZE: usize = 640;
const NUM_ANCHORS: usize = 2;

/// A single detected face (640-space coords unless rescaled): box, 5 landmarks, score.
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: [f32; 4], // x1, y1, x2, y2
    pub kps: [[f32; 2]; 5],
    pub score: f32,
}

/// 2×2 stride-2 pooling over NCHW via reshape + reduce over the two size-2 axes (exact for even dims).
fn pool2x2(x: &Tensor, avg: bool) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    debug_assert!(
        h % 2 == 0 && w % 2 == 0,
        "pool2x2: expected even H/W, got {h}x{w}"
    );
    let r = x.contiguous()?.reshape((n, c, h / 2, 2, w / 2, 2))?;
    // Reduce the higher size-2 axis (5) first so the lower one (3) keeps its index.
    Ok(if avg {
        r.mean(5)?.mean(3)?
    } else {
        r.max(5)?.max(3)?
    })
}

/// Upsample an NCHW map by ×2 (nearest), matching the neck's top-down path.
fn upsample2x(x: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    Ok(x.upsample_nearest2d(h * 2, w * 2)?)
}

/// `[N,C,H,W]` → `[-1, k]` in `(h, w, anchor)` row order: permute NCHW→NHWC so the channel axis is
/// last (it carries the `num_anchors · k` interleave), then flatten. Matches the onnx transpose+reshape.
fn nhwc_flatten(t: &Tensor, k: usize) -> Result<Tensor> {
    let t = t.permute((0, 2, 3, 1))?.contiguous()?;
    let rows = t.elem_count() / k;
    Ok(t.reshape((rows, k))?)
}

/// `Conv(c1,3×3,stride)+Relu → Conv(c2,3×3,s1) → + identity(/downsample) → Relu`.
struct Block {
    conv1: Conv,
    stride: usize,
    conv2: Conv,
    downsample: Option<Conv>, // AvgPool2×2 already applied to its input
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = self.conv1.forward_relu(x, self.stride, 1)?;
        let t = self.conv2.forward(&t, 1, 1)?;
        let identity = match &self.downsample {
            Some(ds) => ds.forward(&pool2x2(x, true)?, 1, 0)?,
            None => x.clone(),
        };
        Ok(t.broadcast_add(&identity)?.relu()?)
    }
}

struct Head {
    scale: f64,
    stem: [Conv; 3],
    cls: Conv,
    reg: Conv,
    kps: Conv,
}

/// Raw per-stride head outputs: scores `[N,1]` (sigmoid), bbox `[N,4]` (× scale), kps `[N,10]`.
struct StrideOut {
    stride: usize,
    scores: Tensor,
    bbox: Tensor,
    kps: Tensor,
}

impl Head {
    fn load(w: &Weights, stride: usize) -> Result<Self> {
        let p = format!("head{stride}");
        Ok(Self {
            // The learned per-level reg scale is a single-element tensor — stored 0-d or `[1]`
            // depending on the converter, so flatten before reading rather than assume rank 0.
            scale: w
                .require(&format!("{p}.scale"))?
                .flatten_all()?
                .to_vec1::<f32>()?[0] as f64,
            stem: [
                Conv::load(w, &format!("{p}.stem0"))?,
                Conv::load(w, &format!("{p}.stem1"))?,
                Conv::load(w, &format!("{p}.stem2"))?,
            ],
            cls: Conv::load(w, &format!("{p}.cls"))?,
            reg: Conv::load(w, &format!("{p}.reg"))?,
            kps: Conv::load(w, &format!("{p}.kps"))?,
        })
    }

    fn forward(&self, stride: usize, x: &Tensor) -> Result<StrideOut> {
        let mut h = x.clone();
        for c in &self.stem {
            h = c.forward_relu(&h, 1, 1)?;
        }
        let scores = sigmoid(&nhwc_flatten(&self.cls.forward(&h, 1, 1)?, 1)?)?;
        let bbox = nhwc_flatten(&self.reg.forward(&h, 1, 1)?, 4)?.affine(self.scale, 0.0)?;
        let kps = nhwc_flatten(&self.kps.forward(&h, 1, 1)?, 10)?;
        Ok(StrideOut {
            stride,
            scores,
            bbox,
            kps,
        })
    }
}

/// SCRFD-10g detector.
pub struct Scrfd {
    stem: [Conv; 3],
    stages: Vec<Vec<Block>>,
    lateral: [Conv; 3],
    fpn: [Conv; 3],
    down: [Conv; 2],
    pafpn: [Conv; 2],
    heads: [Head; 3],
}

impl Scrfd {
    pub(crate) fn from_weights(w: &Weights) -> Result<Self> {
        let mut stages = Vec::with_capacity(STAGE_BLOCKS.len());
        for (si, &nb) in STAGE_BLOCKS.iter().enumerate() {
            let l = si + 1;
            let mut blocks = Vec::with_capacity(nb);
            for b in 0..nb {
                let p = format!("stage{l}.{b}");
                // stages 2-4 block 0: stride 2 + downsample; everything else stride 1, no downsample.
                let has_ds = b == 0 && l > 1;
                blocks.push(Block {
                    conv1: Conv::load(w, &format!("{p}.conv1"))?,
                    stride: if has_ds { 2 } else { 1 },
                    conv2: Conv::load(w, &format!("{p}.conv2"))?,
                    downsample: if has_ds {
                        Some(Conv::load(w, &format!("{p}.downsample"))?)
                    } else {
                        None
                    },
                });
            }
            stages.push(blocks);
        }
        Ok(Self {
            stem: [
                Conv::load(w, "stem.conv0")?,
                Conv::load(w, "stem.conv1")?,
                Conv::load(w, "stem.conv2")?,
            ],
            stages,
            lateral: [
                Conv::load(w, "neck.lateral0")?,
                Conv::load(w, "neck.lateral1")?,
                Conv::load(w, "neck.lateral2")?,
            ],
            fpn: [
                Conv::load(w, "neck.fpn0")?,
                Conv::load(w, "neck.fpn1")?,
                Conv::load(w, "neck.fpn2")?,
            ],
            down: [Conv::load(w, "neck.down0")?, Conv::load(w, "neck.down1")?],
            pafpn: [Conv::load(w, "neck.pafpn0")?, Conv::load(w, "neck.pafpn1")?],
            heads: [Head::load(w, 8)?, Head::load(w, 16)?, Head::load(w, 32)?],
        })
    }

    /// Backbone → (C2 s8, C3 s16, C4 s32).
    fn backbone(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let mut h = self.stem[0].forward_relu(x, 2, 1)?;
        h = self.stem[1].forward_relu(&h, 1, 1)?;
        h = self.stem[2].forward_relu(&h, 1, 1)?;
        h = pool2x2(&h, false)?; // maxpool
        let mut taps = Vec::new();
        for stage in &self.stages {
            for blk in stage {
                h = blk.forward(&h)?;
            }
            taps.push(h.clone());
        }
        // taps = [stage1, stage2(C2), stage3(C3), stage4(C4)]
        Ok((taps[1].clone(), taps[2].clone(), taps[3].clone()))
    }

    /// Full network → the 3 per-stride raw outputs.
    fn forward(&self, x: &Tensor) -> Result<[StrideOut; 3]> {
        let (c2, c3, c4) = self.backbone(x)?;
        let l0 = self.lateral[0].forward(&c2, 1, 0)?;
        let l1 = self.lateral[1].forward(&c3, 1, 0)?;
        let l2 = self.lateral[2].forward(&c4, 1, 0)?;
        // top-down (×2 nearest upsample + add)
        let p4 = l1.broadcast_add(&upsample2x(&l2)?)?;
        let p3 = l0.broadcast_add(&upsample2x(&p4)?)?;
        let f0 = self.fpn[0].forward(&p3, 1, 1)?;
        let f1 = self.fpn[1].forward(&p4, 1, 1)?;
        let f2 = self.fpn[2].forward(&l2, 1, 1)?;
        // bottom-up (downsample 3×3 s2 + add)
        let n4 = f1.broadcast_add(&self.down[0].forward(&f0, 2, 1)?)?;
        let n5 = f2.broadcast_add(&self.down[1].forward(&n4, 2, 1)?)?;
        let out16 = self.pafpn[0].forward(&n4, 1, 1)?;
        let out32 = self.pafpn[1].forward(&n5, 1, 1)?;
        Ok([
            self.heads[0].forward(8, &f0)?,
            self.heads[1].forward(16, &out16)?,
            self.heads[2].forward(32, &out32)?,
        ])
    }

    /// Test/debug hook: the raw per-stride `(stride, scores[N,1], bbox[N,4], kps[N,10])` outputs
    /// (scores sigmoided, bbox scaled) — the onnx graph outputs, used by the network-parity test.
    pub fn raw_outputs(&self, x: &Tensor) -> Result<Vec<(usize, Tensor, Tensor, Tensor)>> {
        Ok(self
            .forward(x)?
            .into_iter()
            .map(|o| (o.stride, o.scores, o.bbox, o.kps))
            .collect())
    }

    /// Detect faces in a preprocessed `[1,3,640,640]` NCHW f32 image (`(rgb-127.5)/128`).
    ///
    /// `det_scale` maps 640-space coords back to the original image (divide); pass 1.0 to keep
    /// 640-space. Returns NMS-filtered detections sorted by score (descending).
    pub fn detect(
        &self,
        x: &Tensor,
        det_scale: f32,
        score_thresh: f32,
        nms_thresh: f32,
    ) -> Result<Vec<Detection>> {
        let outs = self.forward(x)?;
        let mut dets: Vec<Detection> = Vec::new();
        for out in &outs {
            let s = out.stride;
            let w = DET_SIZE / s;
            let readback =
                |a: &Tensor| -> Result<Vec<f32>> { Ok(a.flatten_all()?.to_vec1::<f32>()?) };
            let scores = readback(&out.scores)?;
            let bbox = readback(&out.bbox)?;
            let kps = readback(&out.kps)?;
            let sf = s as f32;
            for (r, &score) in scores.iter().enumerate() {
                // Drop non-finite AND below-threshold scores (a NaN passes `score < thresh` and would
                // later panic the NMS sort).
                if !score.is_finite() || score < score_thresh {
                    continue;
                }
                // anchor centre: row r → (cell, anchor); cell = r / num_anchors; (h,w) = (cell/W, cell%W)
                let cell = r / NUM_ANCHORS;
                let cx = (cell % w) as f32 * sf;
                let cy = (cell / w) as f32 * sf;
                let d = &bbox[r * 4..r * 4 + 4];
                let bb = [
                    cx - d[0] * sf,
                    cy - d[1] * sf,
                    cx + d[2] * sf,
                    cy + d[3] * sf,
                ];
                let kp = &kps[r * 10..r * 10 + 10];
                let mut pts = [[0.0f32; 2]; 5];
                for (i, p) in pts.iter_mut().enumerate() {
                    *p = [cx + kp[i * 2] * sf, cy + kp[i * 2 + 1] * sf];
                }
                dets.push(Detection {
                    bbox: bb,
                    kps: pts,
                    score,
                });
            }
        }
        let mut kept = nms(dets, nms_thresh);
        if det_scale != 1.0 {
            let inv = 1.0 / det_scale;
            for d in &mut kept {
                for v in &mut d.bbox {
                    *v *= inv;
                }
                for p in &mut d.kps {
                    p[0] *= inv;
                    p[1] *= inv;
                }
            }
        }
        Ok(kept)
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Greedy NMS by descending score (insightface uses IoU threshold 0.4).
fn nms(mut dets: Vec<Detection>, thresh: f32) -> Vec<Detection> {
    // `total_cmp` so a NaN score can never panic the sort (decode already drops non-finite scores;
    // belt-and-suspenders for the production readback path).
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Detection> = Vec::new();
    for d in dets {
        if keep.iter().all(|k| iou(&k.bbox, &d.bbox) <= thresh) {
            keep.push(d);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(score: f32, x: f32) -> Detection {
        Detection {
            bbox: [x, 0.0, x + 10.0, 10.0],
            kps: [[0.0; 2]; 5],
            score,
        }
    }

    /// A NaN score must not panic the NMS sort, and the finite scores still come out descending.
    #[test]
    fn nms_sorts_descending_and_survives_nan() {
        let dets = vec![
            det(0.5, 0.0),
            det(f32::NAN, 100.0),
            det(0.9, 200.0),
            det(0.7, 300.0),
        ];
        let kept = nms(dets, 0.4); // disjoint boxes ⇒ all retained, just reordered
        assert_eq!(kept.len(), 4);
        let finite: Vec<f32> = kept
            .iter()
            .map(|d| d.score)
            .filter(|s| s.is_finite())
            .collect();
        assert_eq!(finite, vec![0.9, 0.7, 0.5]);
    }
}
