//! SAM3 geometry/exemplar prompt encoder — candle port of `mlx-gen-sam3`'s `geometry.rs`
//! (`Sam3GeometryEncoder`, the box/point **PVS** prompt path), itself a port of
//! `transformers/models/sam3/modeling_sam3.py` (epic 5482, sc-6244 under sc-5062).
//!
//! A box prompt (normalized cxcywh) is encoded three ways and summed: a direct linear projection of
//! the coordinates, ROI-align pooling of the 72² FPN feature at the box (`roi_align` then a 7×7
//! conv), and a sine position encoding of the box center (+ raw h/w). A positive/negative label
//! embedding is added, a CLS token is appended, and the result is refined by 3 pre-norm transformer
//! layers that cross-attend to the 72² vision feature. The output prompt tokens `[1, N+1, 256]` are
//! concatenated with the text features and fed to the detector + mask decoder as
//! `combined_prompt_features` (see [`crate::model`]).
//!
//! `roi_align` is realized as a host-built bilinear sampling matrix — faithful to
//! `torchvision.ops.roi_align` (`spatial_scale=1`, `sampling_ratio=-1`, `aligned=False`) — then two
//! matmuls (the gather + the conv contraction). Unlike the MLX port (which forces the CPU stream to
//! dodge Metal's reduced-precision matmul), candle's CUDA f32 matmul is full-precision cuBLAS, so it
//! runs on-device.

use std::f32::consts::PI;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::Quant;
use candle_gen::Result;

use crate::common::{join, layer_norm, Linear, Weights};
use crate::config::Sam3GeometryConfig;
use crate::detr::{Attn, Ffn};

const SCALE_2PI: f32 = 2.0 * PI;

/// One pre-norm geometry-encoder layer (`Sam3GeometryEncoderLayer`): prompt self-attn → vision
/// cross-attn (key = vision + pos, value = vision) → ReLU FFN, each residual-added.
struct GeometryLayer {
    ln1_w: Tensor,
    ln1_b: Tensor,
    self_attn: Attn,
    ln2_w: Tensor,
    ln2_b: Tensor,
    cross_attn: Attn,
    ln3_w: Tensor,
    ln3_b: Tensor,
    ffn: Ffn,
    eps: f64,
}

impl GeometryLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3GeometryConfig) -> Result<Self> {
        let (nh, hd) = (cfg.num_attention_heads, cfg.head_dim());
        Ok(Self {
            ln1_w: w.require(&join(prefix, "layer_norm1.weight"))?,
            ln1_b: w.require(&join(prefix, "layer_norm1.bias"))?,
            self_attn: Attn::from_dims(w, &join(prefix, "self_attn"), nh, hd)?,
            ln2_w: w.require(&join(prefix, "layer_norm2.weight"))?,
            ln2_b: w.require(&join(prefix, "layer_norm2.bias"))?,
            cross_attn: Attn::from_dims(w, &join(prefix, "cross_attn"), nh, hd)?,
            ln3_w: w.require(&join(prefix, "layer_norm3.weight"))?,
            ln3_b: w.require(&join(prefix, "layer_norm3.bias"))?,
            ffn: Ffn::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    /// `prompt`: `[1, P, C]`; `vision`: `[1, H·W, C]` (raw 72² feature, flattened); `vision_pos`:
    /// `[1, H·W, C]`. All prompt tokens are valid (PVS path), so attention runs unmasked.
    fn forward(&self, prompt: &Tensor, vision: &Tensor, vision_pos: &Tensor) -> Result<Tensor> {
        let h = layer_norm(prompt, &self.ln1_w, &self.ln1_b, self.eps)?;
        let a = self.self_attn.forward(&h, &h, &h, None)?;
        let x = prompt.add(&a)?;

        let h = layer_norm(&x, &self.ln2_w, &self.ln2_b, self.eps)?;
        let key = vision.add(vision_pos)?;
        let a = self.cross_attn.forward(&h, &key, vision, None)?;
        let x = x.add(&a)?;

        let h = layer_norm(&x, &self.ln3_w, &self.ln3_b, self.eps)?;
        let a = self.ffn.forward(&h)?;
        Ok(x.add(&a)?)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.self_attn.quantize(quant)?;
        self.cross_attn.quantize(quant)?;
        self.ffn.quantize(quant)
    }
}

/// SAM3 geometry/exemplar prompt encoder (`Sam3GeometryEncoder`).
pub struct Sam3GeometryEncoder {
    label_embed: Tensor,  // [2, C]
    cls_embed: Tensor,    // [1, C]
    boxes_direct: Linear, // Linear(4, C)
    boxes_pool_w: Tensor, // Conv2d(C, C, R) weight [C, C, R, R] (torch OIHW)
    boxes_pool_b: Tensor, // [C]
    boxes_pos: Linear,    // Linear(C + 2, C)
    vision_ln_w: Tensor,
    vision_ln_b: Tensor,
    final_proj: Linear,
    prompt_ln_w: Tensor,
    prompt_ln_b: Tensor,
    layers: Vec<GeometryLayer>,
    output_ln_w: Tensor,
    output_ln_b: Tensor,
    cfg: Sam3GeometryConfig,
}

impl Sam3GeometryEncoder {
    /// Load from a `facebook/sam3` weight map. `prefix` is typically
    /// `"detector_model.geometry_encoder"`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3GeometryConfig) -> Result<Self> {
        let layers = (0..cfg.num_layers)
            .map(|i| GeometryLayer::from_weights(w, &join(prefix, &format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            label_embed: w.require(&join(prefix, "label_embed.weight"))?,
            cls_embed: w.require(&join(prefix, "cls_embed.weight"))?,
            boxes_direct: Linear::load(w, &join(prefix, "boxes_direct_project"))?,
            boxes_pool_w: w.require(&join(prefix, "boxes_pool_project.weight"))?,
            boxes_pool_b: w.require(&join(prefix, "boxes_pool_project.bias"))?,
            boxes_pos: Linear::load(w, &join(prefix, "boxes_pos_enc_project"))?,
            vision_ln_w: w.require(&join(prefix, "vision_layer_norm.weight"))?,
            vision_ln_b: w.require(&join(prefix, "vision_layer_norm.bias"))?,
            final_proj: Linear::load(w, &join(prefix, "final_proj"))?,
            prompt_ln_w: w.require(&join(prefix, "prompt_layer_norm.weight"))?,
            prompt_ln_b: w.require(&join(prefix, "prompt_layer_norm.bias"))?,
            layers,
            output_ln_w: w.require(&join(prefix, "output_layer_norm.weight"))?,
            output_ln_b: w.require(&join(prefix, "output_layer_norm.bias"))?,
            cfg: cfg.clone(),
        })
    }

    /// Affine-quantize the geometry encoder's projections to Q4/Q8 (the `final_proj` + the 3 layers'
    /// attention/FFN). The `boxes_direct` (4→256) and `boxes_pos` (258→256) projections auto-skip and
    /// stay dense ([`Linear::quantize`]); the ROI-pool conv + embeddings are dense.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.boxes_direct.quantize(quant)?;
        self.boxes_pos.quantize(quant)?;
        self.final_proj.quantize(quant)?;
        for layer in &mut self.layers {
            layer.quantize(quant)?;
        }
        Ok(())
    }

    /// Encode box prompts into prompt tokens.
    ///
    /// * `boxes`: `[1, N, 4]` normalized cxcywh ∈ [0, 1] (relative to the model input).
    /// * `box_labels`: length `N` (`1` = positive, `0` = negative).
    /// * `vision`: the 72² FPN feature, **NHWC** `[1, H, W, C]` (the level the detector consumes).
    /// * `vision_pos`: the matching flattened sine position embedding `[1, H·W, C]`.
    ///
    /// Returns the geometry prompt tokens `[1, N + 1, C]` (boxes followed by the CLS token).
    pub fn forward(
        &self,
        boxes: &Tensor,
        box_labels: &[i32],
        vision: &Tensor,
        vision_pos: &Tensor,
    ) -> Result<Tensor> {
        let (_, h, w, _) = vision.dims4()?;
        let c = self.cfg.hidden_size;
        let n = boxes.dim(1)?;
        let device = vision.device();
        let eps = self.cfg.layer_norm_eps;

        // (1) direct projection of the box coordinates
        let direct = self.boxes_direct.forward(boxes)?; // [1, N, C]

        // (2) ROI-align pooling of the channel-normalized 72² feature, then the 7×7 conv
        let norm_feat = layer_norm(vision, &self.vision_ln_w, &self.vision_ln_b, eps)?; // [1,H,W,C]
        let boxes_host: Vec<f32> = boxes.flatten_all()?.to_vec1::<f32>()?; // N·4 cxcywh
        let pooled = self.roi_pool(&boxes_host, n, &norm_feat, h, w)?; // [1, N, C]

        // (3) sine position encoding of the box center (+ raw h/w), projected to C
        let pos_enc = box_pos_encoding(&boxes_host, n, c, device)?; // [1, N, C+2]
        let pos = self.boxes_pos.forward(&pos_enc)?; // [1, N, C]

        // label (positive/negative) embedding + the three box encodings
        let lbl_idx = Tensor::from_vec(
            box_labels.iter().map(|&l| l as u32).collect::<Vec<u32>>(),
            n,
            device,
        )?;
        let label = self
            .label_embed
            .index_select(&lbl_idx, 0)?
            .reshape((1, n, c))?;
        let boxes_embed = direct.add(&pooled)?.add(&pos)?.add(&label)?; // [1, N, C]

        // append the always-valid CLS token
        let cls = self.cls_embed.reshape((1, 1, c))?;
        let prompt = Tensor::cat(&[&boxes_embed, &cls], 1)?; // [1, N+1, C]
        let prompt = layer_norm(
            &self.final_proj.forward(&prompt)?,
            &self.prompt_ln_w,
            &self.prompt_ln_b,
            eps,
        )?;

        // refine with transformer layers cross-attending to the raw vision feature
        let vision_flat = vision.reshape((1, h * w, c))?;
        let mut x = prompt;
        for layer in &self.layers {
            x = layer.forward(&x, &vision_flat, vision_pos)?;
        }
        layer_norm(&x, &self.output_ln_w, &self.output_ln_b, eps)
    }

    /// `roi_align` (bilinear ROI pool, `roi_size`²) + the `boxes_pool_project` 7×7 conv, fused as two
    /// matmuls: a host-built sampling matrix `[N·R², H·W]` gathers the pooled grid, then the conv
    /// weight `[C, C·R²]` contracts it to `[1, N, C]`.
    fn roi_pool(
        &self,
        boxes_host: &[f32],
        n: usize,
        norm_feat: &Tensor,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let c = self.cfg.hidden_size;
        let r = self.cfg.roi_size;
        let device = norm_feat.device();

        let s = roi_align_matrix(boxes_host, n, h as i32, w as i32, r as i32);
        let s = Tensor::from_vec(s, (n * r * r, h * w), device)?;
        let vflat = norm_feat.reshape((h * w, c))?;
        // sampled grid: [N·R², C] → [N, C, R²] (the conv's (in, kH, kW) order) → [N, C·R²]
        let sampled = s
            .matmul(&vflat)?
            .reshape((n, r * r, c))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((n, c * r * r))?;
        // boxes_pool_project conv weight [C_out, C_in, R, R] → [C_out, C_in·R²]
        let wflat = self.boxes_pool_w.reshape((c, c * r * r))?;
        let pooled = sampled.matmul(&wflat.t()?.contiguous()?)?; // [N, C]
        Ok(pooled
            .broadcast_add(&self.boxes_pool_b)?
            .reshape((1, n, c))?)
    }
}

/// Sine position encoding of box centers (+ raw height/width), `[1, N, C+2]`. Mirrors
/// `Sam3GeometryEncoder._encode_box_coordinates`: `cat(pos_y, pos_x, height, width)` where each
/// `pos_*` is the `sin(even)/cos(odd)` interleave of `center·2π / dim_t`.
fn box_pos_encoding(boxes_cxcywh: &[f32], n: usize, c: usize, device: &Device) -> Result<Tensor> {
    let npf = c / 2; // 128
    let dim_t: Vec<f32> = (0..npf)
        .map(|i| 10000f32.powf(2.0 * ((i / 2) as f32) / npf as f32))
        .collect();
    let total = 2 * npf + 2; // 258
    let mut out = vec![0f32; n * total];
    let enc = |v: f32, dst: &mut [f32]| {
        let e = v * SCALE_2PI;
        for j in 0..npf / 2 {
            dst[2 * j] = (e / dim_t[2 * j]).sin();
            dst[2 * j + 1] = (e / dim_t[2 * j + 1]).cos();
        }
    };
    for bi in 0..n {
        let (cx, cy, bw, bh) = (
            boxes_cxcywh[bi * 4],
            boxes_cxcywh[bi * 4 + 1],
            boxes_cxcywh[bi * 4 + 2],
            boxes_cxcywh[bi * 4 + 3],
        );
        let base = bi * total;
        enc(cy, &mut out[base..base + npf]); // pos_y first
        enc(cx, &mut out[base + npf..base + 2 * npf]); // then pos_x
        out[base + 2 * npf] = bh; // raw box height
        out[base + 2 * npf + 1] = bw; // raw box width
    }
    Ok(Tensor::from_vec(out, (1, n, total), device)?)
}

/// Host-built `torchvision.ops.roi_align` sampling matrix `[N·R², H·W]` for boxes in normalized
/// cxcywh. Each output cell row holds the bilinear interpolation weights (averaged over the adaptive
/// sample grid) over the flattened H·W feature. `spatial_scale=1`, `sampling_ratio=-1`,
/// `aligned=False`.
fn roi_align_matrix(boxes_cxcywh: &[f32], n: usize, h: i32, w: i32, r: i32) -> Vec<f32> {
    let hw = (h * w) as usize;
    let o = r as usize;
    let (hf, wf) = (h as f32, w as f32);
    let mut s = vec![0f32; n * o * o * hw];
    for bi in 0..n {
        let (cx, cy, bw, bh) = (
            boxes_cxcywh[bi * 4],
            boxes_cxcywh[bi * 4 + 1],
            boxes_cxcywh[bi * 4 + 2],
            boxes_cxcywh[bi * 4 + 3],
        );
        // normalized cxcywh → xyxy → feature coordinates (× W, H)
        let start_w = (cx - 0.5 * bw) * wf;
        let start_h = (cy - 0.5 * bh) * hf;
        let roi_w = ((cx + 0.5 * bw) * wf - start_w).max(1.0); // !aligned → min size 1
        let roi_h = ((cy + 0.5 * bh) * hf - start_h).max(1.0);
        let bin_w = roi_w / r as f32;
        let bin_h = roi_h / r as f32;
        let grid_w = (roi_w / r as f32).ceil().max(1.0) as i32;
        let grid_h = (roi_h / r as f32).ceil().max(1.0) as i32;
        let count = (grid_h * grid_w).max(1) as f32;
        for ph in 0..o {
            for pw in 0..o {
                let row = (bi * o * o + ph * o + pw) * hw;
                let srow = &mut s[row..row + hw];
                for iy in 0..grid_h {
                    let y = start_h + ph as f32 * bin_h + (iy as f32 + 0.5) * bin_h / grid_h as f32;
                    for ix in 0..grid_w {
                        let x =
                            start_w + pw as f32 * bin_w + (ix as f32 + 0.5) * bin_w / grid_w as f32;
                        bilinear_acc(srow, h, w, y, x, 1.0 / count);
                    }
                }
            }
        }
    }
    s
}

/// Accumulate one bilinear sample's four corner weights into a `[H·W]` sampling row. Out-of-range
/// samples contribute nothing; edge handling matches torchvision's `bilinear_interpolate`.
fn bilinear_acc(row: &mut [f32], h: i32, w: i32, y: f32, x: f32, weight: f32) {
    let (hf, wf) = (h as f32, w as f32);
    if y < -1.0 || y > hf || x < -1.0 || x > wf {
        return;
    }
    let mut y = if y <= 0.0 { 0.0 } else { y };
    let mut x = if x <= 0.0 { 0.0 } else { x };
    let mut y_low = y as i32;
    let mut x_low = x as i32;
    let y_high;
    let x_high;
    if y_low >= h - 1 {
        y_low = h - 1;
        y_high = h - 1;
        y = y_low as f32;
    } else {
        y_high = y_low + 1;
    }
    if x_low >= w - 1 {
        x_low = w - 1;
        x_high = w - 1;
        x = x_low as f32;
    } else {
        x_high = x_low + 1;
    }
    let (ly, lx) = (y - y_low as f32, x - x_low as f32);
    let (hy, hx) = (1.0 - ly, 1.0 - lx);
    let idx = |yy: i32, xx: i32| (yy * w + xx) as usize;
    row[idx(y_low, x_low)] += hy * hx * weight;
    row[idx(y_low, x_high)] += hy * lx * weight;
    row[idx(y_high, x_low)] += ly * hx * weight;
    row[idx(y_high, x_high)] += ly * lx * weight;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roi_align_matrix_rows_are_normalized() {
        // A box covering the whole 4×4 feature → every output cell's weights sum to 1.
        let boxes = [0.5f32, 0.5, 1.0, 1.0]; // cxcywh covering [0,1]²
        let s = roi_align_matrix(&boxes, 1, 4, 4, 7);
        let hw = 16;
        for cell in 0..49 {
            let sum: f32 = s[cell * hw..(cell + 1) * hw].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "cell {cell} weight sum {sum}");
        }
    }

    #[test]
    fn bilinear_acc_hits_exact_pixel() {
        // Sampling exactly at an interior pixel center puts all weight on that pixel.
        let mut row = vec![0f32; 16];
        bilinear_acc(&mut row, 4, 4, 2.0, 1.0, 1.0);
        assert!((row[2 * 4 + 1] - 1.0).abs() < 1e-6);
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn box_pos_encoding_has_expected_shape_and_tail() {
        // Tail two columns are the raw box height then width.
        let boxes = [0.3f32, 0.4, 0.2, 0.6];
        let enc = box_pos_encoding(&boxes, 1, 256, &Device::Cpu).unwrap();
        assert_eq!(enc.dims(), &[1, 1, 258]);
        let v = enc.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[256] - 0.6).abs() < 1e-6, "height tail");
        assert!((v[257] - 0.2).abs() < 1e-6, "width tail");
    }
}
