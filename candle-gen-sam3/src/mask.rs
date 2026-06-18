//! SAM3 mask head + instance post-processing — candle port of `mlx-gen-sam3`'s `mask.rs`
//! (`Sam3MaskDecoder` / `Sam3PixelDecoder` / `Sam3MaskEmbedder` + `post_process_instance_segmentation`),
//! itself a port of `transformers/models/sam3/modeling_sam3.py` (epic 5482, sc-6243 under sc-5062).
//!
//! MaskFormer-style: the 200 decoder queries are embedded and dot-producted (`bqc,bchw→bqhw`) against
//! a pixel-embedding FPN built from the backbone features (with the DETR-encoded 72² level swapped in
//! for the coarsest) to yield per-query masks; a 1×1 conv gives the semantic map. Prompt (text)
//! cross-attention conditions the pixel features. Layout NHWC (only the conv/GN/upsample wrappers dip
//! into NCHW). Post-process scores each query `σ(logits)·σ(presence)`, keeps `> threshold`, and
//! binarizes `σ(mask) > 0.5`.

use candle_gen::candle_core::Tensor;
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::gen_core::Quant;
use candle_gen::Result;

use crate::common::{
    conv2d_nhwc, group_norm_nhwc, join, layer_norm, upsample_nearest2d_nhwc, Linear, Weights,
};
use crate::config::Sam3DetrConfig;
use crate::detr::Attn;

const LN_EPS: f64 = 1e-5; // nn.LayerNorm / GroupNorm default eps in the mask decoder
const NUM_GROUPS: usize = 8;

/// 3-layer ReLU MLP embedding the queries for mask prediction (`Sam3MaskEmbedder`, `layers.0..2`,
/// ReLU after the first two).
struct MaskEmbedder {
    layers: Vec<Linear>,
}

impl MaskEmbedder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let layers = (0..3)
            .map(|i| Linear::load(w, &join(prefix, &format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(&h)?;
            if i < 2 {
                h = h.relu()?;
            }
        }
        Ok(h)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        for l in &mut self.layers {
            l.quantize(quant)?;
        }
        Ok(())
    }
}

/// FPN pixel decoder (`Sam3PixelDecoder`): coarse→fine, nearest-upsample + skip-add + conv/GN/ReLU.
struct PixelDecoder {
    convs: Vec<(Tensor, Tensor)>, // OIHW conv3×3 (torch-native) + bias
    norms: Vec<(Tensor, Tensor)>, // GroupNorm weight/bias
}

impl PixelDecoder {
    fn from_weights(w: &Weights, prefix: &str, stages: usize) -> Result<Self> {
        let convs = (0..stages)
            .map(|i| -> Result<(Tensor, Tensor)> {
                Ok((
                    w.require(&join(prefix, &format!("conv_layers.{i}.weight")))?,
                    w.require(&join(prefix, &format!("conv_layers.{i}.bias")))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let norms = (0..stages)
            .map(|i| -> Result<(Tensor, Tensor)> {
                Ok((
                    w.require(&join(prefix, &format!("norms.{i}.weight")))?,
                    w.require(&join(prefix, &format!("norms.{i}.bias")))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { convs, norms })
    }

    /// `features`: NHWC, fine→coarse `[288², 144², 72²]`. Returns the finest pixel embedding NHWC.
    /// The loop runs `features.len()-1` (= 2) conv stages; with `stages = 3` loaded, the last
    /// `conv_layers.2`/`norms.2` pair is loaded-but-unused — exactly as upstream `Sam3PixelDecoder`
    /// (F-017).
    fn forward(&self, features: &[Tensor]) -> Result<Tensor> {
        let mut prev = features[features.len() - 1].clone(); // coarsest (72²)
        for (layer_idx, feat) in features[..features.len() - 1].iter().rev().enumerate() {
            prev = upsample_nearest2d_nhwc(&prev, 2)?; // exact 2× (72→144→288)
            prev = prev.add(feat)?;
            let (cw, cb) = &self.convs[layer_idx];
            prev = conv2d_nhwc(&prev, cw, Some(cb), 1, 1)?; // 3×3 pad 1
            let (nw, nb) = &self.norms[layer_idx];
            prev = group_norm_nhwc(&prev, nw, nb, NUM_GROUPS, LN_EPS)?;
            prev = prev.relu()?;
        }
        Ok(prev)
    }
}

/// SAM3 mask head: prompt-conditioned pixel features + query embeddings → per-instance masks.
pub struct Sam3MaskHead {
    pixel_decoder: PixelDecoder,
    mask_embedder: MaskEmbedder,
    instance_proj_w: Tensor, // OIHW 1×1 (torch-native)
    instance_proj_b: Tensor,
    semantic_proj_w: Tensor, // OIHW 1×1 (→ 1 channel)
    semantic_proj_b: Tensor,
    prompt_attn: Attn,
    prompt_norm_w: Tensor,
    prompt_norm_b: Tensor,
}

/// Mask-head outputs.
pub struct MaskOutput {
    /// `[1, Q, 288, 288]` per-query mask logits.
    pub pred_masks: Tensor,
    /// `[1, 288, 288, 1]` semantic-segmentation logits (NHWC).
    pub semantic_seg: Tensor,
}

impl Sam3MaskHead {
    /// Load from a `facebook/sam3` weight map. `prefix` is typically `"detector_model"`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        let p = join(prefix, "mask_decoder");
        Ok(Self {
            // `num_upsampling_stages = 3` (the checkpoint ships `conv_layers.{0,1,2}` + `norms.{0,1,2}`);
            // the FPN is fed 3 levels (288²,144²,72²) so the loop runs 2 conv steps (F-017).
            pixel_decoder: PixelDecoder::from_weights(w, &join(&p, "pixel_decoder"), 3)?,
            mask_embedder: MaskEmbedder::from_weights(w, &join(&p, "mask_embedder"))?,
            instance_proj_w: w.require(&join(&p, "instance_projection.weight"))?,
            instance_proj_b: w.require(&join(&p, "instance_projection.bias"))?,
            semantic_proj_w: w.require(&join(&p, "semantic_projection.weight"))?,
            semantic_proj_b: w.require(&join(&p, "semantic_projection.bias"))?,
            prompt_attn: Attn::from_weights(w, &join(&p, "prompt_cross_attn"), cfg)?,
            prompt_norm_w: w.require(&join(&p, "prompt_cross_attn_norm.weight"))?,
            prompt_norm_b: w.require(&join(&p, "prompt_cross_attn_norm.bias"))?,
        })
    }

    /// Affine-quantize the mask head's linears to Q4/Q8 (the query mask-embedder MLP + the prompt
    /// cross-attention). The pixel-decoder convs + the instance/semantic 1×1 conv projections are
    /// dense.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.mask_embedder.quantize(quant)?;
        self.prompt_attn.quantize(quant)
    }

    /// `query_hidden`: `[1, Q, D]`; `backbone_features`: NHWC fine→coarse `[288²,144²,72²]`;
    /// `encoder_hidden`: `[1, 72², D]` (DETR-encoded 72² level); `prompt`: text `[1, L, D]`;
    /// `prompt_key_mask`: additive `[1, 1, 1, L]`.
    pub fn forward(
        &self,
        query_hidden: &Tensor,
        backbone_features: &[Tensor],
        encoder_hidden: &Tensor,
        prompt: &Tensor,
        prompt_key_mask: &Tensor,
    ) -> Result<MaskOutput> {
        // prompt cross-attention: encoder features attend to the text prompt
        let normed = layer_norm(
            encoder_hidden,
            &self.prompt_norm_w,
            &self.prompt_norm_b,
            LN_EPS,
        )?;
        let attn = self
            .prompt_attn
            .forward(&normed, prompt, prompt, Some(prompt_key_mask))?;
        let enc = encoder_hidden.add(&attn)?; // [1, 72², D]

        // swap the DETR-encoded 72² level in for the coarsest backbone level, then run the FPN
        let coarse = backbone_features.last().expect("at least one FPN level");
        let (_, h, w, d) = coarse.dims4()?;
        let enc_spatial = enc.reshape((1, h, w, d))?;
        let mut feats: Vec<Tensor> = backbone_features.to_vec();
        *feats.last_mut().unwrap() = enc_spatial;
        let pixel_embed = self.pixel_decoder.forward(&feats)?; // NHWC [1, 288, 288, D]

        // instance masks: dot product of query mask-embeddings with the projected pixel embedding
        let instance = conv2d_nhwc(
            &pixel_embed,
            &self.instance_proj_w,
            Some(&self.instance_proj_b),
            1,
            0,
        )?; // [1, 288, 288, D]
        let mask_emb = self.mask_embedder.forward(query_hidden)?; // [1, Q, D]
        let (_, ph, pw, pd) = instance.dims4()?;
        let inst_flat = instance
            .reshape((1, ph * pw, pd))?
            .transpose(1, 2)?
            .contiguous()?; // [1, D, HW]
        let nq = mask_emb.dim(1)?;
        let pred_masks = mask_emb.matmul(&inst_flat)?.reshape((1, nq, ph, pw))?; // [1, Q, 288, 288]

        let semantic_seg = conv2d_nhwc(
            &pixel_embed,
            &self.semantic_proj_w,
            Some(&self.semantic_proj_b),
            1,
            0,
        )?; // [1, 288, 288, 1]
        Ok(MaskOutput {
            pred_masks,
            semantic_seg,
        })
    }
}

/// One detected instance from the post-process.
pub struct Instance {
    /// `σ(logit)·σ(presence)` confidence.
    pub score: f32,
    /// Query index into the 200 queries.
    pub query: usize,
    /// Box xyxy in pixels at `target_size`.
    pub box_xyxy: [f32; 4],
    /// Binary mask `[h, w]` (U8 0/1) at the mask-head resolution (288²) — caller resizes to the image.
    pub mask: Tensor,
}

/// `post_process_instance_segmentation`: keep queries whose `σ(logits)·σ(presence) > threshold`,
/// binarize `σ(mask) > mask_threshold`. Boxes (xyxy∈[0,1]) are scaled to `target_wh`. Masks are
/// returned at the native 288² resolution (resize-to-image is the caller's concern).
pub fn post_process_instances(
    pred_logits: &Tensor,     // [1, Q]
    pred_boxes: &Tensor,      // [1, Q, 4] xyxy ∈ [0,1]
    presence_logits: &Tensor, // [1, 1]
    pred_masks: &Tensor,      // [1, Q, h, w]
    target_wh: (f32, f32),
    threshold: f32,
    mask_threshold: f32,
) -> Result<Vec<Instance>> {
    let presence = sigmoid(presence_logits)?.flatten_all()?.to_vec1::<f32>()?[0];
    let scores: Vec<f32> = sigmoid(pred_logits)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|&s| s * presence)
        .collect();
    let boxes: Vec<f32> = pred_boxes.flatten_all()?.to_vec1::<f32>()?;
    let (tw, th) = target_wh;
    let (h, w) = (pred_masks.dim(2)?, pred_masks.dim(3)?);

    let mut out = Vec::new();
    for (qi, &score) in scores.iter().enumerate() {
        if score <= threshold {
            continue;
        }
        let b = &boxes[qi * 4..qi * 4 + 4];
        let mask_logits = pred_masks.narrow(1, qi, 1)?; // [1, 1, h, w]
        let mask = sigmoid(&mask_logits)?
            .gt(mask_threshold as f64)?
            .reshape((h, w))?; // U8 [h, w]
        out.push(Instance {
            score,
            query: qi,
            box_xyxy: [b[0] * tw, b[1] * th, b[2] * tw, b[3] * th],
            mask,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    #[test]
    fn post_process_filters_by_score_and_binarizes() {
        let dev = Device::Cpu;
        // presence ~1; query 0 logit high (kept), query 1 low (dropped).
        let presence = Tensor::from_vec(vec![10.0f32], (1, 1), &dev).unwrap();
        let logits = Tensor::from_vec(vec![8.0f32, -8.0], (1, 2), &dev).unwrap();
        let boxes = Tensor::from_vec(
            vec![0.0f32, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0],
            (1, 2, 4),
            &dev,
        )
        .unwrap();
        // [1,2,2,2] masks: query 0 all positive → all-1; query 1 all negative → all-0.
        let masks = Tensor::from_vec(
            vec![5.0f32, 5.0, 5.0, 5.0, -5.0, -5.0, -5.0, -5.0],
            (1, 2, 2, 2),
            &dev,
        )
        .unwrap();
        let inst =
            post_process_instances(&logits, &boxes, &presence, &masks, (100.0, 200.0), 0.5, 0.5)
                .unwrap();
        assert_eq!(inst.len(), 1, "only query 0 passes threshold");
        assert_eq!(inst[0].query, 0);
        assert!(inst[0].score > 0.99);
        // box scaled by target_wh (100,200): [0,0,0.5,0.5] → [0,0,50,100]
        assert_eq!(inst[0].box_xyxy, [0.0, 0.0, 50.0, 100.0]);
        // mask binarized to all-1 (4 px)
        let s = inst[0]
            .mask
            .to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(s, 4.0);
    }
}
