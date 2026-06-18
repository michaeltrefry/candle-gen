//! SAM3 tracker — candle port of `mlx-gen-sam3`'s `tracker.rs` (epic 5482, sc-6245 under sc-5062).
//!
//! The SAM3 tracker is **SAM2.1** architecture (mask decoder 256/2L/8H plus dynamic-multimask via
//! stability; prompt encoder; memory bank for video). It is fed by the **shared PE vision encoder**
//! ([`crate::vision`]) plus its own `tracker_neck` FPN (NOT a separate Hiera trunk). The checkpoint
//! stores it under `tracker_model.*` / `tracker_neck.*` in `transformers`-5 module naming. This is a
//! direct port of the public Apache-2.0 `transformers` reference (`modeling_sam3_tracker_video.py`),
//! mirroring the parity-green MLX twin line-by-line.
//!
//! Layout is **NHWC** end-to-end (matching [`crate::vision`] / [`crate::mask`]); only the
//! conv/transposed-conv/max-pool/group-norm wrappers dip into candle's NCHW. Conv kernels load RAW
//! from `facebook/sam3` (torch-native OIHW / IOHW), so there is NO kernel permute (the MLX side
//! permutes to OHWI because MLX convs are channels-last). Quantization is deferred to sc-6246.

use std::f64::consts::PI;
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::gen_core::Quant;
use candle_gen::{CandleError, Result};

use crate::common::{
    conv2d_nhwc, conv2d_nhwc_grouped, conv_transpose2d_nhwc, join, layer_norm, sdpa, Linear,
    Weights,
};
use crate::config::Sam3VisionConfig;
use crate::vision::{Backbone, FpnLayer};

// --- fixed facebook/sam3 tracker hyperparameters (Sam3TrackerVideoMaskDecoderConfig) -------------
const HIDDEN: usize = 256;
const NUM_HEADS: usize = 8;
const NUM_MASK_TOKENS: usize = 4; // num_multimask_outputs (3) + 1
const LN_EPS: f64 = 1e-6;
const INPUT_SIZE: f32 = 1008.0; // image_size
const STABILITY_DELTA: f32 = 0.05; // dynamic_multimask_stability_delta
const STABILITY_THRESH: f32 = 0.98; // dynamic_multimask_stability_thresh
const NO_OBJ_SCORE: f32 = -1024.0; // logit for "object absent" frames
const MASK_INPUT_SIZE: usize = 288; // prompt encoder mask_input_size (4·1008/14)

/// Take a single index `i` along `axis`, dropping that axis (candle twin of MLX `take_axis(i)`).
fn take1(x: &Tensor, i: usize, axis: usize) -> Result<Tensor> {
    Ok(x.narrow(axis, i, 1)?.squeeze(axis)?)
}

/// Slice `[start, end)` along `axis` (keeps the axis).
fn slice_axis(x: &Tensor, axis: usize, start: usize, end: usize) -> Result<Tensor> {
    Ok(x.narrow(axis, start, end - start)?)
}

fn weight_bias(w: &Weights, prefix: &str) -> Result<(Tensor, Tensor)> {
    Ok((
        w.require(&join(prefix, "weight"))?,
        w.require(&join(prefix, "bias"))?,
    ))
}

fn ln(x: &Tensor, p: &(Tensor, Tensor)) -> Result<Tensor> {
    layer_norm(x, &p.0, &p.1, LN_EPS)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
}

fn scalar(t: &Tensor) -> Result<f32> {
    Ok(to_vec(t)?[0])
}

// --- FeedForward (Sam3TrackerVideoFeedForward) ---------------------------------------------------

/// `proj_in → act → (layers.i → act)* → proj_out → [sigmoid]`. `act` is ReLU throughout the tracker.
struct FeedForward {
    proj_in: Linear,
    layers: Vec<Linear>,
    proj_out: Linear,
    sigmoid_output: bool,
}

impl FeedForward {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_layers: usize,
        sigmoid_output: bool,
    ) -> Result<Self> {
        let layers = (0..num_layers.saturating_sub(2))
            .map(|i| Linear::load(w, &join(prefix, &format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_in: Linear::load(w, &join(prefix, "proj_in"))?,
            layers,
            proj_out: Linear::load(w, &join(prefix, "proj_out"))?,
            sigmoid_output,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.proj_in.forward(x)?.relu()?;
        for l in &self.layers {
            h = l.forward(&h)?.relu()?;
        }
        h = self.proj_out.forward(&h)?;
        if self.sigmoid_output {
            h = sigmoid(&h)?;
        }
        Ok(h)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.proj_in.quantize(quant)?;
        for l in &mut self.layers {
            l.quantize(quant)?;
        }
        self.proj_out.quantize(quant)
    }
}

// --- Attention (Sam3TrackerVideoAttention, with q/k/v down-projection) ---------------------------

/// MHA on `[b, n, hidden]` tokens; q/k/v project to `internal = hidden / downsample`, split into
/// `NUM_HEADS`, SDPA, then `o_proj` back to `hidden`. The head split is derived from the loaded
/// projection width at forward time (so it works for both the full-width self-attn and the
/// half-width cross-attn).
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let l = |n: &str| Linear::load(w, &join(prefix, n));
        Ok(Self {
            q: l("q_proj")?,
            k: l("k_proj")?,
            v: l("v_proj")?,
            o: l("o_proj")?,
            num_heads: NUM_HEADS,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let sep = |x: &Tensor| -> Result<Tensor> {
            let (b, n, c) = x.dims3()?;
            Ok(x.reshape((b, n, self.num_heads, c / self.num_heads))?
                .transpose(1, 2)?
                .contiguous()?) // [b, heads, n, hd]
        };
        let q = sep(&self.q.forward(q)?)?;
        let k = sep(&self.k.forward(k)?)?;
        let v = sep(&self.v.forward(v)?)?;
        let scale = 1.0 / (q.dim(3)? as f64).sqrt();
        let out = sdpa(&q, &k, &v, scale)?;
        let (b, h, n, c) = out.dims4()?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, n, h * c))?;
        self.o.forward(&out)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.q.quantize(quant)?;
        self.k.quantize(quant)?;
        self.v.quantize(quant)?;
        self.o.quantize(quant)
    }
}

// --- Two-way attention block + transformer (Sam3TrackerVideoTwoWayTransformer) -------------------

struct TwoWayBlock {
    self_attn: Attention,
    norm1: (Tensor, Tensor),
    cross_t2i: Attention,
    norm2: (Tensor, Tensor),
    mlp: FeedForward,
    norm3: (Tensor, Tensor),
    cross_i2t: Attention,
    norm4: (Tensor, Tensor),
    skip_first_layer_pe: bool,
}

impl TwoWayBlock {
    fn from_weights(w: &Weights, prefix: &str, skip_first_layer_pe: bool) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::from_weights(w, &join(prefix, "self_attn"))?,
            norm1: weight_bias(w, &join(prefix, "layer_norm1"))?,
            cross_t2i: Attention::from_weights(w, &join(prefix, "cross_attn_token_to_image"))?,
            norm2: weight_bias(w, &join(prefix, "layer_norm2"))?,
            mlp: FeedForward::from_weights(w, &join(prefix, "mlp"), 2, false)?,
            norm3: weight_bias(w, &join(prefix, "layer_norm3"))?,
            cross_i2t: Attention::from_weights(w, &join(prefix, "cross_attn_image_to_token"))?,
            norm4: weight_bias(w, &join(prefix, "layer_norm4"))?,
            skip_first_layer_pe,
        })
    }

    /// `queries`/`keys`: `[b, nq, D]` / `[b, nk, D]`; `query_pe`/`key_pe`: same shapes.
    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        query_pe: &Tensor,
        key_pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = queries.add(query_pe)?;
            queries.add(&self.self_attn.forward(&q, &q, queries)?)?
        };
        queries = ln(&queries, &self.norm1)?;

        let q = queries.add(query_pe)?;
        let k = keys.add(key_pe)?;
        queries = ln(
            &queries.add(&self.cross_t2i.forward(&q, &k, keys)?)?,
            &self.norm2,
        )?;
        queries = ln(&queries.add(&self.mlp.forward(&queries)?)?, &self.norm3)?;

        let q = queries.add(query_pe)?;
        let k = keys.add(key_pe)?;
        let keys = ln(
            &keys.add(&self.cross_i2t.forward(&k, &q, &queries)?)?,
            &self.norm4,
        )?;
        Ok((queries, keys))
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.self_attn.quantize(quant)?;
        self.cross_t2i.quantize(quant)?;
        self.mlp.quantize(quant)?;
        self.cross_i2t.quantize(quant)
    }
}

struct TwoWayTransformer {
    layers: Vec<TwoWayBlock>,
    final_attn: Attention,
    norm_final: (Tensor, Tensor),
}

impl TwoWayTransformer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            layers: vec![
                TwoWayBlock::from_weights(w, &join(prefix, "layers.0"), true)?,
                TwoWayBlock::from_weights(w, &join(prefix, "layers.1"), false)?,
            ],
            final_attn: Attention::from_weights(w, &join(prefix, "final_attn_token_to_image"))?,
            norm_final: weight_bias(w, &join(prefix, "layer_norm_final_attn"))?,
        })
    }

    /// `image_embedding`/`image_pe`: token-flattened `[b, hw, D]`; `point_embedding`: `[b, n, D]`.
    fn forward(
        &self,
        image_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut queries = point_embedding.clone();
        let mut keys = image_embedding.clone();
        for layer in &self.layers {
            let (q, k) = layer.forward(&queries, &keys, point_embedding, image_pe)?;
            queries = q;
            keys = k;
        }
        let q = queries.add(point_embedding)?;
        let k = keys.add(image_pe)?;
        queries = ln(
            &queries.add(&self.final_attn.forward(&q, &k, &keys)?)?,
            &self.norm_final,
        )?;
        Ok((queries, keys))
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(quant)?;
        }
        self.final_attn.quantize(quant)
    }
}

// --- Prompt encoder (Sam3TrackerVideoPromptEncoder, box path) ------------------------------------

/// `PositionEmbeddingRandom` (`shared_embedding`): `cat[sin, cos](2π · (2·coord−1) @ gaussian)`.
/// `gaussian` is `[2, HIDDEN/2]`. Coords are normalized to `[0,1]`.
struct PositionEmbeddingRandom {
    gaussian: Tensor, // [2, 128]
}

impl PositionEmbeddingRandom {
    /// `coords_norm`: `[..., 2]` in `[0,1]`. Returns `[..., HIDDEN]`.
    fn forward(&self, coords_norm: &Tensor) -> Result<Tensor> {
        let dims = coords_norm.dims().to_vec();
        let lead: usize = dims[..dims.len() - 1].iter().product();
        let half = self.gaussian.dim(1)?; // 128
                                          // [.., 2] → [lead, 2] @ [2, 128] → [lead, 128], scaled by 2π.
        let flat = coords_norm.affine(2.0, -1.0)?.reshape((lead, 2))?;
        let proj = flat.matmul(&self.gaussian)?.affine(2.0 * PI, 0.0)?;
        let cat = Tensor::cat(&[&proj.sin()?, &proj.cos()?], 1)?; // [lead, 256]
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = half * 2;
        Ok(cat.reshape(out_dims)?)
    }

    /// Dense positional grid for a `g×g` feature map → NHWC `[1, g, g, HIDDEN]` (each cell at its
    /// pixel center `(i+0.5)/g`).
    fn dense_pe(&self, g: usize) -> Result<Tensor> {
        let mut coords = vec![0f32; g * g * 2];
        for y in 0..g {
            for x in 0..g {
                let idx = (y * g + x) * 2;
                coords[idx] = (x as f32 + 0.5) / g as f32;
                coords[idx + 1] = (y as f32 + 0.5) / g as f32;
            }
        }
        let coords = Tensor::from_vec(coords, (1, g, g, 2), self.gaussian.device())?;
        self.forward(&coords)
    }
}

/// SAM3 tracker prompt encoder — box path (single-frame PVS) plus the empty-point / dense-mask paths
/// the video tracker uses. `point_embed[2]`/`[3]` are the box-corner embeddings; the padding point
/// uses `not_a_point_embed`. Dense embedding for a box-only prompt is the broadcast `no_mask_embed`.
struct PromptEncoder {
    pe: PositionEmbeddingRandom,
    point_embed: Tensor,   // [4, 256]
    not_a_point: Tensor,   // [1, 256]
    no_mask_embed: Tensor, // [1, 256]
    /// `mask_embed` dense-mask path (`Sam3TrackerVideoMaskEmbedding`): conv1 1→4 k2s2, LN, gelu,
    /// conv2 4→16 k2s2, LN, gelu, conv3 16→256 k1 → dense `[1, 72, 72, 256]`. Convs load raw OIHW; the
    /// channels-first LayerNorms are a plain LN over the (last, NHWC) channel axis.
    mask_conv1: (Tensor, Tensor),
    mask_ln1: (Tensor, Tensor),
    mask_conv2: (Tensor, Tensor),
    mask_ln2: (Tensor, Tensor),
    mask_conv3: (Tensor, Tensor),
}

impl PromptEncoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let me = join(prefix, "mask_embed");
        Ok(Self {
            pe: PositionEmbeddingRandom {
                gaussian: w.require(&join(prefix, "shared_embedding.positional_embedding"))?,
            },
            point_embed: w.require(&join(prefix, "point_embed.weight"))?,
            not_a_point: w.require(&join(prefix, "not_a_point_embed.weight"))?,
            no_mask_embed: w.require(&join(prefix, "no_mask_embed.weight"))?,
            mask_conv1: weight_bias(w, &join(&me, "conv1"))?,
            mask_ln1: weight_bias(w, &join(&me, "layer_norm1"))?,
            mask_conv2: weight_bias(w, &join(&me, "conv2"))?,
            mask_ln2: weight_bias(w, &join(&me, "layer_norm2"))?,
            mask_conv3: weight_bias(w, &join(&me, "conv3"))?,
        })
    }

    /// `mask_embed` forward plus the empty-point sparse, for a mask-conditioned (detection-seeded)
    /// frame. `mask_288`: NHWC `[1, 288, 288, 1]`. Returns `(sparse [1, 1, 2, 256] (2× not_a_point),
    /// dense [1, 72, 72, 256])`.
    fn encode_mask_prompt(&self, mask_288: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = conv2d_nhwc(mask_288, &self.mask_conv1.0, Some(&self.mask_conv1.1), 2, 0)?;
        let h = ln(&h, &self.mask_ln1)?.gelu_erf()?;
        let h = conv2d_nhwc(&h, &self.mask_conv2.0, Some(&self.mask_conv2.1), 2, 0)?;
        let h = ln(&h, &self.mask_ln2)?.gelu_erf()?;
        let dense = conv2d_nhwc(&h, &self.mask_conv3.0, Some(&self.mask_conv3.1), 1, 0)?;
        let nap = self.not_a_point.reshape((1, 1, 1, HIDDEN))?;
        let sparse = Tensor::cat(&[&nap, &nap], 2)?;
        Ok((sparse, dense))
    }

    /// `box_xyxy` in **1008-input** pixel space → `(sparse [1, 1, 3, 256], dense [1, g, g, 256])`.
    fn encode_box(&self, box_xyxy: [f32; 4], g: usize) -> Result<(Tensor, Tensor)> {
        let device = self.not_a_point.device();
        let norm = |v: f32| (v + 0.5) / INPUT_SIZE;
        let coords = [
            norm(box_xyxy[0]),
            norm(box_xyxy[1]),
            norm(box_xyxy[2]),
            norm(box_xyxy[3]),
            0.0,
            0.0,
        ];
        let coords = Tensor::from_vec(coords.to_vec(), (1, 1, 3, 2), device)?;
        let emb = self.pe.forward(&coords)?; // [1,1,3,256]
        let pe2 = take1(&self.point_embed, 2, 0)?.reshape((1, 1, 1, HIDDEN))?;
        let pe3 = take1(&self.point_embed, 3, 0)?.reshape((1, 1, 1, HIDDEN))?;
        let row0 = take1(&emb, 0, 2)?.reshape((1, 1, 1, HIDDEN))?.add(&pe2)?;
        let row1 = take1(&emb, 1, 2)?.reshape((1, 1, 1, HIDDEN))?.add(&pe3)?;
        let row2 = self.not_a_point.reshape((1, 1, 1, HIDDEN))?;
        let sparse = Tensor::cat(&[&row0, &row1, &row2], 2)?; // [1,1,3,256]
        let dense = self
            .no_mask_embed
            .reshape((1, 1, 1, HIDDEN))?
            .broadcast_as((1, g, g, HIDDEN))?
            .contiguous()?;
        Ok((sparse, dense))
    }

    /// Empty-prompt encoding for a no-prompt (memory-conditioned) tracking frame: both sparse tokens
    /// collapse to `not_a_point_embed`; dense is the broadcast `no_mask_embed`. Returns
    /// `(sparse [1, 1, 2, 256], dense [1, g, g, 256])`.
    fn encode_empty_point(&self, g: usize) -> Result<(Tensor, Tensor)> {
        let nap = self.not_a_point.reshape((1, 1, 1, HIDDEN))?;
        let sparse = Tensor::cat(&[&nap, &nap], 2)?; // [1,1,2,256]
        let dense = self
            .no_mask_embed
            .reshape((1, 1, 1, HIDDEN))?
            .broadcast_as((1, g, g, HIDDEN))?
            .contiguous()?;
        Ok((sparse, dense))
    }
}

/// `_dynamic_multimask_via_stability`: on a single-mask request, keep mask token 0 if its stability
/// score — the IoU between the `±delta`-thresholded mask areas — is `≥ thresh`; otherwise fall back
/// to the best-predicted-IoU multimask candidate (tokens 1..). `masks`: `[1, NUM_MASK_TOKENS, mg, mg]`;
/// `ious`: `[1, NUM_MASK_TOKENS]`. Returns `(mask [1,1,mg,mg], iou [1,1])`. B=P=1, so the reference's
/// `torch.where` is a scalar choice (evaluated host-side).
fn dynamic_multimask_via_stability(masks: &Tensor, ious: &Tensor) -> Result<(Tensor, Tensor)> {
    let iou_v = to_vec(ious)?;
    let best = 1 + argmax(&iou_v[1..]);
    let single = slice_axis(masks, 1, 0, 1)?; // [1,1,mg,mg]
    let sv = to_vec(&single)?;
    let area_i = sv.iter().filter(|&&x| x > STABILITY_DELTA).count() as f32;
    let area_u = sv.iter().filter(|&&x| x > -STABILITY_DELTA).count() as f32;
    let stability = if area_u > 0.0 { area_i / area_u } else { 1.0 };
    if stability >= STABILITY_THRESH {
        Ok((single, slice_axis(ious, 1, 0, 1)?))
    } else {
        Ok((
            slice_axis(masks, 1, best, best + 1)?,
            slice_axis(ious, 1, best, best + 1)?,
        ))
    }
}

// --- Mask decoder (Sam3TrackerVideoMaskDecoder) --------------------------------------------------

struct MaskDecoder {
    transformer: TwoWayTransformer,
    iou_token: Tensor,       // [1, 256]
    mask_tokens: Tensor,     // [4, 256]
    obj_score_token: Tensor, // [1, 256]
    upscale_conv1: (Tensor, Tensor),
    upscale_layer_norm: (Tensor, Tensor),
    upscale_conv2: (Tensor, Tensor),
    hypernet: Vec<FeedForward>,
    iou_head: FeedForward,
    obj_score_head: FeedForward,
}

impl MaskDecoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let hypernet = (0..NUM_MASK_TOKENS)
            .map(|i| {
                FeedForward::from_weights(
                    w,
                    &join(prefix, &format!("output_hypernetworks_mlps.{i}")),
                    3,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            transformer: TwoWayTransformer::from_weights(w, &join(prefix, "transformer"))?,
            iou_token: w.require(&join(prefix, "iou_token.weight"))?,
            mask_tokens: w.require(&join(prefix, "mask_tokens.weight"))?,
            obj_score_token: w.require(&join(prefix, "obj_score_token.weight"))?,
            upscale_conv1: weight_bias(w, &join(prefix, "upscale_conv1"))?,
            upscale_layer_norm: weight_bias(w, &join(prefix, "upscale_layer_norm"))?,
            upscale_conv2: weight_bias(w, &join(prefix, "upscale_conv2"))?,
            hypernet,
            iou_head: FeedForward::from_weights(w, &join(prefix, "iou_prediction_head"), 3, true)?,
            obj_score_head: FeedForward::from_weights(
                w,
                &join(prefix, "pred_obj_score_head"),
                3,
                false,
            )?,
        })
    }

    /// `image_embedding`/`image_pe`: NHWC `[1, g, g, 256]`; `sparse`: `[1, 1, n, 256]`; `dense`: NHWC
    /// `[1, g, g, 256]`; `high_res`: `[feat_s0 (NHWC, 4g², 32), feat_s1 (NHWC, 2g², 64)]`. Returns
    /// `(masks [1, k, mg, mg], ious [1, k], obj_score [1, 1], mask_tokens_out [1, 4, 256])` with
    /// `multimask_output` selecting the 3 multimask candidates (slice 1..). The full (unsliced)
    /// `mask_tokens_out` is returned so the caller can extract the object-pointer source token.
    fn forward(
        &self,
        image_embedding: &Tensor,
        image_pe: &Tensor,
        sparse: &Tensor,
        dense: &Tensor,
        high_res: &[Tensor; 2],
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let g = image_embedding.dim(1)?;
        // output tokens: [obj_score, iou, mask_tokens(4)] → [1, 1, 6, 256]
        let out_tokens = Tensor::cat(
            &[&self.obj_score_token, &self.iou_token, &self.mask_tokens],
            0,
        )?
        .reshape((1, 1, 2 + NUM_MASK_TOKENS, HIDDEN))?;
        let n_sparse = sparse.dim(2)?;
        let sparse2 = sparse.reshape((1, 1, n_sparse, HIDDEN))?;
        let tokens = Tensor::cat(&[&out_tokens, &sparse2], 2)?.reshape((
            1,
            2 + NUM_MASK_TOKENS + n_sparse,
            HIDDEN,
        ))?;

        // image + dense, flattened to tokens [1, g*g, 256].
        let img = image_embedding.add(dense)?.reshape((1, g * g, HIDDEN))?;
        let img_pe = image_pe.reshape((1, g * g, HIDDEN))?;

        let (hs, src) = self.transformer.forward(&img, &img_pe, &tokens)?;
        let iou_token_out = take1(&hs, 1, 1)?.reshape((1, HIDDEN))?;
        let mask_tokens_out = slice_axis(&hs, 1, 2, 2 + NUM_MASK_TOKENS)?; // [1, 4, 256]
        let obj_score = self
            .obj_score_head
            .forward(&take1(&hs, 0, 1)?.reshape((1, HIDDEN))?)?; // [1, 1]

        // upscale: NHWC src [1, g, g, 256] → +feat_s1 (2g²) → +feat_s0 (4g²).
        let src = src.reshape((1, g, g, HIDDEN))?;
        let up = conv_transpose2d_nhwc(&src, &self.upscale_conv1.0, &self.upscale_conv1.1, 2)?;
        let up = up.add(&high_res[1])?;
        let up = ln(&up, &self.upscale_layer_norm)?.gelu_erf()?;
        let up2 = conv_transpose2d_nhwc(&up, &self.upscale_conv2.0, &self.upscale_conv2.1, 2)?;
        let up2 = up2.add(&high_res[0])?.gelu_erf()?; // [1, 4g, 4g, 32]
        let (mg, ch) = (up2.dim(1)?, up2.dim(3)?);

        // hypernetwork: per mask token MLP → [1, 4, 32]; mask = hyper @ upscaled.
        let hyper: Vec<Tensor> = (0..NUM_MASK_TOKENS)
            .map(|i| {
                self.hypernet[i].forward(&take1(&mask_tokens_out, i, 1)?.reshape((1, HIDDEN))?)
            })
            .collect::<Result<Vec<_>>>()?;
        let hyper = Tensor::stack(&hyper.iter().collect::<Vec<_>>(), 1)?.reshape((
            1,
            NUM_MASK_TOKENS,
            ch,
        ))?;
        let up_flat = up2
            .reshape((1, mg * mg, ch))?
            .transpose(1, 2)?
            .contiguous()?; // [1, ch, mg²]
        let masks = hyper
            .matmul(&up_flat)?
            .reshape((1, NUM_MASK_TOKENS, mg, mg))?;
        let ious = self.iou_head.forward(&iou_token_out)?; // [1, 4]

        let (masks, ious) = if multimask_output {
            (
                slice_axis(&masks, 1, 1, NUM_MASK_TOKENS)?,
                slice_axis(&ious, 1, 1, NUM_MASK_TOKENS)?,
            )
        } else {
            dynamic_multimask_via_stability(&masks, &ious)?
        };
        Ok((masks, ious, obj_score, mask_tokens_out))
    }

    /// Quantize the mask decoder's linears (the two-way transformer + the per-mask hypernetworks +
    /// the IoU / object-score heads). The token embeddings + upscale transposed-convs stay dense.
    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.transformer.quantize(quant)?;
        for h in &mut self.hypernet {
            h.quantize(quant)?;
        }
        self.iou_head.quantize(quant)?;
        self.obj_score_head.quantize(quant)
    }
}

// --- tracker neck (Sam3VisionNeck over tracker_neck.* + conv_s0/s1) ------------------------------

/// The tracker's FPN neck (same `FpnLayer` pyramid as the detector neck, separate `tracker_neck.*`
/// weights) plus the `conv_s0`/`conv_s1` high-res projections (which live under the mask decoder).
struct TrackerNeck {
    fpn_layers: Vec<FpnLayer>,
    conv_s0: (Tensor, Tensor), // 1×1 256→32
    conv_s1: (Tensor, Tensor), // 1×1 256→64
}

impl TrackerNeck {
    fn from_weights(
        w: &Weights,
        neck_prefix: &str,
        decoder_prefix: &str,
        cfg: &Sam3VisionConfig,
    ) -> Result<Self> {
        let fpn_layers = cfg
            .scale_factors
            .iter()
            .enumerate()
            .map(|(i, &scale)| {
                FpnLayer::load(w, &join(neck_prefix, &format!("fpn_layers.{i}")), scale)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fpn_layers,
            conv_s0: weight_bias(w, &join(decoder_prefix, "conv_s0"))?,
            conv_s1: weight_bias(w, &join(decoder_prefix, "conv_s1"))?,
        })
    }

    /// `backbone`: NHWC `[1, g0, g0, C]` PE features. Returns `(image_embedding [1, g, g, 256],
    /// high_res [feat_s0 (4g²,32), feat_s1 (2g²,64)])`. The coarsest FPN level (36²) is dropped.
    fn forward(&self, backbone: &Tensor) -> Result<(Tensor, [Tensor; 2])> {
        let fpn: Vec<Tensor> = self
            .fpn_layers
            .iter()
            .map(|l| l.forward(backbone))
            .collect::<Result<Vec<_>>>()?; // [288²,144²,72²,36²]
        let feat_s0 = conv2d_nhwc(&fpn[0], &self.conv_s0.0, Some(&self.conv_s0.1), 1, 0)?; // 288², 32
        let feat_s1 = conv2d_nhwc(&fpn[1], &self.conv_s1.0, Some(&self.conv_s1.1), 1, 0)?; // 144², 64
        let image_embedding = fpn[2].clone(); // 72², 256
        Ok((image_embedding, [feat_s0, feat_s1]))
    }
}

// --- memory encoder (Sam3TrackerVideoMemoryEncoder, F2) ------------------------------------------

const SIGMOID_SCALE_FOR_MEM: f32 = 20.0;
const SIGMOID_BIAS_FOR_MEM: f32 = -10.0;
const MEM_OUT_CHANNELS: usize = 64; // memory_encoder_output_channels
const MEM_POS_FEATS: usize = MEM_OUT_CHANNELS / 2; // PositionEmbeddingSine num_position_features (32)
const MEM_SINE_TEMPERATURE: f32 = 10000.0;
const NUM_MASKMEM: usize = 7; // num_maskmem (memory bank depth)

/// Separable bilinear-resize weight matrix `W` `[out, in]` for `align_corners=False`
/// (`out = W @ in @ Wᵀ`). Matches `torch.nn.functional.interpolate(mode="bilinear")`.
fn bilinear_resize_matrix(in_size: usize, out_size: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; out_size * in_size];
    let scale = in_size as f32 / out_size as f32;
    for o in 0..out_size {
        let src = ((o as f32 + 0.5) * scale - 0.5).clamp(0.0, (in_size - 1) as f32);
        let x0 = src.floor() as usize;
        let x1 = (x0 + 1).min(in_size - 1);
        let frac = src - x0 as f32;
        data[o * in_size + x0] += 1.0 - frac;
        data[o * in_size + x1] += frac;
    }
    Ok(Tensor::from_vec(data, (out_size, in_size), device)?)
}

/// `PositionEmbeddingSine(num_position_features=N, normalize=True)` over a `g×g` grid → NHWC
/// `[1, g, g, 2N]`. Channel layout is `cat(pos_y[N], pos_x[N])`; within each half the `2k`/`2k+1` pair
/// is `(sin, cos)` at frequency `10000^(k/(N/2))`. `N=32` is the memory encoder's `maskmem_pos_enc`;
/// `N=128` is the neck's `current_vision_pos`.
fn position_embedding_sine(g: usize, num_pos: usize, device: &Device) -> Result<Tensor> {
    let half = num_pos / 2;
    let scale = 2.0 * PI as f32;
    let eps = 1e-6f32;
    let denom = g as f32 + eps;
    let freqs: Vec<f32> = (0..half)
        .map(|k| MEM_SINE_TEMPERATURE.powf((2.0 * k as f32) / num_pos as f32))
        .collect();
    let ch = 2 * num_pos; // 64
    let mut data = vec![0f32; g * g * ch];
    for y in 0..g {
        let ye = (y as f32 + 1.0) / denom * scale;
        for x in 0..g {
            let xe = (x as f32 + 1.0) / denom * scale;
            let base = (y * g + x) * ch;
            for k in 0..half {
                let (py, px) = (ye / freqs[k], xe / freqs[k]);
                data[base + 2 * k] = py.sin();
                data[base + 2 * k + 1] = py.cos();
                data[base + num_pos + 2 * k] = px.sin();
                data[base + num_pos + 2 * k + 1] = px.cos();
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, g, g, ch), device)?)
}

/// `MaskDownSampler`: 4× (k3/s2/p1 conv → channels-first LayerNorm → GELU), channels 1→4→16→64→256,
/// then a 1×1 `final_conv` (256→256). Shrinks `[1,1152,1152,1]` → `[1,72,72,256]`. NHWC.
struct MaskDownSampler {
    layers: Vec<((Tensor, Tensor), (Tensor, Tensor))>, // (conv (w,bias), layer_norm (w,b))
    final_conv: (Tensor, Tensor),
}

impl MaskDownSampler {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let layers = (0..4)
            .map(|i| -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
                let lp = join(prefix, &format!("layers.{i}"));
                Ok((
                    weight_bias(w, &join(&lp, "conv"))?,
                    weight_bias(w, &join(&lp, "layer_norm"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            final_conv: weight_bias(w, &join(prefix, "final_conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for ((cw, cb), norm) in &self.layers {
            x = conv2d_nhwc(&x, cw, Some(cb), 2, 1)?;
            x = ln(&x, norm)?.gelu_erf()?;
        }
        conv2d_nhwc(&x, &self.final_conv.0, Some(&self.final_conv.1), 1, 0)
    }
}

/// `MemoryFuserCXBlock`: ConvNeXt-style residual — 7×7 depthwise conv → channels-first LayerNorm →
/// 1×1 expand (256→1024) → GELU → 1×1 project (1024→256) → per-channel `scale` → +input. NHWC, so the
/// pointwise convs are last-axis linears with no permute.
struct CxBlock {
    depthwise: (Tensor, Tensor), // OIHW [256,1,7,7], depthwise (groups=256)
    norm: (Tensor, Tensor),
    pw1: Linear,
    pw2: Linear,
    scale: Tensor, // [256]
}

impl CxBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            depthwise: weight_bias(w, &join(prefix, "depthwise_conv"))?,
            norm: weight_bias(w, &join(prefix, "layer_norm"))?,
            pw1: Linear::load(w, &join(prefix, "pointwise_conv1"))?,
            pw2: Linear::load(w, &join(prefix, "pointwise_conv2"))?,
            scale: w.require(&join(prefix, "scale"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = conv2d_nhwc_grouped(x, &self.depthwise.0, Some(&self.depthwise.1), 1, 3, HIDDEN)?;
        let h = ln(&h, &self.norm)?;
        let h = self.pw2.forward(&self.pw1.forward(&h)?.gelu_erf()?)?;
        let h = h.broadcast_mul(&self.scale)?;
        Ok(x.add(&h)?)
    }
}

/// A frame's encoded memory: the 64-channel spatial feature map + its sine position encoding, both
/// NHWC `[1, 72, 72, 64]` (f32).
pub struct MemoryFeatures {
    pub features: Tensor,
    pub pos: Tensor,
}

/// `Sam3TrackerVideoMemoryEncoder`: `mask_downsampler` + `feature_projection` (1×1 256→256) +
/// `memory_fuser` (2 CXBlocks) + `projection` (1×1 256→64).
struct MemoryEncoder {
    mask_downsampler: MaskDownSampler,
    feature_projection: (Tensor, Tensor), // 1×1 256→256
    fuser: Vec<CxBlock>,
    projection: (Tensor, Tensor), // 1×1 256→64
}

impl MemoryEncoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mask_downsampler: MaskDownSampler::from_weights(w, &join(prefix, "mask_downsampler"))?,
            feature_projection: weight_bias(w, &join(prefix, "feature_projection"))?,
            fuser: vec![
                CxBlock::from_weights(w, &join(prefix, "memory_fuser.layers.0"))?,
                CxBlock::from_weights(w, &join(prefix, "memory_fuser.layers.1"))?,
            ],
            projection: weight_bias(w, &join(prefix, "projection"))?,
        })
    }

    /// `pix_feat`: NHWC `[1,72,72,256]` raw image embedding; `mask_for_mem`: NHWC `[1,1152,1152,1]`
    /// scaled mask. Returns `(features [1,72,72,64], pos_enc [1,72,72,64])`.
    fn forward(&self, pix_feat: &Tensor, mask_for_mem: &Tensor) -> Result<(Tensor, Tensor)> {
        let masks = self.mask_downsampler.forward(mask_for_mem)?; // [1,72,72,256]
        let mut x = conv2d_nhwc(
            pix_feat,
            &self.feature_projection.0,
            Some(&self.feature_projection.1),
            1,
            0,
        )?;
        x = x.add(&masks)?;
        for layer in &self.fuser {
            x = layer.forward(&x)?;
        }
        let features = conv2d_nhwc(&x, &self.projection.0, Some(&self.projection.1), 1, 0)?; // [1,72,72,64]
        let pos = position_embedding_sine(features.dim(1)?, MEM_POS_FEATS, features.device())?;
        Ok((features, pos))
    }
}

// --- memory attention (Sam3TrackerVideoMemoryAttention, F2) --------------------------------------

const MEM_ATTN_LAYERS: usize = 4;
const MEM_ATTN_HEADS: usize = 1;
const MEM_ATTN_HEAD_DIM: usize = HIDDEN / MEM_ATTN_HEADS; // 256
const ROPE_THETA: f32 = 10000.0;

/// `VisionRotaryEmbedding.create_inv_freq` → `(cos, sin)` tables `[g·g, 256]` for axial 2-D RoPE.
fn build_rope_tables(grid: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let dim = MEM_ATTN_HEAD_DIM; // 256
    let nf = dim / 4; // 64 frequencies per axis
    let freqs: Vec<f32> = (0..nf)
        .map(|j| ROPE_THETA.powf(-(4.0 * j as f32) / dim as f32))
        .collect();
    let seq = grid * grid;
    let (mut cosd, mut sind) = (vec![0f32; seq * dim], vec![0f32; seq * dim]);
    for p in 0..seq {
        let x = (p % grid) as f32;
        let y = (p / grid) as f32;
        for (j, &f) in freqs.iter().enumerate() {
            let (fx, fy) = (x * f, y * f);
            let bx = 2 * j;
            let by = 2 * (nf + j);
            for (pos, ang) in [(bx, fx), (by, fy)] {
                let (c, s) = (ang.cos(), ang.sin());
                cosd[p * dim + pos] = c;
                cosd[p * dim + pos + 1] = c;
                sind[p * dim + pos] = s;
                sind[p * dim + pos + 1] = s;
            }
        }
    }
    Ok((
        Tensor::from_vec(cosd, (seq, dim), device)?,
        Tensor::from_vec(sind, (seq, dim), device)?,
    ))
}

/// `rotate_pairwise`: pairs `(a, b)` → `(−b, a)` along the last (head_dim) axis.
fn rotate_pairwise(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let d = *dims.last().unwrap();
    let mut paired = dims.clone();
    let last = paired.len() - 1;
    paired[last] = d / 2;
    paired.push(2);
    let xr = x.reshape(paired)?;
    let axis = xr.rank() - 1;
    let x1 = xr.narrow(axis, 0, 1)?; // even lanes
    let x2 = xr.narrow(axis, 1, 1)?; // odd lanes
    let stacked = Tensor::cat(&[&x2.neg()?, &x1], axis)?;
    Ok(stacked.reshape(dims)?)
}

/// `apply_rotary_pos_emb_2d`: rotate all of `q`; rotate the leading `seq_k − num_k_exclude` keys
/// (object-pointer tokens pass through). `q`/`k`: `[1, heads, seq, head_dim]`; `cos`/`sin`:
/// `[seq, head_dim]`. `repeat_freqs_k` tiles the tables when the key length is a multiple of `q`.
fn apply_rope_2d(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    num_k_exclude: usize,
    repeat_freqs_k: bool,
) -> Result<(Tensor, Tensor)> {
    let q_embed = q
        .broadcast_mul(cos)?
        .add(&rotate_pairwise(q)?.broadcast_mul(sin)?)?;
    let seq_k = k.dim(2)?;
    let n_rot = seq_k - num_k_exclude;
    let k_rot = slice_axis(k, 2, 0, n_rot)?;
    let q_seq = q.dim(2)?;
    let (cos_k, sin_k) = if repeat_freqs_k && n_rot != q_seq {
        if q_seq == 0 || !n_rot.is_multiple_of(q_seq) {
            return Err(CandleError::Msg(format!(
                "sam3 tracker: rope key length {n_rot} is not a multiple of query length {q_seq}"
            )));
        }
        let rf = n_rot / q_seq;
        let cos_rep: Vec<&Tensor> = (0..rf).map(|_| cos).collect();
        let sin_rep: Vec<&Tensor> = (0..rf).map(|_| sin).collect();
        (Tensor::cat(&cos_rep, 0)?, Tensor::cat(&sin_rep, 0)?)
    } else {
        (cos.clone(), sin.clone())
    };
    let k_embed = k_rot
        .broadcast_mul(&cos_k)?
        .add(&rotate_pairwise(&k_rot)?.broadcast_mul(&sin_k)?)?;
    let k_out = if num_k_exclude > 0 {
        let k_pass = slice_axis(k, 2, n_rot, seq_k)?;
        Tensor::cat(&[&k_embed, &k_pass], 2)?
    } else {
        k_embed
    };
    Ok((q_embed, k_out))
}

/// `RoPEAttention`: q/k/v project to `internal = 256` (downsample 1), split into `MEM_ATTN_HEADS`,
/// axial RoPE on q + the rotated keys, SDPA, then `o_proj`. `kv_in_dim` is 256 (self-attn) or 64
/// (cross-attn over the memory bank).
struct RoPEAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    rope_k_repeat: bool,
}

impl RoPEAttention {
    fn from_weights(w: &Weights, prefix: &str, rope_k_repeat: bool) -> Result<Self> {
        let l = |n: &str| Linear::load(w, &join(prefix, n));
        Ok(Self {
            q: l("q_proj")?,
            k: l("k_proj")?,
            v: l("v_proj")?,
            o: l("o_proj")?,
            rope_k_repeat,
        })
    }

    /// `query`: `[1, seq, 256]`; `key`/`value`: `[1, seq_k, kv_in]`. Returns `[1, seq, 256]`.
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        num_k_exclude: usize,
    ) -> Result<Tensor> {
        let to_heads = |x: &Tensor| -> Result<Tensor> {
            let (b, n, _) = x.dims3()?;
            Ok(x.reshape((b, n, MEM_ATTN_HEADS, MEM_ATTN_HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()?) // [1, heads, seq, head_dim]
        };
        let q = to_heads(&self.q.forward(query)?)?;
        let k = to_heads(&self.k.forward(key)?)?;
        let v = to_heads(&self.v.forward(value)?)?;
        let (q, k) = apply_rope_2d(&q, &k, cos, sin, num_k_exclude, self.rope_k_repeat)?;
        let scale = 1.0 / (MEM_ATTN_HEAD_DIM as f64).sqrt();
        let out = sdpa(&q, &k, &v, scale)?;
        let (b, _, n, _) = out.dims4()?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((
            b,
            n,
            MEM_ATTN_HEADS * MEM_ATTN_HEAD_DIM,
        ))?;
        self.o.forward(&out)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.q.quantize(quant)?;
        self.k.quantize(quant)?;
        self.v.quantize(quant)?;
        self.o.quantize(quant)
    }
}

/// One memory-attention layer: pre-norm self-attn → pre-norm cross-attn over `keys + key_pos` →
/// pre-norm FFN (linear1 → relu → linear2), each a residual add.
struct MemAttnLayer {
    self_attn: RoPEAttention,
    cross_attn: RoPEAttention,
    norm1: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
    norm3: (Tensor, Tensor),
    linear1: Linear,
    linear2: Linear,
}

impl MemAttnLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            self_attn: RoPEAttention::from_weights(w, &join(prefix, "self_attn"), false)?,
            cross_attn: RoPEAttention::from_weights(w, &join(prefix, "cross_attn_image"), true)?,
            norm1: weight_bias(w, &join(prefix, "layer_norm1"))?,
            norm2: weight_bias(w, &join(prefix, "layer_norm2"))?,
            norm3: weight_bias(w, &join(prefix, "layer_norm3"))?,
            linear1: Linear::load(w, &join(prefix, "linear1"))?,
            linear2: Linear::load(w, &join(prefix, "linear2"))?,
        })
    }

    /// `queries`: `[1, seq, 256]`; `keys`/`key_pos`: `[1, seq_k, 64]`.
    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        key_pos: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        num_k_exclude: usize,
    ) -> Result<Tensor> {
        let q = ln(queries, &self.norm1)?;
        let q = self.self_attn.forward(&q, &q, &q, cos, sin, 0)?;
        let queries = queries.add(&q)?;
        let q = ln(&queries, &self.norm2)?;
        let key = keys.add(key_pos)?;
        let q = self
            .cross_attn
            .forward(&q, &key, keys, cos, sin, num_k_exclude)?;
        let queries = queries.add(&q)?;
        let q = ln(&queries, &self.norm3)?;
        let q = self.linear2.forward(&self.linear1.forward(&q)?.relu()?)?;
        Ok(queries.add(&q)?)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.self_attn.quantize(quant)?;
        self.cross_attn.quantize(quant)?;
        self.linear1.quantize(quant)?;
        self.linear2.quantize(quant)
    }
}

/// `Sam3TrackerVideoMemoryAttention`: 4 layers + a final LayerNorm, over precomputed RoPE tables.
struct MemoryAttention {
    layers: Vec<MemAttnLayer>,
    norm: (Tensor, Tensor),
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl MemoryAttention {
    fn from_weights(w: &Weights, prefix: &str, grid: usize, device: &Device) -> Result<Self> {
        let layers = (0..MEM_ATTN_LAYERS)
            .map(|i| MemAttnLayer::from_weights(w, &join(prefix, &format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let (rope_cos, rope_sin) = build_rope_tables(grid, device)?;
        Ok(Self {
            layers,
            norm: weight_bias(w, &join(prefix, "layer_norm"))?,
            rope_cos,
            rope_sin,
        })
    }

    /// `current_vision_features`/`current_vision_pos`: seq-first `[seq, 1, 256]`; `memory`/`memory_pos`:
    /// seq-first `[seq_k, 1, 64]`. Returns the conditioned features batch-first `[1, seq, 256]`.
    fn forward(
        &self,
        current_vision_features: &Tensor,
        current_vision_pos: &Tensor,
        memory: &Tensor,
        memory_pos: &Tensor,
        num_object_pointer_tokens: usize,
    ) -> Result<Tensor> {
        let output = current_vision_features.add(&current_vision_pos.affine(0.1, 0.0)?)?;
        let mut output = output.transpose(1, 0)?.contiguous()?; // [1, seq, 256]
        let mem = memory.transpose(1, 0)?.contiguous()?; // [1, seq_k, 64]
        let mem_pos = memory_pos.transpose(1, 0)?.contiguous()?;
        for layer in &self.layers {
            output = layer.forward(
                &output,
                &mem,
                &mem_pos,
                &self.rope_cos,
                &self.rope_sin,
                num_object_pointer_tokens,
            )?;
        }
        ln(&output, &self.norm)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(quant)?;
        }
        Ok(())
    }
}

// --- single-frame tracker -----------------------------------------------------------------------

/// SAM3 single-frame box-prompt tracker (PVS path) + the video memory primitives. Reuses the shared
/// PE [`Backbone`]; loads the tracker neck + prompt encoder + mask decoder from `tracker_neck.*` /
/// `tracker_model.*`.
pub struct Sam3Tracker {
    backbone: Arc<Backbone>,
    neck: TrackerNeck,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    /// `tracker_model.shared_image_embedding` — a **separate** random-Gaussian table from the prompt
    /// encoder's `shared_embedding`; supplies the dense image positional encoding.
    image_pe_embed: PositionEmbeddingRandom,
    /// `tracker_model.no_memory_embedding` `[1, 1, 256]` — the learned "no memory yet" bias added to
    /// the image embedding on a frame with no memory conditioning.
    no_memory_embedding: Tensor,
    memory_encoder: MemoryEncoder,
    /// `tracker_model.occlusion_spatial_embedding_parameter` `[1, 64]`.
    occlusion: Tensor,
    memory_attention: MemoryAttention,
    /// `tracker_model.object_pointer_proj` — 3-layer FeedForward (256→256).
    object_pointer_proj: FeedForward,
    /// `tracker_model.no_object_pointer` `[1, 256]`.
    no_object_pointer: Tensor,
    /// `tracker_model.memory_temporal_positional_encoding` `[7, 1, 1, 64]`.
    mem_temporal_pos_enc: Tensor,
    /// `tracker_model.temporal_positional_encoding_projection_layer` — Linear 256→64.
    tpos_proj: Linear,
    /// `tracker_model.mask_downsample` — a single conv (1→1, k4s4).
    mask_downsample: (Tensor, Tensor),
    device: Device,
}

const MASK_MEM_SIZE: usize = 1152; // mask_input_size (4·72) · 4

/// A no-prompt (memory-conditioned) tracking-frame prediction: low-res (288²) and high-res (1008²)
/// mask logits, the object pointer stored in the memory bank, and the object-score logit.
pub struct TrackerFrameOutput {
    /// Low-res mask logits `[1, 1, 288, 288]`.
    pub low_res: Tensor,
    /// High-res mask logits `[1, 1, 1008, 1008]` (bilinear-upsampled; fed to the memory encoder).
    pub high_res: Tensor,
    /// Object pointer `[1, 256]`.
    pub object_pointer: Tensor,
    /// Object-score logit (`> 0` ⇒ object present).
    pub object_score: f32,
}

/// A single-frame tracker prediction: the best (argmax-IoU) low-res mask logits + its IoU + the
/// object-score logit.
pub struct TrackerMask {
    /// Low-res mask logits `[mg, mg]` (f32) for the best candidate.
    pub low_res: Tensor,
    pub iou: f32,
    pub object_score: f32,
}

impl Sam3Tracker {
    /// Load from a `facebook/sam3` weight map (PE backbone + `tracker_neck` + `tracker_model`).
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let cfg = Sam3VisionConfig::sam3();
        let backbone = Arc::new(Backbone::from_weights(
            w,
            "detector_model.vision_encoder.backbone",
            &cfg,
        )?);
        Self::from_weights_with_backbone(w, backbone)
    }

    /// Load the tracker reusing an already-loaded (and possibly shared) PE [`Backbone`] (F-028).
    pub(crate) fn from_weights_with_backbone(w: &Weights, backbone: Arc<Backbone>) -> Result<Self> {
        let cfg = Sam3VisionConfig::sam3();
        let device = w
            .require("tracker_model.no_memory_embedding")?
            .device()
            .clone();
        Ok(Self {
            backbone,
            neck: TrackerNeck::from_weights(w, "tracker_neck", "tracker_model.mask_decoder", &cfg)?,
            prompt: PromptEncoder::from_weights(w, "tracker_model.prompt_encoder")?,
            decoder: MaskDecoder::from_weights(w, "tracker_model.mask_decoder")?,
            image_pe_embed: PositionEmbeddingRandom {
                gaussian: w.require("tracker_model.shared_image_embedding.positional_embedding")?,
            },
            no_memory_embedding: w.require("tracker_model.no_memory_embedding")?,
            memory_encoder: MemoryEncoder::from_weights(w, "tracker_model.memory_encoder")?,
            occlusion: w.require("tracker_model.occlusion_spatial_embedding_parameter")?,
            memory_attention: MemoryAttention::from_weights(
                w,
                "tracker_model.memory_attention",
                (INPUT_SIZE as usize) / 14, // 72² RoPE grid
                &device,
            )?,
            object_pointer_proj: FeedForward::from_weights(
                w,
                "tracker_model.object_pointer_proj",
                3,
                false,
            )?,
            no_object_pointer: w.require("tracker_model.no_object_pointer")?,
            mem_temporal_pos_enc: w.require("tracker_model.memory_temporal_positional_encoding")?,
            tpos_proj: Linear::load(
                w,
                "tracker_model.temporal_positional_encoding_projection_layer",
            )?,
            mask_downsample: weight_bias(w, "tracker_model.mask_downsample")?,
            device,
        })
    }

    /// The shared PE [`Backbone`] handle (clone of the `Arc`) — exercised by the F-028 shared-backbone
    /// parity check (the segmenter and tracker must point at one backbone).
    #[cfg(test)]
    pub(crate) fn backbone_arc(&self) -> Arc<Backbone> {
        self.backbone.clone()
    }

    /// Replace the PE backbone with a (typically once-quantized, shared) one — the video model
    /// quantizes the backbone once via the segmenter and reinstalls it here (F-028).
    pub(crate) fn set_backbone(&mut self, backbone: Arc<Backbone>) {
        self.backbone = backbone;
    }

    /// Affine-quantize the whole single-frame tracker to Q4/Q8 (the shared PE backbone + the heads).
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        Arc::make_mut(&mut self.backbone).quantize(quant)?;
        self.quantize_heads(quant)
    }

    /// Quantize everything **except** the PE backbone — the mask decoder, the memory attention, the
    /// object-pointer projection, and the temporal-pos projection. The prompt encoder, memory encoder,
    /// and tracker neck (all convs/embeddings) stay dense. The video model calls this after the single
    /// shared backbone has been quantized once and reinstalled (F-028).
    pub(crate) fn quantize_heads(&mut self, quant: Quant) -> Result<()> {
        self.decoder.quantize(quant)?;
        self.memory_attention.quantize(quant)?;
        self.object_pointer_proj.quantize(quant)?;
        self.tpos_proj.quantize(quant)
    }

    /// The axial 2-D RoPE `(cos, sin)` tables `[72², 256]` the memory attention uses (exposed for
    /// parity validation against the reference `VisionRotaryEmbedding`).
    pub fn memory_attention_rope_tables(&self) -> (Tensor, Tensor) {
        (
            self.memory_attention.rope_cos.clone(),
            self.memory_attention.rope_sin.clone(),
        )
    }

    /// Memory-conditioned features (`_prepare_memory_conditioned_features` non-init branch): fuse a
    /// frame's vision features with the assembled memory bank via memory attention.
    pub fn condition_with_memory(
        &self,
        current_vision_features: &Tensor,
        current_vision_pos: &Tensor,
        memory: &Tensor,
        memory_pos: &Tensor,
        num_object_pointer_tokens: usize,
    ) -> Result<Tensor> {
        self.memory_attention.forward(
            current_vision_features,
            current_vision_pos,
            memory,
            memory_pos,
            num_object_pointer_tokens,
        )
    }

    /// `_single_frame_forward` object-pointer tail: project the SAM mask output token through
    /// `object_pointer_proj`, gated by the object-appearing flag. Returns `[1, 256]`.
    pub fn compute_object_pointer(&self, mask_token: &Tensor, object_score: f32) -> Result<Tensor> {
        let token = mask_token.reshape((1, HIDDEN))?;
        if object_score > 0.0 {
            self.object_pointer_proj.forward(&token)
        } else {
            Ok(self.no_object_pointer.reshape((1, HIDDEN))?)
        }
    }

    /// `_prepare_memory_conditioned_features` (non-init branch) — assemble the per-object memory bank
    /// and fuse it with the current frame's vision features. Returns the conditioned feature map NHWC
    /// `[1, g, g, 256]`.
    pub fn prepare_memory_conditioned_features(
        &self,
        current_vision_features: &Tensor,
        current_vision_pos: &Tensor,
        spatial_mem: &[(i32, Tensor, Tensor)],
        object_pointers: &[(i32, Tensor)],
        max_object_pointers_to_use: i32,
    ) -> Result<Tensor> {
        let seq = current_vision_features.dim(0)?;
        let g = (seq as f64).sqrt().round() as usize;
        if g * g != seq {
            return Err(CandleError::Msg(format!(
                "sam3 tracker: vision feature length {seq} is not a perfect square (g={g})"
            )));
        }

        // Spatial memory: concat each gathered frame's features + (pos + temporal-pos[offset−1]).
        let mut mem_feats: Vec<Tensor> = Vec::new();
        let mut mem_pos: Vec<Tensor> = Vec::new();
        for (offset, feat, pos) in spatial_mem {
            let idx = (offset - 1).rem_euclid(NUM_MASKMEM as i32) as usize; // offset 0 → 6
            let tpos =
                take1(&self.mem_temporal_pos_enc, idx, 0)?.reshape((1, 1, MEM_OUT_CHANNELS))?;
            mem_feats.push(feat.clone());
            mem_pos.push(pos.broadcast_add(&tpos)?);
        }

        // Object pointers: sine temporal PE → project → split 256 into 4×64 memory tokens.
        let mut num_object_pointer_tokens = 0usize;
        if !object_pointers.is_empty() {
            let num_splits = HIDDEN / MEM_OUT_CHANNELS; // 4
            let max_temporal_diff = (max_object_pointers_to_use - 1).max(1) as f32;
            let offsets: Vec<f32> = object_pointers
                .iter()
                .map(|(o, _)| *o as f32 / max_temporal_diff)
                .collect();
            let sine_pe = sine_pe_1d(&offsets, HIDDEN, MEM_SINE_TEMPERATURE, &self.device)?; // [P, 256]
            let proj = self.tpos_proj.forward(&sine_pe)?; // [P, 64]
            let p = object_pointers.len();
            let rows: Vec<Tensor> = object_pointers
                .iter()
                .map(|(_, t)| -> Result<Tensor> { Ok(t.reshape((1, HIDDEN))?) })
                .collect::<Result<Vec<_>>>()?;
            let stacked = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?; // [P, 256]
            let split = stacked.reshape((p * num_splits, 1, MEM_OUT_CHANNELS))?; // [P·4, 1, 64]
                                                                                 // pos embed [P, 64] → [P, 1, 1, 64] → broadcast(4) → [P·4, 1, 64].
            let pe = proj
                .reshape((p, 1, 1, MEM_OUT_CHANNELS))?
                .broadcast_as((p, num_splits, 1, MEM_OUT_CHANNELS))?
                .contiguous()?
                .reshape((p * num_splits, 1, MEM_OUT_CHANNELS))?;
            mem_feats.push(split);
            mem_pos.push(pe);
            num_object_pointer_tokens = p * num_splits;
        }

        let combined_memory = Tensor::cat(&mem_feats.iter().collect::<Vec<_>>(), 0)?;
        let combined_pos = Tensor::cat(&mem_pos.iter().collect::<Vec<_>>(), 0)?;
        let conditioned = self.condition_with_memory(
            current_vision_features,
            current_vision_pos,
            &combined_memory,
            &combined_pos,
            num_object_pointer_tokens,
        )?; // [1, seq, 256] batch-first
        Ok(conditioned.reshape((1, g, g, HIDDEN))?)
    }

    /// Mask prep for `_encode_new_memory`: resize the image-resolution mask logits to 1152² (separable
    /// bilinear), then `sigmoid` (or `>0` binarize for point/box frames), then `·20 −10`. Returns NHWC
    /// `[1, 1152, 1152, 1]`.
    pub fn prepare_mask_for_mem(
        &self,
        pred_high_res: &Tensor,
        is_mask_from_pts: bool,
    ) -> Result<Tensor> {
        let dims = pred_high_res.dims();
        let (in_h, in_w) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        let m = pred_high_res.reshape((in_h, in_w))?;
        let resized = if in_h == MASK_MEM_SIZE && in_w == MASK_MEM_SIZE {
            m
        } else {
            let wh = bilinear_resize_matrix(in_h, MASK_MEM_SIZE, &self.device)?;
            let ww = bilinear_resize_matrix(in_w, MASK_MEM_SIZE, &self.device)?;
            wh.matmul(&m)?.matmul(&ww.t()?.contiguous()?)?
        };
        let prob = if is_mask_from_pts {
            resized.gt(0f64)?.to_dtype(DType::F32)?
        } else {
            sigmoid(&resized)?
        };
        let scaled = prob.affine(SIGMOID_SCALE_FOR_MEM as f64, SIGMOID_BIAS_FOR_MEM as f64)?;
        Ok(scaled.reshape((1, MASK_MEM_SIZE, MASK_MEM_SIZE, 1))?)
    }

    /// `_encode_new_memory`: encode a frame's raw image embedding + its predicted mask into spatial
    /// memory. `pix_feat`: NHWC `[1, 72, 72, 256]`; `pred_high_res`: `[1, 1, 1008, 1008]` mask logits.
    pub fn encode_new_memory(
        &self,
        pix_feat: &Tensor,
        pred_high_res: &Tensor,
        object_score: f32,
        is_mask_from_pts: bool,
    ) -> Result<MemoryFeatures> {
        let mask_for_mem = self.prepare_mask_for_mem(pred_high_res, is_mask_from_pts)?;
        let (mut features, pos) = self.memory_encoder.forward(pix_feat, &mask_for_mem)?;
        if object_score <= 0.0 {
            features =
                features.broadcast_add(&self.occlusion.reshape((1, 1, 1, MEM_OUT_CHANNELS))?)?;
        }
        Ok(MemoryFeatures { features, pos })
    }

    /// Encode a frame's pixels `[1, 3, 1008, 1008]` → `(image_embedding, high_res)`.
    pub fn encode_frame(&self, pixel_values: &Tensor) -> Result<(Tensor, [Tensor; 2])> {
        let backbone = self.backbone_features(pixel_values)?;
        self.encode_frame_from_features(&backbone)
    }

    /// Run **only** the shared PE backbone, returning the raw NHWC `[1, 72, 72, C]` feature map.
    pub fn backbone_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.backbone.forward(pixel_values)
    }

    /// The tracker-neck half of [`Self::encode_frame`], over already-computed backbone features.
    pub fn encode_frame_from_features(&self, features: &Tensor) -> Result<(Tensor, [Tensor; 2])> {
        self.neck.forward(features)
    }

    /// The neck's 72² sine position encoding flattened seq-first `[g², 1, 256]`.
    pub fn frame_position_encoding(&self, g: usize) -> Result<Tensor> {
        Ok(position_embedding_sine(g, HIDDEN / 2, &self.device)?.reshape((g * g, 1, HIDDEN))?)
    }

    /// Box-prompt a pre-encoded frame: `box_xyxy` in **1008-input** space → best low-res mask.
    pub fn segment_encoded(
        &self,
        image_embedding: &Tensor,
        high_res: &[Tensor; 2],
        box_xyxy_1008: [f32; 4],
    ) -> Result<TrackerMask> {
        self.segment_encoded_multimask(image_embedding, high_res, box_xyxy_1008, true)
    }

    /// Like [`Self::segment_encoded`] but choosing the mask-output policy: `true` requests the 3
    /// multimask candidates (box-prompt PVS path), `false` requests a single mask via
    /// `dynamic_multimask_via_stability` (the no-prompt video-frame decode path).
    pub fn segment_encoded_multimask(
        &self,
        image_embedding: &Tensor,
        high_res: &[Tensor; 2],
        box_xyxy_1008: [f32; 4],
        multimask: bool,
    ) -> Result<TrackerMask> {
        let g = image_embedding.dim(1)?;
        // No-memory single-frame path: add the learned no-memory bias (broadcast over the grid).
        let image_embedding =
            image_embedding.broadcast_add(&self.no_memory_embedding.reshape((1, 1, 1, HIDDEN))?)?;
        let (sparse, dense) = self.prompt.encode_box(box_xyxy_1008, g)?;
        let image_pe = self.image_pe_embed.dense_pe(g)?;
        let (masks, ious, obj_score, _mask_tokens) = self.decoder.forward(
            &image_embedding,
            &image_pe,
            &sparse,
            &dense,
            high_res,
            multimask,
        )?;
        let iv = to_vec(&ious)?;
        let best = argmax(&iv);
        let mg = masks.dim(2)?;
        let low_res = take1(&take1(&masks, 0, 0)?, best, 0)?.reshape((mg, mg))?;
        Ok(TrackerMask {
            low_res,
            iou: iv[best],
            object_score: scalar(&obj_score)?,
        })
    }

    /// Decode a no-prompt (memory-conditioned) tracking frame — `_run_single_frame_inference` with no
    /// point/mask inputs. `conditioned_embedding`: NHWC `[1, 72, 72, 256]` (already memory-conditioned).
    pub fn decode_tracked_frame(
        &self,
        conditioned_embedding: &Tensor,
        high_res: &[Tensor; 2],
    ) -> Result<TrackerFrameOutput> {
        let g = conditioned_embedding.dim(1)?;
        let (sparse, dense) = self.prompt.encode_empty_point(g)?;
        let image_pe = self.image_pe_embed.dense_pe(g)?;
        let (masks, ious, obj_score, mask_tokens) = self.decoder.forward(
            conditioned_embedding,
            &image_pe,
            &sparse,
            &dense,
            high_res,
            true,
        )?;
        let object_score = scalar(&obj_score)?;
        let iv = to_vec(&ious)?;
        let best = argmax(&iv);
        let mg = masks.dim(2)?;
        let best_mask = take1(&masks, best, 1)?.reshape((1, 1, mg, mg))?;
        let low_res = if object_score > 0.0 {
            best_mask
        } else {
            Tensor::full(NO_OBJ_SCORE, (1, 1, mg, mg), &self.device)?
        };
        // high-res: separable bilinear 288→1008 (align_corners=False), for the memory encoder.
        let big = INPUT_SIZE as usize;
        let m = low_res.reshape((mg, mg))?;
        let up = bilinear_resize_matrix(mg, big, &self.device)?;
        let high = up
            .matmul(&m)?
            .matmul(&up.t()?.contiguous()?)?
            .reshape((1, 1, big, big))?;
        // object pointer: multimask ⇒ sam_output_token is the best-IoU candidate (token best+1).
        let token = take1(&mask_tokens, best + 1, 1)?;
        let object_pointer = self.compute_object_pointer(&token, object_score)?;
        Ok(TrackerFrameOutput {
            low_res,
            high_res: high,
            object_pointer,
            object_score,
        })
    }

    /// Decode a mask-conditioned (detection-seeded) frame — `_use_mask_as_output`. `raw_embedding`:
    /// NHWC `[1, 72, 72, 256]` (raw, no `no_memory_embedding`); `mask_det`: NHWC `[1, 288, 288, 1]`
    /// binary detection mask.
    pub fn decode_mask_conditioning_frame(
        &self,
        raw_embedding: &Tensor,
        high_res: &[Tensor; 2],
        mask_det: &Tensor,
    ) -> Result<TrackerFrameOutput> {
        let g = raw_embedding.dim(1)?;
        let in_sz = mask_det.dim(1)?;
        let big = INPUT_SIZE as usize;
        // detection mask presence (drives the object score + outer pointer gate).
        let is_appearing = to_vec(mask_det)?.iter().any(|&v| v > 0.0);
        // upsample the binary mask to 1008² and turn it into ± logits.
        let md = mask_det.reshape((in_sz, in_sz))?;
        let up = bilinear_resize_matrix(in_sz, big, &self.device)?;
        let mask_big = up.matmul(&md)?.matmul(&up.t()?.contiguous()?)?; // [1008,1008]
        let high = mask_big
            .affine(SIGMOID_SCALE_FOR_MEM as f64, SIGMOID_BIAS_FOR_MEM as f64)?
            .reshape((1, 1, big, big))?;
        // mask prompt: mask_downsample (k4s4 → 252²) then bilinear up to mask_input_size 288².
        let mask_big_nhwc = mask_big.reshape((1, big, big, 1))?;
        let mds = conv2d_nhwc(
            &mask_big_nhwc,
            &self.mask_downsample.0,
            Some(&self.mask_downsample.1),
            4,
            0,
        )?; // [1,252,252,1]
        let ds = mds.dim(1)?;
        let mds2 = mds.reshape((ds, ds))?;
        let up2 = bilinear_resize_matrix(ds, MASK_INPUT_SIZE, &self.device)?;
        let mask_288 = up2
            .matmul(&mds2)?
            .matmul(&up2.t()?.contiguous()?)?
            .reshape((1, MASK_INPUT_SIZE, MASK_INPUT_SIZE, 1))?;
        let (sparse, dense) = self.prompt.encode_mask_prompt(&mask_288)?;
        // decoder on the RAW image embedding (no no_memory_embedding), multimask=true.
        let image_pe = self.image_pe_embed.dense_pe(g)?;
        let (masks, ious, obj_score, mask_tokens) =
            self.decoder
                .forward(raw_embedding, &image_pe, &sparse, &dense, high_res, true)?;
        let decoder_score = scalar(&obj_score)?;
        let iv = to_vec(&ious)?;
        let best = argmax(&iv);
        // object pointer: best-IoU token, inner-gated by the decoder score, outer-gated by presence.
        let token = take1(&mask_tokens, best + 1, 1)?;
        let inner = self.compute_object_pointer(&token, decoder_score)?;
        let object_pointer = if is_appearing {
            inner
        } else {
            self.no_object_pointer.reshape((1, HIDDEN))?
        };
        let object_score = if is_appearing {
            SIGMOID_SCALE_FOR_MEM + SIGMOID_BIAS_FOR_MEM // 10
        } else {
            SIGMOID_BIAS_FOR_MEM // −10
        };
        let _ = masks; // mask output on this path comes from the detection, not the decoder.
        let low_res = md
            .affine(SIGMOID_SCALE_FOR_MEM as f64, SIGMOID_BIAS_FOR_MEM as f64)?
            .reshape((1, 1, in_sz, in_sz))?;
        Ok(TrackerFrameOutput {
            low_res,
            high_res: high,
            object_pointer,
            object_score,
        })
    }

    /// End-to-end single-frame: pixels + box (1008-input space) → best low-res mask.
    pub fn segment(&self, pixel_values: &Tensor, box_xyxy_1008: [f32; 4]) -> Result<TrackerMask> {
        let (emb, high_res) = self.encode_frame(pixel_values)?;
        self.segment_encoded(&emb, &high_res, box_xyxy_1008)
    }
}

/// `get_1d_sine_pe`: 1-D sinusoidal positional encoding for a set of (already-normalized) positions.
/// `dim` must be even; the first `dim/2` outputs are the sines and the last `dim/2` the cosines, with
/// paired frequencies `temperature^(2·(j//2)/(dim/2))`. Returns `[P, dim]`.
fn sine_pe_1d(positions: &[f32], dim: usize, temperature: f32, device: &Device) -> Result<Tensor> {
    let pe_dim = dim / 2;
    let mut out = vec![0f32; positions.len() * dim];
    for (i, &p) in positions.iter().enumerate() {
        for j in 0..pe_dim {
            let dim_t = temperature.powf((2 * (j / 2)) as f32 / pe_dim as f32);
            let v = p / dim_t;
            out[i * dim + j] = v.sin();
            out[i * dim + pe_dim + j] = v.cos();
        }
    }
    Ok(Tensor::from_vec(out, (positions.len(), dim), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// `rotate_pairwise` maps lanes `(a, b) -> (-b, a)`; applied twice it negates.
    #[test]
    fn rotate_pairwise_squares_to_negation() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &cpu()).unwrap();
        let twice = rotate_pairwise(&rotate_pairwise(&x).unwrap()).unwrap();
        let got = twice.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![-1.0, -2.0, -3.0, -4.0]);
    }

    /// The axial RoPE table has the expected `[grid², 256]` shape, and position 0 (zero angle) is
    /// cos 1 / sin 0.
    #[test]
    fn rope_tables_shape_and_origin() {
        let (cos, sin) = build_rope_tables(72, &cpu()).unwrap();
        assert_eq!(cos.dims(), &[72 * 72, 256]);
        assert_eq!(sin.dims(), &[72 * 72, 256]);
        let c0 = cos
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let s0 = sin
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(c0.iter().all(|&c| (c - 1.0).abs() < 1e-6));
        assert!(s0.iter().all(|&s| s.abs() < 1e-6));
    }

    /// The bilinear-resize matrix rows sum to 1 (a convex combination of source pixels).
    #[test]
    fn bilinear_matrix_rows_sum_to_one() {
        let w = bilinear_resize_matrix(4, 9, &cpu()).unwrap();
        assert_eq!(w.dims(), &[9, 4]);
        let rows = w.sum(1).unwrap().to_vec1::<f32>().unwrap();
        assert!(rows.iter().all(|&r| (r - 1.0).abs() < 1e-5));
    }

    /// `sine_pe_1d` lays sines in the first half and cosines in the second; position 0 → sin 0 / cos 1.
    #[test]
    fn sine_pe_1d_splits_sin_then_cos() {
        let pe = sine_pe_1d(&[0.0], 8, 10000.0, &cpu()).unwrap();
        let v = pe.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v.len(), 8);
        assert!(v[0..4].iter().all(|&s| s.abs() < 1e-6)); // sin(0) = 0
        assert!(v[4..8].iter().all(|&c| (c - 1.0).abs() < 1e-6)); // cos(0) = 1
    }
}
