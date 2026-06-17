//! SAM3 DETR detector — candle port of `mlx-gen-sam3`'s `detr.rs` (the `Sam3DetrEncoder` /
//! `Sam3DetrDecoder` / `Sam3DotProductScoring` path), itself a port of
//! `transformers/models/sam3/modeling_sam3.py` (epic 5482, sc-6242 under sc-5062).
//!
//! Consumes the finest FPN vision feature (72²) + the projected text features (sc-6241) and produces,
//! for 200 object queries: open-vocabulary concept logits (`pred_logits`), refined boxes
//! (`pred_boxes`, xyxy ∈ [0,1]), and the global `presence_logits`. All standard attention — no
//! deformable attention, no NMS, no Hungarian. The decoder's vision cross-attention is biased by a
//! **BoxRPB** relative-position bias (log-scale-encoded box↔grid deltas), and boxes are refined
//! iteratively across the 6 layers. Token layout `[B, seq, C]` — plain row-major, no NHWC/conv (only
//! the input FPN map is flattened from NHWC). Mirrors the MLX module line-by-line (the parity
//! oracle), so the reference cosine bar (>0.999) carries over. Quantization is a later slice
//! (sc-6246), so no `quantize` here.

use std::f32::consts::{LOG2_E, PI};

use candle_gen::candle_core::{Device, Tensor, D};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::{CandleError, Result};

use crate::common::{join, layer_norm, sdpa_masked, Linear, Weights};
use crate::config::Sam3DetrConfig;

const SCALE_2PI: f64 = 2.0 * PI as f64;
const NUM_POS: usize = 128; // sine position features per axis (hidden_size / 2)

/// `dim_t[i] = temperature^(2·(i/2)/NUM_POS)` (host constant).
fn dim_t() -> Vec<f32> {
    (0..NUM_POS)
        .map(|i| 10000f32.powf(2.0 * ((i / 2) as f32) / NUM_POS as f32))
        .collect()
}

/// `sin(even)/cos(odd)` interleave of a `[.., 2k]` raw-angle tensor → `[.., 2k]`
/// (`stack(sin(x[0::2]), cos(x[1::2])).flatten`), the SAM3 sine-embedding convention. Implemented as
/// reshape→narrow→cat (the size-2 lane), avoiding a 6-D stack.
fn sincos_interleave(raw: &Tensor) -> Result<Tensor> {
    let dims = raw.dims().to_vec();
    let last = *dims.last().expect("sincos input has rank >= 1");
    let half = last / 2;
    let mut paired: Vec<usize> = dims[..dims.len() - 1].to_vec();
    paired.push(half);
    paired.push(2);
    let r = raw.reshape(paired)?; // [.., half, 2]
    let ax = r.rank() - 1;
    let even = r.narrow(ax, 0, 1)?; // [.., half, 1]
    let odd = r.narrow(ax, 1, 1)?; // [.., half, 1]
    let stacked = Tensor::cat(&[&even.sin()?, &odd.cos()?], ax)?; // [.., half, 2]
    Ok(stacked.reshape(dims)?)
}

/// `clamp(x, lo, hi)`.
fn clamp(x: &Tensor, lo: f64, hi: f64) -> Result<Tensor> {
    Ok(x.clamp(lo, hi)?)
}

/// `inverse_sigmoid(x, eps=1e-3)` = `log(clamp(x,eps,1) / clamp(1-x,eps,1))`.
fn inverse_sigmoid(x: &Tensor) -> Result<Tensor> {
    let eps = 1e-3;
    let x = clamp(x, 0.0, 1.0)?;
    let x1 = clamp(&x, eps, 1.0)?;
    let x2 = clamp(&x.affine(-1.0, 1.0)?, eps, 1.0)?; // clamp(1 - x)
    Ok(x1.broadcast_div(&x2)?.log()?)
}

/// `(cx,cy,w,h) → (x1,y1,x2,y2)` over the last axis (narrow keeps the size-1 lane, cat re-joins).
fn cxcywh_to_xyxy(b: &Tensor) -> Result<Tensor> {
    let ax = b.rank() - 1;
    let cx = b.narrow(ax, 0, 1)?;
    let cy = b.narrow(ax, 1, 1)?;
    let half_w = b.narrow(ax, 2, 1)?.affine(0.5, 0.0)?;
    let half_h = b.narrow(ax, 3, 1)?.affine(0.5, 0.0)?;
    let x1 = cx.sub(&half_w)?;
    let y1 = cy.sub(&half_h)?;
    let x2 = cx.add(&half_w)?;
    let y2 = cy.add(&half_h)?;
    Ok(Tensor::cat(&[&x1, &y1, &x2, &y2], ax)?)
}

/// Log-scale delta encoding: `d8 = d·8; sign(d8)·log2(|d8|+1)/log2(8)`.
fn log_scale(d: &Tensor) -> Result<Tensor> {
    let d8 = d.affine(8.0, 0.0)?;
    let mag = d8.abs()?.affine(1.0, 1.0)?.log()?; // ln(|d8| + 1)
    let log2 = mag.affine(LOG2_E as f64 / 3.0, 0.0)?; // ln·log2(e)/3 = log2/3 (log2(8) = 3)
    Ok(d8.sign()?.mul(&log2)?)
}

/// Generic multi-head attention (`Sam3Attention`): separate q/k/v/o, optional additive `mask` added
/// to the scores before softmax (a key-padding mask **or** the BoxRPB bias). Shared by the DETR
/// stack and the geometry encoder (sc-6244).
pub(crate) struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attn {
    pub(crate) fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        Self::from_dims(w, prefix, cfg.num_attention_heads, cfg.head_dim())
    }

    /// Construct from explicit head geometry — the geometry encoder reuses the same shape with its
    /// own (numerically identical) config.
    pub(crate) fn from_dims(
        w: &Weights,
        prefix: &str,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            q: Linear::load(w, &join(prefix, "q_proj"))?,
            k: Linear::load(w, &join(prefix, "k_proj"))?,
            v: Linear::load(w, &join(prefix, "v_proj"))?,
            o: Linear::load(w, &join(prefix, "o_proj"))?,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    pub(crate) fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, ql, _) = query.dims3()?;
        let kl = key.dim(1)?;
        let (nh, hd) = (self.num_heads, self.head_dim);
        let heads = |t: Tensor, n: usize| -> Result<Tensor> {
            Ok(t.reshape((b, n, nh, hd))?.transpose(1, 2)?.contiguous()?)
        };
        let q = heads(self.q.forward(query)?, ql)?;
        let k = heads(self.k.forward(key)?, kl)?;
        let v = heads(self.v.forward(value)?, kl)?;
        let o = sdpa_masked(&q, &k, &v, self.scale, mask)?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, ql, nh * hd))?;
        self.o.forward(&o)
    }
}

/// `Sam3MLP` (DETR enc/dec FFN): `fc1` → **ReLU** → `fc2` (`hidden_act = "relu"`). Shared with the
/// geometry encoder (sc-6244).
pub(crate) struct Ffn {
    fc1: Linear,
    fc2: Linear,
}

impl Ffn {
    pub(crate) fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(w, &join(prefix, "fc1"))?,
            fc2: Linear::load(w, &join(prefix, "fc2"))?,
        })
    }
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?.relu()?;
        self.fc2.forward(&h)
    }
}

/// `Sam3DecoderMLP`: a 2- or 3-layer ReLU MLP (relu between layers, no final activation). Layers are
/// named `layer1..layerN`.
struct DecoderMlp {
    layers: Vec<Linear>,
}

impl DecoderMlp {
    fn from_weights(w: &Weights, prefix: &str, num_layers: usize) -> Result<Self> {
        let layers = (1..=num_layers)
            .map(|i| Linear::load(w, &join(prefix, &format!("layer{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        let n = self.layers.len();
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(&h)?;
            if i + 1 < n {
                h = h.relu()?;
            }
        }
        Ok(h)
    }
}

/// `Sam3DotProductScoring`: open-vocab logit = `scale · ⟨query_proj(q), text_proj(meanpool(text))⟩`.
struct DotScoring {
    text_mlp: DecoderMlp,
    text_mlp_out_w: Tensor,
    text_mlp_out_b: Tensor,
    text_proj: Linear,
    query_proj: Linear,
    scale: f64,
    clamp: f64,
    eps: f64,
}

impl DotScoring {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        Ok(Self {
            text_mlp: DecoderMlp::from_weights(w, &join(prefix, "text_mlp"), 2)?,
            text_mlp_out_w: w.require(&join(prefix, "text_mlp_out_norm.weight"))?,
            text_mlp_out_b: w.require(&join(prefix, "text_mlp_out_norm.bias"))?,
            text_proj: Linear::load(w, &join(prefix, "text_proj"))?,
            query_proj: Linear::load(w, &join(prefix, "query_proj"))?,
            scale: 1.0 / (cfg.hidden_size as f64).sqrt(),
            clamp: cfg.score_clamp as f64,
            eps: cfg.layer_norm_eps,
        })
    }

    /// `queries`: `[1, Q, D]`; `text`: `[1, L, D]`; `text_mask`: per-token validity.
    /// Returns `pred_logits` `[1, Q]`.
    fn forward(&self, queries: &Tensor, text: &Tensor, text_mask: &[i32]) -> Result<Tensor> {
        // text_mlp residual + out-norm
        let t = self.text_mlp.forward(text)?;
        let t = t.add(text)?;
        let t = layer_norm(&t, &self.text_mlp_out_w, &self.text_mlp_out_b, self.eps)?;
        // masked mean over valid tokens. Valid positions need not be contiguous — the PVS path
        // (sc-6244) concatenates valid geometry-prompt tokens *after* the text padding, so weight by
        // the mask rather than assuming a leading valid run.
        let l = text_mask.len();
        let device = t.device();
        let isv: Vec<f32> = text_mask
            .iter()
            .map(|&m| if m == 1 { 1.0 } else { 0.0 })
            .collect();
        let n_valid = isv.iter().sum::<f32>().max(1.0) as f64;
        let is_valid = Tensor::from_vec(isv, (1, l, 1), device)?;
        let pooled = t
            .broadcast_mul(&is_valid)?
            .sum(1)?
            .affine(1.0 / n_valid, 0.0)?; // [1, D]
        let proj_text = self.text_proj.forward(&pooled)?; // [1, D]
        let d = proj_text.dim(1)?;
        let proj_q = self.query_proj.forward(queries)?; // [1, Q, D]
                                                        // ⟨q, text⟩ over D → [1, Q]
        let scores = proj_q
            .broadcast_mul(&proj_text.reshape((1, 1, d))?)?
            .sum(D::Minus1)?;
        clamp(&scores.affine(self.scale, 0.0)?, -self.clamp, self.clamp)
    }
}

/// One pre-norm DETR encoder layer: vision self-attn + text cross-attn + FFN.
struct EncoderLayer {
    ln1_w: Tensor,
    ln1_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    ln3_w: Tensor,
    ln3_b: Tensor,
    self_attn: Attn,
    cross_attn: Attn,
    ffn: Ffn,
    eps: f64,
}

impl EncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        Ok(Self {
            ln1_w: w.require(&join(prefix, "layer_norm1.weight"))?,
            ln1_b: w.require(&join(prefix, "layer_norm1.bias"))?,
            ln2_w: w.require(&join(prefix, "layer_norm2.weight"))?,
            ln2_b: w.require(&join(prefix, "layer_norm2.bias"))?,
            ln3_w: w.require(&join(prefix, "layer_norm3.weight"))?,
            ln3_b: w.require(&join(prefix, "layer_norm3.bias"))?,
            self_attn: Attn::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            cross_attn: Attn::from_weights(w, &join(prefix, "cross_attn"), cfg)?,
            ffn: Ffn::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        vis_pos: &Tensor,
        text: &Tensor,
        text_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // vision self-attention (pos added to q/k, not v)
        let h = layer_norm(x, &self.ln1_w, &self.ln1_b, self.eps)?;
        let hp = h.add(vis_pos)?;
        let a = self.self_attn.forward(&hp, &hp, &h, None)?;
        let x = x.add(&a)?;
        // text cross-attention
        let h = layer_norm(&x, &self.ln2_w, &self.ln2_b, self.eps)?;
        let a = self.cross_attn.forward(&h, text, text, text_mask)?;
        let x = x.add(&a)?;
        // FFN
        let h = layer_norm(&x, &self.ln3_w, &self.ln3_b, self.eps)?;
        let a = self.ffn.forward(&h)?;
        Ok(x.add(&a)?)
    }
}

/// One post-norm DETR decoder layer: query self-attn + text cross-attn + vision cross-attn (BoxRPB)
/// + FFN. `hidden` is `[1, 1+Q, D]` (presence token at index 0); `query_pos_padded` is `[1, 1+Q, D]`.
struct DecoderLayer {
    self_attn: Attn,
    self_ln_w: Tensor,
    self_ln_b: Tensor,
    text_attn: Attn,
    text_ln_w: Tensor,
    text_ln_b: Tensor,
    vis_attn: Attn,
    vis_ln_w: Tensor,
    vis_ln_b: Tensor,
    ffn: Ffn,
    mlp_ln_w: Tensor,
    mlp_ln_b: Tensor,
    eps: f64,
}

impl DecoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        Ok(Self {
            self_attn: Attn::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            self_ln_w: w.require(&join(prefix, "self_attn_layer_norm.weight"))?,
            self_ln_b: w.require(&join(prefix, "self_attn_layer_norm.bias"))?,
            text_attn: Attn::from_weights(w, &join(prefix, "text_cross_attn"), cfg)?,
            text_ln_w: w.require(&join(prefix, "text_cross_attn_layer_norm.weight"))?,
            text_ln_b: w.require(&join(prefix, "text_cross_attn_layer_norm.bias"))?,
            vis_attn: Attn::from_weights(w, &join(prefix, "vision_cross_attn"), cfg)?,
            vis_ln_w: w.require(&join(prefix, "vision_cross_attn_layer_norm.weight"))?,
            vis_ln_b: w.require(&join(prefix, "vision_cross_attn_layer_norm.bias"))?,
            ffn: Ffn::from_weights(w, &join(prefix, "mlp"))?,
            mlp_ln_w: w.require(&join(prefix, "mlp_layer_norm.weight"))?,
            mlp_ln_b: w.require(&join(prefix, "mlp_layer_norm.bias"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Tensor,
        query_pos_padded: &Tensor, // [1, 1+Q, D] (presence row = 0)
        text: &Tensor,
        text_mask: Option<&Tensor>,
        vision: &Tensor,
        vis_pos: &Tensor,
        rpb: &Tensor, // [1, nh, 1+Q, HW]
    ) -> Result<Tensor> {
        // self-attention
        let qp = hidden.add(query_pos_padded)?;
        let a = self.self_attn.forward(&qp, &qp, hidden, None)?;
        let x = hidden.add(&a)?;
        let x = layer_norm(&x, &self.self_ln_w, &self.self_ln_b, self.eps)?;
        // text cross-attention
        let qp = x.add(query_pos_padded)?;
        let a = self.text_attn.forward(&qp, text, text, text_mask)?;
        let x = x.add(&a)?;
        let x = layer_norm(&x, &self.text_ln_w, &self.text_ln_b, self.eps)?;
        // vision cross-attention with BoxRPB bias
        let qp = x.add(query_pos_padded)?;
        let kp = vision.add(vis_pos)?;
        let a = self.vis_attn.forward(&qp, &kp, vision, Some(rpb))?;
        let x = x.add(&a)?;
        let x = layer_norm(&x, &self.vis_ln_w, &self.vis_ln_b, self.eps)?;
        // FFN (post-norm)
        let a = self.ffn.forward(&x)?;
        let x = x.add(&a)?;
        layer_norm(&x, &self.mlp_ln_w, &self.mlp_ln_b, self.eps)
    }
}

/// The detector outputs needed downstream (the mask head, sc-6243, adds masks).
pub struct DetectorOutput {
    /// `[1, Q]` concept logits (pre-sigmoid).
    pub pred_logits: Tensor,
    /// `[1, Q, 4]` boxes in xyxy ∈ [0, 1].
    pub pred_boxes: Tensor,
    /// `[1, 1]` global presence logit.
    pub presence_logits: Tensor,
    /// `[1, Q, D]` final decoder query hidden states (output-LN'd) — the mask head consumes these.
    pub query_hidden: Tensor,
    /// `[1, H·W, D]` DETR encoder output (the encoded 72² level) — the mask head consumes this.
    pub encoder_hidden_states: Tensor,
}

/// The DETR detector head: encoder + decoder + presence + scoring. Produces concept logits, boxes,
/// and presence from the 72² FPN feature + projected text features.
pub struct Sam3Detector {
    enc_layers: Vec<EncoderLayer>,
    dec_layers: Vec<DecoderLayer>,
    output_ln_w: Tensor,
    output_ln_b: Tensor,
    box_head: DecoderMlp,
    query_embed: Tensor,      // [Q, D]
    reference_points: Tensor, // [Q, 4]
    presence_token: Tensor,   // [1, D]
    presence_head: DecoderMlp,
    presence_ln_w: Tensor,
    presence_ln_b: Tensor,
    presence_clamp: f64,
    ref_point_head: DecoderMlp,
    box_rpb_x: DecoderMlp,
    box_rpb_y: DecoderMlp,
    scoring: DotScoring,
    cfg: Sam3DetrConfig,
}

impl Sam3Detector {
    /// Load from a `facebook/sam3` weight map. `prefix` is typically `"detector_model"`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        let enc_prefix = join(prefix, "detr_encoder");
        let dec_prefix = join(prefix, "detr_decoder");
        let enc_layers = (0..cfg.num_encoder_layers)
            .map(|i| EncoderLayer::from_weights(w, &join(&enc_prefix, &format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let dec_layers = (0..cfg.num_decoder_layers)
            .map(|i| DecoderLayer::from_weights(w, &join(&dec_prefix, &format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let d = |n: &str| w.require(&join(&dec_prefix, n));
        Ok(Self {
            enc_layers,
            dec_layers,
            output_ln_w: d("output_layer_norm.weight")?,
            output_ln_b: d("output_layer_norm.bias")?,
            box_head: DecoderMlp::from_weights(w, &join(&dec_prefix, "box_head"), 3)?,
            query_embed: d("query_embed.weight")?,
            reference_points: d("reference_points.weight")?,
            presence_token: d("presence_token.weight")?,
            presence_head: DecoderMlp::from_weights(w, &join(&dec_prefix, "presence_head"), 3)?,
            presence_ln_w: d("presence_layer_norm.weight")?,
            presence_ln_b: d("presence_layer_norm.bias")?,
            presence_clamp: cfg.presence_clamp as f64,
            ref_point_head: DecoderMlp::from_weights(w, &join(&dec_prefix, "ref_point_head"), 2)?,
            box_rpb_x: DecoderMlp::from_weights(w, &join(&dec_prefix, "box_rpb_embed_x"), 2)?,
            box_rpb_y: DecoderMlp::from_weights(w, &join(&dec_prefix, "box_rpb_embed_y"), 2)?,
            scoring: DotScoring::from_weights(w, &join(prefix, "dot_product_scoring"), cfg)?,
            cfg: cfg.clone(),
        })
    }

    /// `vision_feature`: the 72² FPN feature **NHWC** `[1, H, W, 256]`; `text`: `[1, L, 256]`.
    /// `text_mask`: per-text-token validity (`1`/`0`). Returns concept logits, boxes, presence.
    pub fn forward(
        &self,
        vision_feature: &Tensor,
        text: &Tensor,
        text_mask: &[i32],
    ) -> Result<DetectorOutput> {
        let (_, h, w, _) = vision_feature.dims4()?;
        let hw = h * w;
        let d = self.cfg.hidden_size;
        let device = vision_feature.device().clone();
        let vision = vision_feature.reshape((1, hw, d))?;
        let vis_pos = sine_position_embedding_flat(h, w, d, &device)?; // [1, HW, D]
        let text_key_mask = text_key_mask(text_mask, &device)?;

        // --- encoder ---
        let mut enc = vision;
        for layer in &self.enc_layers {
            enc = layer.forward(&enc, &vis_pos, text, Some(&text_key_mask))?;
        }

        // --- decoder ---
        let q = self.cfg.num_queries;
        let query_embeds = self.query_embed.reshape((1, q, d))?;
        let presence = self.presence_token.reshape((1, 1, d))?;
        let mut hidden = Tensor::cat(&[&presence, &query_embeds], 1)?; // [1, 1+Q, D]
        let mut reference_boxes = sigmoid(&self.reference_points.reshape((1, q, 4))?)?;

        // The post-loop output reads `last_*`, only set inside the loop; a zero-decoder-layer config
        // would skip the loop and unwrap `None`. Reject it up front (F-015).
        if self.dec_layers.is_empty() {
            return Err(CandleError::Msg(
                "sam3 detr: decoder has zero layers (num_decoder_layers must be >= 1)".into(),
            ));
        }
        let mut last_query_hidden = None;
        let mut last_ref_input = None;
        let mut last_presence = None;
        let mut last_offsets = None;

        for layer in &self.dec_layers {
            // conditional query positions from the current reference boxes
            let query_sine = self.encode_boxes(&reference_boxes)?; // [1, Q, 4*128]
            let query_pos = self.ref_point_head.forward(&query_sine)?; // [1, Q, D]
            let query_pos_padded = query_pos.pad_with_zeros(1, 1, 0)?; // presence row = 0
                                                                       // BoxRPB bias, padded with a zero row for the presence query
            let rpb = self.box_rpb(&reference_boxes, h, w)?; // [1, nh, Q, HW]
            let rpb = rpb.pad_with_zeros(2, 1, 0)?; // [1, nh, 1+Q, HW]

            hidden = layer.forward(
                &hidden,
                &query_pos_padded,
                text,
                Some(&text_key_mask),
                &enc,
                &vis_pos,
                &rpb,
            )?;

            // query hidden (drop presence) → output LN
            let query_hidden = hidden.narrow(1, 1, q)?; // [1, Q, D]
            let query_hidden = layer_norm(
                &query_hidden,
                &self.output_ln_w,
                &self.output_ln_b,
                self.cfg.layer_norm_eps,
            )?;

            // record this layer's reference-box input + outputs (final layer is what we return)
            last_ref_input = Some(reference_boxes.clone());
            last_query_hidden = Some(query_hidden.clone());

            // iterative box refinement. `delta` is box_head(query_hidden) for THIS layer; on the final
            // layer it equals the post-loop `offsets`, so record + reuse it (F-070).
            let delta = self.box_head.forward(&query_hidden)?;
            last_offsets = Some(delta.clone());
            reference_boxes = sigmoid(&delta.add(&inverse_sigmoid(&reference_boxes)?)?)?;

            // presence
            let presence_hidden = hidden.narrow(1, 0, 1)?; // [1, 1, D]
            let p = layer_norm(
                &presence_hidden,
                &self.presence_ln_w,
                &self.presence_ln_b,
                self.cfg.layer_norm_eps,
            )?;
            let p = self.presence_head.forward(&p)?.reshape((1, 1))?;
            last_presence = Some(clamp(&p, -self.presence_clamp, self.presence_clamp)?);
        }

        let query_hidden = last_query_hidden.unwrap();
        let ref_input = last_ref_input.unwrap();
        let presence_logits = last_presence.unwrap();

        // final boxes: sigmoid(inv_sigmoid(ref_input) + box_head(query_hidden)) → xyxy. Reuse the
        // final layer's already-computed box_head output (F-070).
        let offsets = last_offsets.unwrap();
        let boxes_cxcywh = sigmoid(&inverse_sigmoid(&ref_input)?.add(&offsets)?)?;
        let pred_boxes = cxcywh_to_xyxy(&boxes_cxcywh)?;
        let pred_logits = self.scoring.forward(&query_hidden, text, text_mask)?;

        Ok(DetectorOutput {
            pred_logits,
            pred_boxes,
            presence_logits,
            query_hidden,
            encoder_hidden_states: enc,
        })
    }

    /// `encode_boxes` (sine box embedding): `[1, Q, 4]` cxcywh → `[1, Q, 4·128]` (pos_y, x, w, h).
    fn encode_boxes(&self, boxes: &Tensor) -> Result<Tensor> {
        let device = boxes.device();
        let dim_t = Tensor::from_vec(dim_t(), (1, 1, NUM_POS), device)?;
        let comp = |idx: usize| -> Result<Tensor> {
            let e = boxes.narrow(D::Minus1, idx, 1)?.affine(SCALE_2PI, 0.0)?; // [1, Q, 1]
            let raw = e.broadcast_div(&dim_t)?; // [1, Q, 128]
            sincos_interleave(&raw)
        };
        // reference order: cat(pos_y, pos_x, pos_w, pos_h) → indices (1, 0, 2, 3)
        let pos_y = comp(1)?;
        let pos_x = comp(0)?;
        let pos_w = comp(2)?;
        let pos_h = comp(3)?;
        Ok(Tensor::cat(&[&pos_y, &pos_x, &pos_w, &pos_h], D::Minus1)?)
    }

    /// BoxRPB relative-position bias `[1, nh, Q, H·W]` (log-scale-encoded box↔grid deltas).
    fn box_rpb(&self, reference_boxes: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let device = reference_boxes.device();
        let q = reference_boxes.dim(1)?;
        let nh = self.cfg.num_attention_heads;
        let boxes_xyxy = cxcywh_to_xyxy(reference_boxes)?; // [1, Q, 4]
        let coords_h = Tensor::from_vec(
            (0..h).map(|i| i as f32 / h as f32).collect::<Vec<f32>>(),
            (1, 1, h, 1),
            device,
        )?;
        let coords_w = Tensor::from_vec(
            (0..w).map(|i| i as f32 / w as f32).collect::<Vec<f32>>(),
            (1, 1, w, 1),
            device,
        )?;
        // y deltas from box (y1,y2) = indices [1,3]; x deltas from (x1,x2) = [0,2]
        let idx_y = Tensor::from_vec(vec![1u32, 3u32], 2, device)?;
        let idx_x = Tensor::from_vec(vec![0u32, 2u32], 2, device)?;
        let by = boxes_xyxy.index_select(&idx_y, 2)?.reshape((1, q, 1, 2))?;
        let bx = boxes_xyxy.index_select(&idx_x, 2)?.reshape((1, q, 1, 2))?;
        let dy = coords_h.broadcast_sub(&by)?; // [1, Q, H, 2]
        let dx = coords_w.broadcast_sub(&bx)?; // [1, Q, W, 2]
        let dy = self.box_rpb_y.forward(&log_scale(&dy)?)?; // [1, Q, H, nh]
        let dx = self.box_rpb_x.forward(&log_scale(&dx)?)?; // [1, Q, W, nh]
                                                            // rpb[b,q,h,w,head] = dy[b,q,h,head] + dx[b,q,w,head]
        let rpb = dy
            .reshape(vec![1, q, h, 1, nh])?
            .broadcast_add(&dx.reshape(vec![1, q, 1, w, nh])?)?; // [1, Q, H, W, nh]
        Ok(rpb
            .reshape(vec![1, q, h * w, nh])?
            .permute([0, 3, 1, 2])?
            .contiguous()?) // [1, nh, Q, HW]
    }
}

/// Build a key-padding additive mask `[1, 1, 1, L]` (0 valid, −1e9 padded), broadcast over
/// heads/queries.
fn text_key_mask(text_mask: &[i32], device: &Device) -> Result<Tensor> {
    let row: Vec<f32> = text_mask
        .iter()
        .map(|&m| if m == 1 { 0.0 } else { -1e9 })
        .collect();
    let l = row.len();
    Ok(Tensor::from_vec(row, (1, 1, 1, l), device)?)
}

/// Sine position embedding (normalize=True), flattened to `[1, H·W, D]` (host-computed for a fixed
/// grid). Mirrors `Sam3SinePositionEmbedding.build_sine_position_embedding` for the neck.
pub(crate) fn sine_position_embedding_flat(
    h: usize,
    w: usize,
    d: usize,
    device: &Device,
) -> Result<Tensor> {
    let np = d / 2; // 128
    let dt = dim_t();
    let eps = 1e-6f32;
    let mut out = vec![0f32; h * w * d];
    // sin(even)/cos(odd) interleave of v/dim_t into a 128-slice
    let fill = |buf: &mut [f32], v: f32| {
        for j in 0..np / 2 {
            buf[2 * j] = (v / dt[2 * j]).sin();
            buf[2 * j + 1] = (v / dt[2 * j + 1]).cos();
        }
    };
    for hi in 0..h {
        let yv = (hi as f32 + 1.0) / (h as f32 + eps) * SCALE_2PI as f32;
        for wi in 0..w {
            let xv = (wi as f32 + 1.0) / (w as f32 + eps) * SCALE_2PI as f32;
            let base = (hi * w + wi) * d;
            fill(&mut out[base..base + np], yv); // pos_y first
            fill(&mut out[base + np..base + 2 * np], xv); // then pos_x
        }
    }
    Ok(Tensor::from_vec(out, (1, h * w, d), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn cxcywh_to_xyxy_roundtrips_a_unit_box() {
        // center (0.5,0.5) size (0.4,0.2) → (0.3,0.4,0.7,0.6)
        let b = Tensor::from_vec(vec![0.5f32, 0.5, 0.4, 0.2], (1, 1, 4), &cpu()).unwrap();
        let v = cxcywh_to_xyxy(&b)
            .unwrap()
            .reshape(4)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (got, want) in v.iter().zip([0.3f32, 0.4, 0.7, 0.6]) {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
    }

    #[test]
    fn inverse_sigmoid_inverts_sigmoid() {
        let x = Tensor::from_vec(vec![-2.0f32, -0.5, 0.0, 1.3, 3.0], 5, &cpu()).unwrap();
        let round = inverse_sigmoid(&sigmoid(&x).unwrap()).unwrap();
        let a = x.to_vec1::<f32>().unwrap();
        let b = round.to_vec1::<f32>().unwrap();
        let drift = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(drift < 1e-3, "inverse_sigmoid∘sigmoid drift {drift}");
    }

    #[test]
    fn sine_position_embedding_has_expected_shape() {
        let p = sine_position_embedding_flat(72, 72, 256, &cpu()).unwrap();
        assert_eq!(p.dims(), &[1, 5184, 256]);
    }

    #[test]
    fn sincos_interleave_places_sin_even_cos_odd() {
        // raw all-zero → sin=0 on even lanes, cos=1 on odd lanes.
        let raw = Tensor::from_vec(vec![0f32; 8], (1, 8), &cpu()).unwrap();
        let out = sincos_interleave(&raw)
            .unwrap()
            .reshape(8)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }
}
