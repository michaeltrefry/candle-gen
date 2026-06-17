//! SAM3 text encoder — candle port of `mlx-gen-sam3`'s `text.rs` (the CLIP-H text tower
//! `Sam3Config.text_config` + the SAM3 `text_projection` 1024→256), itself a port of
//! `Sam3Model.get_text_features` (epic 5482, sc-6241 under sc-5062).
//!
//! A standard CLIP text transformer: token + learned position embeddings → 24 pre-norm layers
//! (causal **and** key-padding masked) → final LayerNorm, giving `last_hidden_state[1, N, 1024]`;
//! SAM3 then projects every token to 256 to form the prompt conditioning the DETR encoder consumes.
//! Activation is exact GELU (candle `gelu_erf`); LayerNorm eps is **1e-5** (the vision encoder's is
//! 1e-6). The tokenizer is the shipped CLIP `tokenizer.json` (lowercased word-BPE, BOS 49406 /
//! EOS 49407, padded to 32). Unlike the vision tower this is a plain `[b, seq, dim]` transformer —
//! no NHWC/conv layout — so it maps straight onto candle's row-major last-dim ops.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::{CandleError, Result};
use tokenizers::Tokenizer;

use crate::common::{join, layer_norm, sdpa_masked, Linear, Weights};
use crate::config::Sam3TextConfig;

/// One CLIP encoder layer: pre-norm self-attention + pre-norm GELU MLP, both residual.
struct ClipLayer {
    ln1_w: Tensor,
    ln1_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl ClipLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3TextConfig) -> Result<Self> {
        let l = |n: &str| Linear::load(w, &join(prefix, n));
        Ok(Self {
            ln1_w: w.require(&join(prefix, "layer_norm1.weight"))?,
            ln1_b: w.require(&join(prefix, "layer_norm1.bias"))?,
            ln2_w: w.require(&join(prefix, "layer_norm2.weight"))?,
            ln2_b: w.require(&join(prefix, "layer_norm2.bias"))?,
            q: l("self_attn.q_proj")?,
            k: l("self_attn.k_proj")?,
            v: l("self_attn.v_proj")?,
            o: l("self_attn.out_proj")?,
            fc1: l("mlp.fc1")?,
            fc2: l("mlp.fc2")?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let y = layer_norm(x, &self.ln1_w, &self.ln1_b, self.eps)?;
        let y = self.attention(&y, mask)?;
        let x = x.broadcast_add(&y)?;
        let y = layer_norm(&x, &self.ln2_w, &self.ln2_b, self.eps)?;
        let y = self.fc1.forward(&y)?.gelu_erf()?;
        let y = self.fc2.forward(&y)?;
        Ok(x.broadcast_add(&y)?)
    }

    /// Standard multi-head self-attention (no RoPE) over `[b, seq, dim]` with the additive
    /// causal+key-padding `mask` `[1, 1, seq, seq]`.
    fn attention(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, n, _c) = x.dims3()?;
        let (nh, hd) = (self.num_heads, self.head_dim);
        // [b, seq, dim] → [b, seq, nh, hd] → [b, nh, seq, hd]
        let to_heads = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((b, n, nh, hd))?.transpose(1, 2)?.contiguous()?)
        };
        let q = to_heads(self.q.forward(x)?)?;
        let k = to_heads(self.k.forward(x)?)?;
        let v = to_heads(self.v.forward(x)?)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let o = sdpa_masked(&q, &k, &v, scale, Some(mask))?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, n, nh * hd))?;
        self.o.forward(&o)
    }
}

/// SAM3 text encoder: CLIP text tower → final LayerNorm → 1024→256 projection.
pub struct Sam3TextEncoder {
    token_embedding: Tensor,
    position_embedding: Tensor,
    layers: Vec<ClipLayer>,
    final_ln_w: Tensor,
    final_ln_b: Tensor,
    proj: Linear,
    eps: f64,
}

impl Sam3TextEncoder {
    /// Load from a `facebook/sam3` weight map. `clip_prefix` is typically
    /// `"detector_model.text_encoder.text_model"`; `proj_prefix` is `"detector_model.text_projection"`.
    pub fn from_weights(
        w: &Weights,
        clip_prefix: &str,
        proj_prefix: &str,
        cfg: &Sam3TextConfig,
    ) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                ClipLayer::from_weights(w, &join(clip_prefix, &format!("encoder.layers.{i}")), cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding: w.require(&join(clip_prefix, "embeddings.token_embedding.weight"))?,
            position_embedding: w
                .require(&join(clip_prefix, "embeddings.position_embedding.weight"))?,
            layers,
            final_ln_w: w.require(&join(clip_prefix, "final_layer_norm.weight"))?,
            final_ln_b: w.require(&join(clip_prefix, "final_layer_norm.bias"))?,
            proj: Linear::load(w, proj_prefix)?,
            eps: cfg.layer_norm_eps,
        })
    }

    /// Encode `input_ids` `[1, N]` (any int/float dtype — coerced to indices) with a key-padding
    /// `attention_mask` (`1` = real token). Returns the projected text features `[1, N, 256]` (the
    /// DETR-stack conditioning).
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &[i32]) -> Result<Tensor> {
        let ids = input_ids.to_dtype(DType::U32)?;
        let (b, n) = ids.dims2()?;
        let dim = self.token_embedding.dim(1)?;
        let device = self.token_embedding.device().clone();

        // token + position embeddings
        let ids_flat = ids.reshape(b * n)?;
        let tok = self
            .token_embedding
            .index_select(&ids_flat, 0)?
            .reshape((b, n, dim))?;
        // The position-embedding table has only `max_position_embeddings` rows; the tokenizer pads to
        // that, but a direct `forward` with a longer sequence would index it out of bounds. Reject it
        // (F-019, matching the MLX port).
        let max_pos = self.position_embedding.dim(0)?;
        if n > max_pos {
            return Err(CandleError::Msg(format!(
                "sam3 text: sequence length {n} exceeds max_position_embeddings ({max_pos})"
            )));
        }
        let pos = self
            .position_embedding
            .narrow(0, 0, n)?
            .reshape((1, n, dim))?; // [1, N, D]
        let mut x = tok.broadcast_add(&pos)?;

        let mask = causal_padding_mask(n, attention_mask, &device)?;
        for layer in &self.layers {
            x = layer.forward(&x, &mask)?;
        }
        let last_hidden_state = layer_norm(&x, &self.final_ln_w, &self.final_ln_b, self.eps)?;
        // SAM3 projection: every token 1024 → 256.
        self.proj.forward(&last_hidden_state)
    }
}

/// Additive attention mask `[1, 1, N, N]` (f32): position `i` may attend to key `j` iff `j <= i`
/// (causal) **and** `attention_mask[j] == 1` (key-padding); otherwise `-1e9`. Matches HF
/// `CLIPTextTransformer` combining its causal mask with the passed `attention_mask`.
fn causal_padding_mask(n: usize, attention_mask: &[i32], device: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; n * n];
    for (i, row) in m.chunks_mut(n).enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            let padded = attention_mask.get(j).copied().unwrap_or(1) == 0;
            if j > i || padded {
                *slot = -1.0e9;
            }
        }
    }
    Ok(Tensor::from_vec(m, (1, 1, n, n), device)?)
}

/// CLIP tokenizer for SAM3 concept prompts — the shipped `tokenizer.json` (lowercased word-BPE,
/// BOS 49406 / EOS 49407), padded to `max_position_embeddings` (32) with the EOS/pad token. Drives
/// the HF [`tokenizers`] crate directly (the candle convention; no gen-core tokenizer wrapper),
/// reproducing the MLX `TextTokenizer` policy: `add_special_tokens=true` → right-truncate to
/// `max_length` → right-pad to `max_length` with the pad id (and the matching attention mask).
pub struct Sam3Tokenizer {
    inner: Tokenizer,
    max_length: usize,
    pad_token_id: u32,
}

impl Sam3Tokenizer {
    /// Load from the `facebook/sam3` `tokenizer.json`.
    pub fn from_file(tokenizer_json: impl AsRef<Path>, cfg: &Sam3TextConfig) -> Result<Self> {
        let inner = Tokenizer::from_file(tokenizer_json.as_ref())
            .map_err(|e| CandleError::Msg(format!("sam3 tokenizer: {e}")))?;
        Ok(Self {
            inner,
            max_length: cfg.max_position_embeddings,
            pad_token_id: cfg.pad_token_id as u32,
        })
    }

    /// Tokenize a concept phrase → `input_ids[1, max_length]` (U32) + the key-padding
    /// `attention_mask` (length `max_length`, `1` = real token). The ids/mask are padded to
    /// `max_length` exactly as the reference processor (and the MLX `Sam3Tokenizer`) produce.
    pub fn encode(&self, text: &str, device: &Device) -> Result<(Tensor, Vec<i32>)> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| CandleError::Msg(format!("sam3 tokenizer: {e}")))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        let max = self.max_length;
        if ids.len() > max {
            ids.truncate(max); // right-truncation, as HF does for a single sequence
        }
        let mut mask: Vec<i32> = vec![1; ids.len()];
        if ids.len() < max {
            ids.resize(max, self.pad_token_id);
            mask.resize(max, 0);
        }
        let n = ids.len();
        let ids_tensor = Tensor::from_vec(ids, (1, n), device)?;
        Ok((ids_tensor, mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// N=3, last token padded → key 2 blocked everywhere; the strict upper triangle is blocked.
    #[test]
    fn causal_padding_mask_blocks_future_and_padding() {
        let m = causal_padding_mask(3, &[1, 1, 0], &Device::Cpu).unwrap();
        let v = m.reshape(9).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v[0], 0.0); // (0,0) ok
        assert_eq!(v[1], -1e9); // (0,1) future
        assert_eq!(v[3], 0.0); // (1,0) ok
        assert_eq!(v[4], 0.0); // (1,1) ok
        assert_eq!(v[2], -1e9); // (0,2) future+pad
        assert_eq!(v[8], -1e9); // (2,2) padded key
        assert_eq!(v[6], 0.0); // (2,0) ok
    }
}
