//! The full Ideogram 4 DiT: token composition (`[text ; image]`), scalar-`t` AdaLN conditioning, 34
//! blocks, and the affine-less final layer. Port of `Ideogram4Transformer.forward`.
//!
//! Token roles (`indicator`): `LLM_TOKEN_INDICATOR = 3` (text), `OUTPUT_IMAGE_INDICATOR = 2`
//! (image). Text positions carry the projected Qwen3-VL features (`llm_cond_proj`); image positions
//! carry the patchified noise latents (`input_proj`). Both streams live in one sequence, mixed every
//! block by full (segment-masked) attention + interleaved 3D MRoPE.

use candle_gen::candle_core::{DType, Result, Tensor, D};
use candle_gen::candle_nn::{Embedding, Linear, Module};

use super::block::Ideogram4Block;
use super::mrope::Ideogram4MRoPE;
use super::rmsnorm;
use crate::config::Ideogram4DitConfig;
use crate::loader::{linear, Weights};

/// Token role constants (upstream `ideogram4.constants`).
const OUTPUT_IMAGE_INDICATOR: i64 = 2;
const LLM_TOKEN_INDICATOR: i64 = 3;

/// `llm_cond_norm` and the final LayerNorm both use eps 1e-6 (upstream).
const COND_NORM_EPS: f64 = 1e-6;
const FINAL_NORM_EPS: f64 = 1e-6;

pub struct Ideogram4Transformer {
    input_proj: Linear,
    llm_cond_norm: Tensor,
    llm_cond_proj: Linear,
    t_mlp_in: Linear,
    t_mlp_out: Linear,
    adaln_proj: Linear,
    embed_image_indicator: Embedding,
    rotary_emb: Ideogram4MRoPE,
    layers: Vec<Ideogram4Block>,
    final_adaln: Linear,
    final_linear: Linear,
    /// Sinusoidal frequencies for the `t` embedding (`[1, emb_dim/2]`, f32).
    t_freqs: Tensor,
    dtype: DType,
}

impl Ideogram4Transformer {
    /// Load a DiT from a component dir of `.safetensors` (top-level keys: `input_proj.*`,
    /// `layers.{i}.*`, `final_layer.*`, …). `w`'s dtype is the DiT compute dtype (bf16).
    pub fn load(w: &Weights, cfg: &Ideogram4DitConfig) -> Result<Self> {
        let head_dim = cfg.emb_dim / cfg.num_heads;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Ideogram4Block::load(
                w,
                &format!("layers.{i}"),
                cfg.num_heads,
                head_dim,
                cfg.norm_eps,
            )?);
        }
        // Sinusoidal freqs: half = emb_dim/2, lf = ln(1e4)/(half-1), f[d] = exp(-lf·d).
        let half = cfg.emb_dim / 2;
        let lf = (1e4f32).ln() / (half as f32 - 1.0);
        let t_freqs: Vec<f32> = (0..half).map(|d| (-lf * d as f32).exp()).collect();
        let t_freqs = Tensor::from_vec(t_freqs, (1, half), w.device())?;

        let ind_w = w.get("embed_image_indicator.weight")?;
        let embed_image_indicator = Embedding::new(ind_w, cfg.emb_dim);

        Ok(Self {
            input_proj: linear(w, "input_proj", true)?,
            llm_cond_norm: w.get("llm_cond_norm.weight")?,
            llm_cond_proj: linear(w, "llm_cond_proj", true)?,
            t_mlp_in: linear(w, "t_embedding.mlp_in", true)?,
            t_mlp_out: linear(w, "t_embedding.mlp_out", true)?,
            adaln_proj: linear(w, "adaln_proj", true)?,
            embed_image_indicator,
            rotary_emb: Ideogram4MRoPE::new(
                head_dim,
                cfg.rope_theta,
                cfg.mrope_section,
                w.device(),
            )?,
            layers,
            final_adaln: linear(w, "final_layer.adaln_modulation", true)?,
            final_linear: linear(w, "final_layer.linear", true)?,
            t_freqs,
            dtype: w.dtype(),
        })
    }

    /// Sinusoidal scalar-`t` embedding → MLP. `t`: `[B]` in `[0,1]` → `[B, emb_dim]`.
    fn t_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let scaled = (t.to_dtype(DType::F32)? * 1e4)?; // [B]
        let emb = scaled.unsqueeze(1)?.broadcast_mul(&self.t_freqs)?; // [B, half]
        let emb = Tensor::cat(&[emb.sin()?, emb.cos()?], D::Minus1)?.to_dtype(self.dtype)?;
        let h = self.t_mlp_in.forward(&emb)?.silu()?;
        self.t_mlp_out.forward(&h)
    }

    /// Velocity prediction `[B, L, in_channels]` (f32). Inputs follow the upstream packing:
    /// `llm_features [B,L,llm_dim]`, `x [B,L,in_ch]`, `t [B]`, `position_ids [B,L,3]`,
    /// `segment_ids [B,L]`, `indicator [B,L]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        llm_features: &Tensor,
        x: &Tensor,
        t: &Tensor,
        position_ids: &Tensor,
        segment_ids: &Tensor,
        indicator: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (llm_mask, img_mask, img_idx) = role_tensors(indicator, b, l, self.dtype)?;

        let llm_features = llm_features
            .to_dtype(self.dtype)?
            .broadcast_mul(&llm_mask)?;
        let x = x.to_dtype(self.dtype)?.broadcast_mul(&img_mask)?;
        let x = self.input_proj.forward(&x)?.broadcast_mul(&img_mask)?;

        let t_cond = self.t_embedding(t)?.unsqueeze(1)?; // [B,1,emb]
        let adaln_input = self.adaln_proj.forward(&t_cond)?.silu()?; // [B,1,adaln]

        let llm = rmsnorm(&llm_features, &self.llm_cond_norm, COND_NORM_EPS)?;
        let llm = self.llm_cond_proj.forward(&llm)?.broadcast_mul(&llm_mask)?;

        let mut h = (&x + &llm)?;
        h = (h + self.embed_image_indicator.forward(&img_idx)?)?;

        let (cos, sin) = self.rotary_emb.forward(position_ids)?;
        let mask = segment_mask(segment_ids, b, l, h.device())?;

        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask, &adaln_input)?;
        }

        // Final layer: scale = 1 + adaln(silu(c)); linear(layernorm_no_affine(h) · scale).
        let scale = (self.final_adaln.forward(&adaln_input.silu()?)? + 1.0)?;
        let normed = layer_norm_no_affine(&h, FINAL_NORM_EPS)?;
        let out = self.final_linear.forward(&normed.broadcast_mul(&scale)?)?;
        out.to_dtype(DType::F32)
    }
}

/// No-affine LayerNorm over the last dim (computed in f32 for stability, cast back to `x`'s dtype).
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(dt)
}

/// From `indicator [B,L]`: `(llm_mask [B,L,1], img_mask [B,L,1]` at `dtype`, `img_idx [B,L]` u32).
/// `img_idx` = 1 at image tokens, 0 elsewhere (the `embed_image_indicator` lookup index).
fn role_tensors(
    indicator: &Tensor,
    b: usize,
    l: usize,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    let ind: Vec<i64> = indicator
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    let n = b * l;
    let mut llm = vec![0f32; n];
    let mut img = vec![0f32; n];
    let mut idx = vec![0u32; n];
    for (p, &v) in ind.iter().enumerate().take(n) {
        if v == LLM_TOKEN_INDICATOR {
            llm[p] = 1.0;
        }
        if v == OUTPUT_IMAGE_INDICATOR {
            img[p] = 1.0;
            idx[p] = 1;
        }
    }
    let dev = indicator.device();
    Ok((
        Tensor::from_vec(llm, (b, l, 1), dev)?.to_dtype(dtype)?,
        Tensor::from_vec(img, (b, l, 1), dev)?.to_dtype(dtype)?,
        Tensor::from_vec(idx, (b, l), dev)?,
    ))
}

/// Additive attention mask `[B, 1, L, L]` (f32): `0` where two tokens share a `segment_id`, `-inf`
/// otherwise (full bidirectional attention within a packed sample — not causal).
fn segment_mask(
    segment_ids: &Tensor,
    b: usize,
    l: usize,
    dev: &candle_gen::candle_core::Device,
) -> Result<Tensor> {
    let seg: Vec<i64> = segment_ids
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    let mut data = vec![0f32; b * l * l];
    for bi in 0..b {
        for i in 0..l {
            for j in 0..l {
                if seg[bi * l + i] != seg[bi * l + j] {
                    data[(bi * l + i) * l + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(data, (b, 1, l, l), dev)
}
