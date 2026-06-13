//! Gemma-3-12B text encoder — the LTX-2.3 text-encoder backbone. Port of mlx-gen-ltx `gemma.rs`
//! (`GemmaModel::forward`), itself a port of `mlx_vlm` Gemma-3. Returns the **49 hidden states**
//! (scaled embedding + each of 48 layer outputs, the last final-normed) that the LTX feature
//! extractor concatenates and projects.
//!
//! Gemma specifics: RMSNorm scales by **(1 + weight)** (eps 1e-6); token embeddings ×√hidden_size
//! (bf16); **per-layer RoPE base** (local 1e4 on sliding layers `(i+1)%6 != 0`, global 1e6
//! otherwise); **q/k RMSNorm over head_dim** (256); GQA (16 q / 8 kv heads); attention scale
//! `256^-0.5`; MLP `down(gelu_tanh(gate(x)) * up(x))`; norm-sandwich block. Our checkpoint is dense
//! bf16 (no quant). The prompt is ≤ `sliding_window` (1024), so one full causal+padding mask serves
//! every layer (only the RoPE base differs). Runs bf16; RoPE + attention compute in f32 for fidelity.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    ops::rms_norm as candle_rms_norm, ops::softmax_last_dim, Linear, Module, VarBuilder,
};

use crate::config::GemmaConfig;

/// Finite large-negative mask value (bf16 min, as f32) — used instead of -∞ so an all-masked row
/// (a left-padding query position) softmaxes to a finite uniform vector rather than NaN. Those
/// positions are zeroed downstream by the attention-mask multiply in the aggregator.
const MASK_NEG: f32 = -3.389_531_4e38;

/// `weight + 1.0` (Gemma RMSNorm scale), kept bf16.
fn norm_alpha(vb: &VarBuilder, key: &str) -> Result<Tensor> {
    let w = vb.get_unchecked(key)?;
    (w + 1.0)?.to_dtype(DType::BF16)
}

fn linear(vb: &VarBuilder, key: &str) -> Result<Linear> {
    let w = vb
        .get_unchecked(&format!("{key}.weight"))?
        .to_dtype(DType::BF16)?;
    Ok(Linear::new(w, None))
}

struct GemmaLayer {
    input_ln: Tensor,
    post_attn_ln: Tensor,
    pre_ff_ln: Tensor,
    post_ff_ln: Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    rope_base: f64,
}

pub struct GemmaEncoder {
    embed: Tensor, // [vocab, hidden] bf16
    layers: Vec<GemmaLayer>,
    norm: Tensor,
    embed_scale: Tensor, // bf16 scalar √hidden
    cfg: GemmaConfig,
    device: Device,
}

impl GemmaEncoder {
    /// Build from a VarBuilder rooted at `language_model.model.` of a gemma-3-12b-it snapshot.
    pub fn new(vb: VarBuilder, cfg: &GemmaConfig) -> Result<Self> {
        let device = vb.device().clone();
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lb = vb.pp(format!("layers.{i}"));
            let attn = lb.pp("self_attn");
            let rope_base = if cfg.is_global_layer(i) {
                cfg.rope_theta_global
            } else {
                cfg.rope_theta_local
            };
            layers.push(GemmaLayer {
                input_ln: norm_alpha(&lb, "input_layernorm.weight")?,
                post_attn_ln: norm_alpha(&lb, "post_attention_layernorm.weight")?,
                pre_ff_ln: norm_alpha(&lb, "pre_feedforward_layernorm.weight")?,
                post_ff_ln: norm_alpha(&lb, "post_feedforward_layernorm.weight")?,
                q_proj: linear(&attn, "q_proj")?,
                k_proj: linear(&attn, "k_proj")?,
                v_proj: linear(&attn, "v_proj")?,
                o_proj: linear(&attn, "o_proj")?,
                q_norm: norm_alpha(&attn, "q_norm.weight")?,
                k_norm: norm_alpha(&attn, "k_norm.weight")?,
                gate_proj: linear(&lb.pp("mlp"), "gate_proj")?,
                up_proj: linear(&lb.pp("mlp"), "up_proj")?,
                down_proj: linear(&lb.pp("mlp"), "down_proj")?,
                rope_base,
            });
        }
        let embed = vb
            .get_unchecked("embed_tokens.weight")?
            .to_dtype(DType::BF16)?;
        let scale = (cfg.hidden_size as f64).sqrt();
        let embed_scale = Tensor::new(scale as f32, &device)?.to_dtype(DType::BF16)?;
        Ok(Self {
            embed,
            layers,
            norm: norm_alpha(&vb, "norm.weight")?,
            embed_scale,
            cfg: cfg.clone(),
            device,
        })
    }

    fn rms(&self, x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
        candle_rms_norm(&x.contiguous()?, alpha, self.cfg.rms_eps as f32)
    }

    /// NeoX rotate-half RoPE in f32: `x` `[B,H,L,D]`, `cos`/`sin` `[1,1,L,D/2]` → rotated, cast back.
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let o1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
        let o2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
        Tensor::cat(&[&o1, &o2], D::Minus1)?.to_dtype(in_dtype)
    }

    /// Build `(cos, sin)` `[1,1,L,head_dim/2]` (f32) for a given RoPE base.
    fn rope_tables(&self, l: usize, base: f64) -> Result<(Tensor, Tensor)> {
        let d = self.cfg.head_dim;
        let half = d / 2;
        let mut cos = vec![0f32; l * half];
        let mut sin = vec![0f32; l * half];
        for p in 0..l {
            for i in 0..half {
                let inv_freq = base.powf(-(2.0 * i as f64) / d as f64);
                let theta = p as f64 * inv_freq;
                cos[p * half + i] = theta.cos() as f32;
                sin[p * half + i] = theta.sin() as f32;
            }
        }
        Ok((
            Tensor::from_vec(cos, (1, 1, l, half), &self.device)?,
            Tensor::from_vec(sin, (1, 1, l, half), &self.device)?,
        ))
    }

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, kv, l, d) = x.dims4()?;
        x.unsqueeze(2)?
            .broadcast_as((b, kv, n_rep, l, d))?
            .reshape((b, kv * n_rep, l, d))
    }

    #[allow(clippy::too_many_arguments)]
    fn attn(
        &self,
        layer: &GemmaLayer,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kv, d) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
        let q = layer
            .q_proj
            .forward(x)?
            .reshape((b, l, h, d))?
            .transpose(1, 2)?;
        let k = layer
            .k_proj
            .forward(x)?
            .reshape((b, l, kv, d))?
            .transpose(1, 2)?;
        let v = layer
            .v_proj
            .forward(x)?
            .reshape((b, l, kv, d))?
            .transpose(1, 2)?;
        // q/k RMSNorm over head_dim, then per-layer RoPE.
        let q = self.rms(&q.contiguous()?, &layer.q_norm)?;
        let k = self.rms(&k.contiguous()?, &layer.k_norm)?;
        let q = Self::apply_rope(&q, cos, sin)?;
        let k = Self::apply_rope(&k, cos, sin)?;
        // GQA + attention in f32.
        let k = Self::repeat_kv(&k, h / kv)?;
        let v = Self::repeat_kv(&v, h / kv)?;
        let qf = q.to_dtype(DType::F32)?.contiguous()?;
        let kf = k.to_dtype(DType::F32)?.contiguous()?;
        let vf = v.to_dtype(DType::F32)?.contiguous()?;
        let scale = self.cfg.query_pre_attn_scalar.powf(-0.5);
        let scores = (qf.matmul(&kf.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = softmax_last_dim(&scores)?;
        let out = probs.matmul(&vf)?; // (b,h,l,d) f32
        let out = out
            .transpose(1, 2)?
            .reshape((b, l, h * d))?
            .to_dtype(DType::BF16)?;
        layer.o_proj.forward(&out)
    }

    fn mlp(&self, layer: &GemmaLayer, x: &Tensor) -> Result<Tensor> {
        let gate = layer.gate_proj.forward(x)?.gelu()?; // tanh-approx gelu
        let up = layer.up_proj.forward(x)?;
        layer.down_proj.forward(&(gate * up)?)
    }

    fn layer_forward(
        &self,
        layer: &GemmaLayer,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let r = self.attn(layer, &self.rms(x, &layer.input_ln)?, mask, cos, sin)?;
        let h = (x + self.rms(&r, &layer.post_attn_ln)?)?;
        let r = self.mlp(layer, &self.rms(&h, &layer.pre_ff_ln)?)?;
        &h + self.rms(&r, &layer.post_ff_ln)?
    }

    /// Additive causal + left-padding mask `[1,1,L,L]` f32. `valid(i,j) = j<=i && mask01[j]`.
    fn causal_padding_mask(&self, mask01: &[u32], l: usize) -> Result<Tensor> {
        let mut data = vec![0f32; l * l];
        for i in 0..l {
            for j in 0..l {
                let valid = j <= i && mask01[j] != 0;
                data[i * l + j] = if valid { 0.0 } else { MASK_NEG };
            }
        }
        Tensor::from_vec(data, (1, 1, l, l), &self.device)
    }

    /// Run the encoder over `input_ids` `[1,L]` (u32) + `mask01` (1 for valid, left-padded) → the
    /// **49 hidden states** `[1,L,3840]` (bf16).
    pub fn forward(&self, input_ids: &Tensor, mask01: &[u32]) -> Result<Vec<Tensor>> {
        let (b, l) = input_ids.dims2()?;
        let ids = input_ids.reshape((b * l,))?;
        let mut h = self
            .embed
            .index_select(&ids, 0)?
            .reshape((b, l, self.cfg.hidden_size))?;
        h = h.broadcast_mul(&self.embed_scale)?;

        let mask = self.causal_padding_mask(mask01, l)?;
        // Two RoPE tables (local / global base); pick per layer.
        let (cos_l, sin_l) = self.rope_tables(l, self.cfg.rope_theta_local)?;
        let (cos_g, sin_g) = self.rope_tables(l, self.cfg.rope_theta_global)?;

        let mut hiddens = Vec::with_capacity(self.cfg.num_layers + 1);
        hiddens.push(h.clone());
        for (i, layer) in self.layers.iter().enumerate() {
            let (cos, sin) = if self.cfg.is_global_layer(i) {
                (&cos_g, &sin_g)
            } else {
                (&cos_l, &sin_l)
            };
            let _ = layer.rope_base; // base is encoded by the table selection above
            h = self.layer_forward(layer, &h, &mask, cos, sin)?;
            if i < self.cfg.num_layers - 1 {
                hiddens.push(h.clone());
            }
        }
        hiddens.push(self.rms(&h, &self.norm)?);
        Ok(hiddens)
    }
}
