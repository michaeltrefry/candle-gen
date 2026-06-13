//! Qwen-Image's **Qwen2.5-VL** text encoder (text path). Port of `mlx-gen-qwen-image`'s
//! `text_encoder/`. A 28-layer decoder-only LM (hidden 3584, GQA 28q/4kv, head_dim 128) whose
//! **last layer's normed** hidden state — with the leading **34** template tokens dropped — is the
//! transformer's `prompt_embeds` (width = `joint_attention_dim` = 3584).
//!
//! vs the FLUX.2 Qwen3 encoder: **q/k/v have biases** (o_proj does not), there is **no per-head q/k
//! RMSNorm**, RoPE is **half-split (NeoX)** (θ=1e6), and the final `model.norm` IS applied. Runs in
//! **f32** (the fork rounds only the final embeds to bf16); the LM prefix is `model.`.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{
    embedding, linear, linear_no_bias, ops::softmax_last_dim, rms_norm, rotary_emb::rope,
    Embedding, Linear, Module, RmsNorm, VarBuilder,
};

use crate::config::TextEncoderConfig;

struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(head_dim: usize, theta: f32, max_seq: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), device)?;
        let t = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        Ok((
            rope(&q.contiguous()?, &cos, &sin)?,
            rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        Ok(Self {
            q_proj: linear(h, nh * hd, vb.pp("q_proj"))?, // biased
            k_proj: linear(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nh * hd, h, vb.pp("o_proj"))?, // bias-less
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, nh, hd))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, nkv, hd))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, nkv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let (q, k) = rotary.apply(&q, &k)?;
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;
        let scale = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = softmax_last_dim(&scores)?;
        let o = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, s, nh * hd))?;
        self.o_proj.forward(&o)
    }
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, nkv, groups, s, d))?
        .reshape((b, nkv * groups, s, d))
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?.silu()?;
        self.down.forward(&(g * self.up.forward(x)?)?)
    }
}

struct DecoderLayer {
    input_ln: RmsNorm,
    post_ln: RmsNorm,
    attn: Attention,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_ln: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_ln: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&self.input_ln.forward(x)?, rotary, mask)?)?;
        &h + self.mlp.forward(&self.post_ln.forward(&h)?)?
    }
}

/// The Qwen-Image Qwen2.5-VL text encoder.
pub struct QwenTextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: Rotary,
    drop_idx: usize,
}

impl QwenTextEncoder {
    /// Build under the `model.*` prefix.
    pub fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let model = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, model.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let vb_layers = model.pp("layers");
        for i in 0..cfg.n_layers {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, model.pp("norm"))?;
        let rotary = Rotary::new(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.max_length.max(1),
            vb.device(),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            drop_idx: cfg.prompt_drop_idx,
        })
    }

    /// `input_ids` `[1, S]` → `prompt_embeds` `[1, S − drop_idx, 3584]` (f32): the last layer's normed
    /// hidden state with the leading `drop_idx` (=34) template tokens dropped. Single-prompt causal
    /// attention (no padding).
    pub fn prompt_embeds(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let mask = causal_mask(b, s, input_ids.device())?;
        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.rotary, &mask)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        // Drop the leading template tokens.
        let keep = s - self.drop_idx;
        hidden.narrow(1, self.drop_idx, keep)
    }
}

/// Additive causal mask `[B, 1, S, S]` (f32): `0` where `j <= i`, `-inf` otherwise.
fn causal_mask(b: usize, s: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in (i + 1)..s {
                data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (b, 1, s, s), device)
}
