//! Boogu's **Qwen3-VL-8B-Instruct** condition encoder (text path; the vision tower is unused for
//! text-to-image). A 36-layer decoder-only LM whose **last_hidden_state** (all layers + final norm)
//! is the per-token `[1, L, 4096]` instruction features the DiT's caption embedder consumes. Port of
//! `mlx-gen-boogu`'s `text_encoder/`.
//!
//! GQA (32 query / 8 kv heads), bias-less q/k/v/o, **per-head q/k RMSNorm**, HF half-split RoPE
//! (θ = 5e6), SwiGLU MLP, pre-norm causal decoder blocks. The text-only path uses plain 1-D RoPE
//! (Qwen3-VL's MRoPE sections all index the same sequential text position with no image tokens).
//! Runs in **f32** — the proven parity-grade precision for this exact encoder in the sibling ideogram
//! port; the DiT casts the features down to bf16.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::rotary_emb::rope;
use candle_gen::candle_nn::{Embedding, Linear, Module};

use crate::loader::{linear, rmsnorm, Weights};

/// Qwen3-VL-8B text-tower architecture (from `mllm/config.json` `text_config`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BooguTextEncoderConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl BooguTextEncoderConfig {
    pub fn qwen3_vl_8b() -> Self {
        Self {
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
        }
    }
}

/// HF half-split RoPE table (θ over `head_dim`), built once for the max sequence length (f32).
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
        let freqs = t.matmul(&inv_freq)?; // (max_seq, head_dim/2)
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let q = rope(&q.contiguous()?, &cos, &sin)?;
        let k = rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn load(w: &Weights, prefix: &str, cfg: &BooguTextEncoderConfig) -> Result<Self> {
        Ok(Self {
            q_proj: linear(w, &format!("{prefix}.q_proj"), false)?,
            k_proj: linear(w, &format!("{prefix}.k_proj"), false)?,
            v_proj: linear(w, &format!("{prefix}.v_proj"), false)?,
            o_proj: linear(w, &format!("{prefix}.o_proj"), false)?,
            q_norm: w.get(&format!("{prefix}.q_norm.weight"))?,
            k_norm: w.get(&format!("{prefix}.k_norm.weight"))?,
            n_heads: cfg.num_heads,
            n_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);

        let q = self.q_proj.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        // Per-head q/k RMSNorm over the head dim, then transpose to [B, H, S, D].
        let q = rmsnorm(&q, &self.q_norm, self.eps)?.transpose(1, 2)?;
        let k = rmsnorm(&k, &self.k_norm, self.eps)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (q, k) = rotary.apply(&q, &k)?;
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?; // [B, nh, S, D]
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
        self.o_proj.forward(&o)
    }
}

/// Repeat each kv head `groups` times along the head axis ([B,nkv,S,D] → [B,nkv·groups,S,D]).
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, nkv, groups, s, d))?
        .contiguous()?
        .reshape((b, nkv * groups, s, d))
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: linear(w, &format!("{prefix}.gate_proj"), false)?,
            up: linear(w, &format!("{prefix}.up_proj"), false)?,
            down: linear(w, &format!("{prefix}.down_proj"), false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        self.down.forward(&gated)
    }
}

struct DecoderLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Attention,
    mlp: Mlp,
    eps: f64,
}

impl DecoderLayer {
    fn load(w: &Weights, prefix: &str, cfg: &BooguTextEncoderConfig) -> Result<Self> {
        Ok(Self {
            input_ln: w.get(&format!("{prefix}.input_layernorm.weight"))?,
            post_ln: w.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
            attn: Attention::load(w, &format!("{prefix}.self_attn"), cfg)?,
            mlp: Mlp::load(w, &format!("{prefix}.mlp"))?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&rmsnorm(x, &self.input_ln, self.eps)?, rotary, mask)?)?;
        &h + self.mlp.forward(&rmsnorm(&h, &self.post_ln, self.eps)?)?
    }
}

/// The Boogu Qwen3-VL text-path condition encoder.
pub struct BooguTextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    final_norm: Tensor,
    eps: f64,
    device: Device,
}

impl BooguTextEncoder {
    /// Load from the `mllm` weights under `prefix` (`"model.language_model"`).
    pub fn load(
        w: &Weights,
        prefix: &str,
        cfg: &BooguTextEncoderConfig,
        max_seq: usize,
    ) -> Result<Self> {
        let embed_weight = w.get(&format!("{prefix}.embed_tokens.weight"))?;
        let hidden = embed_weight.dim(1)?;
        let embed_tokens = Embedding::new(embed_weight, hidden);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DecoderLayer::load(w, &format!("{prefix}.layers.{i}"), cfg)?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            rotary: Rotary::new(cfg.head_dim, cfg.rope_theta, max_seq.max(1), w.device())?,
            final_norm: w.get(&format!("{prefix}.norm.weight"))?,
            eps: cfg.rms_norm_eps,
            device: w.device().clone(),
        })
    }

    /// `input_ids`: `[1, S]` u32. Returns `last_hidden_state` `[1, S, 4096]` (f32) — all layers run,
    /// final norm applied. Causal (decoder-only); no padding (the candle tokenizer emits none).
    pub fn last_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let mask = causal_mask(b, s, &self.device)?;
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.rotary, &mask)?;
        }
        rmsnorm(&hidden, &self.final_norm, self.eps)
    }
}

/// Additive causal mask `[B, 1, S, S]` (f32): `0` where query `i` may attend key `j` (`j ≤ i`),
/// `-inf` otherwise. No padding term (the candle tokenizer emits no padding).
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
