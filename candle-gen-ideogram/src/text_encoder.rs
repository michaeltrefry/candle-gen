//! Ideogram 4's **Qwen3-VL-8B-Instruct** text encoder (text path only — the vision tower is unused
//! for text-to-image). A 36-layer decoder-only LM whose hidden states at the 13 indices in
//! [`crate::config::EXTRACTED_LAYERS`] (`0,3,…,33,35`) are **interleaved** into the
//! `13·4096 = 53248`-wide features the DiT's `llm_cond_proj` consumes.
//!
//! Adapted from `candle-gen-flux2`'s `Qwen3TextEncoder` (same Qwen3 assembly: GQA 32q/8kv, bias-less
//! q/k/v/o, per-head q/k RMSNorm, HF half-split RoPE, SwiGLU, pre-norm residual blocks, no final
//! norm). Ideogram differs in exactly three ways:
//!   * **θ = 5e6** (klein's Qwen3 is 1e6),
//!   * **13** captured states under the `language_model.*` key prefix (klein concatenates 3 under
//!     `model.*`), and — critically —
//!   * the capture index is the LAYER index whose OUTPUT is taken (`captured[i] = layer_i(hidden)`),
//!     NOT HF `output_hidden_states` (which offsets by one with raw embeddings at index 0); and the
//!     captured states are **interleaved** on the feature axis (`f = h·n + layer`), NOT
//!     block-concatenated — the DiT's `llm_cond_proj` was trained on the interleaved layout; the
//!     wrong order yields a coherent but prompt-agnostic image.
//!
//! The text-only path uses plain 1-D RoPE: Qwen3-VL's MRoPE sections all index the same sequential
//! text position when there are no image tokens, so it reduces to standard RoPE. Runs in **f32**.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{
    embedding, linear_no_bias, ops::softmax_last_dim, rms_norm, rotary_emb::rope, Embedding,
    Linear, Module, RmsNorm, VarBuilder,
};

use crate::config::Ideogram4TextEncoderConfig;

/// HF half-split RoPE table (θ over `head_dim`), built once for the max sequence length.
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
        let (_, _, seq, _) = q.dims4()?;
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
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Ideogram4TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        Ok(Self {
            q_proj: linear_no_bias(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nh * hd, h, vb.pp("o_proj"))?,
            q_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);

        // Project, reshape to [B, H, S, D], apply per-head q/k RMSNorm (over the head_dim axis).
        let q = self.q_proj.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?; // [B, nh, S, D]
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?; // [B, nkv, S, D]
        let v = v.transpose(1, 2)?.contiguous()?;

        let (q, k) = rotary.apply(&q, &k)?;
        // GQA: repeat kv heads to query-head count.
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?; // [B, nh, S, S] + [B, 1, S, S]
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?; // [B, nh, S, D]
        let o = o.transpose(1, 2)?.reshape((b, s, nh * hd))?;
        self.o_proj.forward(&o)
    }
}

/// Repeat each kv head `groups` times along the head axis ([B, nkv, S, D] → [B, nkv·groups, S, D]).
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
    fn new(cfg: &Ideogram4TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?.silu()?;
        let u = self.up.forward(x)?;
        self.down.forward(&(g * u)?)
    }
}

struct DecoderLayer {
    input_ln: RmsNorm,
    post_ln: RmsNorm,
    attn: Attention,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(cfg: &Ideogram4TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
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

/// The Ideogram 4 Qwen3-VL text-path prompt-embeds encoder.
pub struct Ideogram4TextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    /// Layer indices whose OUTPUTS are captured (`captured[i] = layer_i(hidden)`).
    out_layers: Vec<usize>,
}

impl Ideogram4TextEncoder {
    /// Build under the `language_model.*` prefix. The final `language_model.norm` and `lm_head` are
    /// intentionally not loaded — Ideogram uses the raw (pre-final-norm) intermediate states. Only
    /// the first `max(out_layers) + 1` layers are constructed (higher layers cannot affect the kept
    /// states). `max_seq` sizes the RoPE table (use [`crate::config::MAX_TEXT_TOKENS`]).
    pub fn new(
        cfg: &Ideogram4TextEncoderConfig,
        out_layers: &[usize],
        max_seq: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let model = vb.pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, model.pp("embed_tokens"))?;
        let max_layer = *out_layers.iter().max().unwrap_or(&0);
        let mut layers = Vec::with_capacity(max_layer + 1);
        let vb_layers = model.pp("layers");
        for i in 0..=max_layer {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }
        let rotary = Rotary::new(cfg.head_dim, cfg.rope_theta, max_seq.max(1), vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            rotary,
            out_layers: out_layers.to_vec(),
        })
    }

    /// `input_ids` / `attention_mask`: `[B, S]` (ids u32, mask 1=real/0=pad). Returns the
    /// **interleaved** hidden states `[B, S, n·hidden]` (f32) — Ideogram's `llm` features. The final
    /// norm is never applied; only layers up to `max(out_layers)` are run.
    pub fn prompt_embeds(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let mask = build_mask(attention_mask, b, s, input_ids.device())?;
        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;

        // Capture the OUTPUT of layer `i` (index `i`, NOT `i+1`); run up to the last needed layer.
        let mut saved: Vec<(usize, Tensor)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rotary, &mask)?;
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }

        let pick = |idx: usize| -> Result<Tensor> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(format!(
                        "ideogram te: hidden state {idx} not captured"
                    ))
                })
        };
        // INTERLEAVE the layers into the feature axis: each captured `[B,S,H]` → `[B,S,H,1]`, cat on
        // the last axis to `[B,S,H,n]`, reshape to `[B,S,H·n]` so feature `f = h·n + layer`.
        let expanded: Vec<Tensor> = self
            .out_layers
            .iter()
            .map(|&idx| pick(idx)?.unsqueeze(D::Minus1))
            .collect::<Result<_>>()?;
        let stacked = Tensor::cat(&expanded, D::Minus1)?; // [B, S, H, n]
        let (bb, ss, h, n) = stacked.dims4()?;
        stacked.reshape((bb, ss, h * n))
    }
}

/// Additive attention mask `[B, 1, S, S]` (f32): `0` where a query `i` may attend key `j` (causal
/// `j <= i` AND `j` not padding), `-inf` otherwise. Built host-side.
fn build_mask(attention_mask: &Tensor, b: usize, s: usize, device: &Device) -> Result<Tensor> {
    let am: Vec<i64> = attention_mask
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(data, (b, 1, s, s), device)
}
