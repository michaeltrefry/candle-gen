//! LLaVA projector, image-token splice, and Llama-3.1-8B decoder for JoyCaption (candle port).
//!
//! JoyCaption is a `LlavaForConditionalGeneration`: the SigLIP `-2` features are projected into the
//! Llama hidden size (gelu-MLP projector), the expanded image-token placeholders in the prompt are
//! replaced one-for-one by those projected rows, and a Llama-3.1 8B causal decoder generates the
//! caption autoregressively (greedy at `temperature == 0`, else temperature/top-p with a small
//! CTRL-style repetition penalty, ported from the reference sampler).
//!
//! Ported from `mlx-gen`'s `caption/joycaption/language.rs` onto `candle_nn` primitives — candle
//! ships no Llama that accepts pre-spliced `inputs_embeds`, so the decoder is built in-crate.

use candle_gen::candle_core::{DType, Error, IndexOp, Result, Tensor, D};
use candle_gen::candle_nn::{
    embedding, linear, linear_no_bias, ops::silu, ops::softmax_last_dim, rms_norm, Embedding,
    Linear, Module, RmsNorm, VarBuilder,
};
use candle_gen::gen_core::{default_seed, CancelFlag, CaptionFinishReason, CaptionSampling};

use crate::prompt::{
    END_OF_TEXT_TOKEN_ID, EOM_TOKEN_ID, EOT_TOKEN_ID, IMAGE_SEQ_LENGTH, IMAGE_TOKEN_ID,
};

pub const LLAMA_HIDDEN_SIZE: usize = 4096;
pub const LLAMA_INTERMEDIATE_SIZE: usize = 14336;
pub const LLAMA_NUM_LAYERS: usize = 32;
pub const LLAMA_NUM_HEADS: usize = 32;
pub const LLAMA_NUM_KV_HEADS: usize = 8;
pub const LLAMA_HEAD_DIM: usize = 128;
pub const LLAMA_VOCAB_SIZE: usize = 128256;
pub const LLAMA_RMS_NORM_EPS: f64 = 1e-5;
pub const LLAMA_ROPE_THETA: f32 = 500_000.0;
pub const LLAMA_ROPE_FACTOR: f32 = 8.0;
pub const LLAMA_ROPE_LOW_FREQ_FACTOR: f32 = 1.0;
pub const LLAMA_ROPE_HIGH_FREQ_FACTOR: f32 = 4.0;
pub const LLAMA_ORIGINAL_MAX_POSITION_EMBEDDINGS: f32 = 8192.0;

pub const PROJECTOR_IN_SIZE: usize = 1152;
pub const PROJECTOR_HIDDEN_SIZE: usize = 4096;

/// Llama-3.1 generation stop tokens (`generation_config.eos_token_id`): end-of-text, end-of-message,
/// end-of-turn.
pub const STOP_TOKENS: &[i64] = &[END_OF_TEXT_TOKEN_ID, EOM_TOKEN_ID, EOT_TOKEN_ID];

/// CTRL/HF repetition-penalty strength (Keskar et al. 2019): a recently-emitted token's logit is
/// divided by this when positive, multiplied when negative. `1.05` is gentle. A documented port-time
/// deviation from the plain temperature/top-p reference sampler — kept (not removed) to curb
/// JoyCaption's tendency to loop on long captions, matching the mlx provider.
const REPETITION_PENALTY: f32 = 1.05;
/// Sliding window of recent history tokens the repetition penalty looks back over.
const REPETITION_PENALTY_WINDOW: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub rope_low_freq_factor: f32,
    pub rope_high_freq_factor: f32,
    pub rope_original_context: f32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            hidden_size: LLAMA_HIDDEN_SIZE,
            intermediate_size: LLAMA_INTERMEDIATE_SIZE,
            num_layers: LLAMA_NUM_LAYERS,
            num_heads: LLAMA_NUM_HEADS,
            num_kv_heads: LLAMA_NUM_KV_HEADS,
            head_dim: LLAMA_HEAD_DIM,
            vocab_size: LLAMA_VOCAB_SIZE,
            rms_norm_eps: LLAMA_RMS_NORM_EPS,
            rope_theta: LLAMA_ROPE_THETA,
            rope_factor: LLAMA_ROPE_FACTOR,
            rope_low_freq_factor: LLAMA_ROPE_LOW_FREQ_FACTOR,
            rope_high_freq_factor: LLAMA_ROPE_HIGH_FREQ_FACTOR,
            rope_original_context: LLAMA_ORIGINAL_MAX_POSITION_EMBEDDINGS,
        }
    }
}

/// The LLaVA multimodal projector: `linear_1` → exact-gelu → `linear_2` (both with bias).
pub struct LlavaProjector {
    linear_1: Linear,
    linear_2: Linear,
}

impl LlavaProjector {
    /// `vb` points at HF `multi_modal_projector`.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear(PROJECTOR_IN_SIZE, PROJECTOR_HIDDEN_SIZE, vb.pp("linear_1"))?,
            linear_2: linear(PROJECTOR_HIDDEN_SIZE, LLAMA_HIDDEN_SIZE, vb.pp("linear_2"))?,
        })
    }

    /// Project SigLIP features `[b, seq, 1152]` → Llama hidden `[b, seq, 4096]`.
    pub fn forward(&self, vision_features: &Tensor) -> Result<Tensor> {
        // HF LLaVA projector default activation is "gelu" = exact (erf) gelu.
        self.linear_2
            .forward(&self.linear_1.forward(vision_features)?.gelu_erf()?)
    }
}

/// Per-layer rolling K/V cache (`[b, kv_heads, seq, head_dim]`), grown along the sequence axis.
pub struct LlamaKvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl LlamaKvCache {
    fn append(&mut self, i: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let merged = match self.layers[i].take() {
            Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
            None => (k, v),
        };
        self.layers[i] = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }
}

pub struct LlamaDecoder {
    embed_tokens: Embedding,
    layers: Vec<LlamaLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Llama3Rope,
    cfg: LlamaConfig,
}

impl LlamaDecoder {
    /// Load HF `language_model.model.*` + `language_model.lm_head.weight`. `vb` points at
    /// `language_model`.
    pub fn new(cfg: LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let model = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, model.pp("embed_tokens"))?;
        let layers = (0..cfg.num_layers)
            .map(|i| LlamaLayer::new(&cfg, model.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, model.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope: Llama3Rope::new(&cfg)?,
            cfg,
        })
    }

    pub fn new_cache(&self) -> LlamaKvCache {
        LlamaKvCache {
            layers: (0..self.layers.len()).map(|_| None).collect(),
        }
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// Embed token ids `[b, seq]` (u32) → `[b, seq, hidden]`.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// Run pre-embedded tokens at absolute `offset`, append K/V to `cache`, return last-position
    /// logits `[b, vocab]` as f32.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Tensor,
        cache: &mut LlamaKvCache,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, q_len, _) = input_embeds.dims3()?;
        let dtype = input_embeds.dtype();
        let dev = input_embeds.device();
        let k_len = offset + q_len;
        let mask = causal_mask(q_len, k_len, offset, dtype, dev)?;
        // RoPE tables are built on the host (cheap trig) then moved to the model device + dtype.
        let (cos, sin) = self.rope.forward(q_len, offset)?;
        let cos = cos.to_dtype(dtype)?.to_device(dev)?;
        let sin = sin.to_dtype(dtype)?.to_device(dev)?;

        let mut hidden = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask, cache, i)?;
        }
        let last = hidden.i((.., q_len - 1, ..))?; // [b, hidden]
        let normed = self.norm.forward(&last)?;
        let logits = self
            .lm_head
            .forward(&normed.reshape((b, self.cfg.hidden_size))?)?;
        logits.to_dtype(DType::F32)
    }

    /// Decode a single token id `[b, 1]` at `offset`.
    pub fn decode_logits(
        &self,
        input_ids: &Tensor,
        cache: &mut LlamaKvCache,
        offset: usize,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Generate token ids from already-spliced prompt embeddings (`[1, seq, hidden]`). Stop tokens
    /// are not included in the returned list.
    pub fn generate_from_embeds(
        &self,
        prompt_ids: &[i64],
        prompt_embeds: &Tensor,
        sampling: CaptionSampling,
        cancel: &CancelFlag,
        on_token: &mut dyn FnMut(),
    ) -> Result<LanguageGeneration> {
        if prompt_ids.is_empty() {
            return Err(Error::Msg("joycaption: prompt ids are empty".to_owned()));
        }
        if cancel.is_cancelled() {
            return Ok(LanguageGeneration {
                token_ids: Vec::new(),
                finish_reason: CaptionFinishReason::Cancelled,
            });
        }

        let mut history = prompt_ids.to_vec();
        let mut generated = Vec::new();
        let mut cache = self.new_cache();
        // `seed: None` draws a fresh per-call seed so repeated captions vary; an explicit seed
        // reproduces an exact sample (matches the reference + the generators' RNG policy).
        let mut rng = SplitMix64::new(sampling.seed.unwrap_or_else(default_seed));
        let prompt_len = prompt_ids.len();
        let dev = prompt_embeds.device().clone();
        let mut logits = self.decode_logits_from_embeds(prompt_embeds, &mut cache, 0)?;

        for step in 0..sampling.max_new_tokens as usize {
            if cancel.is_cancelled() {
                return Ok(LanguageGeneration {
                    token_ids: generated,
                    finish_reason: CaptionFinishReason::Cancelled,
                });
            }
            let next = sample_token(&logits, &history, sampling, &mut rng)?;
            if STOP_TOKENS.contains(&next) {
                return Ok(LanguageGeneration {
                    token_ids: generated,
                    finish_reason: CaptionFinishReason::StopToken,
                });
            }
            generated.push(next);
            history.push(next);
            on_token();

            if step + 1 == sampling.max_new_tokens as usize {
                break;
            }
            let token = Tensor::from_vec(vec![next as u32], (1, 1), &dev)?;
            logits = self.decode_logits(&token, &mut cache, prompt_len + step)?;
        }
        Ok(LanguageGeneration {
            token_ids: generated,
            finish_reason: CaptionFinishReason::MaxTokens,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanguageGeneration {
    pub token_ids: Vec<i64>,
    pub finish_reason: CaptionFinishReason,
}

struct LlamaLayer {
    input_layernorm: RmsNorm,
    self_attn: LlamaAttention,
    post_attention_layernorm: RmsNorm,
    mlp: LlamaMlp,
}

impl LlamaLayer {
    fn new(cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            self_attn: LlamaAttention::new(cfg, vb.pp("self_attn"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: LlamaMlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        cache: &mut LlamaKvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let h = (x + self
            .self_attn
            .forward(&normed, cos, sin, mask, cache, layer_idx)?)?;
        let normed2 = self.post_attention_layernorm.forward(&h)?;
        &h + self.mlp.forward(&normed2)?
    }
}

struct LlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl LlamaAttention {
    fn new(cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        Ok(Self {
            q_proj: linear_no_bias(h, q_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(q_dim, h, vb.pp("o_proj"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f64).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        cache: &mut LlamaKvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let q = q.transpose(1, 2)?.contiguous()?; // [b, heads, s, hd]
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let (k_all, v_all) = cache.append(layer_idx, k, v)?;

        let groups = self.num_heads / self.num_kv_heads;
        let k_all = repeat_kv(&k_all, groups)?;
        let v_all = repeat_kv(&v_all, groups)?;

        let attn = (q.matmul(&k_all.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&v_all)?.transpose(1, 2)?.reshape((
            b,
            s,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&out)
    }
}

struct LlamaMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl LlamaMlp {
    fn new(cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Llama-3 RoPE with the llama3 frequency rescaling (`rope_scaling.rope_type = "llama3"`).
struct Llama3Rope {
    inv_freq: Vec<f32>,
    head_dim: usize,
}

impl Llama3Rope {
    fn new(cfg: &LlamaConfig) -> Result<Self> {
        let half = cfg.head_dim / 2;
        let low_freq_wavelen = cfg.rope_original_context / cfg.rope_low_freq_factor;
        let high_freq_wavelen = cfg.rope_original_context / cfg.rope_high_freq_factor;
        let inv_freq = (0..half)
            .map(|i| {
                let inv = 1.0 / cfg.rope_theta.powf((2 * i) as f32 / cfg.head_dim as f32);
                let wavelen = 2.0 * std::f32::consts::PI / inv;
                if wavelen > low_freq_wavelen {
                    inv / cfg.rope_factor
                } else if wavelen < high_freq_wavelen {
                    inv
                } else {
                    let smooth = (cfg.rope_original_context / wavelen - cfg.rope_low_freq_factor)
                        / (cfg.rope_high_freq_factor - cfg.rope_low_freq_factor);
                    (1.0 - smooth) * inv / cfg.rope_factor + smooth * inv
                }
            })
            .collect();
        Ok(Self {
            inv_freq,
            head_dim: cfg.head_dim,
        })
    }

    /// `(cos, sin)` each `[1, seq, head_dim]` (f32), for absolute positions `offset..offset+seq`.
    fn forward(&self, seq_len: usize, offset: usize) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let mut cos = Vec::with_capacity(seq_len * self.head_dim);
        let mut sin = Vec::with_capacity(seq_len * self.head_dim);
        debug_assert_eq!(half * 2, self.head_dim);
        for s in 0..seq_len {
            let pos = (offset + s) as f32;
            // emb = concat(freqs, freqs) over the head dim (GPT-NeoX layout).
            let row: Vec<f32> = self.inv_freq.iter().map(|&f| pos * f).collect();
            for _ in 0..2 {
                for &angle in &row {
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
        let dev = candle_gen::candle_core::Device::Cpu;
        let cos = Tensor::from_vec(cos, (1, seq_len, self.head_dim), &dev)?;
        let sin = Tensor::from_vec(sin, (1, seq_len, self.head_dim), &dev)?;
        Ok((cos, sin))
    }
}

/// Apply rotary embeddings (GPT-NeoX rotate-half) to `x` of shape `[b, seq, heads, head_dim]`.
/// `cos`/`sin` are `[1, seq, head_dim]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(2)?; // [1, seq, 1, head_dim]
    let sin = sin.unsqueeze(2)?;
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
}

/// Expand `[b, kv_heads, s, hd]` → `[b, kv_heads*groups, s, hd]` (GQA head replication).
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, kv, s, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv, groups, s, hd))?
        .reshape((b, kv * groups, s, hd))
}

/// Causal additive mask `[1, 1, q_len, k_len]` for a query block starting at `q_offset`.
fn causal_mask(
    q_len: usize,
    k_len: usize,
    q_offset: usize,
    dtype: DType,
    dev: &candle_gen::candle_core::Device,
) -> Result<Tensor> {
    // Finite large-negative (not -inf) — avoids NaNs in bf16 softmax on fully-masked rows.
    let neg = -3.389_531_4e38_f32;
    let mut data = vec![0f32; q_len * k_len];
    for r in 0..q_len {
        let pos = q_offset + r;
        for j in 0..k_len {
            if j > pos {
                data[r * k_len + j] = neg;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, q_len, k_len), dev)?.to_dtype(dtype)
}

/// Splice projected image features into token embeddings. The expanded prompt holds exactly
/// [`IMAGE_SEQ_LENGTH`] **contiguous** image-token positions (from the single `<|image|>` marker);
/// they are overwritten in order by `projected` (`[image_seq, hidden]`).
pub fn splice_image_features(
    token_embeds: &Tensor,
    ids: &[i64],
    projected: &Tensor,
) -> Result<Tensor> {
    let (b, s, h) = token_embeds.dims3()?;
    if b != 1 || ids.len() != s {
        return Err(Error::Msg(format!(
            "joycaption splice: expected [1, {}, hidden] embeds, got [{b}, {s}, {h}]",
            ids.len()
        )));
    }
    let count = ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count();
    if count != IMAGE_SEQ_LENGTH {
        return Err(Error::Msg(format!(
            "joycaption splice: expected {IMAGE_SEQ_LENGTH} image tokens, found {count}"
        )));
    }
    let start = ids
        .iter()
        .position(|&id| id == IMAGE_TOKEN_ID)
        .expect("image token present");
    // The expansion produces one contiguous run; verify it before the contiguous slice_assign.
    if !ids[start..start + IMAGE_SEQ_LENGTH]
        .iter()
        .all(|&id| id == IMAGE_TOKEN_ID)
    {
        return Err(Error::Msg(
            "joycaption splice: image tokens are not contiguous".to_owned(),
        ));
    }
    let proj = projected
        .reshape((1, IMAGE_SEQ_LENGTH, h))?
        .to_dtype(token_embeds.dtype())?;
    token_embeds.slice_assign(&[0..1, start..start + IMAGE_SEQ_LENGTH, 0..h], &proj)
}

// ---- Host-side sampler (backend-agnostic; matches the reference language sampler) ----

fn sample_token(
    logits: &Tensor,
    history: &[i64],
    sampling: CaptionSampling,
    rng: &mut SplitMix64,
) -> Result<i64> {
    let mut v: Vec<f32> = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let vocab = v.len();

    // CTRL/HF-style repetition penalty over the recent history (documented port deviation).
    for &token in history.iter().rev().take(REPETITION_PENALTY_WINDOW) {
        let idx = token as usize;
        if idx < vocab {
            v[idx] = if v[idx] < 0.0 {
                v[idx] * REPETITION_PENALTY
            } else {
                v[idx] / REPETITION_PENALTY
            };
        }
    }

    if sampling.temperature <= 0.0 {
        return Ok(argmax_f32(&v));
    }

    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / sampling.temperature;
    let mut probs: Vec<(usize, f32)> = (0..vocab)
        .map(|i| (i, ((v[i] - max) * inv_t).exp()))
        .collect();

    if sampling.top_p < 1.0 {
        probs = nucleus_select(&probs, sampling.top_p);
    }

    let total: f32 = probs.iter().map(|x| x.1).sum();
    if total <= 0.0 || !total.is_finite() {
        return Ok(argmax_f32(&v));
    }
    let mut target = rng.next_f32() * total;
    for (i, prob) in &probs {
        target -= prob;
        if target <= 0.0 {
            return Ok(*i as i64);
        }
    }
    Ok(probs.last().map(|x| x.0).unwrap_or(0) as i64)
}

/// Top-p (nucleus) selection via a partial max-heap: the highest-probability `(token, weight)` pairs
/// in descending order whose cumulative weight first reaches `top_p · total` (at least one token).
fn nucleus_select(probs: &[(usize, f32)], top_p: f32) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    struct ByWeight(usize, f32);
    impl PartialEq for ByWeight {
        fn eq(&self, o: &Self) -> bool {
            self.1.total_cmp(&o.1) == Ordering::Equal
        }
    }
    impl Eq for ByWeight {}
    impl PartialOrd for ByWeight {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for ByWeight {
        fn cmp(&self, o: &Self) -> Ordering {
            self.1.total_cmp(&o.1)
        }
    }

    let total: f32 = probs.iter().map(|x| x.1).sum();
    let threshold = top_p.max(0.0) * total;
    let mut heap: BinaryHeap<ByWeight> = probs.iter().map(|&(i, p)| ByWeight(i, p)).collect();
    let mut kept: Vec<(usize, f32)> = Vec::new();
    let mut cum = 0.0f32;
    while let Some(ByWeight(i, p)) = heap.pop() {
        kept.push((i, p));
        cum += p;
        if cum >= threshold {
            break;
        }
    }
    kept
}

fn argmax_f32(v: &[f32]) -> i64 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as i64
}

/// SplitMix64 — a tiny, dependency-free, reproducible RNG for the categorical draw (matches the
/// reference sampler so a fixed seed reproduces an exact caption).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn default_llama_config_matches_joycaption() {
        let cfg = LlamaConfig::default();
        assert_eq!(cfg.hidden_size, cfg.num_heads * cfg.head_dim);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(STOP_TOKENS, &[128001, 128008, 128009]);
    }

    #[test]
    fn rope_freqs_have_half_head_dim_entries() {
        let rope = Llama3Rope::new(&LlamaConfig::default()).unwrap();
        assert_eq!(rope.inv_freq.len(), LLAMA_HEAD_DIM / 2);
        let (cos, sin) = rope.forward(4, 0).unwrap();
        assert_eq!(cos.dims(), &[1, 4, LLAMA_HEAD_DIM]);
        assert_eq!(sin.dims(), &[1, 4, LLAMA_HEAD_DIM]);
    }

    #[test]
    fn greedy_sampling_is_argmax() {
        let logits = Tensor::from_vec(vec![0.1f32, 4.0, 2.0], (1, 3), &Device::Cpu).unwrap();
        let mut rng = SplitMix64::new(0);
        let next = sample_token(
            &logits,
            &[],
            CaptionSampling {
                temperature: 0.0,
                top_p: 1.0,
                max_new_tokens: 1,
                seed: None,
                ..Default::default()
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(next, 1);
    }

    #[test]
    fn nucleus_keeps_at_least_one_token() {
        let logits = Tensor::from_vec(vec![5.0f32, 4.0, 1.0], (1, 3), &Device::Cpu).unwrap();
        let mut rng = SplitMix64::new(0);
        let next = sample_token(
            &logits,
            &[],
            CaptionSampling {
                temperature: 0.7,
                top_p: 0.0,
                max_new_tokens: 1,
                seed: None,
                ..Default::default()
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(next, 0);
    }

    #[test]
    fn splice_overwrites_contiguous_image_rows() {
        // hidden = 2, seq = image_seq + 2 text tokens around a contiguous image run.
        let h = 2usize;
        let img = IMAGE_SEQ_LENGTH;
        let s = img + 2;
        let mut ids = vec![10i64];
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, img));
        ids.push(11);
        let embeds = Tensor::zeros((1, s, h), DType::F32, &Device::Cpu).unwrap();
        let proj = Tensor::ones((img, h), DType::F32, &Device::Cpu).unwrap();
        let out = splice_image_features(&embeds, &ids, &proj).unwrap();
        // Text rows stay zero; image rows become one.
        let row0 = out.i((0, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
        let row1 = out.i((0, 1, 0)).unwrap().to_scalar::<f32>().unwrap();
        let row_last = out.i((0, s - 1, 0)).unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(row0, 0.0);
        assert_eq!(row1, 1.0);
        assert_eq!(row_last, 0.0);
    }
}
