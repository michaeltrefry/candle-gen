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

use candle_gen::candle_core::{DType, Device, IndexOp, Result, Tensor};
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

    /// The plain 1-D RoPE `(cos, sin)` `[seq, head_dim/2]` for the text path (sequential positions).
    fn text(&self, seq: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.cos.narrow(0, 0, seq)?, self.sin.narrow(0, 0, seq)?))
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

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[s, head_dim/2]` (the text 1-D or image 3-D MRoPE table);
    /// `mask`: additive causal `[b, 1, s, s]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);

        let q = self.q_proj.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        // Per-head q/k RMSNorm over the head dim, then transpose to [B, H, S, D].
        let q = rmsnorm(&q, &self.q_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = rmsnorm(&k, &self.k_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
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

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&rmsnorm(x, &self.input_ln, self.eps)?, cos, sin, mask)?)?;
        &h + self.mlp.forward(&rmsnorm(&h, &self.post_ln, self.eps)?)?
    }
}

/// Qwen3-VL `text_config.rope_parameters.mrope_section` — the per-axis (T/H/W) frequency counts over
/// `head_dim/2 = 64`. The image path interleaves these across the rotary freqs (the Qwen3-VL form).
const MROPE_SECTION: [usize; 3] = [24, 20, 20];
/// Vision spatial merge — the LM sees one token per `merge²` patches.
const SPATIAL_MERGE: i64 = 2;

/// The Boogu Qwen3-VL text-path condition encoder.
pub struct BooguTextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    final_norm: Tensor,
    eps: f64,
    head_dim: usize,
    rope_theta: f32,
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
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            device: w.device().clone(),
        })
    }

    /// `input_ids`: `[1, S]` u32. Returns `last_hidden_state` `[1, S, 4096]` (f32) — all layers run,
    /// final norm applied. Causal (decoder-only); no padding (the candle tokenizer emits none).
    pub fn last_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let (cos, sin) = self.rotary.text(s)?;
        let mask = causal_mask(b, s, &self.device)?;
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        rmsnorm(&hidden, &self.final_norm, self.eps)
    }

    /// Single-reference image-conditioned forward (Edit) — a thin wrapper over
    /// [`Self::last_hidden_with_images`] for one reference image (`grid_thw`, `image_embeds`
    /// `[n, 4096]`, and its 3 `deepstack` features). Kept for the single-reference call sites and the
    /// component-parity harness.
    pub fn last_hidden_with_image(
        &self,
        input_ids: &Tensor,
        image_embeds: &Tensor,
        deepstack: &[Tensor],
        grid_thw: [i32; 3],
        image_token_id: u32,
    ) -> Result<Tensor> {
        self.last_hidden_with_images(
            input_ids,
            std::slice::from_ref(image_embeds),
            std::slice::from_ref(&deepstack.to_vec()),
            std::slice::from_ref(&grid_thw),
            image_token_id,
        )
    }

    /// Multi-reference image-conditioned forward (Edit). Splices each reference's `image_embeds[k]`
    /// (`[n_k, 4096]`, the vision tower's merged output) into the token embeddings at the k-th
    /// contiguous `image_token_id` block (in input-id order), runs the 36 decoder layers under the
    /// 3-D **interleaved MRoPE** (each image's grid advancing the shared position counter), and injects
    /// each reference's `deepstack[k]` features at its image block after layers 0/1/2 — mirroring
    /// `Qwen3VLTextModel` with multiple `<|image_pad|>` blocks. `grids[k]` is image k's patch grid
    /// `[t, h, w]`. `b = 1`. The block order must match the reference order (the chat template emits
    /// the references' vision blocks before the instruction, in order).
    pub fn last_hidden_with_images(
        &self,
        input_ids: &Tensor,
        image_embeds: &[Tensor],
        deepstack: &[Vec<Tensor>],
        grids: &[[i32; 3]],
        image_token_id: u32,
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let ids: Vec<u32> = input_ids.i(0)?.to_vec1::<u32>()?;

        // Contiguous `<|image_pad|>` blocks, in order; block k carries reference k.
        let blocks = image_blocks(&ids, image_token_id);
        if blocks.len() != image_embeds.len() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "boogu edit: {} image-token blocks in input_ids but {} reference embeds",
                blocks.len(),
                image_embeds.len()
            )));
        }

        // Token embeddings, then splice each reference's vision embeds at its block. Each replacement
        // is the same length as the block it replaces, so earlier splices don't shift later indices.
        let mut hidden = self.embed_tokens.forward(input_ids)?; // [1, s, 4096], f32
        for (k, &(start, len)) in blocks.iter().enumerate() {
            if image_embeds[k].dim(0)? != len {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "boogu edit: reference {k} has {} vision tokens but its image block is {len}",
                    image_embeds[k].dim(0)?
                )));
            }
            let img = image_embeds[k].unsqueeze(0)?.to_dtype(hidden.dtype())?; // [1, n_k, 4096]
            hidden = replace_seq(&hidden, &img, start, start + len, s)?;
        }

        // 3-D interleaved MRoPE (per-image grids) + causal mask.
        let (pt, ph, pw) = mrope_positions(&ids, image_token_id, grids);
        let (cos, sin) = self.mrope_cos_sin(&pt, &ph, &pw)?;
        let mask = causal_mask(b, s, &self.device)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            // Deepstack: after LM layers 0/1/2, add each reference's layer-i feature at its block.
            for (k, &(start, len)) in blocks.iter().enumerate() {
                if i < deepstack[k].len() {
                    let ds = deepstack[k][i].unsqueeze(0)?.to_dtype(hidden.dtype())?; // [1, n_k, 4096]
                    let mid = (slice_seq(&hidden, start, start + len)? + ds)?;
                    hidden = replace_seq(&hidden, &mid, start, start + len, s)?;
                }
            }
        }
        rmsnorm(&hidden, &self.final_norm, self.eps)
    }

    /// Build the interleaved-MRoPE `cos`/`sin` `[s, head_dim/2]` (f32). Each of the `head_dim/2`
    /// frequencies takes its position from the T/H/W axis per the Qwen3-VL interleave: within the
    /// first `mrope_section[1]·3` indices, `j%3==1 → H`, `j%3==2 → W`, else `T` (the tail stays `T`).
    fn mrope_cos_sin(&self, pt: &[i64], ph: &[i64], pw: &[i64]) -> Result<(Tensor, Tensor)> {
        let s = pt.len();
        let hd = self.head_dim;
        let half = hd / 2;
        let sec_h = MROPE_SECTION[1] * 3;
        let sec_w = MROPE_SECTION[2] * 3;
        let inv: Vec<f32> = (0..half)
            .map(|j| (self.rope_theta as f64).powf(-(2.0 * j as f64) / hd as f64) as f32)
            .collect();

        let mut freqs = vec![0f32; s * half];
        for (i, ((&t, &h), &w)) in pt.iter().zip(ph).zip(pw).enumerate() {
            for j in 0..half {
                let pos = if j < sec_h && j % 3 == 1 {
                    h
                } else if j < sec_w && j % 3 == 2 {
                    w
                } else {
                    t
                };
                freqs[i * half + j] = pos as f32 * inv[j];
            }
        }
        let freqs = Tensor::from_vec(freqs, (s, half), &self.device)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }
}

/// Slice `[1, s, d]` along the sequence axis (axis 1) to `[start, end)`.
fn slice_seq(x: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    x.narrow(1, start, end - start)
}

/// Replace `x[:, start:end, :]` with `repl` (`[1, end-start, d]`) via concat of the surrounding slices.
fn replace_seq(x: &Tensor, repl: &Tensor, start: usize, end: usize, s: usize) -> Result<Tensor> {
    let before = x.narrow(1, 0, start)?;
    let after = x.narrow(1, end, s - end)?;
    Tensor::cat(&[&before, repl, &after], 1)
}

/// Contiguous runs of `image_token_id` in `ids`, returned as `(start, len)` in input-id order. Each
/// run is one reference image's `<|image_pad|>` block.
fn image_blocks(ids: &[u32], image_token_id: u32) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            let start = i;
            while i < ids.len() && ids[i] == image_token_id {
                i += 1;
            }
            blocks.push((start, i - start));
        } else {
            i += 1;
        }
    }
    blocks
}

/// 3-D MRoPE positions per token (mirrors `get_rope_index` + `get_vision_position_ids`): text tokens
/// advance `(i, i, i)`; the k-th image block (at running offset `cur`) takes `grids[k] = [t, h, w]`,
/// gets `t = cur`, `h = cur + row`, `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then
/// advances `cur += max(h, w) / merge`. Multiple image blocks consume `grids` in order.
fn mrope_positions(
    ids: &[u32],
    image_token_id: u32,
    grids: &[[i32; 3]],
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i64;
    let mut i = 0usize;
    let mut img_k = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            let g = grids[img_k];
            let (llm_h, llm_w) = (g[1] as i64 / SPATIAL_MERGE, g[2] as i64 / SPATIAL_MERGE);
            let step = (g[1].max(g[2]) as i64) / SPATIAL_MERGE;
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
            img_k += 1;
        } else {
            pt.push(cur);
            ph.push(cur);
            pw.push(cur);
            cur += 1;
            i += 1;
        }
    }
    (pt, ph, pw)
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

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: u32 = 151655;

    #[test]
    fn image_blocks_finds_runs_in_order() {
        // text, text, [4 image], text, [2 image], text.
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        assert_eq!(image_blocks(&ids, IMG), vec![(2, 4), (7, 2)]);
    }

    #[test]
    fn mrope_positions_advance_across_two_images() {
        // Block 0 ↔ grid [1,4,4] (merged 2×2 = 4 tokens, t-step max(4,4)/2 = 2);
        // block 1 ↔ grid [1,4,2] (merged 2×1 = 2 tokens, t-step max(4,2)/2 = 2).
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        let grids = [[1, 4, 4], [1, 4, 2]];
        let (pt, ph, pw) = mrope_positions(&ids, IMG, &grids);
        assert_eq!(pt.len(), ids.len());

        // Leading text advances 0,1.
        assert_eq!((pt[0], pt[1]), (0, 1));
        // Image 0 sits at t-axis = 2 (the running offset); spatial in h/w.
        assert_eq!(&pt[2..6], &[2, 2, 2, 2]);
        assert_eq!(&ph[2..6], &[2, 2, 3, 3]); // rows 0,0,1,1 + offset 2
        assert_eq!(&pw[2..6], &[2, 3, 2, 3]); // cols 0,1,0,1 + offset 2
                                              // Text after image 0: offset advanced by max(4,4)/2 = 2 → 4.
        assert_eq!(pt[6], 4);
        // Image 1 sits at t-axis = 5 (one past the text), 2 tokens (2×1 grid).
        assert_eq!(&pt[7..9], &[5, 5]);
        assert_eq!(&ph[7..9], &[5, 6]); // rows 0,1 + offset 5
        assert_eq!(&pw[7..9], &[5, 5]); // single column
                                        // Trailing text: offset advanced by max(4,2)/2 = 2 → 7.
        assert_eq!(pt[9], 7);
    }
}
