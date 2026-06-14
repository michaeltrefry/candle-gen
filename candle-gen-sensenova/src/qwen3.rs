//! The NEO-Unify **dense dual-path Qwen3** backbone — the candle port of `mlx-gen-sensenova`'s
//! `qwen3.rs` (`modeling_qwen3.py`).
//!
//! Each of the 42 decoder layers carries two parallel **dense** transformer stacks: an
//! *understanding* path and a *generation* path (the `_mot_gen` weights). A forward runs on one path
//! at a time, selected by the caller — the reference's `forward_und` / `forward_gen`, which are the
//! paths T2I uses (text prefill on the understanding path, the denoise loop on the generation path).
//!
//! Per-layer attention (head_dim 128) splits each head into a **temporal** half (normed by
//! `q_norm`/`k_norm`) and a **spatial** half (normed by `q_norm_hw`/`k_norm_hw`); the spatial half
//! splits again into height + width. Three independent RoPE rotations are applied — temporal
//! (`rope_theta`), height and width (`rope_theta_hw`) — then concatenated back to 128. K/V are shared
//! GQA (8 KV heads → 32 query heads). Attention is the reference's eager path
//! (matmul → softmax → matmul) under a block-causal mask, so understanding tokens attend causally
//! while a generation image-block (tokens sharing one temporal index) attends bidirectionally within
//! the block. Everything runs at f32 (the candle port recipe — single matmul dtype).
//!
//! Only the **cached** path is wired (the T2I slice's prefill uses a fresh cache, `past == 0`, which
//! equals the full-sequence block-causal forward; the denoise loop uses the same forward use-only).
//! The non-cached `forward_und`, on-the-fly quantization, and the AdaptableLinear seam the mlx
//! provider carries are dropped — this slice is dense f32, T2I only.

use candle_gen::candle_core::{Device, Result as CResult, Tensor};
use candle_gen::candle_nn::{ops, Linear, Module, VarBuilder};
use candle_gen::Result;

use crate::config::NeoChatConfig;
use crate::distill::DistillLora;

/// Disallowed-attention fill for the additive mask (a large finite negative, matching the candle
/// Kolors slice — avoids `-inf` propagation through the softmax kernel).
const MASK_NEG: f32 = -1e30;

/// Which transformer path a forward runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// Understanding path (text prefill). The reference `forward_und`.
    Und,
    /// Image-generation path (the `_mot_gen` weights). The reference `forward_gen`.
    Gen,
}

/// Per-layer key/value cache for incremental decode. Each entry holds the already-RoPE'd keys and
/// the raw values in `[B, Hkv, S, D]` layout (kv-head count, pre-GQA-expansion).
pub struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    seq_len: usize,
}

impl KvCache {
    /// Total cached sequence length (the `past` prefix length for the next forward).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Persisting append (`update_cache=True`): concat the new K/V onto layer `i` and store it back.
    fn append(&mut self, i: usize, k: Tensor, v: Tensor) -> CResult<(Tensor, Tensor)> {
        let merged = match self.layers[i].take() {
            Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
            None => (k, v),
        };
        self.layers[i] = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }

    /// Non-persisting use (`update_cache=False`): concat past + current for this forward only — the
    /// denoise-loop path (each step runs a fresh image block against the frozen text prefix).
    fn extend(&self, i: usize, k: &Tensor, v: &Tensor) -> CResult<(Tensor, Tensor)> {
        match &self.layers[i] {
            Some((pk, pv)) => Ok((Tensor::cat(&[pk, k], 2)?, Tensor::cat(&[pv, v], 2)?)),
            None => Ok((k.clone(), v.clone())),
        }
    }
}

/// Load a bias-less Linear from `{prefix}.weight` (shapeless via `get_unchecked`).
fn load_linear_no_bias(vb: &VarBuilder, prefix: &str) -> Result<Linear> {
    Ok(Linear::new(
        vb.get_unchecked(&format!("{prefix}.weight"))?,
        None,
    ))
}

/// The per-path attention weights. The projections are bias-less Linears; the QK-norms are dense
/// `[head_dim/2]` weight vectors.
struct AttnPath {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    q_norm_hw: Tensor,
    k_norm_hw: Tensor,
}

impl AttnPath {
    /// `attn_prefix` = `…layers.{i}.self_attn`, `s` = the path suffix (`""` or `"_mot_gen"`).
    fn from_weights(vb: &VarBuilder, attn_prefix: &str, s: &str) -> Result<Self> {
        Ok(Self {
            q_proj: load_linear_no_bias(vb, &format!("{attn_prefix}.q_proj{s}"))?,
            k_proj: load_linear_no_bias(vb, &format!("{attn_prefix}.k_proj{s}"))?,
            v_proj: load_linear_no_bias(vb, &format!("{attn_prefix}.v_proj{s}"))?,
            o_proj: load_linear_no_bias(vb, &format!("{attn_prefix}.o_proj{s}"))?,
            q_norm: vb.get_unchecked(&format!("{attn_prefix}.q_norm{s}.weight"))?,
            k_norm: vb.get_unchecked(&format!("{attn_prefix}.k_norm{s}.weight"))?,
            q_norm_hw: vb.get_unchecked(&format!("{attn_prefix}.q_norm_hw{s}.weight"))?,
            k_norm_hw: vb.get_unchecked(&format!("{attn_prefix}.k_norm_hw{s}.weight"))?,
        })
    }

    /// Merge the distill LoRA into the four projections. Returns the number merged (≤ 4).
    fn merge_distill_lora(
        &mut self,
        lora: &DistillLora,
        attn_prefix: &str,
        s: &str,
    ) -> Result<usize> {
        let mut n = 0;
        for (name, lin) in [
            ("q", &mut self.q_proj),
            ("k", &mut self.k_proj),
            ("v", &mut self.v_proj),
            ("o", &mut self.o_proj),
        ] {
            if let Some(m) = lora.merge_linear(lin, &format!("{attn_prefix}.{name}_proj{s}"))? {
                *lin = m;
                n += 1;
            }
        }
        Ok(n)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn from_weights(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: load_linear_no_bias(vb, &format!("{prefix}.gate_proj"))?,
            up: load_linear_no_bias(vb, &format!("{prefix}.up_proj"))?,
            down: load_linear_no_bias(vb, &format!("{prefix}.down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let gated = (ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        self.down.forward(&gated)
    }

    /// Merge the distill LoRA into the SwiGLU's three linears. Returns the number merged (≤ 3).
    fn merge_distill_lora(&mut self, lora: &DistillLora, prefix: &str) -> Result<usize> {
        let mut n = 0;
        for (name, lin) in [
            ("gate", &mut self.gate),
            ("up", &mut self.up),
            ("down", &mut self.down),
        ] {
            if let Some(m) = lora.merge_linear(lin, &format!("{prefix}.{name}_proj"))? {
                *lin = m;
                n += 1;
            }
        }
        Ok(n)
    }
}

struct Layer {
    input_ln: Tensor,
    input_ln_gen: Tensor,
    post_ln: Tensor,
    post_ln_gen: Tensor,
    attn_und: AttnPath,
    attn_gen: AttnPath,
    mlp_und: Mlp,
    mlp_gen: Mlp,
}

impl Layer {
    fn from_weights(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        Ok(Self {
            input_ln: vb.get_unchecked(&format!("{prefix}.input_layernorm.weight"))?,
            input_ln_gen: vb.get_unchecked(&format!("{prefix}.input_layernorm_mot_gen.weight"))?,
            post_ln: vb.get_unchecked(&format!("{prefix}.post_attention_layernorm.weight"))?,
            post_ln_gen: vb
                .get_unchecked(&format!("{prefix}.post_attention_layernorm_mot_gen.weight"))?,
            attn_und: AttnPath::from_weights(vb, &attn, "")?,
            attn_gen: AttnPath::from_weights(vb, &attn, "_mot_gen")?,
            mlp_und: Mlp::from_weights(vb, &format!("{prefix}.mlp"))?,
            mlp_gen: Mlp::from_weights(vb, &format!("{prefix}.mlp_mot_gen"))?,
        })
    }
}

/// Plain RMSNorm `weight · x / sqrt(mean(x²) + eps)` over the last dim (matches `mlx_rs::fast::rms_norm`
/// and ChatGLM/Qwen RMSNorm — NOT Gemma's `(1 + weight)`). `w` broadcasts over the leading dims.
fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> CResult<Tensor> {
    let last = x.rank() - 1;
    let mean = x.sqr()?.mean_keepdim(last)?; // [.., 1]
    let denom = (mean + eps)?.sqrt()?;
    x.broadcast_div(&denom)?.broadcast_mul(w)
}

/// RoPE cos/sin for arbitrary integer positions over `dim` rotary dims (f32), shaped `[1, S, dim]`.
/// `inv_freq[j] = theta^(-2j/dim)`, `emb = cat(freqs, freqs)`.
fn rope_cos_sin(
    positions: &[i32],
    dim: usize,
    theta: f32,
    device: &Device,
) -> CResult<(Tensor, Tensor)> {
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / dim as f32))
        .collect();
    let s = positions.len();
    let mut emb = Vec::with_capacity(s * dim);
    for &p in positions {
        // cat(freqs, freqs): both halves carry the same `p · inv_freq`.
        for &f in &inv_freq {
            emb.push(p as f32 * f);
        }
        for &f in &inv_freq {
            emb.push(p as f32 * f);
        }
    }
    let emb = Tensor::from_vec(emb, (1, s, dim), device)?;
    Ok((emb.cos()?, emb.sin()?))
}

/// HF half-split rotary: `x·cos + rotate_half(x)·sin`, with `cos`/`sin` `[1,S,dim]` broadcast over the
/// head axis of `x` `[B,S,H,dim]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CResult<Tensor> {
    let d = x.dim(3)?;
    let half = d / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], 3)?; // rotate_half = cat(-x2, x1)
    let cos = cos.unsqueeze(2)?; // [1,S,1,dim]
    let sin = sin.unsqueeze(2)?;
    x.broadcast_mul(&cos)? + rot.broadcast_mul(&sin)?
}

/// Expand `[B,Hkv,S,D]` → `[B,Hkv·groups,S,D]` (GQA), repeating each kv head `groups` times (query
/// head `i` uses KV head `i / groups`).
fn repeat_kv_bhsd(x: &Tensor, groups: usize) -> CResult<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, hkv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, hkv, groups, s, d))?
        .contiguous()?
        .reshape((b, hkv * groups, s, d))
}

/// The dense dual-path Qwen3 backbone: token embeddings, the decoder stack, and the dual final norm.
/// (The `lm_head` is not loaded — the non-think T2I path needs no logits.)
pub struct Qwen3Backbone {
    embed_tokens: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    norm_gen: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
    rope_theta: f32,
    rope_theta_hw: f32,
    device: Device,
}

/// Precomputed tri-axis RoPE tables (`(cos, sin)` per temporal/H/W axis) and the additive block mask
/// for a fixed set of position indexes and cache prefix length. Invariant across the denoise steps of
/// a given cache, so the gen-path loop builds one and reuses it for every step (F-139).
pub struct RopeMask {
    cos_t: Tensor,
    sin_t: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    mask: Tensor,
    n_tokens: usize,
}

impl Qwen3Backbone {
    /// Build from a checkpoint, `prefix` = the `language_model` namespace (e.g. `"language_model"`).
    pub fn from_weights(vb: &VarBuilder, cfg: &NeoChatConfig, prefix: &str) -> Result<Self> {
        let model = format!("{prefix}.model");
        let layers = (0..cfg.llm.num_hidden_layers)
            .map(|i| Layer::from_weights(vb, &format!("{model}.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: vb.get_unchecked(&format!("{model}.embed_tokens.weight"))?,
            layers,
            norm: vb.get_unchecked(&format!("{model}.norm.weight"))?,
            norm_gen: vb.get_unchecked(&format!("{model}.norm_mot_gen.weight"))?,
            num_heads: cfg.llm.num_attention_heads,
            num_kv_heads: cfg.llm.num_key_value_heads,
            head_dim: cfg.llm.head_dim(),
            eps: cfg.llm.rms_norm_eps,
            rope_theta: cfg.llm.rope_theta,
            rope_theta_hw: cfg.llm.rope_theta_hw,
            device: vb.device().clone(),
        })
    }

    /// Token embedding for a single sequence: `ids` `[S]` → `[1, S, hidden]`.
    pub fn embed(&self, ids: &[i32]) -> CResult<Tensor> {
        let s = ids.len();
        let idx: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let idx = Tensor::from_vec(idx, (s,), &self.device)?;
        let g = self.embed_tokens.index_select(&idx, 0)?; // [s, hidden]
        let h = self.embed_tokens.dim(1)?;
        g.reshape((1, s, h))
    }

    /// Merge the 8-step distill LoRA into every layer's **generation-path** attention projections +
    /// SwiGLU (`*_mot_gen`); the understanding path is untouched. Returns the total linears merged
    /// (`7 · layers` when the LoRA carries every target).
    pub fn merge_distill_lora(&mut self, lora: &DistillLora, prefix: &str) -> Result<usize> {
        let mut n = 0;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let attn = format!("{prefix}.model.layers.{i}.self_attn");
            n += layer.attn_gen.merge_distill_lora(lora, &attn, "_mot_gen")?;
            let mlp = format!("{prefix}.model.layers.{i}.mlp_mot_gen");
            n += layer.mlp_gen.merge_distill_lora(lora, &mlp)?;
        }
        Ok(n)
    }

    /// A fresh empty cache (one slot per decoder layer).
    pub fn new_cache(&self) -> KvCache {
        KvCache {
            layers: (0..self.layers.len()).map(|_| None).collect(),
            seq_len: 0,
        }
    }

    /// The incremental-decode forward that backs prefill (Und, `append=true`) and the denoise loop
    /// (Gen, `append=false`). `embeds` `[1, S_new, hidden]` are the new tokens; the `(t,h,w)` rows are
    /// their positions. Returns the final-normed hidden states `[1, S_new, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        embeds: &Tensor,
        temporal: &[i32],
        height: &[i32],
        width: &[i32],
        path: Path,
        cache: &mut KvCache,
        append: bool,
    ) -> CResult<Tensor> {
        let rm = self.prepare_rope_mask(temporal, height, width, cache.len())?;
        self.forward_prepared(embeds, &rm, path, cache, append)
    }

    /// Build the tri-axis RoPE tables + block mask for `(temporal, height, width)` at cache prefix
    /// length `past`. Hoisted so the denoise loop builds it once per cache and reuses it (F-139).
    pub fn prepare_rope_mask(
        &self,
        temporal: &[i32],
        height: &[i32],
        width: &[i32],
        past: usize,
    ) -> CResult<RopeMask> {
        let dt = self.head_dim / 2;
        let dhw = self.head_dim / 4;
        let (cos_t, sin_t) = rope_cos_sin(temporal, dt, self.rope_theta, &self.device)?;
        let (cos_h, sin_h) = rope_cos_sin(height, dhw, self.rope_theta_hw, &self.device)?;
        let (cos_w, sin_w) = rope_cos_sin(width, dhw, self.rope_theta_hw, &self.device)?;
        let mask = cached_block_mask(past, temporal, &self.device)?;
        Ok(RopeMask {
            cos_t,
            sin_t,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            mask,
            n_tokens: temporal.len(),
        })
    }

    /// The decoder stack over `embeds` using a prebuilt [`RopeMask`]. The `RopeMask`'s `past` must
    /// match `cache.len()` (true within a denoise run — use-only `append = false` leaves it fixed).
    pub fn forward_prepared(
        &self,
        embeds: &Tensor,
        rm: &RopeMask,
        path: Path,
        cache: &mut KvCache,
        append: bool,
    ) -> CResult<Tensor> {
        let mut hidden = embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let (input_ln, post_ln, attn, mlp) = match path {
                Path::Und => (
                    &layer.input_ln,
                    &layer.post_ln,
                    &layer.attn_und,
                    &layer.mlp_und,
                ),
                Path::Gen => (
                    &layer.input_ln_gen,
                    &layer.post_ln_gen,
                    &layer.attn_gen,
                    &layer.mlp_gen,
                ),
            };
            let normed = rms_norm(&hidden, input_ln, self.eps)?;
            let attn_out = self.attention_cached(&normed, attn, rm, cache, i, append)?;
            hidden = (hidden + attn_out)?;
            let normed = rms_norm(&hidden, post_ln, self.eps)?;
            hidden = (hidden + mlp.forward(&normed)?)?;
        }
        if append {
            cache.seq_len += rm.n_tokens;
        }
        let final_norm = match path {
            Path::Und => &self.norm,
            Path::Gen => &self.norm_gen,
        };
        rms_norm(&hidden, final_norm, self.eps)
    }

    /// Cached attention: project the new tokens, RoPE q/k, merge with the cache, GQA-expand, attend.
    fn attention_cached(
        &self,
        x: &Tensor,
        a: &AttnPath,
        rm: &RopeMask,
        cache: &mut KvCache,
        layer_idx: usize,
        append: bool,
    ) -> CResult<Tensor> {
        let (b, s, _) = x.dims3()?;
        let hd = self.head_dim;

        // q/k: project + reshape + temporal/spatial norm + tri-axis RoPE, then to [B,H,S,D].
        let q = self
            .qk_rope(
                &a.q_proj.forward(x)?,
                b,
                s,
                self.num_heads,
                &a.q_norm,
                &a.q_norm_hw,
                rm,
            )?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .qk_rope(
                &a.k_proj.forward(x)?,
                b,
                s,
                self.num_kv_heads,
                &a.k_norm,
                &a.k_norm_hw,
                rm,
            )?
            .transpose(1, 2)?
            .contiguous()?;
        let v = a
            .v_proj
            .forward(x)?
            .reshape((b, s, self.num_kv_heads, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        // Merge with the cache (persist or use-only), then GQA-expand the full K/V.
        let (k_all, v_all) = if append {
            cache.append(layer_idx, k, v)?
        } else {
            cache.extend(layer_idx, &k, &v)?
        };
        let groups = self.num_heads / self.num_kv_heads;
        let k_all = repeat_kv_bhsd(&k_all, groups)?;
        let v_all = repeat_kv_bhsd(&v_all, groups)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let scores = (q.matmul(&k_all.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(&rm.mask)?;
        let weights = ops::softmax_last_dim(&scores)?;
        let out = weights.matmul(&v_all)?; // [B,H,S,D]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.num_heads * hd))?;
        a.o_proj.forward(&out)
    }

    /// Project→reshape→(temporal/spatial split)→norm halves→rope(t,h,w)→concat. `proj` is the
    /// already-projected `[B, S, H·head_dim]` tensor.
    #[allow(clippy::too_many_arguments)]
    fn qk_rope(
        &self,
        proj: &Tensor,
        b: usize,
        s: usize,
        heads: usize,
        norm_t: &Tensor,
        norm_hw: &Tensor,
        rm: &RopeMask,
    ) -> CResult<Tensor> {
        let hd = self.head_dim;
        let dt = hd / 2;
        let dhw = hd / 4;
        let x = proj.reshape((b, s, heads, hd))?;
        let t = x.narrow(3, 0, dt)?.contiguous()?; // temporal half [.., 64]
        let sp = x.narrow(3, dt, dt)?.contiguous()?; // spatial half [.., 64]
        let t = rms_norm(&t, norm_t, self.eps)?;
        let hw = rms_norm(&sp, norm_hw, self.eps)?;
        let h = hw.narrow(3, 0, dhw)?.contiguous()?; // height [.., 32]
        let w = hw.narrow(3, dhw, dhw)?.contiguous()?; // width [.., 32]
        let t = apply_rope(&t, &rm.cos_t, &rm.sin_t)?;
        let h = apply_rope(&h, &rm.cos_h, &rm.sin_h)?;
        let w = apply_rope(&w, &rm.cos_w, &rm.sin_w)?;
        Tensor::cat(&[&t, &h, &w], 3)
    }
}

/// Cached additive mask `[1,1,S_new, past+S_new]` (0 / `MASK_NEG`). New query row `r` attends every
/// cached column (`j < past`), and new column `c` iff `temporal[r] == temporal[c]` (same image block
/// → bidirectional) **or** `c <= r` (causal). With `past == 0` this is the full block-causal mask.
fn cached_block_mask(past: usize, temporal: &[i32], device: &Device) -> CResult<Tensor> {
    let q = temporal.len();
    let k = past + q;
    let mut data = vec![0f32; q * k];
    for r in 0..q {
        for j in 0..k {
            let allowed = if j < past {
                true
            } else {
                let c = j - past;
                temporal[r] == temporal[c] || c <= r
            };
            if !allowed {
                data[r * k + j] = MASK_NEG;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, q, k), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn cached_block_mask_text_prefix_is_causal() {
        // past=0, all-distinct temporal → plain causal mask (j <= i allowed).
        let m = cached_block_mask(0, &[0, 1, 2], &Device::Cpu)
            .unwrap()
            .reshape((3, 3))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(m[0][1], MASK_NEG); // future masked
        assert_eq!(m[1][0], 0.0); // past attended
        assert_eq!(m[2][2], 0.0); // self attended
    }

    #[test]
    fn cached_block_mask_image_block_is_bidirectional() {
        // An image block: all tokens share temporal index 5, sitting after a 1-token text prefix.
        let m = cached_block_mask(1, &[5, 5, 5], &Device::Cpu)
            .unwrap()
            .reshape((3, 4))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        // Column 0 is the cached text prefix — always attended.
        for row in &m {
            assert_eq!(row[0], 0.0);
        }
        // Within the block (cols 1..4) every token sees every other (same temporal → bidirectional).
        for row in &m {
            assert_eq!(row[1], 0.0);
            assert_eq!(row[2], 0.0);
            assert_eq!(row[3], 0.0);
        }
    }
}
