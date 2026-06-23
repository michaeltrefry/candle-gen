//! Boogu DiT building blocks: GQA self-attention, the dual-stream joint attention, the SwiGLU FFN,
//! the `LuminaRMSNormZero` modulation, and the three block flavours (plain/context, modulated
//! single-stream, double-stream). Port of `mlx-gen-boogu`'s `transformer/block.rs`.
//!
//! All attention is **bidirectional** and, for the per-sample `B = 1` path, fully unmasked (every
//! token valid) — so SDPA takes no mask. Per-head q/k RMSNorm runs over the head dim before the
//! interleaved RoPE; GQA repeats each kv head to match the query heads.

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, Module};

use super::rope::apply_interleaved_rope;
use crate::loader::{linear, rmsnorm, Weights};

/// diffusers `Attention(eps=1e-5)` — the per-head q/k RMSNorm epsilon.
const QK_EPS: f64 = 1e-5;

/// Join a module prefix with a leaf name.
fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Repeat each kv head `groups` times consecutively ([b,s,hkv,hd] → [b,s,hkv·groups,hd]) —
/// `repeat_interleave` over the head axis, matching the reference.
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, s, hkv, hd) = x.dims4()?;
    x.unsqueeze(3)?
        .expand((b, s, hkv, groups, hd))?
        .contiguous()?
        .reshape((b, s, hkv * groups, hd))
}

/// Max elements in a single attention scores tensor `[b,h,Sq,Sk]` before [`sdpa`] chunks over the query
/// rows. candle CUDA kernels index elements with **i32**, so a scores/probs tensor exceeding `i32::MAX`
/// (~2.147B) silently corrupts its tail. T2I runs attention over `[instruct, noise]` (~4.3k tokens at
/// 1024², 28 heads ⇒ ~0.52B, safe); the **Edit** path prepends each reference's image tokens to the
/// noise tokens (`forward_inner` ⇒ `[instruct, ref(×N), noise]`), so a single 1024² reference roughly
/// doubles the image sequence (~8.4k joint tokens ⇒ `h·Sq·Sk` ~1.98B, and >1 reference goes well past
/// `i32::MAX`) → the trailing query rows get garbage attention → washed-out output (sc-7523). A 1.0B
/// budget keeps each chunk's scores well under the i32 limit while leaving the T2I sizes a single
/// un-chunked pass, so the txt2img / Base / Turbo paths stay byte-identical.
const ATTN_SCORES_BUDGET: usize = 1_000_000_000;

/// Bidirectional, unmasked scaled-dot-product attention. `q`/`k`/`v`: `[b, h, s, hd]` → `[b, h, s, hd]`.
/// Chunks over the query rows when the full `[b,h,Sq,Sk]` scores tensor would exceed
/// [`ATTN_SCORES_BUDGET`] (the candle CUDA i32-index limit). Each query row's softmax is over all keys and
/// independent of the other rows, so the chunked result is numerically identical to the single pass — only
/// the long Edit / multi-reference joint sequences trip it.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    sdpa_budgeted(q, k, v, scale, ATTN_SCORES_BUDGET)
}

/// [`sdpa`] with an explicit per-block scores-element budget (so the chunking is unit-testable with a tiny
/// budget that forces the chunked path on small tensors).
fn sdpa_budgeted(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, budget: usize) -> Result<Tensor> {
    let (b, h, s, _) = q.dims4()?;
    let q = q.contiguous()?;
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;

    // The largest query block whose `[b,h,block,s]` scores tensor stays within budget — the whole `s` for
    // the T2I sizes, so that path stays the unchanged single matmul+softmax+matmul.
    let block = if b * h * s * s <= budget {
        s
    } else {
        (budget / (b * h * s)).max(1)
    };
    if block >= s {
        let scores = (q.matmul(&k_t)? * scale)?;
        let probs = softmax_last_dim(&scores)?;
        return probs.matmul(&v); // [b, h, s, hd]
    }
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < s {
        let len = block.min(s - start);
        let scores = (q.narrow(2, start, len)?.contiguous()?.matmul(&k_t)? * scale)?;
        let probs = softmax_last_dim(&scores)?;
        blocks.push(probs.matmul(&v)?); // [b, h, len, hd]
        start += len;
    }
    Tensor::cat(&blocks, 2) // [b, h, s, hd]
}

// ── GQA self-attention (standard `BooguImageAttnProcessor`) ─────────────────────────────────
pub struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SelfAttention {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            q: linear(w, &join(prefix, "to_q"), false)?,
            k: linear(w, &join(prefix, "to_k"), false)?,
            v: linear(w, &join(prefix, "to_v"), false)?,
            o: linear(w, &join(prefix, "to_out.0"), false)?,
            norm_q: w.get(&join(prefix, "norm_q.weight"))?,
            norm_k: w.get(&join(prefix, "norm_k.weight"))?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim/2]`. Unmasked (B=1 full sequence).
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);

        let q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;

        let q = rmsnorm(&q, &self.norm_q, QK_EPS)?;
        let k = rmsnorm(&k, &self.norm_k, QK_EPS)?;
        let q = apply_interleaved_rope(&q, cos, sin)?;
        let k = apply_interleaved_rope(&k, cos, sin)?;

        let groups = nh / nkv;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = sdpa(&q, &k, &v, self.scale)?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
        self.o.forward(&o)
    }
}

// ── Dual-stream joint attention (`BooguImageDoubleStreamSelfAttnProcessor`) ──────────────────
/// Separate img/instruct QKV projections; the streams are concatenated **instruct-first**, attended
/// jointly, split back, projected by separate `img_out`/`instruct_out`, re-merged, and run through
/// the shared `to_out.0`.
pub struct JointAttention {
    img_q: Linear,
    img_k: Linear,
    img_v: Linear,
    instruct_q: Linear,
    instruct_k: Linear,
    instruct_v: Linear,
    img_out: Linear,
    instruct_out: Linear,
    to_out: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl JointAttention {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let p = |s: &str| join(prefix, s);
        Ok(Self {
            img_q: linear(w, &p("processor.img_to_q"), false)?,
            img_k: linear(w, &p("processor.img_to_k"), false)?,
            img_v: linear(w, &p("processor.img_to_v"), false)?,
            instruct_q: linear(w, &p("processor.instruct_to_q"), false)?,
            instruct_k: linear(w, &p("processor.instruct_to_k"), false)?,
            instruct_v: linear(w, &p("processor.instruct_to_v"), false)?,
            img_out: linear(w, &p("processor.img_out"), false)?,
            instruct_out: linear(w, &p("processor.instruct_out"), false)?,
            to_out: linear(w, &p("to_out.0"), false)?,
            norm_q: w.get(&p("norm_q.weight"))?,
            norm_k: w.get(&p("norm_k.weight"))?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `img`: `[b, Li, D]`, `instruct`: `[b, Lt, D]`, joint `cos`/`sin`: `[1, Lt+Li, head_dim/2]`.
    /// Returns the joint attention output `[b, Lt+Li, D]` (instruct-first).
    pub fn forward(
        &self,
        img: &Tensor,
        instruct: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, li, _) = img.dims3()?;
        let lt = instruct.dim(1)?;
        let hd = self.head_dim;
        let to_heads = |x: &Tensor, proj: &Linear, n: usize, l: usize| -> Result<Tensor> {
            proj.forward(x)?.reshape((b, l, n, hd))
        };

        // Concatenate instruct-first along the sequence axis.
        let q = Tensor::cat(
            &[
                to_heads(instruct, &self.instruct_q, self.heads, lt)?,
                to_heads(img, &self.img_q, self.heads, li)?,
            ],
            1,
        )?;
        let k = Tensor::cat(
            &[
                to_heads(instruct, &self.instruct_k, self.kv_heads, lt)?,
                to_heads(img, &self.img_k, self.kv_heads, li)?,
            ],
            1,
        )?;
        let v = Tensor::cat(
            &[
                to_heads(instruct, &self.instruct_v, self.kv_heads, lt)?,
                to_heads(img, &self.img_v, self.kv_heads, li)?,
            ],
            1,
        )?;

        let q = rmsnorm(&q, &self.norm_q, QK_EPS)?;
        let k = rmsnorm(&k, &self.norm_k, QK_EPS)?;
        let q = apply_interleaved_rope(&q, cos, sin)?;
        let k = apply_interleaved_rope(&k, cos, sin)?;

        let groups = self.heads / self.kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = sdpa(&q, &k, &v, self.scale)?;
        let o = o
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, lt + li, self.heads * hd))?;

        // Split → separate output projections → re-merge → shared output projection. Contiguate the
        // `narrow`ed slices before the Linear matmuls.
        let instruct_part = o.narrow(1, 0, lt)?.contiguous()?;
        let img_part = o.narrow(1, lt, li)?.contiguous()?;
        let merged = Tensor::cat(
            &[
                self.instruct_out.forward(&instruct_part)?,
                self.img_out.forward(&img_part)?,
            ],
            1,
        )?;
        self.to_out.forward(&merged)
    }
}

// ── SwiGLU feed-forward (`LuminaFeedForward`) ───────────────────────────────────────────────
pub struct SwiGlu {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGlu {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: linear(w, &join(prefix, "linear_1"), false)?,
            w2: linear(w, &join(prefix, "linear_2"), false)?,
            w3: linear(w, &join(prefix, "linear_3"), false)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.w1.forward(x)?.silu()? * self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }
}

// ── LuminaRMSNormZero modulation ────────────────────────────────────────────────────────────
/// `emb = linear(silu(temb))` (`1024 → 4·D`), chunked into 4; the returned hidden is
/// `RMSNorm(x)·(1 + scale_msa)`. The caller reuses the other three chunks per its pattern.
pub struct ModNorm {
    linear: Linear,
    norm: Tensor,
    eps: f64,
}

impl ModNorm {
    pub fn load(w: &Weights, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            linear: linear(w, &join(prefix, "linear"), true)?,
            norm: w.get(&join(prefix, "norm.weight"))?,
            eps,
        })
    }

    /// `x`: `[b, s, D]`, `temb`: `[b, 1, 1024]`. Returns `(normed, c1, c2, c3)` where `c1..c3` are
    /// chunks 1/2/3 (each `[b, 1, D]`) and `normed` is `[b, s, D]` = `RMSNorm(x)·(1 + chunk0)`.
    pub fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let emb = self.linear.forward(&temb.silu()?)?; // [b, 1, 4D]
        let chunks = emb.chunk(4, D::Minus1)?;
        let scale_msa = (chunks[0].contiguous()? + 1.0)?;
        let normed = rmsnorm(x, &self.norm, self.eps)?.broadcast_mul(&scale_msa)?;
        Ok((
            normed,
            chunks[1].contiguous()?,
            chunks[2].contiguous()?,
            chunks[3].contiguous()?,
        ))
    }
}

// ── Plain (non-modulated) block — context refiner ───────────────────────────────────────────
pub struct PlainBlock {
    attn: SelfAttention,
    ff: SwiGlu,
    norm1: Tensor,
    norm2: Tensor,
    ffn_norm1: Tensor,
    ffn_norm2: Tensor,
    eps: f64,
}

impl PlainBlock {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::load(w, &join(prefix, "attn"), heads, kv_heads, head_dim)?,
            ff: SwiGlu::load(w, &join(prefix, "feed_forward"))?,
            norm1: w.get(&join(prefix, "norm1.weight"))?,
            norm2: w.get(&join(prefix, "norm2.weight"))?,
            ffn_norm1: w.get(&join(prefix, "ffn_norm1.weight"))?,
            ffn_norm2: w.get(&join(prefix, "ffn_norm2.weight"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let attn = self
            .attn
            .forward(&rmsnorm(x, &self.norm1, self.eps)?, cos, sin)?;
        let x = (x + rmsnorm(&attn, &self.norm2, self.eps)?)?;
        let mlp = self.ff.forward(&rmsnorm(&x, &self.ffn_norm1, self.eps)?)?;
        &x + rmsnorm(&mlp, &self.ffn_norm2, self.eps)?
    }
}

// ── Modulated single-stream / noise-refiner block ───────────────────────────────────────────
pub struct ModBlock {
    attn: SelfAttention,
    ff: SwiGlu,
    norm1: ModNorm,
    norm2: Tensor,
    ffn_norm1: Tensor,
    ffn_norm2: Tensor,
    eps: f64,
}

impl ModBlock {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::load(w, &join(prefix, "attn"), heads, kv_heads, head_dim)?,
            ff: SwiGlu::load(w, &join(prefix, "feed_forward"))?,
            norm1: ModNorm::load(w, &join(prefix, "norm1"), eps)?,
            norm2: w.get(&join(prefix, "norm2.weight"))?,
            ffn_norm1: w.get(&join(prefix, "ffn_norm1.weight"))?,
            ffn_norm2: w.get(&join(prefix, "ffn_norm2.weight"))?,
            eps,
        })
    }

    /// `x`: `[b, s, D]`, `temb`: `[b, 1, 1024]`.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let (normed, gate_msa, scale_mlp, gate_mlp) = self.norm1.forward(x, temb)?;
        let attn = self.attn.forward(&normed, cos, sin)?;
        let x = (x + gate_msa
            .tanh()?
            .broadcast_mul(&rmsnorm(&attn, &self.norm2, self.eps)?)?)?;
        let mlp_in = rmsnorm(&x, &self.ffn_norm1, self.eps)?.broadcast_mul(&(scale_mlp + 1.0)?)?;
        let mlp = self.ff.forward(&mlp_in)?;
        &x + gate_mlp
            .tanh()?
            .broadcast_mul(&rmsnorm(&mlp, &self.ffn_norm2, self.eps)?)?
    }
}

// ── Double-stream block ─────────────────────────────────────────────────────────────────────
pub struct DoubleBlock {
    joint_attn: JointAttention,
    self_attn: SelfAttention,
    img_ff: SwiGlu,
    instruct_ff: SwiGlu,
    img_norm1: ModNorm,
    img_norm2: ModNorm,
    img_norm3: ModNorm,
    instruct_norm1: ModNorm,
    instruct_norm2: ModNorm,
    img_attn_norm: Tensor,
    img_self_attn_norm: Tensor,
    img_ffn_norm1: Tensor,
    img_ffn_norm2: Tensor,
    instruct_attn_norm: Tensor,
    instruct_ffn_norm1: Tensor,
    instruct_ffn_norm2: Tensor,
    eps: f64,
}

impl DoubleBlock {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        let req = |s: &str| w.get(&join(prefix, s));
        Ok(Self {
            joint_attn: JointAttention::load(
                w,
                &join(prefix, "img_instruct_attn"),
                heads,
                kv_heads,
                head_dim,
            )?,
            self_attn: SelfAttention::load(
                w,
                &join(prefix, "img_self_attn"),
                heads,
                kv_heads,
                head_dim,
            )?,
            img_ff: SwiGlu::load(w, &join(prefix, "img_feed_forward"))?,
            instruct_ff: SwiGlu::load(w, &join(prefix, "instruct_feed_forward"))?,
            img_norm1: ModNorm::load(w, &join(prefix, "img_norm1"), eps)?,
            img_norm2: ModNorm::load(w, &join(prefix, "img_norm2"), eps)?,
            img_norm3: ModNorm::load(w, &join(prefix, "img_norm3"), eps)?,
            instruct_norm1: ModNorm::load(w, &join(prefix, "instruct_norm1"), eps)?,
            instruct_norm2: ModNorm::load(w, &join(prefix, "instruct_norm2"), eps)?,
            img_attn_norm: req("img_attn_norm.weight")?,
            img_self_attn_norm: req("img_self_attn_norm.weight")?,
            img_ffn_norm1: req("img_ffn_norm1.weight")?,
            img_ffn_norm2: req("img_ffn_norm2.weight")?,
            instruct_attn_norm: req("instruct_attn_norm.weight")?,
            instruct_ffn_norm1: req("instruct_ffn_norm1.weight")?,
            instruct_ffn_norm2: req("instruct_ffn_norm2.weight")?,
            eps,
        })
    }

    /// `img`: `[b, Li, D]`, `instruct`: `[b, Lt, D]`; joint `cos`/`sin`: `[1, Lt+Li, head_dim/2]`;
    /// image `img_cos`/`img_sin`: `[1, Li, head_dim/2]`; `temb`: `[b, 1, 1024]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        instruct: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        img_cos: &Tensor,
        img_sin: &Tensor,
        temb: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let lt = instruct.dim(1)?;
        let li = img.dim(1)?;

        let (img_n1, img_gate_msa, img_scale_mlp, img_gate_mlp) =
            self.img_norm1.forward(img, temb)?;
        let (img_n2, img_shift_mlp, _, _) = self.img_norm2.forward(img, temb)?;
        let (img_n3, img_gate_self, _, _) = self.img_norm3.forward(img, temb)?;
        let (ins_n1, ins_gate_msa, ins_scale_mlp, ins_gate_mlp) =
            self.instruct_norm1.forward(instruct, temb)?;
        let (ins_n2, ins_shift_mlp, _, _) = self.instruct_norm2.forward(instruct, temb)?;

        // Joint instruct↔img attention, then split back to the two streams.
        let joint = self.joint_attn.forward(&img_n1, &ins_n1, cos, sin)?;
        let instruct_attn_out = joint.narrow(1, 0, lt)?;
        let img_attn_out = joint.narrow(1, lt, li)?;

        // Image self-attention.
        let img_self_out = self.self_attn.forward(&img_n3, img_cos, img_sin)?;

        // Image residual updates.
        let img = (img
            + img_gate_msa.tanh()?.broadcast_mul(&rmsnorm(
                &img_attn_out,
                &self.img_attn_norm,
                self.eps,
            )?)?)?;
        let img = (&img
            + img_gate_self.tanh()?.broadcast_mul(&rmsnorm(
                &img_self_out,
                &self.img_self_attn_norm,
                self.eps,
            )?)?)?;
        let img_mlp_in = img_n2
            .broadcast_mul(&(img_scale_mlp + 1.0)?)?
            .broadcast_add(&img_shift_mlp)?;
        let img_mlp = self
            .img_ff
            .forward(&rmsnorm(&img_mlp_in, &self.img_ffn_norm1, self.eps)?)?;
        let img = (&img
            + img_gate_mlp.tanh()?.broadcast_mul(&rmsnorm(
                &img_mlp,
                &self.img_ffn_norm2,
                self.eps,
            )?)?)?;

        // Instruction residual updates.
        let instruct = (instruct
            + ins_gate_msa.tanh()?.broadcast_mul(&rmsnorm(
                &instruct_attn_out,
                &self.instruct_attn_norm,
                self.eps,
            )?)?)?;
        let ins_mlp_in = ins_n2
            .broadcast_mul(&(ins_scale_mlp + 1.0)?)?
            .broadcast_add(&ins_shift_mlp)?;
        let ins_mlp =
            self.instruct_ff
                .forward(&rmsnorm(&ins_mlp_in, &self.instruct_ffn_norm1, self.eps)?)?;
        let instruct = (&instruct
            + ins_gate_mlp.tanh()?.broadcast_mul(&rmsnorm(
                &ins_mlp,
                &self.instruct_ffn_norm2,
                self.eps,
            )?)?)?;

        Ok((img, instruct))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn chunked_sdpa_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single pass — the guard for the i32-overflow fix (sc-7523).
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let scale = (d as f64).powf(-0.5);
        // Huge budget → single pass; tiny budget (1) → chunked into single-row blocks.
        let single = sdpa_budgeted(&q, &k, &v, scale, usize::MAX).unwrap();
        let chunked = sdpa_budgeted(&q, &k, &v, scale, 1).unwrap();
        assert_eq!(single.dims(), chunked.dims());
        let a = single.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let c = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&c) {
            assert!((x - y).abs() < 1e-6, "chunked sdpa diverged: {x} vs {y}");
        }
    }
}
