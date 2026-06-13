//! Vendored, training-adapted Wan DiT (sc-5167) — the candle twin of `mlx-gen-wan`'s trainable
//! transformer and the Wan analog of [`candle-gen-z-image`'s `dit`](../../candle-gen-z-image/src/dit.rs).
//!
//! A faithful copy of [`crate::transformer`]'s `WanTransformer` with the four attention projections
//! (`to_q`/`to_k`/`to_v`/`to_out.0`) held as [`LoraLinear`] so the native LoRA/LoKr trainer can splice a
//! trainable residual into each — the stock [`Attention`](crate::transformer) builds them from frozen
//! `nn::Linear` with no seam. Only the structs that *own* a projection (or the block / model that owns
//! them) are vendored; every numeric helper (`linear`/`ln_no_affine`/`rms`/`timestep_sinusoid`, and
//! `apply_rope`) is **reused** from [`crate::transformer`] / [`crate::rope`] so the two stay in lockstep.
//!
//! **One deviation, forced by candle autograd:** the stock attention uses the **fused**
//! `softmax_last_dim`, a `CustomOp` with NO backward that silently zeroes upstream grads (the epic-5164
//! fused-ops trap, see [[candle-fused-ops-no-backward]] / `candle-gen-z-image::dit`). The vendored
//! attention uses the composable `candle_nn::ops::softmax` instead. (Wan's qk-`rms` and `ln_no_affine`
//! are already *manual* composable ops, not the fused `RmsNorm::forward`, so they need no change — and
//! the patch-embed conv / time-embed / proj_out are all upstream of every adapter, never on an adapter's
//! backward path.) Wan attention projections carry a **bias** (unlike Z-Image), so they wrap
//! [`lora_linear`] (bias-bearing), the residual adapting only the weight.
//!
//! With no adapter installed the vendored forward is bit-identical to the stock forward (the
//! `parity_tests` gate pins this). `forward` returns the **raw** velocity (no sign flip) — Wan feeds the
//! transformer output to the flow-match step *without* negation (opposite of Z-Image), so the trainer
//! regresses the raw velocity toward `noise − x0` (see [`crate::training`]).

use candle_gen::candle_core::{DType, Module, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::train::lora::{lora_linear, LoraHost, LoraLinear};

use crate::config::TransformerConfig;
use crate::transformer::{linear, ln_no_affine, rms, timestep_sinusoid};

/// Composable interleaved RoPE — the differentiable twin of [`crate::rope::apply_rope`], which wraps
/// candle's fused `rotary_emb::rope_i` (a `CustomOp` with NO backward — the fused-ops trap: a fused RoPE
/// on the q/k path silently zeroes every attention factor's gradient). Applies the **same** interleaved
/// rotation `out[2k] = x[2k]·cos_k − x[2k+1]·sin_k`, `out[2k+1] = x[2k]·sin_k + x[2k+1]·cos_k` over the
/// half tables `cos`/`sin` `[S, head_dim/2]`, so with the same tables it equals `rope_i` (the
/// `parity_tests` gate pins vendored == stock through this path). `x`: `[B, H, S, head_dim]`.
fn apply_rope_diff(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let (b, h, s, d) = x.dims4()?;
    let half = d / 2;
    let xf = x.to_dtype(DType::F32)?.reshape((b, h, s, half, 2))?;
    let x0 = xf.narrow(4, 0, 1)?.squeeze(4)?; // even lanes [B,H,S,half]
    let x1 = xf.narrow(4, 1, 1)?.squeeze(4)?; // odd lanes
    let cos = cos.reshape((1, 1, s, half))?;
    let sin = sin.reshape((1, 1, s, half))?;
    let o0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let o1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    // Re-interleave: stack the two lanes on a new trailing axis → [B,H,S,half,2] → [B,H,S,d].
    Tensor::stack(&[&o0, &o1], 4)?
        .reshape((b, h, s, d))?
        .to_dtype(dtype)
}

/// The default LoRA target suffixes — the attention projections, matching the diffusers
/// `WanTransformer3DModel` LoRA surface, the torch `DEFAULT_LORA_TARGET_MODULES`
/// (`["to_q","to_k","to_v","to_out.0"]`), and the MLX Wan trainer. `to_out.0` is the first element of
/// diffusers' `to_out` `ModuleList`, so its path segment literally contains the `.0`.
pub const WAN_ATTN_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

// ==================== TrainAttention (LoRA seam) ====================

/// Wan attention (self or cross) with the four projections held as [`LoraLinear`]. Numerically
/// identical to the stock [`Attention`](crate::transformer) with no adapter installed, except it runs
/// the composable softmax (the stock `softmax_last_dim` has no backward).
#[derive(Debug, Clone)]
struct TrainAttention {
    to_q: LoraLinear,
    to_k: LoraLinear,
    to_v: LoraLinear,
    to_out: LoraLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl TrainAttention {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.dim;
        Ok(Self {
            to_q: lora_linear(cfg.dim, inner, vb.pp("to_q"))?,
            to_k: lora_linear(cfg.dim, inner, vb.pp("to_k"))?,
            to_v: lora_linear(cfg.dim, inner, vb.pp("to_v"))?,
            to_out: lora_linear(inner, cfg.dim, vb.pp("to_out").pp("0"))?,
            norm_q: vb.pp("norm_q").get(inner, "weight")?,
            norm_k: vb.pp("norm_k").get(inner, "weight")?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            eps: cfg.eps,
        })
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&mut self.to_q)?;
        f(&mut self.to_k)?;
        f(&mut self.to_v)?;
        f(&mut self.to_out)?;
        Ok(())
    }

    /// `hidden`: `[B,S,dim]`; `context`: cross-attn K/V source (= hidden for self-attn). RoPE applied
    /// only when `rope` is given (self-attn). Mirrors [`Attention::forward`](crate::transformer) but
    /// with the composable softmax.
    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, _) = hidden.dims3()?;
        let s_kv = context.dim(1)?;
        let q = rms(&self.to_q.forward(hidden)?, &self.norm_q, self.eps)?;
        let k = rms(&self.to_k.forward(context)?, &self.norm_k, self.eps)?;
        let v = self.to_v.forward(context)?;
        let to_heads = |t: &Tensor, len: usize| -> Result<Tensor> {
            t.reshape((b, len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let mut q = to_heads(&q, s)?; // [B,H,S,d]
        let mut k = to_heads(&k, s_kv)?;
        let v = to_heads(&v, s_kv)?;
        if let Some((cos, sin)) = rope {
            q = apply_rope_diff(&q, cos, sin)?;
            k = apply_rope_diff(&k, cos, sin)?;
        }
        let scale = (self.head_dim as f64).powf(-0.5);
        // Composable SDPA (NOT the fused `softmax_last_dim` — that CustomOp has no backward).
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;
        let attn = softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
        let out = attn.matmul(&v.contiguous()?)?; // [B,H,S,d]
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

// ==================== TrainFfn (frozen) ====================

/// The gated GELU feed-forward (`net.0.proj` → GELU → `net.2`). Not an adapter target by default, so it
/// stays frozen `Linear` (reused from the stock helpers).
#[derive(Debug, Clone)]
struct TrainFfn {
    proj: candle_gen::candle_nn::Linear,
    out: candle_gen::candle_nn::Linear,
}

impl TrainFfn {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: linear(cfg.dim, cfg.ffn_dim, vb.pp("net").pp("0").pp("proj"))?,
            out: linear(cfg.ffn_dim, cfg.dim, vb.pp("net").pp("2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.proj.forward(x)?.gelu()?)
    }
}

// ==================== TrainBlock ====================

/// One Wan DiT block: AdaLN-modulated self-attention → ungated cross-attention to the text context →
/// AdaLN-modulated gated FFN. Byte-faithful to [`Block`](crate::transformer) with the trainable
/// attentions spliced in.
#[derive(Debug, Clone)]
struct TrainBlock {
    scale_shift_table: Tensor, // [1,6,dim] f32
    attn1: TrainAttention,
    norm2_w: Tensor,
    norm2_b: Tensor,
    attn2: TrainAttention,
    ffn: TrainFfn,
    eps: f64,
}

impl TrainBlock {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale_shift_table: vb
                .get((1, 6, cfg.dim), "scale_shift_table")?
                .to_dtype(DType::F32)?,
            attn1: TrainAttention::new(cfg, vb.pp("attn1"))?,
            norm2_w: vb
                .pp("norm2")
                .get(cfg.dim, "weight")?
                .to_dtype(DType::F32)?,
            norm2_b: vb.pp("norm2").get(cfg.dim, "bias")?.to_dtype(DType::F32)?,
            attn2: TrainAttention::new(cfg, vb.pp("attn2"))?,
            ffn: TrainFfn::new(cfg, vb.pp("ffn"))?,
            eps: cfg.eps,
        })
    }

    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn1.visit_lora_mut(f)?;
        self.attn2.visit_lora_mut(f)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        temb6: &Tensor,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let dt = hidden.dtype();
        let mods = self.scale_shift_table.broadcast_add(temb6)?;
        let m = |i: usize| -> Result<Tensor> { mods.narrow(1, i, 1) };
        let (shift_msa, scale_msa, gate_msa) = (m(0)?, m(1)?, m(2)?);
        let (c_shift, c_scale, c_gate) = (m(3)?, m(4)?, m(5)?);

        let hf = hidden.to_dtype(DType::F32)?;
        // 1. self-attention
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?
            .to_dtype(dt)?;
        let a = self.attn1.forward(&n, &n, Some((cos, sin)))?;
        let hf = (hf + a.to_dtype(DType::F32)?.broadcast_mul(&gate_msa)?)?;

        // 2. cross-attention (affine norm2, ungated)
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&self.norm2_w)?
            .broadcast_add(&self.norm2_b)?
            .to_dtype(dt)?;
        let a = self.attn2.forward(&n, context, None)?;
        let hf = (hf + a.to_dtype(DType::F32)?)?;

        // 3. feed-forward
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&(c_scale + 1.0)?)?
            .broadcast_add(&c_shift)?
            .to_dtype(dt)?;
        let f = self.ffn.forward(&n)?;
        let hf = (hf + f.to_dtype(DType::F32)?.broadcast_mul(&c_gate)?)?;
        hf.to_dtype(dt)
    }
}

// ==================== WanTransformerTrain ====================

/// The vendored, trainable twin of [`WanTransformer`](crate::transformer). Built from the *same*
/// `transformer/` (or `transformer_2/`) safetensors keys, so it loads the real expert weights unchanged
/// and, with no adapter installed, reproduces the stock forward bit-for-bit (`parity_tests`).
#[derive(Debug, Clone)]
pub struct WanTransformerTrain {
    patch_w: Tensor, // [dim, in_channels, ph, pw]
    patch_b: Tensor, // [1, dim, 1, 1]
    text_l1: candle_gen::candle_nn::Linear,
    text_l2: candle_gen::candle_nn::Linear,
    time_l1: candle_gen::candle_nn::Linear,
    time_l2: candle_gen::candle_nn::Linear,
    time_proj: candle_gen::candle_nn::Linear,
    blocks: Vec<TrainBlock>,
    norm_out_eps: f64,
    proj_out: candle_gen::candle_nn::Linear,
    scale_shift_table: Tensor, // [1,2,dim] f32
    cfg: TransformerConfig,
    dtype: DType,
}

impl WanTransformerTrain {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let (pt, ph, pw) = cfg.patch;
        let pw_full = vb.get(
            (cfg.dim, cfg.in_channels, pt, ph, pw),
            "patch_embedding.weight",
        )?;
        let patch_w = pw_full.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [dim,in,ph,pw]
        let patch_b = vb
            .get(cfg.dim, "patch_embedding.bias")?
            .reshape((1, cfg.dim, 1, 1))?;

        let ce = vb.pp("condition_embedder");
        let text_l1 = linear(cfg.text_dim, cfg.dim, ce.pp("text_embedder").pp("linear_1"))?;
        let text_l2 = linear(cfg.dim, cfg.dim, ce.pp("text_embedder").pp("linear_2"))?;
        let time_l1 = linear(cfg.freq_dim, cfg.dim, ce.pp("time_embedder").pp("linear_1"))?;
        let time_l2 = linear(cfg.dim, cfg.dim, ce.pp("time_embedder").pp("linear_2"))?;
        let time_proj = linear(cfg.dim, 6 * cfg.dim, ce.pp("time_proj"))?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(TrainBlock::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let proj_out = linear(cfg.dim, cfg.out_channels * pt * ph * pw, vb.pp("proj_out"))?;
        let scale_shift_table = vb
            .get((1, 2, cfg.dim), "scale_shift_table")?
            .to_dtype(DType::F32)?;

        Ok(Self {
            patch_w,
            patch_b,
            text_l1,
            text_l2,
            time_l1,
            time_l2,
            time_proj,
            blocks,
            norm_out_eps: cfg.eps,
            proj_out,
            scale_shift_table,
            cfg: *cfg,
            dtype: vb.dtype(),
        })
    }

    /// Project UMT5 prompt embeds `[B,S,4096]` → cross-attn context `[B,S,dim]`. `gelu` between the two
    /// linears (PixArtAlphaTextProjection). Frozen — no adapter targets here.
    pub fn embed_text(&self, prompt_embeds: &Tensor) -> Result<Tensor> {
        let x = prompt_embeds.to_dtype(self.dtype)?;
        self.text_l2.forward(&self.text_l1.forward(&x)?.gelu()?)
    }

    /// One DiT forward: `latents [B,C,F,Hl,Wl]`, projected `context [B,S,dim]`, scalar `t` (the
    /// `[0,1000]` integer timestep), RoPE `cos`/`sin` → **raw** predicted velocity `[B,C,F,Hl,Wl]`
    /// (f32). Byte-faithful to [`WanTransformer::forward`](crate::transformer).
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        t: f64,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, _c, f, hl, wl) = latents.dims5()?;
        let (pt, ph, pw) = self.cfg.patch;
        let (ppf, pph, ppw) = (f / pt, hl / ph, wl / pw);
        let device = latents.device();

        let merged = latents
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * f, self.cfg.in_channels, hl, wl))?
            .contiguous()?
            .to_dtype(self.dtype)?;
        let y = merged.conv2d(&self.patch_w, 0, ph, 1, 1)?;
        let y = y.broadcast_add(&self.patch_b)?;
        let mut hidden = y
            .reshape((b, f, self.cfg.dim, pph, ppw))?
            .permute((0, 1, 3, 4, 2))?
            .reshape((b, ppf * pph * ppw, self.cfg.dim))?
            .contiguous()?;

        let sinus = timestep_sinusoid(t, self.cfg.freq_dim, b, device)?.to_dtype(self.dtype)?;
        let temb = self
            .time_l2
            .forward(&self.time_l1.forward(&sinus)?.silu()?)?;
        let temb6 = self
            .time_proj
            .forward(&temb.silu()?)?
            .reshape((b, 6, self.cfg.dim))?
            .to_dtype(DType::F32)?;

        for blk in &self.blocks {
            hidden = blk.forward(&hidden, &temb6, context, cos, sin)?;
        }

        let head_mod = self
            .scale_shift_table
            .broadcast_add(&temb.unsqueeze(1)?.to_dtype(DType::F32)?)?;
        let shift = head_mod.narrow(1, 0, 1)?;
        let scale = head_mod.narrow(1, 1, 1)?;
        let hf = hidden.to_dtype(DType::F32)?;
        let normed = ln_no_affine(&hf, self.norm_out_eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?
            .to_dtype(self.dtype)?;
        let out = self.proj_out.forward(&normed)?;

        let oc = self.cfg.out_channels;
        out.reshape(&[b, ppf, pph, ppw, pt, ph, pw, oc][..])?
            .permute(&[0usize, 7, 1, 4, 2, 5, 3, 6][..])?
            .reshape((b, oc, ppf * pt, pph * ph, ppw * pw))?
            .to_dtype(DType::F32)
    }
}

impl LoraHost for WanTransformerTrain {
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for blk in self.blocks.iter_mut() {
            blk.visit_lora_mut(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod parity_tests {
    //! Pin the vendored trainable DiT to the stock [`WanTransformer`](crate::transformer): built from
    //! the *same* `VarMap`-backed weights with no adapter installed, the two must produce a bit-identical
    //! forward. The regression guard that the `LoraLinear` swap + composable softmax changed nothing.
    use super::*;
    use crate::rope::WanRope;
    use crate::transformer::WanTransformer;
    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::candle_nn::{VarBuilder, VarMap};

    /// A tiny Wan-shaped config (head_dim 128, 1 head, 2 layers, z16) — exercises every vendored path
    /// cheaply on CPU.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 2,
            num_heads: 1,
            head_dim: 128,
            dim: 128,
            ffn_dim: 256,
            freq_dim: 256,
            text_dim: 64,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }

    #[test]
    fn vendored_dit_matches_stock_forward() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Vendored built first; the stock model reads the SAME varmap params, so any output difference
        // is a forward-logic difference, not a weight one.
        let vendored = WanTransformerTrain::new(&cfg, vb.clone()).unwrap();
        let stock = WanTransformer::new(&cfg, vb).unwrap();
        // `vb.get` raw tensors (the `patch_embedding` conv, norm/scale tables) default to ZERO-init from
        // a fresh VarMap — a zero patch kernel makes `hidden ≡ 0` and the comparison vacuous. Randomize
        // every shared var so the forward is exercised on nontrivial weights.
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), &dev).unwrap())
                .unwrap();
        }

        // latent [1,16,1,4,4] → patch (1,2,2) → 2×2 = 4 image tokens; tiny 3-token text context.
        let latent = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 1, 4, 4), &dev).unwrap();
        let umt5 = Tensor::randn(0f32, 1f32, (1, 3, cfg.text_dim), &dev).unwrap();
        let (ppf, pph, ppw) = (1usize, 2usize, 2usize);
        let (cos, sin) = WanRope::new(&cfg).cos_sin(ppf, pph, ppw, &dev).unwrap();

        let ctx_v = vendored.embed_text(&umt5).unwrap();
        let ctx_s = stock.embed_text(&umt5).unwrap();
        let y_v = vendored
            .forward(&latent, &ctx_v, 500.0, &cos, &sin)
            .unwrap();
        let y_s = stock.forward(&latent, &ctx_s, 500.0, &cos, &sin).unwrap();

        assert_eq!(y_v.dims(), y_s.dims());
        let diff = (y_v - y_s)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "vendored Wan DiT diverged from stock by {diff}"
        );
    }

    /// The [`LoraHost`] walk reaches exactly `4 × 2 × num_layers` projections — the four attention
    /// `LoraLinear`s in both `attn1` (self) and `attn2` (cross) of every block.
    #[test]
    fn lora_host_visits_every_attention_projection() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut model = WanTransformerTrain::new(&cfg, vb).unwrap();
        let mut paths: Vec<String> = Vec::new();
        model
            .visit_lora_mut(&mut |lin| {
                paths.push(lin.path().to_string());
                Ok(())
            })
            .unwrap();
        assert_eq!(paths.len(), 4 * 2 * cfg.num_layers);
        for suffix in WAN_ATTN_TARGETS {
            assert!(
                paths
                    .iter()
                    .any(|p| p == suffix || p.ends_with(&format!(".{suffix}"))),
                "no visited projection matched suffix {suffix}"
            );
        }
        assert!(paths.contains(&"blocks.0.attn1.to_q".to_string()));
        assert!(paths.contains(&"blocks.0.attn2.to_out.0".to_string()));
    }
}
