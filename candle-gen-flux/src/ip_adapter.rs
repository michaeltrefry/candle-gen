//! The XLabs FLUX **IP-Adapter** weights (sc-5872, epic 5480) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-flux`'s `ip_adapter`. Unlike the SDXL/Kolors IP-Adapter-**Plus** families (a perceiver
//! `Resampler` over penultimate CLIP features), the XLabs FLUX adapter is the **classic** IP-Adapter
//! shape:
//!
//! - an [`FluxImageProjModel`] — a single `Linear(768 → 4·4096)` + `LayerNorm` projecting the **pooled**
//!   CLIP image embedding `[B, 768]` into **4** image-prompt tokens of width 4096 ([`crate::ip_image_encoder`]
//!   produces the pooled embed);
//! - **19** per-double-block decoupled K/V projectors (`Linear(4096 → 3072)`) — one pair per FLUX
//!   double block, applied as a decoupled cross-attention whose residual is injected into the image
//!   stream ([`crate::ip_dit`]).
//!
//! [`FluxIpInjector`] precomputes the 4 image tokens once (constant across the denoise) and exposes
//! [`double_block_residual`](FluxIpInjector::double_block_residual) — the per-block residual the forked
//! DiT adds. With `scale == 0.0` every residual is `None`, so the forked DiT renders byte-identically to
//! the plain (stock) FLUX path — that is the no-IP arm of the validation ablation.

use candle_core::{Result, Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::CandleError;
use candle_gen::Result as GenResult;
use candle_gen_sdxl::weights::Weights;

/// Image-prompt token count (the XLabs `ImageProjModel` emits 4 tokens).
const NUM_TOKENS: usize = 4;
/// Image-prompt token width = the FLUX context width the K/V projectors consume (4096).
const CROSS_ATTN_DIM: usize = 4096;
/// FLUX has 19 double-stream blocks; the XLabs adapter carries exactly one K/V pair per block.
const NUM_DOUBLE_BLOCKS: usize = 19;
/// The `ImageProjModel` LayerNorm epsilon (diffusers default).
const PROJ_LN_EPS: f64 = 1e-5;

/// Projects the pooled CLIP image embedding `[B, 768]` into `[B, 4, 4096]` image-prompt tokens — a
/// `Linear(768 → 4·4096)` + a per-token `LayerNorm(4096)`. The XLabs `ip_adapter_proj_model`.
struct FluxImageProjModel {
    /// `Linear(clip_embed_dim → num_tokens·cross_attn_dim)` (+ bias).
    proj: Linear,
    /// `LayerNorm(cross_attn_dim)` applied per token.
    norm: LayerNorm,
    num_tokens: usize,
    cross_attn_dim: usize,
}

impl FluxImageProjModel {
    fn from_weights(w: &Weights, prefix: &str) -> GenResult<Self> {
        let proj = Linear::new(
            w.require(&format!("{prefix}.proj.weight"))?,
            Some(w.require(&format!("{prefix}.proj.bias"))?),
        );
        let norm = LayerNorm::new(
            w.require(&format!("{prefix}.norm.weight"))?,
            w.require(&format!("{prefix}.norm.bias"))?,
            PROJ_LN_EPS,
        );
        Ok(Self {
            proj,
            norm,
            num_tokens: NUM_TOKENS,
            cross_attn_dim: CROSS_ATTN_DIM,
        })
    }

    /// `[B, clip_embed_dim]` pooled CLIP image embeds → `[B, num_tokens, cross_attn_dim]` tokens.
    fn forward(&self, image_embeds: &Tensor) -> Result<Tensor> {
        let b = image_embeds.dim(0)?;
        let x = self.proj.forward(image_embeds)?; // [B, num_tokens·cross_attn_dim]
        let x = x.reshape((b, self.num_tokens, self.cross_attn_dim))?;
        self.norm.forward(&x)
    }
}

/// The XLabs FLUX IP-Adapter weights: the [`FluxImageProjModel`] + the 19 per-double-block decoupled
/// K/V projectors (`Linear(cross_attn_dim → hidden_size)` each). Loaded from the
/// `XLabs-AI/flux-ip-adapter` `ip_adapter.safetensors`.
pub struct FluxIpAdapter {
    proj_model: FluxImageProjModel,
    /// `(k_proj, v_proj)` per double block, in block order — `double_blocks.{i}.processor.\
    /// ip_adapter_double_stream_{k,v}_proj`. Each `Linear(4096 → 3072)` (+ bias).
    blocks: Vec<(Linear, Linear)>,
}

impl FluxIpAdapter {
    /// Load from the XLabs `ip_adapter.safetensors` (`Weights` key→Tensor map). Validates that the
    /// checkpoint carries exactly the 19 double-block adapters FLUX has (rejecting a longer one loudly,
    /// rather than silently ignoring extra pairs).
    pub fn from_weights(w: &Weights) -> GenResult<Self> {
        let proj_model = FluxImageProjModel::from_weights(w, "ip_adapter_proj_model")?;
        let mut blocks = Vec::with_capacity(NUM_DOUBLE_BLOCKS);
        for i in 0..NUM_DOUBLE_BLOCKS {
            let p = format!("double_blocks.{i}.processor.ip_adapter_double_stream");
            let k = Linear::new(
                w.require(&format!("{p}_k_proj.weight"))?,
                Some(w.require(&format!("{p}_k_proj.bias"))?),
            );
            let v = Linear::new(
                w.require(&format!("{p}_v_proj.weight"))?,
                Some(w.require(&format!("{p}_v_proj.bias"))?),
            );
            blocks.push((k, v));
        }
        let extra = format!(
            "double_blocks.{NUM_DOUBLE_BLOCKS}.processor.ip_adapter_double_stream_k_proj.weight"
        );
        if w.contains(&extra) {
            return Err(CandleError::Msg(format!(
                "flux ip-adapter: checkpoint carries more than {NUM_DOUBLE_BLOCKS} double-block \
                 adapters"
            )));
        }
        Ok(Self { proj_model, blocks })
    }

    /// Project the pooled CLIP image embedding `[B, 768]` into the `[B, 4, 4096]` image-prompt tokens.
    pub fn tokens(&self, image_embeds: &Tensor) -> Result<Tensor> {
        self.proj_model.forward(image_embeds)
    }

    /// The number of adapted double blocks (= 19 for a valid XLabs checkpoint).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

/// A bound IP injector for one render: the adapter + the precomputed image tokens `[B, 4, 4096]` + the
/// `ip_adapter_scale`. The forked DiT calls [`double_block_residual`](Self::double_block_residual) per
/// double block. The tokens are computed once (they do not change across the denoise), so the per-block
/// cost is just the K/V projection + a tiny (4-key) cross-attention.
pub struct FluxIpInjector<'a> {
    adapter: &'a FluxIpAdapter,
    /// `[B, num_tokens, cross_attn_dim]` image-prompt tokens (precomputed once).
    tokens: Tensor,
    /// The `ip_adapter_scale` weight applied to the decoupled-cross-attention residual. `0.0` ⇒ every
    /// block residual is `None` (the no-IP ablation arm).
    scale: f64,
}

impl<'a> FluxIpInjector<'a> {
    /// Bind `adapter` to the precomputed image `tokens` (`[B, num_tokens, cross_attn_dim]`) at `scale`.
    pub fn new(adapter: &'a FluxIpAdapter, tokens: Tensor, scale: f64) -> Self {
        Self {
            adapter,
            tokens,
            scale,
        }
    }

    /// The decoupled-cross-attention residual for double block `block_idx`, given that block's
    /// **post-QkNorm, pre-RoPE** image query `img_q` `[B, heads, img_seq, head_dim]`. Returns the
    /// residual `[B, img_seq, heads·head_dim]` scaled by `ip_adapter_scale`, or `None` when the scale is
    /// 0 or the block has no adapter. The image tokens carry no RoPE (position-less keys), so the query
    /// attends them un-rotated (diffusers' `FluxIPAdapterAttnProcessor`).
    pub fn double_block_residual(
        &self,
        block_idx: usize,
        img_q: &Tensor,
    ) -> Result<Option<Tensor>> {
        if self.scale == 0.0 || block_idx >= self.adapter.blocks.len() {
            return Ok(None);
        }
        let (k_proj, v_proj) = &self.adapter.blocks[block_idx];
        let (b, num_tokens, _) = self.tokens.dims3()?;
        let (_, heads, img_seq, head_dim) = img_q.dims4()?;
        // K/V projection of the image tokens → per-head `[B, heads, num_tokens, head_dim]`.
        let to_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, num_tokens, heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let k = to_heads(k_proj.forward(&self.tokens)?)?;
        let v = to_heads(v_proj.forward(&self.tokens)?)?;
        // Decoupled cross-attention (no RoPE, no mask): scale = 1/√head_dim, matching the FLUX heads.
        let q = img_q.contiguous()?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?;
        // `[B, heads, img_seq, head_dim]` → `[B, img_seq, hidden]`. `transpose` makes the layout
        // non-contiguous; `contiguous` before `reshape` matches the SDXL vision-encoder pattern (GPU
        // layout strictness — sc-5488).
        let o = o
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, img_seq, heads * head_dim))?;
        Ok(Some((o * self.scale)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn randn(shape: &[usize], dev: &Device) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, dev).unwrap()
    }

    /// `FluxImageProjModel::forward` projects `[B, embed]` → `[B, num_tokens, cross_attn_dim]` and
    /// LayerNorms each token (tiny dims; the real 768→4·4096 shape is the GPU validation).
    #[test]
    fn image_proj_model_shapes() {
        let dev = Device::Cpu;
        let (embed, num_tokens, cross) = (6usize, 2usize, 5usize);
        let proj = Linear::new(
            randn(&[num_tokens * cross, embed], &dev),
            Some(randn(&[num_tokens * cross], &dev)),
        );
        let norm = LayerNorm::new(randn(&[cross], &dev), randn(&[cross], &dev), PROJ_LN_EPS);
        let m = FluxImageProjModel {
            proj,
            norm,
            num_tokens,
            cross_attn_dim: cross,
        };
        let tokens = m.forward(&randn(&[3, embed], &dev)).unwrap();
        assert_eq!(tokens.dims(), &[3, num_tokens, cross]);
        assert!(tokens
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    /// `double_block_residual` produces `[B, img_seq, heads·head_dim]` from a per-head image query and
    /// the image tokens; `scale = 0` short-circuits to `None` (the no-IP ablation), as does an
    /// out-of-range block index. Tiny, shape-derived dims (no real 4096/3072 allocation).
    #[test]
    fn double_block_residual_shapes_and_gates() {
        let dev = Device::Cpu;
        let (b, num_tokens, cross) = (1usize, 4usize, 8usize);
        let (heads, head_dim, img_seq) = (2usize, 3usize, 5usize);
        let hidden = heads * head_dim; // 6
                                       // One-block adapter: k/v project cross → hidden.
        let mk = || Linear::new(randn(&[hidden, cross], &dev), Some(randn(&[hidden], &dev)));
        let proj = Linear::new(
            randn(&[num_tokens * cross, cross], &dev),
            Some(randn(&[num_tokens * cross], &dev)),
        );
        let proj_model = FluxImageProjModel {
            proj,
            norm: LayerNorm::new(randn(&[cross], &dev), randn(&[cross], &dev), PROJ_LN_EPS),
            num_tokens,
            cross_attn_dim: cross,
        };
        let adapter = FluxIpAdapter {
            proj_model,
            blocks: vec![(mk(), mk())],
        };
        let tokens = randn(&[b, num_tokens, cross], &dev);
        let img_q = randn(&[b, heads, img_seq, head_dim], &dev);

        // scale > 0, valid block → a finite residual of the right shape.
        let inj = FluxIpInjector::new(&adapter, tokens.clone(), 0.7);
        let r = inj
            .double_block_residual(0, &img_q)
            .unwrap()
            .expect("residual");
        assert_eq!(r.dims(), &[b, img_seq, hidden]);
        assert!(r
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));

        // Out-of-range block index → None.
        assert!(inj.double_block_residual(1, &img_q).unwrap().is_none());

        // scale == 0 → None for every block (the no-IP arm).
        let off = FluxIpInjector::new(&adapter, tokens, 0.0);
        assert!(off.double_block_residual(0, &img_q).unwrap().is_none());
        assert_eq!(adapter.num_blocks(), 1);
    }
}
