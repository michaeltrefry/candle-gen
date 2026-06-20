//! SCAIL-2's CLIP image encoder — the open-CLIP **XLM-RoBERTa ViT-H/14** *visual tower* (upstream
//! `wan/modules/clip.py` `VisionTransformer`, the "onlyvisual" checkpoint).
//!
//! The reference image is encoded to the `[1, 257, 1280]` features the DiT's `img_emb` consumes
//! (`Scail2Dit`'s `clip_fea`). This is the **`use_31_block=True`** path: patch-embed → prepend cls →
//! add pos → pre-norm → run only the **first 31 of 32** transformer blocks, returning the
//! **penultimate** hidden state — *no* `post_norm`, *no* `head` projection, *no* pooling.
//!
//! A standard pre-norm ViT: Conv2d(3→1280, 14×14, stride 14, no bias) read as an `[out, 3·14·14]`
//! Linear (stride==kernel, like the DiT patch stems); 32 blocks with `x = x + attn(norm1(x))` then
//! `x = x + mlp(norm2(x))`; **fused `to_qkv`** (`[3·dim, dim]`); **exact GELU**; LayerNorm eps 1e-5.
//! Runs entirely in f32 (the encoder is small and conditioning-only).
//!
//! Image preprocessing (224² bicubic resize, `[-1,1]→[0,1]`, CLIP mean/std normalize) is the caller's
//! concern ([`crate::resize::clip_preprocess`]); [`ScailClip::encode`] takes a preprocessed
//! `[B, 3, 224, 224]` pixel tensor.

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};

use crate::common::{conv_as_linear, linear, ln_affine, sdpa};

/// open-CLIP ViT visual-tower geometry. The shipped SCAIL-2 CLIP is always [`ClipVisionConfig::vit_h_14`].
#[derive(Clone, Debug)]
pub struct ClipVisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub dim: usize,
    pub num_heads: usize,
    /// Total transformer blocks (32 for ViT-H/14). The `use_31_block` path runs `num_layers - 1`.
    pub num_layers: usize,
    /// FFN inner dim (mlp_ratio 4 → 5120 for ViT-H/14).
    pub mlp_dim: usize,
    pub eps: f64,
}

impl ClipVisionConfig {
    /// open-CLIP XLM-RoBERTa ViT-H/14 (the SCAIL-2 / Wan-I2V image encoder).
    pub fn vit_h_14() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            dim: 1280,
            num_heads: 16,
            num_layers: 32,
            mlp_dim: 5120,
            eps: 1e-5,
        }
    }

    /// Blocks actually executed by the `use_31_block` penultimate path.
    pub fn run_layers(&self) -> usize {
        self.num_layers - 1
    }

    /// Tokens per image (`(image_size / patch_size)² + 1` cls).
    pub fn num_tokens(&self) -> usize {
        let p = self.image_size / self.patch_size;
        p * p + 1
    }
}

/// One pre-norm ViT block: `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`.
struct ClipBlock {
    norm1_w: Tensor,
    norm1_b: Tensor,
    to_qkv: Linear,
    proj: Linear,
    norm2_w: Tensor,
    norm2_b: Tensor,
    mlp0: Linear,
    mlp2: Linear,
    n: usize,
    d: usize,
    scale: f64,
    eps: f64,
}

impl ClipBlock {
    fn new(vb: &VarBuilder, i: usize, cfg: &ClipVisionConfig) -> Result<Self> {
        let p = vb.pp("transformer").pp(i);
        let head_dim = cfg.dim / cfg.num_heads;
        Ok(Self {
            norm1_w: p.pp("norm1").get(cfg.dim, "weight")?,
            norm1_b: p.pp("norm1").get(cfg.dim, "bias")?,
            to_qkv: linear(cfg.dim, 3 * cfg.dim, p.pp("attn").pp("to_qkv"))?,
            proj: linear(cfg.dim, cfg.dim, p.pp("attn").pp("proj"))?,
            norm2_w: p.pp("norm2").get(cfg.dim, "weight")?,
            norm2_b: p.pp("norm2").get(cfg.dim, "bias")?,
            mlp0: linear(cfg.dim, cfg.mlp_dim, p.pp("mlp").pp("0"))?,
            mlp2: linear(cfg.mlp_dim, cfg.dim, p.pp("mlp").pp("2"))?,
            n: cfg.num_heads,
            d: head_dim,
            scale: (head_dim as f64).powf(-0.5),
            eps: cfg.eps,
        })
    }

    fn attn(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (n, d) = (self.n, self.d);
        // Fused qkv: [b, s, 3·dim] → [b, s, 3, n, d], split the "3" axis.
        let qkv = self.to_qkv.forward(x)?.reshape((b, s, 3, n, d))?;
        let parts = qkv.chunk(3, 2)?;
        let head = |t: &Tensor| -> Result<Tensor> {
            t.reshape((b, s, n, d))?.transpose(1, 2)?.contiguous()
        };
        let q = head(&parts[0])?;
        let k = head(&parts[1])?;
        let v = head(&parts[2])?;
        let out = sdpa(&q, &k, &v, self.scale)?;
        let out = out.transpose(1, 2)?.reshape((b, s, n * d))?;
        self.proj.forward(&out)
    }

    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        self.mlp2.forward(&self.mlp0.forward(x)?.gelu_erf()?)
    }

    /// `x`: `[B, L, dim]` (f32).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = ln_affine(x, &self.norm1_w, &self.norm1_b, self.eps)?;
        let x = (x + self.attn(&h)?)?;
        let h = ln_affine(&x, &self.norm2_w, &self.norm2_b, self.eps)?;
        x + self.mlp(&h)?
    }
}

/// The SCAIL-2 CLIP visual tower (penultimate-feature extractor).
pub struct ScailClip {
    patch_embedding: Linear, // Conv2d(3→dim, p×p) as [dim, 3·p·p]
    cls: Tensor,             // [1, 1, dim]
    pos: Tensor,             // [1, num_tokens, dim]
    pre_norm_w: Tensor,
    pre_norm_b: Tensor,
    blocks: Vec<ClipBlock>, // run_layers (31 for ViT-H/14)
    cfg: ClipVisionConfig,
}

impl ScailClip {
    /// Load the visual tower from a `VarBuilder` over the de-prefixed `onlyvisual` state dict
    /// (`patch_embedding.weight`, `cls_embedding`, `pos_embedding`, `pre_norm.*`, `transformer.{i}.*`).
    /// Only the `run_layers` blocks the penultimate path needs are loaded — `post_norm`/`head` skipped.
    /// The `VarBuilder` should be f32.
    pub fn new(vb: VarBuilder, cfg: &ClipVisionConfig) -> Result<Self> {
        let patch_embedding = conv_as_linear(
            cfg.dim,
            3,
            &[cfg.patch_size, cfg.patch_size],
            "patch_embedding.weight",
            None,
            &vb,
        )?;
        let mut blocks = Vec::with_capacity(cfg.run_layers());
        for i in 0..cfg.run_layers() {
            blocks.push(ClipBlock::new(&vb, i, cfg)?);
        }
        Ok(Self {
            patch_embedding,
            cls: vb.get((1, 1, cfg.dim), "cls_embedding")?,
            pos: vb.get((1, cfg.num_tokens(), cfg.dim), "pos_embedding")?,
            pre_norm_w: vb.get(cfg.dim, "pre_norm.weight")?,
            pre_norm_b: vb.get(cfg.dim, "pre_norm.bias")?,
            blocks,
            cfg: cfg.clone(),
        })
    }

    /// Encode a preprocessed pixel tensor `[B, 3, image_size, image_size]` (f32) → penultimate CLIP
    /// features `[B, num_tokens, dim]` (e.g. `[1, 257, 1280]` for ViT-H/14 at 224²).
    pub fn encode(&self, pixel: &Tensor) -> Result<Tensor> {
        let p = self.cfg.patch_size;
        let (b, _c, h, wd) = pixel.dims4()?;
        let (nh, nw) = (h / p, wd / p);
        let dim = self.cfg.dim;

        // Patchify [B,3,H,W] → [B, nh·nw, 3·p·p] (feature order (c, kh, kw) matches the conv flatten),
        // then the conv-as-Linear patch embed.
        let tokens = pixel
            .reshape((b, 3, nh, p, nw, p))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b, nh * nw, 3 * p * p))?;
        let x = self.patch_embedding.forward(&tokens)?;

        // Prepend cls, add positional, pre-norm.
        let cls = self.cls.broadcast_as((b, 1, dim))?;
        let x = Tensor::cat(&[&cls, &x], 1)?; // [B, nh·nw+1, dim]
        let x = x.broadcast_add(&self.pos)?;
        let mut x = ln_affine(&x, &self.pre_norm_w, &self.pre_norm_b, self.cfg.eps)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}
