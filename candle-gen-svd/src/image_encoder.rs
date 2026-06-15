//! SVD image-conditioning encoder — the `CLIPVisionModelWithProjection` (OpenCLIP ViT-H/14) that
//! turns the input frame into the `image_embeds` fed to the UNet cross-attention. candle port of
//! diffusers `CLIPVisionTransformer` + the `CLIPVisionModelWithProjection` head (the IP-Adapter image
//! tower): patch-conv (bias-free) + CLS `class_embedding` + position embedding + `pre_layrnorm` → 32
//! encoder layers (gelu MLP) → CLS pool → `post_layernorm` → `visual_projection` (1280→1024, no bias).
//!
//! Mirrors `mlx-gen-svd`'s `image_encoder.rs` (which reuses `mlx-gen-sdxl`'s ViT-H body). Structured on
//! `candle_nn` primitives following the `candle-gen-joycaption` SigLIP tower template.

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{
    conv2d_no_bias, embedding, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, LayerNorm,
    Linear, Module, VarBuilder,
};

use crate::config::ImageEncoderConfig;

/// One ViT encoder layer: `LN1 → MHA → +res → LN2 → gelu-MLP → +res`.
struct EncoderLayer {
    layer_norm1: LayerNorm,
    attn: Attention,
    layer_norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncoderLayer {
    fn load(cfg: &ImageEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            layer_norm1: layer_norm(h, cfg.layer_norm_eps, vb.pp("layer_norm1"))?,
            attn: Attention::load(cfg, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(h, cfg.layer_norm_eps, vb.pp("layer_norm2"))?,
            fc1: linear(h, cfg.intermediate_size, vb.pp("mlp").pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, h, vb.pp("mlp").pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.layer_norm1.forward(x)?;
        let x = (x + self.attn.forward(&y)?)?;
        let y = self.layer_norm2.forward(&x)?;
        // laion CLIP-ViT-H uses exact (erf) gelu.
        let y = self.fc2.forward(&self.fc1.forward(&y)?.gelu_erf()?)?;
        &x + y
    }
}

/// Bidirectional multi-head self-attention (CLIP vision, no causal mask).
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(cfg: &ImageEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let head_dim = h / cfg.num_attention_heads;
        Ok(Self {
            q: linear(h, h, vb.pp("q_proj"))?,
            k: linear(h, h, vb.pp("k_proj"))?,
            v: linear(h, h, vb.pp("v_proj"))?,
            out: linear(h, h, vb.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, n, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q.forward(x)?)?;
        let k = split(self.k.forward(x)?)?;
        let v = split(self.v.forward(x)?)?;
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn = softmax_last_dim(&attn)?;
        let out =
            attn.matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, n, self.num_heads * self.head_dim))?;
        self.out.forward(&out)
    }
}

/// The SVD image encoder: the ViT-H body + the projection head.
pub struct SvdImageEncoder {
    patch_embedding: Conv2d,
    class_embedding: Tensor,    // [hidden]
    position_embedding: Tensor, // [1, num_pos, hidden]
    pre_layrnorm: LayerNorm,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    visual_projection: Linear, // [proj, hidden], no bias
    hidden_size: usize,
}

impl SvdImageEncoder {
    /// Build from the SVD `image_encoder/model.safetensors` VarBuilder (`vision_model.*` body +
    /// top-level `visual_projection.weight`).
    pub fn new(cfg: &ImageEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let vm = vb.pp("vision_model");
        let emb = vm.pp("embeddings");
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d_no_bias(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            emb.pp("patch_embedding"),
        )?;
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2);
        let num_pos = num_patches + 1; // + CLS
        let class_embedding = emb.get(cfg.hidden_size, "class_embedding")?;
        let position_embedding = embedding(num_pos, cfg.hidden_size, emb.pp("position_embedding"))?
            .embeddings()
            .reshape((1, num_pos, cfg.hidden_size))?;
        let pre_layrnorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vm.pp("pre_layrnorm"))?;
        let enc = vm.pp("encoder").pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| EncoderLayer::load(cfg, enc.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vm.pp("post_layernorm"))?;
        let visual_projection = linear_no_bias(
            cfg.hidden_size,
            cfg.projection_dim,
            vb.pp("visual_projection"),
        )?;
        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            pre_layrnorm,
            layers,
            post_layernorm,
            visual_projection,
            hidden_size: cfg.hidden_size,
        })
    }

    /// `pixel_values` NCHW `[B, 3, 224, 224]` (CLIP-normalized) → `image_embeds` `[B, projection_dim]`
    /// (f32). Mirrors diffusers `image_encoder(image).image_embeds`: run the tower → CLS token →
    /// `post_layernorm` → `visual_projection`.
    pub fn image_embeds(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let dtype = self.patch_embedding.weight().dtype();
        let pixel_values = pixel_values.to_dtype(dtype)?;
        // Patch-embed: [B, 3, 224, 224] → [B, hidden, 16, 16] → [B, 256, hidden].
        let patches = self.patch_embedding.forward(&pixel_values)?;
        let (b, c, h, w) = patches.dims4()?;
        let patches = patches.reshape((b, c, h * w))?.transpose(1, 2)?; // [B, 256, hidden]
                                                                        // Prepend the CLS class_embedding → [B, 257, hidden].
        let cls = self
            .class_embedding
            .reshape((1, 1, self.hidden_size))?
            .broadcast_as((b, 1, self.hidden_size))?;
        let x = Tensor::cat(&[&cls, &patches], 1)?;
        let mut x = x.broadcast_add(&self.position_embedding)?;
        x = self.pre_layrnorm.forward(&x)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        // CLS pool → post_layernorm → visual_projection.
        let cls = x.narrow(1, 0, 1)?.squeeze(1)?; // [B, hidden]
        let pooled = self.post_layernorm.forward(&cls)?;
        let embeds = self.visual_projection.forward(&pooled)?; // [B, proj]
        embeds.to_dtype(DType::F32)
    }
}
