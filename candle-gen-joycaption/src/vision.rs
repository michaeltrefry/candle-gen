//! SigLIP-so400m vision tower + image processor for JoyCaption (candle port).
//!
//! JoyCaption's LLaVA vision tower is `google/siglip2-so400m-patch14-384`. This module ports the
//! image preprocessing (PIL-bicubic resize → `[-1, 1]` normalize → NCHW pixels) and the
//! hidden-state-producing vision transformer. JoyCaption reads the **penultimate** hidden state
//! ([`VISION_FEATURE_LAYER`] = `-2`) with the `"full"` select strategy (all 729 patch tokens), so
//! `forward` collects the embeddings output plus one tensor per encoder layer and returns layer
//! `-2` — the final `post_layernorm` is never applied (and its weights are not loaded).
//!
//! candle-transformers ships a SigLIP `VisionModel`, but its `Encoder`/`EncoderLayer` are private
//! and it only returns the post-layernorm output — it cannot surface the intermediate `-2` hidden
//! state JoyCaption needs — so the tower is ported here on top of `candle_nn` primitives.

use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::candle_nn::{
    self, conv2d, layer_norm, linear, ops::softmax_last_dim, Conv2d, LayerNorm, Linear, Module,
    VarBuilder,
};
use candle_gen::gen_core::imageops::resize_bicubic_u8;
use candle_gen::gen_core::Image;

pub const SIGLIP_IMAGE_SIZE: usize = 384;
pub const SIGLIP_PATCH_SIZE: usize = 14;
pub const SIGLIP_HIDDEN_SIZE: usize = 1152;
pub const SIGLIP_INTERMEDIATE_SIZE: usize = 4304;
pub const SIGLIP_NUM_LAYERS: usize = 27;
pub const SIGLIP_NUM_HEADS: usize = 16;
pub const SIGLIP_LAYER_NORM_EPS: f64 = 1e-6;
/// SigLIP normalizes RGB to `[-1, 1]` (mean 0.5, std 0.5 per channel).
pub const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];
/// JoyCaption reads the **penultimate** SigLIP hidden state (HF `vision_feature_layer = -2`).
pub const VISION_FEATURE_LAYER: i32 = -2;

/// SigLIP-so400m/14@384 geometry: `(384 / 14)^2 = 27^2 = 729` patch tokens.
#[derive(Clone, Copy, Debug)]
pub struct SiglipVisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_eps: f64,
}

impl Default for SiglipVisionConfig {
    fn default() -> Self {
        Self {
            image_size: SIGLIP_IMAGE_SIZE,
            patch_size: SIGLIP_PATCH_SIZE,
            num_channels: 3,
            hidden_size: SIGLIP_HIDDEN_SIZE,
            intermediate_size: SIGLIP_INTERMEDIATE_SIZE,
            num_hidden_layers: SIGLIP_NUM_LAYERS,
            num_attention_heads: SIGLIP_NUM_HEADS,
            layer_norm_eps: SIGLIP_LAYER_NORM_EPS,
        }
    }
}

impl SiglipVisionConfig {
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub fn num_patches(&self) -> usize {
        self.grid() * self.grid()
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// RGB uint8 → SigLIP pixel tensor `[1, 3, 384, 384]` (NCHW), bicubic-resized and `[-1, 1]`-normalized.
#[derive(Clone, Debug)]
pub struct SiglipImageProcessor {
    pub size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for SiglipImageProcessor {
    fn default() -> Self {
        Self {
            size: SIGLIP_IMAGE_SIZE,
            mean: SIGLIP_MEAN,
            std: SIGLIP_STD,
        }
    }
}

impl SiglipImageProcessor {
    /// Preprocess an RGB [`Image`] into a `[1, 3, size, size]` pixel tensor on `device` at `dtype`.
    /// Resize is PIL-`BICUBIC`-exact (`gen_core::imageops`), matching the reference image processor.
    pub fn preprocess(&self, image: &Image, device: &Device, dtype: DType) -> Result<Tensor> {
        let (w, h) = (image.width as usize, image.height as usize);
        let expected = w * h * 3;
        if image.pixels.len() != expected {
            return Err(Error::Msg(format!(
                "joycaption siglip: expected {expected} RGB pixels for {w}x{h}, got {}",
                image.pixels.len()
            )));
        }
        // Resized HWC f32 in [0, 255].
        let resized: Vec<f32> = if w == self.size && h == self.size {
            image.pixels.iter().map(|&p| p as f32).collect()
        } else {
            resize_bicubic_u8(&image.pixels, h, w, self.size, self.size)
        };

        // HWC [0,255] → NCHW normalized: (v/255 - mean) / std.
        let plane = self.size * self.size;
        let mut chw = vec![0f32; 3 * plane];
        for y in 0..self.size {
            for x in 0..self.size {
                let px = (y * self.size + x) * 3;
                for c in 0..3 {
                    let v = resized[px + c];
                    chw[c * plane + y * self.size + x] = (v / 255.0 - self.mean[c]) / self.std[c];
                }
            }
        }
        Tensor::from_vec(chw, (1, 3, self.size, self.size), device)?.to_dtype(dtype)
    }
}

/// The SigLIP vision tower. Loads `vision_tower.vision_model.*` (the `post_layernorm` is skipped —
/// JoyCaption's `-2` feature layer is taken before it).
pub struct SiglipVisionTower {
    patch_embedding: Conv2d,
    /// Position embedding `[1, num_patches, hidden]`, added to the patch tokens.
    position_embedding: Tensor,
    layers: Vec<SiglipEncoderLayer>,
    cfg: SiglipVisionConfig,
}

impl SiglipVisionTower {
    /// `vb` points at the HF `vision_model` module (e.g. `vision_tower.vision_model`).
    pub fn new(cfg: SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let emb = vb.pp("embeddings");
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            emb.pp("patch_embedding"),
        )?;
        let position_embedding = candle_nn::embedding(
            cfg.num_patches(),
            cfg.hidden_size,
            emb.pp("position_embedding"),
        )?
        .embeddings()
        .reshape((1, cfg.num_patches(), cfg.hidden_size))?;
        let enc = vb.pp("encoder").pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| SiglipEncoderLayer::new(&cfg, enc.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding,
            position_embedding,
            layers,
            cfg,
        })
    }

    fn embeddings(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // [b, 3, 384, 384] → conv → [b, hidden, 27, 27] → [b, 729, hidden] + position.
        let x = self.patch_embedding.forward(pixel_values)?;
        let (b, c, h, w) = x.dims4()?;
        let x = x.reshape((b, c, h * w))?.transpose(1, 2)?;
        x.broadcast_add(&self.position_embedding)
    }

    /// Run the tower and return the **penultimate** hidden state (`[b, 729, hidden]`), the feature
    /// JoyCaption projects into the Llama hidden size.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embeddings(pixel_values)?;
        // HF hidden_states: embeddings output, then one per encoder layer (before post_layernorm).
        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(hidden.clone());
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
            hidden_states.push(hidden.clone());
        }
        let len = hidden_states.len() as i32;
        let idx = (len + VISION_FEATURE_LAYER).rem_euclid(len) as usize;
        Ok(hidden_states.swap_remove(idx))
    }

    pub fn config(&self) -> &SiglipVisionConfig {
        &self.cfg
    }
}

struct SiglipEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: SiglipAttention,
    layer_norm2: LayerNorm,
    mlp: SiglipMlp,
}

impl SiglipEncoderLayer {
    fn new(cfg: &SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm1"))?,
            self_attn: SiglipAttention::new(cfg, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm2"))?,
            mlp: SiglipMlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.layer_norm1.forward(x)?;
        let x = (x + self.self_attn.forward(&y)?)?;
        let y = self.layer_norm2.forward(&x)?;
        &x + self.mlp.forward(&y)?
    }
}

struct SiglipAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SiglipAttention {
    fn new(cfg: &SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            q_proj: linear(h, h, vb.pp("q_proj"))?,
            k_proj: linear(h, h, vb.pp("k_proj"))?,
            v_proj: linear(h, h, vb.pp("v_proj"))?,
            out_proj: linear(h, h, vb.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            scale: (cfg.head_dim() as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, n, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q_proj.forward(x)?)?;
        let k = split(self.k_proj.forward(x)?)?;
        let v = split(self.v_proj.forward(x)?)?;
        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = softmax_last_dim(&attn)?;
        let out =
            attn.matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, n, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

struct SiglipMlp {
    fc1: Linear,
    fc2: Linear,
}

impl SiglipMlp {
    fn new(cfg: &SiglipVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SigLIP uses gelu_pytorch_tanh (the tanh approximation), which is candle's `Tensor::gelu`.
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::IndexOp;

    #[test]
    fn config_matches_joycaption_siglip() {
        let cfg = SiglipVisionConfig::default();
        assert_eq!(cfg.num_patches(), 729);
        assert_eq!(cfg.head_dim(), 72);
        assert_eq!(cfg.grid(), 27);
    }

    #[test]
    fn preprocess_normalizes_rgb_to_minus_one_one() {
        let image = Image {
            width: 384,
            height: 384,
            pixels: [0u8, 128, 255].repeat(384 * 384),
        };
        let t = SiglipImageProcessor::default()
            .preprocess(&image, &Device::Cpu, DType::F32)
            .expect("preprocess");
        assert_eq!(t.dims(), &[1, 3, 384, 384]);
        // Channel 0 is all 0 → -1.0; channel 2 is all 255 → 1.0.
        let r = t.i((0, 0, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
        let b = t.i((0, 2, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(r, -1.0);
        assert_eq!(b, 1.0);
    }

    #[test]
    fn preprocess_rejects_bad_rgb_buffer() {
        let image = Image {
            width: 2,
            height: 2,
            pixels: vec![0u8; 3],
        };
        assert!(SiglipImageProcessor::default()
            .preprocess(&image, &Device::Cpu, DType::F32)
            .is_err());
    }

    #[test]
    fn feature_layer_index_resolves_to_penultimate() {
        // 28 hidden states (embeddings + 27 layers); -2 → index 26.
        let len = (SIGLIP_NUM_LAYERS + 1) as i32;
        let idx = (len + VISION_FEATURE_LAYER).rem_euclid(len);
        assert_eq!(idx, 26);
    }
}
