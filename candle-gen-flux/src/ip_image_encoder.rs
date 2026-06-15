//! The XLabs FLUX IP-Adapter **image encoder** (sc-5872, epic 5480): the CLIP **ViT-L/14** tower +
//! its pooled-projection head, producing the `[1, 768]` image embedding the [`crate::ip_adapter`]
//! `FluxImageProjModel` consumes.
//!
//! This is the **classic** (non-"plus") IP-Adapter image path, so — unlike the SDXL/Kolors towers,
//! which feed the **penultimate** hidden states into a Resampler — XLabs FLUX uses the **pooled,
//! projected** CLIP embedding: the full tower's `last_hidden_state` → class token (position 0) →
//! `post_layernorm` → `visual_projection` (`openai/clip-vit-large-patch14`'s `image_embeds`). The
//! tower itself is the reused [`ClipVisionEncoder`] (`VisionConfig::vit_l_14`); the projection head
//! (`post_layernorm` + `visual_projection`) lives here because the SDXL tower omits it (it only ever
//! needs the penultimate features).

use candle_core::{IndexOp, Tensor};
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::gen_core::Image;
use candle_gen::Result as GenResult;
use candle_gen_sdxl::ip_adapter::preprocess_clip_image_sized;
use candle_gen_sdxl::vision_encoder::check_layer_count;
use candle_gen_sdxl::weights::Weights;
use candle_gen_sdxl::{ClipVisionEncoder, VisionConfig};

/// CLIP LayerNorm epsilon (the same `post_layernorm` eps the tower uses).
const LN_EPS: f64 = 1e-5;

/// The XLabs FLUX IP-Adapter image encoder: a CLIP ViT-L/14 tower + the pooled-projection head.
pub struct FluxIpImageEncoder {
    encoder: ClipVisionEncoder,
    /// `vision_model.post_layernorm` — applied to the class token of `last_hidden_state`.
    post_ln: LayerNorm,
    /// `visual_projection` (`Linear(1024 → 768)`, **no bias**) — the pooled→image-embed head.
    visual_projection: Linear,
    /// The CLIP crop size (224 for ViT-L/14).
    image_size: usize,
}

impl FluxIpImageEncoder {
    /// Load the ViT-L/14 tower + the projection head from an `openai/clip-vit-large-patch14`-layout
    /// checkpoint (`vision_model.*` + `visual_projection.weight`). The checkpoint's layer count is
    /// validated against the ViT-L config (catches a ViT-H/ViT-L mixup loudly).
    pub fn from_weights(w: &Weights) -> GenResult<Self> {
        let cfg = VisionConfig::vit_l_14();
        check_layer_count(w, &cfg)?;
        let encoder = ClipVisionEncoder::from_weights(w, &cfg)?;
        let post_ln = LayerNorm::new(
            w.require("vision_model.post_layernorm.weight")?,
            w.require("vision_model.post_layernorm.bias")?,
            LN_EPS,
        );
        // `visual_projection` is bias-free in CLIP.
        let visual_projection = Linear::new(w.require("visual_projection.weight")?, None);
        Ok(Self {
            encoder,
            post_ln,
            visual_projection,
            image_size: cfg.image_size,
        })
    }

    /// Encode `image` → the pooled CLIP image embedding `[1, 768]` (at the tower's weight dtype):
    /// preprocess → full tower (`last_hidden_state`) → class token → `post_layernorm` →
    /// `visual_projection`.
    pub fn image_embeds(&self, image: &Image) -> GenResult<Tensor> {
        let device = self.visual_projection.weight().device().clone();
        let dtype = self.encoder.dtype();
        let px = preprocess_clip_image_sized(image, self.image_size, &device)?.to_dtype(dtype)?;
        let last = self.encoder.last_hidden(&px)?; // [1, num_positions, 1024]
        let cls = last.i((.., 0))?; // [1, 1024] — the class token
        let pooled = self.post_ln.forward(&cls)?;
        Ok(self.visual_projection.forward(&pooled)?) // [1, 768]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    /// Build a tiny synthetic CLIP-vision checkpoint (random weights) + the projection head, enough to
    /// drive `from_weights` + `image_embeds`. Mirrors `candle-gen-sdxl::vision_encoder`'s `tiny_weights`.
    fn tiny_checkpoint(cfg: &VisionConfig, proj_dim: usize, dev: &Device) -> Weights {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let randn = |shape: &[usize]| Tensor::randn(0f32, 1f32, shape, dev).unwrap();
        let p = "vision_model";
        m.insert(
            format!("{p}.embeddings.patch_embedding.weight"),
            randn(&[cfg.hidden, cfg.num_channels, cfg.patch, cfg.patch]),
        );
        m.insert(
            format!("{p}.embeddings.class_embedding"),
            randn(&[cfg.hidden]),
        );
        m.insert(
            format!("{p}.embeddings.position_embedding.weight"),
            randn(&[cfg.num_positions(), cfg.hidden]),
        );
        m.insert(format!("{p}.pre_layrnorm.weight"), randn(&[cfg.hidden]));
        m.insert(format!("{p}.pre_layrnorm.bias"), randn(&[cfg.hidden]));
        let mlp = cfg.hidden * 4;
        for i in 0..cfg.num_layers {
            let l = format!("{p}.encoder.layers.{i}");
            for ln in ["layer_norm1", "layer_norm2"] {
                m.insert(format!("{l}.{ln}.weight"), randn(&[cfg.hidden]));
                m.insert(format!("{l}.{ln}.bias"), randn(&[cfg.hidden]));
            }
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                m.insert(
                    format!("{l}.self_attn.{proj}.weight"),
                    randn(&[cfg.hidden, cfg.hidden]),
                );
                m.insert(format!("{l}.self_attn.{proj}.bias"), randn(&[cfg.hidden]));
            }
            m.insert(format!("{l}.mlp.fc1.weight"), randn(&[mlp, cfg.hidden]));
            m.insert(format!("{l}.mlp.fc1.bias"), randn(&[mlp]));
            m.insert(format!("{l}.mlp.fc2.weight"), randn(&[cfg.hidden, mlp]));
            m.insert(format!("{l}.mlp.fc2.bias"), randn(&[cfg.hidden]));
        }
        // The projection head this module loads (the SDXL tower omits these).
        m.insert(format!("{p}.post_layernorm.weight"), randn(&[cfg.hidden]));
        m.insert(format!("{p}.post_layernorm.bias"), randn(&[cfg.hidden]));
        m.insert(
            "visual_projection.weight".into(),
            randn(&[proj_dim, cfg.hidden]),
        );
        Weights::from_map(m)
    }

    /// `from_weights` wires the tower + projection head, and `image_embeds` produces a finite
    /// `[1, proj_dim]` pooled embedding (tiny dims; the real 1024→768 shape is the GPU validation).
    #[test]
    fn image_embeds_shape_and_finite() {
        let dev = Device::Cpu;
        // A tiny ViT-L-shaped tower: hidden 16, 2 layers, 2 heads, patch 2, 8px → 5×... grid 4 → 17.
        let cfg = VisionConfig {
            hidden: 16,
            num_layers: 2,
            num_heads: 2,
            patch: 2,
            image_size: 8,
            num_channels: 3,
            quick_gelu: true,
        };
        let proj_dim = 6;
        let w = tiny_checkpoint(&cfg, proj_dim, &dev);
        // `from_weights` hardcodes vit_l_14 (24 layers); build the encoder directly for the tiny tower.
        let encoder = ClipVisionEncoder::from_weights(&w, &cfg).unwrap();
        let post_ln = LayerNorm::new(
            w.require("vision_model.post_layernorm.weight").unwrap(),
            w.require("vision_model.post_layernorm.bias").unwrap(),
            LN_EPS,
        );
        let visual_projection = Linear::new(w.require("visual_projection.weight").unwrap(), None);
        let enc = FluxIpImageEncoder {
            encoder,
            post_ln,
            visual_projection,
            image_size: cfg.image_size,
        };
        let img = Image {
            width: 10,
            height: 7,
            pixels: vec![128u8; 10 * 7 * 3],
        };
        let embeds = enc.image_embeds(&img).unwrap();
        assert_eq!(embeds.dims(), &[1, proj_dim]);
        assert!(embeds
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
}
