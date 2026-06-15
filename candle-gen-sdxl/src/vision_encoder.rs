//! CLIP **ViT vision tower** — the IP-Adapter image encoder (sc-5488, epic 5480), the candle
//! (Windows/CUDA) sibling of `mlx-gen-sdxl::vision_encoder`. This is the one net-new module the
//! general IP-Adapter port needs: candle-gen had no CLIP *image* encoder (only the text tower, via
//! candle-transformers, drives the SDXL/FLUX conditioners).
//!
//! The transformer body is the **same** pre-norm self-attention + MLP stack as the CLIP *text*
//! encoder — identical `self_attn.{q,k,v,out}_proj` / `mlp.fc1|fc2` naming — so only the *front*
//! differs: a patch-conv + class-token + learned-position embedding (`vision_model.embeddings`) and a
//! `pre_layrnorm` (HF's spelling), and there is **no causal mask** (full bidirectional attention).
//! IP-Adapter "plus" consumes the **penultimate** hidden state (`hidden_states[-2]`, raw — before
//! `post_layernorm` / `visual_projection`), so [`penultimate`](ClipVisionEncoder::penultimate) runs
//! only the first `num_layers - 1` encoder layers (that output *is* `hidden_states[-2]`).
//!
//! **candle vs mlx port note.** The patch embedding is a `conv2d`. mlx is NHWC/OHWI so the MLX port
//! transposes the conv weight `OIHW → OHWI` on load; **candle `Tensor::conv2d` is NCHW/OIHW = the
//! on-disk diffusers layout**, so there is NO transpose and the input pixel tensor is NCHW
//! `[B, 3, H, W]` (the [`crate::ip_adapter::preprocess_clip_image_sized`] output) directly. SDPA runs
//! in f32 then casts back (the f16 production path is identity-directional, mirroring the Resampler).

use candle_core::{DType, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::{CandleError, Result};

use crate::weights::Weights;

/// CLIP's LayerNorm epsilon (shared with the text encoder).
const LN_EPS: f64 = 1e-5;

/// CLIP vision tower config. Defaults are the ViT-H/14 used by `h94/IP-Adapter`'s `image_encoder`.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub patch: usize,
    pub image_size: usize,
    pub num_channels: usize,
    /// MLP activation: `quick_gelu` (`x·sigmoid(1.702x)`, OpenAI CLIP-L) vs exact `gelu` (the laion
    /// OpenCLIP ViT-H tower). `config.json` `hidden_act` — they differ, and over the tower's depth the
    /// wrong one drifts the embeds (~0.93 cosine vs torch on ViT-L; mlx-gen sc-3622).
    pub quick_gelu: bool,
}

impl VisionConfig {
    /// OpenCLIP ViT-H/14 (`h94/IP-Adapter` `models/image_encoder/config.json`): 1280-wide, 32 layers,
    /// 16 heads (head_dim 80), patch 14, 224px → 257 tokens, **exact** gelu, LN eps 1e-5. The image
    /// tower the SDXL IP-Adapter-Plus (`ip-adapter-plus_sdxl_vit-h`) conditions on.
    pub fn vit_h_14() -> Self {
        Self {
            hidden: 1280,
            num_layers: 32,
            num_heads: 16,
            patch: 14,
            image_size: 224,
            num_channels: 3,
            quick_gelu: false,
        }
    }

    /// OpenAI CLIP ViT-L/14 (`openai/clip-vit-large-patch14`, `vision_model.*`): 1024-wide, 24 layers,
    /// 16 heads (head_dim 64), patch 14, 224px → 257 tokens, `quick_gelu`. The image tower the XLabs
    /// FLUX IP-Adapter conditions on (the 1024→768 projection head lives in the consumer, not here).
    pub fn vit_l_14() -> Self {
        Self {
            hidden: 1024,
            num_layers: 24,
            num_heads: 16,
            patch: 14,
            image_size: 224,
            num_channels: 3,
            quick_gelu: true,
        }
    }

    /// OpenAI CLIP ViT-L/14-**336** (`openai/clip-vit-large-patch14-336`): identical to
    /// [`vit_l_14`](Self::vit_l_14) but 336px → 24×24 patches → 577 tokens. The image tower the
    /// **Kolors** IP-Adapter-Plus conditions on (`quick_gelu`, LN eps 1e-5).
    pub fn vit_l_14_336() -> Self {
        Self {
            hidden: 1024,
            num_layers: 24,
            num_heads: 16,
            patch: 14,
            image_size: 336,
            num_channels: 3,
            quick_gelu: true,
        }
    }

    /// Token count = 1 class token + (image_size / patch)² patches (= 257 for ViT-H/14).
    pub fn num_positions(&self) -> usize {
        let grid = self.image_size / self.patch;
        grid * grid + 1
    }
}

/// `candle_nn::LayerNorm` from `{prefix}.weight` + `{prefix}.bias`.
fn layer_norm(w: &Weights, prefix: &str) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        w.require(&format!("{prefix}.weight"))?,
        w.require(&format!("{prefix}.bias"))?,
        LN_EPS,
    ))
}

/// `candle_nn::Linear` (`[out, in]` weight + bias) from `{prefix}.weight` / `{prefix}.bias`. Every
/// CLIP attention/MLP projection is biased.
fn linear(w: &Weights, prefix: &str) -> Result<Linear> {
    Ok(Linear::new(
        w.require(&format!("{prefix}.weight"))?,
        Some(w.require(&format!("{prefix}.bias"))?),
    ))
}

/// QuickGELU `x · sigmoid(1.702·x)` (OpenAI CLIP-L/ViT-L). Computed in the input dtype to mirror the
/// MLX `gelu_quick` (the working dtype, not an f32 upcast).
fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    // sigmoid(1.702x) = 1 / (1 + exp(-1.702x)); affine(mul, add) computes mul·t + add.
    let s = x
        .affine(1.702, 0.0)?
        .neg()?
        .exp()?
        .affine(1.0, 1.0)?
        .recip()?;
    Ok((x * s)?)
}

/// One CLIP vision encoder layer (pre-norm self-attention + pre-norm MLP, both residual). Mirrors the
/// CLIP text encoder layer; the only structural difference upstream is the absence of a causal mask.
struct VisionEncoderLayer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    quick_gelu: bool,
}

impl VisionEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let head_dim = cfg.hidden / cfg.num_heads;
        Ok(Self {
            ln1: layer_norm(w, &format!("{prefix}.layer_norm1"))?,
            ln2: layer_norm(w, &format!("{prefix}.layer_norm2"))?,
            q: linear(w, &format!("{prefix}.self_attn.q_proj"))?,
            k: linear(w, &format!("{prefix}.self_attn.k_proj"))?,
            v: linear(w, &format!("{prefix}.self_attn.v_proj"))?,
            out: linear(w, &format!("{prefix}.self_attn.out_proj"))?,
            fc1: linear(w, &format!("{prefix}.mlp.fc1"))?,
            fc2: linear(w, &format!("{prefix}.mlp.fc2"))?,
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            quick_gelu: cfg.quick_gelu,
        })
    }

    /// `[B, N, D]` → `[B, heads, N, head_dim]`.
    fn to_heads(&self, a: &Tensor) -> Result<Tensor> {
        let (b, n, _) = a.dims3()?;
        Ok(a.reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// Bidirectional multi-head self-attention (no mask). Scores/softmax in f32, cast back.
    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let in_dtype = x.dtype();
        let q = self.to_heads(&self.q.forward(x)?)?.to_dtype(DType::F32)?;
        let k = self.to_heads(&self.k.forward(x)?)?.to_dtype(DType::F32)?;
        let v = self.to_heads(&self.v.forward(x)?)?.to_dtype(DType::F32)?;
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * self.scale)?;
        let probs = softmax(&scores, D::Minus1)?;
        let o = probs.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
        let o = o
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, self.num_heads * self.head_dim))?;
        self.out.forward(&o).map_err(Into::into)
    }

    /// `x`: `[B, N, D]`. Pre-norm attention residual, then pre-norm MLP residual.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.attention(&self.ln1.forward(x)?)?;
        let x = (x + y)?;
        let y = self.fc1.forward(&self.ln2.forward(&x)?)?;
        let y = if self.quick_gelu {
            quick_gelu(&y)?
        } else {
            y.gelu_erf()?
        };
        let y = self.fc2.forward(&y)?;
        Ok((x + y)?)
    }
}

/// A loaded CLIP ViT vision tower (transformer body + patch/class/position embeddings + `pre_layrnorm`).
pub struct ClipVisionEncoder {
    /// Patch conv weight, candle NCHW `[hidden, channels, patch, patch]` (no bias) — the on-disk layout.
    patch_embedding: Tensor,
    /// `[hidden]` learned class token.
    class_embedding: Tensor,
    /// `[num_positions, hidden]` learned position table.
    position_embedding: Tensor,
    pre_ln: LayerNorm,
    layers: Vec<VisionEncoderLayer>,
    patch: usize,
    hidden: usize,
}

impl ClipVisionEncoder {
    /// Load from an `image_encoder` checkpoint (`vision_model.*` prefix). The patch conv weight stays
    /// NCHW `[out, in, kH, kW]` (no transpose — candle `conv2d` is NCHW, unlike the MLX NHWC port).
    pub fn from_weights(w: &Weights, cfg: &VisionConfig) -> Result<Self> {
        let p = "vision_model";
        Ok(Self {
            patch_embedding: w.require(&format!("{p}.embeddings.patch_embedding.weight"))?,
            class_embedding: w.require(&format!("{p}.embeddings.class_embedding"))?,
            position_embedding: w.require(&format!("{p}.embeddings.position_embedding.weight"))?,
            pre_ln: layer_norm(w, &format!("{p}.pre_layrnorm"))?,
            layers: (0..cfg.num_layers)
                .map(|i| {
                    VisionEncoderLayer::from_weights(w, &format!("{p}.encoder.layers.{i}"), cfg)
                })
                .collect::<Result<Vec<_>>>()?,
            patch: cfg.patch,
            hidden: cfg.hidden,
        })
    }

    /// Embed `pixel_values` (candle NCHW `[B, 3, H, W]`, CLIP-normalized) → `[B, num_positions, hidden]`,
    /// then `pre_layrnorm`. This is HF's `hidden_states[0]` (the input to encoder layer 0).
    fn embed(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let b = pixel_values.dim(0)?;
        // Patch conv: stride = kernel = patch, no padding/dilation, single group, no bias →
        // [B, hidden, grid, grid].
        let patches = pixel_values.conv2d(&self.patch_embedding, 0, self.patch, 1, 1)?;
        let (_, _, gh, gw) = patches.dims4()?;
        let n_patch = gh * gw;
        // [B, hidden, grid, grid] → [B, hidden, n_patch] → [B, n_patch, hidden].
        let patches = patches
            .reshape((b, self.hidden, n_patch))?
            .transpose(1, 2)?
            .contiguous()?;
        // Prepend the class token, broadcast over the batch → [B, 1, hidden].
        let cls = self
            .class_embedding
            .reshape((1, 1, self.hidden))?
            .broadcast_as((b, 1, self.hidden))?;
        let x = Tensor::cat(&[&cls, &patches], 1)?; // [B, 1+n_patch, hidden]
                                                    // Add the learned position table (one row per token).
        let num_pos = self.position_embedding.dim(0)?;
        let pos = self.position_embedding.reshape((1, num_pos, self.hidden))?;
        let x = x.broadcast_add(&pos)?;
        self.pre_ln.forward(&x).map_err(Into::into)
    }

    /// The **penultimate** hidden state `[B, num_positions, hidden]` — the IP-Adapter "plus" image
    /// features fed to the Resampler. Runs only the first `num_layers - 1` encoder layers: that output
    /// *is* `hidden_states[-2]`, so forwarding the whole tower and discarding the last layer would be
    /// wasted work.
    pub fn penultimate(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.embed(pixel_values)?;
        let keep = self.layers.len().saturating_sub(1);
        for layer in &self.layers[..keep] {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    /// The **last** hidden state `[B, num_positions, hidden]` — HF's `last_hidden_state` (the full
    /// encoder output, before `post_layernorm` / `visual_projection`). Runs *all* `num_layers` encoder
    /// layers, unlike [`penultimate`](Self::penultimate). This is what the classic (non-"plus")
    /// IP-Adapter image embedding needs: the consumer takes the class token (position 0) of this output,
    /// applies `post_layernorm`, then `visual_projection` to get the pooled `image_embeds` (the
    /// projection head lives in the consumer — e.g. the XLabs FLUX IP-Adapter's `FluxIpImageEncoder` —
    /// not in this tower, sc-5872).
    pub fn last_hidden(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.embed(pixel_values)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    /// The compute dtype (the patch-embedding weight's dtype).
    pub fn dtype(&self) -> DType {
        self.patch_embedding.dtype()
    }
}

/// Reject a vision-encoder checkpoint whose `vision_model.encoder.layers.{n}` count disagrees with the
/// config — a wrong config (ViT-H vs ViT-L) would otherwise load a truncated/over-long tower and only
/// surface as a silent quality regression.
pub fn check_layer_count(w: &Weights, cfg: &VisionConfig) -> Result<()> {
    let found = w
        .keys()
        .filter_map(|k| {
            k.strip_prefix("vision_model.encoder.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .map(|m| m + 1);
    match found {
        Some(n) if n == cfg.num_layers => Ok(()),
        Some(n) => Err(CandleError::Msg(format!(
            "clip vision encoder: checkpoint has {n} layers but the config expects {} (wrong \
             ViT-H/ViT-L variant?)",
            cfg.num_layers
        ))),
        None => Err(CandleError::Msg(
            "clip vision encoder: no `vision_model.encoder.layers.{n}` keys found".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    /// Build a tiny synthetic CLIP-vision checkpoint for `cfg` (random weights), enough to drive
    /// `from_weights` + `penultimate`.
    fn tiny_weights(cfg: &VisionConfig, dev: &Device) -> Weights {
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
        let inner = cfg.hidden;
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
                    randn(&[inner, cfg.hidden]),
                );
                m.insert(format!("{l}.self_attn.{proj}.bias"), randn(&[inner]));
            }
            m.insert(format!("{l}.mlp.fc1.weight"), randn(&[mlp, cfg.hidden]));
            m.insert(format!("{l}.mlp.fc1.bias"), randn(&[mlp]));
            m.insert(format!("{l}.mlp.fc2.weight"), randn(&[cfg.hidden, mlp]));
            m.insert(format!("{l}.mlp.fc2.bias"), randn(&[cfg.hidden]));
        }
        Weights::from_map(m)
    }

    /// `num_positions` = 1 + (image_size/patch)²: 257 for ViT-H/14 (224/14=16), 577 for ViT-L/14-336.
    #[test]
    fn num_positions_matches_grid() {
        assert_eq!(VisionConfig::vit_h_14().num_positions(), 257);
        assert_eq!(VisionConfig::vit_l_14().num_positions(), 257);
        assert_eq!(VisionConfig::vit_l_14_336().num_positions(), 577);
    }

    /// The tower forwards a synthetic (tiny-config) checkpoint to a finite penultimate
    /// `[B, num_positions, hidden]` — exercising the patch conv (NCHW, no transpose), the class/pos
    /// embeddings, `pre_layrnorm`, and the bidirectional encoder layers (both gelu variants). Numerical
    /// parity vs the real OpenCLIP weights is the GPU validation; this pins the port's structure.
    #[test]
    fn penultimate_shape_and_finite() {
        let dev = Device::Cpu;
        // A tiny ViT: hidden 32, 3 layers, 4 heads, patch 2, 8px → grid 4 → 17 tokens.
        let cfg = VisionConfig {
            hidden: 32,
            num_layers: 3,
            num_heads: 4,
            patch: 2,
            image_size: 8,
            num_channels: 3,
            quick_gelu: false,
        };
        let w = tiny_weights(&cfg, &dev);
        check_layer_count(&w, &cfg).unwrap();
        let enc = ClipVisionEncoder::from_weights(&w, &cfg).unwrap();
        let px = Tensor::randn(0f32, 1f32, (2, 3, 8, 8), &dev).unwrap();
        let out = enc.penultimate(&px).unwrap();
        assert_eq!(out.dims(), &[2, cfg.num_positions(), cfg.hidden]);
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "penultimate not finite");
    }

    /// `last_hidden` runs the full tower (one more layer than `penultimate`) to a finite
    /// `[B, num_positions, hidden]`, and differs from `penultimate` (the extra layer transforms the
    /// state) — the XLabs FLUX IP-Adapter's pooled-embedding path (sc-5872).
    #[test]
    fn last_hidden_runs_full_tower() {
        let dev = Device::Cpu;
        let cfg = VisionConfig {
            hidden: 32,
            num_layers: 3,
            num_heads: 4,
            patch: 2,
            image_size: 8,
            num_channels: 3,
            quick_gelu: true, // ViT-L uses quick-gelu (XLabs FLUX tower)
        };
        let w = tiny_weights(&cfg, &dev);
        let enc = ClipVisionEncoder::from_weights(&w, &cfg).unwrap();
        let px = Tensor::randn(0f32, 1f32, (1, 3, 8, 8), &dev).unwrap();
        let last = enc.last_hidden(&px).unwrap();
        assert_eq!(last.dims(), &[1, cfg.num_positions(), cfg.hidden]);
        let lv = last.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(lv.iter().all(|v| v.is_finite()), "last_hidden not finite");
        // The final encoder layer actually transforms the state, so last != penultimate.
        let penult = enc.penultimate(&px).unwrap();
        let pv = penult.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_diff = lv
            .iter()
            .zip(&pv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "last_hidden should differ from penultimate"
        );
    }

    /// The quick-gelu variant also forwards to a finite penultimate (exercises the `x·sigmoid(1.702x)`
    /// path used by the ViT-L towers).
    #[test]
    fn quick_gelu_variant_forwards() {
        let dev = Device::Cpu;
        let cfg = VisionConfig {
            hidden: 16,
            num_layers: 2,
            num_heads: 2,
            patch: 2,
            image_size: 4,
            num_channels: 3,
            quick_gelu: true,
        };
        let w = tiny_weights(&cfg, &dev);
        let enc = ClipVisionEncoder::from_weights(&w, &cfg).unwrap();
        let px = Tensor::randn(0f32, 1f32, (1, 3, 4, 4), &dev).unwrap();
        let out = enc.penultimate(&px).unwrap();
        assert_eq!(out.dims(), &[1, cfg.num_positions(), cfg.hidden]);
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    /// A wrong-variant config (layer-count mismatch) is rejected loudly.
    #[test]
    fn check_layer_count_rejects_mismatch() {
        let dev = Device::Cpu;
        let cfg = VisionConfig {
            hidden: 8,
            num_layers: 2,
            num_heads: 2,
            patch: 2,
            image_size: 4,
            num_channels: 3,
            quick_gelu: false,
        };
        let w = tiny_weights(&cfg, &dev);
        // The checkpoint has 2 layers; asking for 4 must error.
        let wrong = VisionConfig {
            num_layers: 4,
            ..cfg.clone()
        };
        assert!(check_layer_count(&w, &wrong).is_err());
        assert!(check_layer_count(&w, &cfg).is_ok());
    }
}
