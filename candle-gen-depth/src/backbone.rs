//! DINOv2 ViT-S/14 backbone — the Depth Anything V2 encoder. Faithful candle port of the HF
//! `transformers` `Dinov2Backbone` (`modeling_dinov2.py`) for the `backbone.*` weight tree of
//! `depth-anything/Depth-Anything-V2-Small-hf`. The candle twin of `mlx-gen-depth`'s `backbone.rs`.
//!
//! Pipeline: `Conv2d` patch-embed (kernel=stride=14, NHWC body) → prepend a learned CLS token + add
//! the learned absolute `position_embeddings` → 12 standard pre-norm transformer layers → the final
//! `layernorm` applied to each captured state. Each layer is two residual sub-blocks: (a) LN → MHSA
//! (separate Q/K/V linears, full SDPA) → LayerScale; (b) LN → MLP (fc1 → exact-GELU → fc2) →
//! LayerScale.
//!
//! For the DPT neck this backbone returns the **per-layer output hidden states** of the four
//! `out_indices` layers ([3,6,9,12], 1-based → captured at layer-output indices [2,5,8,11]). Each is
//! `[B, grid²+1, hidden]` *including* the CLS token; the neck drops the CLS token itself (matching
//! `transformers`).
//!
//! Fixed-size note: the host preprocessor always feeds the default 518² square, so the token grid is
//! exactly 37×37 (1369 patches + 1 CLS = 1370) and the shipped `position_embeddings` (length 1370) is
//! added **directly** — no DINOv2 pos-embed interpolation is needed.

use candle_gen::candle_core::Tensor;
use candle_gen::Result;

use crate::common::{conv2d_nhwc, join, layer_norm, sdpa, Linear, Weights};
use crate::config::DepthAnythingConfig;

/// One DINOv2 transformer layer (`backbone.encoder.layer.{i}`).
struct Dinov2Layer {
    norm1_w: Tensor,
    norm1_b: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ls1: Tensor, // layer_scale1.lambda1  [hidden]
    norm2_w: Tensor,
    norm2_b: Tensor,
    fc1: Linear,
    fc2: Linear,
    ls2: Tensor, // layer_scale2.lambda1  [hidden]
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    eps: f64,
}

impl Dinov2Layer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DepthAnythingConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            norm1_w: w.require(&p("norm1.weight"))?,
            norm1_b: w.require(&p("norm1.bias"))?,
            q: Linear::load(w, &p("attention.attention.query"))?,
            k: Linear::load(w, &p("attention.attention.key"))?,
            v: Linear::load(w, &p("attention.attention.value"))?,
            out: Linear::load(w, &p("attention.output.dense"))?,
            ls1: w.require(&p("layer_scale1.lambda1"))?,
            norm2_w: w.require(&p("norm2.weight"))?,
            norm2_b: w.require(&p("norm2.bias"))?,
            fc1: Linear::load(w, &p("mlp.fc1"))?,
            fc2: Linear::load(w, &p("mlp.fc2"))?,
            ls2: w.require(&p("layer_scale2.lambda1"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            scale: (cfg.head_dim() as f64).powf(-0.5),
            eps: cfg.layer_norm_eps,
        })
    }

    /// `x`: `[B, N, C]` (N = 1 CLS + grid² patches) → `[B, N, C]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let (h, hd) = (self.num_heads, self.head_dim);

        // --- self-attention sub-block (pre-norm + LayerScale + residual) ---
        let hn = layer_norm(x, &self.norm1_w, &self.norm1_b, self.eps)?;
        let to_heads = |t: &Tensor| -> Result<Tensor> {
            Ok(t.reshape((b, n, h, hd))?.transpose(1, 2)?.contiguous()?)
        };
        let q = to_heads(&self.q.forward(&hn)?)?;
        let k = to_heads(&self.k.forward(&hn)?)?;
        let v = to_heads(&self.v.forward(&hn)?)?;
        let attn = sdpa(&q, &k, &v, self.scale)?; // [b, h, n, hd]
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, h * hd))?;
        let attn = self.out.forward(&attn)?;
        let attn = attn.broadcast_mul(&self.ls1)?;
        let x = x.add(&attn)?;

        // --- MLP sub-block (pre-norm + LayerScale + residual) ---
        let hn = layer_norm(&x, &self.norm2_w, &self.norm2_b, self.eps)?;
        let y = self.fc1.forward(&hn)?.gelu_erf()?; // exact GELU
        let y = self.fc2.forward(&y)?;
        let y = y.broadcast_mul(&self.ls2)?;
        Ok(x.add(&y)?)
    }
}

/// The DINOv2 ViT backbone (patch-embed + CLS/pos embed + 12 layers + final LN).
pub struct Dinov2Backbone {
    proj_w: Tensor, // patch_embeddings.projection.weight, OIHW [embed, 3, 14, 14] (candle-native)
    proj_b: Tensor, // [embed]
    cls_token: Tensor,
    pos_embed: Tensor, // [1, grid²+1, embed]
    layers: Vec<Dinov2Layer>,
    final_ln_w: Tensor,
    final_ln_b: Tensor,
    cfg: DepthAnythingConfig,
}

impl Dinov2Backbone {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: DepthAnythingConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Dinov2Layer::from_weights(w, &p(&format!("encoder.layer.{i}")), &cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_w: w.require(&p("embeddings.patch_embeddings.projection.weight"))?,
            proj_b: w.require(&p("embeddings.patch_embeddings.projection.bias"))?,
            cls_token: w.require(&p("embeddings.cls_token"))?,
            pos_embed: w.require(&p("embeddings.position_embeddings"))?,
            layers,
            final_ln_w: w.require(&p("layernorm.weight"))?,
            final_ln_b: w.require(&p("layernorm.bias"))?,
            cfg,
        })
    }

    pub fn config(&self) -> &DepthAnythingConfig {
        &self.cfg
    }

    /// `pixel_values`: NHWC `[B, H, W, 3]` (H=W=image_size, ImageNet-normalized) → the four captured
    /// hidden states (outputs of the `out_indices` layers), each `[B, grid²+1, hidden]` **including**
    /// the CLS token. The final `layernorm` is applied to the captured states (the DPT reassemble
    /// stage in `transformers` consumes the normalized hidden states for DA-V2).
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let b = pixel_values.dim(0)?;
        let embed = self.cfg.hidden_size;

        // Patch embed: conv (stride=patch, no pad) → NHWC [B, g, g, embed] → [B, g², embed].
        let y = conv2d_nhwc(
            pixel_values,
            &self.proj_w,
            Some(&self.proj_b),
            self.cfg.patch_size,
            0,
        )?;
        let g = y.dim(1)?;
        let mut x = y.reshape((b, g * g, embed))?;

        // Prepend CLS, add absolute position embedding.
        let cls = self.cls_token.broadcast_as((b, 1, embed))?;
        x = Tensor::cat(&[&cls, &x], 1)?;
        x = x.broadcast_add(&self.pos_embed)?;

        let capture = self.cfg.capture_layers();
        let mut out = Vec::with_capacity(4);
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if capture.contains(&idx) {
                // DA-V2 applies the backbone's final LayerNorm to each captured state.
                out.push(layer_norm(
                    &x,
                    &self.final_ln_w,
                    &self.final_ln_b,
                    self.cfg.layer_norm_eps,
                )?);
            }
        }
        Ok(out)
    }
}
