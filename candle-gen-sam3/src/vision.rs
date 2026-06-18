//! SAM3 PE vision encoder — candle port of `mlx-gen-sam3`'s `vision.rs` (`Sam3ViTModel` PE backbone +
//! `Sam3VisionNeck` FPN), itself a port of `transformers/models/sam3/modeling_sam3.py` (epic 5482,
//! sc-6240 under sc-5062).
//!
//! The backbone is an isotropic windowed ViT (NOT SAM2's hierarchical Hiera): patch-embed (conv
//! stride 14, no bias) → tiled absolute position embedding → a front LayerNorm → 32 pre-norm
//! transformer layers. Most layers run **windowed** attention (window 24); layers [7,15,23,31] run
//! **global** attention. Every layer applies **2D axial RoPE** to q/k (the rotary table is fixed per
//! layer: window-sized for windowed layers, grid-sized + down-scaled for global layers). No
//! LayerScale (`layer_scale_init_value` is None in the shipped config).
//!
//! The neck runs one FPN branch per scale factor [4,2,1,0.5] over the 72² backbone grid, yielding
//! four 256-channel feature maps at 288²/144²/72²/36². The body runs **NHWC** (channels-last) to
//! mirror the MLX module; only the conv/transposed-conv/max-pool wrappers dip into candle's NCHW.

use std::sync::Arc;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::Quant;
use candle_gen::{CandleError, Result};

use crate::common::{
    conv2d_nhwc, conv_transpose2d_nhwc, join, layer_norm, maxpool2d_nhwc, sdpa, Linear, Weights,
};
use crate::config::Sam3VisionConfig;

/// Partition NHWC `x` into `window`×`window` windows (zero-padding to a multiple of `window`).
/// Returns the `[-1, window, window, c]` windows + the padded `(hp, wp)`. (Port of SAM3
/// `window_partition`.) The MLX 6-D permute `[0,1,3,2,4,5]` is just an axis-2↔3 swap → `transpose`.
fn window_partition(x: &Tensor, window: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;
    let pad_h = (window - h % window) % window;
    let pad_w = (window - w % window) % window;
    let x = if pad_h > 0 || pad_w > 0 {
        x.pad_with_zeros(1, 0, pad_h)?.pad_with_zeros(2, 0, pad_w)?
    } else {
        x.clone()
    };
    let (hp, wp) = (h + pad_h, w + pad_w);
    let (nwh, nww) = (hp / window, wp / window);
    let windows = x
        .reshape(vec![b, nwh, window, nww, window, c])?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((b * nwh * nww, window, window, c))?;
    Ok((windows, (hp, wp)))
}

/// Inverse of [`window_partition`]: stitch windows back to `[b, h, w, c]`, cropping padding.
fn window_unpartition(
    windows: &Tensor,
    window: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
) -> Result<Tensor> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let (nwh, nww) = (hp / window, wp / window);
    let num_per_image = nwh * nww;
    let c = windows.dim(3)?;
    let b = windows.dim(0)? / num_per_image;
    let x = windows
        .reshape(vec![b, nwh, nww, window, window, c])?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((b, hp, wp, c))?;
    if hp > h || wp > w {
        Ok(x.narrow(1, 0, h)?.narrow(2, 0, w)?)
    } else {
        Ok(x)
    }
}

/// Precomputed 2D-axial RoPE `(cos, sin)`, each `[end·end, head_dim]`, for a fixed feature grid.
/// `freqs[j] = θ^(-(4j)/head_dim)` over `j∈[0, head_dim/4)`; per position `i` the row is
/// `[x·freqs, y·freqs]` (x = i%end, y = i/end, both ·`scale`) then `repeat_interleave(2)`.
#[derive(Clone)]
struct RopeTable {
    cos: Tensor,
    sin: Tensor,
}

impl RopeTable {
    fn new(end: usize, scale: f32, theta: f64, head_dim: usize, device: &Device) -> Result<Self> {
        let quarter = head_dim / 4;
        let freqs: Vec<f32> = (0..quarter)
            .map(|j| (1.0 / theta.powf((4 * j) as f64 / head_dim as f64)) as f32)
            .collect();
        let n = end * end;
        let mut cos = Vec::with_capacity(n * head_dim);
        let mut sin = Vec::with_capacity(n * head_dim);
        for i in 0..n {
            let x = (i % end) as f32 * scale;
            let y = (i / end) as f32 * scale;
            // 32 values [x·freqs (16), y·freqs (16)], each then duplicated (repeat_interleave 2).
            let row = freqs
                .iter()
                .map(|&f| x * f)
                .chain(freqs.iter().map(|&f| y * f));
            for v in row {
                let (c, s) = (v.cos(), v.sin());
                cos.push(c);
                cos.push(c);
                sin.push(s);
                sin.push(s);
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (n, head_dim), device)?,
            sin: Tensor::from_vec(sin, (n, head_dim), device)?,
        })
    }
}

/// `rotate_pairwise(x)`: pairwise `(a, b) -> (-b, a)` over the last dim (the SAM3 interleaved
/// convention, paired with `repeat_interleave(2)` cos/sin).
fn rotate_pairwise(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let hd = *dims.last().expect("rope input has rank >= 1");
    let mut paired = dims.clone();
    let last = paired.len() - 1;
    paired[last] = hd / 2;
    paired.push(2);
    let xr = x.reshape(paired)?;
    let axis = xr.rank() - 1;
    let x1 = xr.narrow(axis, 0, 1)?; // even lane
    let x2 = xr.narrow(axis, 1, 1)?; // odd lane
    let stacked = Tensor::cat(&[&x2.neg()?, &x1], axis)?;
    Ok(stacked.reshape(dims)?)
}

/// `q_embed = q·cos + rotate_pairwise(q)·sin`. `q`: `[b, nh, seq, hd]`; `cos`/`sin`: `[seq, hd]`.
fn apply_rope(q: &Tensor, table: &RopeTable) -> Result<Tensor> {
    let a = q.broadcast_mul(&table.cos)?;
    let b = rotate_pairwise(q)?.broadcast_mul(&table.sin)?;
    Ok(a.add(&b)?)
}

/// Two-layer GELU MLP (`mlp.fc1` → exact-gelu → `mlp.fc2`). exact GELU = candle `gelu_erf`.
#[derive(Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(w, &join(prefix, "fc1"))?,
            fc2: Linear::load(w, &join(prefix, "fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?.gelu_erf()?;
        self.fc2.forward(&h)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.fc1.quantize(quant)?;
        self.fc2.quantize(quant)
    }
}

/// RoPE self-attention (separate q/k/v/o projections). Operates on NHWC `[b, H, W, C]`
/// (`b = batch·num_windows` for windowed layers).
#[derive(Clone)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(w: &Weights, prefix: &str, num_heads: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            q: Linear::load(w, &join(prefix, "q_proj"))?,
            k: Linear::load(w, &join(prefix, "k_proj"))?,
            v: Linear::load(w, &join(prefix, "v_proj"))?,
            o: Linear::load(w, &join(prefix, "o_proj"))?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RopeTable) -> Result<Tensor> {
        let (b, h, w, _c) = x.dims4()?;
        let (nh, hd) = (self.num_heads, self.head_dim);
        let seq = h * w;
        // [b,H,W,C] → [b, seq, nh, hd] → [b, nh, seq, hd]
        let to_heads = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((b, seq, nh, hd))?.transpose(1, 2)?.contiguous()?)
        };
        let q = apply_rope(&to_heads(self.q.forward(x)?)?, rope)?;
        let k = apply_rope(&to_heads(self.k.forward(x)?)?, rope)?;
        let v = to_heads(self.v.forward(x)?)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let attn = sdpa(&q, &k, &v, scale)?;
        let out = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, h, w, nh * hd))?;
        self.o.forward(&out)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.q.quantize(quant)?;
        self.k.quantize(quant)?;
        self.v.quantize(quant)?;
        self.o.quantize(quant)
    }
}

/// One pre-norm ViT layer: (windowed) RoPE attention + GELU MLP.
#[derive(Clone)]
struct ViTLayer {
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    attn: Attention,
    mlp: Mlp,
    rope: RopeTable,
    /// 0 ⇒ global attention over the full grid; else windowed with this side.
    window: usize,
    eps: f64,
}

impl ViTLayer {
    fn load(
        w: &Weights,
        prefix: &str,
        cfg: &Sam3VisionConfig,
        global: bool,
        device: &Device,
    ) -> Result<Self> {
        let hd = cfg.head_dim();
        // Rotary table: windowed layers use a window grid at scale 1; global layers use the full
        // token grid scaled by `window_size / grid` (so positions span the same rotary range).
        let (end, scale) = if global {
            (cfg.grid(), cfg.window_size as f32 / cfg.grid() as f32)
        } else {
            (cfg.window_size, 1.0)
        };
        Ok(Self {
            norm1_w: w.require(&join(prefix, "layer_norm1.weight"))?,
            norm1_b: w.require(&join(prefix, "layer_norm1.bias"))?,
            norm2_w: w.require(&join(prefix, "layer_norm2.weight"))?,
            norm2_b: w.require(&join(prefix, "layer_norm2.bias"))?,
            attn: Attention::load(w, &join(prefix, "attention"), cfg.num_attention_heads, hd)?,
            mlp: Mlp::load(w, &join(prefix, "mlp"))?,
            rope: RopeTable::new(end, scale, cfg.rope_theta, hd, device)?,
            window: if global { 0 } else { cfg.window_size },
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = layer_norm(x, &self.norm1_w, &self.norm1_b, self.eps)?;
        let attended = if self.window > 0 {
            let (h, w) = (normed.dim(1)?, normed.dim(2)?);
            let (windows, pad_hw) = window_partition(&normed, self.window)?;
            let a = self.attn.forward(&windows, &self.rope)?;
            window_unpartition(&a, self.window, pad_hw, (h, w))?
        } else {
            self.attn.forward(&normed, &self.rope)?
        };
        let x = x.broadcast_add(&attended)?;
        let mlp_in = layer_norm(&x, &self.norm2_w, &self.norm2_b, self.eps)?;
        Ok(x.broadcast_add(&self.mlp.forward(&mlp_in)?)?)
    }

    fn quantize(&mut self, quant: Quant) -> Result<()> {
        self.attn.quantize(quant)?;
        self.mlp.quantize(quant)
    }
}

/// PE ViT backbone: patch-embed → tiled position embedding → front LayerNorm → layers. Shared (via
/// `Arc`) between the detector neck and the tracker neck (the video model loads it once; F-028).
/// `Clone` is cheap (candle tensors are `Arc`-backed) — the video model clones the dense backbone to
/// quantize it once and reinstall the quantized copy into both consumers.
#[derive(Clone)]
pub(crate) struct Backbone {
    patch_w: Tensor, // OIHW (torch-native), no bias
    pos_embed: Tensor,
    front_norm_w: Tensor,
    front_norm_b: Tensor,
    layers: Vec<ViTLayer>,
    patch_size: usize,
    grid: usize,
    pretrain_grid: usize,
    eps: f64,
}

impl Backbone {
    pub(crate) fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3VisionConfig) -> Result<Self> {
        let patch_w = w.require(&join(
            prefix,
            "embeddings.patch_embeddings.projection.weight",
        ))?;
        let device = patch_w.device().clone();
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let global = cfg.global_attn_indexes.contains(&i);
                ViTLayer::load(
                    w,
                    &join(prefix, &format!("layers.{i}")),
                    cfg,
                    global,
                    &device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_w,
            pos_embed: w.require(&join(prefix, "embeddings.position_embeddings"))?,
            front_norm_w: w.require(&join(prefix, "layer_norm.weight"))?,
            front_norm_b: w.require(&join(prefix, "layer_norm.bias"))?,
            layers,
            patch_size: cfg.patch_size,
            grid: cfg.grid(),
            pretrain_grid: cfg.pretrain_grid(),
            eps: cfg.layer_norm_eps,
        })
    }

    /// Tile the `[1, pg², C]` position embedding to `[1, grid, grid, C]`. The shipped config has
    /// `grid` an exact multiple of `pg` (72 = 3·24), so this is exact tiling (no crop).
    fn tiled_pos(&self) -> Result<Tensor> {
        let pg = self.pretrain_grid;
        let c = self.pos_embed.dim(2)?;
        if !self.grid.is_multiple_of(pg) {
            return Err(CandleError::Msg(format!(
                "sam3 vision: token grid {} is not a multiple of the position-embedding grid {} \
                 (non-exact tiling not implemented)",
                self.grid, pg
            )));
        }
        let reps = self.grid / pg;
        let p = self.pos_embed.reshape((1, pg, pg, c))?;
        Ok(p.repeat((1, reps, reps, 1))?) // [1, grid, grid, C]
    }

    /// `pixel_values`: NCHW `[1, 3, 1008, 1008]`. Returns the backbone feature map NHWC
    /// `[1, grid, grid, C]`. The patch-embed conv runs directly on NCHW (torch-native OIHW kernel);
    /// the rest of the backbone is channels-last.
    pub(crate) fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let x = pixel_values.conv2d(&self.patch_w, 0, self.patch_size, 1, 1)?; // [1, C, grid, grid]
        let x = x.permute([0, 2, 3, 1])?.contiguous()?; // NHWC
        let mut x = x.broadcast_add(&self.tiled_pos()?)?;
        x = layer_norm(&x, &self.front_norm_w, &self.front_norm_b, self.eps)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    /// Affine-quantize the 32 ViT layers' attention/MLP projections to Q4/Q8 (the bulk of the model's
    /// ~445M params). The patch-embed conv, position embedding, and LayerNorms stay dense.
    pub(crate) fn quantize(&mut self, quant: Quant) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(quant)?;
        }
        Ok(())
    }
}

/// One FPN branch (`Sam3FPNLayer`): scale the backbone map, then `proj1` (1×1) → `proj2` (3×3).
pub(crate) struct FpnLayer {
    /// Transposed-conv up-scale stages (IOHW kernel + bias, torch-native), applied in order with
    /// exact-gelu between consecutive stages (matches `nn.GELU()` in the scale_factor==4 branch).
    up_stages: Vec<(Tensor, Tensor)>,
    /// True for scale_factor 0.5: a 2×2 max-pool downsample instead of transposed convs.
    downsample: bool,
    proj1_w: Tensor,
    proj1_b: Tensor,
    proj2_w: Tensor,
    proj2_b: Tensor,
}

impl FpnLayer {
    pub(crate) fn load(w: &Weights, prefix: &str, scale: f32) -> Result<Self> {
        // Branch on an integer code (`scale·2` → 8/4/2/1) to avoid float-literal matching.
        // scale_layers indices: ConvTranspose at 0 (and 2 for scale 4), GELU at 1 (no weights),
        // MaxPool at 0 for scale 0.5 (no weights).
        let code = (scale * 2.0).round() as i32;
        let up_indices: &[usize] = match code {
            8 => &[0, 2], // scale 4.0: two transposed convs (72→144→288)
            4 => &[0],    // scale 2.0: one transposed conv (72→144)
            _ => &[],     // scale 1.0 / 0.5: no transposed conv
        };
        let up_stages = up_indices
            .iter()
            .map(|&i| -> Result<(Tensor, Tensor)> {
                Ok((
                    w.require(&join(prefix, &format!("scale_layers.{i}.weight")))?,
                    w.require(&join(prefix, &format!("scale_layers.{i}.bias")))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            up_stages,
            downsample: code == 1, // scale 0.5
            proj1_w: w.require(&join(prefix, "proj1.weight"))?,
            proj1_b: w.require(&join(prefix, "proj1.bias"))?,
            proj2_w: w.require(&join(prefix, "proj2.weight"))?,
            proj2_b: w.require(&join(prefix, "proj2.bias"))?,
        })
    }

    /// `x`: NHWC `[1, 72, 72, 1024]`. Returns NHWC `[1, Hs, Ws, fpn_dim]`.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        let n = self.up_stages.len();
        for (i, (w, b)) in self.up_stages.iter().enumerate() {
            h = conv_transpose2d_nhwc(&h, w, b, 2)?;
            if i + 1 < n {
                h = h.gelu_erf()?; // GELU only *between* the two transposed convs (scale 4)
            }
        }
        if self.downsample {
            h = maxpool2d_nhwc(&h, 2)?;
        }
        let h = conv2d_nhwc(&h, &self.proj1_w, Some(&self.proj1_b), 1, 0)?; // 1×1
        conv2d_nhwc(&h, &self.proj2_w, Some(&self.proj2_b), 1, 1) // 3×3 pad 1
    }
}

/// SAM3 vision encoder: PE backbone + FPN neck. Produces the multi-scale FPN feature maps the
/// detector + tracker share.
pub struct Sam3VisionEncoder {
    backbone: Arc<Backbone>,
    fpn_layers: Vec<FpnLayer>,
}

impl Sam3VisionEncoder {
    /// Load from a `facebook/sam3` weight map. `prefix` is typically
    /// `"detector_model.vision_encoder"`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3VisionConfig) -> Result<Self> {
        let backbone = Arc::new(Backbone::from_weights(w, &join(prefix, "backbone"), cfg)?);
        Self::from_weights_with_backbone(w, prefix, cfg, backbone)
    }

    /// Load the FPN neck only, reusing an already-loaded (and possibly shared) PE [`Backbone`]. The
    /// video model uses this so the segmenter and the tracker share **one** backbone (F-028).
    pub(crate) fn from_weights_with_backbone(
        w: &Weights,
        prefix: &str,
        cfg: &Sam3VisionConfig,
        backbone: Arc<Backbone>,
    ) -> Result<Self> {
        let fpn_layers = cfg
            .scale_factors
            .iter()
            .enumerate()
            .map(|(i, &scale)| {
                FpnLayer::load(w, &join(prefix, &format!("neck.fpn_layers.{i}")), scale)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            backbone,
            fpn_layers,
        })
    }

    /// `pixel_values`: NCHW `[1, 3, 1008, 1008]`. Returns the FPN feature maps as **NHWC**
    /// `[1, Hs, Ws, fpn_dim]`, fine→coarse (288²/144²/72²/36²), one per `scale_factors` entry.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let features = self.backbone_features(pixel_values)?;
        self.fpn_from_backbone(&features)
    }

    /// Run **only** the PE ViT backbone (shared by the detector neck and the tracker neck), returning
    /// the NHWC `[1, grid, grid, C]` feature map. The video pipeline runs this once per frame and
    /// feeds both necks, avoiding a second backbone pass.
    pub fn backbone_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.backbone.forward(pixel_values)
    }

    /// Run the detector FPN neck over already-computed backbone features. Returns the FPN maps NHWC,
    /// fine→coarse.
    pub fn fpn_from_backbone(&self, features: &Tensor) -> Result<Vec<Tensor>> {
        self.fpn_layers
            .iter()
            .map(|l| l.forward(features))
            .collect()
    }

    /// The shared PE [`Backbone`] handle (clone of the `Arc`) — used by the video model to reinstall a
    /// once-quantized backbone, and by the F-028 shared-backbone parity check.
    pub(crate) fn backbone_arc(&self) -> Arc<Backbone> {
        self.backbone.clone()
    }

    /// Affine-quantize the shared PE backbone in place. `Arc::make_mut` clones the backbone first iff
    /// it is shared (the video model holds it in both the segmenter and the tracker), so a standalone
    /// segmenter quantizes in place. The FPN neck is all convs and stays dense.
    pub(crate) fn quantize_backbone(&mut self, quant: Quant) -> Result<()> {
        Arc::make_mut(&mut self.backbone).quantize(quant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// `window_unpartition ∘ window_partition` is identity over the unpadded region (exercises the
    /// pad-then-crop path with H/W not divisible by the window).
    #[test]
    fn window_partition_round_trips() {
        let (h, w, c) = (12usize, 10usize, 4usize);
        let vals: Vec<f32> = (0..(h * w * c) as i64).map(|i| i as f32).collect();
        let x = Tensor::from_vec(vals, (1, h, w, c), &cpu()).unwrap();
        let (windows, pad_hw) = window_partition(&x, 8).unwrap();
        assert_eq!(pad_hw, (16, 16));
        let back = window_unpartition(&windows, 8, pad_hw, (h, w)).unwrap();
        assert_eq!(back.dims(), &[1, h, w, c]);
        let a = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b);
    }

    /// `rotate_pairwise` maps lanes `(a, b) -> (-b, a)`; applied twice it negates (`x -> -x`).
    #[test]
    fn rotate_pairwise_squares_to_negation() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &cpu()).unwrap();
        let twice = rotate_pairwise(&rotate_pairwise(&x).unwrap()).unwrap();
        let got = twice.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![-1.0, -2.0, -3.0, -4.0]);
    }

    /// The RoPE table has the expected `[end², head_dim]` shape and unit-magnitude cos/sin pairs
    /// (row 0 is all-zero angle → cos 1, sin 0).
    #[test]
    fn rope_table_shape_and_origin() {
        let t = RopeTable::new(24, 1.0, 10000.0, 64, &cpu()).unwrap();
        assert_eq!(t.cos.dims(), &[576, 64]);
        assert_eq!(t.sin.dims(), &[576, 64]);
        let cos0 = t
            .cos
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let sin0 = t
            .sin
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(cos0.iter().all(|&c| (c - 1.0).abs() < 1e-6));
        assert!(sin0.iter().all(|&s| s.abs() < 1e-6));
    }
}
