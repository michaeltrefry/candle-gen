//! SVD VAE — `AutoencoderKLTemporalDecoder`: a standard 2-D SD VAE **encoder** plus a
//! **spatio-temporal decoder**. candle port of diffusers
//! `models/autoencoders/autoencoder_kl_temporal_decoder.py` (+ the `SpatioTemporalResBlock` /
//! `TemporalResnetBlock` / `AlphaBlender` building blocks and the `Mid/UpBlockTemporalDecoder`).
//!
//! NCHW throughout (`[B·F, C, H, W]` spatial, `[B, C, F, H, W]` temporal, frame axis = the temporal
//! conv axis). Weights load in diffusers layout (`[O, I, kH, kW]`) with no transpose. diffusers
//! normalizes the SVD VAE at **eps 1e-6** (spatial / encoder / `conv_norm_out`) and **1e-5**
//! (temporal). Mirrors `mlx-gen-svd`'s `vae.rs` (which runs NHWC); validated against the same f32
//! reference.

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::VaeConfig;
use crate::conv3d::TemporalConv3d;

const GN_GROUPS: usize = 32;
/// Spatial / encoder / `conv_norm_out` GroupNorm epsilon (diffusers `resnet_eps` default).
const EPS_SPATIAL: f64 = 1e-6;
/// Temporal `TemporalResnetBlock` GroupNorm epsilon (the `SpatioTemporalResBlock` `temporal_eps`).
const EPS_TEMPORAL: f64 = 1e-5;

/// GroupNorm over the channel axis (dim 1) for an arbitrary-rank `[B, C, ...]` tensor — normalize per
/// group over `(C/G)·spatial`, then affine. Hand-written (rather than `candle_nn::GroupNorm`) to pin
/// the exact eps per block and guarantee 4-D + 5-D support. `pub(crate)` so the transformer + UNet
/// reuse it.
pub(crate) struct GroupNormW {
    weight: Tensor, // [C]
    bias: Tensor,   // [C]
    groups: usize,
    eps: f64,
}

impl GroupNormW {
    pub(crate) fn load(channels: usize, groups: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(channels, "weight")?,
            bias: vb.get(channels, "bias")?,
            groups,
            eps,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (b, c) = (dims[0], dims[1]);
        let spatial: usize = dims[2..].iter().product();
        let xr = x.reshape((b, self.groups, (c / self.groups) * spatial))?;
        let mean = xr.mean_keepdim(2)?;
        let xc = xr.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(2)?;
        let xn = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let xn = xn.reshape(dims.clone())?;
        let mut wshape = vec![1usize; dims.len()];
        wshape[1] = c;
        xn.broadcast_mul(&self.weight.reshape(wshape.clone())?)?
            .broadcast_add(&self.bias.reshape(wshape)?)
    }
}

/// A 2-D conv with explicit pad/stride. Weight `[O, I, k, k]` (diffusers layout), bias `[1, O, 1, 1]`.
/// `pub(crate)` so the UNet reuses it.
pub(crate) struct Conv2dW {
    w: Tensor,
    b: Tensor,
    pad: usize,
    stride: usize,
}

impl Conv2dW {
    pub(crate) fn load(
        in_c: usize,
        out_c: usize,
        k: usize,
        pad: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            w: vb.get((out_c, in_c, k, k), "weight")?.contiguous()?,
            b: vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1))?,
            pad,
            stride,
        })
    }

    /// `x`: `[N, C, H, W]`.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv2d(&self.w, self.pad, self.stride, 1, 1)?
            .broadcast_add(&self.b)
    }
}

/// Spatial `ResnetBlock2D` (temb-free in the VAE): GroupNorm→SiLU→Conv3×3 ×2 + a 1×1-conv residual
/// shortcut when channels change. NCHW `[B·F, C, H, W]`.
struct SpatialResnet {
    norm1: GroupNormW,
    conv1: Conv2dW,
    norm2: GroupNormW,
    conv2: Conv2dW,
    shortcut: Option<Conv2dW>,
}

impl SpatialResnet {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: GroupNormW::load(in_c, GN_GROUPS, EPS_SPATIAL, vb.pp("norm1"))?,
            conv1: Conv2dW::load(in_c, out_c, 3, 1, 1, vb.pp("conv1"))?,
            norm2: GroupNormW::load(out_c, GN_GROUPS, EPS_SPATIAL, vb.pp("norm2"))?,
            conv2: Conv2dW::load(out_c, out_c, 3, 1, 1, vb.pp("conv2"))?,
            shortcut: if in_c != out_c {
                Some(Conv2dW::load(in_c, out_c, 1, 0, 1, vb.pp("conv_shortcut"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let y = self.conv2.forward(&self.norm2.forward(&y)?.silu()?)?;
        let residual = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        residual + y
    }
}

/// `TemporalResnetBlock` (temb-free in the VAE): GroupNorm→SiLU→Conv3d`(3,1,1)` ×2 over the frame
/// axis + an optional 1×1×1 shortcut. NCDHW `[B, C, F, H, W]`.
struct TemporalResnet {
    norm1: GroupNormW,
    conv1: TemporalConv3d,
    norm2: GroupNormW,
    conv2: TemporalConv3d,
    shortcut: Option<TemporalConv3d>,
}

impl TemporalResnet {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: GroupNormW::load(in_c, GN_GROUPS, EPS_TEMPORAL, vb.pp("norm1"))?,
            conv1: TemporalConv3d::load(in_c, out_c, 3, vb.pp("conv1"))?,
            norm2: GroupNormW::load(out_c, GN_GROUPS, EPS_TEMPORAL, vb.pp("norm2"))?,
            conv2: TemporalConv3d::load(out_c, out_c, 3, vb.pp("conv2"))?,
            shortcut: if in_c != out_c {
                Some(TemporalConv3d::load(
                    in_c,
                    out_c,
                    1,
                    vb.pp("conv_shortcut"),
                )?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let y = self.conv2.forward(&self.norm2.forward(&y)?.silu()?)?;
        let residual = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        residual + y
    }
}

/// `SpatioTemporalResBlock` (VAE flavor): spatial pass on `[B·F, C, H, W]`, then the temporal pass on
/// `[B, C, F, H, W]`, blended by `AlphaBlender` (`merge_strategy="learned"`,
/// `switch_spatial_to_temporal_mix=True`): `out = (1−σ(mix))·spatial + σ(mix)·temporal`.
struct SpatioTemporalResBlock {
    spatial: SpatialResnet,
    temporal: TemporalResnet,
    mix_factor: Tensor, // [1]
}

impl SpatioTemporalResBlock {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            spatial: SpatialResnet::load(in_c, out_c, vb.pp("spatial_res_block"))?,
            temporal: TemporalResnet::load(out_c, out_c, vb.pp("temporal_res_block"))?,
            mix_factor: vb.get(1, "time_mixer.mix_factor")?,
        })
    }

    fn forward(&self, x: &Tensor, num_frames: usize) -> Result<Tensor> {
        let spatial = self.spatial.forward(x)?; // [B·F, C_out, H, W]
        let (bf, c, h, w) = spatial.dims4()?;
        let b = bf / num_frames;
        // [B·F, C, H, W] → [B, F, C, H, W] → [B, C, F, H, W]
        let spatial5 = spatial
            .reshape((b, num_frames, c, h, w))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        let temporal = self.temporal.forward(&spatial5)?; // [B, C, F, H, W]

        // AlphaBlender: alpha = σ(mix); switched → out = (1−alpha)·spatial + alpha·temporal.
        let alpha = candle_gen::candle_nn::ops::sigmoid(&self.mix_factor)?; // [1]
        let one_minus = alpha.affine(-1.0, 1.0)?; // 1 − alpha
        let blended = spatial5
            .broadcast_mul(&one_minus)?
            .add(&temporal.broadcast_mul(&alpha)?)?;
        // [B, C, F, H, W] → [B, F, C, H, W] → [B·F, C, H, W]
        blended
            .permute((0, 2, 1, 3, 4))?
            .reshape((bf, c, h, w))?
            .contiguous()
    }
}

/// Single-head spatial self-attention (diffusers `Attention`, `residual_connection=True`,
/// `norm_num_groups=32`, eps 1e-6) — the VAE mid-block attention. NCHW, per-frame.
struct VaeAttention {
    gn: GroupNormW,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl VaeAttention {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gn: GroupNormW::load(channels, GN_GROUPS, EPS_SPATIAL, vb.pp("group_norm"))?,
            q: linear(channels, channels, vb.pp("to_q"))?,
            k: linear(channels, channels, vb.pp("to_k"))?,
            v: linear(channels, channels, vb.pp("to_v"))?,
            out: linear(channels, channels, vb.pp("to_out").pp("0"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let y = self.gn.forward(x)?;
        // [B, C, H, W] → [B, H·W, C] sequence for the Linear projections.
        let seq = y.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;
        let q = self.q.forward(&seq)?; // [B, HW, C]
        let k = self.k.forward(&seq)?;
        let v = self.v.forward(&seq)?;
        let scale = (c as f64).powf(-0.5);
        // Single head: scores [B, HW, HW] → softmax → out [B, HW, C].
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?;
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?; // [B, HW, C]
        let o = self.out.forward(&o)?;
        // [B, HW, C] → [B, C, H, W]
        let o = o.transpose(1, 2)?.reshape((b, c, h, w))?;
        x + o
    }
}

/// Encoder down-block: a run of spatial resnets, then an optional stride-2 downsample.
struct EncDownBlock {
    resnets: Vec<SpatialResnet>,
    downsample: Option<Conv2dW>,
}

impl EncDownBlock {
    fn load(
        in_c: usize,
        out_c: usize,
        num_resnets: usize,
        add_down: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let resnets = (0..num_resnets)
            .map(|j| {
                let ic = if j == 0 { in_c } else { out_c };
                SpatialResnet::load(ic, out_c, rvb.pp(j))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            resnets,
            // The SD downsample conv is stride-2 / pad-0 with an asymmetric (right/bottom) pre-pad.
            downsample: if add_down {
                Some(Conv2dW::load(
                    out_c,
                    out_c,
                    3,
                    0,
                    2,
                    vb.pp("downsamplers").pp("0").pp("conv"),
                )?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some(conv) = &self.downsample {
            // Asymmetric (right/bottom) zero-pad, then stride-2 / pad-0 conv (the SD downsample).
            x = x
                .pad_with_zeros(D::Minus1, 0, 1)?
                .pad_with_zeros(D::Minus2, 0, 1)?;
            x = conv.forward(&x)?;
        }
        Ok(x)
    }
}

/// The standard 2-D SD VAE encoder (image → latent moments).
struct Encoder {
    conv_in: Conv2dW,
    down_blocks: Vec<EncDownBlock>,
    mid_res0: SpatialResnet,
    mid_attn: VaeAttention,
    mid_res1: SpatialResnet,
    norm_out: GroupNormW,
    conv_out: Conv2dW,
}

impl Encoder {
    fn load(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let ch = &cfg.block_out_channels;
        let n = ch.len();
        let last = ch[n - 1];
        let dvb = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(n);
        let mut input_channel = ch[0];
        for (i, &output_channel) in ch.iter().enumerate() {
            down_blocks.push(EncDownBlock::load(
                input_channel,
                output_channel,
                cfg.layers_per_block,
                i < n - 1,
                dvb.pp(i),
            )?);
            input_channel = output_channel;
        }
        let mvb = vb.pp("mid_block");
        Ok(Self {
            conv_in: Conv2dW::load(cfg.in_channels, ch[0], 3, 1, 1, vb.pp("conv_in"))?,
            down_blocks,
            mid_res0: SpatialResnet::load(last, last, mvb.pp("resnets").pp("0"))?,
            mid_attn: VaeAttention::load(last, mvb.pp("attentions").pp("0"))?,
            mid_res1: SpatialResnet::load(last, last, mvb.pp("resnets").pp("1"))?,
            norm_out: GroupNormW::load(last, GN_GROUPS, EPS_SPATIAL, vb.pp("conv_norm_out"))?,
            // 2·latent moments (mean+logvar).
            conv_out: Conv2dW::load(last, 2 * cfg.latent_channels, 3, 1, 1, vb.pp("conv_out"))?,
        })
    }

    /// `x`: NCHW `[B, 3, H, W]` → moments `[B, 8, H/8, W/8]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        for db in &self.down_blocks {
            x = db.forward(&x)?;
        }
        x = self.mid_res0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x)?;
        let x = self.norm_out.forward(&x)?.silu()?;
        self.conv_out.forward(&x)
    }
}

/// Decoder mid block (`MidBlockTemporalDecoder`, `num_layers=2`): res0 → spatial attn → res1.
struct DecMidBlock {
    res0: SpatioTemporalResBlock,
    attn: VaeAttention,
    res1: SpatioTemporalResBlock,
}

impl DecMidBlock {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            res0: SpatioTemporalResBlock::load(channels, channels, vb.pp("resnets").pp("0"))?,
            attn: VaeAttention::load(channels, vb.pp("attentions").pp("0"))?,
            res1: SpatioTemporalResBlock::load(channels, channels, vb.pp("resnets").pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor, num_frames: usize) -> Result<Tensor> {
        let x = self.res0.forward(x, num_frames)?;
        let x = self.attn.forward(&x)?;
        self.res1.forward(&x, num_frames)
    }
}

/// Decoder up-block (`UpBlockTemporalDecoder`): a run of spatio-temporal resnets, then an optional
/// nearest-2× + conv upsample.
struct DecUpBlock {
    resnets: Vec<SpatioTemporalResBlock>,
    upsample: Option<Conv2dW>,
}

impl DecUpBlock {
    fn load(
        in_c: usize,
        out_c: usize,
        num_resnets: usize,
        add_up: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let resnets = (0..num_resnets)
            .map(|j| {
                let ic = if j == 0 { in_c } else { out_c };
                SpatioTemporalResBlock::load(ic, out_c, rvb.pp(j))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            resnets,
            upsample: if add_up {
                Some(Conv2dW::load(
                    out_c,
                    out_c,
                    3,
                    1,
                    1,
                    vb.pp("upsamplers").pp("0").pp("conv"),
                )?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor, num_frames: usize) -> Result<Tensor> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x, num_frames)?;
        }
        if let Some(conv) = &self.upsample {
            let (n, c, h, w) = x.dims4()?;
            let up = x.upsample_nearest2d(h * 2, w * 2)?;
            let _ = (n, c);
            x = conv.forward(&up)?;
        }
        Ok(x)
    }
}

/// The temporal decoder (latent → frames).
struct TemporalDecoder {
    conv_in: Conv2dW,
    mid: DecMidBlock,
    up_blocks: Vec<DecUpBlock>,
    norm_out: GroupNormW,
    conv_out: Conv2dW,
    time_conv_out: TemporalConv3d,
}

impl TemporalDecoder {
    fn load(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let ch = &cfg.block_out_channels;
        let n = ch.len();
        let last = ch[n - 1];
        let num_resnets = cfg.layers_per_block + 1;
        let reversed: Vec<usize> = ch.iter().rev().copied().collect();
        let uvb = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(n);
        let mut prev = reversed[0];
        for (i, &out_c) in reversed.iter().enumerate() {
            up_blocks.push(DecUpBlock::load(
                prev,
                out_c,
                num_resnets,
                i < n - 1,
                uvb.pp(i),
            )?);
            prev = out_c;
        }
        Ok(Self {
            conv_in: Conv2dW::load(cfg.latent_channels, last, 3, 1, 1, vb.pp("conv_in"))?,
            mid: DecMidBlock::load(last, vb.pp("mid_block"))?,
            up_blocks,
            norm_out: GroupNormW::load(ch[0], GN_GROUPS, EPS_SPATIAL, vb.pp("conv_norm_out"))?,
            conv_out: Conv2dW::load(ch[0], cfg.out_channels, 3, 1, 1, vb.pp("conv_out"))?,
            time_conv_out: TemporalConv3d::load(
                cfg.out_channels,
                cfg.out_channels,
                3,
                vb.pp("time_conv_out"),
            )?,
        })
    }

    /// `z`: NCHW `[B·F, 4, H/8, W/8]` → frames NCHW `[B·F, 3, H, W]`.
    fn forward(&self, z: &Tensor, num_frames: usize) -> Result<Tensor> {
        let mut x = self.conv_in.forward(z)?;
        x = self.mid.forward(&x, num_frames)?;
        for ub in &self.up_blocks {
            x = ub.forward(&x, num_frames)?;
        }
        let x = self.norm_out.forward(&x)?.silu()?;
        let x = self.conv_out.forward(&x)?; // [B·F, 3, H, W]

        // `time_conv_out` over the frame axis: [B·F, C, H, W] → [B, C, F, H, W] → Conv3d → back.
        let (bf, c, h, w) = x.dims4()?;
        let b = bf / num_frames;
        let x5 = x
            .reshape((b, num_frames, c, h, w))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        let x5 = self.time_conv_out.forward(&x5)?;
        x5.permute((0, 2, 1, 3, 4))?
            .reshape((bf, c, h, w))?
            .contiguous()
    }
}

/// The SVD `AutoencoderKLTemporalDecoder`. `encode_mode` produces the latent mean (the
/// `latent_dist.mode()` the SVD pipeline conditions on); `decode` reconstructs frames from a latent
/// (the caller divides by `scaling_factor` first — this VAE has no `post_quant_conv`).
pub struct SvdVae {
    encoder: Encoder,
    /// `quant_conv` `[8, 8, 1, 1]` → an `[8, 8]` Linear over the moment channels.
    quant: Linear,
    decoder: TemporalDecoder,
    scaling_factor: f32,
}

impl SvdVae {
    /// Build from a VarBuilder rooted at the `vae/` safetensors (keys `encoder.*`, `decoder.*`,
    /// `quant_conv.*`).
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let moments = 2 * cfg.latent_channels;
        // quant_conv is a 1×1 conv [8,8,1,1] — load as an [8,8] Linear over the moment channels.
        let qw = vb
            .get((moments, moments, 1, 1), "quant_conv.weight")?
            .reshape((moments, moments))?;
        let qb = vb.get(moments, "quant_conv.bias")?;
        Ok(Self {
            encoder: Encoder::load(cfg, vb.pp("encoder"))?,
            quant: Linear::new(qw, Some(qb)),
            decoder: TemporalDecoder::load(cfg, vb.pp("decoder"))?,
            scaling_factor: cfg.scaling_factor,
        })
    }

    /// The trained latent scale (`z = z · scaling_factor` for the diffusion model; the pipeline divides
    /// by it before [`decode`](Self::decode)).
    pub fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }

    /// Encode `[B, 3, H, W]` (NCHW, roughly `[-1, 1]`) → latent **mean** `[B, 4, H/8, W/8]` (raw,
    /// **unscaled** — `latent_dist.mode()`). This conditioning latent is fed through unscaled by
    /// design: diffusers SVD `_encode_vae_image` does NOT apply `scaling_factor` to it.
    pub fn encode_mode(&self, image: &Tensor) -> Result<Tensor> {
        let m = self.encoder.forward(image)?; // [B, 8, h, w]
        let (b, c, h, w) = m.dims4()?;
        // quant_conv (1×1) over channels: [B, C, h, w] → [B, h·w, C] → Linear → [B, C, h, w].
        let seq = m.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;
        let moments = self
            .quant
            .forward(&seq)?
            .transpose(1, 2)?
            .reshape((b, c, h, w))?;
        // `DiagonalGaussian.mode()` = the mean = the first half of the channel axis.
        let half = c / 2;
        moments.narrow(1, 0, half)?.contiguous()
    }

    /// Decode a latent `[B·F, 4, H/8, W/8]` (NCHW, already divided by `scaling_factor`) → frames
    /// `[B·F, 3, H, W]`. Mirrors diffusers `vae.decode(z, num_frames)`.
    pub fn decode(&self, z: &Tensor, num_frames: usize) -> Result<Tensor> {
        self.decoder.forward(z, num_frames)
    }
}
