//! The Qwen-Image **AutoencoderKLQwenImage** decoder (decode-only). Port of `mlx-gen-qwen-image`'s
//! `vae/`, run in candle-native NCHW f32.
//!
//! It is a **causal-Conv3d** (video) VAE, but for a single image the temporal axis is `T = 1`. A
//! CausalConv3d left-pads time by `kD-1` (zeros) then does a valid `kD`-tap conv, so on a length-1
//! frame **only the last depth tap survives** — each `[O,I,kD,kH,kW]` conv3d weight reduces to the
//! 2-D slice `weight[:, :, kD-1, :, :]` and a plain conv2d. (candle has no conv3d.) The temporal
//! `time_conv` of the upsamplers is unused (skipped, like the fork).
//!
//! Two more non-obvious points: the norm is a **channel-L2 normalization** (NOT GroupNorm and NOT a
//! feature-axis RMSNorm) — `x / max(‖x‖₂ over C, 1e-12) · √C · gamma` — and the latent is
//! de-normalized as `z·std + mean` with per-channel constants before `post_quant_conv`.

use candle_gen::candle_core::{DType, IndexOp, Result, Tensor};
use candle_gen::candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

const NORM_EPS: f64 = 1e-12;

const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

/// Load a `CausalConv3d` (`[O,I,kD,kH,kW]`) as a candle `Conv2d`, keeping only the last depth tap.
fn causal_conv2d(
    in_c: usize,
    out_c: usize,
    k: usize,
    pad: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let w = vb.get((out_c, in_c, k, k, k), "weight")?;
    let w2 = w.narrow(2, k - 1, 1)?.squeeze(2)?.contiguous()?; // [O,I,kH,kW]
    let b = vb.get(out_c, "bias")?;
    Ok(Conv2d::new(
        w2,
        Some(b),
        Conv2dConfig {
            padding: pad,
            ..Default::default()
        },
    ))
}

/// Load a native 2-D conv (`[O,I,kH,kW]` on disk — the spatial resample + attention 1×1 convs).
fn conv2d_native(
    in_c: usize,
    out_c: usize,
    k: usize,
    pad: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let w = vb.get((out_c, in_c, k, k), "weight")?.contiguous()?;
    let b = vb.get(out_c, "bias")?;
    Ok(Conv2d::new(
        w,
        Some(b),
        Conv2dConfig {
            padding: pad,
            ..Default::default()
        },
    ))
}

/// A channel-L2 norm weight (`gamma`), stored as `[1, C, 1, 1]`.
struct ChanNorm {
    gamma: Tensor,
    sqrt_c: f64,
}

impl ChanNorm {
    fn new(channels: usize, vb: VarBuilder, key: &str) -> Result<Self> {
        // gamma ships as [C,1,1,1] (resnet/norm_out) or [C,1,1] (attention) — flatten to [C].
        let g = vb
            .get_unchecked(key)?
            .flatten_all()?
            .reshape((1, channels, 1, 1))?;
        Ok(Self {
            gamma: g,
            sqrt_c: (channels as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B,C,H,W]. L2 over channel axis (1), keepdim.
        let l2 = (x.sqr()?.sum_keepdim(1)? + NORM_EPS)?.sqrt()?;
        let normed = x.broadcast_div(&l2)?;
        (normed * self.sqrt_c)?.broadcast_mul(&self.gamma)
    }
}

struct Resnet {
    norm1: ChanNorm,
    conv1: Conv2d,
    norm2: ChanNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl Resnet {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: ChanNorm::new(in_c, vb.pp("norm1"), "gamma")?,
            conv1: causal_conv2d(in_c, out_c, 3, 1, vb.pp("conv1"))?,
            norm2: ChanNorm::new(out_c, vb.pp("norm2"), "gamma")?,
            conv2: causal_conv2d(out_c, out_c, 3, 1, vb.pp("conv2"))?,
            shortcut: if in_c != out_c {
                Some(causal_conv2d(in_c, out_c, 1, 0, vb.pp("conv_shortcut"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let h = self.conv2.forward(&self.norm2.forward(&h)?.silu()?)?;
        let res = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        h + res
    }
}

struct MidAttention {
    norm: ChanNorm,
    qkv: Conv2d,
    proj: Conv2d,
    channels: usize,
}

impl MidAttention {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: ChanNorm::new(channels, vb.pp("norm"), "gamma")?,
            qkv: conv2d_native(channels, channels * 3, 1, 0, vb.pp("to_qkv"))?,
            proj: conv2d_native(channels, channels, 1, 0, vb.pp("proj"))?,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let normed = self.norm.forward(x)?;
        let qkv = self.qkv.forward(&normed)?; // [B, 3C, H, W]
        let qkv = qkv.reshape((b, 3, c, h * w))?;
        let q = qkv.i((.., 0))?.transpose(1, 2)?.contiguous()?; // [B, HW, C]
        let k = qkv.i((.., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., 2))?.transpose(1, 2)?.contiguous()?;
        let scale = (self.channels as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let attn = candle_gen::candle_nn::ops::softmax_last_dim(&attn)?;
        let o = attn.matmul(&v)?; // [B, HW, C]
        let o = o.transpose(1, 2)?.reshape((b, c, h, w))?;
        x + self.proj.forward(&o)?
    }
}

struct Upsampler {
    conv: Conv2d,
}

impl Upsampler {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        // The spatial resample conv ships as a native 2-D conv at `resample.1`.
        Ok(Self {
            conv: conv2d_native(in_c, out_c, 3, 1, vb.pp("resample").pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let up = x.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&up)
    }
}

struct UpBlock {
    resnets: Vec<Resnet>,
    upsampler: Option<Upsampler>,
}

impl UpBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(u) = &self.upsampler {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

/// The Qwen-Image VAE (decode-only).
pub struct QwenVae {
    mean: Tensor, // [1,16,1,1]
    std: Tensor,  // [1,16,1,1]
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    mid_resnet0: Resnet,
    mid_attn: MidAttention,
    mid_resnet1: Resnet,
    up_blocks: Vec<UpBlock>,
    norm_out: ChanNorm,
    conv_out: Conv2d,
}

impl QwenVae {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let mean = Tensor::from_vec(LATENTS_MEAN.to_vec(), (1, 16, 1, 1), device)?;
        let std = Tensor::from_vec(LATENTS_STD.to_vec(), (1, 16, 1, 1), device)?;
        let post_quant_conv = causal_conv2d(16, 16, 1, 0, vb.pp("post_quant_conv"))?;

        let dec = vb.pp("decoder");
        let conv_in = causal_conv2d(16, 384, 3, 1, dec.pp("conv_in"))?;
        let mid = dec.pp("mid_block");
        let mid_resnet0 = Resnet::new(384, 384, mid.pp("resnets").pp("0"))?;
        let mid_attn = MidAttention::new(384, mid.pp("attentions").pp("0"))?;
        let mid_resnet1 = Resnet::new(384, 384, mid.pp("resnets").pp("1"))?;

        // (resnet0_in, block_width, upsampler_out?) per up_block — read from the checkpoint shapes.
        let up_cfg: [(usize, usize, Option<usize>); 4] = [
            (384, 384, Some(192)),
            (192, 384, Some(192)),
            (192, 192, Some(96)),
            (96, 96, None),
        ];
        let mut up_blocks = Vec::with_capacity(4);
        for (i, &(in_c, width, up_out)) in up_cfg.iter().enumerate() {
            let ub = dec.pp("up_blocks").pp(i);
            let mut resnets = Vec::with_capacity(3);
            for j in 0..3 {
                let rin = if j == 0 { in_c } else { width };
                resnets.push(Resnet::new(rin, width, ub.pp("resnets").pp(j))?);
            }
            let upsampler = match up_out {
                Some(out) => Some(Upsampler::new(width, out, ub.pp("upsamplers").pp("0"))?),
                None => None,
            };
            up_blocks.push(UpBlock { resnets, upsampler });
        }

        let norm_out = ChanNorm::new(96, dec.pp("norm_out"), "gamma")?;
        let conv_out = causal_conv2d(96, 3, 3, 1, dec.pp("conv_out"))?;

        Ok(Self {
            mean,
            std,
            post_quant_conv,
            conv_in,
            mid_resnet0,
            mid_attn,
            mid_resnet1,
            up_blocks,
            norm_out,
            conv_out,
        })
    }

    /// Decode VAE latents `[1, 16, H/8, W/8]` (NCHW) → RGB `[1, 3, H, W]` in `[-1, 1]`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let l = latents.to_dtype(DType::F32)?;
        // De-normalize: z·std + mean.
        let l = l.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let l = self.post_quant_conv.forward(&l)?;
        let mut h = self.conv_in.forward(&l)?;
        h = self.mid_resnet0.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet1.forward(&h)?;
        for ub in &self.up_blocks {
            h = ub.forward(&h)?;
        }
        let h = self.norm_out.forward(&h)?.silu()?;
        self.conv_out.forward(&h)
    }
}
