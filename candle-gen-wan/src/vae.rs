//! The **`AutoencoderKLWan`** (z48, `is_residual`) decoder — a port of diffusers
//! `autoencoder_kl_wan.py`, decode-only. Causal-Conv3d temporal VAE: latent `[B,48,T,H,W]` →
//! `[B,3, 1+(T-1)·4, 16H, 16W]` in `[-1,1]`.
//!
//! diffusers streams the decode frame-by-frame with a `feat_cache` (the causal temporal cache);
//! that is mathematically identical to a single pass over all `T` frames with the causal
//! left-padding ([`crate::conv3d`]) — except the temporal **upsampling**, where the first latent
//! frame is passed through un-doubled and the rest are doubled via the `time_conv` channel
//! interleave (the `first_chunk` rule). We reproduce that here in one pass.
//!
//! `WanRMS_norm` is a **channel-L2 normalization** over the channel axis (`x / max(‖x‖₂, 1e-12) ·
//! √C · γ`), NOT GroupNorm; weights ship as `.gamma`.

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::VarBuilder;

use crate::config::{VaeConfig, LATENTS_MEAN, LATENTS_STD};
use crate::conv3d::{CausalConv3d, Ctx};

const NORM_EPS: f64 = 1e-12;

/// Channel-L2 norm (`F.normalize(dim=channel) · √C · γ`). Works on 4-D `[N,C,H,W]` and 5-D
/// `[B,C,T,H,W]` tensors (channel axis 1).
struct ChanNorm {
    gamma: Tensor, // [C]
    sqrt_c: f64,
}

impl ChanNorm {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get_unchecked("gamma")?.flatten_all()?;
        Ok(Self {
            gamma,
            sqrt_c: (channels as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let l2 = x.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(NORM_EPS, 1e30)?;
        let normed = (x.broadcast_div(&l2)? * self.sqrt_c)?;
        let c = self.gamma.dim(0)?;
        let gshape = match x.rank() {
            5 => vec![1, c, 1, 1, 1],
            4 => vec![1, c, 1, 1],
            _ => vec![1, c],
        };
        normed.broadcast_mul(&self.gamma.reshape(gshape)?)
    }
}

/// A native 2-D conv applied per video frame (resample / attention 1×1 convs).
struct Conv2dW {
    w: Tensor,
    b: Tensor, // [1,O,1,1]
    pad: usize,
}

impl Conv2dW {
    fn load(in_c: usize, out_c: usize, k: usize, pad: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w: vb.get((out_c, in_c, k, k), "weight")?.contiguous()?,
            b: vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1))?,
            pad,
        })
    }
    /// `x`: `[N, C, H, W]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv2d(&self.w, self.pad, 1, 1, 1)?.broadcast_add(&self.b)
    }
}

fn causal(
    in_c: usize,
    out_c: usize,
    kernel: (usize, usize, usize),
    vb: VarBuilder,
) -> Result<CausalConv3d> {
    CausalConv3d::load(in_c, out_c, kernel, vb)
}

struct Resnet {
    norm1: ChanNorm,
    conv1: CausalConv3d,
    norm2: ChanNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl Resnet {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: ChanNorm::new(in_c, vb.pp("norm1"))?,
            conv1: causal(in_c, out_c, (3, 3, 3), vb.pp("conv1"))?,
            norm2: ChanNorm::new(out_c, vb.pp("norm2"))?,
            conv2: causal(out_c, out_c, (3, 3, 3), vb.pp("conv2"))?,
            shortcut: if in_c != out_c {
                Some(causal(in_c, out_c, (1, 1, 1), vb.pp("conv_shortcut"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let h = match &self.shortcut {
            Some(c) => c.forward(x, ctx)?,
            None => x.clone(),
        };
        let y = self.conv1.forward(&self.norm1.forward(x)?.silu()?, ctx)?;
        let y = self.conv2.forward(&self.norm2.forward(&y)?.silu()?, ctx)?;
        y + h
    }

    fn reset_cache(&self) {
        self.conv1.reset_cache();
        self.conv2.reset_cache();
        if let Some(c) = &self.shortcut {
            c.reset_cache();
        }
    }
}

struct MidAttn {
    norm: ChanNorm,
    qkv: Conv2dW,
    proj: Conv2dW,
    channels: usize,
}

impl MidAttn {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: ChanNorm::new(channels, vb.pp("norm"))?,
            qkv: Conv2dW::load(channels, channels * 3, 1, 0, vb.pp("to_qkv"))?,
            proj: Conv2dW::load(channels, channels, 1, 0, vb.pp("proj"))?,
            channels,
        })
    }

    /// `x`: `[B,C,T,H,W]`. Per-frame spatial self-attention.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let merged = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        let xn = self.norm.forward(&merged)?;
        let qkv = self.qkv.forward(&xn)?; // [BT,3C,H,W]
        let qkv = qkv
            .reshape((b * t, 1, 3 * c, h * w))?
            .permute((0, 1, 3, 2))?
            .contiguous()?; // [BT,1,HW,3C]
        let q = qkv.narrow(3, 0, c)?.contiguous()?;
        let k = qkv.narrow(3, c, c)?.contiguous()?;
        let v = qkv.narrow(3, 2 * c, c)?.contiguous()?;
        let scale = (self.channels as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = softmax_last_dim(&scores)?;
        let o = attn.matmul(&v)?; // [BT,1,HW,C]
        let o = o
            .squeeze(1)?
            .permute((0, 2, 1))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        let o = self.proj.forward(&o)?;
        let o = o
            .reshape((b, t, c, h, w))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        x + o
    }
}

/// Parameter-free `DupUp3D` shortcut (channel-duplicate upsample). `first_chunk` drops the leading
/// `factor_t-1` temporal frames to align with the causal main-path temporal expansion.
struct Dup {
    out_c: usize,
    factor_t: usize,
    factor_s: usize,
    repeats: usize,
}

impl Dup {
    fn new(in_c: usize, out_c: usize, factor_t: usize, factor_s: usize) -> Self {
        let factor = factor_t * factor_s * factor_s;
        Self {
            out_c,
            factor_t,
            factor_s,
            repeats: out_c * factor / in_c,
        }
    }

    fn apply(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        // repeat_interleave channels: [B,C,T,H,W] → [B,C,repeats,T,H,W] → [B,C*repeats,T,H,W].
        let x = x
            .unsqueeze(2)?
            .broadcast_as((b, c, self.repeats, t, h, w))?
            .reshape((b, c * self.repeats, t, h, w))?;
        let (ft, fs) = (self.factor_t, self.factor_s);
        let x = x
            .reshape(&[b, self.out_c, ft, fs, fs, t, h, w][..])?
            .permute(&[0usize, 1, 5, 2, 6, 3, 7, 4][..])? // [B,out,t,ft,h,fs,w,fs]
            .reshape((b, self.out_c, t * ft, h * fs, w * fs))?
            .contiguous()?;
        // Drop the leading ft-1 duplicated frames so the shortcut aligns with the causal main-path
        // temporal expansion (the "first frame un-doubled" rule). Single pass: always (the clip's
        // leading frames). Streaming: only on the first latent frame — later chunks keep all t·ft
        // frames, matching the temporal upsampler which doubles them.
        let drop_leading = ft > 1 && (!ctx.streaming || ctx.first_chunk);
        if drop_leading {
            let tt = x.dim(2)?;
            x.narrow(2, ft - 1, tt - (ft - 1))
        } else {
            Ok(x)
        }
    }
}

enum Upsampler {
    /// Temporal (3D): `time_conv` doubling + spatial 2× conv.
    Temporal {
        time_conv: CausalConv3d,
        resample: Conv2dW,
    },
    /// Spatial-only (2D): nearest-2× + conv.
    Spatial { resample: Conv2dW },
}

impl Upsampler {
    /// Per-frame nearest-2× upsample then the 3×3 resample conv. `x`: `[B,C,T,H,W]`.
    fn spatial(resample: &Conv2dW, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let merged = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        let up = merged.upsample_nearest2d(h * 2, w * 2)?;
        let y = resample.forward(&up)?;
        let (_, oc, oh, ow) = y.dims4()?;
        y.reshape((b, t, oc, oh, ow))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()
    }

    /// Double `t` frames → `2t` via the channel-interleave of `time_conv` (a `[B,2C,t,H,W]` output).
    fn double_temporal(time_conv: &CausalConv3d, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let tc = time_conv.forward(x, ctx)?; // [B,2C,t,H,W]
        tc.reshape((b, 2, c, t, h, w))?
            .permute((0, 2, 3, 1, 4, 5))? // [B,C,t,2,H,W]
            .reshape((b, c, 2 * t, h, w))?
            .contiguous()
    }

    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        match self {
            Upsampler::Spatial { resample } => Self::spatial(resample, x),
            Upsampler::Temporal {
                time_conv,
                resample,
            } => {
                let x_t = if ctx.streaming {
                    // Per-frame: the first latent frame passes un-doubled (and never touches the
                    // time_conv cache, matching the single-pass `first`); every later frame is
                    // doubled through the streaming time_conv.
                    if ctx.first_chunk {
                        x.clone()
                    } else {
                        Self::double_temporal(time_conv, x, ctx)?
                    }
                } else {
                    // Single pass: frame 0 un-doubled, frames 1.. doubled in one time_conv call.
                    let t = x.dim(2)?;
                    if t > 1 {
                        let first = x.narrow(2, 0, 1)?;
                        let rest = x.narrow(2, 1, t - 1)?;
                        let doubled = Self::double_temporal(time_conv, &rest, ctx)?;
                        Tensor::cat(&[&first, &doubled], 2)?
                    } else {
                        x.narrow(2, 0, 1)?
                    }
                };
                Self::spatial(resample, &x_t)
            }
        }
    }

    fn reset_cache(&self) {
        if let Upsampler::Temporal { time_conv, .. } = self {
            time_conv.reset_cache();
        }
    }
}

struct UpBlock {
    resnets: Vec<Resnet>,
    upsampler: Option<Upsampler>,
    dup: Option<Dup>,
}

impl UpBlock {
    fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let x_copy = x.clone();
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h, ctx)?;
        }
        if let Some(up) = &self.upsampler {
            h = up.forward(&h, ctx)?;
        }
        if let Some(dup) = &self.dup {
            h = (h + dup.apply(&x_copy, ctx)?)?;
        }
        Ok(h)
    }

    fn reset_cache(&self) {
        for r in &self.resnets {
            r.reset_cache();
        }
        if let Some(up) = &self.upsampler {
            up.reset_cache();
        }
    }
}

pub struct WanVae {
    mean: Tensor, // [1,48,1,1,1]
    std: Tensor,
    post_quant_conv: CausalConv3d,
    conv_in: CausalConv3d,
    mid_resnet0: Resnet,
    mid_attn: MidAttn,
    mid_resnet1: Resnet,
    up_blocks: Vec<UpBlock>,
    norm_out: ChanNorm,
    conv_out: CausalConv3d,
    patch_size: usize,
    out_channels: usize,
}

impl WanVae {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let mean = Tensor::from_vec(LATENTS_MEAN.to_vec(), (1, 48, 1, 1, 1), device)?;
        let std = Tensor::from_vec(LATENTS_STD.to_vec(), (1, 48, 1, 1, 1), device)?;
        let post_quant_conv = causal(cfg.z_dim, cfg.z_dim, (1, 1, 1), vb.pp("post_quant_conv"))?;

        let dec = vb.pp("decoder");
        // dims = [base*4] + base*[reversed dim_mult] = [1024, 1024,1024,512,256] for base=256.
        let b = cfg.base_dim;
        let dims = [b * 4, b * 4, b * 4, b * 2, b];
        let conv_in = causal(cfg.z_dim, dims[0], (3, 3, 3), dec.pp("conv_in"))?;

        let mid = dec.pp("mid_block");
        let mid_resnet0 = Resnet::new(dims[0], dims[0], mid.pp("resnets").pp("0"))?;
        let mid_attn = MidAttn::new(dims[0], mid.pp("attentions").pp("0"))?;
        let mid_resnet1 = Resnet::new(dims[0], dims[0], mid.pp("resnets").pp("1"))?;

        // temperal_upsample = [true, true, false]; up_flag = i != 3.
        let temporal = [true, true, false, false];
        let mut up_blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let (in_c, out_c) = (dims[i], dims[i + 1]);
            let up_flag = i != 3;
            let ub = dec.pp("up_blocks").pp(i);
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks + 1);
            let mut cur = in_c;
            for j in 0..(cfg.num_res_blocks + 1) {
                resnets.push(Resnet::new(cur, out_c, ub.pp("resnets").pp(j))?);
                cur = out_c;
            }
            let (upsampler, dup) = if up_flag {
                let up = if temporal[i] {
                    Upsampler::Temporal {
                        time_conv: causal(
                            out_c,
                            out_c * 2,
                            (3, 1, 1),
                            ub.pp("upsampler").pp("time_conv"),
                        )?,
                        resample: Conv2dW::load(
                            out_c,
                            out_c,
                            3,
                            1,
                            ub.pp("upsampler").pp("resample").pp("1"),
                        )?,
                    }
                } else {
                    Upsampler::Spatial {
                        resample: Conv2dW::load(
                            out_c,
                            out_c,
                            3,
                            1,
                            ub.pp("upsampler").pp("resample").pp("1"),
                        )?,
                    }
                };
                let factor_t = if temporal[i] { 2 } else { 1 };
                (Some(up), Some(Dup::new(in_c, out_c, factor_t, 2)))
            } else {
                (None, None)
            };
            up_blocks.push(UpBlock {
                resnets,
                upsampler,
                dup,
            });
        }

        let norm_out = ChanNorm::new(dims[4], dec.pp("norm_out"))?;
        let conv_out = causal(
            dims[4],
            cfg.conv_out_channels,
            (3, 3, 3),
            dec.pp("conv_out"),
        )?;

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
            patch_size: cfg.patch_size,
            out_channels: cfg.out_channels,
        })
    }

    /// Decode latents `[B,48,T,H,W]` → RGB frames `[B,3, 1+(T-1)·4, 16H, 16W]` in `[-1,1]`.
    ///
    /// **Streams one latent frame at a time** (sc-5176): the original single pass decoded every frame
    /// at once, spiking VAE memory ~60 GB on a 320²×17 clip (OOM). Each `CausalConv3d` carries its
    /// causal `feat_cache` across frames, so this is bit-equivalent to [`Self::decode_full`] while
    /// bounding peak memory to ~one frame's activations. Frame 0 expands to 1 output frame, each later
    /// latent frame to 4 (the two temporal upsamplers) — total `1+(T-1)·4`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.unnormalize(z)?;
        let t_lat = z.dim(2)?;
        self.reset_caches();
        let mut out: Option<Tensor> = None;
        for i in 0..t_lat {
            let zi = z.narrow(2, i, 1)?.contiguous()?;
            let oi = self.decode_inner(&zi, &Ctx::streaming(i == 0))?;
            out = Some(match out {
                Some(o) => Tensor::cat(&[&o, &oi], 2)?,
                None => oi,
            });
        }
        self.reset_caches();
        out.expect("decode needs >= 1 latent frame")
            .clamp(-1f32, 1f32)
    }

    /// Single-pass decode over all frames (the original path). Retained for the streaming-parity test
    /// (`decode` must match this bit-for-bit); not used in production (it OOMs on real clips).
    pub fn decode_full(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.unnormalize(z)?;
        self.decode_inner(&z, &Ctx::single_pass())?
            .clamp(-1f32, 1f32)
    }

    /// `z_pixel = z·std + mean` in f32 (the inverse of the encoder's per-channel normalize).
    fn unnormalize(&self, z: &Tensor) -> Result<Tensor> {
        z.to_dtype(DType::F32)?
            .broadcast_mul(&self.std)?
            .broadcast_add(&self.mean)
    }

    /// The decoder graph for one chunk (`ctx.streaming` selects the per-frame `feat_cache` path).
    fn decode_inner(&self, z: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let mut h = self.post_quant_conv.forward(z, ctx)?;
        h = self.conv_in.forward(&h, ctx)?;
        h = self.mid_resnet0.forward(&h, ctx)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet1.forward(&h, ctx)?;
        for ub in &self.up_blocks {
            h = ub.forward(&h, ctx)?;
        }
        let h = self.norm_out.forward(&h)?.silu()?;
        let h = self.conv_out.forward(&h, ctx)?; // [B,12,T',H8,W8]
        self.unpatchify(&h)
    }

    /// Drop every streaming `feat_cache` (called around the [`Self::decode`] frame loop).
    fn reset_caches(&self) {
        self.post_quant_conv.reset_cache();
        self.conv_in.reset_cache();
        self.mid_resnet0.reset_cache();
        self.mid_resnet1.reset_cache();
        for ub in &self.up_blocks {
            ub.reset_cache();
        }
        self.conv_out.reset_cache();
    }

    /// 12 → 3 channels, 2× spatial (inverse of the encoder's 2×2 patchify).
    fn unpatchify(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.patch_size;
        let (b, _cp, t, h, w) = x.dims5()?;
        let c = self.out_channels;
        x.reshape(&[b, c, p, p, t, h, w][..])?
            .permute(&[0usize, 1, 4, 5, 3, 6, 2][..])? // [B,c,T,H,p,W,p]
            .reshape((b, c, t, h * p, w * p))?
            .contiguous()
    }
}
