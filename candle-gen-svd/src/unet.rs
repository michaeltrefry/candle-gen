//! SVD `UNetSpatioTemporalConditionModel` — the spatiotemporal denoising UNet. candle port of
//! diffusers `unet_spatio_temporal_condition.py` + the `*SpatioTemporal` blocks in `unet_3d_blocks.py`.
//!
//! A conv stem; sinusoidal timestep + `added_time_ids` micro-conditioning → a 1280-wide `emb`; a down
//! (3× `CrossAttnDownBlockSpatioTemporal` + 1× `DownBlockSpatioTemporal`) / mid
//! (`UNetMidBlockSpatioTemporal`) / up (`UpBlockSpatioTemporal` + 3× `CrossAttnUpBlockSpatioTemporal`)
//! stack of [`SpatioTemporalResBlock`]s and [`TransformerSpatioTemporal`]s; a conv head. Predicts the
//! per-frame `v` for one denoise step. NCHW spatial (`[B·F, C, H, W]`), NCDHW temporal (`[B, C, F, H, W]`).
//!
//! Per-block GroupNorm eps matches diffusers: **1e-6** for the `CrossAttnDownBlockSpatioTemporal`
//! resnets, **1e-5** for the plain down / mid / all up blocks + `conv_norm_out`. The
//! `SpatioTemporalResBlock` is temb-aware and blends `σ(mix)·spatial + (1−σ)·temporal`
//! (`learned_with_images`, no switch). Unlike MLX, candle's `VarBuilder` needs explicit channel dims,
//! so the (skip-concat) channel flow is computed here rather than inferred from weight shapes.

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::UnetConfig;
use crate::conv3d::TemporalConv3d;
use crate::embeddings::{sinusoidal_timestep, TimestepEmbedding};
use crate::transformer::TransformerSpatioTemporal;
use crate::vae::{Conv2dW, GroupNormW};

const GN_GROUPS: usize = 32;
/// `CrossAttnDownBlockSpatioTemporal` resnet epsilon (diffusers hardcodes `eps=1e-6` there).
const EPS_CROSS_DOWN: f64 = 1e-6;
/// Plain down / mid / up resnet + `conv_norm_out` epsilon (the `resnet_eps=1e-5` the UNet passes).
const EPS_OTHER: f64 = 1e-5;
const CROSS_DIM: usize = 1024;

/// `[B, D] → [B·F, D]` (diffusers `repeat_interleave(F, dim=0)`).
fn repeat_interleave_2d(x: &Tensor, f: usize) -> Result<Tensor> {
    let (b, d) = x.dims2()?;
    x.reshape((b, 1, d))?
        .broadcast_as((b, f, d))?
        .reshape((b * f, d))?
        .contiguous()
}

/// `[B, S, D] → [B·F, S, D]` (diffusers `repeat_interleave(F, dim=0)`).
fn repeat_interleave_3d(x: &Tensor, f: usize) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    x.reshape((b, 1, s, d))?
        .broadcast_as((b, f, s, d))?
        .reshape((b * f, s, d))?
        .contiguous()
}

/// Temb-aware spatial `ResnetBlock2D`: GroupNorm→SiLU→Conv3×3, + projected temb, GroupNorm→SiLU→
/// Conv3×3, + (1×1-conv) residual. NCHW `[B·F, C, H, W]`, `temb` `[B·F, 1280]`.
struct SpatialResnet {
    norm1: GroupNormW,
    conv1: Conv2dW,
    temb_proj: Linear,
    norm2: GroupNormW,
    conv2: Conv2dW,
    shortcut: Option<Conv2dW>,
}

impl SpatialResnet {
    fn load(in_c: usize, out_c: usize, temb_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: GroupNormW::load(in_c, GN_GROUPS, eps, vb.pp("norm1"))?,
            conv1: Conv2dW::load(in_c, out_c, 3, 1, 1, vb.pp("conv1"))?,
            temb_proj: linear(temb_dim, out_c, vb.pp("time_emb_proj"))?,
            norm2: GroupNormW::load(out_c, GN_GROUPS, eps, vb.pp("norm2"))?,
            conv2: Conv2dW::load(out_c, out_c, 3, 1, 1, vb.pp("conv2"))?,
            shortcut: if in_c != out_c {
                Some(Conv2dW::load(in_c, out_c, 1, 0, 1, vb.pp("conv_shortcut"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let tp = self.temb_proj.forward(&temb.silu()?)?; // [B·F, out_c]
        let (bf, oc) = tp.dims2()?;
        let y = y.broadcast_add(&tp.reshape((bf, oc, 1, 1))?)?;
        let y = self.conv2.forward(&self.norm2.forward(&y)?.silu()?)?;
        let residual = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        residual + y
    }
}

/// Temb-aware temporal `TemporalResnetBlock`: Conv3d`(3,1,1)` over the frame axis, + projected temb.
/// NCDHW `[B, C, F, H, W]`, `temb` `[B, F, 1280]`.
struct TemporalResnet {
    norm1: GroupNormW,
    conv1: TemporalConv3d,
    temb_proj: Linear,
    norm2: GroupNormW,
    conv2: TemporalConv3d,
}

impl TemporalResnet {
    fn load(c: usize, temb_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: GroupNormW::load(c, GN_GROUPS, eps, vb.pp("norm1"))?,
            conv1: TemporalConv3d::load(c, c, 3, vb.pp("conv1"))?,
            temb_proj: linear(temb_dim, c, vb.pp("time_emb_proj"))?,
            norm2: GroupNormW::load(c, GN_GROUPS, eps, vb.pp("norm2"))?,
            conv2: TemporalConv3d::load(c, c, 3, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let tp = self.temb_proj.forward(&temb.silu()?)?; // [B, F, c]
        let (b, f, c) = tp.dims3()?;
        let tp = tp.transpose(1, 2)?.reshape((b, c, f, 1, 1))?; // [B, c, F, 1, 1]
        let y = y.broadcast_add(&tp)?;
        let y = self.conv2.forward(&self.norm2.forward(&y)?.silu()?)?;
        x + y
    }
}

/// `SpatioTemporalResBlock` (UNet flavor): spatial pass then temporal pass, blended
/// `σ(mix)·spatial + (1−σ)·temporal`.
struct SpatioTemporalResBlock {
    spatial: SpatialResnet,
    temporal: TemporalResnet,
    mix_factor: Tensor,
}

impl SpatioTemporalResBlock {
    fn load(in_c: usize, out_c: usize, temb_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            spatial: SpatialResnet::load(in_c, out_c, temb_dim, eps, vb.pp("spatial_res_block"))?,
            temporal: TemporalResnet::load(out_c, temb_dim, eps, vb.pp("temporal_res_block"))?,
            mix_factor: vb.get(1, "time_mixer.mix_factor")?,
        })
    }

    fn forward(&self, x: &Tensor, temb: &Tensor, num_frames: usize) -> Result<Tensor> {
        let spatial = self.spatial.forward(x, temb)?; // [B·F, out_c, H, W]
        let (bf, c, h, w) = spatial.dims4()?;
        let b = bf / num_frames;
        let spatial5 = spatial
            .reshape((b, num_frames, c, h, w))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?; // [B, C, F, H, W]
        let temb5 = temb.reshape((b, num_frames, temb.dim(1)?))?; // [B, F, 1280]
        let temporal = self.temporal.forward(&spatial5, &temb5)?; // [B, C, F, H, W]

        let alpha = sigmoid(&self.mix_factor)?;
        let one_minus = alpha.affine(-1.0, 1.0)?;
        // learned_with_images, no switch → α·spatial + (1−α)·temporal.
        let blended = spatial5
            .broadcast_mul(&alpha)?
            .add(&temporal.broadcast_mul(&one_minus)?)?;
        blended
            .permute((0, 2, 1, 3, 4))?
            .reshape((bf, c, h, w))?
            .contiguous()
    }
}

/// One down block: resnets (optionally each followed by a transformer) + an optional downsample.
struct DownBlock {
    resnets: Vec<SpatioTemporalResBlock>,
    attentions: Option<Vec<TransformerSpatioTemporal>>,
    downsampler: Option<Conv2dW>,
}

impl DownBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_c: usize,
        out_c: usize,
        temb_dim: usize,
        num_resnets: usize,
        eps: f64,
        cross_attn: Option<usize>, // Some(heads) if a CrossAttn block
        add_down: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let resnets = (0..num_resnets)
            .map(|j| {
                let ic = if j == 0 { in_c } else { out_c };
                SpatioTemporalResBlock::load(ic, out_c, temb_dim, eps, rvb.pp(j))
            })
            .collect::<Result<Vec<_>>>()?;
        let attentions = match cross_attn {
            Some(heads) => {
                let avb = vb.pp("attentions");
                Some(
                    (0..num_resnets)
                        .map(|j| {
                            TransformerSpatioTemporal::load(out_c, CROSS_DIM, heads, 1, avb.pp(j))
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            None => None,
        };
        Ok(Self {
            resnets,
            attentions,
            downsampler: if add_down {
                // Downsample2D(use_conv, padding=1): conv 3×3, stride 2, pad 1.
                Some(Conv2dW::load(
                    out_c,
                    out_c,
                    3,
                    1,
                    2,
                    vb.pp("downsamplers").pp("0").pp("conv"),
                )?)
            } else {
                None
            },
        })
    }

    /// Returns the block output and its per-resnet (+ downsample) skip residuals.
    fn forward(
        &self,
        x: &Tensor,
        temb: &Tensor,
        context: &Tensor,
        num_frames: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut x = x.clone();
        let mut res = Vec::new();
        for (i, r) in self.resnets.iter().enumerate() {
            x = r.forward(&x, temb, num_frames)?;
            if let Some(attns) = &self.attentions {
                x = attns[i].forward(&x, context, num_frames)?;
            }
            res.push(x.clone());
        }
        if let Some(conv) = &self.downsampler {
            x = conv.forward(&x)?;
            res.push(x.clone());
        }
        Ok((x, res))
    }
}

/// The mid block: resnet → (transformer → resnet)×num_layers.
struct MidBlock {
    res0: SpatioTemporalResBlock,
    pairs: Vec<(TransformerSpatioTemporal, SpatioTemporalResBlock)>,
}

impl MidBlock {
    fn load(
        channels: usize,
        temb_dim: usize,
        heads: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let res0 = SpatioTemporalResBlock::load(
            channels,
            channels,
            temb_dim,
            EPS_OTHER,
            vb.pp("resnets").pp("0"),
        )?;
        let pairs = (0..num_layers)
            .map(|i| -> Result<_> {
                let attn = TransformerSpatioTemporal::load(
                    channels,
                    CROSS_DIM,
                    heads,
                    1,
                    vb.pp("attentions").pp(i),
                )?;
                let resnet = SpatioTemporalResBlock::load(
                    channels,
                    channels,
                    temb_dim,
                    EPS_OTHER,
                    vb.pp("resnets").pp(i + 1),
                )?;
                Ok((attn, resnet))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { res0, pairs })
    }

    fn forward(
        &self,
        x: &Tensor,
        temb: &Tensor,
        context: &Tensor,
        num_frames: usize,
    ) -> Result<Tensor> {
        let mut x = self.res0.forward(x, temb, num_frames)?;
        for (attn, resnet) in &self.pairs {
            x = attn.forward(&x, context, num_frames)?;
            x = resnet.forward(&x, temb, num_frames)?;
        }
        Ok(x)
    }
}

/// One up block: per resnet, concat the popped skip then resnet (optionally + transformer); then an
/// optional upsample.
struct UpBlock {
    resnets: Vec<SpatioTemporalResBlock>,
    attentions: Option<Vec<TransformerSpatioTemporal>>,
    upsampler: Option<Conv2dW>,
}

impl UpBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_c: usize,
        out_c: usize,
        prev_out: usize,
        temb_dim: usize,
        num_resnets: usize,
        cross_attn: Option<usize>,
        add_up: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let resnets = (0..num_resnets)
            .map(|j| {
                // diffusers: res_skip = in_c if last resnet else out_c; resnet_in = prev_out if j==0 else out_c.
                let res_skip = if j == num_resnets - 1 { in_c } else { out_c };
                let resnet_in = if j == 0 { prev_out } else { out_c };
                SpatioTemporalResBlock::load(
                    resnet_in + res_skip,
                    out_c,
                    temb_dim,
                    EPS_OTHER,
                    rvb.pp(j),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let attentions = match cross_attn {
            Some(heads) => {
                let avb = vb.pp("attentions");
                Some(
                    (0..num_resnets)
                        .map(|j| {
                            TransformerSpatioTemporal::load(out_c, CROSS_DIM, heads, 1, avb.pp(j))
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            None => None,
        };
        Ok(Self {
            resnets,
            attentions,
            upsampler: if add_up {
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

    fn forward(
        &self,
        x: &Tensor,
        temb: &Tensor,
        context: &Tensor,
        skips: &mut Vec<Tensor>,
        num_frames: usize,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, r) in self.resnets.iter().enumerate() {
            let skip = skips.pop().expect("up block: skip residual underflow");
            x = Tensor::cat(&[&x, &skip], 1)?; // channel concat
            x = r.forward(&x, temb, num_frames)?;
            if let Some(attns) = &self.attentions {
                x = attns[i].forward(&x, context, num_frames)?;
            }
        }
        if let Some(conv) = &self.upsampler {
            let (_, _, h, w) = x.dims4()?;
            x = conv.forward(&x.upsample_nearest2d(h * 2, w * 2)?)?;
        }
        Ok(x)
    }
}

/// The SVD spatiotemporal conditional UNet.
pub struct SvdUnet {
    conv_in: Conv2dW,
    time_embedding: TimestepEmbedding,
    add_embedding: TimestepEmbedding,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNormW,
    conv_out: Conv2dW,
    time_proj_dim: usize,
    add_time_proj_dim: usize,
    num_time_ids: usize,
    dtype: DType,
}

impl SvdUnet {
    /// Build from a VarBuilder rooted at the `unet/` safetensors.
    pub fn new(cfg: &UnetConfig, vb: VarBuilder) -> Result<Self> {
        let boc = &cfg.block_out_channels;
        let heads = &cfg.num_attention_heads;
        let n = boc.len();
        let temb_dim = boc[0] * 4; // diffusers time_embed_dim.

        // Down: CrossAttn for every block but the last (a plain DownBlock).
        let dvb = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(n);
        let mut input_channel = boc[0];
        for i in 0..n {
            let output_channel = boc[i];
            let is_last = i == n - 1;
            down_blocks.push(DownBlock::load(
                input_channel,
                output_channel,
                temb_dim,
                cfg.layers_per_block,
                if is_last { EPS_OTHER } else { EPS_CROSS_DOWN },
                if is_last { None } else { Some(heads[i]) },
                !is_last,
                dvb.pp(i),
            )?);
            input_channel = output_channel;
        }

        let mid_block = MidBlock::load(
            boc[n - 1],
            temb_dim,
            heads[n - 1],
            cfg.transformer_layers_per_block,
            vb.pp("mid_block"),
        )?;

        // Up: reversed; the first is a plain UpBlock, the rest CrossAttn. All use eps 1e-5.
        let rev_boc: Vec<usize> = boc.iter().rev().copied().collect();
        let rev_heads: Vec<usize> = heads.iter().rev().copied().collect();
        let uvb = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(n);
        let mut prev_output = rev_boc[0];
        for i in 0..n {
            let output_channel = rev_boc[i];
            let input_channel = rev_boc[(i + 1).min(n - 1)];
            let is_first = i == 0;
            let is_last = i == n - 1;
            up_blocks.push(UpBlock::load(
                input_channel,
                output_channel,
                prev_output,
                temb_dim,
                cfg.layers_per_block + 1,
                if is_first { None } else { Some(rev_heads[i]) },
                !is_last,
                uvb.pp(i),
            )?);
            prev_output = output_channel;
        }

        Ok(Self {
            conv_in: Conv2dW::load(cfg.in_channels, boc[0], 3, 1, 1, vb.pp("conv_in"))?,
            time_embedding: TimestepEmbedding::load(
                boc[0],
                temb_dim,
                temb_dim,
                vb.pp("time_embedding"),
            )?,
            add_embedding: TimestepEmbedding::load(
                cfg.projection_class_embeddings_input_dim,
                temb_dim,
                temb_dim,
                vb.pp("add_embedding"),
            )?,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out: GroupNormW::load(boc[0], GN_GROUPS, EPS_OTHER, vb.pp("conv_norm_out"))?,
            conv_out: Conv2dW::load(boc[0], cfg.out_channels, 3, 1, 1, vb.pp("conv_out"))?,
            time_proj_dim: boc[0],
            add_time_proj_dim: cfg.addition_time_embed_dim,
            num_time_ids: 3,
            dtype: vb.dtype(),
        })
    }

    /// Predict per-frame `v` for one denoise step.
    /// - `sample`: `[B, F, 8, H, W]` (4 noise latent + 4 image-latent channel-concat).
    /// - `timestep`: the scheduler model-timestep (`0.25·ln σ`), broadcast to the batch.
    /// - `image_embeds`: CLIP image conditioning `[B, ctx, 1024]` (repeated over frames internally).
    /// - `added_time_ids`: `[B, 3]` (`[fps−1, motion_bucket_id, noise_aug_strength]`).
    ///
    /// Returns `[B, F, 4, H, W]` (f32).
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: f32,
        image_embeds: &Tensor,
        added_time_ids: &Tensor,
        num_frames: usize,
    ) -> Result<Tensor> {
        let (b, f, in_ch, h, w) = sample.dims5()?;
        let device = sample.device();
        let sample = sample.to_dtype(self.dtype)?;
        let image_embeds = image_embeds.to_dtype(self.dtype)?;

        // Timestep embedding: sinusoidal (f32) → cast → MLP.
        let t = Tensor::from_vec(vec![timestep; b], b, device)?;
        let temb = sinusoidal_timestep(&t, self.time_proj_dim, device)?.to_dtype(self.dtype)?; // [B, 320]
        let mut emb = self.time_embedding.forward(&temb)?; // [B, 1280]

        // `added_time_ids` micro-conditioning.
        let flat = added_time_ids.flatten_all()?; // [B·3]
        let time_embeds = sinusoidal_timestep(&flat, self.add_time_proj_dim, device)?; // [B·3, 256]
        let time_embeds = time_embeds
            .reshape((b, self.add_time_proj_dim * self.num_time_ids))?
            .to_dtype(self.dtype)?; // [B, 768]
        let aug = self.add_embedding.forward(&time_embeds)?; // [B, 1280]
        emb = (emb + aug)?;

        // Flatten frames; repeat conditioning over frames.
        let sample = sample.reshape((b * f, in_ch, h, w))?;
        let emb = repeat_interleave_2d(&emb, f)?; // [B·F, 1280]
        let context = repeat_interleave_3d(&image_embeds, f)?; // [B·F, ctx, 1024]

        // Conv stem; collect skip residuals (starting with the stem output).
        let mut x = self.conv_in.forward(&sample)?;
        let mut skips: Vec<Tensor> = vec![x.clone()];
        for block in &self.down_blocks {
            let (out, res) = block.forward(&x, &emb, &context, num_frames)?;
            x = out;
            skips.extend(res);
        }

        x = self.mid_block.forward(&x, &emb, &context, num_frames)?;

        for block in &self.up_blocks {
            x = block.forward(&x, &emb, &context, &mut skips, num_frames)?;
        }

        let x = self.conv_norm_out.forward(&x)?.silu()?;
        let x = self.conv_out.forward(&x)?; // [B·F, 4, H, W]
        let (_, oc, oh, ow) = x.dims4()?;
        x.reshape((b, f, oc, oh, ow))?.to_dtype(DType::F32)
    }
}
