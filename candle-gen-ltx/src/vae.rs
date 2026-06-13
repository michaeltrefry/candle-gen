//! LTX-2.3 **video VAE decoder** (`CausalVideoAutoencoder`, latent 128-ch, patch 4, 8× temporal /
//! 32× spatial) — port of mlx-gen-ltx `vae.rs` (`LTX2VideoDecoder`). T2V needs only `decode`; the
//! encoder (I2V) is deferred.
//!
//! Decode: denormalize `latent·std + mean` → `conv_in 128→1024` → 9 up_blocks (`Res` groups +
//! `DepthToSpace` upsamplers) → pixel-norm (eps 1e-8) → SiLU → `conv_out 128→48` → unpatchify(×4).
//! All convs are non-causal (frame-replication temporal pad). pixel_norm = `x/√(mean(x² over C)+eps)`
//! (no √C, no γ). Runs **f32**.
//!
//! Block execution order (the config `decoder_blocks` list is encoder-order; the decoder reverses
//! it): `Res(2), Up(2,2,2), Res(2), Up(2,2,2), Res(4), Up(2,1,1), Res(6), Up(1,2,2), Res(4)`. Each
//! `Up` with temporal stride 2 doubles then drops the first frame, so latent T=7 → 49 pixel frames;
//! spatial 15 → 480 px (×2×2×2 then unpatchify ×4).

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

use crate::conv3d::CausalConv3d;

const DEC_NORM_EPS: f64 = 1e-8;

/// `x / sqrt(mean(x² over C, keepdims) + eps)` — LTX PixelNorm (channel axis = 1, no √C, no γ).
fn pixel_norm(x: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)?;
    let sumsq = x.sqr()?.sum_keepdim(1)?;
    let mean = (sumsq / c as f64)?;
    let denom = (mean + DEC_NORM_EPS)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// Decoder residual block (`ResnetBlock3DSimple`): pixel-norm → SiLU → conv → pixel-norm → SiLU →
/// conv → residual add. Channels constant (no shortcut).
struct DecResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl DecResBlock {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::load(vb.clone(), &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::load(vb, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm(x)?)?;
        let h = self.conv1.forward(&h, false)?;
        let h = candle_gen::candle_nn::ops::silu(&pixel_norm(&h)?)?;
        let h = self.conv2.forward(&h, false)?;
        h + x
    }
}

/// `DepthToSpaceUpsample` (residual=false): conv → depth-to-space → (st>1) drop first temporal frame.
struct DepthToSpace {
    conv: CausalConv3d,
    st: usize,
    sh: usize,
    sw: usize,
}

impl DepthToSpace {
    fn load(vb: VarBuilder, prefix: &str, stride: (usize, usize, usize)) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::load(vb, &format!("{prefix}.conv.conv"))?,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
        })
    }

    /// `(B, C·st·sh·sw, D, H, W) -> (B, C, D·st, H·sh, W·sw)`.
    fn depth_to_space(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_packed, d, h, w) = x.dims5()?;
        let (st, sh, sw) = (self.st, self.sh, self.sw);
        let c = c_packed / (st * sh * sw);
        let x = x.reshape([b, c, st, sh, sw, d, h, w].as_slice())?;
        // transpose to (B, C, D, st, H, sh, W, sw) = axes [0,1,5,2,6,3,7,4].
        let x = x.permute([0usize, 1, 5, 2, 6, 3, 7, 4].as_slice())?;
        x.reshape((b, c, d * st, h * sh, w * sw))?.contiguous()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x, false)?;
        let x = self.depth_to_space(&x)?;
        if self.st > 1 {
            let t = x.dim(2)?;
            x.narrow(2, 1, t - 1)
        } else {
            Ok(x)
        }
    }
}

enum UpLayer {
    Res(Vec<DecResBlock>),
    Up(DepthToSpace),
}

/// One decoder block in execution order: a res group of `n` blocks, or an upsampler with `stride`.
enum DBlock {
    Res(usize),
    Up((usize, usize, usize)),
}

/// The fixed LTX-2.3 decoder block order (config `decoder_blocks` already reversed to execution order).
const DECODER_BLOCKS: [DBlock; 9] = [
    DBlock::Res(2),
    DBlock::Up((2, 2, 2)),
    DBlock::Res(2),
    DBlock::Up((2, 2, 2)),
    DBlock::Res(4),
    DBlock::Up((2, 1, 1)),
    DBlock::Res(6),
    DBlock::Up((1, 2, 2)),
    DBlock::Res(4),
];

/// `(B, C·p², F, H, W) -> (B, C, F, H·p, W·p)` (spatial-only unpatchify, patch_size_t = 1).
fn unpatchify(x: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c_packed, f, h, w) = x.dims5()?;
    let c = c_packed / (p * p);
    // (B, C, 1, p, p, F, H, W) -> transpose (0,1,5,2,6,4,7,3) -> (B, C, F, H·p, W·p).
    let x = x.reshape([b, c, 1, p, p, f, h, w].as_slice())?;
    let x = x.permute([0usize, 1, 5, 2, 6, 4, 7, 3].as_slice())?;
    x.reshape((b, c, f, h * p, w * p))?.contiguous()
}

/// The LTX-2.3 video VAE (decoder only, T2V).
pub struct LtxVideoVae {
    conv_in: CausalConv3d,
    up_blocks: Vec<UpLayer>,
    conv_out: CausalConv3d,
    mean: Tensor, // [1, 128, 1, 1, 1]
    std: Tensor,  // [1, 128, 1, 1, 1]
    patch_size: usize,
}

impl LtxVideoVae {
    /// Build from a VarBuilder rooted at the `vae.` prefix of the checkpoint.
    pub fn new(vb: VarBuilder, latent_channels: usize, patch_size: usize) -> Result<Self> {
        let dec = vb.pp("decoder");
        let mut up_blocks = Vec::with_capacity(DECODER_BLOCKS.len());
        for (idx, block) in DECODER_BLOCKS.iter().enumerate() {
            let prefix = format!("up_blocks.{idx}");
            up_blocks.push(match block {
                DBlock::Res(n) => {
                    let mut blocks = Vec::with_capacity(*n);
                    for j in 0..*n {
                        blocks.push(DecResBlock::load(
                            dec.clone(),
                            &format!("{prefix}.res_blocks.{j}"),
                        )?);
                    }
                    UpLayer::Res(blocks)
                }
                DBlock::Up(stride) => {
                    UpLayer::Up(DepthToSpace::load(dec.clone(), &prefix, *stride)?)
                }
            });
        }
        let stats = vb.pp("per_channel_statistics");
        let mean = stats
            .get_unchecked("mean-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        let std = stats
            .get_unchecked("std-of-means")?
            .reshape((1, latent_channels, 1, 1, 1))?;
        Ok(Self {
            conv_in: CausalConv3d::load(dec.clone(), "conv_in.conv")?,
            up_blocks,
            conv_out: CausalConv3d::load(dec, "conv_out.conv")?,
            mean,
            std,
            patch_size,
        })
    }

    /// Decode a normalized latent `[B, 128, F', H', W']` → video `[B, 3, F, 32·H', 32·W']` in ~[-1,1].
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        // Denormalize: x · std + mean.
        let x =
            (latent.broadcast_mul(&self.std)? + self.mean.broadcast_as(latent.shape())?.clone())?;
        let mut x = self.conv_in.forward(&x, false)?;
        for layer in &self.up_blocks {
            x = match layer {
                UpLayer::Res(blocks) => {
                    let mut h = x;
                    for b in blocks {
                        h = b.forward(&h)?;
                    }
                    h
                }
                UpLayer::Up(u) => u.forward(&x)?,
            };
        }
        let x = pixel_norm(&x)?;
        let x = candle_gen::candle_nn::ops::silu(&x)?;
        let x = self.conv_out.forward(&x, false)?;
        unpatchify(&x, self.patch_size)
    }
}
