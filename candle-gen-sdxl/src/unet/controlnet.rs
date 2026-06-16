//! SDXL **ControlNet** branch (sc-5491, epic 5480) — the candle twin of `mlx-gen-sdxl`'s
//! `unet::controlnet`. A diffusers `ControlNetModel` is an *encoder copy* of the SDXL UNet: the same
//! `conv_in` + timestep/`add_embedding` embeddings + down/mid stack (built from the SAME vendored
//! blocks, loading the identical `down_blocks.*` / `mid_block.*` keys), plus three net-new pieces:
//!   - `controlnet_cond_embedding` — a tiny conv stack (3→16→32→96→256→`b_channels`, three stride-2
//!     convs ⇒ 8× downsample) that embeds the control image to latent resolution and is *added* to
//!     `conv_in(latents)`;
//!   - `controlnet_down_blocks` — one 1×1 "zero-conv" projection per down residual (9 for SDXL);
//!   - `controlnet_mid_block` — one 1×1 zero-conv for the mid output.
//!
//! [`ControlNet::forward`] returns the per-down-block + mid [`ControlResiduals`] (scaled by
//! `conditioning_scale`); the InstantID UNet adds them into its skip connections + mid output. The
//! control's `encoder_hidden_states` is a **caller-supplied parameter** (text for a tile-CN; the 16
//! face tokens for InstantID), so the branch is a generic primitive, not InstantID-specific.
//!
//! This forces the SDXL **`add_embedding`** the vendored UNet lacks (the time embedding is otherwise
//! just `time_proj ∘ time_embedding`): the diffusers `get_aug_embed` — `add_time_proj` over the 6-vec
//! `time_ids` concatenated with the pooled text embeds, through a 2-layer MLP, added to the time emb.
//! The InstantID inference UNet (sc-5491 phase 2c) reuses this exact construction.

use candle_core::Tensor;
use candle_nn::ops::silu;
use candle_nn::{self as nn, Module, VarBuilder};

use candle_gen::{CandleError, Result};

use super::conv::{conv2d, Conv2d};
use super::embeddings::{TimestepEmbedding, Timesteps};
use super::unet_2d::{BlockConfig, UNet2DConditionModelConfig};
use super::unet_2d_blocks::{
    CrossAttnDownBlock2D, CrossAttnDownBlock2DConfig, DownBlock2D, DownBlock2DConfig,
    UNetMidBlock2DCrossAttn, UNetMidBlock2DCrossAttnConfig,
};

/// The canonical SDXL UNet sub-config (`stabilityai/stable-diffusion-xl-base-1.0/unet/config.json`) —
/// 3 blocks `320/640/1280`, transformer depths `[—, 2, 10]`, 5/10/20 heads, `cross_attention_dim 2048`,
/// linear projection. Shared by the ControlNet (and the sc-5491 phase-2c InstantID UNet); mirrors the
/// (private) copies in `training.rs` / `pipeline.rs`. `pub` so the candle-gen-kolors IP-Adapter
/// provider (sc-5488) builds the same vendored stack from the SDXL-family Kolors UNet.
pub fn sdxl_unet_config() -> UNet2DConditionModelConfig {
    let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    UNet2DConditionModelConfig {
        center_input_sample: false,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        blocks: vec![
            bc(320, None, 5),
            bc(640, Some(2), 10),
            bc(1280, Some(10), 20),
        ],
        layers_per_block: 2,
        downsample_padding: 1,
        mid_block_scale_factor: 1.,
        norm_num_groups: 32,
        norm_eps: 1e-5,
        cross_attention_dim: 2048,
        sliced_attention_size: None,
        use_linear_projection: true,
    }
}

/// Config for an SDXL [`ControlNet`]: the encoder-copy UNet geometry + the ControlNet-specific extras
/// (the `add_embedding` projection dims, the conditioning-embedding conv channels).
#[derive(Clone, Debug)]
pub struct ControlNetConfig {
    /// The down/mid geometry (shared with the SDXL UNet — the ControlNet is an encoder copy).
    pub unet: UNet2DConditionModelConfig,
    /// `add_time_proj` sinusoidal width per `time_ids` element (diffusers `addition_time_embed_dim`, 256).
    pub addition_time_embed_dim: usize,
    /// `add_embedding` input width (diffusers `projection_class_embeddings_input_dim`): pooled text
    /// (1280) + `time_ids_len`·`addition_time_embed_dim` (6·256) = 2816 for SDXL.
    pub projection_class_embeddings_input_dim: usize,
    /// Control-image channel count (3, RGB).
    pub conditioning_channels: usize,
    /// `controlnet_cond_embedding` block channels (diffusers default `(16, 32, 96, 256)`).
    pub cond_block_out_channels: Vec<usize>,
}

impl ControlNetConfig {
    /// The stock SDXL ControlNet config (the InstantID IdentityNet + any diffusers SDXL ControlNet).
    pub fn sdxl() -> Self {
        Self {
            unet: sdxl_unet_config(),
            addition_time_embed_dim: 256,
            projection_class_embeddings_input_dim: 2816,
            conditioning_channels: 3,
            cond_block_out_channels: vec![16, 32, 96, 256],
        }
    }

    /// The **Kolors** ControlNet config (`Kwai-Kolors/Kolors-ControlNet-*`, sc-5489). Kolors is an
    /// SDXL-family UNet, so the down/mid geometry + the conditioning-embedding conv stack are identical
    /// to SDXL; the one delta is the `add_embedding` projection input — `projection_class_embeddings_
    /// input_dim = 5632` (the ChatGLM3 pooled **4096** + `time_ids_len`·`addition_time_embed_dim`
    /// (6·256) = 1536), vs SDXL's 2816 (pooled 1280). The Kolors `ControlNetModel` ALSO ships its own
    /// `encoder_hid_proj` (4096→2048) — loaded + applied by the `candle-gen-kolors` control provider, NOT
    /// here, since [`ControlNet::forward`] takes the cross-attention `encoder_x` already projected (the
    /// branch stays a generic primitive, exactly as for the SDXL/InstantID `encoder_x`).
    pub fn kolors() -> Self {
        Self {
            unet: sdxl_unet_config(),
            addition_time_embed_dim: 256,
            projection_class_embeddings_input_dim: 5632,
            conditioning_channels: 3,
            cond_block_out_channels: vec![16, 32, 96, 256],
        }
    }
}

/// `ControlNetConditioningEmbedding`: `conv_in(cond_ch→16) → SiLU → [block → SiLU]×6 → conv_out
/// (256→out_channels)`. The six blocks alternate stride 1 / stride 2 (three stride-2 ⇒ 8× downsample to
/// latent resolution). No trailing SiLU after `conv_out`.
struct CondEmbedding {
    conv_in: Conv2d,
    blocks: Vec<Conv2d>,
    conv_out: Conv2d,
}

impl CondEmbedding {
    fn new(
        vs: VarBuilder,
        conditioning_channels: usize,
        block_out_channels: &[usize],
        out_channels: usize,
    ) -> Result<Self> {
        let cfg = |stride| nn::Conv2dConfig {
            padding: 1,
            stride,
            ..Default::default()
        };
        let conv_in = conv2d(
            conditioning_channels,
            block_out_channels[0],
            3,
            cfg(1),
            vs.pp("conv_in"),
        )?;
        // diffusers: for i in 0..len-1 { Conv(c[i]→c[i], s1); Conv(c[i]→c[i+1], s2) }.
        let mut blocks = Vec::with_capacity((block_out_channels.len() - 1) * 2);
        let vs_b = vs.pp("blocks");
        for i in 0..block_out_channels.len() - 1 {
            let (ci, co) = (block_out_channels[i], block_out_channels[i + 1]);
            blocks.push(conv2d(
                ci,
                ci,
                3,
                cfg(1),
                vs_b.pp(blocks.len().to_string()),
            )?);
            blocks.push(conv2d(
                ci,
                co,
                3,
                cfg(2),
                vs_b.pp(blocks.len().to_string()),
            )?);
        }
        let conv_out = conv2d(
            *block_out_channels.last().unwrap(),
            out_channels,
            3,
            cfg(1),
            vs.pp("conv_out"),
        )?;
        Ok(Self {
            conv_in,
            blocks,
            conv_out,
        })
    }

    /// `control`: NCHW `[B, cond_ch, H, W]` in `[0,1]` → `[B, out_channels, H/8, W/8]`.
    fn forward(&self, control: &Tensor) -> Result<Tensor> {
        let mut e = silu(&self.conv_in.forward(control)?)?;
        for b in &self.blocks {
            e = silu(&b.forward(&e)?)?;
        }
        Ok(self.conv_out.forward(&e)?)
    }
}

/// The control residuals from one ControlNet forward, already scaled by `conditioning_scale`.
pub struct ControlResiduals {
    /// Per-down-block residuals (matching the UNet's collected skip residuals 1:1 — 9 for SDXL).
    pub down: Vec<Tensor>,
    /// The mid-block residual.
    pub mid: Tensor,
}

impl ControlResiduals {
    /// Element-wise sum with another branch's residuals — the diffusers `MultiControlNetModel` rule:
    /// each sub-ControlNet's down samples + mid sample are summed before injection. Both branches must
    /// produce the same number of down residuals (they share the UNet skip geometry). Used to combine
    /// e.g. InstantID's IdentityNet + an OpenPose ControlNet.
    pub fn add(&self, other: &ControlResiduals) -> Result<ControlResiduals> {
        if self.down.len() != other.down.len() {
            return Err(CandleError::Msg(format!(
                "controlnet residual sum: branch down counts differ ({} vs {})",
                self.down.len(),
                other.down.len()
            )));
        }
        let mut down = Vec::with_capacity(self.down.len());
        for (a, b) in self.down.iter().zip(&other.down) {
            down.push((a + b)?);
        }
        Ok(ControlResiduals {
            down,
            mid: (&self.mid + &other.mid)?,
        })
    }
}

/// A down block of the encoder copy — `Basic` (no cross-attn) or `CrossAttn`. Mirrors the UNet's
/// `UNetDownBlock`, kept local so the ControlNet doesn't couple to that private enum.
enum CnDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

impl CnDownBlock {
    fn forward(
        &self,
        xs: &Tensor,
        temb: &Tensor,
        ehs: &Tensor,
    ) -> candle_core::Result<(Tensor, Vec<Tensor>)> {
        match self {
            CnDownBlock::Basic(b) => b.forward(xs, Some(temb)),
            CnDownBlock::CrossAttn(b) => b.forward(xs, Some(temb), Some(ehs)),
        }
    }
}

/// An SDXL ControlNet (UNet encoder copy + conditioning embedding + zero-conv heads). Built with the
/// vendored blocks' **math** attention (`use_flash_attn = false`): the vendored flash path is an
/// `unimplemented!()` stub, so the InstantID lane runs the materialized attention.
pub struct ControlNet {
    conv_in: Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    add_time_proj: Timesteps,
    add_embedding: TimestepEmbedding,
    down_blocks: Vec<CnDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    cond_embedding: CondEmbedding,
    /// One 1×1 zero-conv per down residual.
    down_zero: Vec<Conv2d>,
    /// The 1×1 zero-conv for the mid output.
    mid_zero: Conv2d,
    addition_time_embed_dim: usize,
}

/// The channel count of each collected down residual (`conv_in` output, then each down block's resnet
/// outputs + its optional downsample), in skip order — pins the 1×1 zero-conv in/out channels.
fn residual_channels(cfg: &UNet2DConditionModelConfig) -> Vec<usize> {
    let n = cfg.blocks.len();
    let mut chans = vec![cfg.blocks[0].out_channels]; // conv_in residual
    for (i, b) in cfg.blocks.iter().enumerate() {
        for _ in 0..cfg.layers_per_block {
            chans.push(b.out_channels);
        }
        if i < n - 1 {
            chans.push(b.out_channels); // downsample residual
        }
    }
    chans
}

impl ControlNet {
    /// Build from a diffusers SDXL `ControlNetModel` `VarBuilder` (diffusers key layout) + `cfg`.
    pub fn new(vs: VarBuilder, cfg: &ControlNetConfig) -> Result<Self> {
        let u = &cfg.unet;
        let n = u.blocks.len();
        let b_channels = u.blocks[0].out_channels;
        let bl_channels = u.blocks.last().unwrap().out_channels;
        let time_embed_dim = b_channels * 4;

        let conv_in = conv2d(
            4,
            b_channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv_in"),
        )?;
        let time_proj = Timesteps::new(b_channels, u.flip_sin_to_cos, u.freq_shift);
        let time_embedding =
            TimestepEmbedding::new(vs.pp("time_embedding"), b_channels, time_embed_dim)?;
        let add_time_proj =
            Timesteps::new(cfg.addition_time_embed_dim, u.flip_sin_to_cos, u.freq_shift);
        let add_embedding = TimestepEmbedding::new(
            vs.pp("add_embedding"),
            cfg.projection_class_embeddings_input_dim,
            time_embed_dim,
        )?;

        let vs_db = vs.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(n);
        for i in 0..n {
            let BlockConfig {
                out_channels,
                use_cross_attn,
                attention_head_dim,
            } = u.blocks[i];
            let in_channels = if i > 0 {
                u.blocks[i - 1].out_channels
            } else {
                b_channels
            };
            let db_cfg = DownBlock2DConfig {
                num_layers: u.layers_per_block,
                resnet_eps: u.norm_eps,
                resnet_groups: u.norm_num_groups,
                add_downsample: i < n - 1,
                downsample_padding: u.downsample_padding,
                ..Default::default()
            };
            if let Some(transformer_layers_per_block) = use_cross_attn {
                let c = CrossAttnDownBlock2DConfig {
                    downblock: db_cfg,
                    attn_num_head_channels: attention_head_dim,
                    cross_attention_dim: u.cross_attention_dim,
                    sliced_attention_size: u.sliced_attention_size,
                    use_linear_projection: u.use_linear_projection,
                    transformer_layers_per_block,
                };
                down_blocks.push(CnDownBlock::CrossAttn(CrossAttnDownBlock2D::new(
                    vs_db.pp(i.to_string()),
                    in_channels,
                    out_channels,
                    Some(time_embed_dim),
                    false,
                    c,
                )?));
            } else {
                down_blocks.push(CnDownBlock::Basic(DownBlock2D::new(
                    vs_db.pp(i.to_string()),
                    in_channels,
                    out_channels,
                    Some(time_embed_dim),
                    db_cfg,
                )?));
            }
        }

        let mid_transformer_layers = u.blocks.last().and_then(|b| b.use_cross_attn).unwrap_or(1);
        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            resnet_eps: u.norm_eps,
            output_scale_factor: u.mid_block_scale_factor,
            cross_attn_dim: u.cross_attention_dim,
            attn_num_head_channels: u.blocks.last().unwrap().attention_head_dim,
            resnet_groups: Some(u.norm_num_groups),
            use_linear_projection: u.use_linear_projection,
            transformer_layers_per_block: mid_transformer_layers,
            ..Default::default()
        };
        let mid_block = UNetMidBlock2DCrossAttn::new(
            vs.pp("mid_block"),
            bl_channels,
            Some(time_embed_dim),
            false,
            mid_cfg,
        )?;

        // Zero-conv heads (1×1, no pad), one per down residual + one for mid.
        let zero_cfg = nn::Conv2dConfig::default();
        let res_chans = residual_channels(u);
        let vs_dz = vs.pp("controlnet_down_blocks");
        let down_zero = res_chans
            .iter()
            .enumerate()
            .map(|(i, &c)| conv2d(c, c, 1, zero_cfg, vs_dz.pp(i.to_string())))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mid_zero = conv2d(
            bl_channels,
            bl_channels,
            1,
            zero_cfg,
            vs.pp("controlnet_mid_block"),
        )?;

        let cond_embedding = CondEmbedding::new(
            vs.pp("controlnet_cond_embedding"),
            cfg.conditioning_channels,
            &cfg.cond_block_out_channels,
            b_channels,
        )?;

        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            add_time_proj,
            add_embedding,
            down_blocks,
            mid_block,
            cond_embedding,
            down_zero,
            mid_zero,
            addition_time_embed_dim: cfg.addition_time_embed_dim,
        })
    }

    /// Precompute the conditioning embedding for the fixed control image — the `cond_embedding` conv
    /// stack, which is **step-invariant** (depends only on `control`, not the latents/timestep). Run
    /// once per generation and stored, so the ~30-step denoise loop doesn't re-evaluate it (F-069).
    /// `control`: NCHW `[B, 3, H, W]` in `[0,1]` → `[B, b_channels, H/8, W/8]`.
    pub fn embed_cond(&self, control: &Tensor) -> Result<Tensor> {
        self.cond_embedding.forward(control)
    }

    /// The SDXL time + `add_embedding` (`get_aug_embed`): `time_embedding(time_proj(t)) +
    /// add_embedding(cat[pooled, add_time_proj(time_ids)])`. Shared verbatim with the InstantID UNet.
    fn temb(
        &self,
        timestep: f64,
        text_emb: &Tensor,
        time_ids: &Tensor,
        batch: usize,
    ) -> Result<Tensor> {
        let dtype = text_emb.dtype();
        let dev = text_emb.device();
        let emb = (Tensor::ones(batch, dtype, dev)? * timestep)?;
        let emb = self
            .time_embedding
            .forward(&self.time_proj.forward(&emb)?)?; // [B, time_embed_dim]
        let (b, l) = time_ids.dims2()?;
        let time_embeds = self
            .add_time_proj
            .forward(&time_ids.flatten_all()?)? // [B·L, addition_time_embed_dim]
            .reshape((b, l * self.addition_time_embed_dim))?; // [B, L·addition_time_embed_dim]
        let add_embeds = Tensor::cat(&[text_emb, &time_embeds], 1)?; // [B, projection_input_dim]
        let aug = self.add_embedding.forward(&add_embeds)?; // [B, time_embed_dim]
        Ok((emb + aug)?)
    }

    /// Compute the control residuals for one denoise step.
    /// - `x`: NCHW latents `[B, 4, H/8, W/8]` (the same CFG-batched input the UNet sees).
    /// - `cond_embed`: the precomputed [`embed_cond`](Self::embed_cond) embedding for the fixed control.
    /// - `encoder_x`: cross-attention conditioning `[B, S, D]` — the face tokens for InstantID; generic.
    /// - `text_emb`: pooled text embeds `[B, 1280]`; `time_ids`: the SDXL micro-conditioning `[B, 6]`.
    /// - `scale`: `conditioning_scale` applied to every residual.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        cond_embed: &Tensor,
        timestep: f64,
        encoder_x: &Tensor,
        text_emb: &Tensor,
        time_ids: &Tensor,
        scale: f64,
    ) -> Result<ControlResiduals> {
        let batch = x.dim(0)?;
        let temb = self.temb(timestep, text_emb, time_ids, batch)?;

        // conv_in + the (precomputed, step-invariant) conditioning embedding.
        let mut h = (self.conv_in.forward(x)? + cond_embed)?;

        // Down — collect skip residuals (starting with the stem+cond output).
        let mut residuals: Vec<Tensor> = vec![h.clone()];
        for block in &self.down_blocks {
            let (out, res) = block.forward(&h, &temb, encoder_x)?;
            h = out;
            residuals.extend(res);
        }

        // Mid.
        let h = self.mid_block.forward(&h, Some(&temb), Some(encoder_x))?;

        // Zero-conv heads + scale. Each collected skip residual pairs with exactly one zero-conv.
        if residuals.len() != self.down_zero.len() {
            return Err(CandleError::Msg(format!(
                "sdxl controlnet: {} down residuals but {} zero-conv heads — control branch geometry \
                 does not match the loaded down_zero convs",
                residuals.len(),
                self.down_zero.len()
            )));
        }
        let mut down = Vec::with_capacity(residuals.len());
        for (r, z) in residuals.iter().zip(&self.down_zero) {
            down.push((z.forward(r)? * scale)?);
        }
        let mid = (self.mid_zero.forward(&h)? * scale)?;
        Ok(ControlResiduals { down, mid })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    /// A tiny SDXL-shaped ControlNet config (one basic + one cross-attn down block, cross-attn mid)
    /// that exercises every path cheaply on CPU: conv_in + the add_embedding, the vendored down/mid
    /// blocks, the conditioning-embedding 8× downsample, and the zero-conv heads.
    fn tiny_cfg() -> ControlNetConfig {
        let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim,
        };
        ControlNetConfig {
            unet: UNet2DConditionModelConfig {
                center_input_sample: false,
                flip_sin_to_cos: true,
                freq_shift: 0.,
                blocks: vec![bc(32, None, 8), bc(64, Some(1), 8)],
                layers_per_block: 1,
                downsample_padding: 1,
                mid_block_scale_factor: 1.,
                norm_num_groups: 32,
                norm_eps: 1e-5,
                cross_attention_dim: 16,
                sliced_attention_size: None,
                use_linear_projection: false,
            },
            addition_time_embed_dim: 8,
            // pooled(16) + time_ids_len(2)·addition_time_embed_dim(8) = 32.
            projection_class_embeddings_input_dim: 32,
            conditioning_channels: 3,
            cond_block_out_channels: vec![16, 32, 96, 256],
        }
    }

    #[test]
    fn kolors_config_differs_from_sdxl_only_in_add_embedding_input() {
        let sdxl = ControlNetConfig::sdxl();
        let kolors = ControlNetConfig::kolors();
        // The one Kolors delta: the `add_embedding` projection input is 5632 (ChatGLM3 pooled 4096 +
        // 6·256) vs SDXL's 2816 (pooled 1280 + 6·256). The down/mid geometry + cond-embed convs match.
        assert_eq!(sdxl.projection_class_embeddings_input_dim, 2816);
        assert_eq!(kolors.projection_class_embeddings_input_dim, 4096 + 6 * 256);
        assert_eq!(kolors.addition_time_embed_dim, sdxl.addition_time_embed_dim);
        assert_eq!(kolors.conditioning_channels, sdxl.conditioning_channels);
        assert_eq!(kolors.cond_block_out_channels, sdxl.cond_block_out_channels);
        assert_eq!(
            residual_channels(&kolors.unet),
            residual_channels(&sdxl.unet)
        );
    }

    #[test]
    fn residual_channel_sequence_matches_sdxl() {
        // SDXL: conv_in(320) + block0[320,320,320(ds)] + block1[640,640,640(ds)] + block2[1280,1280].
        let chans = residual_channels(&sdxl_unet_config());
        assert_eq!(
            chans,
            vec![320, 320, 320, 320, 640, 640, 640, 1280, 1280],
            "SDXL has 9 down residuals (one per zero-conv head)"
        );
    }

    #[test]
    fn controlnet_forward_residual_shapes_and_scale() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cfg = tiny_cfg();
        let cn = ControlNet::new(vb, &cfg).unwrap();

        let b = 2usize;
        // latents [B,4,16,16]; control [B,3,128,128] → cond_embed [B,32,16,16] (8× downsample).
        let x = Tensor::randn(0f32, 1f32, (b, 4, 16, 16), &dev).unwrap();
        let control = Tensor::rand(0f32, 1f32, (b, 3, 128, 128), &dev).unwrap();
        let ehs = Tensor::randn(0f32, 1f32, (b, 5, 16), &dev).unwrap(); // [B, S, cross_attention_dim]
        let pooled = Tensor::randn(0f32, 1f32, (b, 16), &dev).unwrap();
        let time_ids = Tensor::randn(0f32, 1f32, (b, 2), &dev).unwrap();

        let cond_embed = cn.embed_cond(&control).unwrap();
        assert_eq!(cond_embed.dims(), &[b, 32, 16, 16]);

        let res = cn
            .forward(&x, &cond_embed, 500.0, &ehs, &pooled, &time_ids, 0.8)
            .unwrap();
        // tiny geometry: conv_in + block0[32,32(ds)] + block1[64] = 4 down residuals.
        assert_eq!(res.down.len(), 4);
        for d in &res.down {
            assert!(d
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()));
        }
        assert_eq!(res.mid.dim(0).unwrap(), b);

        // scale = 0 zeroes every residual (the zero-conv outputs scaled to nothing) — confirms the
        // conditioning_scale is applied to both down and mid.
        let z = cn
            .forward(&x, &cond_embed, 500.0, &ehs, &pooled, &time_ids, 0.0)
            .unwrap();
        let maxabs = |t: &Tensor| {
            t.abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        for d in &z.down {
            assert_eq!(maxabs(d), 0.0);
        }
        assert_eq!(maxabs(&z.mid), 0.0);
    }

    #[test]
    fn residual_sum_is_elementwise_and_rejects_mismatch() {
        let dev = Device::Cpu;
        let t = |v: &[f32]| Tensor::from_vec(v.to_vec(), v.len(), &dev).unwrap();
        let a = ControlResiduals {
            down: vec![t(&[1.0, 2.0, 3.0]), t(&[4.0, 5.0])],
            mid: t(&[10.0, 20.0]),
        };
        let b = ControlResiduals {
            down: vec![t(&[0.5, 0.5, 0.5]), t(&[-1.0, 1.0])],
            mid: t(&[1.0, -2.0]),
        };
        let s = a.add(&b).unwrap();
        assert_eq!(s.down[0].to_vec1::<f32>().unwrap(), vec![1.5, 2.5, 3.5]);
        assert_eq!(s.down[1].to_vec1::<f32>().unwrap(), vec![3.0, 6.0]);
        assert_eq!(s.mid.to_vec1::<f32>().unwrap(), vec![11.0, 18.0]);

        let bad = ControlResiduals {
            down: vec![t(&[0.0])],
            mid: t(&[0.0, 0.0]),
        };
        assert!(a.add(&bad).is_err());
    }
}
