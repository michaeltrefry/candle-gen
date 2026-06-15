//! 2D UNet Denoising Models
//!
//! The 2D Unet models take as input a noisy sample and the current diffusion
//! timestep and return a denoised version of the input.
use super::attention::CrossAttention;
use super::conv::{conv2d, Conv2d};
use super::embeddings::{TimestepEmbedding, Timesteps};
use super::unet_2d_blocks::*;
use candle_core::{DType, Device, Result, Tensor};
use candle_gen::train::gradient_checkpoint::Segment;
use candle_gen::train::lora::{LoraHost, LoraLinear};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    pub out_channels: usize,
    /// When `None` no cross-attn is used, when `Some(d)` then cross-attn is used and `d` is the
    /// number of transformer blocks to be used.
    pub use_cross_attn: Option<usize>,
    pub attention_head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct UNet2DConditionModelConfig {
    pub center_input_sample: bool,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f64,
    pub blocks: Vec<BlockConfig>,
    pub layers_per_block: usize,
    pub downsample_padding: usize,
    pub mid_block_scale_factor: f64,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    pub cross_attention_dim: usize,
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
}

impl Default for UNet2DConditionModelConfig {
    fn default() -> Self {
        Self {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig {
                    out_channels: 320,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 640,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 1280,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 1280,
                    use_cross_attn: None,
                    attention_head_dim: 8,
                },
            ],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
pub(crate) enum UNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

#[derive(Debug)]
enum UNetUpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

#[derive(Debug)]
pub struct UNet2DConditionModel {
    conv_in: Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<UNetUpBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: Conv2d,
    // SDXL `add_embedding` (`get_aug_embed`), loaded only for the InstantID inference path
    // ([`with_add_embedding`](UNet2DConditionModel::with_add_embedding)); the vendored UNet's plain
    // time embedding omits it. `None` on the training / stock build, so `forward` is unaffected (sc-5491).
    add_time_proj: Option<Timesteps>,
    add_embedding: Option<TimestepEmbedding>,
    addition_time_embed_dim: usize,
    span: tracing::Span,
    config: UNet2DConditionModelConfig,
}

impl UNet2DConditionModel {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        use_flash_attn: bool,
        config: UNet2DConditionModelConfig,
    ) -> Result<Self> {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = config.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = conv2d(in_channels, b_channels, 3, conv_cfg, vs.pp("conv_in"))?;

        let time_proj = Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift);
        let time_embedding =
            TimestepEmbedding::new(vs.pp("time_embedding"), b_channels, time_embed_dim)?;

        let vs_db = vs.pp("down_blocks");
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let in_channels = if i > 0 {
                    config.blocks[i - 1].out_channels
                } else {
                    b_channels
                };
                let db_cfg = DownBlock2DConfig {
                    num_layers: config.layers_per_block,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_downsample: i < n_blocks - 1,
                    downsample_padding: config.downsample_padding,
                    ..Default::default()
                };
                if let Some(transformer_layers_per_block) = use_cross_attn {
                    let config = CrossAttnDownBlock2DConfig {
                        downblock: db_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block,
                    };
                    let block = CrossAttnDownBlock2D::new(
                        vs_db.pp(i.to_string()),
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        use_flash_attn,
                        config,
                    )?;
                    Ok(UNetDownBlock::CrossAttn(block))
                } else {
                    let block = DownBlock2D::new(
                        vs_db.pp(i.to_string()),
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        db_cfg,
                    )?;
                    Ok(UNetDownBlock::Basic(block))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // https://github.com/huggingface/diffusers/blob/a76f2ad538e73b34d5fe7be08c8eb8ab38c7e90c/src/diffusers/models/unet_2d_condition.py#L462
        let mid_transformer_layers_per_block = match config.blocks.last() {
            None => 1,
            Some(block) => block.use_cross_attn.unwrap_or(1),
        };
        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            resnet_eps: config.norm_eps,
            output_scale_factor: config.mid_block_scale_factor,
            cross_attn_dim: config.cross_attention_dim,
            attn_num_head_channels: bl_attention_head_dim,
            resnet_groups: Some(config.norm_num_groups),
            use_linear_projection: config.use_linear_projection,
            transformer_layers_per_block: mid_transformer_layers_per_block,
            ..Default::default()
        };

        let mid_block = UNetMidBlock2DCrossAttn::new(
            vs.pp("mid_block"),
            bl_channels,
            Some(time_embed_dim),
            use_flash_attn,
            mid_cfg,
        )?;

        let vs_ub = vs.pp("up_blocks");
        let up_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig {
                    out_channels,
                    use_cross_attn,
                    attention_head_dim,
                } = config.blocks[n_blocks - 1 - i];

                // Enable automatic attention slicing if the config sliced_attention_size is set to 0.
                let sliced_attention_size = match config.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    _ => config.sliced_attention_size,
                };

                let prev_out_channels = if i > 0 {
                    config.blocks[n_blocks - i].out_channels
                } else {
                    bl_channels
                };
                let in_channels = {
                    let index = if i == n_blocks - 1 {
                        0
                    } else {
                        n_blocks - i - 2
                    };
                    config.blocks[index].out_channels
                };
                let ub_cfg = UpBlock2DConfig {
                    num_layers: config.layers_per_block + 1,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_upsample: i < n_blocks - 1,
                    ..Default::default()
                };
                if let Some(transformer_layers_per_block) = use_cross_attn {
                    let config = CrossAttnUpBlock2DConfig {
                        upblock: ub_cfg,
                        attn_num_head_channels: attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                        sliced_attention_size,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block,
                    };
                    let block = CrossAttnUpBlock2D::new(
                        vs_ub.pp(i.to_string()),
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        use_flash_attn,
                        config,
                    )?;
                    Ok(UNetUpBlock::CrossAttn(block))
                } else {
                    let block = UpBlock2D::new(
                        vs_ub.pp(i.to_string()),
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        ub_cfg,
                    )?;
                    Ok(UNetUpBlock::Basic(block))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let conv_norm_out = nn::group_norm(
            config.norm_num_groups,
            b_channels,
            config.norm_eps,
            vs.pp("conv_norm_out"),
        )?;
        let conv_out = conv2d(b_channels, out_channels, 3, conv_cfg, vs.pp("conv_out"))?;
        let span = tracing::span!(tracing::Level::TRACE, "unet2d");
        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            add_time_proj: None,
            add_embedding: None,
            addition_time_embed_dim: 0,
            span,
            config,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.forward_with_additional_residuals(xs, timestep, encoder_hidden_states, None, None)
    }

    pub fn forward_with_additional_residuals(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        down_block_additional_residuals: Option<&[Tensor]>,
        mid_block_additional_residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 1. time (the plain SDXL time embedding; the InstantID path folds in the `add_embedding`).
        let bsize = xs.dim(0)?;
        let emb = (Tensor::ones(bsize, xs.dtype(), xs.device())? * timestep)?;
        let emb = self.time_proj.forward(&emb)?;
        let emb = self.time_embedding.forward(&emb)?;
        self.run_blocks(
            xs,
            &emb,
            encoder_hidden_states,
            down_block_additional_residuals,
            mid_block_additional_residual,
        )
    }

    /// The conv-in → down → mid → up → conv-out body, parameterized on the precomputed time embedding
    /// `emb` (so the plain `forward` and the InstantID [`forward_instantid`](Self::forward_instantid) —
    /// which differ only in whether the SDXL `add_embedding` is folded into `emb` — share it) + the
    /// optional ControlNet residuals. Byte-identical to the previous inline body (the vendored-vs-stock
    /// parity test pins it).
    fn run_blocks(
        &self,
        xs: &Tensor,
        emb: &Tensor,
        encoder_hidden_states: &Tensor,
        down_block_additional_residuals: Option<&[Tensor]>,
        mid_block_additional_residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_bsize, _channels, height, width) = xs.dims4()?;
        let n_blocks = self.config.blocks.len();
        let num_upsamplers = n_blocks - 1;
        let default_overall_up_factor = 2usize.pow(num_upsamplers as u32);
        let forward_upsample_size =
            height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0;
        // 0. center input if necessary
        let xs = if self.config.center_input_sample {
            ((xs * 2.0)? - 1.0)?
        } else {
            xs.clone()
        };
        // 2. pre-process
        let xs = self.conv_in.forward(&xs)?;
        // 3. down
        let mut down_block_res_xs = vec![xs.clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (_xs, res_xs) = match down_block {
                UNetDownBlock::Basic(b) => b.forward(&xs, Some(emb))?,
                UNetDownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(emb), Some(encoder_hidden_states))?
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = _xs;
        }

        let new_down_block_res_xs =
            if let Some(down_block_additional_residuals) = down_block_additional_residuals {
                let mut v = vec![];
                // A previous version of this code had a bug because of the addition being made
                // in place via += hence modifying the input of the mid block.
                for (i, residuals) in down_block_additional_residuals.iter().enumerate() {
                    v.push((&down_block_res_xs[i] + residuals)?)
                }
                v
            } else {
                down_block_res_xs
            };
        let mut down_block_res_xs = new_down_block_res_xs;

        // 4. mid
        let xs = self
            .mid_block
            .forward(&xs, Some(emb), Some(encoder_hidden_states))?;
        let xs = match mid_block_additional_residual {
            None => xs,
            Some(m) => (m + xs)?,
        };
        // 5. up
        let mut xs = xs;
        let mut upsample_size = None;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up_block {
                UNetUpBlock::Basic(b) => b.resnets.len(),
                UNetUpBlock::CrossAttn(b) => b.upblock.resnets.len(),
            };
            let res_xs = down_block_res_xs.split_off(down_block_res_xs.len() - n_resnets);
            if i < n_blocks - 1 && forward_upsample_size {
                let (_, _, h, w) = down_block_res_xs.last().unwrap().dims4()?;
                upsample_size = Some((h, w))
            }
            xs = match up_block {
                UNetUpBlock::Basic(b) => b.forward(&xs, &res_xs, Some(emb), upsample_size)?,
                UNetUpBlock::CrossAttn(b) => b.forward(
                    &xs,
                    &res_xs,
                    Some(emb),
                    upsample_size,
                    Some(encoder_hidden_states),
                )?,
            };
        }
        // 6. post-process
        let xs = self.conv_norm_out.forward(&xs)?;
        let xs = nn::ops::silu(&xs)?;
        self.conv_out.forward(&xs)
    }
}

impl LoraHost for UNet2DConditionModel {
    /// Walk every cross-attention block (down/mid/up) and visit its adaptable projections. The
    /// `Basic` (non-cross-attn) down/up blocks have no `SpatialTransformer`, hence no targets.
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for db in self.down_blocks.iter_mut() {
            if let UNetDownBlock::CrossAttn(b) = db {
                b.visit_lora_mut(f)?;
            }
        }
        self.mid_block.visit_lora_mut(f)?;
        for ub in self.up_blocks.iter_mut() {
            if let UNetUpBlock::CrossAttn(b) = ub {
                b.visit_lora_mut(f)?;
            }
        }
        Ok(())
    }
}

impl UNet2DConditionModel {
    /// The PEFT module paths of every adaptable attention projection (`to_q`/`to_k`/`to_v`/`to_out.0`
    /// across the down/mid/up cross-attention blocks), in deterministic walk order. Drives trainer
    /// target resolution + the `mid_block` exclusion check for LoKr (sc-2640).
    pub fn lora_target_paths(&mut self) -> candle_gen::Result<Vec<String>> {
        let mut paths = Vec::new();
        LoraHost::visit_lora_mut(self, &mut |lin| {
            paths.push(lin.path().to_string());
            Ok(())
        })?;
        Ok(paths)
    }
}

/// Gradient-checkpointing decomposition of the forward (sc-5165): the trainer drives these pieces
/// through [`candle_gen::train::gradient_checkpoint::checkpointed_backward`] so each down/mid/up block
/// is recomputed in the backward instead of retained. The pieces reproduce
/// [`forward`](UNet2DConditionModel::forward) exactly (no additional residuals; `upsample_size` is
/// `None`, valid for the square, /32-bucketed training resolutions), so the checkpointed grads are the
/// dense grads (modulo float reassociation) — pinned by the trainer's dense-vs-checkpoint parity test.
impl UNet2DConditionModel {
    /// The frozen time embedding (`timestep` → `[bsize, time_embed_dim]`): `time_proj` ∘
    /// `time_embedding`, exactly as `forward`'s step 1. No adapter; the trainer detaches it as a
    /// per-step constant shared across every block segment.
    pub fn time_embed(
        &self,
        timestep: f64,
        bsize: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let emb = (Tensor::ones(bsize, dtype, device)? * timestep)?;
        let emb = self.time_proj.forward(&emb)?;
        self.time_embedding.forward(&emb)
    }

    /// The pre-process prelude: optional `center_input_sample`, then `conv_in` (`forward` steps 0+2).
    /// Its output is both the first hidden state and the first skip residual (`res₀`).
    pub fn conv_in_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.config.center_input_sample {
            ((xs * 2.0)? - 1.0)?
        } else {
            xs.clone()
        };
        self.conv_in.forward(&xs)
    }

    /// The post-process tail: `conv_norm_out` → silu → `conv_out` (`forward` step 6). Frozen (no
    /// adapter); the trainer folds it into its loss segment, so it is recomputed cheaply in the
    /// backward like any other checkpointed work.
    pub fn head_out(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv_norm_out.forward(xs)?;
        let xs = nn::ops::silu(&xs)?;
        self.conv_out.forward(&xs)
    }

    /// Build the down→mid→up [`Segment`]s over a `[hidden, res…]` state vector. The trainer seeds the
    /// input state `[conv_in_out, conv_in_out]` (`hidden`, `res₀`), then appends a final loss segment
    /// mapping the post-up `[hidden]` → `[loss]` (via [`head_out`](Self::head_out)).
    ///
    /// State convention: index 0 is the live hidden; indices `1..` are the skip residuals in append
    /// order. A down block appends its residuals (so the up blocks pop the last `n_resnets` LIFO,
    /// matching `forward`'s `split_off`); the mid block passes residuals through untouched.
    pub fn block_segments<'a>(&'a self, emb: &'a Tensor, ehs: &'a Tensor) -> Vec<Segment<'a>> {
        let mut segs: Vec<Segment<'a>> = Vec::with_capacity(self.down_blocks.len() * 2 + 1);

        // Down: [hidden, res…] → [new_hidden, res…, new_res…] (append the produced residuals).
        for i in 0..self.down_blocks.len() {
            segs.push(Box::new(move |st: &[Tensor]| {
                let (new_hidden, new_res) = match &self.down_blocks[i] {
                    UNetDownBlock::Basic(b) => b.forward(&st[0], Some(emb))?,
                    UNetDownBlock::CrossAttn(b) => b.forward(&st[0], Some(emb), Some(ehs))?,
                };
                let mut out = Vec::with_capacity(st.len() + new_res.len());
                out.push(new_hidden);
                out.extend_from_slice(&st[1..]);
                out.extend(new_res);
                Ok(out)
            }));
        }

        // Mid: [hidden, res…] → [new_hidden, res…] (residuals untouched).
        segs.push(Box::new(move |st: &[Tensor]| {
            let new_hidden = self.mid_block.forward(&st[0], Some(emb), Some(ehs))?;
            let mut out = Vec::with_capacity(st.len());
            out.push(new_hidden);
            out.extend_from_slice(&st[1..]);
            Ok(out)
        }));

        // Up: pop the last `n_resnets` residuals (LIFO), [hidden, res…] → [new_hidden, res'…].
        for j in 0..self.up_blocks.len() {
            segs.push(Box::new(move |st: &[Tensor]| {
                let n = match &self.up_blocks[j] {
                    UNetUpBlock::Basic(b) => b.resnets.len(),
                    UNetUpBlock::CrossAttn(b) => b.upblock.resnets.len(),
                };
                let res = &st[1..];
                let split = res.len() - n;
                let res_for_block = &res[split..];
                let new_hidden = match &self.up_blocks[j] {
                    UNetUpBlock::Basic(b) => b.forward(&st[0], res_for_block, Some(emb), None)?,
                    UNetUpBlock::CrossAttn(b) => {
                        b.forward(&st[0], res_for_block, Some(emb), None, Some(ehs))?
                    }
                };
                let mut out = Vec::with_capacity(1 + split);
                out.push(new_hidden);
                out.extend_from_slice(&res[..split]);
                Ok(out)
            }));
        }

        segs
    }
}

/// InstantID inference surface on the vendored SDXL UNet (sc-5491, epic 5480): the SDXL
/// `add_embedding` the plain forward omits, the decoupled IP-Adapter cross-attention install +
/// per-generation token set, and the ControlNet-residual forward. All additive — the training / stock
/// `forward` is untouched (the IP branch is inert until installed + set, and `add_embedding` is `None`).
impl UNet2DConditionModel {
    /// Load the SDXL `add_embedding` (`get_aug_embed`) for the InstantID path: the vendored UNet's
    /// plain time embedding omits the pooled-text + `time_ids` micro-conditioning the SDXL identity
    /// render needs. `vs` is the UNet `VarBuilder` (the `add_embedding.*` keys are in the stock SDXL
    /// `unet/` checkpoint); `addition_time_embed_dim` is 256 and `projection_input_dim` is diffusers
    /// `projection_class_embeddings_input_dim` (2816 = pooled 1280 + 6·256).
    pub fn with_add_embedding(
        mut self,
        vs: nn::VarBuilder,
        addition_time_embed_dim: usize,
        projection_input_dim: usize,
    ) -> Result<Self> {
        let time_embed_dim = self.config.blocks[0].out_channels * 4;
        self.add_time_proj = Some(Timesteps::new(
            addition_time_embed_dim,
            self.config.flip_sin_to_cos,
            self.config.freq_shift,
        ));
        self.add_embedding = Some(TimestepEmbedding::new(
            vs.pp("add_embedding"),
            projection_input_dim,
            time_embed_dim,
        )?);
        self.addition_time_embed_dim = addition_time_embed_dim;
        Ok(self)
    }

    /// The SDXL time + `add_embedding` (`get_aug_embed`): `time_embedding(time_proj(t)) +
    /// add_embedding(cat[pooled, add_time_proj(time_ids)])`. Errors if `add_embedding` is not loaded.
    fn instantid_temb(
        &self,
        timestep: f64,
        text_emb: &Tensor,
        time_ids: &Tensor,
        xs: &Tensor,
    ) -> Result<Tensor> {
        let (add_time_proj, add_embedding) = match (&self.add_time_proj, &self.add_embedding) {
            (Some(p), Some(e)) => (p, e),
            _ => {
                return Err(candle_core::Error::Msg(
                    "instantid: UNet add_embedding not loaded (call with_add_embedding)".into(),
                ))
            }
        };
        let bsize = xs.dim(0)?;
        let emb = (Tensor::ones(bsize, xs.dtype(), xs.device())? * timestep)?;
        let emb = self
            .time_embedding
            .forward(&self.time_proj.forward(&emb)?)?;
        let (b, l) = time_ids.dims2()?;
        let time_embeds = add_time_proj
            .forward(&time_ids.flatten_all()?)?
            .reshape((b, l * self.addition_time_embed_dim))?;
        let add_embeds = Tensor::cat(&[text_emb, &time_embeds], 1)?;
        let aug = add_embedding.forward(&add_embeds)?;
        emb + aug
    }

    /// InstantID inference forward: the SDXL UNet with the `add_embedding` micro-conditioning
    /// (`text_emb` pooled + `time_ids`), the decoupled IP-Adapter cross-attention (active once
    /// [`set_ip_context`](Self::set_ip_context) has set the face tokens), and the optional IdentityNet /
    /// OpenPose ControlNet residuals. Reuses the shared [`run_blocks`](Self::run_blocks).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_instantid(
        &self,
        xs: &Tensor,
        timestep: f64,
        encoder_hidden_states: &Tensor,
        text_emb: &Tensor,
        time_ids: &Tensor,
        down_block_additional_residuals: Option<&[Tensor]>,
        mid_block_additional_residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let emb = self.instantid_temb(timestep, text_emb, time_ids, xs)?;
        self.run_blocks(
            xs,
            &emb,
            encoder_hidden_states,
            down_block_additional_residuals,
            mid_block_additional_residual,
        )
    }

    /// Install the IP-Adapter decoupled K/V `pairs` into the cross-attentions in the diffusers
    /// `attn_processors` order (down → up → mid; see [`visit_cross_attn_mut`](Self::visit_cross_attn_mut))
    /// — 70 pairs for SDXL. Errors on a count mismatch (too few or too many) rather than silently
    /// leaving cross-attentions un-/over-installed.
    pub fn install_ip_adapter(&mut self, pairs: Vec<(Tensor, Tensor)>) -> Result<()> {
        let mut pairs = pairs.into_iter();
        self.visit_cross_attn_mut(&mut |xa| match pairs.next() {
            Some((k, v)) => {
                xa.install_ip(k, v);
                Ok(())
            }
            None => Err(candle_core::Error::Msg(
                "ip_adapter: fewer K/V pairs than UNet cross-attentions".into(),
            )),
        })?;
        if pairs.next().is_some() {
            return Err(candle_core::Error::Msg(
                "ip_adapter: more K/V pairs than UNet cross-attentions".into(),
            ));
        }
        Ok(())
    }

    /// Set (or clear, with `None`) the IP tokens + scale on every cross-attention. Constant across the
    /// denoise, so call once per generation before the loop; `forward_instantid` then runs the decoupled
    /// branch. `None` reverts to plain SDXL.
    pub fn set_ip_context(&mut self, tokens: Option<&Tensor>, scale: f64) -> Result<()> {
        self.visit_cross_attn_mut(&mut |xa| {
            xa.set_ip(tokens, scale);
            Ok(())
        })
    }

    /// Walk every cross-attention (`attn2`) in diffusers `attn_processors` order — **down blocks, then
    /// up blocks, then mid last** — applying `f`. This is NOT the forward order (down → mid → up): the
    /// diffusers `attn_processors` dict follows `named_children` *registration* order, and the UNet
    /// assigns `down_blocks`, `up_blocks`, then `mid_block`, so the mid-block cross-attns come last. The
    /// saved `ip_adapter.{n}` indices number the pairs in exactly this order, so
    /// [`install_ip_adapter`](Self::install_ip_adapter) (ordered pair consume) MUST match it or a
    /// wrong-dim K/V pair lands on a block (a 640-dim pair on a 1280-dim block → matmul shape mismatch).
    /// [`set_ip_context`](Self::set_ip_context) sets the same tokens everywhere, so it is order-agnostic.
    fn visit_cross_attn_mut(
        &mut self,
        f: &mut dyn FnMut(&mut CrossAttention) -> Result<()>,
    ) -> Result<()> {
        for db in self.down_blocks.iter_mut() {
            if let UNetDownBlock::CrossAttn(b) = db {
                b.visit_cross_attn_mut(f)?;
            }
        }
        for ub in self.up_blocks.iter_mut() {
            if let UNetUpBlock::CrossAttn(b) = ub {
                b.visit_cross_attn_mut(f)?;
            }
        }
        self.mid_block.visit_cross_attn_mut(f)?;
        Ok(())
    }
}

#[cfg(test)]
mod instantid_tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    /// A tiny SDXL-shaped config: one basic + one cross-attn down block, cross-attn mid, mirrored up.
    /// Cross-attns: down1 (1) + mid (1) + up0 (2) = 4 — so the IP install consumes 4 pairs.
    fn tiny_cfg() -> UNet2DConditionModelConfig {
        UNet2DConditionModelConfig {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig {
                    out_channels: 32,
                    use_cross_attn: None,
                    attention_head_dim: 8,
                },
                BlockConfig {
                    out_channels: 64,
                    use_cross_attn: Some(1),
                    attention_head_dim: 8,
                },
            ],
            layers_per_block: 1,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 16,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }

    fn maxdiff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn forward_instantid_folds_add_embedding_and_threads_ip() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cfg = tiny_cfg();

        // Without `with_add_embedding`, the InstantID forward errors loudly (no silent plain-time path).
        let bare = UNet2DConditionModel::new(vb.clone(), 4, 4, false, cfg.clone()).unwrap();
        let (b, dev_ref) = (2usize, &dev);
        let x = Tensor::randn(0f32, 1f32, (b, 4, 16, 16), dev_ref).unwrap();
        let ehs = Tensor::randn(0f32, 1f32, (b, 5, 16), dev_ref).unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (b, 16), dev_ref).unwrap();
        let time_ids = Tensor::randn(0f32, 1f32, (b, 2), dev_ref).unwrap();
        assert!(bare
            .forward_instantid(&x, 500.0, &ehs, &pooled, &time_ids, None, None)
            .is_err());

        // pooled(16) + time_ids_len(2)·addition_time_embed_dim(8) = 32.
        let mut unet = UNet2DConditionModel::new(vb.clone(), 4, 4, false, cfg)
            .unwrap()
            .with_add_embedding(vb, 8, 32)
            .unwrap();

        // The InstantID forward folds in the (random, nonzero) add_embedding, so it differs from the
        // plain time-only `forward`; its shape matches the latents.
        let base = unet
            .forward_instantid(&x, 500.0, &ehs, &pooled, &time_ids, None, None)
            .unwrap();
        assert_eq!(base.dims(), &[b, 4, 16, 16]);
        let plain = unet.forward(&x, 500.0, &ehs).unwrap();
        assert!(
            maxdiff(&base, &plain) > 1e-4,
            "add_embedding must change the time embedding vs the plain forward"
        );

        // Install one K/V pair per cross-attn (inner 64, cross_attention_dim 16) + set tokens.
        let pair = || {
            (
                Tensor::randn(0f32, 1f32, (64, 16), &dev).unwrap(),
                Tensor::randn(0f32, 1f32, (64, 16), &dev).unwrap(),
            )
        };
        unet.install_ip_adapter(vec![pair(), pair(), pair(), pair()])
            .unwrap();
        let ip_tokens = Tensor::randn(0f32, 1f32, (b, 3, 16), &dev).unwrap();
        unet.set_ip_context(Some(&ip_tokens), 0.8).unwrap();
        let with_ip = unet
            .forward_instantid(&x, 500.0, &ehs, &pooled, &time_ids, None, None)
            .unwrap();
        assert!(
            maxdiff(&with_ip, &base) > 1e-4,
            "the decoupled IP branch must change the output once installed + set"
        );

        // Clearing the tokens reverts to the no-IP InstantID output.
        unet.set_ip_context(None, 0.0).unwrap();
        let cleared = unet
            .forward_instantid(&x, 500.0, &ehs, &pooled, &time_ids, None, None)
            .unwrap();
        assert!(
            maxdiff(&cleared, &base) < 1e-6,
            "clearing the IP tokens reverts to the no-IP output"
        );
    }

    /// An install with the wrong number of K/V pairs (too few / too many) errors rather than leaving
    /// cross-attentions un-/over-installed.
    #[test]
    fn install_ip_adapter_rejects_pair_count_mismatch() {
        let dev = Device::Cpu;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &dev);
        let pair = || {
            (
                Tensor::randn(0f32, 1f32, (64, 16), &dev).unwrap(),
                Tensor::randn(0f32, 1f32, (64, 16), &dev).unwrap(),
            )
        };
        let mut unet = UNet2DConditionModel::new(vb, 4, 4, false, tiny_cfg()).unwrap();
        assert!(unet.install_ip_adapter(vec![pair(), pair()]).is_err()); // too few (needs 4)
        let mut unet2 = UNet2DConditionModel::new(
            VarBuilder::from_varmap(&VarMap::new(), DType::F32, &dev),
            4,
            4,
            false,
            tiny_cfg(),
        )
        .unwrap();
        assert!(unet2
            .install_ip_adapter(vec![pair(), pair(), pair(), pair(), pair()])
            .is_err()); // too many
    }
}
