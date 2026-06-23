//! The Boogu mixed single/double-stream DiT (`BooguImageTransformer2DModel`) forward. Port of
//! `mlx-gen-boogu`'s `transformer/mod.rs`.
//!
//! Two entry points share one inner path: [`BooguTransformer::forward`] (text-to-image) and
//! [`BooguTransformer::forward_edit`] (single-reference text+image-to-image).
//!
//! Text-to-image flow (the reference-image blocks stay dormant):
//! ```text
//!   time_caption_embed:  temb = TimestepEmbedder(sinusoid(t·scale));  caption = Linear(RMSNorm(instr))
//!   patchify(p=2, 16→64) → x_embedder                                 → img tokens  [1, Li, 3360]
//!   context_refiner ×2  (no modulation)        on instruct tokens     [1, Lt, 3360]
//!   noise_refiner   ×2  (modulated)            on img tokens
//!   double_stream   ×8  (joint instruct↔img attn + img self-attn)
//!   fuse → [instruct; img]                                            [1, Lt+Li, 3360]
//!   single_stream   ×32 (modulated)            on the joint sequence
//!   norm_out (LuminaLayerNormContinuous + temb) → Linear(3360→64)
//!   unpatchify(img tokens)                                            → velocity [1, 16, H, W]
//! ```
//!
//! Per-sample `B = 1`: true-CFG runs this twice (cond/uncond) rather than padding a batch, so every
//! attention is full/unmasked. The instruction features arrive already trimmed to the valid token
//! count (the candle tokenizer emits no padding), so `cap_len = instruction_hidden.dim(1)`.

pub mod block;
pub mod rope;

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{Linear, Module};

use crate::config::BooguConfig;
use crate::loader::{layernorm_noaffine, linear, rmsnorm, Weights};
use block::{DoubleBlock, ModBlock, PlainBlock};
use rope::RopeTables;

/// The Boogu DiT. Carries the text-to-image modules plus the reference-image conditioning path
/// (`ref_image_patch_embedder` + `ref_image_refiner` + `image_index_embedding`) the Edit forward
/// exercises; the T2I forward simply leaves those dormant.
pub struct BooguTransformer {
    cfg: BooguConfig,
    device: Device,
    dtype: DType,
    x_embedder: Linear,
    ref_image_patch_embedder: Linear,
    image_index_embedding: Tensor,
    caption_norm: Tensor,
    caption_linear: Linear,
    time_lin1: Linear,
    time_lin2: Linear,
    context_refiner: Vec<PlainBlock>,
    noise_refiner: Vec<ModBlock>,
    ref_image_refiner: Vec<ModBlock>,
    double_stream: Vec<DoubleBlock>,
    single_stream: Vec<ModBlock>,
    norm_out_lin1: Linear,
    norm_out_lin2: Linear,
}

impl BooguTransformer {
    /// Build from a loaded `transformer/` weight set.
    pub fn load(w: &Weights, cfg: &BooguConfig) -> Result<Self> {
        let (heads, kv, hd) = (cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim());
        let eps = cfg.norm_eps;

        let plain = |name: String| PlainBlock::load(w, &name, heads, kv, hd, eps);
        let mod_ = |name: String| ModBlock::load(w, &name, heads, kv, hd, eps);
        let dbl = |name: String| DoubleBlock::load(w, &name, heads, kv, hd, eps);

        Ok(Self {
            cfg: cfg.clone(),
            device: w.device().clone(),
            dtype: w.dtype(),
            x_embedder: linear(w, "x_embedder", true)?,
            ref_image_patch_embedder: linear(w, "ref_image_patch_embedder", true)?,
            image_index_embedding: w.get("image_index_embedding")?,
            caption_norm: w.get("time_caption_embed.caption_embedder.0.weight")?,
            caption_linear: linear(w, "time_caption_embed.caption_embedder.1", true)?,
            time_lin1: linear(w, "time_caption_embed.timestep_embedder.linear_1", true)?,
            time_lin2: linear(w, "time_caption_embed.timestep_embedder.linear_2", true)?,
            context_refiner: (0..cfg.num_refiner_layers)
                .map(|i| plain(format!("context_refiner.{i}")))
                .collect::<Result<_>>()?,
            noise_refiner: (0..cfg.num_refiner_layers)
                .map(|i| mod_(format!("noise_refiner.{i}")))
                .collect::<Result<_>>()?,
            ref_image_refiner: (0..cfg.num_refiner_layers)
                .map(|i| mod_(format!("ref_image_refiner.{i}")))
                .collect::<Result<_>>()?,
            double_stream: (0..cfg.num_double_stream_layers)
                .map(|i| dbl(format!("double_stream_layers.{i}")))
                .collect::<Result<_>>()?,
            single_stream: (0..cfg.num_single_stream_layers())
                .map(|i| mod_(format!("single_stream_layers.{i}")))
                .collect::<Result<_>>()?,
            norm_out_lin1: linear(w, "norm_out.linear_1", true)?,
            norm_out_lin2: linear(w, "norm_out.linear_2", true)?,
        })
    }

    /// Text-to-image velocity prediction.
    ///
    /// - `latent`: `[1, 16, H, W]` (H, W multiples of `patch_size`),
    /// - `timestep`: `[1]` f32 (raw, pre-scale),
    /// - `instruction_hidden`: `[1, L, 4096]` raw Qwen3-VL `last_hidden_state` (already trimmed).
    ///
    /// Returns the velocity `[1, 16, H, W]`.
    pub fn forward(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        instruction_hidden: &Tensor,
    ) -> Result<Tensor> {
        self.forward_inner(latent, None, timestep, instruction_hidden)
    }

    /// Edit (single-reference text+image-to-image) velocity prediction. Identical to [`Self::forward`]
    /// but with a clean reference latent `ref_latent` (`[1, 16, rH, rW]`, the VAE-encoded reference)
    /// packed — after its own `ref_image_patch_embedder` + `image_index_embedding` + `ref_image_refiner`
    /// — *before* the noise tokens in the combined image sequence.
    pub fn forward_edit(
        &self,
        latent: &Tensor,
        ref_latent: &Tensor,
        timestep: &Tensor,
        instruction_hidden: &Tensor,
    ) -> Result<Tensor> {
        self.forward_inner(latent, Some(ref_latent), timestep, instruction_hidden)
    }

    fn forward_inner(
        &self,
        latent: &Tensor,
        ref_latent: Option<&Tensor>,
        timestep: &Tensor,
        instruction_hidden: &Tensor,
    ) -> Result<Tensor> {
        let p = self.cfg.patch_size;
        let (_, _, h, w) = latent.dims4()?;
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;
        let dt = self.dtype;
        let axes_dim = self.cfg.axes_dim_rope[0];
        let theta = self.cfg.rope_theta;

        let latent = latent.to_dtype(dt)?;
        // The candle tokenizer emits no padding, so every instruction token is valid.
        let instruct = instruction_hidden.to_dtype(dt)?;
        let cap_len = instruct.dim(1)?;

        // Timestep + caption embedding.
        let temb = self.timestep_embed(timestep)?; // [1, 1, 1024]
        let caption = self.caption_linear.forward(&rmsnorm(
            &instruct,
            &self.caption_norm,
            self.cfg.norm_eps,
        )?)?; // [1, cap, 3360]

        // Patchify the noise latent → target image tokens.
        let img = self.x_embedder.forward(&patchify(&latent, p)?)?; // [1, img_len, 3360]

        // Reference image (Edit): patch-embed + add the per-image index embedding (single ref ⇒ 0).
        let (ref_tokens, rope) = match ref_latent {
            Some(rl) => {
                let rl = rl.to_dtype(dt)?;
                let (_, _, rh, rw) = rl.dims4()?;
                let (rht, rwt) = (rh / p, rw / p);
                let ref_t = self.ref_image_patch_embedder.forward(&patchify(&rl, p)?)?;
                let idx0 = self
                    .image_index_embedding
                    .narrow(0, 0, 1)?
                    .reshape((1, 1, self.cfg.hidden_size))?
                    .to_dtype(dt)?;
                let ref_t = ref_t.broadcast_add(&idx0)?;
                let rope = RopeTables::build_edit(
                    cap_len,
                    rht,
                    rwt,
                    ht,
                    wt,
                    axes_dim,
                    theta,
                    &self.device,
                )?;
                (Some(ref_t), rope)
            }
            None => (
                None,
                RopeTables::build_t2i(cap_len, ht, wt, axes_dim, theta, &self.device)?,
            ),
        };

        let (text_cos, text_sin) = rope.text()?;
        let (noise_cos, noise_sin) = rope.image()?;
        let (comb_cos, comb_sin) = rope.combined_image()?;
        let (joint_cos, joint_sin) = rope.joint();

        // Context refinement (instruction stream).
        let mut instruct_h = caption;
        for blk in &self.context_refiner {
            instruct_h = blk.forward(&instruct_h, &text_cos, &text_sin)?;
        }

        // Noise refinement (target image stream).
        let mut img = img;
        for blk in &self.noise_refiner {
            img = blk.forward(&img, &noise_cos, &noise_sin, &temb)?;
        }

        // Reference refinement, then prepend the refined reference tokens to form the combined
        // image sequence `[ref; noise]` (Edit). T2I leaves the combined sequence as the noise tokens.
        let mut img = match ref_tokens {
            Some(mut ref_t) => {
                let (ref_cos, ref_sin) = rope.ref_image()?;
                for blk in &self.ref_image_refiner {
                    ref_t = blk.forward(&ref_t, &ref_cos, &ref_sin, &temb)?;
                }
                Tensor::cat(&[&ref_t, &img], 1)?
            }
            None => img,
        };

        // Dual-stream blocks (joint instruct↔combined-image attn + combined-image self-attn).
        for blk in &self.double_stream {
            let (ni, nt) = blk.forward(
                &img,
                &instruct_h,
                &joint_cos,
                &joint_sin,
                &comb_cos,
                &comb_sin,
                &temb,
            )?;
            img = ni;
            instruct_h = nt;
        }

        // Fuse to the joint sequence, then single-stream blocks.
        let mut joint = Tensor::cat(&[&instruct_h, &img], 1)?; // [1, cap+ref+img, 3360]
        for blk in &self.single_stream {
            joint = blk.forward(&joint, &joint_cos, &joint_sin, &temb)?;
        }

        // Continuous-AdaLN output projection (LuminaLayerNormContinuous, eps 1e-6, no affine).
        let scale = self.norm_out_lin1.forward(&temb.silu()?)?; // [1, 1, 3360]
        let normed = layernorm_noaffine(&joint, 1e-6)?;
        let normed = normed.broadcast_mul(&(scale + 1.0)?)?;
        let out = self.norm_out_lin2.forward(&normed)?; // [1, cap+ref+img, 64]

        // Unpatchify the trailing target-image tokens into the velocity (reference tokens, when
        // present, are dropped — only the noise/target slice is the prediction).
        let total = out.dim(1)?;
        let img_tokens = out.narrow(1, total - img_len, img_len)?;
        unpatchify(&img_tokens, ht, wt, p, self.cfg.out_channels)
    }

    /// `Lumina2CombinedTimestepCaptionEmbedding` timestep branch:
    /// `sinusoid(timestep · timestep_scale, 256) → Linear → SiLU → Linear` → `[1, 1, 1024]`.
    fn timestep_embed(&self, timestep: &Tensor) -> Result<Tensor> {
        let scaled = (timestep.to_dtype(DType::F32)? * self.cfg.timestep_scale as f64)?;
        let proj = sinusoidal_timestep(&scaled, 256, &self.device)?.to_dtype(self.dtype)?; // [1, 256]
        let t = self.time_lin1.forward(&proj)?;
        let t = t.silu()?;
        let t = self.time_lin2.forward(&t)?; // [1, 1024]
        t.unsqueeze(1) // [1, 1, 1024]
    }
}

/// diffusers `get_timestep_embedding(x, dim, flip_sin_to_cos=True, downscale_freq_shift=0,
/// max_period=10000)`: `freq_i = 10000^(−i/half)`, `emb = x·freq`, `concat([cos, sin], -1)` (cos
/// first). `x`: `[N]` → `[N, dim]`. Built in f32.
fn sinusoidal_timestep(x: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let neg_ln = -(10000f64.ln()) as f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (neg_ln * i as f32 / half as f32).exp())
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), device)?; // [1, half]
    let n = x.dim(0)?;
    let emb = x.reshape((n, 1))?.broadcast_mul(&freqs)?; // [N, half]
    Tensor::cat(&[emb.cos()?, emb.sin()?], D::Minus1) // [N, dim]
}

/// `c (h p1) (w p2) -> (h w) (p1 p2 c)` with batch: `[1, C, H, W] → [1, (H/p)(W/p), p·p·C]`.
fn patchify(latent: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c, h, w) = latent.dims4()?;
    let (ht, wt) = (h / p, w / p);
    let x = latent.reshape((b, c, ht, p, wt, p))?; // B, C, h, p1, w, p2
    let x = x.permute((0, 2, 4, 3, 5, 1))?; // B, h, w, p1, p2, C
    x.contiguous()?.reshape((b, ht * wt, p * p * c))
}

/// `(h w) (p1 p2 c) -> c (h p1) (w p2)` with batch: `[1, (h)(w), p·p·C] → [1, C, h·p, w·p]`.
fn unpatchify(tokens: &Tensor, ht: usize, wt: usize, p: usize, c: usize) -> Result<Tensor> {
    let b = tokens.dim(0)?;
    // `tokens` is a `narrow`ed slice of the output sequence; contiguate before reshape.
    let x = tokens.contiguous()?.reshape((b, ht, wt, p, p, c))?; // B, h, w, p1, p2, C
    let x = x.permute((0, 5, 1, 3, 2, 4))?; // B, C, h, p1, w, p2
    x.contiguous()?.reshape((b, c, ht * p, wt * p))
}
