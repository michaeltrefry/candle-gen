//! The Krea 2 dense single-stream DiT (`Krea2Transformer2DModel` / reference `mmdit.py`
//! `SingleStreamDiT`) forward. Port of `mlx-gen-krea`'s `transformer/mod.rs`.
//!
//! ```text
//!   img_in:        img tokens = Linear(patchify(latent, p=2))          [b, img_len, 6144]
//!   time_embed:    t   = Linear(GELU(Linear(sinusoid(timestep))))      [b, 1, 6144]
//!   time_mod_proj: tvec = Linear(GELU(t))                              [b, 1, 6·6144]   (shared modulation)
//!   text_fusion:   ctx = aggregate(stacked 12 Qwen3-VL layers)         [b, cap, 2560]
//!   txt_in:        ctx = Linear(GELU(Linear(RMSNorm(ctx))))            [b, cap, 6144]
//!   combined = [ctx ; img]                                            [b, cap+img_len, 6144]
//!   28× transformer_blocks (gated single-stream, DoubleSharedModulation, 3-axis RoPE)
//!   final_layer:   (1+scale)·RMSNorm(x) + shift → Linear(6144→64)      [b, cap+img_len, 64]
//!   slice image tokens → unpatchify                                   → velocity [b, 16, H, W]
//! ```
//!
//! Per-sample `B = 1`: the text stream arrives already trimmed to its valid length (the candle
//! tokenizer emits no padding) and the whole sequence runs **unmasked**.

pub mod block;
pub mod rope;

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{Linear, Module};

use crate::config::Krea2Config;
use crate::loader::{linear, Weights};
use block::{RmsScale, SingleStreamBlock, TextFusionTransformer};
use rope::RopeTables;

/// The Krea 2 single-stream DiT.
pub struct Krea2Transformer {
    cfg: Krea2Config,
    device: Device,
    dtype: DType,
    img_in: Linear,
    time_embed_l1: Linear,
    time_embed_l2: Linear,
    time_mod_proj: Linear,
    txt_in_norm: RmsScale,
    txt_in_l1: Linear,
    txt_in_l2: Linear,
    text_fusion: TextFusionTransformer,
    blocks: Vec<SingleStreamBlock>,
    final_norm: RmsScale,
    final_linear: Linear,
    final_sstable: Tensor, // [1, 2, hidden]
}

impl Krea2Transformer {
    /// Build from a loaded `transformer/` weight set.
    pub fn load(w: &Weights, cfg: &Krea2Config) -> Result<Self> {
        let (heads, kv, hd, eps) = (
            cfg.num_attention_heads,
            cfg.num_kv_heads,
            cfg.attention_head_dim,
            cfg.norm_eps,
        );
        let (theads, tkv) = (cfg.text_num_attention_heads, cfg.text_num_kv_heads);
        let hidden = cfg.hidden_size;

        let final_sstable = w
            .get("final_layer.scale_shift_table")?
            .reshape((1, 2, hidden))?;

        Ok(Self {
            cfg: cfg.clone(),
            device: w.device().clone(),
            dtype: w.dtype(),
            img_in: linear(w, "img_in", true)?,
            time_embed_l1: linear(w, "time_embed.linear_1", true)?,
            time_embed_l2: linear(w, "time_embed.linear_2", true)?,
            time_mod_proj: linear(w, "time_mod_proj", true)?,
            txt_in_norm: RmsScale::load(w, "txt_in.norm.weight", eps)?,
            txt_in_l1: linear(w, "txt_in.linear_1", true)?,
            txt_in_l2: linear(w, "txt_in.linear_2", true)?,
            text_fusion: TextFusionTransformer::load(
                w,
                cfg.num_layerwise_text_blocks,
                cfg.num_refiner_text_blocks,
                theads,
                tkv,
                hd,
                eps,
            )?,
            blocks: (0..cfg.num_layers)
                .map(|i| {
                    SingleStreamBlock::load(
                        w,
                        &format!("transformer_blocks.{i}"),
                        heads,
                        kv,
                        hd,
                        hidden,
                        eps,
                    )
                })
                .collect::<Result<_>>()?,
            final_norm: RmsScale::load(w, "final_layer.norm.weight", eps)?,
            final_linear: linear(w, "final_layer.linear", true)?,
            final_sstable,
        })
    }

    /// Velocity prediction.
    ///
    /// - `latent`: `[b, 16, H, W]` (H, W multiples of `patch_size`),
    /// - `timestep`: `[b]` f32 (raw flow time in `[0, 1]`),
    /// - `context`: `[b, n_tokens, num_text_layers, text_hidden]` — the stacked Qwen3-VL select-layer
    ///   hidden states (sc-7569), already trimmed to the valid token count (no padding).
    ///
    /// Returns the velocity `[b, 16, H, W]`.
    pub fn forward(&self, latent: &Tensor, timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        let cfg = &self.cfg;
        let p = cfg.patch_size;
        let dt = self.dtype;
        let (_, _, h, w) = latent.dims4()?;
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;
        let latent_ch = cfg.in_channels / (p * p);

        let cap_len = context.dim(1)?;
        let context = context.to_dtype(dt)?;

        // Image patch embed.
        let img = self.img_in.forward(&patchify(&latent.to_dtype(dt)?, p)?)?; // [b, img_len, hidden]

        // Timestep embed → `t`; shared modulation `tvec = time_mod_proj(GELU(t))`.
        let t_sin = temb(timestep, cfg.timestep_embed_dim, &self.device)?.to_dtype(dt)?; // [b, 1, tdim]
        let t = self
            .time_embed_l2
            .forward(&self.time_embed_l1.forward(&t_sin)?.gelu()?)?; // [b, 1, hidden]
        let tvec = self.time_mod_proj.forward(&t.gelu()?)?; // [b, 1, 6·hidden]

        // Text fusion (12 layers → 1) then the text input projection.
        let ctx = self.text_fusion.forward(&context)?; // [b, cap, text_hidden]
        let ctx = self.txt_in_norm.forward(&ctx)?;
        let ctx = self
            .txt_in_l2
            .forward(&self.txt_in_l1.forward(&ctx)?.gelu()?)?; // [b, cap, hidden]

        // Fuse to the joint sequence and run the single-stream stack under the joint RoPE.
        let mut combined = Tensor::cat(&[&ctx, &img], 1)?; // [b, cap+img_len, hidden]
        let rope = RopeTables::build_t2i(
            cap_len,
            ht,
            wt,
            cfg.axes_dims_rope,
            cfg.rope_theta as f64,
            &self.device,
        )?;
        let (rcos, rsin) = rope.joint();
        for blk in &self.blocks {
            combined = blk.forward(&combined, &tvec, &rcos, &rsin)?;
        }

        // Continuous-AdaLN output (SimpleModulation on `t`), then slice the image tokens + unpatchify.
        let out = self.final_layer(&combined, &t)?; // [b, cap+img_len, in_channels]
        let img_out = out.narrow(1, cap_len, img_len)?;
        unpatchify(&img_out, ht, wt, p, latent_ch)
    }

    /// Reference `LastLayer`: `SimpleModulation(t) = t + scale_shift_table` → `(scale, shift)`;
    /// `Linear((1+scale)·RMSNorm(x) + shift)`.
    fn final_layer(&self, x: &Tensor, t: &Tensor) -> Result<Tensor> {
        let m = t.broadcast_add(&self.final_sstable)?; // [b, 2, hidden] (t broadcasts 1→2)
        let scale = m.narrow(1, 0, 1)?; // [b, 1, hidden]
        let shift = m.narrow(1, 1, 1)?;
        let normed = self
            .final_norm
            .forward(x)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        self.final_linear.forward(&normed)
    }
}

/// Reference `temb`: `freqs = exp(−ln(1e4)·arange(half)/half)`, `args = (timestep·1e3)·freqs`,
/// `concat([cos, sin], −1)` (cos-first). `timestep`: `[b]` → `[b, 1, dim]` (a per-sample vector that
/// broadcasts over the sequence). Built in f32 (the reference upcasts).
///
/// `pub(crate)` so the trainable DiT ([`crate::train_dit`]) shares the exact embedding (parity).
pub(crate) fn temb(timestep: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let neg_ln = -(10000f64.ln()) as f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (neg_ln * i as f32 / half as f32).exp())
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), device)?; // [1, half]
    let b = timestep.dim(0)?;
    let scaled = (timestep.to_dtype(DType::F32)?.reshape((b, 1, 1))? * 1000.0)?; // [b, 1, 1]
    let args = scaled.broadcast_mul(&freqs.reshape((1, 1, half))?)?; // [b, 1, half]
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1) // [b, 1, dim]
}

/// Reference `rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`: `[b, C, H, W] →
/// [b, (H/p)(W/p), C·p·p]` with **channel-major** patch flattening (NOT boogu's `(ph pw c)`).
///
/// `pub(crate)` so the trainable DiT ([`crate::train_dit`]) patchifies identically.
pub(crate) fn patchify(latent: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c, h, w) = latent.dims4()?;
    let (ht, wt) = (h / p, w / p);
    let x = latent.reshape((b, c, ht, p, wt, p))?; // b, c, ht, ph, wt, pw
    let x = x.permute((0, 2, 4, 1, 3, 5))?; // b, ht, wt, c, ph, pw
    x.contiguous()?.reshape((b, ht * wt, c * p * p))
}

/// Inverse of [`patchify`] (`"b (h w) (c ph pw) -> b c (h ph) (w pw)"`): `[b, (h)(w), C·p·p] →
/// [b, C, h·p, w·p]`.
///
/// `pub(crate)` so the trainable DiT ([`crate::train_dit`]) unpatchifies identically.
pub(crate) fn unpatchify(
    tokens: &Tensor,
    ht: usize,
    wt: usize,
    p: usize,
    c: usize,
) -> Result<Tensor> {
    let b = tokens.dim(0)?;
    let x = tokens.contiguous()?.reshape((b, ht, wt, c, p, p))?; // b, ht, wt, c, ph, pw
    let x = x.permute((0, 3, 1, 4, 2, 5))?; // b, c, ht, ph, wt, pw
    x.contiguous()?.reshape((b, c, ht * p, wt * p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn patchify_roundtrips_channel_major() {
        let dev = Device::Cpu;
        // [1, 4, 4, 6] with p=2 → 2×3 grid, 4·2·2 = 16 packed channels.
        let x = Tensor::arange(0f32, (4 * 4 * 6) as f32, &dev)
            .unwrap()
            .reshape((1, 4, 4, 6))
            .unwrap();
        let packed = patchify(&x, 2).unwrap();
        assert_eq!(packed.dims(), &[1, 2 * 3, 4 * 2 * 2]);
        let back = unpatchify(&packed, 2, 3, 2, 4).unwrap();
        assert_eq!(back.dims(), x.dims());
        let a = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "patchify∘unpatchify must be the identity");
    }

    #[test]
    fn temb_is_cos_first_and_scaled() {
        let dev = Device::Cpu;
        // t = 0 → all angles 0 → cos half = 1, sin half = 0.
        let t = Tensor::from_vec(vec![0f32], (1,), &dev).unwrap();
        let e = temb(&t, 8, &dev).unwrap();
        assert_eq!(e.dims(), &[1, 1, 8]);
        let v = e.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v[..4].iter().all(|&x| (x - 1.0).abs() < 1e-6),
            "cos-first half = 1 at t=0"
        );
        assert!(
            v[4..].iter().all(|&x| x.abs() < 1e-6),
            "sin half = 0 at t=0"
        );
    }
}
