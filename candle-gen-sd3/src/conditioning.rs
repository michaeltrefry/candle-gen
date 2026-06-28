//! SD3.5 triple text-encoder **aggregator** (sc-7876, epic 7982).
//!
//! SD3.5 conditions on three text encoders — CLIP-L, OpenCLIP bigG, and T5-XXL — combined into two
//! tensors fed to the MMDiT:
//!
//! - **pooled** `[B, 2048]` = `cat(CLIP-L pooled [768], CLIP-bigG pooled [1280])`. This is added to
//!   the timestep embedding (NOT to the token sequence) — it conditions the AdaLN modulation.
//! - **context** `[B, 333, 4096]` (at the SD3.5 defaults) = the token sequence the joint blocks
//!   attend over. Built in two steps, exactly as the public diffusers `StableDiffusion3Pipeline`:
//!   1. CLIP context = `cat(CLIP-L penultimate [77, 768], CLIP-bigG penultimate [77, 1280])` →
//!      `[77, 2048]`, then **zero-padded on the hidden axis** to `[77, 4096]`
//!      (`joint_attention_dim`). The pad is on the *trailing* hidden dims (diffusers
//!      `F.pad(clip, (0, t5_dim - clip_concat_dim))`).
//!   2. context = `cat([clip_padded [77, 4096], t5 [t5_len, 4096]], dim=seq)` → `[77 + t5_len, 4096]`.
//!
//! This module owns the **aggregation** — the parity-critical concat/pad/order that the spike
//! flagged. The actual CLIP/T5 forward (loading the encoders, penultimate-layer extraction, EOS
//! pooling) is wired in C2's pipeline; keeping the aggregator a pure tensor transform lets the
//! ordering be unit-tested on CPU with synthetic encoder outputs (no weights/GPU needed), the same
//! correctness bar epic 7841 used.

#[cfg(test)]
use candle_gen::candle_core::IndexOp;
use candle_gen::candle_core::{DType, Device, Result, Tensor, D};

use crate::config::Sd3Config;

/// The raw per-encoder outputs the aggregator combines. Produced by the encoders in C2; here they
/// are the inputs to the pure aggregation so the ordering is testable in isolation.
///
/// All tensors carry a leading batch axis `B`.
pub struct EncoderOutputs {
    /// CLIP-L penultimate hidden state `[B, clip_seq_len, clip_l_dim]` (768-wide).
    pub clip_l_hidden: Tensor,
    /// CLIP-bigG penultimate hidden state `[B, clip_seq_len, clip_g_dim]` (1280-wide).
    pub clip_g_hidden: Tensor,
    /// CLIP-L pooled/projected output `[B, clip_l_dim]` (768-wide).
    pub clip_l_pooled: Tensor,
    /// CLIP-bigG pooled/projected output `[B, clip_g_dim]` (1280-wide).
    pub clip_g_pooled: Tensor,
    /// T5-XXL encoder sequence `[B, t5_seq_len, t5_dim]` (4096-wide).
    pub t5_hidden: Tensor,
}

/// The two SD3.5 conditioning tensors fed to the MMDiT.
pub struct Sd3Conditioning {
    /// `[B, pooled_dim]` (2048) — added to the timestep embedding.
    pub pooled: Tensor,
    /// `[B, context_seq_len, joint_attention_dim]` (333 × 4096 at defaults) — the joint token
    /// sequence.
    pub context: Tensor,
}

/// Build the SD3.5 pooled + context conditioning from the three encoders' outputs.
///
/// Order and padding match the public diffusers `StableDiffusion3Pipeline._get_clip_prompt_embeds`
/// + `encode_prompt`:
/// - pooled = `cat([clip_l_pooled, clip_g_pooled], dim=-1)`;
/// - clip_context = `cat([clip_l_hidden, clip_g_hidden], dim=-1)` then right-pad the hidden axis to
///   `joint_attention_dim` with zeros;
/// - context = `cat([clip_context, t5_hidden], dim=seq)`.
pub fn aggregate(cfg: &Sd3Config, enc: &EncoderOutputs) -> Result<Sd3Conditioning> {
    // ---- pooled [B, 2048] ----
    let pooled = Tensor::cat(&[&enc.clip_l_pooled, &enc.clip_g_pooled], D::Minus1)?;

    // ---- CLIP context [B, 77, 2048] -> zero-pad hidden axis to [B, 77, 4096] ----
    let clip_context = Tensor::cat(&[&enc.clip_l_hidden, &enc.clip_g_hidden], D::Minus1)?;
    // The concatenated CLIP width must be the configured `clip_concat_dim` (768 + 1280 = 2048); a
    // mismatch means a mis-shaped encoder output, caught here before the pad rather than producing a
    // silently wrong context.
    let clip_w = clip_context.dim(D::Minus1)?;
    if clip_w != cfg.clip_concat_dim {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "sd3 aggregator: concatenated CLIP context width {clip_w} != configured \
             clip_concat_dim {} (clip_l_dim {} + clip_g_dim {})",
            cfg.clip_concat_dim, cfg.clip_l_dim, cfg.clip_g_dim
        )));
    }
    let clip_padded = pad_hidden_to(&clip_context, cfg.joint_attention_dim)?;

    // ---- context = cat([clip_padded, t5], seq) -> [B, 333, 4096] ----
    let context = Tensor::cat(&[&clip_padded, &enc.t5_hidden], 1)?;

    Ok(Sd3Conditioning { pooled, context })
}

/// Right-pad the LAST (hidden) axis of `x` `[..., h]` to width `target` with zeros (`F.pad(x, (0,
/// target - h))`). Errors if `x` is already wider than `target`.
fn pad_hidden_to(x: &Tensor, target: usize) -> Result<Tensor> {
    let h = x.dim(D::Minus1)?;
    if h == target {
        return Ok(x.clone());
    }
    if h > target {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "sd3 aggregator: clip context hidden {h} exceeds joint_attention_dim {target}"
        )));
    }
    let mut shape = x.dims().to_vec();
    *shape.last_mut().unwrap() = target - h;
    let pad = Tensor::zeros(shape, x.dtype(), x.device())?;
    Tensor::cat(&[x, &pad], D::Minus1)
}

/// Build zeroed encoder outputs at the config's shapes for a given batch (the "empty/unconditioned"
/// path and a test fixture). C2's CFG uses this to assemble the negative branch when a T5/CLIP empty
/// encode is degenerate; here it doubles as the structural-test fixture.
pub fn zeroed_outputs(
    cfg: &Sd3Config,
    batch: usize,
    dtype: DType,
    device: &Device,
) -> Result<EncoderOutputs> {
    Ok(EncoderOutputs {
        clip_l_hidden: Tensor::zeros((batch, cfg.clip_seq_len, cfg.clip_l_dim), dtype, device)?,
        clip_g_hidden: Tensor::zeros((batch, cfg.clip_seq_len, cfg.clip_g_dim), dtype, device)?,
        clip_l_pooled: Tensor::zeros((batch, cfg.clip_l_dim), dtype, device)?,
        clip_g_pooled: Tensor::zeros((batch, cfg.clip_g_dim), dtype, device)?,
        t5_hidden: Tensor::zeros((batch, cfg.t5_seq_len, cfg.t5_dim), dtype, device)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn fixture(cfg: &Sd3Config, batch: usize) -> EncoderOutputs {
        let dev = Device::Cpu;
        // Distinctive fill values per source so the concat ORDER is observable in the output.
        EncoderOutputs {
            clip_l_hidden: Tensor::full(1f32, (batch, cfg.clip_seq_len, cfg.clip_l_dim), &dev)
                .unwrap(),
            clip_g_hidden: Tensor::full(2f32, (batch, cfg.clip_seq_len, cfg.clip_g_dim), &dev)
                .unwrap(),
            clip_l_pooled: Tensor::full(3f32, (batch, cfg.clip_l_dim), &dev).unwrap(),
            clip_g_pooled: Tensor::full(4f32, (batch, cfg.clip_g_dim), &dev).unwrap(),
            t5_hidden: Tensor::full(5f32, (batch, cfg.t5_seq_len, cfg.t5_dim), &dev).unwrap(),
        }
    }

    #[test]
    fn aggregate_shapes_match_sd35_defaults() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        // pooled = 768 + 1280 = 2048.
        assert_eq!(out.pooled.dims(), &[1, 2048]);
        // context = (77 + 256) x 4096 = 333 x 4096.
        assert_eq!(out.context.dims(), &[1, 333, 4096]);
    }

    #[test]
    fn pooled_concat_order_is_l_then_g() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        let v = out.pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // First 768 from CLIP-L (filled 3), next 1280 from bigG (filled 4).
        assert!(
            v[..768].iter().all(|&x| x == 3.0),
            "CLIP-L pooled goes first"
        );
        assert!(
            v[768..2048].iter().all(|&x| x == 4.0),
            "bigG pooled goes second"
        );
    }

    #[test]
    fn context_layout_is_clip_padded_then_t5() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        // Token 0 (a CLIP token): hidden = [CLIP-L 768 = 1, bigG 1280 = 2, zero-pad 2048 = 0].
        let tok0 = out.context.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        assert!(tok0[..768].iter().all(|&x| x == 1.0), "clip-l region");
        assert!(tok0[768..2048].iter().all(|&x| x == 2.0), "bigg region");
        assert!(
            tok0[2048..4096].iter().all(|&x| x == 0.0),
            "zero-pad region"
        );
        // Token 77 (the first T5 token): all 5 across the full 4096 width.
        let tok_t5 = out.context.i((0, 77)).unwrap().to_vec1::<f32>().unwrap();
        assert!(
            tok_t5.iter().all(|&x| x == 5.0),
            "t5 region is full-width 4096"
        );
    }

    #[test]
    fn t5_length_drives_context_seq() {
        let mut cfg = Sd3Config::large();
        cfg.t5_seq_len = 512;
        let enc = fixture(&cfg, 2);
        let out = aggregate(&cfg, &enc).unwrap();
        assert_eq!(out.context.dims(), &[2, 77 + 512, 4096]);
        assert_eq!(out.pooled.dims(), &[2, 2048]);
    }

    #[test]
    fn aggregate_rejects_misshaped_clip_width() {
        // A config whose clip_concat_dim disagrees with clip_l_dim + clip_g_dim trips the guard.
        let mut cfg = Sd3Config::large();
        cfg.clip_concat_dim = 999; // != 768 + 1280
        let enc = fixture(&cfg, 1);
        assert!(aggregate(&cfg, &enc).is_err());
    }

    #[test]
    fn zeroed_outputs_aggregate_to_correct_shape() {
        let cfg = Sd3Config::large();
        let enc = zeroed_outputs(&cfg, 1, DType::F32, &Device::Cpu).unwrap();
        let out = aggregate(&cfg, &enc).unwrap();
        assert_eq!(out.context.dims(), &[1, 333, 4096]);
        assert_eq!(out.pooled.dims(), &[1, 2048]);
    }
}
