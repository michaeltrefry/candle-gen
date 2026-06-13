//! Qwen-Image configuration, ported from `mlx-gen-qwen-image`. Dims are the production
//! `qwen_image` values (T2I).

/// Registry id for Qwen-Image txt2img.
pub const MODEL_ID: &str = "qwen_image";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
/// Fork-verbatim default; production callers pass ~20–50 (Qwen-Image T2I is not distilled — the
/// distilled Lightning path is deferred).
pub const DEFAULT_STEPS: u32 = 4;
/// True-CFG guidance scale default.
pub const DEFAULT_GUIDANCE: f32 = 4.0;
/// Single space — the negative prompt used when a CFG request omits one.
pub const NEGATIVE_FALLBACK: &str = " ";

/// Both image dims must be multiples of 16 (VAE /8 then the DiT 2×2 patch).
pub const SIZE_MULTIPLE: u32 = 16;

/// VAE latent channels.
pub const LATENT_CHANNELS: usize = 16;
/// DiT 2×2 patch — the packed token feature width is `LATENT_CHANNELS * PATCH² = 64`.
pub const PATCH: usize = 2;

/// The Qwen-Image dual-stream MMDiT dims.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformerConfig {
    /// Packed latent channels entering the transformer (`LATENT_CHANNELS * PATCH² = 64`).
    pub in_channels: usize,
    /// VAE latent channels leaving the transformer (16).
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Text embed width from Qwen2.5-VL (3584).
    pub joint_attention_dim: usize,
    /// Sinusoidal timestep-embedding width (256).
    pub timestep_channels: usize,
    /// 3-axis (frame, height, width) RoPE.
    pub axes_dim: [usize; 3],
    pub rope_theta: f32,
    pub eps: f64,
}

impl TransformerConfig {
    pub fn qwen_image() -> Self {
        Self {
            in_channels: LATENT_CHANNELS * PATCH * PATCH,
            out_channels: LATENT_CHANNELS,
            num_layers: 60,
            num_heads: 24,
            head_dim: 128,
            joint_attention_dim: 3584,
            timestep_channels: 256,
            axes_dim: [16, 56, 56],
            rope_theta: 10_000.0,
            eps: 1e-6,
        }
    }

    /// `num_heads * head_dim` — the model width (3072).
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// The Qwen2.5-VL language-model (text path) dims.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    /// Tokens dropped off the front of the encoded sequence (the QwenImage system-prompt prefix).
    pub prompt_drop_idx: usize,
    pub max_length: usize,
    pub pad_token_id: i32,
}

impl TextEncoderConfig {
    pub fn qwen_image() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            n_layers: 28,
            n_heads: 28,
            n_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 18944,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            prompt_drop_idx: 34,
            max_length: 1058,
            pad_token_id: 151643,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_match_fork() {
        let t = TransformerConfig::qwen_image();
        assert_eq!(t.num_layers, 60);
        assert_eq!(t.inner_dim(), 3072);
        assert_eq!(t.in_channels, 64);
        assert_eq!(t.out_channels, 16);
        assert_eq!(t.joint_attention_dim, 3584);
        assert_eq!(t.axes_dim.iter().sum::<usize>(), t.head_dim);

        let e = TextEncoderConfig::qwen_image();
        assert_eq!(e.hidden_size, 3584);
        assert_eq!(e.n_layers, 28);
        assert_eq!(e.n_heads / e.n_kv_heads, 7);
        assert_eq!(e.hidden_size, t.joint_attention_dim);
    }
}
