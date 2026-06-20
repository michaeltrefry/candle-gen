//! Ideogram 4.0 configuration — constants read directly from the official
//! `ideogram-ai/ideogram-4-fp8` checkpoint configs (sc-5984), ported from `mlx-gen-ideogram`'s
//! `config.rs` to candle (`usize`/`f64` idioms). The Rust modules and the weight provisioning agree
//! on every dimension.
//!
//! Pipeline (`model_index.json` = `Ideogram4Pipeline`):
//!   * `FlowMatchEulerDiscreteScheduler` → the resolution-aware [`crate::scheduler`] logit-normal
//!     schedule the real `Ideogram4Pipeline` actually samples with.
//!   * `Qwen3VLModel` text encoder (text path only) → [`crate::text_encoder`].
//!   * `Ideogram4Transformer2DModel` **transformer** + **unconditional_transformer**
//!     (two full DiTs — asymmetric CFG for the quality variant).
//!   * `AutoencoderKLFlux2` VAE — the FLUX.2 VAE, reused from `candle-gen-flux2`.

/// Registry id for the quality variant (asymmetric two-DiT CFG, 48-step default).
pub const MODEL_ID: &str = "ideogram_4";

/// Registry id for the few-step **turbo** variant — the CFG-free single-DiT path driven by the
/// ostris TurboTime LoRA (mlx-gen #488). Same base weights as [`MODEL_ID`]; the snapshot adds the
/// bundled LoRA ([`TURBO_LORA_FILE`]) and needs no unconditional DiT.
pub const MODEL_ID_TURBO: &str = "ideogram_4_turbo";

/// HF repo for the gated source weights (fp8 reference release). The candle lane needs bf16 (or an
/// fp8→bf16 dequant at load) — the MLX-quantized turnkey `SceneWorks/ideogram-4-mlx` is not
/// candle-readable.
pub const IDEOGRAM_4_FP8_REPO: &str = "ideogram-ai/ideogram-4-fp8";

/// Filename of the bundled TurboTime LoRA inside a turbo snapshot directory (sibling of
/// `transformer/`). The turbo loader installs it onto the conditional DiT at load.
pub const TURBO_LORA_FILE: &str = "turbo_lora.safetensors";

/// TurboTime ships **no** alpha/config tensor → the ai-toolkit default scale of 1.0 (mlx-gen #488).
pub const TURBO_LORA_SCALE: f32 = 1.0;

/// Turbo default step count (mlx-gen #488 spike: 1024²/8-step quality ≥ the 128-step 2-DiT render).
pub const DEFAULT_TURBO_STEPS: u32 = 8;

// ── Defaults / limits ────────────────────────────────────────────────────────────────────
pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
/// Native resolution range: 256–2048, multiples of 16, aspect up to 6:1.
pub const RES_MIN: u32 = 256;
pub const RES_MAX: u32 = 2048;
pub const RES_MULTIPLE: u32 = 16;
/// Both image dims must be multiples of 16 (VAE /8 then the DiT 2×2 patch). Alias of
/// [`RES_MULTIPLE`] for parity with the other candle-gen crates' `SIZE_MULTIPLE`.
pub const SIZE_MULTIPLE: u32 = 16;

/// Quality default step count — the `V4_QUALITY_48` preset (the reference `__call__` default is 128;
/// 48 renders cleanly at a fraction of the cost over two DiTs). (sc-5988)
pub const DEFAULT_STEPS: u32 = 48;
/// Reference `__call__` default `guidance_scale=7.0` (asymmetric CFG: `v = g·cond + (1−g)·uncond`).
pub const DEFAULT_GUIDANCE: f32 = 7.0;
/// Ideogram 4 reference scheduler mean (`mu`) — the logit-normal schedule's `known_mean`. The
/// ComfyUI "Ideogram 4 Scheduler" node defaults to `mu=0.0, std=1.75`. See [`crate::scheduler`].
pub const DEFAULT_MU: f64 = 0.0;

/// Default img2img (Remix) strength when an edit `Reference` carries no explicit strength (slice 3).
pub const DEFAULT_IMG2IMG_STRENGTH: f32 = 0.6;
/// Default mask-inpaint strength when an edit supplies a `Mask` without an explicit strength.
pub const DEFAULT_INPAINT_STRENGTH: f32 = 0.85;

/// Max text tokens the model accepts (Qwen3-VL context budget used by Ideogram).
pub const MAX_TEXT_TOKENS: usize = 2048;

/// VAE latent channels (32-ch `AutoencoderKLFlux2`).
pub const LATENT_CHANNELS: usize = 32;
/// DiT 2×2 patch — the packed token feature width is `LATENT_CHANNELS * PATCH² = 128`
/// (= [`Ideogram4DitConfig::in_channels`]).
pub const PATCH: usize = 2;
/// VAE spatial downscale (8×).
pub const AE_SCALE: usize = 8;

/// `Ideogram4Transformer2DModel` dims (transformer/config.json). Single-stream, 34 layers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ideogram4DitConfig {
    /// `num_heads * head_dim` — the model width (4608).
    pub emb_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// SwiGLU intermediate (12288).
    pub mlp_dim: usize,
    /// AdaLN modulation width (512).
    pub adaln_dim: usize,
    /// Input latent channels (`LATENT_CHANNELS * PATCH² = 128`).
    pub in_channels: usize,
    /// Concatenated TE feature width (`13 * 4096 = 53248`).
    pub llm_features_dim: usize,
    /// 3-axis (t, h, w) interleaved MRoPE frequency grouping.
    pub mrope_section: [usize; 3],
    pub rope_theta: f32,
    pub norm_eps: f64,
}

impl Ideogram4DitConfig {
    pub const fn v4() -> Self {
        Self {
            emb_dim: 4608,
            num_layers: 34,
            num_heads: 18,
            head_dim: 256,
            mlp_dim: 12288,
            adaln_dim: 512,
            in_channels: 128,
            llm_features_dim: 53248,
            mrope_section: [24, 20, 20],
            rope_theta: 5_000_000.0,
            norm_eps: 1e-5,
        }
    }
}

/// `Qwen3VLModel` text stack (text_encoder/config.json `text_config`). Text path only — the vision
/// tower is unused for text-to-image. Ideogram concatenates the hidden states from
/// [`EXTRACTED_LAYERS`] (13 of them) → `13 * 4096 = 53248` features fed to the DiT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ideogram4TextEncoderConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    /// Text-only path uses plain 1-D RoPE (MRoPE sections all index the same sequential text
    /// position when there are no image tokens), so only `rope_theta` matters here.
    pub rope_theta: f32,
    pub vocab_size: usize,
}

impl Ideogram4TextEncoderConfig {
    pub const fn qwen3_vl_8b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            vocab_size: 151936,
        }
    }
}

/// The 13 Qwen3-VL hidden-state layers Ideogram concatenates: the OUTPUTS of layers
/// `(0, 3, 6, …, 33, 35)` (upstream `_get_qwen3_vl_embeddings` `captured[layer_idx]` — the state
/// right *after* running layer `idx`, NOT HF `output_hidden_states` indexing).
/// `len * hidden_size = 13 * 4096 = 53248 = Ideogram4DitConfig.llm_features_dim`.
pub const EXTRACTED_LAYERS: [usize; 13] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35];

/// Qwen `<|endoftext|>` pad token id.
pub const PAD_TOKEN_ID: i32 = 151643;

const _: () = assert!(
    EXTRACTED_LAYERS.len() * Ideogram4TextEncoderConfig::qwen3_vl_8b().hidden_size
        == Ideogram4DitConfig::v4().llm_features_dim
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_are_internally_consistent() {
        let d = Ideogram4DitConfig::v4();
        assert_eq!(d.emb_dim, d.num_heads * d.head_dim);
        assert_eq!(d.in_channels, LATENT_CHANNELS * PATCH * PATCH);
        let e = Ideogram4TextEncoderConfig::qwen3_vl_8b();
        assert_eq!(EXTRACTED_LAYERS.len() * e.hidden_size, d.llm_features_dim);
        assert_eq!(e.num_heads % e.num_kv_heads, 0);
        assert_eq!(*EXTRACTED_LAYERS.iter().max().unwrap(), 35);
        assert!(*EXTRACTED_LAYERS.iter().max().unwrap() < e.num_layers);
    }
}
