//! FLUX.2 configuration, ported from `mlx-gen-flux2`'s `config.rs` (itself lifted from the frozen
//! mflux fork). Two variants are wired here: **klein-9b** (distilled, Qwen3 TE, 4-step CFG-free) and
//! **dev** (the 32B flagship — Mistral TE + a 48/48 DiT, guidance-distilled). The struct is kept
//! dimension-parametric so the two share every module; only a handful of fields differ.

/// Registry id for FLUX.2-klein-9B txt2img.
pub const FLUX2_KLEIN_9B_ID: &str = "flux2_klein_9b";
/// Registry id for FLUX.2-dev txt2img (the undistilled 32B flagship: embedded guidance + more steps).
pub const FLUX2_DEV_ID: &str = "flux2_dev";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
/// Distilled klein default — the fork generates in 4 steps.
pub const DEFAULT_STEPS: u32 = 4;
/// Distilled klein runs at guidance 1.0 (no CFG); >1.0 enables a classifier-free negative pass.
pub const DEFAULT_GUIDANCE: f32 = 1.0;

/// FLUX.2-dev is guidance-distilled (embedded scalar, the FLUX.1-dev pattern): ~28 steps (24–50) at
/// guidance ~4.0 — NOT true CFG with a negative prompt. (BFL reference call: `guidance_scale=4`,
/// `num_inference_steps=50`, with 28 a good speed/quality trade-off.)
pub const DEFAULT_STEPS_DEV: u32 = 28;
pub const DEFAULT_GUIDANCE_DEV: f32 = 4.0;

/// Both image dims must be multiples of 16 (VAE /8 then the DiT's 2×2 patch) for a clean pack.
pub const SIZE_MULTIPLE: u32 = 16;

/// A pre-quantized-snapshot manifest (the candle twin of mlx sc-5917): the `quantization` block
/// (`{ "bits", "group_size" }`) an install-time convert job writes into a component's `config.json`.
/// Its presence on disk flips the matching loader from the dense path to building each Linear (and
/// the TE token embedding) directly from packed parts — so no dense bf16 weight is ever materialized,
/// which is what keeps the dev load-time memory floor under the ceiling (60 GB DiT + 45 GB TE bf16
/// would peak ~105 GB dense). Consume-side mirror of `mlx_gen_flux2::config::Flux2Quant`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flux2Quant {
    pub bits: i32,
    pub group_size: i32,
}

/// The FLUX.2 txt2img variants this crate registers. `klein_9b` is distilled (4-step, CFG-free);
/// `dev` is the guidance-distilled 32B flagship (Mistral TE + 48/48 DiT, embedded guidance ~4 over
/// ~28 steps). Edit / ControlNet / klein weight-variants are tracked separately (epic 6564 stories
/// 3–4) and are not in this enum yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flux2Variant {
    /// FLUX.2-klein-9b, distilled, txt2img.
    Klein9b,
    /// FLUX.2-dev, guidance-distilled txt2img: larger MMDiT (48 single blocks, 48 heads, joint 15360)
    /// + the Mistral text encoder; embedded guidance ~4 over ~28 steps.
    Dev,
}

impl Flux2Variant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Klein9b => FLUX2_KLEIN_9B_ID,
            Self::Dev => FLUX2_DEV_ID,
        }
    }

    /// The dimension-parametric model config for this variant.
    pub fn config(self) -> Flux2Config {
        match self {
            Self::Klein9b => Flux2Config::klein_9b(),
            Self::Dev => Flux2Config::dev(),
        }
    }

    /// Default denoise steps. Distilled klein = 4; guidance-distilled dev ≈ 28 (range 24–50).
    pub fn default_steps(self) -> u32 {
        match self {
            Self::Klein9b => DEFAULT_STEPS,
            Self::Dev => DEFAULT_STEPS_DEV,
        }
    }

    /// Default guidance. klein runs CFG-free (1.0); dev uses embedded guidance ~4.0.
    pub fn default_guidance(self) -> f32 {
        match self {
            Self::Klein9b => DEFAULT_GUIDANCE,
            Self::Dev => DEFAULT_GUIDANCE_DEV,
        }
    }

    /// Whether the guidance scale is consumed as an **embedded scalar** fed into the transformer's
    /// guidance embedder (the guidance-distilled dev, FLUX.1-dev pattern) rather than as a true-CFG
    /// dual-forward over a negative prompt. dev = `true` (single forward, no negative pass); klein =
    /// `false` (distilled, CFG-free at guidance 1.0; >1 runs a classifier-free negative pass).
    pub fn uses_embedded_guidance(self) -> bool {
        matches!(self, Self::Dev)
    }

    /// The dev variant (Mistral TE + 48/48 DiT + the embedded-guidance forward).
    pub fn is_dev(self) -> bool {
        matches!(self, Self::Dev)
    }
}

/// Dimension-parametric FLUX.2 model dimensions. Field values come from the fork's `ModelConfig`
/// + the FLUX.2 module constructors; klein-9b and dev are both expressed here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flux2Config {
    // --- MMDiT transformer ---
    /// Double (joint img+txt) blocks. 9b: 8, dev: 8.
    pub num_double_layers: usize,
    /// Single (fused parallel attention+SwiGLU) blocks. 9b: 24, dev: 48.
    pub num_single_layers: usize,
    /// Attention heads. 9b: 32, dev: 48.
    pub num_heads: usize,
    /// Per-head dim (constant across variants). `inner_dim = num_heads * head_dim` (9b: 4096, dev: 6144).
    pub head_dim: usize,
    /// Latent channels entering/leaving the transformer = `num_latent_channels * 4` (2×2 patch).
    pub in_channels: usize,
    pub out_channels: usize,
    /// Text-embedding width entering the joint blocks = `3 * te_hidden_size` (concat of 3 TE
    /// hidden-state layers). 9b: 12288, dev: 15360.
    pub joint_attention_dim: usize,
    /// Single-block SwiGLU expansion ratio (`mlp_hidden = mlp_ratio * inner_dim`). Both: 3.0.
    pub mlp_ratio: f32,
    /// Sinusoidal timestep-embedding width feeding `time_guidance_embed.linear_1` (both: 256).
    pub timestep_channels: usize,

    // --- 4-axis RoPE over ids (t, h, w, layer) ---
    pub axes_dim: [usize; 4],
    pub rope_theta: f32,

    // --- decoder-LM text encoder (Qwen3 for klein, Mistral for dev) ---
    pub te_hidden_size: usize,
    pub te_intermediate_size: usize,
    pub te_n_layers: usize,
    pub te_n_heads: usize,
    pub te_n_kv_heads: usize,
    pub te_head_dim: usize,
    pub te_rope_theta: f32,
    pub te_rms_norm_eps: f64,
    /// Per-head q/k RMSNorm before RoPE — Qwen3 (klein) has it, Mistral (dev) does not.
    pub te_qk_norm: bool,
    /// Token-embedding vocab size. Qwen3 (klein): 151936; Mistral (dev): 131072.
    pub te_vocab_size: usize,
    /// The on-disk weight prefix for the TE's decoder tree. klein's Qwen3 lives under `model.*`; dev's
    /// Mistral is the language tower of a `Mistral3ForConditionalGeneration`, under `language_model.model.*`.
    pub te_prefix: &'static str,
    /// Hidden-state indices (index 0 = embeddings, index k = output of layer k-1) concatenated into
    /// `prompt_embeds`. klein: (9, 18, 27) → 3·4096 = 12288; dev: (10, 20, 30) → 3·5120 = 15360.
    pub te_out_layers: [usize; 3],
    pub max_sequence_length: usize,

    // --- VAE / latent geometry (identical across variants) ---
    pub num_latent_channels: usize,
    pub vae_scale_factor: usize,
}

impl Flux2Config {
    /// FLUX.2-klein-9b.
    pub fn klein_9b() -> Self {
        Self {
            num_double_layers: 8,
            num_single_layers: 24,
            num_heads: 32,
            head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            joint_attention_dim: 12288,
            mlp_ratio: 3.0,
            timestep_channels: 256,
            axes_dim: [32, 32, 32, 32],
            rope_theta: 2000.0,
            te_hidden_size: 4096,
            te_intermediate_size: 12288,
            te_n_layers: 36,
            te_n_heads: 32,
            te_n_kv_heads: 8,
            te_head_dim: 128,
            te_rope_theta: 1_000_000.0,
            te_rms_norm_eps: 1e-6,
            te_qk_norm: true,
            te_vocab_size: 151936,
            te_prefix: "model",
            te_out_layers: [9, 18, 27],
            max_sequence_length: 512,
            num_latent_channels: 32,
            vae_scale_factor: 8,
        }
    }

    /// FLUX.2-dev — the same MMDiT arch as klein, scaled up: 48 single blocks, 48 heads (inner 6144),
    /// `joint_attention_dim` 15360 (= 3 × the Mistral TE hidden 5120). The VAE + RoPE + patch geometry
    /// are identical to klein; only the block/head counts, the joint width, and the text-encoder dims
    /// change. The TE is the **Mistral** language tower of a `Mistral3ForConditionalGeneration` (under
    /// `language_model.model.*`, no per-head q/k-norm, θ=1e9, eps 1e-5). Values from the dev
    /// `transformer/config.json` + `text_encoder/config.json`.
    pub fn dev() -> Self {
        Self {
            num_double_layers: 8,
            num_single_layers: 48,
            num_heads: 48,
            head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            joint_attention_dim: 15360,
            mlp_ratio: 3.0,
            timestep_channels: 256,
            axes_dim: [32, 32, 32, 32],
            rope_theta: 2000.0,
            te_hidden_size: 5120,
            te_intermediate_size: 32768,
            te_n_layers: 40,
            te_n_heads: 32,
            te_n_kv_heads: 8,
            te_head_dim: 128,
            te_rope_theta: 1_000_000_000.0,
            te_rms_norm_eps: 1e-5,
            te_qk_norm: false,
            te_vocab_size: 131072,
            te_prefix: "language_model.model",
            te_out_layers: [10, 20, 30],
            max_sequence_length: 512,
            num_latent_channels: 32,
            vae_scale_factor: 8,
        }
    }

    /// `num_heads * head_dim` — the transformer inner width (9b: 4096, dev: 6144).
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Single-block SwiGLU hidden width (`mlp_ratio * inner_dim`, 9b: 12288, dev: 18432).
    pub fn single_mlp_hidden(&self) -> usize {
        (self.mlp_ratio * self.inner_dim() as f32) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klein_9b_dims_match_fork() {
        let c = Flux2Config::klein_9b();
        assert_eq!(c.num_double_layers, 8);
        assert_eq!(c.num_single_layers, 24);
        assert_eq!(c.num_heads, 32);
        assert_eq!(c.inner_dim(), 4096);
        assert_eq!(c.joint_attention_dim, 3 * c.te_hidden_size);
        assert_eq!(c.in_channels, c.num_latent_channels * 4);
        assert_eq!(c.single_mlp_hidden(), 12288);
        // RoPE axes sum to the head dim; each axis emits dim/2 freqs → cos/sin width head_dim/2.
        assert_eq!(c.axes_dim.iter().sum::<usize>(), c.head_dim);
        assert_eq!(c.te_head_dim * c.te_n_heads, c.te_hidden_size);
        // klein's Qwen3 TE carries per-head q/k norm and lives under `model.*`.
        assert!(c.te_qk_norm);
        assert_eq!(c.te_prefix, "model");
    }

    #[test]
    fn dev_dims_match_reference() {
        let c = Flux2Config::dev();
        assert_eq!(c.num_double_layers, 8);
        assert_eq!(c.num_single_layers, 48);
        assert_eq!(c.num_heads, 48);
        assert_eq!(c.inner_dim(), 6144);
        // joint_attention_dim = 3 × the Mistral TE hidden (5120) = 15360.
        assert_eq!(c.joint_attention_dim, 3 * c.te_hidden_size);
        assert_eq!(c.joint_attention_dim, 15360);
        assert_eq!(c.in_channels, c.num_latent_channels * 4);
        assert_eq!(c.single_mlp_hidden(), 18432);
        assert_eq!(c.axes_dim.iter().sum::<usize>(), c.head_dim);
        // Mistral decouples head_dim from hidden/heads: q/k/v inner = n_heads·head_dim = 4096, but the
        // residual stream (hidden_size) is 5120 — so q_proj maps 5120→4096 and o_proj 4096→5120.
        assert_eq!(c.te_n_heads * c.te_head_dim, 4096);
        assert_ne!(c.te_n_heads * c.te_head_dim, c.te_hidden_size);
        assert_eq!(c.te_hidden_size, 5120);
        assert_eq!(c.te_out_layers, [10, 20, 30]);
        // Mistral: no per-head q/k-norm, θ=1e9, eps 1e-5, language_model.model.* prefix.
        assert!(!c.te_qk_norm);
        assert_eq!(c.te_rope_theta, 1_000_000_000.0);
        assert_eq!(c.te_rms_norm_eps, 1e-5);
        assert_eq!(c.te_prefix, "language_model.model");
    }

    #[test]
    fn variant_identity_and_defaults() {
        assert_eq!(Flux2Variant::Klein9b.id(), FLUX2_KLEIN_9B_ID);
        assert_eq!(Flux2Variant::Dev.id(), FLUX2_DEV_ID);
        // klein: 4-step distilled, CFG-free, no embedded guidance.
        assert_eq!(Flux2Variant::Klein9b.default_steps(), 4);
        assert_eq!(Flux2Variant::Klein9b.default_guidance(), 1.0);
        assert!(!Flux2Variant::Klein9b.uses_embedded_guidance());
        assert!(!Flux2Variant::Klein9b.is_dev());
        // dev: ~28-step guidance-distilled, embedded guidance ~4.
        assert_eq!(Flux2Variant::Dev.default_steps(), 28);
        assert_eq!(Flux2Variant::Dev.default_guidance(), 4.0);
        assert!(Flux2Variant::Dev.uses_embedded_guidance());
        assert!(Flux2Variant::Dev.is_dev());
        assert_eq!(Flux2Variant::Dev.config().num_single_layers, 48);
        assert_eq!(Flux2Variant::Klein9b.config().num_single_layers, 24);
    }
}
