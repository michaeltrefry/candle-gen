//! Chroma family configuration — the candle (Windows/CUDA) port of `mlx-gen-chroma`'s `config.rs`,
//! lifted from the diffusers `ChromaTransformer2DModel` / `ChromaPipeline` reference.
//!
//! Chroma is a FLUX.1-schnell-derived DiT: same MMDiT skeleton (19 dual + 38 single blocks,
//! `inner_dim = 3072`, 24 heads × 128, FluxPosEmbed RoPE), but the FLUX `time_text_embed`
//! (timestep + guidance + pooled-CLIP) is replaced by a `distilled_guidance_layer` (Approximator)
//! that generates *all* per-block modulation, conditioning is **T5-XXL only** (no CLIP / no pooled),
//! and generation uses **true CFG** (a real negative prompt), not FLUX's distilled guidance.
//!
//! The candle deviations from the mlx descriptor are the two backend-correct ones the SDXL / FLUX /
//! Z-Image candle slices already make: `backend = "candle"` and `mac_only = false`. Like those
//! slices this v1 wires **txt2img only** — LoRA/LoKr and Q4/Q8 are NOT advertised (and are rejected
//! at load rather than silently dropped); ControlNet / IP-Adapter are later ports.

use candle_gen::gen_core::{Capabilities, Modality, ModelDescriptor};

pub const CHROMA1_HD_ID: &str = "chroma1_hd";
pub const CHROMA1_BASE_ID: &str = "chroma1_base";
pub const CHROMA1_FLASH_ID: &str = "chroma1_flash";

/// The base flow-match sampler name (matches the mlx descriptor's advertised sampler).
pub const DEFAULT_SAMPLER: &str = "flow_match";

/// Chroma works in the VAE's /8 latent and the DiT packs that 2×2, so both image dims must be
/// multiples of **16** for a clean pack. Enforced in the provider's `validate`.
pub const SIZE_MULTIPLE: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaVariant {
    /// `lodestones/Chroma1-HD` — the high-detail full-CFG model.
    Hd,
    /// `lodestones/Chroma1-Base` — the base full-CFG model (beta-spaced sigmas).
    Base,
    /// `lodestones/Chroma1-Flash` — the few-step distilled model.
    Flash,
}

impl ChromaVariant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Hd => CHROMA1_HD_ID,
            Self::Base => CHROMA1_BASE_ID,
            Self::Flash => CHROMA1_FLASH_ID,
        }
    }

    /// Default denoise steps. HD/Base run full CFG; Flash is a few-step distilled checkpoint.
    pub fn default_steps(self) -> u32 {
        match self {
            Self::Hd | Self::Base => 28,
            Self::Flash => 8,
        }
    }

    /// Default true-CFG scale. Flash is distilled toward CFG≈1 (single forward).
    pub fn default_true_cfg(self) -> f32 {
        match self {
            Self::Hd | Self::Base => 4.0,
            Self::Flash => 1.0,
        }
    }

    /// Static flow-match time `shift` (diffusers `FlowMatchEulerDiscreteScheduler`,
    /// `use_dynamic_shifting=false`): `σ' = shift·σ / (1 + (shift-1)·σ)`. HD's `scheduler_config.json`
    /// pins `shift=3.0`; Flash pins `1.0`. **Base** ships `use_beta_sigmas=true` (a beta-spaced
    /// schedule, not a shifted linspace), so its shift is inert (`1.0`).
    pub fn sigma_shift(self) -> f32 {
        match self {
            Self::Hd => 3.0,
            Self::Base | Self::Flash => 1.0,
        }
    }

    /// Base ships `use_beta_sigmas=true` — a beta-distribution sigma spacing (see [`crate::beta`])
    /// instead of the shifted linspace HD/Flash use.
    pub fn use_beta_sigmas(self) -> bool {
        matches!(self, Self::Base)
    }

    /// The candle descriptor for this variant — the txt2img surface sc-5484 actually wires. Chroma
    /// uses real classifier-free guidance with a true negative prompt (`supports_true_cfg` +
    /// `supports_negative_prompt`), and NO distilled guidance-scalar embedding
    /// (`supports_guidance = false`). LoRA/LoKr and quantization are deferred (the Python fallback's
    /// job until candle wires them), so they are not advertised and are rejected at load.
    pub fn descriptor(self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.id(),
            family: "chroma",
            backend: "candle",
            modality: Modality::Image,
            capabilities: Capabilities {
                supports_negative_prompt: true,
                supports_guidance: false,
                supports_true_cfg: true,
                // v1 = T2I only. ControlNet / IP-Adapter / img2img are later ports.
                conditioning: vec![],
                supports_lora: false,
                supports_lokr: false,
                // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123) plus the legacy
                // aliases (`flow_match` / `linear`), which fall back to euler / the native per-variant
                // schedule (N3) so a request the worker builds for either backend still validates.
                samplers: candle_gen::menu_with_aliases(
                    candle_gen::curated_sampler_names(),
                    &[DEFAULT_SAMPLER],
                ),
                schedulers: candle_gen::menu_with_aliases(
                    candle_gen::curated_scheduler_names(),
                    &["linear"],
                ),
                supported_guidance_methods: vec![],
                min_size: 256,
                max_size: 2048,
                max_count: 8,
                // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
                mac_only: false,
                supported_quants: &[],
                supports_kv_cache: false,
                // The static-shift / beta sigma schedule is applied inside the candle pipeline, so the
                // worker needs no sigma-shift loader hint (matches the candle FLUX/Z-Image slices).
                requires_sigma_shift: false,
            },
        }
    }
}

/// Static dims of `ChromaTransformer2DModel` (diffusers defaults — identical across the three
/// variants; only the weights and sampling profile differ).
#[derive(Clone, Copy, Debug)]
pub struct ChromaTransformerConfig {
    pub in_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub approximator_num_channels: usize,
    pub approximator_hidden_dim: usize,
    pub approximator_layers: usize,
}

impl Default for ChromaTransformerConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            num_layers: 19,
            num_single_layers: 38,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 4096,
            axes_dims_rope: [16, 56, 56],
            approximator_num_channels: 64,
            approximator_hidden_dim: 5120,
            approximator_layers: 5,
        }
    }
}

impl ChromaTransformerConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// The modulation index length `out_dim = 3·N_single + 2·6·N_double + 2` — the number of
    /// `[inner_dim]` modulation rows the Approximator emits (`pooled_temb` is `[B, mod_index_len, inner]`).
    pub fn mod_index_len(&self) -> usize {
        3 * self.num_single_layers + 2 * 6 * self.num_layers + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_defaults_match_mlx_provider() {
        assert_eq!(ChromaVariant::Hd.default_steps(), 28);
        assert_eq!(ChromaVariant::Base.default_steps(), 28);
        assert_eq!(ChromaVariant::Flash.default_steps(), 8);
        assert_eq!(ChromaVariant::Hd.default_true_cfg(), 4.0);
        assert_eq!(ChromaVariant::Flash.default_true_cfg(), 1.0);
        assert_eq!(ChromaVariant::Hd.sigma_shift(), 3.0);
        assert_eq!(ChromaVariant::Flash.sigma_shift(), 1.0);
        assert!(ChromaVariant::Base.use_beta_sigmas());
        assert!(!ChromaVariant::Hd.use_beta_sigmas());
        assert!(!ChromaVariant::Flash.use_beta_sigmas());
    }

    #[test]
    fn mod_index_len_is_344() {
        // 3·38 + 2·6·19 + 2 = 114 + 228 + 2 = 344.
        assert_eq!(ChromaTransformerConfig::default().mod_index_len(), 344);
        assert_eq!(ChromaTransformerConfig::default().inner_dim(), 3072);
    }

    #[test]
    fn descriptors_advertise_only_wired_txt2img_surface() {
        for v in [ChromaVariant::Hd, ChromaVariant::Base, ChromaVariant::Flash] {
            let d = v.descriptor();
            assert_eq!(d.backend, "candle");
            assert_eq!(d.modality, Modality::Image);
            assert!(d.capabilities.supports_true_cfg);
            assert!(d.capabilities.supports_negative_prompt);
            assert!(!d.capabilities.supports_guidance);
            assert!(!d.capabilities.mac_only);
            assert!(d.capabilities.conditioning.is_empty());
            assert!(!d.capabilities.supports_lora);
            assert!(!d.capabilities.supports_lokr);
            assert!(d.capabilities.supported_quants.is_empty());
            assert_eq!(d.capabilities.min_size, 256);
            assert_eq!(d.capabilities.max_size, 2048);
            assert_eq!(d.capabilities.max_count, 8);
        }
    }

    /// epic 7114 P4 (sc-7123): each variant advertises the full curated sampler/scheduler menu plus
    /// the legacy `flow_match` / `linear` aliases (which fall back to euler / the native schedule).
    #[test]
    fn descriptors_advertise_curated_sampler_scheduler_menu() {
        for v in [ChromaVariant::Hd, ChromaVariant::Base, ChromaVariant::Flash] {
            let caps = v.descriptor().capabilities;
            for s in candle_gen::curated_sampler_names() {
                assert!(caps.samplers.contains(&s), "missing sampler {s}");
            }
            for s in candle_gen::curated_scheduler_names() {
                assert!(caps.schedulers.contains(&s), "missing scheduler {s}");
            }
            assert!(caps.samplers.contains(&DEFAULT_SAMPLER)); // legacy "flow_match" alias retained
            assert!(caps.schedulers.contains(&"linear")); // legacy scheduler alias retained
            assert!(caps.samplers.contains(&"euler")); // the N1 default integrator
        }
    }
}
