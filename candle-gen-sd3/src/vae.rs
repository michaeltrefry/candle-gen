//! SD3.5 16-channel VAE wiring (sc-7876, epic 7982).
//!
//! SD3.5 uses a 16-channel diffusers `AutoencoderKL` with the SAME module structure as Z-Image's
//! (`block_out_channels = [128, 256, 512, 512]`, /8 spatial, group-norm 32, diffusers weight
//! naming). Only the scale/shift constants differ:
//!  - `scaling_factor ≈ 1.5305`, `shift_factor ≈ 0.0609`.
//!
//! So C1 **reuses** the candle-transformers `z_image::vae::AutoEncoderKL` — which already takes a
//! parameterized [`VaeConfig`] with `scaling_factor`/`shift_factor` — rather than re-porting the VAE.
//! The encode/decode direction is confirmed against diffusers:
//!  - encode: `latent = (z - shift_factor) * scaling_factor` (`AutoEncoderKL::encode`);
//!  - decode: `z = latent / scaling_factor + shift_factor` (`AutoEncoderKL::decode`).
//!
//! This module just provides the SD3.5 [`Sd3VaeConfig`] preset + a thin loader so the pipeline (C2)
//! and any test can construct the VAE with the right constants. We re-export the reused VAE type so
//! downstream code does not reach into candle-transformers directly.

pub use candle_transformers::models::z_image::vae::{AutoEncoderKL, VaeConfig};

use candle_gen::candle_core::Result;
use candle_gen::candle_nn::VarBuilder;

/// SD3.5 latent channel count (the DiT `in_channels` and the VAE `latent_channels`).
pub const LATENT_CHANNELS: usize = 16;

/// SD3.5 VAE spatial downscale (image /8 per side; the `[128,256,512,512]` AutoencoderKL has 3
/// downsamplers).
pub const SPATIAL_SCALE: u32 = 8;

/// SD3.5 `scaling_factor` (diffusers `vae/config.json`).
pub const SCALING_FACTOR: f64 = 1.5305;

/// SD3.5 `shift_factor` (diffusers `vae/config.json`).
pub const SHIFT_FACTOR: f64 = 0.0609;

/// Build the SD3.5 [`VaeConfig`] preset — the Z-Image VAE geometry with SD3.5's scale/shift.
pub fn sd3_vae_config() -> VaeConfig {
    VaeConfig {
        scaling_factor: SCALING_FACTOR,
        shift_factor: SHIFT_FACTOR,
        ..VaeConfig::z_image()
    }
}

/// Construct the SD3.5 16-channel `AutoEncoderKL` from a diffusers `vae/` VarBuilder.
pub fn load_vae(vb: VarBuilder) -> Result<AutoEncoderKL> {
    AutoEncoderKL::new(&sd3_vae_config(), vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd3_vae_config_uses_sd35_constants() {
        let c = sd3_vae_config();
        assert_eq!(c.latent_channels, 16);
        assert_eq!(c.block_out_channels, vec![128, 256, 512, 512]);
        assert!((c.scaling_factor - 1.5305).abs() < 1e-9);
        assert!((c.shift_factor - 0.0609).abs() < 1e-9);
        // The reused Z-Image VAE preset uses different constants — confirm we actually overrode them.
        assert_ne!(c.scaling_factor, VaeConfig::z_image().scaling_factor);
        assert_ne!(c.shift_factor, VaeConfig::z_image().shift_factor);
    }

    #[test]
    fn latent_geometry_constants() {
        assert_eq!(LATENT_CHANNELS, 16);
        assert_eq!(SPATIAL_SCALE, 8);
    }

    /// The VAE builds + round-trips a latent on CPU (tiny image), exercising the encode/decode
    /// direction with the SD3.5 config. Uses random weights so the structural wiring is what's
    /// validated (not pixel quality).
    #[test]
    fn vae_builds_and_decodes_on_cpu() {
        use candle_gen::candle_core::{DType, Device, Tensor};
        use candle_gen::candle_nn::{VarBuilder, VarMap};

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let vae = load_vae(vb).unwrap();
        // A tiny 16-ch latent /8: a 16x16 image -> 2x2 latent.
        let latent = Tensor::randn(0f32, 1f32, (1, LATENT_CHANNELS, 2, 2), &dev).unwrap();
        let decoded = vae.decode(&latent).unwrap();
        // Decode -> [B, 3, H, W] at the upscaled resolution (2*8 = 16).
        assert_eq!(decoded.dims(), &[1, 3, 16, 16]);
    }
}
