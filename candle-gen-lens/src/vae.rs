//! Lens VAE decode (sc-5113) — a thin shim over the **already-ported** Flux.2 `AutoencoderKLFlux2`
//! ([`candle_gen_flux2::vae::Flux2Vae`]). The Lens latent space *is* the Flux.2 one (32-ch latent,
//! 2×2 patchify into the 128-ch transformer space, BatchNorm-stats normalization), so the whole
//! `LensPipeline._decode` reduces to: reshape the DiT output into the packed NCHW grid and call the
//! shared decode.
//!
//! ## Why the reshape is the whole shim
//! The reference `_decode` does `rearrange(b (h w) (c p1 p2) -> b c (h p1) (w p2))` then
//! `_patchify_latents` (re-pack 2×2) → bn de-normalize → `_unpatchify_latents` → `vae.decode`. The
//! rearrange-then-patchify pair is an **identity** that collapses to a plain reshape from
//! `[B, h·w, 128]` to the packed grid `[B, 128, h, w]` (the DiT's 128 channels already carry the
//! `c·4 + p1·2 + p2` packing, exactly [`Flux2Vae::decode_packed`]'s expected channel order). The
//! bn de-normalize (`x·std + mean`, `std = √(running_var + 1e-4)`), the 2×2 unpatchify, and the
//! AutoencoderKL decode are then the shared Flux.2 path verbatim — only the checkpoint differs
//! (the Lens `vae/` snapshot, loaded into the same `Flux2Vae`).

use candle_gen::candle_core::{DType, Result, Tensor};

pub use candle_gen_flux2::vae::Flux2Vae;

/// Decode the Lens DiT output into an image. `dit_out`: `[B, h·w, 128]` (the transformer's packed
/// patch-space velocity at the final step); `(latent_h, latent_w)` is the packed latent grid
/// (`= height/16, width/16`). Returns `[B, 3, H, W]` (NCHW) in ~`[−1, 1]`, where `H = latent_h·16`,
/// `W = latent_w·16` (2× unpatchify × 8× VAE upsample).
pub fn decode(
    vae: &Flux2Vae,
    dit_out: &Tensor,
    latent_h: usize,
    latent_w: usize,
) -> Result<Tensor> {
    let (b, _, c) = dit_out.dims3()?; // [B, h·w, 128]
    let packed = dit_out
        .reshape((b, latent_h, latent_w, c))?
        .permute((0, 3, 1, 2))? // [B, h, w, 128] → [B, 128, h, w] (NCHW)
        .contiguous()?;
    vae.decode_packed(&packed)
}

/// Convert a decoded image `[B, 3, H, W]` in `[−1, 1]` to `u8` `[0, 255]` (`(x.clamp(−1,1)+1)·127.5`),
/// matching the reference `_to_pil` quantization.
pub fn to_uint8(image: &Tensor) -> Result<Tensor> {
    let x = image.to_dtype(DType::F32)?.clamp(-1f32, 1f32)?;
    ((x + 1.0)? * 127.5)?.to_dtype(DType::U8)
}
