//! Pipeline glue for LTX-2.3 txt2video: latent geometry, deterministic CPU-seeded noise (sc-3673),
//! latent token flatten/unflatten, and frames → `gen_core::Image`.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::gen_core::Image;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::config::{LATENT_CHANNELS, SPATIAL_SCALE, TEMPORAL_SCALE};

/// Latent dims `(t_lat, h_lat, w_lat)` for `frames × height × width`: temporal `(F-1)/8 + 1`, spatial
/// `/32`.
pub fn latent_dims(frames: u32, width: u32, height: u32) -> (usize, usize, usize) {
    let t_lat = (frames as usize - 1) / TEMPORAL_SCALE + 1;
    let h_lat = height as usize / SPATIAL_SCALE;
    let w_lat = width as usize / SPATIAL_SCALE;
    (t_lat, h_lat, w_lat)
}

/// Deterministic N(0,1) latent noise `[1, 128, t_lat, h_lat, w_lat]` (f32) — CPU `StdRng` (ChaCha),
/// launch-portable per seed.
pub fn create_noise(
    seed: u64,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    device: &Device,
) -> Result<Tensor> {
    let n = LATENT_CHANNELS * t_lat * h_lat * w_lat;
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Tensor::from_vec(data, (1, LATENT_CHANNELS, t_lat, h_lat, w_lat), device)
}

/// `[B, 128, F, H, W]` → `[B, S, 128]` packed tokens (C-major over F,H,W).
pub fn flatten_latent(latent: &Tensor) -> Result<Tensor> {
    let (b, c, f, h, w) = latent.dims5()?;
    latent
        .reshape((b, c, f * h * w))?
        .transpose(1, 2)?
        .contiguous()
}

/// `[B, S, 128]` velocity → `[B, 128, F, H, W]`.
pub fn unflatten_latent(tokens: &Tensor, f: usize, h: usize, w: usize) -> Result<Tensor> {
    let (b, _s, c) = tokens.dims3()?;
    tokens
        .transpose(1, 2)?
        .reshape((b, c, f, h, w))?
        .contiguous()
}

/// Decoded video `[1, 3, T, H, W]` in `[-1, 1]` → one RGB8 [`Image`] per frame.
pub fn frames_to_images(decoded: &Tensor) -> Result<Vec<Image>> {
    let u8s = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .to_device(&Device::Cpu)?;
    let (_b, c, t, h, w) = u8s.dims5()?;
    let frames = u8s.squeeze(0)?; // [3,T,H,W]
    let mut out = Vec::with_capacity(t);
    for ti in 0..t {
        let frame = frames.narrow(1, ti, 1)?.squeeze(1)?; // [3,H,W]
        let pixels = frame.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        debug_assert_eq!(c, 3);
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        });
    }
    Ok(out)
}
