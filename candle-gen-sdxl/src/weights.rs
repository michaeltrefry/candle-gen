//! A small safetensors key→`Tensor` map for the IP-Adapter / ControlNet loads (sc-5491) — the candle
//! twin of `mlx_gen::weights::Weights`. The stock SDXL UNet/VAE build through a `VarBuilder`, but the
//! IP-Adapter Resampler mixes a learned-`latents` tensor with fused-projection Linears, and the
//! ControlNet adds the per-residual zero-convs, so a raw key→`Tensor` map (cast to the compute dtype on
//! load) is the natural loader for both.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{safetensors as cst, DType, Tensor};

use candle_gen::candle_core::Device;
use candle_gen::{CandleError, Result};

/// A loaded checkpoint weight map (every tensor coerced to the requested compute dtype on load).
pub struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    /// Load every tensor from a `.safetensors` file onto `device`, casting to `dtype` (f16 in
    /// production, f32 for CPU parity), matching how `mlx-gen-sdxl` casts the IP-Adapter bundle to the
    /// UNet dtype before building.
    pub fn from_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let raw = cst::load(path, device)?;
        let mut map = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            // Only re-cast FLOATING tensors to the compute dtype. Integer buffers — e.g. the CLIP
            // image encoder's I64 `position_ids` (`h94/IP-Adapter` `models/image_encoder`) — are left
            // as-is: casting an int index buffer to f16 is meaningless, and on CUDA (sm_120) the
            // int→f16 cast kernel isn't compiled, so `to_dtype` there fails with
            // `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")` (sc-5488). The consumers
            // here only `require()` the float weights, so the untouched buffer is simply never read.
            let is_float = matches!(
                v.dtype(),
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            );
            let v = if is_float && v.dtype() != dtype {
                v.to_dtype(dtype)?
            } else {
                v
            };
            map.insert(k, v);
        }
        Ok(Self { map })
    }

    /// Fetch a required tensor, erroring (not panicking) when a checkpoint is missing a key.
    pub fn require(&self, key: &str) -> Result<Tensor> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CandleError::Msg(format!("missing tensor: {key}")))
    }

    /// Whether `key` is present (e.g. the ControlNet's optional `encoder_hid_proj`).
    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Iterate the tensor keys (drives the `ip_adapter.{n}` index discovery in
    /// [`load_ip_kv_pairs`](crate::ip_adapter::load_ip_kv_pairs)).
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    /// Build directly from an in-memory map — tests (including cross-crate ones, e.g. the FLUX
    /// IP-Adapter image-encoder fixtures, sc-5872) construct synthetic weights without a file, and a
    /// caller can assemble a checkpoint programmatically.
    pub fn from_map(map: HashMap<String, Tensor>) -> Self {
        Self { map }
    }
}
