//! Weight loading for the Ideogram 4 DiT — a thin shape-inferring wrapper over candle's
//! [`MmapedSafetensors`], mirroring `mlx-gen-ideogram`'s `Weights`/`lin` interface so the
//! `transformer` port stays a near-1:1 translation. [`linear`] builds a [`Linear`] from the actual
//! `{base}.weight` (and optional `{base}.bias`) tensor shapes, so dims that aren't in the public
//! config (e.g. the `t_embedding` MLP hidden width) need no hardcoding.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::candle_nn::Linear;

/// An mmaped component-directory of `.safetensors`, loading tensors at a fixed compute dtype.
pub struct Weights {
    st: MmapedSafetensors,
    device: Device,
    dtype: DType,
}

impl Weights {
    /// mmap every `*.safetensors` in `dir` (sorted; later files win on name collision).
    pub fn from_dir(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| Error::Msg(format!("ideogram: read {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(Error::Msg(format!(
                "ideogram: no .safetensors in {}",
                dir.display()
            )));
        }
        // SAFETY: read-only mmap of weight files; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
        })
    }

    /// Load `name` at the component dtype.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(self.dtype)
    }

    /// Load `name` forcing f32 (norm weights / tables that want full precision).
    pub fn get_f32(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(DType::F32)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.st.get(name).is_ok()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Build a [`Linear`] from `{base}.weight` (+ `{base}.bias` when `bias`), inferring in/out dims from
/// the stored tensor shape (`[out, in]`, PyTorch/HF convention).
pub fn linear(w: &Weights, base: &str, bias: bool) -> Result<Linear> {
    let weight = w.get(&format!("{base}.weight"))?;
    let bias = if bias {
        Some(w.get(&format!("{base}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}
