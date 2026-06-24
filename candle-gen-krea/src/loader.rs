//! Weight loading for the Krea 2 DiT + Qwen3-VL-4B condition encoder — a thin shape-inferring wrapper
//! over candle's [`MmapedSafetensors`], mirroring `candle-gen-boogu`/`candle-gen-ideogram`'s `Weights`
//! interface so the port stays a near-1:1 translation of `mlx-gen-krea` (whose `Weights::from_dir`
//! loads the identity-keyed diffusers checkpoint directly). [`linear`] builds a [`Linear`] from the
//! actual `{base}.weight` (+ optional `{base}.bias`) tensor shapes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::Linear;

/// An mmaped component-directory of `.safetensors`, loading tensors at a fixed compute dtype.
///
/// An optional in-memory `overlay` (installed by [`set_overlay`](Weights::set_overlay)) takes priority
/// over the mmap for the keys it holds — the inference-side LoRA/LoKr adapter merge (sc-7836) folds its
/// deltas into the targeted dense weights on the CPU in f32, then installs them here so
/// [`crate::transformer::Krea2Transformer::load`] reads the **merged** weight without re-mmapping or
/// touching the untargeted bulk of the model. Overlay tensors are stored CPU-side (where the merge runs)
/// and moved to `device` / cast to the requested dtype on read, exactly like the mmap path.
pub struct Weights {
    st: MmapedSafetensors,
    device: Device,
    dtype: DType,
    overlay: HashMap<String, Tensor>,
}

impl Weights {
    /// mmap every `*.safetensors` in `dir` (sorted; later files win on name collision).
    pub fn from_dir(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| {
                candle_gen::candle_core::Error::Msg(format!("krea: read {}: {e}", dir.display()))
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea: no .safetensors in {}",
                dir.display()
            )));
        }
        // SAFETY: read-only mmap of weight files; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
        })
    }

    /// mmap a single `.safetensors` file (used by the committed parity fixtures).
    pub fn from_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::new(path)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
        })
    }

    /// Load `name` at the component dtype — from the [`overlay`](Weights::set_overlay) if present
    /// (adapter-merged weight), else the mmap.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.overlay.get(name) {
            return t.to_device(&self.device)?.to_dtype(self.dtype);
        }
        self.st.load(name, &self.device)?.to_dtype(self.dtype)
    }

    /// Load `name` preserving its on-disk dtype (e.g. int `input_ids` in a parity fixture). The overlay
    /// only ever holds merged DiT weights (never raw-dtype tensors), so this stays the mmap path.
    pub fn get_raw(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)
    }

    /// Load `name` forcing f32 (the `+1` norm weights and other precision-sensitive scalars) — from the
    /// overlay if present, else the mmap.
    pub fn get_f32(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.overlay.get(name) {
            return t.to_device(&self.device)?.to_dtype(DType::F32);
        }
        self.st.load(name, &self.device)?.to_dtype(DType::F32)
    }

    /// Load `name` onto the **CPU** at its on-disk dtype. Used by the inference-side adapter merge
    /// ([`crate::adapters`]), which reconstructs LoRA/LoKr deltas on the CPU (matching the CPU-loaded
    /// adapter factors) and folds them into the base weight before installing the [`overlay`](Weights::set_overlay).
    pub(crate) fn get_cpu(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &Device::Cpu)
    }

    /// Install an in-memory `overlay` of (CPU-resident) tensors that take priority over the mmap for the
    /// keys they cover — the adapter-merged dense weights (sc-7836). Replaces any prior overlay.
    pub(crate) fn set_overlay(&mut self, overlay: HashMap<String, Tensor>) {
        self.overlay = overlay;
    }

    pub fn contains(&self, name: &str) -> bool {
        self.overlay.contains_key(name) || self.st.get(name).is_ok()
    }

    /// All tensor keys in the component (for architecture validation).
    pub fn keys(&self) -> Vec<String> {
        self.st.tensors().into_iter().map(|(k, _)| k).collect()
    }

    /// On-disk shape of `name` (for architecture validation), or `None` if absent. The overlay never
    /// changes a weight's shape, so the mmap is authoritative.
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.st.get(name).ok().map(|v| v.shape().to_vec())
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

/// Standard RMSNorm over the last dim with weight `w` and eps (candle's fused op). Used by the Qwen3-VL
/// text encoder (whose norm weight is applied directly, NOT the DiT's `+1` convention).
pub(crate) fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    candle_gen::candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)
}

/// Load a `+1` RMSNorm weight (the reference `RMSNorm(weight = scale + 1.0)`): the on-disk `scale` is
/// centered at 0, so pre-fold the `+1` into an **f32** weight at load. Pairs with [`rms_scale`], which
/// always reduces in f32. Mirrors `mlx-gen-krea`'s `RmsScale`.
pub(crate) fn rms_scale_weight(w: &Weights, key: &str) -> Result<Tensor> {
    w.get_f32(key)? + 1.0
}

/// Apply a pre-folded `+1` RMSNorm (`weight` already = `scale + 1`, f32) over the last dim, computing
/// in f32 and casting back to `x`'s dtype — the byte-equivalent of the reference
/// `F.rms_norm(x.float(), weight).to(dtype)`.
pub(crate) fn rms_scale(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let y = candle_gen::candle_nn::ops::rms_norm(
        &x.to_dtype(DType::F32)?.contiguous()?,
        weight,
        eps as f32,
    )?;
    y.to_dtype(dt)
}
