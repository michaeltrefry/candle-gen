//! Shared SAM3 leaf helpers for the candle port: a safetensors weight map ([`Weights`]), a
//! per-last-dim [`Linear`], NHWC↔NCHW conv wrappers, and a small no-mask [`sdpa`] / [`layer_norm`].
//! The candle twin of how `mlx-gen-sam3` uses `mlx_gen::weights::Weights` + `AdaptableLinear`.
//!
//! Layout: SAM3 loads the RAW `facebook/sam3` torch checkpoint, whose conv kernels are OIHW
//! (`conv2d`) / IOHW (`conv_transpose2d`) — already candle-native — so we DON'T permute kernels (the
//! MLX side does, because MLX convs are OHWI). We only transpose *activations* NHWC↔NCHW around each
//! conv so the transformer body stays channels-last and mirrors the MLX modules line-by-line.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{safetensors, DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::softmax;
use candle_gen::candle_nn::{GroupNorm, LayerNorm, Module};
use candle_gen::{CandleError, Result};

/// A loaded SAM3 weight map. Tensors are coerced to f32 on load — the parity oracle is f32 and SAM3
/// fits comfortably in f32 on the target box; the Q8/Q4 quant path lands in a later slice (sc-6246).
pub struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    /// Load every tensor from one `.safetensors` file onto `device`, coercing to f32.
    pub fn from_file(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let mut map = HashMap::new();
        Self::extend_from(&mut map, path.as_ref(), device)?;
        Ok(Self { map })
    }

    /// Load + merge every `*.safetensors` shard in `dir` (the sharded `facebook/sam3` checkpoint).
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| CandleError::Msg(format!("read_dir {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(CandleError::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        let mut map = HashMap::new();
        for shard in &shards {
            Self::extend_from(&mut map, shard, device)?;
        }
        Ok(Self { map })
    }

    fn extend_from(map: &mut HashMap<String, Tensor>, path: &Path, device: &Device) -> Result<()> {
        let raw = safetensors::load(path, device)?;
        for (k, v) in raw {
            let v = if v.dtype() == DType::F32 {
                v
            } else {
                v.to_dtype(DType::F32)?
            };
            map.insert(k, v);
        }
        Ok(())
    }

    /// Fetch a required tensor, erroring (not panicking) when a checkpoint is missing a key.
    pub fn require(&self, key: &str) -> Result<Tensor> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CandleError::Msg(format!("missing tensor: {key}")))
    }

    /// Fetch an optional tensor (e.g. a `.bias` that some projections omit).
    pub fn get(&self, key: &str) -> Option<Tensor> {
        self.map.get(key).cloned()
    }
}

/// `"{prefix}.{leaf}"` (or just `leaf` when `prefix` is empty) — the empty-prefix-aware key join
/// (mirrors `mlx-gen-sam3`'s `util::join`).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// A dense linear over the LAST dim: weight `[out, in]` (torch/candle native), optional bias `[out]`.
/// Applies to any `[.., in]` tensor by flattening the leading dims — robust for both the NHWC
/// `[b,H,W,C]` projections and the `[b,nh,seq,hd]` head tensors the SAM3 modules feed it. The fused
/// `addmm` semantics match the reference `nn.Linear` (and the MLX `AdaptableLinear::dense`).
pub(crate) struct Linear {
    /// Pre-transposed `[in, out]`, contiguous (so the per-call matmul is a plain `[lead,in]@[in,out]`).
    weight_t: Tensor,
    bias: Option<Tensor>,
    out_features: usize,
}

impl Linear {
    /// Load `{name}.weight` (+ optional `{name}.bias`).
    pub fn load(w: &Weights, name: &str) -> Result<Self> {
        let weight = w.require(&format!("{name}.weight"))?; // [out, in]
        let out_features = weight.dim(0)?;
        Ok(Self {
            weight_t: weight.t()?.contiguous()?,
            bias: w.get(&format!("{name}.bias")),
            out_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let in_features = *dims.last().expect("linear input has rank >= 1");
        let lead: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((lead, in_features))?;
        let mut y = x2.matmul(&self.weight_t)?; // [lead, out]
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        let mut out_shape = dims;
        *out_shape.last_mut().unwrap() = self.out_features;
        Ok(y.reshape(out_shape)?)
    }
}

/// LayerNorm over the last dim with explicit weight/bias (eps as the reference's f64).
pub(crate) fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let ln = LayerNorm::new(w.clone(), b.clone(), eps);
    Ok(ln.forward(x)?)
}

/// Scaled-dot-product attention, no mask. `q`/`k`/`v`: `[b, nh, seq, hd]` → `[b, nh, seq, hd]`.
pub(crate) fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    sdpa_masked(q, k, v, scale, None)
}

/// Scaled-dot-product attention with an optional **additive** mask, broadcast onto the
/// `[b, nh, seq_q, seq_k]` scores before softmax (`-1e9` at blocked positions, `0` elsewhere — the
/// CLIP causal+key-padding convention). `q`/`k`/`v`: `[b, nh, seq, hd]`; `mask`: any shape that
/// broadcasts to the scores (e.g. `[1, 1, seq_q, seq_k]`). Mirrors the reference / MLX
/// `scaled_dot_product_attention(..., mask, None)`.
pub(crate) fn sdpa_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let mut attn = (q.contiguous()?.matmul(&kt)? * scale)?; // [b, nh, seq_q, seq_k]
    if let Some(m) = mask {
        attn = attn.broadcast_add(m)?;
    }
    let attn = softmax(&attn, D::Minus1)?;
    Ok(attn.matmul(&v.contiguous()?)?)
}

/// `conv2d` on an NHWC activation with a torch-native OIHW kernel (loaded as-is). Transposes
/// NHWC→NCHW, runs candle `conv2d`, adds the optional `[O]` bias, transposes back to NHWC.
pub(crate) fn conv2d_nhwc(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let xc = x.permute([0, 3, 1, 2])?.contiguous()?; // NHWC → NCHW
    let mut y = xc.conv2d(w, padding, stride, 1, 1)?; // [N, O, H', W']
    if let Some(b) = bias {
        y = y.broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)?;
    }
    Ok(y.permute([0, 2, 3, 1])?.contiguous()?) // NCHW → NHWC
}

/// `conv_transpose2d` on an NHWC activation with a torch-native IOHW kernel (loaded as-is), pad 0 /
/// output_pad 0, plus the `[O]` bias.
pub(crate) fn conv_transpose2d_nhwc(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    stride: usize,
) -> Result<Tensor> {
    let xc = x.permute([0, 3, 1, 2])?.contiguous()?;
    let y = xc.conv_transpose2d(w, 0, 0, stride, 1)?; // padding, output_padding, stride, dilation
    let y = y.broadcast_add(&bias.reshape((1, bias.elem_count(), 1, 1))?)?;
    Ok(y.permute([0, 2, 3, 1])?.contiguous()?)
}

/// `k×k` max-pool (stride `k`) on an NHWC activation.
pub(crate) fn maxpool2d_nhwc(x: &Tensor, k: usize) -> Result<Tensor> {
    let xc = x.permute([0, 3, 1, 2])?.contiguous()?;
    let y = xc.max_pool2d(k)?;
    Ok(y.permute([0, 2, 3, 1])?.contiguous()?)
}

/// GroupNorm over an NHWC activation (the mask decoder runs channels-last). candle's [`GroupNorm`]
/// normalizes channel-dim-1 (NCHW), so transpose NHWC→NCHW, normalize, transpose back. The channel
/// count is read from the activation; `weight`/`bias` are the `[C]` affine params.
pub(crate) fn group_norm_nhwc(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    num_groups: usize,
    eps: f64,
) -> Result<Tensor> {
    let xc = x.permute([0, 3, 1, 2])?.contiguous()?; // NHWC → NCHW
    let c = xc.dim(1)?;
    let gn = GroupNorm::new(weight.clone(), bias.clone(), c, num_groups, eps)?;
    Ok(gn.forward(&xc)?.permute([0, 2, 3, 1])?.contiguous()?) // NCHW → NHWC
}

/// Nearest-neighbour `factor`× upsample of an NHWC activation (the FPN pixel decoder's 2× upsample).
/// candle's `upsample_nearest2d` works on the trailing two (NCHW H/W) dims, so transpose around it.
pub(crate) fn upsample_nearest2d_nhwc(x: &Tensor, factor: usize) -> Result<Tensor> {
    let xc = x.permute([0, 3, 1, 2])?.contiguous()?; // NHWC → NCHW
    let (_, _, h, w) = xc.dims4()?;
    let y = xc.upsample_nearest2d(h * factor, w * factor)?;
    Ok(y.permute([0, 2, 3, 1])?.contiguous()?) // NCHW → NHWC
}
