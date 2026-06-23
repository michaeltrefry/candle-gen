//! `BooguImageTransformer2DModel` configuration — parsed from a diffusers
//! `transformer/config.json`, or constructed directly via [`BooguConfig::base`].
//!
//! Mirrors `mlx-gen-boogu`'s `config.rs` (the reference `BooguImageTransformer2DModel.__init__`
//! config surface). The Base and Turbo checkpoints share the same architecture (only the DiT weights
//! differ), so one config covers both.

use std::path::Path;

use candle_gen::CandleError;

type Result<T> = std::result::Result<T, CandleError>;

/// Architecture config for the Boogu mixed single/double-stream DiT.
#[derive(Debug, Clone, PartialEq)]
pub struct BooguConfig {
    pub patch_size: usize,
    pub in_channels: usize,
    /// Defaults to `in_channels` when the checkpoint leaves `out_channels` null.
    pub out_channels: usize,
    pub hidden_size: usize,
    /// Total transformer layers = double-stream + single-stream.
    pub num_layers: usize,
    pub num_double_stream_layers: usize,
    /// Depth of each refiner stack (context / noise / ref-image).
    pub num_refiner_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub multiple_of: usize,
    pub ffn_dim_multiplier: Option<f64>,
    pub norm_eps: f64,
    /// Per-axis (t, h, w) RoPE sub-dimensions; must sum to `hidden_size / num_attention_heads`.
    pub axes_dim_rope: [usize; 3],
    pub axes_lens: [usize; 3],
    pub rope_theta: f32,
    /// Qwen3-VL hidden width fed to the caption embedder (per-layer, before reduce).
    pub instruction_feat_dim: usize,
    pub num_instruction_feat_layers: usize,
    /// `"mean"` (reduce over layers, keep dim) or `"concat"` (sum dims). Base uses `"mean"`.
    pub reduce_type: String,
    pub timestep_scale: f32,
}

impl BooguConfig {
    /// Boogu-Image-0.1-Base / -Turbo architecture (verified from the published `transformer/config.json`).
    pub fn base() -> Self {
        Self {
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            hidden_size: 3360,
            num_layers: 40,
            num_double_stream_layers: 8,
            num_refiner_layers: 2,
            num_attention_heads: 28,
            num_kv_heads: 7,
            multiple_of: 256,
            ffn_dim_multiplier: None,
            norm_eps: 1e-5,
            axes_dim_rope: [40, 40, 40],
            axes_lens: [2048, 1664, 1664],
            rope_theta: 10000.0,
            instruction_feat_dim: 4096,
            num_instruction_feat_layers: 1,
            reduce_type: "mean".to_string(),
            timestep_scale: 1000.0,
        }
    }

    /// Parse `<root>/transformer/config.json`. Missing scalar fields fall back to [`BooguConfig::base`]
    /// values; the validated invariants (RoPE-sum, double ≤ total) are checked here.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let path = root.as_ref().join("transformer").join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| CandleError::Msg(format!("boogu: read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| CandleError::Msg(format!("boogu: parse {}: {e}", path.display())))?;
        Self::from_json(&v)
    }

    /// Build from an already-parsed `config.json` value.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let d = BooguConfig::base();
        let u = |k: &str, dflt: usize| {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(dflt)
        };
        let f = |k: &str, dflt: f64| v.get(k).and_then(serde_json::Value::as_f64).unwrap_or(dflt);

        let axes_dim = read_triple(v.get("axes_dim_rope"), d.axes_dim_rope);
        let axes_lens = read_triple(v.get("axes_lens"), d.axes_lens);

        let instr = v.get("instruction_feature_configs");
        let instruction_feat_dim = instr
            .and_then(|o| o.get("instruction_feat_dim"))
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(d.instruction_feat_dim);
        // The reference reads both spellings across versions; accept either, default 1.
        let num_instruction_feat_layers = instr
            .and_then(|o| {
                o.get("num_instruction_feat_layers")
                    .or_else(|| o.get("num_instruction_feature_layers"))
            })
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(d.num_instruction_feat_layers);
        let reduce_type = instr
            .and_then(|o| o.get("reduce_type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or(&d.reduce_type)
            .to_string();

        let in_channels = u("in_channels", d.in_channels);
        let out_channels = v
            .get("out_channels")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(in_channels);

        let cfg = Self {
            patch_size: u("patch_size", d.patch_size),
            in_channels,
            out_channels,
            hidden_size: u("hidden_size", d.hidden_size),
            num_layers: u("num_layers", d.num_layers),
            num_double_stream_layers: u("num_double_stream_layers", d.num_double_stream_layers),
            num_refiner_layers: u("num_refiner_layers", d.num_refiner_layers),
            num_attention_heads: u("num_attention_heads", d.num_attention_heads),
            num_kv_heads: u("num_kv_heads", d.num_kv_heads),
            multiple_of: u("multiple_of", d.multiple_of),
            ffn_dim_multiplier: v
                .get("ffn_dim_multiplier")
                .and_then(serde_json::Value::as_f64),
            norm_eps: f("norm_eps", d.norm_eps),
            axes_dim_rope: axes_dim,
            axes_lens,
            rope_theta: d.rope_theta,
            instruction_feat_dim,
            num_instruction_feat_layers,
            reduce_type,
            timestep_scale: f("timestep_scale", d.timestep_scale as f64) as f32,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Invariants mirrored from the reference `__init__`.
    pub fn validate(&self) -> Result<()> {
        if self.head_dim() != self.axes_dim_rope.iter().sum::<usize>() {
            return Err(CandleError::Msg(format!(
                "boogu: head_dim ({}) must equal sum(axes_dim_rope) ({})",
                self.head_dim(),
                self.axes_dim_rope.iter().sum::<usize>()
            )));
        }
        if self.num_double_stream_layers > self.num_layers {
            return Err(CandleError::Msg(format!(
                "boogu: num_double_stream_layers ({}) > num_layers ({})",
                self.num_double_stream_layers, self.num_layers
            )));
        }
        if self.num_attention_heads % self.num_kv_heads != 0 {
            return Err(CandleError::Msg(format!(
                "boogu: num_attention_heads ({}) not divisible by num_kv_heads ({})",
                self.num_attention_heads, self.num_kv_heads
            )));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_single_stream_layers(&self) -> usize {
        self.num_layers - self.num_double_stream_layers
    }
}

fn read_triple(v: Option<&serde_json::Value>, dflt: [usize; 3]) -> [usize; 3] {
    match v.and_then(serde_json::Value::as_array) {
        Some(a) if a.len() == 3 => {
            let mut out = dflt;
            for (i, x) in a.iter().enumerate() {
                if let Some(n) = x.as_u64() {
                    out[i] = n as usize;
                }
            }
            out
        }
        _ => dflt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_config_invariants() {
        let c = BooguConfig::base();
        c.validate().unwrap();
        assert_eq!(c.head_dim(), 120);
        assert_eq!(c.head_dim(), c.axes_dim_rope.iter().sum::<usize>());
        assert_eq!(c.num_single_stream_layers(), 32);
    }

    #[test]
    fn from_json_overrides_and_validates() {
        let v: serde_json::Value = serde_json::json!({
            "patch_size": 2, "in_channels": 16, "out_channels": null,
            "hidden_size": 3360, "num_layers": 40, "num_double_stream_layers": 8,
            "num_refiner_layers": 2, "num_attention_heads": 28, "num_kv_heads": 7,
            "multiple_of": 256, "ffn_dim_multiplier": null, "norm_eps": 1e-5,
            "axes_dim_rope": [40, 40, 40], "axes_lens": [2048, 1664, 1664],
            "instruction_feature_configs": {
                "instruction_feat_dim": 4096, "num_instruction_feature_layers": 1, "reduce_type": "mean"
            },
            "timestep_scale": 1000.0
        });
        let c = BooguConfig::from_json(&v).unwrap();
        assert_eq!(c, BooguConfig::base());
        assert_eq!(c.out_channels, 16); // null → in_channels
    }

    #[test]
    fn bad_rope_sum_rejected() {
        let mut c = BooguConfig::base();
        c.axes_dim_rope = [40, 40, 41];
        assert!(c.validate().is_err());
    }
}
