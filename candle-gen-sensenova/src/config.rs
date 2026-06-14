//! SenseNova-U1 (NEO-Unify) model configuration — the `neo_chat` `config.json` layout. The candle
//! port of `mlx-gen-sensenova`'s `config.rs` (pure `serde_json`, backend-agnostic).
//!
//! Reads the on-disk `config.json` via `serde_json::Value` (the provider convention — no `serde`
//! derive). The first-class (and only wired) target is the **8B-MoT** checkpoint, whose
//! `llm_config.model_type` is `"qwen3"` — the **dense** dual-path backbone. (The sparse-MoE A3B
//! variant the mlx provider future-proofs for is out of scope for this candle slice.)

use std::path::Path;

use serde_json::Value;

use candle_gen::{CandleError, Result};

/// The dense Qwen3 backbone config (`llm_config`). Extends the stock Qwen3 knobs with the spatial
/// rotary axes (`rope_theta_hw` / `max_position_embeddings_hw`) layered on top of the temporal one.
#[derive(Clone, Debug, PartialEq)]
pub struct NeoLlmConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub attention_bias: bool,
    /// Temporal RoPE base.
    pub rope_theta: f32,
    /// Spatial (height/width) RoPE base.
    pub rope_theta_hw: f32,
    pub max_position_embeddings: usize,
    pub max_position_embeddings_hw: usize,
}

impl NeoLlmConfig {
    /// Per-head dimension. The checkpoint sets `head_dim` explicitly (128); fall back to
    /// `hidden_size / num_attention_heads` if a config omits it.
    pub fn head_dim(&self) -> usize {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    fn from_value(v: &Value) -> Self {
        Self {
            model_type: get_str(v, "model_type", "qwen3"),
            hidden_size: get_usize(v, "hidden_size", 4096),
            intermediate_size: get_usize(v, "intermediate_size", 12288),
            num_hidden_layers: get_usize(v, "num_hidden_layers", 42),
            num_attention_heads: get_usize(v, "num_attention_heads", 32),
            num_key_value_heads: get_usize(v, "num_key_value_heads", 8),
            head_dim: get_usize(v, "head_dim", 128),
            rms_norm_eps: get_f64(v, "rms_norm_eps", 1e-6),
            vocab_size: get_usize(v, "vocab_size", 151936),
            attention_bias: get_bool(v, "attention_bias", false),
            rope_theta: get_f32(v, "rope_theta", 5_000_000.0),
            rope_theta_hw: get_f32(v, "rope_theta_hw", 10_000.0),
            max_position_embeddings: get_usize(v, "max_position_embeddings", 262_144),
            max_position_embeddings_hw: get_usize(v, "max_position_embeddings_hw", 10_000),
        }
    }
}

/// The NEO vision config (`vision_config`). For the 8B-MoT checkpoint the vision module is **not** a
/// transformer — only a Conv `patch_embedding` (3→`hidden_size`, kernel/stride `patch_size`) + 2D
/// RoPE + a Conv `dense_embedding` (`hidden_size`→`llm_hidden_size`, kernel/stride 2). The same
/// embedder structure backs both the understanding-path `vision_model` and the generation-path
/// `fm_modules.vision_model_mot_gen` (only the latter is used by the T2I slice).
#[derive(Clone, Debug, PartialEq)]
pub struct NeoVisionConfig {
    pub hidden_size: usize,
    pub llm_hidden_size: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub downsample_ratio: f32,
    pub rope_theta_vision: f32,
    pub max_position_embeddings_vision: usize,
}

impl NeoVisionConfig {
    fn from_value(v: &Value) -> Self {
        Self {
            hidden_size: get_usize(v, "hidden_size", 1024),
            llm_hidden_size: get_usize(v, "llm_hidden_size", 4096),
            num_channels: get_usize(v, "num_channels", 3),
            patch_size: get_usize(v, "patch_size", 16),
            downsample_ratio: get_f32(v, "downsample_ratio", 0.5),
            rope_theta_vision: get_f32(v, "rope_theta_vision", 10_000.0),
            max_position_embeddings_vision: get_usize(v, "max_position_embeddings_vision", 10_000),
        }
    }
}

/// The top-level NEO-Unify config (`config.json`, `model_type == "neo_chat"`): the dense Qwen3
/// backbone + the vision embedder + the flow-matching image-generation knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct NeoChatConfig {
    pub model_type: String,
    pub template: Option<String>,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    /// `lm_head` is a distinct tensor when `false` (the 8B-MoT case); tied to `embed_tokens`
    /// otherwise (in which case there is no `language_model.lm_head.weight`).
    pub tie_word_embeddings: bool,
    pub downsample_ratio: f32,
    pub patch_size: usize,

    // ---- flow-matching image generation ----
    pub timestep_shift: f32,
    pub time_schedule: String,
    pub time_shift_type: String,
    pub base_shift: f32,
    pub max_shift: f32,
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    pub noise_scale_mode: String,
    pub noise_scale: f32,
    pub noise_scale_max_value: f32,
    /// Reference sequence length the resolution-mode noise scale is normalised against
    /// (`noise_scale = sqrt(image_seq_len / noise_scale_base_image_seq_len) · noise_scale`).
    pub noise_scale_base_image_seq_len: usize,
    /// When `true` (the 8B-MoT case) the checkpoint carries a `fm_modules.noise_scale_embedder`.
    pub add_noise_scale_embedding: bool,
    pub fm_head_dim: usize,
    /// Number of Linear layers in the `fm_head` `Sequential` (GELU-interleaved → weights at indices
    /// `0, 2, …`). `2` for the 8B-MoT.
    pub fm_head_layers: usize,
    pub fm_head_mlp_ratio: f32,
    /// `false` for the 8B-MoT → the conv pixel decoders are absent; the pixel path is
    /// `fm_head` → unpatchify.
    pub use_pixel_head: bool,
    pub use_adaln: bool,

    pub llm: NeoLlmConfig,
    pub vision: NeoVisionConfig,
}

impl NeoChatConfig {
    /// Parse a `config.json` `Value` (the `neo_chat` layout).
    ///
    /// Per-field defaults follow the provider convention, but this config additionally carries
    /// generation-math scalars (`timestep_shift`, `noise_scale_*`, …) where a silent default means
    /// *wrong images*, not a load error. So gate on the `llm_config`/`vision_config` sub-objects
    /// being present-and-object before trusting those defaults: a snapshot missing either is corrupt
    /// or mislabeled, and must fail at load rather than fabricate an 8B-MoT and render garbage
    /// (F-145). Fields *within* each present sub-object still default individually.
    pub fn from_config_json(v: &Value) -> Result<Self> {
        let llm = NeoLlmConfig::from_value(require_object(v, "llm_config")?);
        let vision = NeoVisionConfig::from_value(require_object(v, "vision_config")?);
        Ok(Self {
            model_type: get_str(v, "model_type", "neo_chat"),
            template: v
                .get("template")
                .and_then(Value::as_str)
                .map(str::to_string),
            eos_token_id: get_usize(v, "eos_token_id", 151_645) as u32,
            pad_token_id: get_usize(v, "pad_token_id", 151_643) as u32,
            tie_word_embeddings: get_bool(v, "tie_word_embeddings", false),
            downsample_ratio: get_f32(v, "downsample_ratio", 0.5),
            patch_size: get_usize(v, "patch_size", 16),
            timestep_shift: get_f32(v, "timestep_shift", 1.0),
            time_schedule: get_str(v, "time_schedule", "standard"),
            time_shift_type: get_str(v, "time_shift_type", "exponential"),
            base_shift: get_f32(v, "base_shift", 0.5),
            max_shift: get_f32(v, "max_shift", 1.15),
            base_image_seq_len: get_usize(v, "base_image_seq_len", 64),
            max_image_seq_len: get_usize(v, "max_image_seq_len", 4096),
            noise_scale_mode: get_str(v, "noise_scale_mode", "resolution"),
            noise_scale: get_f32(v, "noise_scale", 1.0),
            noise_scale_max_value: get_f32(v, "noise_scale_max_value", 8.0),
            noise_scale_base_image_seq_len: get_usize(v, "noise_scale_base_image_seq_len", 64),
            add_noise_scale_embedding: get_bool(v, "add_noise_scale_embedding", true),
            fm_head_dim: get_usize(v, "fm_head_dim", 1536),
            fm_head_layers: get_usize(v, "fm_head_layers", 2),
            fm_head_mlp_ratio: get_f32(v, "fm_head_mlp_ratio", 1.0),
            use_pixel_head: get_bool(v, "use_pixel_head", false),
            // The reference key is `use_adaLN` (camelCase).
            use_adaln: get_bool(v, "use_adaLN", false),
            llm,
            vision,
        })
    }

    /// Read and parse `<root>/config.json`.
    pub fn from_dir(root: impl AsRef<Path>) -> Result<Self> {
        let path = root.as_ref().join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| CandleError::Msg(format!("sensenova: reading {}: {e}", path.display())))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| CandleError::Msg(format!("sensenova: parsing {}: {e}", path.display())))?;
        Self::from_config_json(&v)
    }
}

/// Require a sub-object (`llm_config`/`vision_config`) to be present and a JSON object. A snapshot
/// missing one is corrupt or mislabeled — error rather than silently default the whole object
/// (F-145). `null`, a scalar, or an array all fail the `is_object` check.
fn require_object<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    match v.get(key) {
        Some(o) if o.is_object() => Ok(o),
        _ => Err(CandleError::Msg(format!(
            "sensenova: config.json is missing the `{key}` object (corrupt or wrong snapshot); \
             refusing to fall back to 8B-MoT defaults for generation-math scalars"
        ))),
    }
}

fn get_str(v: &Value, key: &str, default: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

fn get_usize(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(default)
}

fn get_f32(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(Value::as_f64)
        .map(|n| n as f32)
        .unwrap_or(default)
}

fn get_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn get_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// A minimal `config.json` carrying the 8B-MoT structural values (the parser ignores the many
/// fields it does not model — `min_pixels`, `P_mean`, …). Shared by the config and loader tests.
#[cfg(test)]
const MOT_8B_CONFIG: &str = r#"{
      "model_type": "neo_chat",
      "template": "neo1_0",
      "tie_word_embeddings": false,
      "downsample_ratio": 0.5,
      "patch_size": 16,
      "timestep_shift": 1.0,
      "time_schedule": "standard",
      "time_shift_type": "exponential",
      "base_shift": 0.5,
      "max_shift": 1.15,
      "base_image_seq_len": 64,
      "max_image_seq_len": 4096,
      "noise_scale_mode": "resolution",
      "add_noise_scale_embedding": true,
      "noise_scale_max_value": 8.0,
      "fm_head_dim": 1536,
      "fm_head_layers": 2,
      "fm_head_mlp_ratio": 1,
      "use_pixel_head": false,
      "use_adaLN": false,
      "llm_config": {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 42,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000.0,
        "rope_theta_hw": 10000.0,
        "max_position_embeddings": 262144,
        "max_position_embeddings_hw": 10000,
        "vocab_size": 151936,
        "attention_bias": false
      },
      "vision_config": {
        "hidden_size": 1024,
        "llm_hidden_size": 4096,
        "num_channels": 3,
        "patch_size": 16,
        "downsample_ratio": 0.5,
        "rope_theta_vision": 10000.0,
        "max_position_embeddings_vision": 10000
      }
    }"#;

/// The parsed 8B-MoT config fixture, shared with the loader tests.
#[cfg(test)]
pub(crate) fn mot_8b() -> NeoChatConfig {
    NeoChatConfig::from_config_json(&serde_json::from_str(MOT_8B_CONFIG).unwrap())
        .expect("8B-MoT fixture has llm_config + vision_config")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_8b_mot_as_dense() {
        let c = mot_8b();
        assert_eq!(c.model_type, "neo_chat");
        assert_eq!(c.template.as_deref(), Some("neo1_0"));
        assert!(!c.tie_word_embeddings, "8B-MoT has a distinct lm_head");
        assert!(
            !c.use_pixel_head,
            "8B-MoT pixel path is fm_head, not a conv decoder"
        );
        assert!(c.add_noise_scale_embedding);
        assert_eq!(c.fm_head_layers, 2);

        let llm = &c.llm;
        assert_eq!(llm.num_hidden_layers, 42);
        assert_eq!(llm.head_dim(), 128);
        assert_eq!(llm.num_attention_heads, 32);
        assert_eq!(llm.num_key_value_heads, 8);

        assert_eq!(c.vision.hidden_size, 1024);
        assert_eq!(c.vision.llm_hidden_size, 4096);
    }

    #[test]
    fn errors_on_missing_subconfigs() {
        let no_llm: Value = serde_json::from_str(r#"{"vision_config":{}}"#).unwrap();
        let err =
            NeoChatConfig::from_config_json(&no_llm).expect_err("missing llm_config must error");
        assert!(err.to_string().contains("llm_config"), "got: {err}");

        let no_vision: Value = serde_json::from_str(r#"{"llm_config":{}}"#).unwrap();
        let err = NeoChatConfig::from_config_json(&no_vision)
            .expect_err("missing vision_config must error");
        assert!(err.to_string().contains("vision_config"), "got: {err}");

        let null_llm: Value =
            serde_json::from_str(r#"{"llm_config":null,"vision_config":{}}"#).unwrap();
        assert!(NeoChatConfig::from_config_json(&null_llm).is_err());

        let ok: Value = serde_json::from_str(r#"{"llm_config":{},"vision_config":{}}"#).unwrap();
        assert!(NeoChatConfig::from_config_json(&ok).is_ok());
    }
}
