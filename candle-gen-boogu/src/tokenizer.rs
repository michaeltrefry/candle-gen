//! Boogu instruction tokenization — the Qwen3-VL chat template + tokenizer that turns a text prompt
//! into the `input_ids` the condition encoder consumes. Port of `mlx-gen-boogu`'s `tokenizer.rs`.
//!
//! The reference builds messages `[system, user]` and calls `apply_chat_template(...,
//! add_generation_prompt=False)` (no trailing assistant turn). For text-to-image the system prompt is
//! [`SYSTEM_PROMPT_T2I`]; the CFG-negative is the **empty** instruction with [`SYSTEM_PROMPT_DROP`].
//! We render the exact ChatML string ourselves and encode with `add_special_tokens=false` (the
//! `<|im_start|>`/`<|im_end|>` markers are literal special tokens already in the string).

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::{CandleError, Result};

/// Text-to-image system prompt (reference `SYSTEM_PROMPT_4_T2I`).
pub const SYSTEM_PROMPT_T2I: &str = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.";

/// Empty-instruction (CFG negative) / unified-edit system prompt (reference `SYSTEM_PROMPT_DROP` ==
/// `SYSTEM_PROMPT_4_TI2I_UNIFIED`).
pub const SYSTEM_PROMPT_DROP: &str = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";

/// Qwen3-VL vision marker tokens (`mllm/tokenizer.json` added tokens). The processor expands a single
/// `<|image_pad|>` into `merged` copies; we render the expanded block directly.
const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
const IMAGE_PAD: &str = "<|image_pad|>";

/// Render the ChatML string for a `(system, user)` turn pair with no generation prompt:
/// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n`.
fn render_chat(system: &str, user: &str) -> String {
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n")
}

/// Render the ChatML string for an image-conditioned `(system, user)` turn, with the reference image
/// block (`<|vision_start|>` + `num_image_tokens`×`<|image_pad|>` + `<|vision_end|>`) prepended to the
/// user text — the Qwen3-VL chat template + processor expansion for `content = [image, text]` (image
/// first, no separator, then the instruction).
fn render_chat_with_image(system: &str, user: &str, num_image_tokens: usize) -> String {
    let pads = IMAGE_PAD.repeat(num_image_tokens);
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{VISION_START}{pads}{VISION_END}{user}<|im_end|>\n"
    )
}

/// The Boogu condition tokenizer: the snapshot's `mllm/tokenizer.json` wrapped so we can render the
/// Boogu chat templates and encode them. Builds `input_ids` directly on the model device.
pub struct BooguTokenizer {
    inner: TextTokenizer,
    device: Device,
}

impl BooguTokenizer {
    /// Load from a snapshot's `mllm/tokenizer.json`.
    pub fn from_snapshot(root: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let inner = TextTokenizer::from_file(
            root.as_ref().join("mllm").join("tokenizer.json"),
            TokenizerConfig {
                // We render the chat string ourselves and call `encode_ids` directly, so the config
                // template/padding are unused; keep them inert.
                max_length: 1280,
                pad_token_id: 151643, // Qwen <|endoftext|>; unused (no padding on this path)
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("boogu: load mllm tokenizer: {e}")))?;
        Ok(Self {
            inner,
            device: device.clone(),
        })
    }

    /// Encode a rendered chat string to a `[1, L]` u32 `input_ids` tensor (`add_special_tokens=false`).
    fn encode(&self, text: &str) -> Result<Tensor> {
        let ids = self
            .inner
            .encode_ids(text, false)
            .map_err(|e| CandleError::Msg(format!("boogu: tokenize: {e}")))?;
        if ids.is_empty() {
            return Err(CandleError::Msg("boogu: empty token sequence".into()));
        }
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let len = ids.len();
        Ok(Tensor::from_vec(ids, (1, len), &self.device)?)
    }

    /// Encode the **positive** text-to-image instruction → `input_ids` `[1, L]`.
    pub fn encode_t2i(&self, prompt: &str) -> Result<Tensor> {
        self.encode(&render_chat(SYSTEM_PROMPT_T2I, prompt))
    }

    /// Encode the CFG **negative** (empty instruction with the drop system prompt) → `[1, L]`.
    pub fn encode_negative(&self) -> Result<Tensor> {
        self.encode(&render_chat(SYSTEM_PROMPT_DROP, ""))
    }

    /// Encode the **edit** instruction (text-only) → `input_ids` `[1, L]`. The TI2I unified system
    /// prompt ([`SYSTEM_PROMPT_DROP`]) is shared with image editing, so the CFG negative is just
    /// [`Self::encode_negative`] (empty user text, same system prompt).
    pub fn encode_edit(&self, instruction: &str) -> Result<Tensor> {
        self.encode(&render_chat(SYSTEM_PROMPT_DROP, instruction))
    }

    /// Encode the **image-conditioned edit** instruction → `input_ids` `[1, L]`, with the reference
    /// image's `num_image_tokens` (= merged vision tokens) `<|image_pad|>` placeholders spliced into
    /// the user turn. The text encoder then replaces those placeholder embeddings with the vision
    /// tower's output ([`crate::text_encoder::BooguTextEncoder::last_hidden_with_image`]).
    pub fn encode_edit_with_image(
        &self,
        instruction: &str,
        num_image_tokens: usize,
    ) -> Result<Tensor> {
        self.encode(&render_chat_with_image(
            SYSTEM_PROMPT_DROP,
            instruction,
            num_image_tokens,
        ))
    }
}
