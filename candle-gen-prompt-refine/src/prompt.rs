//! Llama-3.2 chat formatting, special-token ids, and the capability surface for the prompt-refine
//! text-LLM provider.
//!
//! The Rust `tokenizers` crate does not apply chat templates, so — like `candle-gen-joycaption` —
//! the Llama-3 instruct template is hand-assembled here as a literal string. The added special-token
//! strings (`<|start_header_id|>` etc.) map to their single token ids when encoded with
//! `add_special_tokens=false`; the `<|begin_of_text|>` BOS is prepended as an id at the call site
//! (matching the JoyCaption decoder path).

use candle_gen::gen_core::TextLlmCapabilities;

/// Registered text-LLM id (the worker's `prompt_refine` job resolves this).
pub const PROMPT_REFINE_ID: &str = "prompt_refine";
/// Descriptor family.
pub const PROMPT_REFINE_FAMILY: &str = "llama";
/// The default refinement checkpoint the worker downloads (informational; the provider loads from a
/// `WeightsSource::Dir` snapshot the caller resolves). A small, text-only, uncensored instruct LLM,
/// matching the Python `PromptRefiner`'s `DEFAULT_REFINE_MODEL`.
pub const DEFAULT_REFINE_MODEL: &str = "huihui-ai/Llama-3.2-3B-Instruct-abliterated";

/// `<|begin_of_text|>` — the BOS the HF Llama-3 chat template starts with, prepended as an id.
pub const BEGIN_OF_TEXT_TOKEN_ID: u32 = 128000;
/// `<|end_of_text|>` — a generation stop token.
pub const END_OF_TEXT_TOKEN_ID: u32 = 128001;
/// `<|eot_id|>` (end-of-turn) — the instruct-tuned model's primary stop token.
pub const EOT_TOKEN_ID: u32 = 128009;

/// Tokenizer context budget (Llama-3.2 supports far more; this bounds the prompt + system turn).
pub const DEFAULT_MAX_CONTEXT_TOKENS: usize = 8192;

/// Capability bounds (advertised + enforced by `validate`).
pub const MAX_PROMPT_CHARS: usize = 8000;
pub const MAX_SYSTEM_CHARS: usize = 32000;
pub const MAX_NEW_TOKENS_CAP: u32 = 2048;

/// The prompt-refine provider's advertised capability surface (candle backend, not mac-only).
pub fn capabilities() -> TextLlmCapabilities {
    TextLlmCapabilities {
        max_prompt_chars: MAX_PROMPT_CHARS,
        max_system_chars: MAX_SYSTEM_CHARS,
        supports_system_prompt: true,
        max_new_tokens: MAX_NEW_TOKENS_CAP,
        mac_only: false,
    }
}

/// Hand-assemble the Llama-3.2 instruct chat text for an optional `system` message + a `user` turn,
/// ending at the assistant generation prompt. The `<|begin_of_text|>` BOS is **not** included (the
/// caller prepends its id). When `system` is empty, the system block is omitted entirely (the model
/// accepts a user-only turn).
pub fn build_chat_text(system: &str, user: &str) -> String {
    let mut s = String::new();
    let system = system.trim();
    if !system.is_empty() {
        s.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        s.push_str(system);
        s.push_str("<|eot_id|>");
    }
    s.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
    s.push_str(user.trim());
    s.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_text_with_system_has_both_turns_and_assistant_prompt() {
        let t = build_chat_text("You are a rewriter.", "a cat");
        assert!(t.starts_with(
            "<|start_header_id|>system<|end_header_id|>\n\nYou are a rewriter.<|eot_id|>"
        ));
        assert!(t.contains("<|start_header_id|>user<|end_header_id|>\n\na cat<|eot_id|>"));
        assert!(t.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
        // BOS is prepended as an id, never embedded as a literal.
        assert!(!t.contains("<|begin_of_text|>"));
    }

    #[test]
    fn chat_text_without_system_omits_the_system_block() {
        let t = build_chat_text("   ", "a cat");
        assert!(!t.contains("system<|end_header_id|>"));
        assert!(t.starts_with("<|start_header_id|>user<|end_header_id|>\n\na cat<|eot_id|>"));
    }

    #[test]
    fn capabilities_advertise_the_prompt_refine_surface() {
        let c = capabilities();
        assert!(c.supports_system_prompt);
        assert!(!c.mac_only);
        assert_eq!(c.max_new_tokens, MAX_NEW_TOKENS_CAP);
        assert!(c.max_prompt_chars > 0 && c.max_system_chars > 0);
    }
}
