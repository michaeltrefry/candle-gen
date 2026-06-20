//! # candle-gen-prompt-refine
//!
//! A text-in / text-out **instruction-LLM** provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) implementation of the gen-core [`TextLlm`](candle_gen::gen_core::TextLlm) contract
//! (sc-5500). It runs **Llama-3.2-3B-Instruct** (the abliterated prompt-refine checkpoint by default)
//! to rewrite a user's prompt, replacing the worker's Python `prompt_refine.py`
//! (`AutoModelForCausalLM` + `model.generate`).
//!
//! Built directly on `candle_transformers::models::llama` (Llama-3.2 is fully supported there —
//! rope-scaling is auto-loaded from the snapshot's `config.json`), so there is no vendored decoder.
//! The Llama-3 chat template is hand-assembled in [`prompt`] (the `tokenizers` crate applies no
//! template), and decoding is a standard KV-cached greedy/top-p sampling loop via
//! `candle_transformers::generation::LogitsProcessor`.
//!
//! **Generic by contract.** This is a plain instruction LLM: the caller supplies the `system` message
//! (the prompt-rewrite rules + the model's prompt guide) and the `user` prompt, and gets back the raw
//! model text. The product-specific prompt assembly and any output cleanup (`<think>` stripping,
//! fence/quote trimming) live at the caller's edge — the worker — keeping this a reusable seam.
//! Registered as id `"prompt_refine"`, `backend = "candle"`, `mac_only = false`.

pub mod prompt;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::TextLlmRegistration;
use candle_gen::gen_core::tokenizer::{
    ChatTemplate, ConstraintDecodeTable, TextTokenizer, TokenizerConfig,
};
use candle_gen::gen_core::{
    self, default_seed, JsonState, LoadSpec, Progress, TextLlm, TextLlmConstraint,
    TextLlmDescriptor, TextLlmFinishReason, TextLlmOutput, TextLlmRequest, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks};

use prompt::{
    build_chat_text, capabilities, BEGIN_OF_TEXT_TOKEN_ID, DEFAULT_MAX_CONTEXT_TOKENS,
    END_OF_TEXT_TOKEN_ID, EOT_TOKEN_ID, PROMPT_REFINE_FAMILY, PROMPT_REFINE_ID,
};

/// The loaded model + tokenizer + resolved stop tokens (cached after the first `generate`).
struct Engine {
    llama: Llama,
    cfg: Config,
    tokenizer: TextTokenizer,
    stop_ids: Vec<u32>,
    device: Device,
    dtype: candle_gen::candle_core::DType,
    /// Per-token decode table for grammar-constrained decoding (sc-6585), filled lazily on the first
    /// constrained request — the Engine is shared via `Arc`, so the cache is a `OnceLock` and a plain
    /// free-text refine never builds it.
    constraint_table: OnceLock<ConstraintDecodeTable>,
}

/// How a token id behaves under a JSON constraint: never-content special/added tokens and the
/// end-of-text stop token are handled outside the grammar; ordinary tokens are masked by it.
#[derive(Clone, Copy, PartialEq)]
enum TokenKind {
    Ordinary,
    Special,
    Stop,
}

impl Engine {
    fn load(root: &std::path::Path, device: &Device) -> CResult<Self> {
        // Config (carries the Llama-3.2 rope_scaling block, auto-deserialized).
        let cfg_path = root.join("config.json");
        let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| {
            CandleError::Msg(format!(
                "prompt-refine: read config {}: {e}",
                cfg_path.display()
            ))
        })?;
        let llama_cfg: LlamaConfig = serde_json::from_slice(&cfg_bytes)
            .map_err(|e| CandleError::Msg(format!("prompt-refine: parse config.json: {e}")))?;
        let cfg = llama_cfg.into_config(false);

        let dtype = candle_gen::default_dtype();
        let mut files: Vec<PathBuf> = std::fs::read_dir(root)
            .map_err(|e| {
                CandleError::Msg(format!(
                    "prompt-refine: read snapshot {}: {e}",
                    root.display()
                ))
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "prompt-refine: no .safetensors in snapshot dir {}",
                root.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; the standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
        let llama = Llama::load(vb, &cfg)?;

        let tokenizer = TextTokenizer::from_file(
            root.join("tokenizer.json"),
            TokenizerConfig {
                max_length: DEFAULT_MAX_CONTEXT_TOKENS,
                pad_token_id: END_OF_TEXT_TOKEN_ID as i32,
                chat_template: ChatTemplate::None, // the template is hand-assembled in `prompt`
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("prompt-refine: load tokenizer: {e}")))?;

        Ok(Self {
            stop_ids: stop_ids_from_config(&cfg),
            llama,
            cfg,
            tokenizer,
            device: device.clone(),
            dtype,
            constraint_table: OnceLock::new(),
        })
    }
}

/// Generation stop tokens: the config's `eos_token_id` (single or multiple) unioned with the
/// Llama-3 `<|eot_id|>` and `<|end_of_text|>` (instruct models stop on `<|eot_id|>`, which some
/// `config.json`s omit from `eos_token_id`).
fn stop_ids_from_config(cfg: &Config) -> Vec<u32> {
    let mut ids = match &cfg.eos_token_id {
        Some(LlamaEosToks::Single(t)) => vec![*t],
        Some(LlamaEosToks::Multiple(v)) => v.clone(),
        None => Vec::new(),
    };
    for must in [EOT_TOKEN_ID, END_OF_TEXT_TOKEN_ID] {
        if !ids.contains(&must) {
            ids.push(must);
        }
    }
    ids
}

/// The candle prompt-refine text-LLM provider. Lazily loads weights on the first `generate`.
pub struct PromptRefiner {
    descriptor: TextLlmDescriptor,
    root: PathBuf,
    device: Device,
    engine: Mutex<Option<Arc<Engine>>>,
}

impl PromptRefiner {
    fn engine(&self) -> CResult<Arc<Engine>> {
        let mut guard = self
            .engine
            .lock()
            .expect("prompt-refine engine cache mutex poisoned");
        if let Some(e) = guard.as_ref() {
            return Ok(e.clone());
        }
        let engine = Arc::new(Engine::load(&self.root, &self.device)?);
        *guard = Some(engine.clone());
        Ok(engine)
    }

    fn run(
        &self,
        req: &TextLlmRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<TextLlmOutput> {
        let engine = self.engine()?;

        // Build the Llama-3 chat text (system + user → assistant prompt), map to ids without
        // auto-specials (the template carries the header/eot tokens as literal strings), then prepend
        // the <|begin_of_text|> BOS id — mirroring the JoyCaption decoder path.
        let chat = build_chat_text(&req.system, &req.prompt);
        let encoded = engine
            .tokenizer
            .encode_ids(&chat, false)
            .map_err(|e| CandleError::Msg(format!("prompt-refine: tokenize: {e}")))?;
        let mut tokens: Vec<u32> = Vec::with_capacity(encoded.len() + 1);
        tokens.push(BEGIN_OF_TEXT_TOKEN_ID);
        tokens.extend(encoded.into_iter().map(|i| i as u32));

        // Sampling: temperature < 1e-7 → greedy argmax (seed unused); else top-p nucleus. Seed is
        // caller-pinned or a fresh per-call draw, so a fixed seed reproduces the rewrite bit-for-bit.
        let seed = req.sampling.seed.unwrap_or_else(default_seed);
        let temperature = if req.sampling.temperature < 1e-7 {
            None
        } else {
            Some(req.sampling.temperature as f64)
        };
        let top_p = match temperature {
            Some(_) if req.sampling.top_p < 1.0 => Some(req.sampling.top_p as f64),
            _ => None,
        };
        let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);
        let mut cache = Cache::new(true, engine.dtype, &engine.cfg, &engine.device)?;

        // Constraint setup (sc-6585): classify each token id once (stop / never-content special /
        // grammar-masked ordinary) and start the JSON grammar; per-step masking only runs when a
        // constraint is set.
        let constraint = (req.constraint == Some(TextLlmConstraint::Json)).then(|| {
            engine
                .constraint_table
                .get_or_init(|| engine.tokenizer.constraint_decode_table())
        });
        let kinds: Vec<TokenKind> = match constraint {
            Some(table) => {
                let vocab = table.pieces.len();
                let mut kinds = vec![TokenKind::Ordinary; vocab];
                for &id in &table.special {
                    if (id as usize) < vocab {
                        kinds[id as usize] = TokenKind::Special;
                    }
                }
                for &id in &engine.stop_ids {
                    if (id as usize) < vocab {
                        kinds[id as usize] = TokenKind::Stop;
                    }
                }
                kinds
            }
            None => Vec::new(),
        };
        let mut json_state = JsonState::start();

        let total = req.sampling.max_new_tokens;
        let mut generated: Vec<u32> = Vec::new();
        let mut index_pos = 0usize;
        let mut finish = TextLlmFinishReason::MaxTokens;
        for step in 0..total {
            // Cooperative cancel between tokens → return the partial reply marked Cancelled (the
            // pre-inference already-cancelled case is the typed Err in `generate`).
            if req.cancel.is_cancelled() {
                finish = TextLlmFinishReason::Cancelled;
                break;
            }
            // With the KV cache, feed the whole prompt on step 0 and one token thereafter.
            let (context_size, context_index) = if step > 0 {
                (1usize, index_pos)
            } else {
                (tokens.len(), 0usize)
            };
            let ctxt = &tokens[tokens.len() - context_size..];
            let input = Tensor::new(ctxt, &engine.device)?.unsqueeze(0)?;
            let logits = engine.llama.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?; // (1, vocab) → (vocab,)
            index_pos += ctxt.len();

            let next = if let Some(table) = constraint {
                // Mask the (vocab,) logits to grammar-valid tokens before sampling: the stop token
                // only when the JSON value is already complete, never a non-stop special, otherwise
                // gated by feeding the token's decoded text to the JSON grammar.
                let can_stop = json_state.can_stop();
                let mut v: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let vocab = v.len();
                for (id, slot) in v.iter_mut().enumerate() {
                    let allowed = match kinds[id] {
                        TokenKind::Stop => can_stop,
                        TokenKind::Special => false,
                        TokenKind::Ordinary => json_state.advance(&table.pieces[id]).is_some(),
                    };
                    if !allowed {
                        *slot = f32::NEG_INFINITY;
                    }
                }
                let masked = Tensor::from_vec(v, vocab, &engine.device)?;
                logits_processor.sample(&masked)?
            } else {
                logits_processor.sample(&logits)?
            };
            tokens.push(next);
            if engine.stop_ids.contains(&next) {
                finish = TextLlmFinishReason::StopToken;
                break;
            }
            // Advance the JSON grammar by the accepted (non-stop) token's text — it was in the allowed
            // set, so this never rejects.
            if let Some(table) = constraint {
                if let Some(piece) = table.pieces.get(next as usize) {
                    if let Some(advanced) = json_state.advance(piece) {
                        json_state = advanced;
                    }
                }
            }
            generated.push(next);
            // 1-based step count = tokens emitted so far (monotone, ≤ total).
            on_progress(Progress::Step {
                current: generated.len() as u32,
                total,
            });
        }

        let text = engine
            .tokenizer
            .decode(&generated, true)
            .map(|t| t.trim().to_owned())
            .map_err(|e| CandleError::Msg(format!("prompt-refine: detokenize: {e}")))?;

        Ok(TextLlmOutput {
            text,
            generated_tokens: Some(generated.len() as u32),
            finish_reason: Some(finish),
        })
    }
}

impl TextLlm for PromptRefiner {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(PROMPT_REFINE_ID, req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<TextLlmOutput> {
        self.validate(req)?;
        // An already-cancelled request returns the typed `Canceled` before any inference (or weight
        // load) runs — the TextLlm pre-inference cancellation contract (sc-5500).
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        Ok(self.run(req, on_progress)?)
    }
}

/// The prompt-refine text-LLM descriptor (candle backend; not mac-only).
pub fn descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROMPT_REFINE_ID,
        family: PROMPT_REFINE_FAMILY,
        backend: "candle",
        capabilities: capabilities(),
    }
}

/// Construct a lazy candle prompt-refine provider. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a Llama-3.2-3B-Instruct snapshot (`config.json`, `tokenizer.json`,
/// `model-*.safetensors`). Adapters / quantization are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn TextLlm>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "prompt-refine expects a snapshot directory (config.json, tokenizer.json, \
                 model-*.safetensors), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle prompt-refine does not support LoRA/LoKr".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle prompt-refine does not support on-the-fly quantization".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(PromptRefiner {
        descriptor: descriptor(),
        root,
        device,
        engine: Mutex::new(None),
    }))
}

inventory::submit! {
    TextLlmRegistration { descriptor, load }
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn descriptor_advertises_prompt_refine_surface() {
        let d = descriptor();
        assert_eq!(d.id, PROMPT_REFINE_ID);
        assert_eq!(d.family, "llama");
        assert_eq!(d.backend, "candle");
        assert!(d.capabilities.supports_system_prompt);
        assert!(!d.capabilities.mac_only);
        assert_eq!(d.capabilities.max_new_tokens, prompt::MAX_NEW_TOKENS_CAP);
    }

    #[test]
    fn registers_and_resolves_as_candle_textllm() {
        // Lazy load: a nonexistent dir still resolves (weights are only touched at generate time).
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_textllm(PROMPT_REFINE_ID, &spec).expect("registered");
        assert_eq!(t.descriptor().id, PROMPT_REFINE_ID);
        assert_eq!(t.descriptor().backend, "candle");
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_rejects_adapters() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&spec).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn validate_rejects_empty_prompt_and_overlong_tokens() {
        let p = PromptRefiner {
            descriptor: descriptor(),
            root: "/nonexistent".into(),
            device: Device::Cpu,
            engine: Mutex::new(None),
        };
        // empty prompt
        assert!(p.validate(&TextLlmRequest::default()).is_err());
        // max_new_tokens over the advertised cap
        let mut req = TextLlmRequest {
            prompt: "rewrite this".to_owned(),
            ..Default::default()
        };
        req.sampling.max_new_tokens = prompt::MAX_NEW_TOKENS_CAP + 1;
        assert!(p.validate(&req).is_err());
    }
}
