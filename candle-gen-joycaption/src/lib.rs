//! # candle-gen-joycaption
//!
//! The **JoyCaption** image-captioning provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-joycaption`. JoyCaption is a `LlavaForConditionalGeneration`:
//! a SigLIP-so400m vision tower ([`vision`]) → a gelu-MLP multimodal projector → a Llama-3.1-8B
//! decoder ([`language`]). It implements the backend-neutral gen-core
//! [`Captioner`](candle_gen::gen_core::Captioner) contract (image → caption text), not `Generator`.
//!
//! There is **no** candle-transformers reference: the contract needs the SigLIP **`-2`** hidden
//! state (`vision_feature_layer = -2`, `"full"` 729 tokens) and a Llama that consumes pre-spliced
//! `inputs_embeds`, neither of which candle-transformers exposes — so the tower, the projector, the
//! image-feature splice, and the autoregressive decoder are all ported from scratch on `candle_nn`.
//!
//! **Caption (sc-3699):** [`JoyCaptioner::caption`] preprocesses the image → SigLIP `-2` features →
//! projector → splices the 729 projected rows over the expanded image-token placeholders → runs the
//! Llama-3.1 decoder autoregressively (greedy or temperature/top-p with a small repetition penalty)
//! → detokenizes. Registered under `"fancyfeast/llama-joycaption-beta-one-hf-llava"`.
//!
//! **Dtype:** the whole assembly runs **bf16** (the checkpoint's native dtype); logits upcast to f32
//! for sampling. `backend = "candle"`, `mac_only = false`.

pub mod language;
pub mod prompt;
pub mod vision;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::registry::CaptionerRegistration;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor, LoadSpec, Progress,
    WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use language::{LlamaConfig, LlamaDecoder, LlavaProjector};
use prompt::{
    build_chat_text, capabilities, expand_image_tokens, BEGIN_OF_TEXT_TOKEN_ID,
    DEFAULT_MAX_CONTEXT_TOKENS, JOY_CAPTION_FAMILY, JOY_CAPTION_MODEL_ID, PAD_TOKEN_ID,
};
use vision::{SiglipImageProcessor, SiglipVisionConfig, SiglipVisionTower};

/// The JoyCaption checkpoint is bf16 (SigLIP2 + Llama-3.1); the whole assembly runs at this dtype.
const MODEL_DTYPE: DType = DType::BF16;

/// The loaded, weight-bearing model components + tokenizer (cached after the first caption).
struct Engine {
    vision: SiglipVisionTower,
    projector: LlavaProjector,
    llama: LlamaDecoder,
    processor: SiglipImageProcessor,
    tokenizer: TextTokenizer,
}

impl Engine {
    fn load(root: &std::path::Path, device: &Device) -> CResult<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root)
            .map_err(|e| {
                CandleError::Msg(format!("joycaption: read snapshot {}: {e}", root.display()))
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(CandleError::Msg(format!(
                "joycaption: no .safetensors in snapshot dir {}",
                root.display()
            )));
        }
        // SAFETY: mmap of read-only weight files; the standard candle loading path.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, MODEL_DTYPE, device)? };

        let vision = SiglipVisionTower::new(
            SiglipVisionConfig::default(),
            vb.pp("vision_tower").pp("vision_model"),
        )?;
        let projector = LlavaProjector::new(vb.pp("multi_modal_projector"))?;
        let llama = LlamaDecoder::new(LlamaConfig::default(), vb.pp("language_model"))?;

        let tokenizer = TextTokenizer::from_file(
            root.join("tokenizer.json"),
            TokenizerConfig {
                max_length: DEFAULT_MAX_CONTEXT_TOKENS,
                pad_token_id: PAD_TOKEN_ID as i32,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("joycaption: load tokenizer: {e}")))?;

        Ok(Self {
            vision,
            projector,
            llama,
            processor: SiglipImageProcessor::default(),
            tokenizer,
        })
    }
}

pub struct JoyCaptioner {
    descriptor: CaptionerDescriptor,
    root: PathBuf,
    device: Device,
    engine: Mutex<Option<Arc<Engine>>>,
}

impl JoyCaptioner {
    fn engine(&self) -> CResult<Arc<Engine>> {
        let mut guard = self
            .engine
            .lock()
            .expect("joycaption engine cache mutex poisoned");
        if let Some(e) = guard.as_ref() {
            return Ok(e.clone());
        }
        let engine = Arc::new(Engine::load(&self.root, &self.device)?);
        *guard = Some(engine.clone());
        Ok(engine)
    }

    fn run(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<CaptionOutput> {
        let engine = self.engine()?;

        // Vision: preprocess → SigLIP -2 features → project into the Llama hidden size.
        let pixels = engine
            .processor
            .preprocess(&req.image, &self.device, MODEL_DTYPE)?;
        let vision_features = engine.vision.forward(&pixels)?; // [1, 729, 1152]
        let projected = engine.projector.forward(&vision_features)?; // [1, 729, 4096]

        // Prompt: wrap the (already-constructed) request prompt in the Llama-3 chat template, map to
        // ids (no auto special tokens — the template carries them as literal strings), prepend the
        // <|begin_of_text|> BOS the HF chat template starts with, then expand the single image marker
        // into 729 placeholders.
        let chat = build_chat_text(&req.prompt);
        let mut ids: Vec<i64> = engine
            .tokenizer
            .encode_ids(&chat, false)
            .map_err(|e| CandleError::Msg(format!("joycaption: tokenize: {e}")))?
            .into_iter()
            .map(|i| i as i64)
            .collect();
        ids.insert(0, BEGIN_OF_TEXT_TOKEN_ID);
        let ids = expand_image_tokens(&ids);

        // Embed ids, splice the projected image rows over the image-token positions.
        let input_ids = Tensor::from_vec(
            ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            (1, ids.len()),
            &self.device,
        )?;
        let embeds = engine.llama.embed(&input_ids)?;
        let spliced = language::splice_image_features(&embeds, &ids, &projected)?;

        // Autoregressive generation, reporting one progress step per emitted token.
        let total = req.sampling.max_new_tokens;
        let mut produced = 0u32;
        let mut on_token = || {
            produced += 1;
            on_progress(Progress::Step {
                current: produced,
                total,
            });
        };
        let gen = engine.llama.generate_from_embeds(
            &ids,
            &spliced,
            req.sampling,
            &req.cancel,
            &mut on_token,
        )?;

        let toks: Vec<u32> = gen.token_ids.iter().map(|&i| i as u32).collect();
        let text = engine
            .tokenizer
            .decode(&toks, true)
            .map(|t| t.trim().to_owned())
            .map_err(|e| CandleError::Msg(format!("joycaption: detokenize: {e}")))?;

        Ok(CaptionOutput {
            text,
            generated_tokens: Some(gen.token_ids.len() as u32),
            finish_reason: Some(gen.finish_reason),
        })
    }
}

impl Captioner for JoyCaptioner {
    fn descriptor(&self) -> &CaptionerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &CaptionRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(JOY_CAPTION_MODEL_ID, req)?;
        Ok(())
    }

    fn caption(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<CaptionOutput> {
        self.validate(req)?;
        // An already-cancelled request returns the typed `Canceled` before any inference (or even a
        // weight load) runs — the captioner cancellation contract (sc-4895).
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        Ok(self.run(req, on_progress)?)
    }
}

/// The JoyCaption captioner descriptor (candle backend; not mac-only).
pub fn descriptor() -> CaptionerDescriptor {
    CaptionerDescriptor {
        id: JOY_CAPTION_MODEL_ID,
        family: JOY_CAPTION_FAMILY,
        backend: "candle",
        capabilities: capabilities(),
    }
}

/// Construct a lazy candle JoyCaption captioner. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `fancyfeast/llama-joycaption-beta-one-hf-llava` snapshot (`config.json`,
/// `tokenizer.json`, `model-*.safetensors`). Adapters / quantization are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Captioner>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "joycaption expects a snapshot directory (config.json, tokenizer.json, \
                 model-*.safetensors), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle joycaption does not support LoRA/LoKr".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle joycaption does not support on-the-fly quantization".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(JoyCaptioner {
        descriptor: descriptor(),
        root,
        device,
        engine: Mutex::new(None),
    }))
}

inventory::submit! {
    CaptionerRegistration { descriptor, load }
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    #[test]
    fn descriptor_advertises_joycaption_surface() {
        let d = descriptor();
        assert_eq!(d.id, JOY_CAPTION_MODEL_ID);
        assert_eq!(d.family, "joycaption");
        assert_eq!(d.backend, "candle");
        assert!(d.capabilities.supports_custom_prompt);
        assert!(!d.capabilities.mac_only);
        assert_eq!(d.capabilities.max_new_tokens, 1024);
        assert!(d.capabilities.caption_types.contains(&"Straightforward"));
        assert!(d.capabilities.caption_lengths.contains(&"medium-length"));
    }

    #[test]
    fn registers_and_resolves_as_candle_captioner() {
        // Lazy load: a nonexistent dir still resolves (weights are only touched at caption time).
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let c = registry::load_captioner(JOY_CAPTION_MODEL_ID, &spec).expect("registered");
        assert_eq!(c.descriptor().id, JOY_CAPTION_MODEL_ID);
        assert_eq!(c.descriptor().backend, "candle");
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
}
