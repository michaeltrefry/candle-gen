//! # candle-gen-joycaption
//!
//! JoyCaption captioner registration for [`candle-gen`](candle_gen), served by candle-llm's LLaVA
//! VLM provider. The candle (Windows/CUDA) sibling of `mlx-gen-joycaption`.
//!
//! The model — SigLIP-so400m vision tower + LLaVA projector + image splice + Llama-3.1 decode + the
//! LLaVA chat-input format — lives in [`candle-llm`](https://github.com/SceneWorks/candle-llm) as the
//! `candle-llava` [`core_llm::TextLlm`] vision provider (story 7634). This crate is the thin consumer
//! seam (story 7692, the candle twin of mlx's sc-7265): it keeps the SceneWorks caption **product
//! policy** (caption types / length templates / capability bounds / the default system prompt, in
//! [`prompt`]) and adapts the backend-neutral [`gen_core::Captioner`] contract the worker calls onto
//! the unified engine — building the prompt text + image request and streaming the provider's tokens
//! back.
//!
//! Unlike mlx-llm's dedicated `mlx-joycaption` provider (which injects JoyCaption's default system
//! prompt itself), candle-llm's `candle-llava` is a *generic* LLaVA provider that renders whatever
//! messages it is given through the model's own chat template. So this shim supplies the JoyCaption
//! system prompt as an explicit `System` message — it is SceneWorks product content, not model code —
//! and the engine owns the chat-input format, image-token splice, and decode.
//!
//! `backend = "candle"`, `mac_only = false`. Registered under
//! `"fancyfeast/llama-joycaption-beta-one-hf-llava"`.

pub mod prompt;

use candle_gen::gen_core::core_llm::{
    self, Content, ImageRef, LoadSpec as CoreLoadSpec, Message, ModelRequirements, Role, Sampling,
    StreamEvent, TextLlm, TextLlmRequest,
};
use candle_gen::gen_core::registry::CaptionerRegistration;
use candle_gen::gen_core::{
    CaptionFinishReason, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor, Error,
    LoadSpec, Progress, Result, WeightsSource,
};

use prompt::{build_prompt, capabilities, JOY_CAPTION_FAMILY, JOY_CAPTION_MODEL_ID, SYSTEM_PROMPT};

// Force-link the candle-llm engine so the `candle-llava` provider's `inventory::submit!` into
// core-llm's registry survives the linker — this crate resolves the provider through
// `core_llm::load_for_model_with` and never names another candle-llm symbol.
use candle_llm as _;

/// The JoyCaption captioner descriptor (candle backend; not mac-only).
pub fn descriptor() -> CaptionerDescriptor {
    CaptionerDescriptor {
        id: JOY_CAPTION_MODEL_ID,
        family: JOY_CAPTION_FAMILY,
        backend: "candle",
        capabilities: capabilities(),
    }
}

/// Construct a candle JoyCaption captioner. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a `fancyfeast/llama-joycaption-beta-one-hf-llava` snapshot (`config.json`, `tokenizer.json`,
/// `model-*.safetensors`). Adapters / quantization are rejected (not wired). The provider — and its
/// weights — are resolved eagerly, exactly as the worker's `load_captioner` call site expects (it
/// loads at job time with the real snapshot present, mirroring the mlx lane).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
    Ok(Box::new(load_joycaption(spec)?))
}

/// The concrete-typed loader behind [`load`].
pub fn load_joycaption(spec: &LoadSpec) -> Result<JoyCaptioner> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "joycaption expects a snapshot directory (config.json, tokenizer.json, \
                 model-*.safetensors), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(Error::Unsupported(
            "candle joycaption does not support LoRA/LoKr".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Unsupported(
            "candle joycaption does not support on-the-fly quantization".into(),
        ));
    }

    // Model-first resolution: the `candle-llava` vision provider's weightless `can_load` claims the
    // LLaVA snapshot; `with_vision()` disambiguates it from any text-only provider.
    let provider = core_llm::load_for_model_with(
        &CoreLoadSpec {
            source: root.to_string_lossy().into_owned(),
            quantize: None,
        },
        &ModelRequirements::default().with_vision(),
    )
    .map_err(map_core_err)?;

    Ok(JoyCaptioner {
        descriptor: descriptor(),
        provider,
    })
}

/// The JoyCaption captioner: a thin adapter from the [`gen_core::Captioner`] contract onto the
/// candle-llm `candle-llava` vision provider.
pub struct JoyCaptioner {
    descriptor: CaptionerDescriptor,
    provider: Box<dyn TextLlm>,
}

/// Fill the request prompt from the caption options when the caller left it empty — the JoyCaption
/// type/length template (or the custom-prompt override, which `build_prompt` returns as-is). Mirrors
/// `mlx-gen-joycaption`'s `normalized_request` so both backends accept an options-only request
/// (SceneWorks' worker sends `prompt = custom_prompt`, empty for the normal type/length flow).
fn normalized_request(req: &CaptionRequest) -> CaptionRequest {
    let mut out = req.clone();
    if out.prompt.trim().is_empty() {
        out.prompt = build_prompt(&out.options);
    }
    out
}

impl Captioner for JoyCaptioner {
    fn descriptor(&self) -> &CaptionerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &CaptionRequest) -> Result<()> {
        let req = normalized_request(req);
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, &req)
    }

    fn caption(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CaptionOutput> {
        let req = normalized_request(req);
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, &req)?;
        // An already-cancelled request returns the typed `Canceled` before any inference runs — the
        // captioner cancellation contract (sc-4895 / the testkit pre-cancel check).
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // The JoyCaption default system prompt (product policy) + one user turn carrying the image and
        // the (product-policy) prompt text. The LLaVA chat-input format and image-token splice are
        // applied inside the provider, so the consumer passes plain text + an image and nothing
        // model-specific. (mlx-llm's dedicated provider injects this system prompt itself; the generic
        // candle-llava provider does not, so the shim supplies it here.)
        let image = ImageRef::new(req.image.width, req.image.height, req.image.pixels.clone())
            .map_err(Error::Msg)?;
        let messages = vec![
            Message {
                role: Role::System,
                content: vec![Content::Text(SYSTEM_PROMPT.to_owned())],
                thinking: None,
            },
            Message {
                role: Role::User,
                content: vec![Content::Image(image), Content::Text(req.prompt.clone())],
                thinking: None,
            },
        ];

        // The provider polls its own `core_llm::CancelFlag`; bridge the gen-core flag onto it by
        // mirroring on each streamed token so the provider's next-step cancel check trips promptly.
        let core_cancel = core_llm::CancelFlag::new();
        let request = TextLlmRequest {
            messages,
            sampling: Sampling {
                temperature: req.sampling.temperature,
                top_p: req.sampling.top_p,
                // CaptionSampling exposes no top-k; disabled (0) matches the prior engine sampler.
                top_k: 0,
                repetition_penalty: req.sampling.repetition_penalty,
                repetition_context: req.sampling.repetition_context,
            },
            max_new_tokens: req.sampling.max_new_tokens,
            seed: req.sampling.seed,
            cancel: core_cancel.clone(),
            ..Default::default()
        };

        // Report one progress step per emitted token (the testkit's Progress-monotonicity check) and
        // bridge cancellation on every token.
        let total = req.sampling.max_new_tokens;
        let gen_cancel = req.cancel.clone();
        let bridge_cancel = core_cancel;
        let mut produced = 0u32;
        let mut on_event = |ev: StreamEvent| {
            if let StreamEvent::Token { .. } = ev {
                produced += 1;
                on_progress(Progress::Step {
                    current: produced,
                    total,
                });
                if gen_cancel.is_cancelled() {
                    bridge_cancel.cancel();
                }
            }
        };
        let out = self
            .provider
            .generate(&request, &mut on_event)
            .map_err(map_core_err)?;

        Ok(CaptionOutput {
            text: out.text.trim().to_owned(),
            generated_tokens: Some(out.usage.generated_tokens),
            finish_reason: out.finish_reason.map(map_finish),
        })
    }
}

/// Map a core-llm engine error onto the gen-core captioner error, preserving the **typed**
/// cancellation the contract (and the conformance suite) require.
fn map_core_err(e: core_llm::Error) -> Error {
    match e {
        core_llm::Error::Canceled => Error::Canceled,
        other => Error::Msg(other.to_string()),
    }
}

fn map_finish(f: core_llm::FinishReason) -> CaptionFinishReason {
    match f {
        // JoyCaption stops on an EOS/stop token or exhausts its token budget; it ships no content
        // filter, so that arm is unreachable but kept total (treated as a model-initiated stop).
        core_llm::FinishReason::Stop | core_llm::FinishReason::ContentFilter => {
            CaptionFinishReason::StopToken
        }
        core_llm::FinishReason::Length => CaptionFinishReason::MaxTokens,
        core_llm::FinishReason::Cancelled => CaptionFinishReason::Cancelled,
    }
}

inventory::submit! {
    CaptionerRegistration { descriptor, load }
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, CaptionOptions};

    #[test]
    fn normalize_builds_prompt_from_options_when_empty() {
        // sc-5189: the normal type/length flow sends no prompt; the provider must derive it (else
        // `caption` fails "prompt is required"), mirroring mlx-gen-joycaption.
        let req = CaptionRequest {
            options: CaptionOptions {
                caption_type: "Descriptive".to_owned(),
                caption_length: "long".to_owned(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(req.prompt.trim().is_empty());
        let normalized = normalized_request(&req);
        assert!(
            !normalized.prompt.trim().is_empty(),
            "empty prompt must be built from the caption options"
        );
    }

    #[test]
    fn normalize_preserves_an_explicit_prompt() {
        let req = CaptionRequest {
            prompt: "Describe the lighting only.".to_owned(),
            ..Default::default()
        };
        assert_eq!(
            normalized_request(&req).prompt,
            "Describe the lighting only."
        );
    }

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
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_rejects_adapters() {
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&spec).err().expect("err"),
            Error::Unsupported(_)
        ));
    }
}
