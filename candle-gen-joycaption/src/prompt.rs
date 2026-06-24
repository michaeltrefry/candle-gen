//! JoyCaption caption **product policy** — the SceneWorks caption prompt map, the default system
//! prompt, the capability surface, and trigger-word post-processing. Pure string logic, backend
//! agnostic.
//!
//! The model's chat-input format (the LLaVA/Llama-3 chat wrapper) and the special token ids used to
//! live here too; they moved into candle-llm's `candle-llava` provider when this crate was repointed
//! onto the unified engine (sc-7692). What remains is product content only.

use candle_gen::gen_core::{CaptionCapabilities, CaptionOptions};

pub const JOY_CAPTION_MODEL_ID: &str = "fancyfeast/llama-joycaption-beta-one-hf-llava";
pub const JOY_CAPTION_FAMILY: &str = "joycaption";

/// JoyCaption's default system prompt. Passed by the shim as an explicit `System` message (the
/// generic `candle-llava` provider does not inject a model-specific system prompt of its own).
pub const SYSTEM_PROMPT: &str = "You are a helpful image captioner.";

pub const JOY_NAME_OPTION: &str =
    "If there is a person/character in the image you must refer to them as {name}.";

pub const CAPTION_TYPES: &[&str] = &[
    "Descriptive",
    "Descriptive (Casual)",
    "Straightforward",
    "Stable Diffusion Prompt",
    "MidJourney",
    "Danbooru tag list",
    "e621 tag list",
    "Rule34 tag list",
    "Booru-like tag list",
    "Art Critic",
    "Product Listing",
    "Social Media Post",
];

pub const CAPTION_LENGTHS: &[&str] = &["any", "very short", "short", "medium-length", "long"];

const PROMPT_TEMPLATES: &[(&str, [&str; 3])] = &[
    (
        "Descriptive",
        [
            "Write a detailed description for this image.",
            "Write a detailed description for this image in {word_count} words or less.",
            "Write a {length} detailed description for this image.",
        ],
    ),
    (
        "Descriptive (Casual)",
        [
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
    ),
    (
        "Straightforward",
        [
            "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
            "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
            "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
        ],
    ),
    (
        "Stable Diffusion Prompt",
        [
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
            "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        ],
    ),
    (
        "MidJourney",
        [
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
    ),
    (
        "Danbooru tag list",
        [
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
        ],
    ),
    (
        "e621 tag list",
        [
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
        ],
    ),
    (
        "Rule34 tag list",
        [
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
        ],
    ),
    (
        "Booru-like tag list",
        [
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
    ),
    (
        "Art Critic",
        [
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
    ),
    (
        "Product Listing",
        [
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
    ),
    (
        "Social Media Post",
        [
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    ),
];

pub fn capabilities() -> CaptionCapabilities {
    CaptionCapabilities {
        caption_types: CAPTION_TYPES.to_vec(),
        caption_lengths: CAPTION_LENGTHS.to_vec(),
        supports_custom_prompt: true,
        supports_low_vram: true,
        min_image_size: 1,
        max_image_size: 8192,
        max_prompt_chars: 4000,
        max_name_chars: 120,
        max_extra_options: 16,
        max_extra_option_chars: 500,
        max_trigger_words: 32,
        max_trigger_word_chars: 120,
        max_new_tokens: 1024,
        // candle is the Windows/CUDA lane → NOT mac-only (unlike the mlx provider).
        mac_only: false,
    }
}

pub fn build_prompt(options: &CaptionOptions) -> String {
    let custom = options.custom_prompt.trim();
    if !custom.is_empty() {
        return custom.to_owned();
    }
    let caption_length = options.caption_length.as_str();
    let template_index = if caption_length == "any" {
        0
    } else if !caption_length.is_empty() && caption_length.chars().all(|c| c.is_ascii_digit()) {
        1
    } else {
        2
    };
    let mut prompt = templates_for(&options.caption_type)[template_index].to_owned();
    if !options.extra_options.is_empty() {
        prompt.push(' ');
        prompt.push_str(&options.extra_options.join(" "));
    }
    prompt
        .replace("{name}", name_or_placeholder(options))
        .replace("{length}", caption_length)
        .replace("{word_count}", caption_length)
}

pub fn apply_trigger_words(caption: &str, trigger_words: &[String]) -> String {
    let cleaned = caption.split_whitespace().collect::<Vec<_>>().join(" ");
    let lower_caption = cleaned.to_lowercase();
    let mut parts: Vec<String> = trigger_words
        .iter()
        .map(|word| word.trim())
        .filter(|word| !word.is_empty())
        .filter(|word| !lower_caption.contains(&word.to_lowercase()))
        .map(ToOwned::to_owned)
        .collect();
    if !cleaned.is_empty() {
        parts.push(cleaned);
    }
    parts.join(", ")
}

fn templates_for(caption_type: &str) -> &'static [&'static str; 3] {
    PROMPT_TEMPLATES
        .iter()
        .find(|(kind, _)| *kind == caption_type)
        .map(|(_, templates)| templates)
        .unwrap_or(&PROMPT_TEMPLATES[0].1)
}

fn name_or_placeholder(options: &CaptionOptions) -> &str {
    if options.name_input.is_empty() {
        "{NAME}"
    } else {
        &options.name_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(kind: &str, length: &str) -> CaptionOptions {
        CaptionOptions {
            caption_type: kind.to_owned(),
            caption_length: length.to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn prompt_defaults_match_sceneworks() {
        assert_eq!(
            build_prompt(&CaptionOptions::default()),
            "Write a long detailed description for this image."
        );
        assert_eq!(
            build_prompt(&options("Descriptive", "any")),
            "Write a detailed description for this image."
        );
        assert_eq!(
            build_prompt(&options("Descriptive", "85")),
            "Write a detailed description for this image in 85 words or less."
        );
    }

    #[test]
    fn custom_prompt_overrides_template() {
        let prompt = build_prompt(&CaptionOptions {
            custom_prompt: "  Describe only the outfit.  ".to_owned(),
            ..Default::default()
        });
        assert_eq!(prompt, "Describe only the outfit.");
    }

    #[test]
    fn trigger_words_are_prepended_only_when_missing() {
        let trigger_words = vec!["mika_token".to_owned(), "hat".to_owned()];
        assert_eq!(
            apply_trigger_words("A portrait of Mika wearing a hat.", &trigger_words),
            "mika_token, A portrait of Mika wearing a hat."
        );
        assert_eq!(
            apply_trigger_words("   ", &trigger_words),
            "mika_token, hat"
        );
    }
}
