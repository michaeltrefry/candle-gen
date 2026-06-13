//! JoyCaption captioning smoke driver — resolves THIS crate's inventory-registered captioner through
//! `gen_core::registry::load_captioner(…)`, runs a real `caption` against a local JoyCaption snapshot
//! and a real input image, and prints the generated caption. The human-eyeball check behind sc-3699.
//!
//! ```text
//! cargo run --release --example joycaption-caption --features cuda -- \
//!   --snapshot "C:\Users\…\models--fancyfeast--llama-joycaption-beta-one-hf-llava\snapshots\<hash>" \
//!   --image "C:\path\to\photo.jpg" \
//!   --type Descriptive --length long --max-new-tokens 256 --temperature 0.6 --seed 42
//! ```

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, CaptionOptions, CaptionRequest, CaptionSampling, LoadSpec, Progress, WeightsSource,
};
use candle_gen_joycaption::prompt::{build_prompt, JOY_CAPTION_MODEL_ID};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("JOYCAPTION_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set JOYCAPTION_SNAPSHOT)")?;
    let image_path = arg(&args, "--image").ok_or("pass --image <file>")?;

    let caption_type = arg(&args, "--type").unwrap_or_else(|| "Descriptive".into());
    let caption_length = arg(&args, "--length").unwrap_or_else(|| "long".into());
    let custom_prompt = arg(&args, "--prompt").unwrap_or_default();
    let max_new_tokens: u32 = arg(&args, "--max-new-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let temperature: f32 = arg(&args, "--temperature")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.6);
    let top_p: f32 = arg(&args, "--top-p")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.9);
    let seed: Option<u64> = arg(&args, "--seed").and_then(|s| s.parse().ok());

    // Decode the input image to RGB8 → gen_core::Image.
    let rgb = image::open(&image_path)?.to_rgb8();
    let (w, h) = rgb.dimensions();
    let image = gen_core::Image {
        width: w,
        height: h,
        pixels: rgb.into_raw(),
    };

    let options = CaptionOptions {
        caption_type,
        caption_length,
        custom_prompt,
        ..Default::default()
    };
    let prompt = build_prompt(&options);
    println!(
        "[smoke] snapshot={snapshot}\n[smoke] image={image_path} ({w}x{h})\n[smoke] prompt={prompt:?}"
    );

    candle_gen_joycaption::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let captioner = gen_core::registry::load_captioner(JOY_CAPTION_MODEL_ID, &spec)?;
    println!(
        "[smoke] resolved captioner id={} backend={}",
        captioner.descriptor().id,
        captioner.descriptor().backend
    );

    let req = CaptionRequest {
        image,
        prompt,
        options,
        sampling: CaptionSampling {
            temperature,
            top_p,
            max_new_tokens,
            seed,
        },
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };

    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            print!("\r[smoke] token {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    };
    let t0 = std::time::Instant::now();
    let out = captioner.caption(&req, &mut on_progress)?;
    let secs = t0.elapsed().as_secs_f32();

    println!(
        "\n[smoke] {} token(s) in {secs:.1}s (finish: {:?})",
        out.generated_tokens.unwrap_or(0),
        out.finish_reason
    );
    println!("\n=== caption ===\n{}\n===============", out.text);
    Ok(())
}
