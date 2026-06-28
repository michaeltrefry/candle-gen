//! SD3.5 txt2img smoke driver — exercises the full candle-gen seam end-to-end on a real GPU:
//! `gen_core::registry::load(<id>, …)` resolves THIS crate's inventory-registered generator, runs
//! [`Generator::generate`] against a local SD3.5 diffusers snapshot, and writes each `gen_core::Image`
//! to PNG.
//!
//! The human-eyeball check behind sc-7877 (the worker, not this example, owns asset writes in
//! production). The SD3.5 weights are gated (Stability Community License, HF-account-bound), so this
//! is run only where a snapshot is already present. Build with the CUDA backend on Windows/Blackwell:
//!
//! ```text
//! cargo run --release --example sd3-txt2img --features cuda -- \
//!   --snapshot "C:\…\stable-diffusion-3.5-large" \
//!   --variant large --prompt "a rusty robot holding a lit candle" --steps 28 --cfg 4.0 --out out.png
//! # Turbo (distilled, 4-step, CFG-off):
//! cargo run --release --example sd3-txt2img --features cuda -- \
//!   --snapshot "C:\…\stable-diffusion-3.5-large-turbo" --variant turbo --steps 4 --out turbo.png
//! ```
//!
//! The snapshot must be the diffusers multi-component tree (`tokenizer*/`, `text_encoder*/`,
//! `transformer/`, `vae/`).

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("SD35_SNAPSHOT").ok())
        .ok_or(
            "pass --snapshot <dir> (or set SD35_SNAPSHOT) pointing at an SD3.5 diffusers snapshot",
        )?;
    let variant = arg(&args, "--variant").unwrap_or_else(|| "large".into());
    let model_id = match variant.as_str() {
        "turbo" | "large-turbo" | "large_turbo" => candle_gen_sd3::MODEL_ID_TURBO,
        _ => candle_gen_sd3::MODEL_ID,
    };
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed"
            .into()
    });
    let negative = arg(&args, "--negative");
    let default_steps = if model_id == candle_gen_sd3::MODEL_ID_TURBO {
        4
    } else {
        28
    };
    let steps: u32 = arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_steps);
    let cfg: Option<f32> = arg(&args, "--cfg").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let count: u32 = arg(&args, "--count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sd3_smoke.png"));

    println!(
        "[smoke] snapshot={snapshot} id={model_id}\n[smoke] {width}x{height} steps={steps} \
         seed={seed} count={count}\n[smoke] prompt={prompt:?}"
    );

    // Force-link the provider so its `inventory::submit!` registration survives the linker (we reach
    // it only through the gen_core registry below).
    candle_gen_sd3::force_link();

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load(model_id, &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={}",
        gen.descriptor().id,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: negative,
        width,
        height,
        count,
        seed: Some(seed),
        steps: Some(steps),
        guidance: cfg,
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            print!("\r[smoke] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[smoke] decoding"),
    };

    let t = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let secs = t.elapsed().as_secs_f32();
    let images = match output {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => return Err("expected images, got video".into()),
    };
    println!("[smoke] {} image(s) in {secs:.1}s", images.len());

    for (i, img) in images.iter().enumerate() {
        let path = if images.len() == 1 {
            out.clone()
        } else {
            out.with_file_name(format!(
                "{}_{i}.png",
                out.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sd3_smoke")
            ))
        };
        let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(&path)?;
        println!(
            "[smoke] wrote {} ({}x{})",
            path.display(),
            img.width,
            img.height
        );
    }
    Ok(())
}
