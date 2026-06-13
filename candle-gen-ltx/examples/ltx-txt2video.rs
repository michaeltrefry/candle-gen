//! LTX-2.3 (distilled 22B) txt2video smoke driver — resolves THIS crate's inventory-registered
//! generator through `gen_core::registry::load("ltx_2_3_distilled", …)`, runs a real `generate`
//! against a local LTX-2.3 snapshot + a Gemma-3-12B encoder snapshot, and writes each decoded frame
//! to PNG. The human-eyeball check behind sc-3698.
//!
//! ```text
//! cargo run --release --example ltx-txt2video --features cuda -- \
//!   --snapshot "C:\Users\…\models--Lightricks--LTX-2.3\snapshots\<hash>" \
//!   --gemma "C:\Users\…\models--google--gemma-3-12b-it\snapshots\<hash>" \
//!   --prompt "a fluffy cat walking across a sunny garden, cinematic" \
//!   --width 704 --height 480 --frames 49 --seed 42 --out ltx_smoke
//! ```

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
        .or_else(|| std::env::var("LTX_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set LTX_SNAPSHOT)")?;
    // The Gemma encoder is a separate snapshot; the provider reads LTX_GEMMA_DIR.
    if let Some(g) = arg(&args, "--gemma") {
        std::env::set_var("LTX_GEMMA_DIR", g);
    }
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a fluffy cat walking across a sunny garden, gentle camera pan, cinematic, highly detailed"
            .into()
    });
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(704);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(480);
    let frames: Option<u32> = arg(&args, "--frames").and_then(|s| s.parse().ok());
    let fps: Option<u32> = arg(&args, "--fps").and_then(|s| s.parse().ok());
    let out = arg(&args, "--out").unwrap_or_else(|| "ltx_smoke".into());

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] {width}x{height} frames={frames:?} seed={seed}\n\
         [smoke] prompt={prompt:?}"
    );

    candle_gen_ltx::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("ltx_2_3_distilled", &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={} modality={:?}",
        gen.descriptor().id,
        gen.descriptor().backend,
        gen.descriptor().modality
    );

    let req = GenerationRequest {
        prompt,
        width,
        height,
        count: 1,
        seed: Some(seed),
        frames,
        fps,
        sampler: Some("rectified-flow".into()),
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
    let t0 = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let secs = t0.elapsed().as_secs_f32();
    let (frames, fps) = match output {
        GenerationOutput::Video { frames, fps, .. } => (frames, fps),
        GenerationOutput::Images(_) => return Err("expected video, got images".into()),
    };
    println!("[smoke] {} frame(s) @ {fps}fps in {secs:.1}s", frames.len());

    std::fs::create_dir_all(&out)?;
    for (i, f) in frames.iter().enumerate() {
        let buf = image::RgbImage::from_raw(f.width, f.height, f.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(PathBuf::from(&out).join(format!("frame_{i:03}.png")))?;
    }
    println!(
        "[smoke] wrote {} frames to {}/ ({}x{})",
        frames.len(),
        out,
        frames[0].width,
        frames[0].height
    );
    Ok(())
}
