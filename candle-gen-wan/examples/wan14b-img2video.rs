//! Wan2.2 **I2V-A14B** (dual-expert MoE, channel-concat) img2video smoke driver — resolves THIS
//! crate's generator through `gen_core::registry::load("wan2_2_i2v_14b", …)`, feeds a conditioning
//! image as a `Conditioning::Reference` (the first video frame), runs a real `generate` against a
//! local Wan2.2-I2V-A14B diffusers snapshot, and writes each decoded frame to PNG. (sc-5174.)
//!
//! ```text
//! cargo run --release --example wan14b-img2video --features cuda -- \
//!   --snapshot "C:\Users\…\models--Wan-AI--Wan2.2-I2V-A14B-Diffusers\snapshots\<hash>" \
//!   --image first_frame.png \
//!   --prompt "the camera pushes in as the subject turns to face us, cinematic" \
//!   --width 320 --height 320 --frames 17 --steps 20 --seed 42 --out wan14b_i2v_smoke
//! ```

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn load_image(path: &str) -> Result<Image> {
    let rgb = image::open(path)?.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("WAN14B_I2V_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set WAN14B_I2V_SNAPSHOT)")?;
    let image_path = arg(&args, "--image").ok_or("pass --image <path-to-first-frame.png>")?;
    let prompt = arg(&args, "--prompt")
        .unwrap_or_else(|| "the camera slowly pushes in, cinematic, highly detailed".into());
    let negative = arg(&args, "--negative");
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let guidance: Option<f32> = arg(&args, "--guidance").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(320);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(320);
    let frames: Option<u32> = arg(&args, "--frames").and_then(|s| s.parse().ok());
    let fps: Option<u32> = arg(&args, "--fps").and_then(|s| s.parse().ok());
    let sampler = arg(&args, "--sampler");
    let out = arg(&args, "--out").unwrap_or_else(|| "wan14b_i2v_smoke".into());

    let image = load_image(&image_path)?;
    println!(
        "[smoke] snapshot={snapshot}\n[smoke] image={image_path} ({}x{})\n[smoke] {width}x{height} \
         frames={frames:?} steps={steps:?} guidance={guidance:?} sampler={sampler:?} seed={seed}\n\
         [smoke] prompt={prompt:?}",
        image.width, image.height
    );

    candle_gen_wan::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("wan2_2_i2v_14b", &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={} modality={:?}",
        gen.descriptor().id,
        gen.descriptor().backend,
        gen.descriptor().modality
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: negative,
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps,
        guidance,
        frames,
        fps,
        sampler,
        conditioning: vec![Conditioning::Reference {
            image,
            strength: None,
        }],
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
