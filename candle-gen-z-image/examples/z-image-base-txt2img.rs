//! Base (non-Turbo) Z-Image txt2img smoke driver (sc-8414) — the real-CFG sibling of
//! `z-image-txt2img.rs`. Resolves THIS crate's inventory-registered **base** generator via
//! `gen_core::registry::load("z_image", …)`, runs [`Generator::generate`] against a local
//! `Tongyi-MAI/Z-Image` (base) snapshot with classifier-free guidance + a negative prompt over the
//! static **shift=6.0** schedule, and writes each `gen_core::Image` to PNG.
//!
//! Build with the CUDA backend on the Windows/Blackwell box (GPU 1 only):
//!
//! ```text
//! set CUDA_VISIBLE_DEVICES=1
//! cargo run --release --example z-image-base-txt2img --features cuda -- \
//!   --snapshot "C:\Users\…\models--Tongyi-MAI--Z-Image\snapshots\<hash>" \
//!   --prompt "a photo of a rusty robot holding a lit candle" \
//!   --negative "blurry, low quality" --guidance 4.0 --steps 50 --seed 42 --out base.png
//! ```
//!
//! The snapshot must be the diffusers multi-component tree (`tokenizer/`, `text_encoder/`,
//! `transformer/`, `vae/`). Unlike Turbo, the base is undistilled, so `--guidance` (default 4.0) and
//! `--negative` are honored and the default is **50 steps**.

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
        .or_else(|| std::env::var("Z_IMAGE_BASE_SNAPSHOT").ok())
        .ok_or(
            "pass --snapshot <dir> (or set Z_IMAGE_BASE_SNAPSHOT) pointing at a Tongyi-MAI/Z-Image \
             (base) diffusers snapshot",
        )?;
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed"
            .into()
    });
    let negative = arg(&args, "--negative").unwrap_or_else(|| "blurry, low quality".into());
    let guidance: f32 = arg(&args, "--guidance")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.0);
    let steps: u32 = arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
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
        .unwrap_or_else(|| PathBuf::from("z_image_base_smoke.png"));

    println!(
        "[base smoke] snapshot={snapshot}\n[base smoke] {width}x{height} steps={steps} guidance={guidance} seed={seed} count={count}\n[base smoke] prompt={prompt:?} negative={negative:?}"
    );

    // Force-link the provider so its base `inventory::submit!` registration survives the linker.
    candle_gen_z_image::force_link();

    if args.iter().any(|a| a == "--no-accel") {
        candle_gen_z_image::set_accel_attn(false);
    }

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("z_image", &spec)?;
    println!(
        "[base smoke] resolved engine id={} backend={} supports_guidance={}",
        gen.descriptor().id,
        gen.descriptor().backend,
        gen.descriptor().capabilities.supports_guidance
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: Some(negative),
        guidance: Some(guidance),
        width,
        height,
        count,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            print!("\r[base smoke] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[base smoke] decoding"),
    };
    let t_call = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let gen_s = t_call.elapsed().as_secs_f32();
    let images = match output {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => return Err("expected images, got video".into()),
    };
    println!("[base smoke] {} image(s) in {gen_s:.1}s", images.len());

    for (i, img) in images.iter().enumerate() {
        let path = if images.len() == 1 {
            out.clone()
        } else {
            out.with_file_name(format!(
                "{}_{i}.png",
                out.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("z_image_base_smoke")
            ))
        };
        let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(&path)?;
        println!(
            "[base smoke] wrote {} ({}x{})",
            path.display(),
            img.width,
            img.height
        );
    }
    Ok(())
}
