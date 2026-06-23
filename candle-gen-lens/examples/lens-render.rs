//! sc-7544 acceptance: a real **end-to-end Lens Q4/Q8 render** on Blackwell at the **cap=80 packaging
//! baseline** (the multi-arch fatbin). The substitute for the story's "flux2_dev Q4 render" (flux2
//! rejects on-the-fly quant; its dev Q4 is an unstaged pre-quant snapshot). Unlike the synthetic DiT
//! parity test this drives the *full* pipeline — gpt-oss encoder → real text features → Q4/Q8 DiT
//! denoise → VAE — so every input is in-distribution; a coherent, all-finite image proves the
//! quantized GGUF `QMatMul` kernels work end-to-end on sm_120 at cap=80 (the broken sm_80-only build
//! rendered black/NaN).
//!
//! ```text
//! set CUDA_COMPUTE_CAP=80
//! cargo run -p candle-gen-lens --example lens-render --features cuda --release -- \
//!   --snapshot "C:\Users\…\models--microsoft--Lens-Turbo\snapshots\<hash>" \
//!   --quant q4 --prompt "a red apple on a wooden table" --out lens_q4.png
//! ```
//! `--snapshot` (or `LENS_SNAPSHOT`) is the Lens-Turbo snapshot root (text_encoder/ transformer/ vae/
//! tokenizer/). `--quant q4|q8|dense` (default q4). Exits non-zero if the render is degenerate
//! (constant/black image) — the failure mode of the unfixed quant kernels.

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource,
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
        .or_else(|| std::env::var("LENS_SNAPSHOT").ok())
        .ok_or(
            "pass --snapshot <dir> (or set LENS_SNAPSHOT) pointing at a Lens-Turbo snapshot root",
        )?;
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a red apple on a wooden table, soft daylight, sharp focus, highly detailed".into()
    });
    let quant = match arg(&args, "--quant")
        .or_else(|| std::env::var("LENS_QUANT").ok())
        .as_deref()
    {
        Some("q8") | Some("Q8") => Some(Quant::Q8),
        Some("dense") | Some("none") => None,
        _ => Some(Quant::Q4),
    };
    let qtag = match &quant {
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        _ => "dense",
    };
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("lens_{qtag}.png")));

    println!(
        "[lens-render] snapshot={snapshot}\n[lens-render] quant={qtag} {width}x{height} \
         steps={steps:?} seed={seed}\n[lens-render] prompt={prompt:?}"
    );

    candle_gen_lens::force_link();
    let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let gen = gen_core::registry::load("lens_turbo", &spec)?;
    println!(
        "[lens-render] resolved engine id={} backend={}",
        gen.descriptor().id,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        prompt,
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps,
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            print!("\r[lens-render] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[lens-render] decoding"),
    };
    let t0 = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let secs = t0.elapsed().as_secs_f32();
    let images = match output {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => return Err("expected images, got video".into()),
    };
    let img = &images[0];
    println!("\n[lens-render] {} image(s) in {secs:.1}s", images.len());

    // Degeneracy check: the unfixed (sm_80-only) quant kernels render a constant/black image. A real
    // render has spread across the channels.
    let (mut lo, mut hi, mut sum) = (255u8, 0u8, 0u64);
    for &p in &img.pixels {
        lo = lo.min(p);
        hi = hi.max(p);
        sum += p as u64;
    }
    let mean = sum as f64 / img.pixels.len().max(1) as f64;
    println!(
        "[lens-render] pixels: min={lo} max={hi} mean={mean:.1} ({}x{})",
        img.width, img.height
    );

    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or("invalid RGB buffer dimensions")?;
    buf.save(&out)?;
    println!("[lens-render] wrote {}", out.display());

    if lo == hi {
        return Err(format!(
            "DEGENERATE render: constant image (all pixels = {lo}). On Blackwell this is the unfixed \
             sm_80-only quant kernel no-op — the multi-arch fatbin is missing/broken."
        )
        .into());
    }
    println!(
        "[lens-render] OK — coherent, all-finite Lens {qtag} render on {}",
        gen.descriptor().backend
    );
    Ok(())
}
