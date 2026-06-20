//! GPU validation harness for the candle Ideogram 4 provider (sc-6596). Loads a bf16 snapshot (see
//! `convert_fp8`), renders one image via the registered generator, and writes a PNG.
//!
//!   cargo run -p candle-gen-ideogram --example render --features cuda -- \
//!       ideogram_4 /d/ideogram-4-bf16 "a neon city skyline at dusk" 1024 1024 0 42 out.png
//!
//! Args: <model_id> <snapshot_dir> <prompt> [width=1024] [height=1024] [steps=0→default] [seed=42]
//!       [out=render.png]

use candle_gen::gen_core::{
    registry, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Force-link the crate so its `inventory::submit!` registrations aren't dead-stripped (the
    // example otherwise references no symbol from the lib).
    candle_gen_ideogram::force_link();
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).cloned().unwrap_or_else(|| "ideogram_4".into());
    let snapshot = a
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/d/ideogram-4-bf16".into());
    let prompt = a
        .get(3)
        .cloned()
        .unwrap_or_else(|| "a neon city skyline at dusk, ultra detailed".into());
    let width: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let height: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let steps: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0); // 0 → model default
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = a.get(8).cloned().unwrap_or_else(|| "render.png".into());

    println!("model={model} snapshot={snapshot} {width}x{height} steps={steps} seed={seed}");
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.into()));
    let gen = registry::load(&model, &spec)?;
    println!(
        "loaded: id={} family={} backend={}",
        gen.descriptor().id,
        gen.descriptor().family,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        prompt,
        width,
        height,
        steps: if steps == 0 { None } else { Some(steps) },
        seed: Some(seed),
        count: 1,
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    let mut last = 0u32;
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            if current == 1 || current == total || current >= last + 8 {
                println!(
                    "  step {current}/{total}  ({:.1}s)",
                    t0.elapsed().as_secs_f32()
                );
                last = current;
            }
        } else if let Progress::Decoding = p {
            println!("  decoding ({:.1}s)", t0.elapsed().as_secs_f32());
        }
    };

    let result = gen.generate(&req, &mut on_progress)?;
    let GenerationOutput::Images(images) = result else {
        return Err("expected images".into());
    };
    let img = images.into_iter().next().ok_or("no image")?;
    println!(
        "rendered {}x{} in {:.1}s",
        img.width,
        img.height,
        t0.elapsed().as_secs_f32()
    );
    let buf =
        image::RgbImage::from_raw(img.width, img.height, img.pixels).ok_or("bad image buffer")?;
    buf.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
