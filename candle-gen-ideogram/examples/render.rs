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

/// Wrap a plain prompt into Ideogram's minimal JSON caption (mirrors the worker's
/// `ideogram_caption::ensure_caption_prompt`). Ideogram 4 expects a JSON caption; a raw plain-text
/// prompt is out-of-distribution and stochastically renders the "Image blocked by safety filter"
/// placeholder. An already-caption prompt passes through unchanged.
fn to_caption(prompt: &str) -> String {
    let p = prompt.trim();
    if p.starts_with('{') && p.contains("compositional_deconstruction") {
        return p.to_string();
    }
    let q: String = p
        .chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            _ => vec![c],
        })
        .collect();
    format!(
        "{{\"high_level_description\": \"{q}\", \"compositional_deconstruction\": \
         {{\"background\": \"{q}\", \"elements\": []}}}}"
    )
}

/// Heuristic copy of the worker's `ideogram_caption::looks_like_placeholder`: detect Ideogram 4's
/// "Image blocked by safety filter" placeholder (a flat gray card with baked text) so the harness can
/// reseed past a residual one (sc-6501 — rare even with a JSON caption, but seed-dependent). Flat-gray
/// mean/std band + near-zero colorful fraction.
fn looks_like_placeholder(pixels: &[u8], width: u32, height: u32) -> bool {
    let expected = (width as usize) * (height as usize) * 3;
    if pixels.len() < 3 || pixels.len() != expected {
        return false;
    }
    let n = (pixels.len() / 3) as f64;
    let (mut sum, mut sum_sq, mut colorful) = (0.0f64, 0.0f64, 0usize);
    for px in pixels.chunks_exact(3) {
        let (r, g, b) = (px[0] as u16, px[1] as u16, px[2] as u16);
        if r.max(g).max(b) - r.min(g).min(b) > 24 {
            colorful += 1;
        }
        let luma = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
        sum += luma;
        sum_sq += luma * luma;
    }
    let mean = sum / n;
    let std = ((sum_sq / n) - mean * mean).max(0.0).sqrt();
    (colorful as f64 / n) <= 0.02 && (90.0..=165.0).contains(&mean) && (2.0..=30.0).contains(&std)
}

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

    // Reseed past a residual safety placeholder (sc-6501) so the harness emits a real image — the
    // same detect-and-recover the worker's macOS Ideogram path does.
    let caption = to_caption(&prompt);
    let mut img = None;
    for attempt in 0..6u64 {
        let seed_try = seed.wrapping_add(attempt);
        let req = GenerationRequest {
            prompt: caption.clone(),
            width,
            height,
            steps: if steps == 0 { None } else { Some(steps) },
            seed: Some(seed_try),
            count: 1,
            ..Default::default()
        };
        let GenerationOutput::Images(images) = gen.generate(&req, &mut on_progress)? else {
            return Err("expected images".into());
        };
        let candidate = images.into_iter().next().ok_or("no image")?;
        if !looks_like_placeholder(&candidate.pixels, candidate.width, candidate.height) {
            img = Some(candidate);
            break;
        }
        println!(
            "  Ideogram safety placeholder at seed {seed_try}; reseeding (attempt {})",
            attempt + 1
        );
        img = Some(candidate);
    }
    let img = img.ok_or("no image")?;
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
