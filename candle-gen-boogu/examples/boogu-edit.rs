//! GPU-validation harness for the candle Boogu **Edit** lane (sc-7523): load the Edit snapshot, take
//! a reference PNG + an edit instruction, render one edited image, write a PNG.
//!
//! ```text
//! cargo run -p candle-gen-boogu --example boogu-edit --features cuda --release -- \
//!   D:\models\Boogu-Image-0.1-Edit ref.png "make it autumn" 0 0 0 42 edit_out.png
//! ```
//! Arg order: <snapshot_dir> <reference.png> <instruction> [width] [height] [steps(0=default)] \
//!            [seed] [out.png]
//!
//! `width`/`height` are the OUTPUT generation size; `0` (the default) keeps the reference's own
//! (snapped-to-multiple-of-16) dimensions, so the edit preserves resolution. The reference is
//! VAE-encoded at its own dims and must be a multiple of 16 per side; this harness snaps it down to
//! the nearest multiple of 16 (≥ 256) with a Lanczos resize so any input PNG works.

use candle_gen::gen_core::{
    registry, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};
use image::imageops::FilterType;

/// Snap `n` down to the nearest multiple of 16, floored at 256 (the engine's min size).
fn snap16(n: u32) -> u32 {
    (n - n % 16).max(256)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    candle_gen_boogu::force_link();

    let a: Vec<String> = std::env::args().collect();
    let snapshot = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| "D:/models/Boogu-Image-0.1-Edit".into());
    let ref_path = a.get(2).cloned().unwrap_or_else(|| "ref.png".into());
    let instruction = a.get(3).cloned().unwrap_or_else(|| "make it autumn".into());
    let arg_w: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let arg_h: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
    let steps: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0); // 0 → engine default
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = a.get(8).cloned().unwrap_or_else(|| "boogu_edit.png".into());

    // Load the reference PNG and snap it to a multiple-of-16 RGB8 buffer.
    let img = image::open(&ref_path)?.to_rgb8();
    let (rw, rh) = (snap16(img.width()), snap16(img.height()));
    let img = if (rw, rh) != (img.width(), img.height()) {
        eprintln!(
            "snapping reference {}x{} -> {rw}x{rh} (multiple of 16)",
            img.width(),
            img.height()
        );
        image::imageops::resize(&img, rw, rh, FilterType::Lanczos3)
    } else {
        img
    };
    let reference = Image {
        width: rw,
        height: rh,
        pixels: img.into_raw(),
    };

    // Output size: default to the reference dims (preserve resolution), else the CLI override.
    let width = if arg_w == 0 { rw } else { arg_w };
    let height = if arg_h == 0 { rh } else { arg_h };

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.into()));
    let gen = registry::load("boogu_image_edit", &spec)?;

    let req = GenerationRequest {
        prompt: instruction,
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: if steps == 0 { None } else { Some(steps) },
        conditioning: vec![Conditioning::Reference {
            image: reference,
            strength: None,
        }],
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => eprintln!("step {current}/{total}"),
        Progress::Decoding => eprintln!("decoding…"),
    };

    let GenerationOutput::Images(images) = gen.generate(&req, &mut on_progress)? else {
        return Err("expected images".into());
    };
    let result = images.into_iter().next().ok_or("no image")?;

    let buf = image::RgbImage::from_raw(result.width, result.height, result.pixels)
        .ok_or("bad image buffer")?;
    buf.save(&out)?;
    eprintln!("wrote {out} ({width}x{height})");
    Ok(())
}
