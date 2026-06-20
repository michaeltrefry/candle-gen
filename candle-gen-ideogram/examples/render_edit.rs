//! GPU validation harness for the candle Ideogram 4 **edit** lane (sc-6598): img2img / Remix
//! (`Reference`) and mask inpaint (`Reference` + `Mask`). Loads a bf16 snapshot, a source PNG, and an
//! optional mask, then renders one edited image and writes a PNG.
//!
//!   cargo run -p candle-gen-ideogram --example render_edit --features cuda -- \
//!       ideogram_4 /d/ideogram-4-bf16 "a green apple on a wooden table" source.png - 0.6 \
//!       768 768 0 42 out_img2img.png
//!
//! Args: <model_id> <snapshot_dir> <prompt> <source.png> <mask.png|-|centerbox> [strength=0.6]
//!       [width=0→source] [height=0→source] [steps=0→default] [seed=42] [out=render_edit.png]
//!
//! `mask` accepts `-` (no mask → plain img2img), a PNG path, or the literal `centerbox` (synthesize a
//! centered 50%×50% white box on black → inpaint just the middle).

use candle_gen::gen_core::{
    registry, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
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

/// Load an RGB8 PNG into a gen-core [`Image`].
fn load_image(path: &str) -> Result<Image, Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_rgb8();
    let (width, height) = (img.width(), img.height());
    Ok(Image {
        width,
        height,
        pixels: img.into_raw(),
    })
}

/// Synthesize a centered 50%×50% white box on black (RGB), `width×height`.
fn center_box_mask(width: u32, height: u32) -> Image {
    let (w, h) = (width as usize, height as usize);
    let (x0, x1) = (w / 4, w - w / 4);
    let (y0, y1) = (h / 4, h - h / 4);
    let mut pixels = vec![0u8; w * h * 3];
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) * 3;
            pixels[i] = 255;
            pixels[i + 1] = 255;
            pixels[i + 2] = 255;
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        .unwrap_or_else(|| "a green apple on a wooden table".into());
    let source_path = a.get(4).cloned().ok_or("missing <source.png>")?;
    let mask_arg = a.get(5).cloned().unwrap_or_else(|| "-".into());
    let strength: f32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.6);
    let mut width: u32 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut height: u32 = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0);
    let steps: u32 = a.get(9).and_then(|s| s.parse().ok()).unwrap_or(0);
    let seed: u64 = a.get(10).and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = a
        .get(11)
        .cloned()
        .unwrap_or_else(|| "render_edit.png".into());

    let source = load_image(&source_path)?;
    // Default the output grid to the source's dims (rounded down to a multiple of 16).
    if width == 0 {
        width = source.width / 16 * 16;
    }
    if height == 0 {
        height = source.height / 16 * 16;
    }
    println!(
        "model={model} snapshot={snapshot} {width}x{height} steps={steps} seed={seed} \
         strength={strength} source={source_path} ({}x{}) mask={mask_arg}",
        source.width, source.height
    );

    let mut conditioning = vec![Conditioning::Reference {
        image: source,
        strength: Some(strength),
    }];
    match mask_arg.as_str() {
        "-" => {}
        "centerbox" => conditioning.push(Conditioning::Mask {
            image: center_box_mask(width, height),
        }),
        path => conditioning.push(Conditioning::Mask {
            image: load_image(path)?,
        }),
    }

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.into()));
    let gen = registry::load(&model, &spec)?;
    println!(
        "loaded: id={} family={} backend={}",
        gen.descriptor().id,
        gen.descriptor().family,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        prompt: to_caption(&prompt),
        width,
        height,
        steps: if steps == 0 { None } else { Some(steps) },
        seed: Some(seed),
        count: 1,
        conditioning,
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    let mut last = 0u32;
    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            if current == 1 || current == total || current >= last + 4 {
                println!(
                    "  step {current}/{total}  ({:.1}s)",
                    t0.elapsed().as_secs_f32()
                );
                last = current;
            }
        }
        Progress::Decoding => println!("  decoding ({:.1}s)", t0.elapsed().as_secs_f32()),
    };

    let GenerationOutput::Images(images) = gen.generate(&req, &mut on_progress)? else {
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
