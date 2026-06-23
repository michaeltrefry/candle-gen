//! GPU-validation harness for the candle Boogu provider: load a snapshot by engine id, render one
//! image from a prompt, write a PNG.
//!
//! ```text
//! cargo run -p candle-gen-boogu --example boogu-txt2img --features cuda --release -- \
//!   boogu_image D:\models\Boogu-Image-0.1-Base "a red apple on a wooden table" 1024 1024 0 42 out.png
//! ```
//! Arg order: <model_id> <snapshot_dir> <prompt> [width] [height] [steps(0=default)] [seed] [out.png]

use candle_gen::gen_core::{
    registry, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    candle_gen_boogu::force_link();

    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).cloned().unwrap_or_else(|| "boogu_image".into());
    let snapshot = a
        .get(2)
        .cloned()
        .unwrap_or_else(|| "D:/models/Boogu-Image-0.1-Base".into());
    let prompt = a
        .get(3)
        .cloned()
        .unwrap_or_else(|| "a red apple on a wooden table".into());
    let width: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let height: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let steps: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0); // 0 → engine default
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(42);
    let out = a
        .get(8)
        .cloned()
        .unwrap_or_else(|| "boogu_render.png".into());

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.into()));
    let gen = registry::load(&model, &spec)?;

    let req = GenerationRequest {
        prompt,
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: if steps == 0 { None } else { Some(steps) },
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            eprintln!("step {current}/{total}");
        }
        Progress::Decoding => eprintln!("decoding…"),
    };

    let GenerationOutput::Images(images) = gen.generate(&req, &mut on_progress)? else {
        return Err("expected images".into());
    };
    let img = images.into_iter().next().ok_or("no image")?;

    let buf =
        image::RgbImage::from_raw(img.width, img.height, img.pixels).ok_or("bad image buffer")?;
    buf.save(&out)?;
    eprintln!("wrote {out} ({}x{})", width, height);
    Ok(())
}
