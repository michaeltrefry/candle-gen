//! Smoke driver for the candle SenseNova-U1 T2I provider — encode the produced `gen_core::Image`
//! (RGB8) to PNG so a real GPU run can be eyeballed. Build with `--features cuda` on the Blackwell box.
//!
//! ```text
//! set SENSENOVA_SNAPSHOT=C:\Users\…\snapshots\<hash>
//! cargo run -p candle-gen-sensenova --features cuda --release --example sensenova-txt2img -- \
//!     "a fox reading a book by candlelight" out.png
//! ```

use std::error::Error;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

fn main() -> Result<(), Box<dyn Error>> {
    let snap = std::env::var("SENSENOVA_SNAPSHOT")
        .map_err(|_| "set SENSENOVA_SNAPSHOT to a SenseNova-U1-8B-MoT snapshot dir")?;
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "a fox reading a book by candlelight".to_string());
    let out_path = args
        .next()
        .unwrap_or_else(|| "sensenova-out.png".to_string());

    let spec = LoadSpec::new(WeightsSource::Dir(snap.into()));
    let generator = candle_gen_sensenova::load(&spec)?;

    let req = GenerationRequest {
        prompt,
        width: 512,
        height: 512,
        steps: Some(8),
        guidance: Some(4.0),
        seed: Some(0),
        count: 1,
        ..Default::default()
    };
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            eprintln!("step {current}/{total}");
        }
    };
    let out = generator.generate(&req, &mut on_progress)?;
    let GenerationOutput::Images(images) = out else {
        return Err("expected image output".into());
    };
    let img = images.into_iter().next().ok_or("no image produced")?;
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels)
        .ok_or("pixel buffer size mismatch")?;
    buf.save(&out_path)?;
    eprintln!("wrote {out_path}");
    Ok(())
}
