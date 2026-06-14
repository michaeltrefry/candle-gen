//! Real-weights smoke for the candle SenseNova-U1 **understanding** surface (sc-5501): VQA
//! ([`T2iModel::vqa`]) + Document-Studio interleave ([`T2iModel::interleave_gen`]). Drives the
//! off-registry `load_understanding` entry the worker uses. Build with `--features cuda` on the
//! Blackwell box; the dense base model is ~35 GB.
//!
//! ```text
//! set SENSENOVA_SNAPSHOT=C:\Users\…\snapshots\<hash>
//! cargo run -p candle-gen-sensenova --features cuda --release --example sensenova-understanding
//! ```

use std::error::Error;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::CancelFlag;
use candle_gen_sensenova::{
    load_understanding, tensor_to_image, Sampler, T2iOptions, INTERLEAVE_SYSTEM_MESSAGE,
};

/// A synthetic `[3,H,W]` RGB gradient in `[0,1]` on CPU (the engine relocates it to the model device):
/// red ramps left→right, green ramps top→bottom, blue constant — identifiable colors for VQA.
fn gradient(w: usize, h: usize) -> Result<Tensor, Box<dyn Error>> {
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = x as f32 / w as f32;
            data[h * w + y * w + x] = y as f32 / h as f32;
            data[2 * h * w + y * w + x] = 0.5;
        }
    }
    Ok(Tensor::from_vec(data, (3, h, w), &Device::Cpu)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let snap = std::env::var("SENSENOVA_SNAPSHOT")
        .map_err(|_| "set SENSENOVA_SNAPSHOT to a SenseNova-U1-8B-MoT snapshot dir")?;
    eprintln!("loading {snap} …");
    let (model, tok) = load_understanding(std::path::Path::new(&snap))?;
    eprintln!("loaded.");

    // ---- VQA: image + question → text answer ----
    let img = gradient(512, 512)?;
    eprintln!("running VQA …");
    let answer = model.vqa(
        &tok,
        "What colors appear in this image?",
        std::slice::from_ref(&img),
        64,
        Sampler::Greedy,
    )?;
    eprintln!("VQA answer: {answer:?}");
    assert!(!answer.trim().is_empty(), "VQA answer should be non-empty");

    // ---- Interleave: prompt → ordered text + generated images (short think-mode rollout) ----
    let opts = T2iOptions {
        cfg_scale: 4.0,
        img_cfg_scale: 1.0,
        num_steps: 8,
        timestep_shift: 3.0,
        seed: 42,
        think_mode: true,
        ..Default::default()
    };
    let cancel = CancelFlag::new();
    eprintln!("running interleave …");
    let out = model.interleave_gen(
        &tok,
        "Generate an illustration of a single red circle on a white background, then briefly describe it.",
        &[],
        512,
        512,
        &opts,
        INTERLEAVE_SYSTEM_MESSAGE,
        512,
        4,
        &cancel,
    )?;
    eprintln!("interleave text: {:?}", out.text);
    eprintln!("interleave images: {}", out.images.len());
    assert!(
        !out.images.is_empty(),
        "interleave should produce >= 1 image"
    );
    let img0 = tensor_to_image(&out.images[0])?;
    let buf = image::RgbImage::from_raw(img0.width, img0.height, img0.pixels)
        .ok_or("pixel buffer size mismatch")?;
    buf.save("sensenova-interleave-0.png")?;
    eprintln!(
        "wrote sensenova-interleave-0.png ({}x{})",
        img0.width, img0.height
    );
    eprintln!("SMOKE OK");
    Ok(())
}
