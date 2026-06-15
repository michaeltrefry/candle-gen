//! Wan-VACE controllable-video smoke driver (sc-5494) — resolves THIS crate's generator through
//! `gen_core::registry::load("wan_vace", …)`, builds a control clip + per-frame mask (+ optional
//! reference image) as one `Conditioning::ControlClip`, runs a real `generate` against a local
//! Wan2.1-VACE-14B diffusers snapshot, and writes each decoded frame to PNG.
//!
//! The control clip + mask are either loaded from a directory of PNGs (`--control-dir` / `--mask-dir`)
//! or synthesized: a moving bright square over a colour gradient, with a mask in one of three shapes
//! (`--mask`):
//!   - `center` (default): a centred box is regenerated (white) while the surround is kept (black) —
//!     tests that VACE honours the kept region while generating the masked region.
//!   - `extend`: the first ~⅓ of frames are kept (black), the rest regenerated (white) over a neutral
//!     grey span — the extend_clip / video_bridge shape (the kept frames should match the control).
//!   - `all`: every pixel is regenerated — the control video is pure `reactive` conditioning.
//!
//! ```text
//! cargo run --release --example vace_smoke --features cuda -- \
//!   --snapshot "C:\Users\…\models--Wan-AI--Wan2.1-VACE-14B-diffusers\snapshots\<hash>" \
//!   --prompt "a person walking through a sunlit garden, cinematic" \
//!   --mask center --width 512 --height 512 --frames 13 --steps 20 --seed 42 --out vace_smoke
//! ```

use std::path::PathBuf;

use candle_gen::gen_core::{
    self, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    ReplacementMode, WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn load_image(path: &std::path::Path) -> Result<Image> {
    let rgb = image::open(path)?.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Load every PNG in `dir`, sorted by name, as `Image`s.
fn load_png_dir(dir: &str) -> Result<Vec<Image>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no .png frames in {dir}").into());
    }
    paths.iter().map(|p| load_image(p)).collect()
}

/// A synthetic control frame: a colour gradient background + a bright square that translates across
/// the clip (so the control video carries real structure + motion).
fn synth_control(width: u32, height: u32, t: usize, frames: usize) -> Image {
    let (w, h) = (width as usize, height as usize);
    let mut px = vec![0u8; w * h * 3];
    let frac = if frames > 1 {
        t as f32 / (frames - 1) as f32
    } else {
        0.0
    };
    let sq = (w.min(h) / 4).max(1);
    let sx = ((w - sq) as f32 * frac) as usize; // square slides left→right over the clip
    let sy = (h - sq) / 2;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Gradient background.
            px[i] = (255 * x / w.max(1)) as u8;
            px[i + 1] = (255 * y / h.max(1)) as u8;
            px[i + 2] = 128;
            // Moving bright square.
            if x >= sx && x < sx + sq && y >= sy && y < sy + sq {
                px[i] = 255;
                px[i + 1] = 255;
                px[i + 2] = 255;
            }
        }
    }
    Image {
        width,
        height,
        pixels: px,
    }
}

/// A solid mask frame (`v` on every channel — 0 = keep the control frame, 255 = regenerate).
fn solid_mask(width: u32, height: u32, v: u8) -> Image {
    Image {
        width,
        height,
        pixels: vec![v; (width as usize) * (height as usize) * 3],
    }
}

/// A centred-box mask: the centre half-W × half-H rectangle is 255 (regenerate), the surround 0 (keep).
fn center_mask(width: u32, height: u32) -> Image {
    let (w, h) = (width as usize, height as usize);
    let (bw, bh) = (w / 2, h / 2);
    let (x0, y0) = ((w - bw) / 2, (h - bh) / 2);
    let mut px = vec![0u8; w * h * 3];
    for y in y0..y0 + bh {
        for x in x0..x0 + bw {
            let i = (y * w + x) * 3;
            px[i] = 255;
            px[i + 1] = 255;
            px[i + 2] = 255;
        }
    }
    Image {
        width,
        height,
        pixels: px,
    }
}

/// Per-frame NaN / degeneracy report for a decoded frame (the SVD lesson: a "passes" smoke that only
/// checks range/motion is fooled by noise — always print stats AND view a frame).
fn frame_stats(f: &Image) -> (f64, f64, u8, u8) {
    let n = f.pixels.len().max(1) as f64;
    let mean = f.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = f
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let min = *f.pixels.iter().min().unwrap_or(&0);
    let max = *f.pixels.iter().max().unwrap_or(&0);
    (mean, var.sqrt(), min, max)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("VACE_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set VACE_SNAPSHOT)")?;
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a person walking through a sunlit garden, cinematic, highly detailed".into()
    });
    let negative = arg(&args, "--negative");
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let guidance: Option<f32> = arg(&args, "--guidance").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let frames: usize = arg(&args, "--frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(13);
    let sampler = arg(&args, "--sampler");
    let mask_shape = arg(&args, "--mask").unwrap_or_else(|| "center".into());
    let reference = arg(&args, "--reference");
    let out = arg(&args, "--out").unwrap_or_else(|| "vace_smoke".into());

    // Control frames: a directory of PNGs, else synthesized. frames % 4 must be 1 (z16 VAE chunk).
    let control: Vec<Image> = match arg(&args, "--control-dir") {
        Some(dir) => load_png_dir(&dir)?,
        None => (0..frames)
            .map(|t| synth_control(width, height, t, frames))
            .collect(),
    };
    let n = control.len();
    if n % 4 != 1 {
        return Err(format!("control frame count must be 1 + 4·k (got {n})").into());
    }

    // Mask frames: a directory of PNGs, else synthesized per `--mask`.
    let mask: Vec<Image> = match arg(&args, "--mask-dir") {
        Some(dir) => load_png_dir(&dir)?,
        None => match mask_shape.as_str() {
            "all" => (0..n).map(|_| solid_mask(width, height, 255)).collect(),
            "extend" => {
                let keep = (n / 3).max(1);
                (0..n)
                    .map(|i| solid_mask(width, height, if i < keep { 0 } else { 255 }))
                    .collect()
            }
            // "center" (default): a centred box is regenerated, the surround kept.
            _ => (0..n).map(|_| center_mask(width, height)).collect(),
        },
    };

    let mut conditioning = vec![Conditioning::ControlClip {
        frames: control,
        mask,
        masking_strength: 1.0,
        start_frame: 0,
        mode: ReplacementMode::default(),
    }];
    if let Some(ref_path) = &reference {
        conditioning.push(Conditioning::Reference {
            image: load_image(std::path::Path::new(ref_path))?,
            strength: None,
        });
    }

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] {width}x{height} control_frames={n} mask={mask_shape} \
         reference={reference:?} steps={steps:?} guidance={guidance:?} sampler={sampler:?} seed={seed}\n\
         [smoke] prompt={prompt:?}"
    );

    candle_gen_wan::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("wan_vace", &spec)?;
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
        sampler,
        conditioning,
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
    let (frames_out, fps) = match output {
        GenerationOutput::Video { frames, fps, .. } => (frames, fps),
        GenerationOutput::Images(_) => return Err("expected video, got images".into()),
    };
    println!(
        "[smoke] {} frame(s) @ {fps}fps in {secs:.1}s",
        frames_out.len()
    );

    std::fs::create_dir_all(&out)?;
    let mut degenerate = true;
    for (i, f) in frames_out.iter().enumerate() {
        let (mean, std, min, max) = frame_stats(f);
        if std > 1.0 {
            degenerate = false;
        }
        println!("[smoke] frame {i:03}: mean={mean:.1} std={std:.1} min={min} max={max}");
        let buf = image::RgbImage::from_raw(f.width, f.height, f.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(PathBuf::from(&out).join(format!("frame_{i:03}.png")))?;
    }
    println!(
        "[smoke] wrote {} frames to {}/ ({}x{})",
        frames_out.len(),
        out,
        frames_out[0].width,
        frames_out[0].height
    );
    if degenerate {
        return Err(
            "DEGENERATE: every frame is ~constant (std ≤ 1) — open a frame to confirm".into(),
        );
    }
    println!("[smoke] non-degenerate ✓  — OPEN A FRAME to confirm coherent video (range+motion alone pass for noise)");
    Ok(())
}
