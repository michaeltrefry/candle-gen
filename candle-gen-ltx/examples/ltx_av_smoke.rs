//! LTX-2.3 synchronized **audio+video** smoke driver (sc-5495) — resolves THIS crate's generator
//! through `gen_core::registry::load("ltx_2_3_distilled", …)`, runs a real `generate` against a local
//! LTX-2.3 snapshot, and writes each decoded video frame to PNG **plus** the synchronized audio track
//! to a 16-bit PCM WAV. Prints per-frame + audio stats with a degeneracy guard (the SVD lesson: a
//! "passes" smoke that only checks range is fooled by noise — always view a frame AND listen).
//!
//! ```text
//! cargo run --release --example ltx_av_smoke --features cuda -- \
//!   --snapshot "C:\Users\…\models--Lightricks--LTX-2.3\snapshots\<hash>" \
//!   --gemma-dir "C:\Users\…\models--…gemma-3-12b…\snapshots\<hash>" \
//!   --prompt "a dog barking in a sunlit garden, cinematic" \
//!   --width 512 --height 320 --frames 25 --fps 24 --seed 42 --out ltx_av_smoke
//! ```

use std::io::Write;
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

/// Write interleaved-stereo f32 samples (`[-1,1]`) as a 16-bit PCM WAV.
fn write_wav(path: &PathBuf, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    let bits = 16u16;
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * block_align as u32;
    let data_len = (samples.len() * 2) as u32;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // PCM fmt chunk size
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn frame_stats(pixels: &[u8]) -> (f64, f64) {
    let n = pixels.len().max(1) as f64;
    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("LTX_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set LTX_SNAPSHOT)")?;
    if let Some(g) = arg(&args, "--gemma-dir") {
        std::env::set_var("LTX_GEMMA_DIR", g);
    }
    let prompt = arg(&args, "--prompt")
        .unwrap_or_else(|| "a dog barking in a sunlit garden, cinematic, highly detailed".into());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(320);
    let frames: u32 = arg(&args, "--frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let fps: u32 = arg(&args, "--fps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let out = arg(&args, "--out").unwrap_or_else(|| "ltx_av_smoke".into());

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] {width}x{height} frames={frames} fps={fps} seed={seed}\n\
         [smoke] prompt={prompt:?}"
    );

    candle_gen_ltx::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("ltx_2_3_distilled", &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={} modality={:?}",
        gen.descriptor().id,
        gen.descriptor().backend,
        gen.descriptor().modality
    );

    let req = GenerationRequest {
        prompt,
        width,
        height,
        count: 1,
        seed: Some(seed),
        frames: Some(frames),
        fps: Some(fps),
        ..Default::default()
    };

    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            print!("\r[smoke] step {current}/{total}   ");
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[smoke] decoding"),
    };
    let t0 = std::time::Instant::now();
    let output = gen.generate(&req, &mut on_progress)?;
    let secs = t0.elapsed().as_secs_f32();
    let (frames_out, out_fps, audio) = match output {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        GenerationOutput::Images(_) => return Err("expected video, got images".into()),
    };
    println!(
        "[smoke] {} frame(s) @ {out_fps}fps in {secs:.1}s",
        frames_out.len()
    );

    std::fs::create_dir_all(&out)?;
    let mut degenerate = true;
    for (i, f) in frames_out.iter().enumerate() {
        let (mean, std) = frame_stats(&f.pixels);
        if std > 1.0 {
            degenerate = false;
        }
        if i == 0 || i + 1 == frames_out.len() {
            println!("[smoke] frame {i:03}: mean={mean:.1} std={std:.1}");
        }
        let buf = image::RgbImage::from_raw(f.width, f.height, f.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(PathBuf::from(&out).join(format!("frame_{i:03}.png")))?;
    }
    println!("[smoke] wrote {} frames to {out}/", frames_out.len());

    // Audio.
    match &audio {
        Some(a) => {
            let n = a.samples.len().max(1) as f64;
            let mean = a.samples.iter().map(|&s| s as f64).sum::<f64>() / n;
            let std = (a
                .samples
                .iter()
                .map(|&s| (s as f64 - mean).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            let peak = a.samples.iter().fold(0f32, |m, &s| m.max(s.abs()));
            let dur = a.samples.len() as f64 / (a.channels.max(1) as f64 * a.sample_rate as f64);
            println!(
                "[smoke] audio: {} samples, {}ch @ {}Hz = {dur:.2}s  mean={mean:.4} std={std:.4} peak={peak:.3}",
                a.samples.len(),
                a.channels,
                a.sample_rate
            );
            let wav = PathBuf::from(&out).join("audio.wav");
            write_wav(&wav, &a.samples, a.sample_rate, a.channels)?;
            println!("[smoke] wrote {}", wav.display());
            if std < 1e-4 {
                return Err("DEGENERATE AUDIO: ~constant signal (std < 1e-4)".into());
            }
        }
        None => return Err("expected synchronized audio, got None".into()),
    }

    if degenerate {
        return Err("DEGENERATE VIDEO: every frame is ~constant (std ≤ 1)".into());
    }
    println!("[smoke] non-degenerate ✓  — OPEN A FRAME + PLAY audio.wav to confirm coherent A/V");
    Ok(())
}
