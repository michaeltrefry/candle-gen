//! SDXL txt2img smoke driver — exercises the full candle-gen seam end-to-end on a real GPU:
//! `gen_core::registry::load("sdxl", …)` resolves THIS crate's inventory-registered generator, runs
//! [`Generator::generate`] against a local SDXL snapshot, and writes each `gen_core::Image` to PNG.
//!
//! This is the human-eyeball check behind sc-3675 (the worker, not this example, owns asset writes in
//! production). Build with the CUDA backend on the Windows/Blackwell box:
//!
//! ```text
//! cargo run --release --example sdxl-txt2img --features cuda -- \
//!   --snapshot "C:\Users\…\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\<hash>" \
//!   --prompt "a photo of a rusty robot holding a lit candle" --steps 30 --seed 42 --out out.png
//! ```
//!
//! For the sc-6128 few-step **Lightning** eyeball, point `--snapshot` at a RealVisXL Lightning (or
//! SDXL-Lightning) checkpoint and select the sampler — CFG is forced off, so guidance is ignored:
//!
//! ```text
//! cargo run --release --example sdxl-txt2img --features cuda -- \
//!   --snapshot "…\RealVisXL Lightning snapshot…" --sampler lightning --steps 5 --out lightning.png
//! ```
//!
//! The snapshot must be the diffusers multi-component tree (`unet/`, `text_encoder/`,
//! `text_encoder_2/`); the model-agnostic CLIP tokenizers + fp16-VAE-fix resolve from the HF cache.

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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Snapshot dir: --snapshot, else $SDXL_SNAPSHOT.
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("SDXL_SNAPSHOT").ok())
        .ok_or(
            "pass --snapshot <dir> (or set SDXL_SNAPSHOT) pointing at an SDXL diffusers snapshot",
        )?;
    let prompt = arg(&args, "--prompt")
        .unwrap_or_else(|| "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed".into());
    let negative = arg(&args, "--negative").unwrap_or_default();
    let steps: u32 = arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let guidance: f32 = arg(&args, "--guidance")
        .and_then(|s| s.parse().ok())
        .unwrap_or(7.0);
    // `--sampler lightning` exercises the sc-6128 few-step Euler-trailing path (CFG-off) — the
    // RealVisXL Lightning eyeball: `--sampler lightning --steps 5`. Omitted ⇒ the DDIM default.
    let sampler = arg(&args, "--sampler");
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let count: u32 = arg(&args, "--count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    // `--repeat N` calls generate() N times on the SAME generator — exercises the sc-5037 UNet/VAE
    // cache: call 1 is cold (loads + caches), calls 2+ are warm (no UNet/VAE disk re-read). Per-call
    // wall-clock is printed so the warm speedup is visible; the last call's images are the ones saved.
    let repeat: u32 = arg(&args, "--repeat")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let out = arg(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sdxl_smoke.png"));

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] {width}x{height} steps={steps} guidance={guidance} sampler={sampler:?} seed={seed} count={count}\n[smoke] prompt={prompt:?}"
    );

    // Force-link the provider so its `inventory::submit!` registration survives the linker (we reach
    // it only through the gen_core registry below — see `candle_gen_sdxl::force_link`).
    candle_gen_sdxl::force_link();

    // `--no-flash` exercises the runtime toggle (sc-3674): turn fused flash-attention off even on a
    // flash-attn build (the worker drives this from the UI setting). No effect on a non-flash build.
    if args.iter().any(|a| a == "--no-flash") {
        candle_gen_sdxl::set_flash_attn(false);
    }
    // `--no-tiling` turns VAE tiling off (sc-4987) so a bench can compare the tiled vs monolithic VAE
    // decode peak VRAM at the same resolution. Default is on (tiles above 512² output).
    if args.iter().any(|a| a == "--no-tiling") {
        candle_gen_sdxl::set_vae_tiling(false);
    }

    // Resolve through the registry — proves the inventory seam (THIS crate's `submit!` is linked).
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let gen = gen_core::registry::load("sdxl", &spec)?;
    println!(
        "[smoke] resolved engine id={} backend={}",
        gen.descriptor().id,
        gen.descriptor().backend
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: if negative.is_empty() {
            None
        } else {
            Some(negative)
        },
        width,
        height,
        count,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(guidance),
        sampler: sampler.clone(),
        ..Default::default()
    };

    // Repeat generate() on the same generator; record each call's wall-clock to show the sc-5037 warm
    // speedup (cold call 1 loads+caches UNet/VAE; warm calls skip the disk re-read). Per Step event we
    // also stamp a mark: the first interval (call-start → step 1) carries model load + CUDA cold-start,
    // so it is NOT one of the inter-step deltas — every delta between consecutive marks is a warm,
    // steady-state denoise step. Marks are kept from the LAST call for the mean s/step figure.
    let mut call_secs: Vec<f32> = Vec::with_capacity(repeat as usize);
    let mut marks: Vec<std::time::Instant> = Vec::new();
    let mut images = Vec::new();
    for call in 0..repeat {
        marks.clear();
        let mut on_progress = |p: Progress| match p {
            Progress::Step { current, total } => {
                marks.push(std::time::Instant::now());
                print!(
                    "\r[smoke] call {}/{repeat} step {current}/{total}   ",
                    call + 1
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            Progress::Decoding => println!("\n[smoke] call {}/{repeat} decoding", call + 1),
        };
        let t_call = std::time::Instant::now();
        let output = gen.generate(&req, &mut on_progress)?;
        call_secs.push(t_call.elapsed().as_secs_f32());
        images = match output {
            GenerationOutput::Images(imgs) => imgs,
            GenerationOutput::Video { .. } => return Err("expected images, got video".into()),
        };
    }
    let gen_s = *call_secs.last().unwrap();
    if repeat > 1 {
        let warm = &call_secs[1..];
        let warm_mean = warm.iter().sum::<f32>() / warm.len() as f32;
        println!(
            "[smoke] per-call wall-clock: cold={:.2}s warm_mean={:.2}s (calls: {})",
            call_secs[0],
            warm_mean,
            call_secs
                .iter()
                .map(|s| format!("{s:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let deltas: Vec<f32> = marks
        .windows(2)
        .map(|w| (w[1] - w[0]).as_secs_f32())
        .collect();
    let mean_step = if deltas.is_empty() {
        0.0
    } else {
        deltas.iter().sum::<f32>() / deltas.len() as f32
    };
    let flash = cfg!(feature = "flash-attn") && candle_gen_sdxl::flash_attn_enabled();
    let tiling = candle_gen_sdxl::vae_tiling_enabled();
    println!(
        "[smoke] {} image(s) in {gen_s:.1}s total; steady-state {:.3}s/step (flash_attn={flash} vae_tiling={tiling})",
        images.len(),
        mean_step
    );

    // Sidecar with the run facts — the harness's cmd-subprocess stdout capture is lossy, so persist
    // the bench numbers to a file that can be read back.
    let _ = std::fs::write(
        out.with_extension("meta.txt"),
        format!(
            "engine=sdxl backend={} flash_attn={flash} vae_tiling={tiling}\n{width}x{height} steps={steps} guidance={guidance} seed={seed} count={count}\ngen_total_s={gen_s:.2} steady_per_step_s={mean_step:.3}\nimages={}\n",
            gen.descriptor().backend,
            images.len()
        ),
    );

    for (i, img) in images.iter().enumerate() {
        let path = if images.len() == 1 {
            out.clone()
        } else {
            out.with_file_name(format!(
                "{}_{i}.png",
                out.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sdxl_smoke")
            ))
        };
        let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(&path)?;
        println!(
            "[smoke] wrote {} ({}x{})",
            path.display(),
            img.width,
            img.height
        );
    }
    Ok(())
}
