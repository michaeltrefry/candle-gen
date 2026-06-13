//! sc-5114 — Lens schedule + CFG parity vs the vendor `LensImagePipeline` (diffusers
//! `FlowMatchEulerDiscreteScheduler`).
//!
//! **Weight-free** (CPU, no GPU): checks, against `scripts/dump_lens_schedule_golden.py`, that for
//! both the Turbo (4-step) and base (20-step) counts the Rust [`candle_gen_lens::schedule`]
//! reproduces (a) the shifted sigmas, (b) the per-step transformer timesteps (= shifted sigma; the
//! golden stores `sigma·1000`), (c) a single flow-match Euler `step`, and (d) the norm-rescaled CFG —
//! all near-bit (f32). Gated on the golden file existing (it needs the lens-venv to dump):
//!   LENS_SCHEDULE_GOLDENS — lens_schedule_golden.safetensors (default: .scratch/lens-schedule-goldens/…)
//! Run:  cargo test -p candle-gen-lens --test schedule_parity -- --nocapture

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen_lens::schedule::{cfg_rescale, euler_step, lens_sigmas, timesteps};

const LATENT: usize = 64; // 64×64 = 4096 seq_len (matches the dump's SEQ_LEN)

fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch");
    Ok(a.iter()
        .zip(&b)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs())))
}

fn peak_rel(a: &Tensor, b: &Tensor) -> Result<f32> {
    let bv = b.flatten_all()?.to_vec1::<f32>()?;
    let denom = bv.iter().fold(0f32, |m, &y| m.max(y.abs())).max(1e-12);
    Ok(max_abs(a, b)? / denom)
}

#[test]
fn lens_schedule_matches_reference() -> Result<()> {
    let goldens_path = std::env::var("LENS_SCHEDULE_GOLDENS").unwrap_or_else(|_| {
        ".scratch/lens-schedule-goldens/lens_schedule_golden.safetensors".to_string()
    });
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!(
            "SKIP: goldens not found at {goldens_path} (run scripts/dump_lens_schedule_golden.py)"
        );
        return Ok(());
    }
    let dev = Device::Cpu;
    let g = candle_gen::candle_core::safetensors::load(&goldens_path, &dev)?;

    for n in [4usize, 20] {
        let sigmas = lens_sigmas(n, LATENT, LATENT);

        // (a) shifted sigmas (n+1, trailing 0).
        let got_sigmas = Tensor::from_slice(&sigmas, sigmas.len(), &dev)?;
        let d_sig = max_abs(&got_sigmas, &g[&format!("sigmas_{n}")])?;

        // (b) per-step timesteps = shifted sigma; the golden stores sigma·1000.
        let ts = timesteps(&sigmas);
        let got_ts = Tensor::from_slice(ts, ts.len(), &dev)?;
        let want_ts = (g[&format!("timesteps_{n}")].clone() * (1.0 / 1000.0))?;
        let d_ts = max_abs(&got_ts, &want_ts)?;

        // (c) one flow-match Euler step at index 0.
        let got_step = euler_step(
            &g[&format!("step_in_{n}")],
            &g[&format!("step_noise_{n}")],
            &sigmas,
            0,
        )?;
        let d_step = peak_rel(&got_step, &g[&format!("step_out_{n}")])?;

        eprintln!(
            "n={n}: sigmas Δ={d_sig:.3e} | timesteps Δ={d_ts:.3e} | step peak_rel={d_step:.3e}"
        );
        assert!(d_sig < 1e-5, "n={n} sigmas Δ {d_sig:.3e}");
        assert!(d_ts < 1e-5, "n={n} timesteps Δ {d_ts:.3e}");
        assert!(d_step < 1e-4, "n={n} step peak_rel {d_step:.3e}");
    }

    // (d) norm-rescaled CFG (guidance 5.0).
    let got_cfg = cfg_rescale(&g["cfg_cond"], &g["cfg_uncond"], 5.0)?;
    let d_cfg = peak_rel(&got_cfg, &g["cfg_out"])?;
    eprintln!("cfg: peak_rel={d_cfg:.3e}");
    assert!(d_cfg < 1e-4, "cfg peak_rel {d_cfg:.3e}");
    eprintln!("ALL PASS");
    Ok(())
}
