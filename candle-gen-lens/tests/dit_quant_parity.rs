//! sc-5117 acceptance gate: the candle Lens DiT Q4/Q8 quantization must stay within tolerance of the
//! dense bf16 DiT — the candle twin of `mlx-gen-lens`'s `dit_quant_parity.rs` (sc-3175).
//!
//! Loads the real `transformer/` weights at **bf16** (the production DiT dtype the quant path runs
//! at), runs the dense forward over the `dit_parity` golden's synthetic inputs, then quantizes the DiT
//! ([`LensTransformer::quantize`]) and re-runs — asserting Q8 is near-lossless and Q4 stays coherent
//! vs the dense bf16 DiT. This is a self-consistent load-time-quant gate (no torch reference needed):
//! a real quantization bug (wrong block type, transposed `QMatMul`, dropped bias) would crater the
//! cosine, while the expected 4-bit floor on a *single* full forward leaves Q4 coherent.
//!
//! Heavy + machine-specific (loads the ~8 GB bf16 DiT and needs the GPU for the GGUF `QMatMul`
//! kernels), so it is **gated** on the same env vars as `dit_parity` and skips cleanly when they are
//! unset (CPU CI has neither weights nor a GPU):
//!   LENS_DIT_DIR     — the Lens-Turbo `transformer` snapshot dir (config.json + model-*.safetensors)
//!   LENS_DIT_GOLDENS — lens_dit_golden.safetensors (default: .scratch/lens-dit-goldens/…)
//! Run with the `cuda` feature (absolute goldens path — cargo test cwd is the crate dir):
//!   cargo test -p candle-gen-lens --features cuda --test dit_quant_parity -- --nocapture

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::Quant;
use candle_gen_lens::transformer::{LensDitConfig, LensTransformer};

/// Cosine similarity over all elements (flattened), computed in f64 on CPU.
fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch in cosine");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    Ok((dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32)
}

/// Peak relative error `max|a-b| / max|b|`, in f64 on CPU.
fn peak_rel(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_diff = 0f64;
    let mut max_b = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((*x - *y).abs() as f64);
        max_b = max_b.max((*y).abs() as f64);
    }
    Ok((max_diff / max_b.max(1e-12)) as f32)
}

#[test]
fn lens_dit_quant_matches_dense() -> Result<()> {
    let dit_dir = match std::env::var("LENS_DIT_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set LENS_DIT_DIR to the Lens-Turbo transformer snapshot dir");
            return Ok(());
        }
    };
    let goldens_path = std::env::var("LENS_DIT_GOLDENS")
        .unwrap_or_else(|_| ".scratch/lens-dit-goldens/lens_dit_golden.safetensors".to_string());
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!(
            "SKIP: goldens not found at {goldens_path} (run scripts/dump_lens_dit_golden.py)"
        );
        return Ok(());
    }

    let device = candle_gen::default_device()
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!("device: {device:?}");

    let cfg = LensDitConfig::lens();

    // Golden inputs (the same synthetic batch the reference / dit_parity uses), as **bf16** — the
    // production DiT dtype the quant path runs at.
    let g = candle_gen::candle_core::safetensors::load(&goldens_path, &device)?;
    let bf16 = |k: &str| -> Result<Tensor> { g[k].to_dtype(DType::BF16) };
    let grid = g["grid_fhw"].to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let (frame, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
    let timestep = g["timestep"].to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
    let hidden = bf16("hidden_states")?;
    let feats: Vec<Tensor> = (0..cfg.num_text_layers)
        .map(|i| bf16(&format!("feat_{i}")))
        .collect::<Result<_>>()?;
    let txt_len = feats[0].dim(1)?;
    eprintln!("grid=({frame},{h},{w}) timestep={timestep:.5} txt_len={txt_len}");

    let run = |dit: &LensTransformer| -> Result<Tensor> {
        dit.forward(&hidden, &feats, None, timestep, frame, h, w)
    };

    // Dense bf16 reference, then quantize the *same* instance to Q8 and re-run.
    eprintln!("loading transformer (bf16)…");
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dit_dir)
        .expect("read transformer dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in {dit_dir}");
    // SAFETY: mmap of read-only weight files (the standard candle loading path).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::BF16, &device)? };

    let mut dit = LensTransformer::new(&cfg, vb.clone())?;
    let dense = run(&dit)?;
    dit.quantize(Quant::Q8)?;
    let q8 = run(&dit)?;
    let q8_cos = cosine(&q8, &dense)?;
    eprintln!(
        "Q8 vs dense bf16: cosine {q8_cos:.6}  peak_rel {:.3e}",
        peak_rel(&q8, &dense)?
    );

    // Fresh dense DiT → Q4.
    let mut dit = LensTransformer::new(&cfg, vb)?;
    dit.quantize(Quant::Q4)?;
    let q4 = run(&dit)?;
    let q4_cos = cosine(&q4, &dense)?;
    eprintln!(
        "Q4 vs dense bf16: cosine {q4_cos:.6}  peak_rel {:.3e}",
        peak_rel(&q4, &dense)?
    );

    // Q8 is near-lossless. Q4 is lossier — a single full DiT forward quantizes `img_in`/`txt_in`/
    // `proj_out` + the attention projections + SwiGLU MLPs across all 48 blocks, so the per-forward
    // cosine sits at the 4-bit floor (in line with the Q4 precedent across the codebase) — coherent,
    // not collapsed. The denoise runs many such forwards; the e2e render stays coherent (the registry
    // exposes Q4 for the memory-constrained tier).
    assert!(
        q8_cos > 0.99,
        "Q8 DiT cosine {q8_cos:.6} ≤ 0.99 — not near-lossless"
    );
    assert!(
        q4_cos > 0.80,
        "Q4 DiT cosine {q4_cos:.6} ≤ 0.80 — collapsed, not a coherent quantization"
    );
    eprintln!("ALL PASS");
    Ok(())
}
