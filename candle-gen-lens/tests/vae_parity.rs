//! sc-5113 acceptance gate: the Lens VAE decode shim must match the vendor `_decode`.
//!
//! Loads the real `vae/` weights (the cached `microsoft/Lens-Turbo` snapshot, a diffusers
//! `AutoencoderKLFlux2`) **as f32** into the shared `candle_gen_flux2::Flux2Vae`, decodes a synthetic
//! DiT-shaped latent via the Lens shim, and compares the decoded pixels against
//! `scripts/dump_lens_vae_golden.py`. f32 both sides → a tight correctness gate for the bn-stats
//! de-normalization + 2×2 unpatchify + AutoencoderKL decode.
//!
//! Heavy + machine-specific (needs the VAE weights + GPU), so it is **gated** on env vars and skips
//! cleanly when they are unset:
//!   LENS_VAE_DIR     — the Lens-Turbo `vae` snapshot dir (config.json + diffusion_pytorch_model.safetensors)
//!   LENS_VAE_GOLDENS — lens_vae_golden.safetensors (default: .scratch/lens-vae-goldens/…)
//! Run with the `cuda` feature (absolute goldens path — cargo test cwd is the crate dir):
//!   cargo test -p candle-gen-lens --features cuda --test vae_parity -- --nocapture

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_lens::vae::{decode, Flux2Vae};

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
fn lens_vae_matches_reference() -> Result<()> {
    let vae_dir = match std::env::var("LENS_VAE_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set LENS_VAE_DIR to the Lens-Turbo vae snapshot dir");
            return Ok(());
        }
    };
    let goldens_path = std::env::var("LENS_VAE_GOLDENS")
        .unwrap_or_else(|_| ".scratch/lens-vae-goldens/lens_vae_golden.safetensors".to_string());
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!(
            "SKIP: goldens not found at {goldens_path} (run scripts/dump_lens_vae_golden.py)"
        );
        return Ok(());
    }

    let device = candle_gen::default_device()
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!("device: {device:?}");

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&vae_dir)
        .expect("read vae dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in {vae_dir}");
    // SAFETY: mmap of read-only weight files (the standard candle loading path).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device)? };
    let vae = Flux2Vae::new(vb)?;

    let g = candle_gen::candle_core::safetensors::load(&goldens_path, &device)?;
    let grid = g["grid_hw"].to_dtype(DType::U32)?.to_vec1::<u32>()?;
    let (lh, lw) = (grid[0] as usize, grid[1] as usize);
    let latents = g["latents"].to_dtype(DType::F32)?;
    eprintln!("grid=({lh},{lw}) latents={:?}", latents.dims());

    let out = decode(&vae, &latents, lh, lw)?;
    let golden = g["out"].to_dtype(DType::F32)?;
    assert_eq!(out.dims(), golden.dims(), "decoded shape mismatch");
    let pr = peak_rel(&out, &golden)?;
    let cos = cosine(&out, &golden)?;
    eprintln!(
        "vae decode: peak_rel={pr:.3e} cosine={cos:.7} out={:?}",
        out.dims()
    );

    // Single conv-decode pass (no 48-block accumulation); CUDA-vs-CPU f32 should be tight.
    assert!(cos > 0.999, "vae decode cosine {cos:.7} ≤ 0.999");
    assert!(pr < 2e-2, "vae decode peak_rel {pr:.3e} ≥ 2e-2");
    eprintln!("ALL PASS");
    Ok(())
}
