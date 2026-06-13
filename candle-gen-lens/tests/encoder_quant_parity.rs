//! sc-5111 — gpt-oss encoder **Q4/Q8** parity vs the bf16 reference.
//!
//! Loads the `text_encoder` with the MoE experts transcoded MXFP4 → GGUF `Q8_0` / `Q4_0` (the ~12 GB
//! path — [`GptOssTextEncoder::new_quant`]) and asserts the captured hidden states still track the
//! **bf16** reference captures in the same golden the dense sc-5108 gate uses (`gptoss_goldens`, dumped
//! from the HF `LensGptOssEncoder`). The dense candle bf16 encoder's own worst cosine vs this golden is
//! ~0.9993 (sc-5108, the candle-CUDA-vs-torch bf16 floor); quantizing the experts must stay near that
//! floor — Q8 near-lossless, Q4 coherent — confirming the transcode is wired correctly (a wrong
//! pack / nibble order / requant would collapse the cosine).
//!
//! Each loads **only** its quantized encoder (and the test prints the load wall-clock), so under
//! `--test-threads=1` the process peaks at the ~12 GB it advertises (run `nvidia-smi` alongside to
//! confirm) rather than the ~40 GB dense bf16 stack — the whole point of the story.
//!
//! Gated on env vars; skips cleanly when unset. Run with the `cuda` feature, single-threaded so the
//! two encoders load sequentially (not 2× resident at once):
//!   cargo test -p candle-gen-lens --features cuda --test encoder_quant_parity -- --nocapture --test-threads=1
//!   LENS_TEXT_ENCODER_DIR — the Lens `text_encoder` snapshot dir (config.json + model-*.safetensors)
//!   LENS_GOLDENS          — gptoss_goldens.safetensors (default: .scratch/gptoss-goldens/…)

use candle_gen::candle_core::quantized::GgmlDType;
use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_lens::text_encoder::{Config, GptOssTextEncoder, DEFAULT_SELECTED_LAYERS};

/// Cosine similarity over all elements (flattened), computed in f32 on CPU.
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

/// Build the encoder at `quant`, run it over the golden ids, and return the worst capture cosine vs
/// the bf16 reference (`cap_{layer}`). `None` skips cleanly (missing env / goldens).
fn worst_capture_cosine(quant: GgmlDType) -> Result<Option<f32>> {
    let Ok(te_dir) = std::env::var("LENS_TEXT_ENCODER_DIR") else {
        eprintln!("SKIP: set LENS_TEXT_ENCODER_DIR to the Lens text_encoder snapshot dir");
        return Ok(None);
    };
    let goldens_path = std::env::var("LENS_GOLDENS")
        .unwrap_or_else(|_| ".scratch/gptoss-goldens/gptoss_goldens.safetensors".to_string());
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!("SKIP: goldens not found at {goldens_path} (run scripts/dump_gptoss_goldens.py)");
        return Ok(None);
    }

    let device = candle_gen::default_device()
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
    eprintln!("device: {device:?}  quant: {quant:?}");

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&te_dir)
        .expect("read text_encoder dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors in {te_dir}");

    // Load + transcode the experts to Q4/Q8 (attention/router/embeddings stay bf16). Time it: the
    // on-device unpack should make this far quicker than the old per-nibble host loop.
    let t0 = std::time::Instant::now();
    // SAFETY: mmap of read-only weight files (the standard candle loading path).
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::BF16, &device)? };
    let encoder = GptOssTextEncoder::new_quant(&Config::gpt_oss_20b(), vb, Some(quant))?;
    eprintln!(
        "loaded {quant:?} encoder in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        eprintln!(
            "GPU mem (MiB) resident after {quant:?} load:\n{}",
            String::from_utf8_lossy(&o.stdout).trim()
        );
    }

    let goldens = candle_gen::candle_core::safetensors::load(&goldens_path, &device)?;
    let input_ids = goldens["input_ids"].to_dtype(DType::U32)?;
    let seq = input_ids.dim(0)?;
    let input_ids = input_ids.reshape((1, seq))?;

    let caps = encoder.capture(&input_ids, &DEFAULT_SELECTED_LAYERS)?;
    let mut worst = 1f32;
    for (cap, &layer) in caps.iter().zip(DEFAULT_SELECTED_LAYERS.iter()) {
        let Some(golden) = goldens.get(&format!("cap_{layer:02}")) else {
            continue;
        };
        let c = cosine(&cap.squeeze(0)?, golden)?;
        eprintln!("  {quant:?} capture[L{layer:>2}]: cosine={c:.6}");
        // Guard the `min` against NaN (a broken quant path makes the captures NaN, and `f32::min`
        // silently *ignores* NaN — which would let the assertions below pass on garbage).
        assert!(
            c.is_finite(),
            "{quant:?} capture[L{layer}] cosine is non-finite — broken quant path"
        );
        worst = worst.min(c);
    }
    eprintln!("worst {quant:?} capture cosine: {worst:.6}  (dense bf16 floor ~0.9993)");
    Ok(Some(worst))
}

#[test]
fn encoder_q8_matches_reference() -> Result<()> {
    // Q8 is near-lossless: it must essentially hold the dense bf16 floor (a hair under, for the added
    // 8-bit block rounding).
    if let Some(worst) = worst_capture_cosine(GgmlDType::Q8_0)? {
        assert!(
            worst > 0.995,
            "Q8 worst capture cosine {worst:.6} ≤ 0.995 — quant degraded well past the bf16 floor"
        );
        eprintln!("Q8 PASS");
    }
    Ok(())
}

#[test]
fn encoder_q4_matches_reference() -> Result<()> {
    // Q4 (the ~12 GB target) is lossier but must stay coherent — the captures still track the bf16
    // reference, not collapse.
    if let Some(worst) = worst_capture_cosine(GgmlDType::Q4_0)? {
        assert!(
            worst > 0.95,
            "Q4 worst capture cosine {worst:.6} ≤ 0.95 — not a coherent quantization"
        );
        eprintln!("Q4 PASS");
    }
    Ok(())
}
