//! Offline one-time provisioning: convert the gated `ideogram-ai/ideogram-4-fp8` checkpoint to a
//! candle-readable **bf16** snapshot (the candle analogue of `tools/convert_ideogram4_to_mlx.py`,
//! minus the MLX-quant step — and Python-free).
//!
//! Ideogram 4 ships **fp8 weight-only**: every Linear weight is `float8_e4m3fn` plus a per-output-row
//! f32 `{key}_scale`; biases / norms / the VAE are bf16/int already. This dequantizes
//! `w_bf16 = (fp8.f32 * scale[:,None]).bf16()`, drops the folded `_scale` tensors, and re-emits each
//! component as a single bf16 `model.safetensors` with the key names preserved (the loaders own the
//! module mapping). Tokenizer / scheduler / config / model_index are copied verbatim.
//!
//! Run (CPU; no CUDA needed):
//!   cargo run -p candle-gen-ideogram --example convert_fp8 -- <src_fp8_dir> <out_bf16_dir>

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::{save, MmapedSafetensors};
use candle_gen::candle_core::{DType, Device, Result, Tensor};

/// Components carrying weights to convert (scheduler/tokenizer are copied verbatim).
const WEIGHT_COMPONENTS: &[&str] = &[
    "transformer",
    "unconditional_transformer",
    "text_encoder",
    "vae",
];
const COPY_DIRS: &[&str] = &["scheduler", "tokenizer"];
const COPY_FILES: &[&str] = &["model_index.json", "README.md", "LICENSE.md"];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let src = PathBuf::from(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("/d/ideogram-4-fp8"),
    );
    let out = PathBuf::from(
        args.get(2)
            .map(String::as_str)
            .unwrap_or("/d/ideogram-4-bf16"),
    );
    let only = args.get(3).cloned(); // optional single-component filter
    println!("src: {}\nout: {}", src.display(), out.display());
    std::fs::create_dir_all(&out).map_err(wrap)?;

    // Dequant on the default device — the GPU when built `--features cuda` (elementwise fp8→bf16 over
    // ~9.3B params is far faster there than debug-mode CPU); falls back to CPU otherwise.
    let dev = candle_gen::default_device().map_err(|e| wrap_msg(e.to_string()))?;
    println!("device: {dev:?}");
    for comp in WEIGHT_COMPONENTS {
        if let Some(o) = &only {
            if o != comp {
                continue;
            }
        }
        convert_component(comp, &src, &out, &dev)?;
    }

    if only.is_none() {
        for d in COPY_DIRS {
            let s = src.join(d);
            if s.is_dir() {
                copy_dir(&s, &out.join(d))?;
                println!("copied {d}/");
            }
        }
        for f in COPY_FILES {
            if src.join(f).is_file() {
                std::fs::copy(src.join(f), out.join(f)).map_err(wrap)?;
            }
        }
    }
    println!("DONE");
    Ok(())
}

fn convert_component(comp: &str, src: &Path, out: &Path, dev: &Device) -> Result<()> {
    let dir = src.join(comp);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(wrap)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "{comp}: no .safetensors in {} (download incomplete?)",
            dir.display()
        )));
    }
    // SAFETY: read-only mmap.
    let st = unsafe { MmapedSafetensors::multi(&files)? };
    let names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let present: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();

    let mut out_map: HashMap<String, Tensor> = HashMap::new();
    let (mut n_fp8, mut n_pass, mut n_scale) = (0u32, 0u32, 0u32);
    for name in &names {
        if name.ends_with("_scale") {
            n_scale += 1;
            continue; // folded into its weight
        }
        let t = st.load(name, dev)?;
        if t.dtype() == DType::F8E4M3 {
            let scale_key = format!("{name}_scale");
            if !present.contains(scale_key.as_str()) {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "{comp}: fp8 tensor {name} has no sibling {scale_key}"
                )));
            }
            let scale = st.load(&scale_key, dev)?.to_dtype(DType::F32)?; // [out]
            let mut sh = vec![1usize; t.rank()];
            sh[0] = t.dim(0)?;
            let scale = scale.reshape(sh)?;
            let w = t.to_dtype(DType::F32)?.broadcast_mul(&scale)?;
            out_map.insert(name.clone(), w.to_dtype(DType::BF16)?);
            n_fp8 += 1;
        } else {
            // Passthrough: floats → bf16, integer buffers keep their dtype.
            let conv = if t.dtype().is_int() {
                t
            } else {
                t.to_dtype(DType::BF16)?
            };
            out_map.insert(name.clone(), conv);
            n_pass += 1;
        }
    }

    let comp_out = out.join(comp);
    std::fs::create_dir_all(&comp_out).map_err(wrap)?;
    if src.join(comp).join("config.json").is_file() {
        std::fs::copy(
            src.join(comp).join("config.json"),
            comp_out.join("config.json"),
        )
        .map_err(wrap)?;
    }
    save(&out_map, comp_out.join("model.safetensors"))?;
    println!(
        "[{comp}] fp8_dequant={n_fp8} passthrough={n_pass} scales_folded={n_scale} -> {} tensors",
        out_map.len()
    );
    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).map_err(wrap)?;
    for e in std::fs::read_dir(src).map_err(wrap)? {
        let e = e.map_err(wrap)?;
        let p = e.path();
        if p.is_file() {
            std::fs::copy(&p, dst.join(e.file_name())).map_err(wrap)?;
        }
    }
    Ok(())
}

fn wrap(e: std::io::Error) -> candle_gen::candle_core::Error {
    candle_gen::candle_core::Error::Msg(format!("io: {e}"))
}

fn wrap_msg(m: String) -> candle_gen::candle_core::Error {
    candle_gen::candle_core::Error::Msg(m)
}
