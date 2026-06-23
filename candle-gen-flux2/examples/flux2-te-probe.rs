//! FLUX.2-dev Mistral TE diagnostic (sc-7457 black-image debug). Loads ONLY the dev text encoder on
//! the CPU (f32, ~86 GB — fits system RAM, no GPU OOM), tokenizes a prompt, and prints `prompt_embeds`
//! magnitude stats for the **dense** encoder and again after **Q4** quantization. Isolates whether the
//! catastrophic ±2e10 prompt-embeds magnitude (which overflows the DiT softmax → NaN → black render)
//! comes from Q4 quantization or from the TE architecture itself. Also dumps the attention-mask
//! padding side (left vs right) to test the all-masked-row softmax-NaN hypothesis.
//!
//! ```text
//! cargo run --release --example flux2-te-probe -- --snapshot "D:\models\FLUX.2-dev"
//! ```

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::Quant;
use candle_gen_flux2::config::Flux2Config;
use candle_gen_flux2::text_encoder::Qwen3TextEncoder;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn stats(name: &str, t: &Tensor) -> Result<()> {
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let (mut mn, mut mx, mut s, mut bad, mut absmax) =
        (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0usize, 0f32);
    for &x in &v {
        if x.is_finite() {
            mn = mn.min(x);
            mx = mx.max(x);
            absmax = absmax.max(x.abs());
        } else {
            bad += 1;
        }
        s += x as f64;
    }
    println!(
        "[probe] {name}: shape={:?} min={mn:.4} max={mx:.4} absmax={absmax:.4} mean={:.6} nonfinite={bad}",
        t.dims(),
        s / v.len().max(1) as f64
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot").unwrap_or_else(|| r"D:\models\FLUX.2-dev".into());
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed"
            .into()
    });
    let snapshot = PathBuf::from(snapshot);
    let cpu = Device::Cpu;
    let cfg = Flux2Config::dev();

    // --- tokenize (dev: Mistral pad=11, [INST] chat template, pad to 512) ---
    let tok = TextTokenizer::from_file(
        snapshot.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: cfg.max_sequence_length,
            pad_token_id: 11,
            chat_template: ChatTemplate::Flux2DevMistral,
            pad_to_max_length: true,
        },
    )?;
    let out = tok.tokenize(&prompt)?;
    let len = out.ids.len();
    let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    let mask: Vec<i64> = out.mask.iter().map(|&m| m as i64).collect();
    let real: usize = mask.iter().filter(|&&m| m == 1).count();
    println!("[probe] tokens len={len} real={real} pad={}", len - real);
    let head: Vec<i64> = mask.iter().take(16).copied().collect();
    let tail: Vec<i64> = mask.iter().rev().take(8).copied().collect();
    println!("[probe] mask head(16)={head:?}  mask tail(8, reversed)={tail:?}");
    let id_head: Vec<u32> = ids.iter().take(8).copied().collect();
    println!("[probe] id head(8)={id_head:?}");

    let input_ids = Tensor::from_vec(ids, (1, len), &cpu)?;
    let attn = Tensor::from_vec(mask, (1, len), &cpu)?;

    // --- build the dev TE dense on CPU (f32) ---
    let te_dir = snapshot.join("text_encoder");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&te_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    println!(
        "[probe] loading {} text_encoder shards (f32, CPU)...",
        files.len()
    );
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &cpu)? };
    let mut te = Qwen3TextEncoder::new(&cfg, vb)?;
    println!("[probe] dense TE built; encoding...");
    let pe_dense = te.prompt_embeds(&input_ids, &attn)?;
    stats("DENSE prompt_embeds", &pe_dense)?;

    // --- quantize the SAME encoder to Q4 (on CPU) and re-encode ---
    println!("[probe] quantizing TE to Q4 (CPU)...");
    te.quantize(Quant::Q4, &cpu)?;
    let pe_q4 = te.prompt_embeds(&input_ids, &attn)?;
    stats("Q4    prompt_embeds", &pe_q4)?;

    // --- CUDA Q4: the exact render path (CPU-stage dense → quantize_onto GPU) ---
    match Device::new_cuda(0) {
        Ok(gpu) => {
            println!("[probe] building fresh dense TE → quantize_onto CUDA (Q4)...");
            let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &cpu)? };
            let mut te_g = Qwen3TextEncoder::new(&cfg, vb2)?;
            te_g.quantize(Quant::Q4, &gpu)?;
            let ids_g = input_ids.to_device(&gpu)?;
            let attn_g = attn.to_device(&gpu)?;
            let pe_g = te_g.prompt_embeds(&ids_g, &attn_g)?;
            stats("Q4-CUDA prompt_embeds", &pe_g)?;
        }
        Err(e) => println!("[probe] no CUDA device ({e}); skipping GPU Q4"),
    }

    Ok(())
}
