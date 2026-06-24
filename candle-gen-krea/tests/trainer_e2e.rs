//! End-to-end real-weight smoke for the candle Krea 2 **trainer** (sc-7577) — the production
//! `KreaTrainer` driven through the gen-core registry exactly as the SceneWorks worker will, on the
//! real Krea-2-Raw 12B base. The candle twin of `candle-gen-z-image/tests/trainer_e2e.rs`, on the
//! flow-match objective.
//!
//! `#[ignore]`d (not feature-gated, so the bodies are compile-checked in normal CI). Needs the real
//! Krea-2-Raw snapshot (`KREA_RAW_DIR`, or `KREA_TURBO_DIR` — Raw ≡ Turbo architecture, so a Turbo
//! snapshot also exercises the load+train path) and enough VRAM for the bf16 12B DiT (~24 GB resident
//! alongside the f32 VAE encoder / text encoder during caching). On the Windows/Blackwell box
//! (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set KREA_RAW_DIR=D:\models\Krea-2-Raw
//! cargo test -p candle-gen-krea --features cuda --release --test trainer_e2e -- --ignored --nocapture
//! ```
//!
//! What it proves (the sc-7577 AC):
//!  - **prepare→cache→train→save** lifecycle: a tiny captioned-PNG dataset is VAE-mean / Qwen3-VL-4B
//!    encoded + cached, the optimizer drives the flow-match velocity loss down (windowed median falls —
//!    convergence on real data, not just finite), and a PEFT/LoKr adapter is written with the expected
//!    bare-dotted DiT keys (`transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}`) plus the
//!    `networkType`/`rank`/`alpha`/`baseModel`/`family` metadata the Turbo cross-apply policy reads.
//!  - **the 112-target default surface**: a LoRA adapts exactly the 28 single-stream blocks' four
//!    attention projections (28 × 4).
//!  - **gradient checkpointing** (the memory-bounded backward) converges + saves end-to-end (its
//!    bit-exact dense-vs-checkpoint grad parity is the f32 `dense_and_checkpoint_grads_match` unit gate).
//!  - **launch-portable determinism**: the same seed produces the same adapter bytes, run to run.
//!
//! The train→**infer** round-trip (a Raw-trained adapter applied at Turbo inference, sc-7837) is
//! exercised by [`krea_lora_roundtrip_raw_to_turbo`] now that the inference-apply seam has landed
//! (sc-7836): train on Raw → merge across the 112-target surface (nothing skipped) → render at Turbo
//! with the adapter (coherent + visibly adapted; scale-0 ≡ base byte-exact). Cross-framework numeric
//! parity with MLX/torch is explicitly a non-goal (different autograd + RNG); candle-internal
//! determinism + behavioural parity (converges; well-formed adapter; coherent adapted render) is what
//! is guaranteed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, registry, AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest,
    Image, LoadSpec, NetworkType, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::{merge_into_weights, Krea2Config};

/// The Krea-2-Raw base snapshot dir — `KREA_RAW_DIR`, or `KREA_TURBO_DIR` (architecture-identical).
fn snapshot() -> Option<PathBuf> {
    std::env::var("KREA_RAW_DIR")
        .or_else(|_| std::env::var("KREA_TURBO_DIR"))
        .ok()
        .map(PathBuf::from)
}

/// Two solid-colour swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(96, 96);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = dir.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("a solid colour swatch number {i}"),
        });
    }
    items
}

/// A tiny config: small rank, low resolution (64 → 8×8 latent → 4×4 = 16 image tokens), few steps.
/// **bf16** compute — the Krea DiT's native dtype and what inference loads. The trainable adapter
/// factors / loss / grads / optimizer state stay f32 regardless (master weights).
fn config(network_type: NetworkType, steps: u32, grad_ckpt: bool) -> TrainingConfig {
    TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-3,
        steps,
        resolution: 64,
        save_every: 0,
        seed: 7,
        network_type,
        decompose_factor: -1,
        train_dtype: "bf16".to_string(),
        gradient_checkpointing: grad_ckpt,
        ..Default::default()
    }
}

struct RunOut {
    losses: Vec<f32>,
    adapter_path: PathBuf,
}

/// Train through the registry and collect the per-step losses + the adapter path.
fn run(
    tmp: &Path,
    file_name: &str,
    network_type: NetworkType,
    steps: u32,
    grad_ckpt: bool,
) -> RunOut {
    let items = make_dataset(tmp);
    // Reference the provider crate so its `inventory::submit!` trainer registration is linked in.
    candle_gen_krea::force_link();

    let mut trainer = gen_core::load_trainer(
        "krea_2_raw",
        &LoadSpec::new(WeightsSource::Dir(
            snapshot().expect("KREA_RAW_DIR / KREA_TURBO_DIR"),
        )),
    )
    .expect("krea_2_raw candle trainer should be registered");

    let req = TrainingRequest {
        items,
        config: config(network_type, steps, grad_ckpt),
        output_dir: tmp.join("out"),
        file_name: file_name.to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut losses: Vec<f32> = Vec::new();
    let mut cached = 0u32;
    let out = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Caching { current, .. } => cached = current,
            TrainingProgress::Training { loss, .. } => losses.push(loss),
            _ => {}
        })
        .expect("training should succeed");

    assert_eq!(cached, 2, "both dataset items should be cached");
    assert_eq!(out.steps, steps, "all micro-steps should run");
    assert_eq!(losses.len() as u32, steps);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf losses (not diverging)"
    );
    assert!(out.adapter_path.exists(), "adapter file should be written");
    println!("[{file_name}] losses: {losses:?}");
    RunOut {
        losses,
        adapter_path: out.adapter_path,
    }
}

/// Assert the windowed loss **median** falls (robust to the heavy-tailed per-σ flow-match loss — each
/// step samples a fresh σ, and a high-σ draw spikes that step regardless of progress). Compares the
/// first-quarter vs last-quarter median.
fn assert_converged(tag: &str, losses: &[f32]) {
    let q = (losses.len() / 4).max(1);
    let median = |s: &[f32]| {
        let mut v = s.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (first_q, last_q) = (median(&losses[..q]), median(&losses[losses.len() - q..]));
    println!("[{tag}] loss-median first-quarter {first_q:.5} -> last-quarter {last_q:.5}");
    assert!(
        last_q < first_q * 0.9,
        "[{tag}] windowed loss-median should fall on real data: {first_q:.5} -> {last_q:.5}"
    );
}

/// The adapter file's header metadata.
fn read_meta(path: &Path) -> HashMap<String, String> {
    let bytes = std::fs::read(path).unwrap();
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    md.metadata().clone().unwrap_or_default()
}

/// Sorted tensor keys of the adapter file (via candle's loader).
fn adapter_keys(path: &Path) -> Vec<String> {
    let map = candle_gen::candle_core::safetensors::load(path, &Device::Cpu).unwrap();
    let mut keys: Vec<String> = map.into_keys().collect();
    keys.sort();
    keys
}

/// LoRA: trains + converges, writes a bare-dotted PEFT adapter over the 112-target attention surface,
/// with the expected metadata; and the same seed reproduces the same adapter bytes.
#[test]
#[ignore = "needs real Krea-2-Raw weights + a GPU; run with --features cuda --release --ignored"]
fn krea_lora_trains_and_is_well_formed() {
    if snapshot().is_none() {
        eprintln!("skipping: set KREA_RAW_DIR (or KREA_TURBO_DIR)");
        return;
    }
    let tmp = std::env::temp_dir().join("krea_trainer_e2e_lora");
    let out = run(&tmp, "krea_lora.safetensors", NetworkType::Lora, 24, false);
    assert_converged("lora", &out.losses);

    // 28 blocks × 4 attention projections = 112 targets; each writes lora_A/lora_B/alpha (3 tensors).
    let keys = adapter_keys(&out.adapter_path);
    let a = keys
        .iter()
        .filter(|k| k.ends_with(".lora_A.weight"))
        .count();
    let b = keys
        .iter()
        .filter(|k| k.ends_with(".lora_B.weight"))
        .count();
    assert_eq!(
        a, 112,
        "112 LoRA-A factors (28 blocks × 4 attn projs), got {a}"
    );
    assert_eq!(b, 112, "112 LoRA-B factors, got {b}");
    assert!(
        keys.iter().all(|k| k.starts_with("transformer_blocks.")),
        "DiT family adapters use bare-dotted keys (no base_model.model.unet. prefix)"
    );

    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lora"));
    assert_eq!(meta.get("rank").map(String::as_str), Some("8"));
    assert_eq!(
        meta.get("baseModel").map(String::as_str),
        Some("krea_2_raw")
    );
    assert_eq!(meta.get("family").map(String::as_str), Some("krea_2"));

    // Determinism: a second run at the same seed writes byte-identical factors.
    let out2 = run(
        &tmp,
        "krea_lora_2.safetensors",
        NetworkType::Lora,
        24,
        false,
    );
    let (b1, b2) = (
        std::fs::read(&out.adapter_path).unwrap(),
        std::fs::read(&out2.adapter_path).unwrap(),
    );
    assert_eq!(b1, b2, "same seed must produce a byte-identical adapter");
}

/// LoKr: trains + converges and writes a bare-keyed LoKr adapter with the LoKr metadata.
#[test]
#[ignore = "needs real Krea-2-Raw weights + a GPU; run with --features cuda --release --ignored"]
fn krea_lokr_trains_and_is_well_formed() {
    if snapshot().is_none() {
        eprintln!("skipping: set KREA_RAW_DIR (or KREA_TURBO_DIR)");
        return;
    }
    let tmp = std::env::temp_dir().join("krea_trainer_e2e_lokr");
    let out = run(&tmp, "krea_lokr.safetensors", NetworkType::Lokr, 24, false);
    assert_converged("lokr", &out.losses);

    let keys = adapter_keys(&out.adapter_path);
    let w1 = keys.iter().filter(|k| k.ends_with(".lokr_w1")).count();
    assert_eq!(
        w1, 112,
        "112 LoKr w1 factors (28 blocks × 4 attn projs), got {w1}"
    );
    assert!(
        keys.iter().all(|k| k.starts_with("transformer_blocks.")),
        "bare-dotted LoKr keys"
    );
    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lokr"));
    assert_eq!(
        meta.get("baseModel").map(String::as_str),
        Some("krea_2_raw")
    );
}

/// Gradient checkpointing: the memory-bounded backward converges + saves end-to-end on real weights.
#[test]
#[ignore = "needs real Krea-2-Raw weights + a GPU; run with --features cuda --release --ignored"]
fn krea_lora_gradient_checkpointing() {
    if snapshot().is_none() {
        eprintln!("skipping: set KREA_RAW_DIR (or KREA_TURBO_DIR)");
        return;
    }
    let tmp = std::env::temp_dir().join("krea_trainer_e2e_ckpt");
    let out = run(
        &tmp,
        "krea_lora_ckpt.safetensors",
        NetworkType::Lora,
        24,
        true,
    );
    assert_converged("lora-ckpt", &out.losses);
    assert_eq!(
        adapter_keys(&out.adapter_path)
            .iter()
            .filter(|k| k.ends_with(".lora_A.weight"))
            .count(),
        112
    );
}

// ── sc-7837: the real Raw→Turbo round trip (train on Raw → apply at Turbo inference) ──────────────

/// The Turbo render snapshot — `KREA_TURBO_DIR`, distinct from the Raw training snapshot ([`snapshot`]).
fn turbo_dir() -> Option<PathBuf> {
    std::env::var("KREA_TURBO_DIR").ok().map(PathBuf::from)
}

/// `n` copies of a vivid solid-colour swatch + a shared caption — a strong, low-frequency concept that
/// survives the train(64²)→render(1024²) resolution jump and the Raw→Turbo cross-apply, so the effect
/// is *visible* rather than a subtle texture shift.
fn make_concept_dataset(dir: &Path, rgb: [u8; 3], caption: &str, n: usize) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    (0..n)
        .map(|i| {
            let mut img = image::RgbImage::new(96, 96);
            for px in img.pixels_mut() {
                *px = image::Rgb(rgb);
            }
            let path = dir.join(format!("c{i}.png"));
            img.save(&path).unwrap();
            TrainingItem {
                image_path: path,
                caption: caption.to_string(),
            }
        })
        .collect()
}

/// Train a LoRA/LoKr on the Raw snapshot through the registry with a custom dataset + config; returns
/// the written adapter path (asserts finite losses + that the file exists).
fn train_concept(
    items: Vec<TrainingItem>,
    cfg: TrainingConfig,
    out_dir: &Path,
    file_name: &str,
) -> PathBuf {
    candle_gen_krea::force_link();
    let mut trainer = gen_core::load_trainer(
        "krea_2_raw",
        &LoadSpec::new(WeightsSource::Dir(
            snapshot().expect("KREA_RAW_DIR / KREA_TURBO_DIR"),
        )),
    )
    .expect("krea_2_raw trainer registered");
    let req = TrainingRequest {
        items,
        config: cfg,
        output_dir: out_dir.to_path_buf(),
        file_name: file_name.to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    let mut losses: Vec<f32> = Vec::new();
    let out = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { loss, .. } = p {
                losses.push(loss);
            }
        })
        .expect("training succeeds");
    assert!(losses.iter().all(|l| l.is_finite()), "no NaN/Inf losses");
    assert!(out.adapter_path.exists(), "adapter written");
    println!(
        "[train_concept {file_name}] {} steps, last loss {:.5}",
        losses.len(),
        losses.last().copied().unwrap_or(f32::NAN)
    );
    out.adapter_path
}

/// Render one 1024² image at `krea_2_turbo` with `adapters` merged into the DiT (the sc-7836 path).
fn render_turbo(root: &Path, prompt: &str, adapters: Vec<AdapterSpec>) -> Image {
    candle_gen_krea::force_link();
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_adapters(adapters);
    let gen = registry::load("krea_2_turbo", &spec).expect("load krea_2_turbo");
    let req = GenerationRequest {
        prompt: prompt.into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };
    let GenerationOutput::Images(mut imgs) = gen.generate(&req, &mut |_| {}).expect("generate")
    else {
        panic!("expected images");
    };
    imgs.remove(0)
}

fn save_png(img: &Image, name: &str) {
    let dir = std::env::temp_dir().join("krea_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    println!("  saved {}", path.display());
}

/// Mean of one RGB channel (`off` = 0/1/2 → R/G/B) over an RGB8 buffer.
fn mean_channel(px: &[u8], off: usize) -> f64 {
    let sum: f64 = px.iter().skip(off).step_by(3).map(|&v| v as f64).sum();
    sum / (px.len() as f64 / 3.0)
}

/// The sc-7837 round trip on **real** weights: train a LoRA on Krea-2-Raw (a vivid green concept), then
/// apply it at `krea_2_turbo` inference. Asserts the trained adapter merges across the full 112-target
/// surface (nothing skipped — the `assert_reloads` mirror), scale-0 ≡ base byte-exact, and a non-zero
/// scale yields a coherent, differing render; logs the green-channel shift and saves PNGs for the
/// "visibly adapted" eyeball (the qualitative half of the AC).
#[test]
#[ignore = "needs Krea-2-Raw (train) + Krea-2-Turbo (render) + GPU; --features cuda --release"]
fn krea_lora_roundtrip_raw_to_turbo() {
    let (Some(_raw), Some(turbo)) = (snapshot(), turbo_dir()) else {
        eprintln!("skipping: set KREA_RAW_DIR and KREA_TURBO_DIR");
        return;
    };
    let tmp = std::env::temp_dir().join("krea_rt");

    // 1) Train a deliberately over-fit LoRA on a vivid emerald-green concept. **Gradient checkpointing
    //    on** — the retained backward OOMs on the 12B DiT (~24 GB resident leaves no VRAM headroom for
    //    the full activation graph); the checkpointed backward is memory-bounded.
    let items = make_concept_dataset(
        &tmp.join("data"),
        [25, 200, 70],
        "a vivid emerald green field",
        4,
    );
    let mut cfg = config(NetworkType::Lora, 150, true);
    cfg.learning_rate = 1.5e-3;
    let adapter = train_concept(items, cfg, &tmp.join("out"), "krea_green.safetensors");

    // 2) The trained adapter merges across the full attention surface of the real Turbo DiT.
    let tcfg = Krea2Config::from_snapshot(&turbo).expect("turbo config");
    let mut w = Weights::from_dir(&turbo.join("transformer"), &Device::Cpu, DType::BF16).unwrap();
    let report = merge_into_weights(
        &mut w,
        &tcfg,
        &[AdapterSpec::new(adapter.clone(), 1.0, AdapterKind::Lora)],
    )
    .expect("merge trained adapter");
    assert_eq!(report.merged, 112, "every trained target merges");
    assert_eq!(report.skipped_keys, 0, "nothing skipped");

    // 3) Render base / scale-0 / scale-1 at Turbo (same prompt + seed isolates the LoRA's effect).
    let prompt = "a photograph of a city street at noon";
    let base = render_turbo(&turbo, prompt, vec![]);
    let zero = render_turbo(
        &turbo,
        prompt,
        vec![AdapterSpec::new(adapter.clone(), 0.0, AdapterKind::Lora)],
    );
    assert_eq!(
        base.pixels, zero.pixels,
        "scale-0 adapter ≡ base render (byte-exact)"
    );

    let adapted = render_turbo(
        &turbo,
        prompt,
        vec![AdapterSpec::new(adapter.clone(), 1.0, AdapterKind::Lora)],
    );
    assert_eq!((adapted.width, adapted.height), (1024, 1024), "output dims");
    let diff = base
        .pixels
        .iter()
        .zip(&adapted.pixels)
        .filter(|(a, b)| a != b)
        .count();
    let (gb, ga) = (
        mean_channel(&base.pixels, 1),
        mean_channel(&adapted.pixels, 1),
    );
    let (rb, ra) = (
        mean_channel(&base.pixels, 0),
        mean_channel(&adapted.pixels, 0),
    );
    println!(
        "[roundtrip] changed_px={diff}/{} green {gb:.1}->{ga:.1} red {rb:.1}->{ra:.1}",
        base.pixels.len()
    );
    save_png(&base, "rt_base");
    save_png(&adapted, "rt_adapter");
    assert!(
        diff > base.pixels.len() / 100,
        "a trained adapter must change the render"
    );
}
