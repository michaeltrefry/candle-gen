//! End-to-end verification of the candle SDXL **trainer** (sc-5165 #5) — the production `SdxlTrainer`
//! driven through the gen-core registry exactly as the SceneWorks worker will, on real weights + a
//! real CUDA GPU. The candle twin of `mlx-gen-sdxl/tests/trainer_e2e.rs`.
//!
//! `#[ignore]`d + `cfg(feature = "cuda")` — needs the real `stabilityai/stable-diffusion-xl-base-1.0`
//! snapshot (the HF cache, or `SDXL_SNAPSHOT`) and a CUDA GPU. On the Windows/Blackwell box
//! (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set SDXL_SNAPSHOT=C:\Users\…\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\<hash>
//! cargo test -p candle-gen-sdxl --features cuda --release --test trainer_e2e -- --ignored --nocapture
//! ```
//!
//! What it proves:
//!  - **prepare→cache→train→save** lifecycle: a tiny captioned-PNG dataset is VAE/dual-CLIP-encoded +
//!    cached, the optimizer drives the epsilon loss down (windowed mean falls — convergence on real
//!    data, not just finite), and a PEFT/LoKr adapter is written with the expected keys + metadata.
//!  - **train→infer round-trip** (closes the loop with sc-5165's inference merge): the produced
//!    adapter reloads through the real candle inference merge ([`candle_gen_sdxl::merge_adapters`]) —
//!    every trained target merges, nothing is skipped — and a full `generate` with the adapter renders
//!    a finite, correctly-sized image on the GPU.
//!  - **gradient checkpointing** trains + converges + reloads identically (the recompute path).
//!  - **launch-portable determinism** (the achievable "parity" guarantee, see below): the same seed
//!    produces the same adapter, run to run.
//!
//! **On "parity vs torch/MLX":** cross-framework *numeric* parity is explicitly NOT a goal here — the
//! candle and torch/MLX trainers use different autograd, RNG algorithms, and (on candle) the
//! deterministic CPU-seeded noise/timesteps of sc-3673, so the trained weights differ bit-for-bit even
//! at a shared seed (same reasoning as `tests/conformance.rs`'s sampler note). What IS guaranteed, and
//! tested, is the candle-internal launch-portable determinism the worker relies on (same seed ⇒ same
//! adapter) and behavioural parity (converges on real data; the adapter reloads + renders). The
//! numeric cross-check against the reference is done at the component level by the trainer's own unit
//! gates (e.g. `dense_and_checkpoint_grads_match`) and the train↔infer reconstruction-parity tests.
#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};

/// The SDXL base snapshot dir — `SDXL_SNAPSHOT` or the first HF-cache snapshot.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .expect("USERPROFILE/HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir (set SDXL_SNAPSHOT to override)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
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

/// A tiny config: small rank, low resolution (64 → 8×8 latent), few steps. f32 compute so the
/// convergence signal + the determinism check are not masked by bf16's ~3-digit rounding noise (the
/// bf16 production path is exercised by the trainer's own memory-path unit gates).
fn config(network_type: NetworkType, steps: u32, gradient_checkpointing: bool) -> TrainingConfig {
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
        gradient_checkpointing,
        train_dtype: "f32".to_string(),
        ..Default::default()
    }
}

struct RunOut {
    losses: Vec<f32>,
    adapter_path: PathBuf,
}

/// Train through the registry and collect the per-step losses + the adapter path.
fn run(tmp: &Path, file_name: &str, network_type: NetworkType, steps: u32, gc: bool) -> RunOut {
    let items = make_dataset(tmp);
    // Reference the provider crate so its `inventory::submit!` trainer registration is linked into
    // this test binary (else dead-stripped — the test names nothing else from the crate's trainer).
    assert_eq!(candle_gen_sdxl::MODEL_ID, "sdxl");

    let mut trainer =
        gen_core::load_trainer("sdxl", &LoadSpec::new(WeightsSource::Dir(snapshot())))
            .expect("sdxl candle trainer should be registered");

    let req = TrainingRequest {
        items,
        config: config(network_type, steps, gc),
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

/// Assert the windowed loss-mean falls: per-step loss is dominated by timestep variance (each step
/// samples a fresh integer t + noise), so compare first-quarter vs last-quarter means.
fn assert_converged(tag: &str, losses: &[f32]) {
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    println!("[{tag}] loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}");
    assert!(
        last_q < first_q * 0.9,
        "[{tag}] windowed loss-mean should fall on real data: {first_q:.5} -> {last_q:.5}"
    );
}

/// The adapter file's header metadata.
fn read_meta(path: &Path) -> HashMap<String, String> {
    let bytes = std::fs::read(path).unwrap();
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    md.metadata().clone().unwrap_or_default()
}

/// Reload the produced adapter through the REAL candle inference merge onto a fresh base UNet tensor
/// map, asserting every trained target merges and nothing is skipped — the train→infer round-trip.
fn assert_reloads(adapter_path: &Path, kind: AdapterKind, n_targets: usize) {
    let unet_file = snapshot().join("unet/diffusion_pytorch_model.fp16.safetensors");
    let mut base = candle_gen::candle_core::safetensors::load(&unet_file, &Device::Cpu)
        .expect("load base unet tensors");
    let report = candle_gen_sdxl::merge_adapters(
        &mut base,
        &[AdapterSpec::new(adapter_path.to_path_buf(), 1.0, kind)],
    )
    .expect("trained adapter should reload through the inference merge");
    assert_eq!(
        report.merged, n_targets,
        "every trained target should merge ({report:?})"
    );
    assert_eq!(
        report.skipped_keys, 0,
        "no adapter key should be skipped ({report:?})"
    );
}

/// LoRA: trains + converges, writes a PEFT adapter, and reloads + renders through candle inference.
#[test]
#[ignore = "needs real SDXL weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn sdxl_trainer_lora_trains_reloads_and_renders() {
    let tmp = std::env::temp_dir().join("candle_sdxl_trainer_lora_e2e");
    let out = run(
        &tmp,
        "swatch_lora.safetensors",
        NetworkType::Lora,
        64,
        false,
    );
    assert_converged("sdxl-lora", &out.losses);

    // The adapter carries PEFT keys under the SDXL prefix + reload metadata.
    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lora"));
    let tensors =
        candle_gen::candle_core::safetensors::load(&out.adapter_path, &Device::Cpu).unwrap();
    let n_targets = tensors
        .keys()
        .filter(|k| k.ends_with(".lora_A.weight"))
        .count();
    assert!(n_targets > 0, "adapter should contain LoRA factors");
    assert!(
        tensors
            .keys()
            .any(|k| k.starts_with("base_model.model.unet.") && k.ends_with(".to_q.lora_A.weight")),
        "adapter should carry PEFT-prefixed attention LoRA keys"
    );

    // Round-trip 1 — merge report: every target merges, nothing skipped.
    assert_reloads(&out.adapter_path, AdapterKind::Lora, n_targets);

    // Round-trip 2 — full generate with the adapter merged: a finite, correctly-sized image.
    render_finite_with_adapter(&out.adapter_path, AdapterKind::Lora);
    println!("[sdxl-lora] e2e OK — {n_targets} targets reload + merge, render finite");
}

/// LoKr: trains + converges, writes a LoKr adapter (with `decomposeFactor`), and reloads.
#[test]
#[ignore = "needs real SDXL weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn sdxl_trainer_lokr_trains_and_reloads() {
    let tmp = std::env::temp_dir().join("candle_sdxl_trainer_lokr_e2e");
    let out = run(
        &tmp,
        "swatch_lokr.safetensors",
        NetworkType::Lokr,
        64,
        false,
    );
    assert_converged("sdxl-lokr", &out.losses);

    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lokr"));
    assert!(meta.contains_key("decomposeFactor"));
    let tensors =
        candle_gen::candle_core::safetensors::load(&out.adapter_path, &Device::Cpu).unwrap();
    let n_targets = tensors.keys().filter(|k| k.ends_with(".lokr_w1")).count();
    assert!(n_targets > 0, "adapter should contain LoKr factor keys");

    assert_reloads(&out.adapter_path, AdapterKind::Lokr, n_targets);
    render_finite_with_adapter(&out.adapter_path, AdapterKind::Lokr);
    println!("[sdxl-lokr] e2e OK — {n_targets} targets reload + merge, render finite");
}

/// The gradient-checkpointing path trains + converges + reloads identically (sc-5165 #3 on real data).
#[test]
#[ignore = "needs real SDXL weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn sdxl_trainer_gradient_checkpointing_converges() {
    let tmp = std::env::temp_dir().join("candle_sdxl_trainer_gc_e2e");
    let out = run(
        &tmp,
        "swatch_lora_gc.safetensors",
        NetworkType::Lora,
        64,
        true,
    );
    assert_converged("sdxl-gc", &out.losses);
    let tensors =
        candle_gen::candle_core::safetensors::load(&out.adapter_path, &Device::Cpu).unwrap();
    let n_targets = tensors
        .keys()
        .filter(|k| k.ends_with(".lora_A.weight"))
        .count();
    assert_reloads(&out.adapter_path, AdapterKind::Lora, n_targets);
    println!("[sdxl-gc] e2e OK — checkpointed training converged + reloads");
}

/// Launch-portable determinism: the same seed produces the same adapter, run to run (the achievable
/// reproducibility guarantee the worker relies on — see the module header on cross-framework parity).
#[test]
#[ignore = "needs real SDXL weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn sdxl_trainer_same_seed_is_reproducible() {
    let a = run(
        &std::env::temp_dir().join("candle_sdxl_trainer_det_a"),
        "det_a.safetensors",
        NetworkType::Lora,
        6,
        false,
    );
    let b = run(
        &std::env::temp_dir().join("candle_sdxl_trainer_det_b"),
        "det_b.safetensors",
        NetworkType::Lora,
        6,
        false,
    );
    // Same seed ⇒ identical per-step losses.
    assert_eq!(
        a.losses, b.losses,
        "same-seed runs should give identical losses"
    );
    // …and identical adapter factors.
    let ta = candle_gen::candle_core::safetensors::load(&a.adapter_path, &Device::Cpu).unwrap();
    let tb = candle_gen::candle_core::safetensors::load(&b.adapter_path, &Device::Cpu).unwrap();
    assert_eq!(ta.len(), tb.len(), "same key set");
    for (k, va) in &ta {
        let vb = tb
            .get(k)
            .unwrap_or_else(|| panic!("missing key {k} in run B"));
        let max = (va.to_dtype(candle_gen::candle_core::DType::F32).unwrap()
            - vb.to_dtype(candle_gen::candle_core::DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
        assert!(
            max < 1e-6,
            "factor {k} diverged across same-seed runs by {max}"
        );
    }
    println!("[sdxl-det] e2e OK — same seed reproduces the adapter bit-for-bit");
}

/// Load a generator with `adapter_path` merged and run a tiny `generate`, asserting a finite,
/// correctly-sized image. Exercises the full merge→build→denoise→decode path on the GPU — if the
/// merge matched no target it would error inside `generate`, so a finite render proves the merge ran.
fn render_finite_with_adapter(adapter_path: &Path, kind: AdapterKind) {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![AdapterSpec::new(
        adapter_path.to_path_buf(),
        1.0,
        kind,
    )]);
    let g = candle_gen_sdxl::load(&spec).expect("load generator with adapter");
    let req = GenerationRequest {
        prompt: "a solid colour swatch".into(),
        width: 512,
        height: 512,
        steps: Some(2),
        seed: Some(1),
        count: 1,
        ..Default::default()
    };
    let out = g.generate(&req, &mut |_| {}).expect("adapted generate");
    let GenerationOutput::Images(imgs) = out else {
        panic!("expected images");
    };
    assert_eq!(imgs.len(), 1);
    assert_eq!((imgs[0].width, imgs[0].height), (512, 512));
    assert_eq!(imgs[0].pixels.len(), 512 * 512 * 3, "RGB8 buffer");
    // Sanity: not a degenerate all-zero buffer.
    assert!(
        imgs[0].pixels.iter().any(|&p| p != 0),
        "rendered image should not be all-black"
    );
}
