//! End-to-end verification of the candle Z-Image **trainer** (sc-5166) — the production
//! `ZImageTrainer` driven through the gen-core registry exactly as the SceneWorks worker will, on
//! real weights + a real CUDA GPU. The candle twin of `candle-gen-sdxl/tests/trainer_e2e.rs`, on the
//! flow-match objective.
//!
//! `#[ignore]`d + `cfg(feature = "cuda")` — needs the real `Tongyi-MAI/Z-Image-Turbo` snapshot (the
//! HF cache, or `Z_IMAGE_SNAPSHOT`) and a CUDA GPU. On the Windows/Blackwell box (v143 vcvars + CUDA
//! on PATH):
//!
//! ```text
//! set Z_IMAGE_SNAPSHOT=C:\Users\…\models--Tongyi-MAI--Z-Image-Turbo\snapshots\<hash>
//! cargo test -p candle-gen-z-image --features cuda --release --test trainer_e2e -- --ignored --nocapture
//! ```
//!
//! What it proves:
//!  - **prepare→cache→train→save** lifecycle: a tiny captioned-PNG dataset is VAE-mean / Qwen-encoded
//!    and cached, the optimizer drives the flow-match velocity loss down (windowed mean falls —
//!    convergence on real data, not just finite), and a PEFT/LoKr adapter is written with the expected
//!    bare-dotted DiT keys plus metadata.
//!  - **train→infer round-trip** (closes the loop with sc-5166's inference merge): the produced
//!    adapter reloads through the real candle inference merge ([`candle_gen_z_image::merge_adapters`])
//!    — every trained target merges, nothing is skipped — and a full `generate` with the adapter
//!    renders a finite, correctly-sized image on the GPU.
//!  - **launch-portable determinism**: the same seed produces the same adapter, run to run.
//!  - **gradient checkpointing** (sc-5246): the same LoRA recipe with `gradient_checkpointing = true`
//!    converges + reloads + renders on the GPU — the memory-bounded backward runs end-to-end (the
//!    bit-exact dense-vs-checkpoint grad parity is the f32 `dense_and_checkpoint_grads_match` unit gate).
//!
//! **On "parity vs torch/MLX":** cross-framework *numeric* parity is explicitly NOT a goal (different
//! autograd, RNG algorithms, and the deterministic CPU-seeded noise/σ of sc-3673). What IS guaranteed
//! and tested is candle-internal launch-portable determinism (same seed ⇒ same adapter) and
//! behavioural parity (converges on real data; the adapter reloads + renders). The numeric
//! cross-checks are the trainer's own unit gates (`backward_reaches_lora_factors`,
//! `one_optimizer_step_descends`) and the reconstruction-parity tests in [`candle_gen::train::lora`].
#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::{
    self, AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, Image,
    LoadSpec, NetworkType, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};

/// The Z-Image base snapshot dir — `Z_IMAGE_SNAPSHOT` or the first HF-cache snapshot.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("Z_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .expect("USERPROFILE/HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir (set Z_IMAGE_SNAPSHOT to override)")
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

/// A tiny config: small rank, low resolution (64 → 8×8 latent → 16 image tokens), few steps. **bf16**
/// compute — the model's native dtype (unlike SDXL's fp16 family) and what inference loads, so the
/// merged adapter exercises the exact precision it will run at. (f32 would double the resident DiT to
/// ~25 GB and OOM during caching alongside the f32 text encoder; bf16 is also the trainer's default.)
/// The trainable adapter factors / loss / grads / optimizer state stay f32 regardless (master
/// weights), so the convergence + determinism signals are not bf16-noisy at the parameter level.
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

/// Train through the registry and collect the per-step losses + the adapter path. `grad_ckpt` selects
/// the gradient-checkpointed backward (sc-5246) over the dense one.
fn run(
    tmp: &Path,
    file_name: &str,
    network_type: NetworkType,
    steps: u32,
    grad_ckpt: bool,
) -> RunOut {
    let items = make_dataset(tmp);
    // Reference the provider crate so its `inventory::submit!` trainer registration is linked in.
    assert_eq!(candle_gen_z_image::MODEL_ID, "z_image_turbo");

    let mut trainer = gen_core::load_trainer(
        "z_image_turbo",
        &LoadSpec::new(WeightsSource::Dir(snapshot())),
    )
    .expect("z_image_turbo candle trainer should be registered");

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

/// Assert the windowed loss **median** falls: the flow-match per-step loss is heavy-tailed — each
/// step samples a fresh σ, and a high-σ draw (where the velocity target `noise − x0` is large) spikes
/// that step's loss regardless of training progress. The *mean* of a 16-step window is dominated by a
/// couple of those spikes; the *median* tracks the central training trend, so we compare the
/// first-quarter vs last-quarter median (robust to the σ outliers).
fn assert_converged(tag: &str, losses: &[f32]) {
    let q = losses.len() / 4;
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

/// Load the (possibly sharded) base DiT `transformer/` tensors into one CPU map.
fn load_transformer_base(snap: &Path) -> HashMap<String, Tensor> {
    let dir = snap.join("transformer");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("transformer/ dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let mut base = HashMap::new();
    for f in &files {
        base.extend(candle_gen::candle_core::safetensors::load(f, &Device::Cpu).unwrap());
    }
    base
}

/// Reload the produced adapter through the REAL candle inference merge onto a fresh base DiT tensor
/// map, asserting every trained target merges and nothing is skipped — the train→infer round-trip.
fn assert_reloads(adapter_path: &Path, kind: AdapterKind, n_targets: usize) {
    let mut base = load_transformer_base(&snapshot());
    let report = candle_gen_z_image::merge_adapters(
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

/// LoRA: trains + converges, writes a bare-dotted PEFT adapter, reloads + renders through inference.
#[test]
#[ignore = "needs real Z-Image weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn z_image_trainer_lora_trains_reloads_and_renders() {
    let tmp = std::env::temp_dir().join("candle_zimage_trainer_lora_e2e");
    let out = run(
        &tmp,
        "swatch_lora.safetensors",
        NetworkType::Lora,
        64,
        false,
    );
    assert_converged("zimage-lora", &out.losses);

    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lora"));
    let tensors =
        candle_gen::candle_core::safetensors::load(&out.adapter_path, &Device::Cpu).unwrap();
    let n_targets = tensors
        .keys()
        .filter(|k| k.ends_with(".lora_A.weight"))
        .count();
    assert!(n_targets > 0, "adapter should contain LoRA factors");
    // DiT family writes BARE dotted keys (no `base_model.model.unet.` prefix) — attention projections.
    assert!(
        tensors
            .keys()
            .any(|k| k.ends_with(".attention.to_q.lora_A.weight")),
        "adapter should carry bare-dotted attention LoRA keys"
    );

    assert_reloads(&out.adapter_path, AdapterKind::Lora, n_targets);
    render_finite_with_adapter(&out.adapter_path, AdapterKind::Lora);
    println!("[zimage-lora] e2e OK — {n_targets} targets reload + merge, render finite");
}

/// Gradient checkpointing (sc-5246): the **same** LoRA recipe with `gradient_checkpointing = true`
/// must still converge, reload, and render on real CUDA weights — proving the memory-bounded backward
/// (retained pre-main refiners + checkpointed main `layers` + the stitched boundary cotangent) runs
/// end-to-end on the GPU and descends the flow-match objective, not just on the CPU parity gate. (The
/// adapter is NOT asserted bit-identical to the dense run: on bf16 CUDA the recompute + stitch reorder
/// the reductions, so they agree numerically — the f32 `dense_and_checkpoint_grads_match` unit gate —
/// but need not match bit-for-bit. Run-to-run determinism within the checkpoint path still holds.)
#[test]
#[ignore = "needs real Z-Image weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn z_image_trainer_gradient_checkpointing_trains_and_reloads() {
    let tmp = std::env::temp_dir().join("candle_zimage_trainer_ckpt_e2e");
    let out = run(&tmp, "swatch_ckpt.safetensors", NetworkType::Lora, 64, true);
    assert_converged("zimage-ckpt", &out.losses);

    let tensors =
        candle_gen::candle_core::safetensors::load(&out.adapter_path, &Device::Cpu).unwrap();
    let n_targets = tensors
        .keys()
        .filter(|k| k.ends_with(".lora_A.weight"))
        .count();
    assert!(
        n_targets > 0,
        "checkpointed adapter should contain LoRA factors"
    );
    // The checkpointed backward must reach BOTH the retained pre-main refiners and the checkpointed
    // main layers — assert the adapter carries a refiner key AND a main-layer key.
    assert!(
        tensors
            .keys()
            .any(|k| k.contains("refiner") && k.ends_with(".attention.to_q.lora_A.weight")),
        "checkpointed adapter should carry refiner attention keys (the retained pre-main path)"
    );
    assert!(
        tensors
            .keys()
            .any(|k| k.starts_with("layers.") && k.ends_with(".attention.to_q.lora_A.weight")),
        "checkpointed adapter should carry main-layer attention keys (the checkpointed path)"
    );

    assert_reloads(&out.adapter_path, AdapterKind::Lora, n_targets);
    render_finite_with_adapter(&out.adapter_path, AdapterKind::Lora);
    println!("[zimage-ckpt] e2e OK — gradient-checkpointed train converges, {n_targets} targets reload + render");
}

/// LoKr: trains + converges, writes a LoKr adapter (with `decomposeFactor`), and reloads.
#[test]
#[ignore = "needs real Z-Image weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn z_image_trainer_lokr_trains_and_reloads() {
    let tmp = std::env::temp_dir().join("candle_zimage_trainer_lokr_e2e");
    // 128 steps: the LoKr Kronecker reparam descends slower than LoRA at the same lr/rank, so it needs
    // more steps to show a clear median fall on this tiny over-fit task.
    let out = run(
        &tmp,
        "swatch_lokr.safetensors",
        NetworkType::Lokr,
        128,
        false,
    );
    assert_converged("zimage-lokr", &out.losses);

    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lokr"));
    assert!(meta.contains_key("decomposeFactor"));
    let tensors =
        candle_gen::candle_core::safetensors::load(&out.adapter_path, &Device::Cpu).unwrap();
    let n_targets = tensors.keys().filter(|k| k.ends_with(".lokr_w1")).count();
    assert!(n_targets > 0, "adapter should contain LoKr factor keys");

    assert_reloads(&out.adapter_path, AdapterKind::Lokr, n_targets);
    render_finite_with_adapter(&out.adapter_path, AdapterKind::Lokr);
    println!("[zimage-lokr] e2e OK — {n_targets} targets reload + merge, render finite");
}

/// sc-8650: with a sample cadence set, the trainer renders preview images from the **in-progress
/// adapter** and emits them as [`TrainingProgress::Sample`] — each a valid, non-empty RGB8 bitmap. This
/// is the candle twin of the MLX sc-5637 `*_emits_preview_samples` smokes.
#[test]
#[ignore = "needs real Z-Image weights + a GPU; run with --features cuda --release --ignored"]
fn z_image_trainer_emits_preview_samples() {
    if !snapshot().exists() {
        eprintln!("skipping: set Z_IMAGE_SNAPSHOT (or populate the HF cache)");
        return;
    }
    let tmp = std::env::temp_dir().join("candle_zimage_trainer_samples_e2e");
    let items = make_dataset(&tmp);
    // Reference the provider crate so its `inventory::submit!` trainer registration is linked in.
    assert_eq!(candle_gen_z_image::MODEL_ID, "z_image_turbo");
    let mut trainer = gen_core::load_trainer(
        "z_image_turbo",
        &LoadSpec::new(WeightsSource::Dir(snapshot())),
    )
    .expect("z_image_turbo candle trainer should be registered");

    // 4 steps, render a preview every 2 steps over 2 prompts → 2 cadences × 2 prompts = 4 Sample events.
    let mut cfg = config(NetworkType::Lora, 4, false);
    cfg.sample_every = 2;
    cfg.sample_steps = 4;
    cfg.sample_guidance_scale = 1.0;
    cfg.sample_prompts = vec![
        "a solid red swatch".to_string(),
        "a solid blue swatch".to_string(),
    ];

    let req = TrainingRequest {
        items,
        config: cfg,
        output_dir: tmp.join("out"),
        file_name: "swatch_samples.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut samples: Vec<(u32, u32, u32, String, Image)> = Vec::new();
    trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Sample {
                step,
                index,
                total,
                prompt,
                image,
            } = p
            {
                samples.push((step, index, total, prompt, image));
            }
        })
        .expect("training with sampling succeeds");

    assert_eq!(
        samples.len(),
        4,
        "expected 4 preview samples (2 cadences × 2 prompts), got {}",
        samples.len()
    );
    for (step, index, total, prompt, img) in &samples {
        assert!(
            *step == 2 || *step == 4,
            "preview at a cadence step, got {step}"
        );
        assert_eq!(*total, 2, "two prompts per cadence");
        assert!((1..=2).contains(index), "1-based prompt index, got {index}");
        assert!(img.width > 0 && img.height > 0, "non-empty preview dims");
        assert_eq!(
            img.pixels.len(),
            (img.width * img.height * 3) as usize,
            "RGB8 row-major: pixels.len() == w*h*3"
        );
        assert!(
            img.pixels.iter().any(|&b| b != 0),
            "preview from the in-progress adapter is not all-black (prompt {prompt:?})"
        );
        println!(
            "[zimage sample] step {step} {index}/{total} {}x{} prompt {prompt:?}",
            img.width, img.height
        );
    }
}

/// Launch-portable determinism: the same seed produces the same adapter, run to run (the achievable
/// reproducibility guarantee the worker relies on — see the module header on cross-framework parity).
#[test]
#[ignore = "needs real Z-Image weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn z_image_trainer_same_seed_is_reproducible() {
    let a = run(
        &std::env::temp_dir().join("candle_zimage_trainer_det_a"),
        "det_a.safetensors",
        NetworkType::Lora,
        6,
        false,
    );
    let b = run(
        &std::env::temp_dir().join("candle_zimage_trainer_det_b"),
        "det_b.safetensors",
        NetworkType::Lora,
        6,
        false,
    );
    assert_eq!(
        a.losses, b.losses,
        "same-seed runs should give identical losses"
    );
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
    println!("[zimage-det] e2e OK — same seed reproduces the adapter bit-for-bit");
}

/// Load a generator with `adapter_path` merged and run a tiny `generate`, asserting a finite,
/// correctly-sized image. Exercises the full merge→build→denoise→decode path on the GPU — if the
/// merge matched no target it would error inside `load`/`generate`, so a finite render proves it ran.
fn render_finite_with_adapter(adapter_path: &Path, kind: AdapterKind) {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![AdapterSpec::new(
        adapter_path.to_path_buf(),
        1.0,
        kind,
    )]);
    let g = candle_gen_z_image::load(&spec).expect("load generator with adapter");
    let req = GenerationRequest {
        prompt: "a solid colour swatch".into(),
        width: 256,
        height: 256,
        steps: Some(4),
        seed: Some(1),
        count: 1,
        ..Default::default()
    };
    let out = g.generate(&req, &mut |_| {}).expect("adapted generate");
    let GenerationOutput::Images(imgs) = out else {
        panic!("expected images");
    };
    assert_eq!(imgs.len(), 1);
    assert_eq!((imgs[0].width, imgs[0].height), (256, 256));
    assert_eq!(imgs[0].pixels.len(), 256 * 256 * 3, "RGB8 buffer");
    assert!(
        imgs[0].pixels.iter().any(|&p| p != 0),
        "rendered image should not be all-black"
    );
}
