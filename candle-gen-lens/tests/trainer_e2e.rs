//! End-to-end verification of the candle **Lens trainer** (sc-5147) — the production `LensTrainer`
//! driven through the gen-core registry exactly as the SceneWorks worker will, on real `microsoft/Lens`
//! weights + a real CUDA GPU. The candle twin of `candle-gen-z-image/tests/trainer_e2e.rs`, on the Lens
//! flow-match (no-negation) objective with the gpt-oss text front-end.
//!
//! `#[ignore]`d + `cfg(feature = "cuda")` — needs the real Lens snapshot (the HF cache, or
//! `LENS_BASE_SNAPSHOT`, including the ~40 GB gpt-oss encoder) and a CUDA GPU. **Run serially**
//! (`--test-threads=1`): each test loads the ~40 GB encoder (the determinism test, two runs), so the
//! cargo-default parallel runner would try to hold several at once and OOM even a 96 GB card. On the
//! Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set LENS_BASE_SNAPSHOT=C:\Users\…\models--microsoft--Lens\snapshots\<hash>
//! cargo test -p candle-gen-lens --features cuda --release --test trainer_e2e -- --ignored --nocapture --test-threads=1
//! ```
//!
//! What it proves:
//!  - **prepare→cache→train→save** lifecycle: a tiny captioned-PNG dataset is VAE-encoded (the neural
//!    encode shim, posterior mean) + gpt-oss-encoded (4 captured layers) and cached; training descends
//!    the flow-match velocity loss on the `transformer/` DiT; a bare-dotted PEFT (LoRA) / `lokr_w*`
//!    (LoKr) adapter is written over the fused attention targets.
//!  - **train→infer round-trip**: the adapter reloads through the REAL candle inference merge
//!    ([`candle_gen_lens::adapters::merge_adapters`]) onto the `transformer/` base — every trained target
//!    merges, nothing skipped — and a full `lens` `generate` with the adapter renders a finite,
//!    non-black image on the GPU.
//!  - **launch-portable determinism**: the same seed produces the same adapter, run to run.
//!
//! **On "parity vs torch":** cross-framework *numeric* parity is NOT a goal (different autograd / RNG /
//! the deterministic CPU-seeded noise of sc-3673). What IS tested is candle-internal determinism (same
//! seed ⇒ same adapter) and behaviour (converges on real data; reloads + renders). The numeric
//! cross-checks are the trainer's unit gates + the reconstruction-parity tests in `candle_gen::train::lora`.
#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::{
    self, AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, Image,
    LoadSpec, NetworkType, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};

/// The `microsoft/Lens` base snapshot dir — `LENS_BASE_SNAPSHOT` or the first HF-cache snapshot.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("LENS_BASE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .expect("USERPROFILE/HOME");
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--microsoft--Lens/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir (set LENS_BASE_SNAPSHOT to override)")
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

/// A tiny config: small rank, low resolution (64 → 4×4 latent → 16 image tokens), few steps. **bf16**
/// — the DiT's native dtype + what inference loads. Adapter factors / loss / grads stay f32.
fn config(network_type: NetworkType, steps: u32) -> TrainingConfig {
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
        gradient_checkpointing: true,
        ..Default::default()
    }
}

struct RunOut {
    losses: Vec<f32>,
    adapter_path: PathBuf,
}

/// Train through the registry and collect the per-step losses + the adapter path.
fn run(tmp: &Path, file_name: &str, network_type: NetworkType, steps: u32) -> RunOut {
    let items = make_dataset(tmp);
    assert_eq!(candle_gen_lens::MODEL_ID_BASE, "lens");

    let mut trainer =
        gen_core::load_trainer("lens", &LoadSpec::new(WeightsSource::Dir(snapshot())))
            .expect("lens candle trainer should be registered");

    let req = TrainingRequest {
        items,
        config: config(network_type, steps),
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
    assert!(losses.iter().all(|l| l.is_finite()), "no NaN/Inf losses");
    assert!(out.adapter_path.exists(), "adapter should be written");
    println!("[{file_name}] losses: {losses:?}");
    RunOut {
        losses,
        adapter_path: out.adapter_path,
    }
}

/// Median of a slice.
fn median(s: &[f32]) -> f32 {
    let mut v = s.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Assert the windowed loss **median** falls (first quarter vs last quarter). Median (not mean) because
/// the flow-match per-step loss is heavy-tailed in the sampled timestep.
fn assert_converged(tag: &str, losses: &[f32]) {
    let q = (losses.len() / 4).max(1);
    let (first_q, last_q) = (median(&losses[..q]), median(&losses[losses.len() - q..]));
    println!("[{tag}] loss-median {first_q:.5} -> {last_q:.5}");
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

/// Load the (possibly sharded) `transformer/` base tensors into a CPU map.
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

/// Reload the adapter through the REAL candle inference merge onto the `transformer/` base, asserting
/// every trained target merges and nothing is skipped — the train→infer round-trip.
fn assert_reloads(path: &Path, kind: AdapterKind) {
    let mut base = load_transformer_base(&snapshot());
    let report = candle_gen_lens::adapters::merge_adapters(
        &mut base,
        &[AdapterSpec::new(path.to_path_buf(), 1.0, kind)],
    )
    .unwrap_or_else(|e| panic!("adapter should reload through the inference merge: {e}"));
    assert!(
        report.merged > 0,
        "every trained target should merge ({report:?})"
    );
    assert_eq!(
        report.skipped_keys, 0,
        "no adapter key skipped ({report:?})"
    );
    println!("[reload] merged {} targets", report.merged);
}

/// Count the LoRA/LoKr targets in the adapter file (the `.lora_A.weight` / `.lokr_w1` keys).
fn count_targets(path: &Path, kind: AdapterKind) -> usize {
    let tensors = candle_gen::candle_core::safetensors::load(path, &Device::Cpu).unwrap();
    let suffix = match kind {
        AdapterKind::Lora => ".lora_A.weight",
        AdapterKind::Lokr => ".lokr_w1",
    };
    tensors.keys().filter(|k| k.ends_with(suffix)).count()
}

/// Load the `lens` generator with the adapter merged and run a tiny `generate`, asserting a finite,
/// correctly-sized, non-black image. Exercises the full merge→build→denoise→decode path on the GPU.
fn render_finite_with_adapter(path: &Path, kind: AdapterKind) {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![AdapterSpec::new(
        path.to_path_buf(),
        1.0,
        kind,
    )]);
    let g = gen_core::load("lens", &spec).expect("load lens generator with adapter");
    let req = GenerationRequest {
        prompt: "a solid colour swatch".into(),
        width: 256,
        height: 256,
        steps: Some(4),
        seed: Some(1),
        ..Default::default()
    };
    let out = g.generate(&req, &mut |_| {}).expect("adapted generate");
    let GenerationOutput::Images(images) = out else {
        panic!("expected images");
    };
    assert!(!images.is_empty(), "should render at least one image");
    for im in &images {
        assert_eq!((im.width, im.height), (256, 256));
        assert_eq!(im.pixels.len(), 256 * 256 * 3, "RGB8 buffer");
        assert!(
            im.pixels.iter().any(|&p| p != 0),
            "rendered image should not be all-black"
        );
    }
}

/// LoRA: trains + converges, writes a bare-dotted PEFT adapter over the fused attention targets,
/// reloads through the inference merge, and renders a finite image.
#[test]
#[ignore = "needs real microsoft/Lens weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn lens_trainer_lora_trains_reloads_and_renders() {
    let tmp = std::env::temp_dir().join("candle_lens_trainer_lora_e2e");
    let out = run(&tmp, "swatch_lora.safetensors", NetworkType::Lora, 64);
    assert_converged("lens-lora", &out.losses);

    assert_eq!(
        read_meta(&out.adapter_path)
            .get("networkType")
            .map(String::as_str),
        Some("lora")
    );
    let tensors =
        candle_gen::candle_core::safetensors::load(&out.adapter_path, &Device::Cpu).unwrap();
    assert!(
        tensors
            .keys()
            .any(|k| k.ends_with(".attn.img_qkv.lora_A.weight")),
        "adapter should carry bare-dotted fused-QKV LoRA keys"
    );
    let n = count_targets(&out.adapter_path, AdapterKind::Lora);
    assert!(n > 0, "adapter should contain LoRA targets");

    assert_reloads(&out.adapter_path, AdapterKind::Lora);
    render_finite_with_adapter(&out.adapter_path, AdapterKind::Lora);
    println!("[lens-lora] e2e OK — {n} targets reload + merge, image renders finite");
}

/// LoKr: trains + converges (slower Kronecker reparam ⇒ more steps), writes a LoKr adapter with
/// `decomposeFactor`, reloads, and renders.
#[test]
#[ignore = "needs real microsoft/Lens weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn lens_trainer_lokr_trains_reloads_and_renders() {
    let tmp = std::env::temp_dir().join("candle_lens_trainer_lokr_e2e");
    let out = run(&tmp, "swatch_lokr.safetensors", NetworkType::Lokr, 110);
    assert_converged("lens-lokr", &out.losses);

    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lokr"));
    assert!(meta.contains_key("decomposeFactor"));
    let n = count_targets(&out.adapter_path, AdapterKind::Lokr);
    assert!(n > 0, "adapter should contain LoKr factor keys");

    assert_reloads(&out.adapter_path, AdapterKind::Lokr);
    render_finite_with_adapter(&out.adapter_path, AdapterKind::Lokr);
    println!("[lens-lokr] e2e OK — {n} targets reload + merge, image renders finite");
}

/// Launch-portable determinism: the same seed produces the same adapter, run to run.
#[test]
#[ignore = "needs real microsoft/Lens weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn lens_trainer_same_seed_is_reproducible() {
    let a = run(
        &std::env::temp_dir().join("candle_lens_trainer_det_a"),
        "det_a.safetensors",
        NetworkType::Lora,
        4,
    );
    let b = run(
        &std::env::temp_dir().join("candle_lens_trainer_det_b"),
        "det_b.safetensors",
        NetworkType::Lora,
        4,
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
    println!("[lens-det] e2e OK — same seed reproduces the adapter bit-for-bit");
}

/// sc-8650: with a sample cadence set, the trainer renders preview images from the **in-progress
/// adapter** and emits them as [`TrainingProgress::Sample`] — each a valid, non-empty RGB8 bitmap. This
/// is the candle Lens twin of the MLX sc-5637 `*_emits_preview_samples` smokes. Lens uses real CFG, so
/// the preview render runs at a guidance > 1.
#[test]
#[ignore = "needs real microsoft/Lens weights + a GPU; run with --features cuda --release --ignored"]
fn lens_trainer_emits_preview_samples() {
    if !snapshot().exists() {
        eprintln!("skipping: set LENS_BASE_SNAPSHOT (or populate the HF cache)");
        return;
    }
    let tmp = std::env::temp_dir().join("candle_lens_trainer_samples_e2e");
    let items = make_dataset(&tmp);
    assert_eq!(candle_gen_lens::MODEL_ID_BASE, "lens");
    let mut trainer =
        gen_core::load_trainer("lens", &LoadSpec::new(WeightsSource::Dir(snapshot())))
            .expect("lens candle trainer should be registered");

    // 4 steps, render a preview every 2 steps over 2 prompts → 2 cadences × 2 prompts = 4 Sample events.
    let mut cfg = config(NetworkType::Lora, 4);
    cfg.sample_every = 2;
    cfg.sample_steps = 4;
    cfg.sample_guidance_scale = 4.0;
    cfg.sample_prompts = vec![
        "a solid red swatch".to_string(),
        "a solid blue swatch".to_string(),
    ];

    let req = TrainingRequest {
        items,
        config: cfg,
        output_dir: tmp.join("out"),
        file_name: "lens_samples.safetensors".to_string(),
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
            "[lens sample] step {step} {index}/{total} {}x{} prompt {prompt:?}",
            img.width, img.height
        );
    }
}
