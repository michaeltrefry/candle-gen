//! End-to-end verification of the candle Wan2.2 **A14B MoE trainer** (sc-5167) — the production
//! `WanMoeTrainer` driven through the gen-core registry exactly as the SceneWorks worker will, on real
//! `Wan-AI/Wan2.2-T2V-A14B-Diffusers` weights + a real CUDA GPU. The candle twin of
//! `candle-gen-z-image/tests/trainer_e2e.rs`, on the dual-expert MoE flow-match objective.
//!
//! `#[ignore]`d + `cfg(feature = "cuda")` — needs the real A14B snapshot (the HF cache, or
//! `WAN_T2V_14B_SNAPSHOT`) and a CUDA GPU. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set WAN_T2V_14B_SNAPSHOT=C:\Users\…\models--Wan-AI--Wan2.2-T2V-A14B-Diffusers\snapshots\<hash>
//! cargo test -p candle-gen-wan --features cuda --release --test trainer_e2e -- --ignored --nocapture
//! ```
//!
//! What it proves:
//!  - **prepare→cache→train→save** lifecycle on the **dual-expert MoE**: a tiny captioned-PNG dataset is
//!    z16-VAE-mean / UMT5-encoded and cached; training alternates the high-noise (`transformer/`) and
//!    low-noise (`transformer_2/`) experts, each descending the flow-match velocity loss on its own
//!    timestep band; a `{stem}.high_noise` / `{stem}.low_noise` PEFT/LoKr pair is written with the
//!    expected bare-dotted DiT keys + metadata.
//!  - **train→infer round-trip**: each expert's adapter reloads through the REAL candle inference merge
//!    ([`candle_gen_wan::adapters::merge_adapters`]) onto its matching base (high→`transformer/`,
//!    low→`transformer_2/`) — every trained target merges, nothing skipped — and a full A14B `generate`
//!    with the high/low pair (tagged by [`gen_core::MoeExpert`]) renders finite video frames on the GPU.
//!  - **launch-portable determinism**: the same seed produces the same adapter pair, run to run.
//!
//! **On "parity vs torch/MLX":** cross-framework *numeric* parity is NOT a goal (different autograd /
//! RNG / the deterministic CPU-seeded noise of sc-3673). What IS tested is candle-internal determinism
//! (same seed ⇒ same adapter) and behaviour (converges on real data; reloads + renders). The numeric
//! cross-checks are the trainer's unit gates + the reconstruction-parity tests in `candle_gen::train::lora`.
#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::{
    self, AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, Image,
    LoadSpec, MoeExpert, NetworkType, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest, WeightsSource,
};

/// The Wan A14B (T2V) base snapshot dir — `WAN_T2V_14B_SNAPSHOT` or the first HF-cache snapshot.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("WAN_T2V_14B_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .expect("USERPROFILE/HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir (set WAN_T2V_14B_SNAPSHOT to override)")
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

/// A tiny config: small rank, low resolution (64 → 8×8 latent → 4×4=16 image tokens), few steps.
/// **bf16** — the experts' native dtype and what inference loads (f32 would be ~56 GB for two 14B
/// experts). Adapter factors / loss / grads stay f32 (master weights).
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
    high_path: PathBuf,
    low_path: PathBuf,
}

/// The low-noise sibling path of a `{stem}.high_noise.{ext}` adapter (the trainer writes the pair;
/// `TrainingOutput.adapter_path` reports the high-noise file).
fn low_sibling(high: &Path) -> PathBuf {
    let s = high
        .to_string_lossy()
        .replace(".high_noise.", ".low_noise.");
    PathBuf::from(s)
}

/// Train through the registry and collect the per-step losses + the high/low adapter paths.
fn run(tmp: &Path, file_name: &str, network_type: NetworkType, steps: u32) -> RunOut {
    let items = make_dataset(tmp);
    assert_eq!(candle_gen_wan::config::MODEL_ID_T2V_14B, "wan2_2_t2v_14b");

    let mut trainer = gen_core::load_trainer(
        "wan2_2_t2v_14b",
        &LoadSpec::new(WeightsSource::Dir(snapshot())),
    )
    .expect("wan2_2_t2v_14b candle trainer should be registered");

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
    let high_path = out.adapter_path.clone();
    let low_path = low_sibling(&high_path);
    assert!(high_path.exists(), "high_noise adapter should be written");
    assert!(low_path.exists(), "low_noise adapter should be written");
    println!("[{file_name}] losses: {losses:?}");
    RunOut {
        losses,
        high_path,
        low_path,
    }
}

/// Median of a slice.
fn median(s: &[f32]) -> f32 {
    let mut v = s.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Assert the windowed loss **median** falls for **each expert's** stream. Training alternates experts
/// (odd step → high, even → low), so the combined stream is bimodal (the high-noise band has larger
/// targets → larger loss); split it and check each expert's first-quarter vs last-quarter median falls.
/// Median (not mean) because the flow-match per-step loss is heavy-tailed in the sampled σ.
fn assert_converged_per_expert(tag: &str, losses: &[f32]) {
    let high: Vec<f32> = losses.iter().step_by(2).copied().collect(); // steps 1,3,5… (odd → high)
    let low: Vec<f32> = losses.iter().skip(1).step_by(2).copied().collect(); // steps 2,4,6… (low)
    for (which, s) in [("high", &high), ("low", &low)] {
        let q = (s.len() / 4).max(1);
        let (first_q, last_q) = (median(&s[..q]), median(&s[s.len() - q..]));
        println!("[{tag}/{which}] loss-median {first_q:.5} -> {last_q:.5}");
        assert!(
            last_q < first_q * 0.9,
            "[{tag}/{which}] windowed loss-median should fall on real data: {first_q:.5} -> {last_q:.5}"
        );
    }
}

/// The adapter file's header metadata.
fn read_meta(path: &Path) -> HashMap<String, String> {
    let bytes = std::fs::read(path).unwrap();
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    md.metadata().clone().unwrap_or_default()
}

/// Load one (possibly sharded) expert's base tensors into a CPU map.
fn load_expert_base(snap: &Path, sub: &str) -> HashMap<String, Tensor> {
    let dir = snap.join(sub);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|_| panic!("{sub}/ dir"))
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

/// Reload each expert's adapter through the REAL candle inference merge onto its matching base
/// (high→`transformer/`, low→`transformer_2/`), asserting every trained target merges and nothing is
/// skipped — the per-expert train→infer round-trip.
fn assert_reloads_pair(high: &Path, low: &Path, kind: AdapterKind) {
    for (sub, path) in [("transformer", high), ("transformer_2", low)] {
        let mut base = load_expert_base(&snapshot(), sub);
        let report = candle_gen_wan::adapters::merge_adapters(
            &mut base,
            &[AdapterSpec::new(path.to_path_buf(), 1.0, kind)],
        )
        .unwrap_or_else(|e| panic!("{sub} adapter should reload through the inference merge: {e}"));
        assert!(
            report.merged > 0,
            "{sub}: every trained target should merge ({report:?})"
        );
        assert_eq!(
            report.skipped_keys, 0,
            "{sub}: no adapter key skipped ({report:?})"
        );
        println!("[reload/{sub}] merged {} targets", report.merged);
    }
}

/// Count the LoRA/LoKr targets in one adapter file (the `.lora_A.weight` / `.lokr_w1` keys).
fn count_targets(path: &Path, kind: AdapterKind) -> usize {
    let tensors = candle_gen::candle_core::safetensors::load(path, &Device::Cpu).unwrap();
    let suffix = match kind {
        AdapterKind::Lora => ".lora_A.weight",
        AdapterKind::Lokr => ".lokr_w1",
    };
    tensors.keys().filter(|k| k.ends_with(suffix)).count()
}

/// Load the A14B generator with the high/low adapter pair merged (tagged by [`MoeExpert`]) and run a
/// tiny `generate`, asserting finite, correctly-sized, non-black video frames. Exercises the full
/// per-expert merge→build→denoise→decode path on the GPU.
fn render_finite_with_pair(high: &Path, low: &Path, kind: AdapterKind) {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![
        AdapterSpec::new(high.to_path_buf(), 1.0, kind).with_moe_expert(MoeExpert::High),
        AdapterSpec::new(low.to_path_buf(), 1.0, kind).with_moe_expert(MoeExpert::Low),
    ]);
    let g = candle_gen_wan::wan14b::load_t2v_14b(&spec).expect("load A14B generator with adapters");
    let req = GenerationRequest {
        prompt: "a solid colour swatch".into(),
        width: 256,
        height: 256,
        steps: Some(4),
        frames: Some(1),
        seed: Some(1),
        ..Default::default()
    };
    let out = g.generate(&req, &mut |_| {}).expect("adapted generate");
    let GenerationOutput::Video { frames, .. } = out else {
        panic!("expected video frames");
    };
    assert!(!frames.is_empty(), "should render at least one frame");
    for f in &frames {
        assert_eq!((f.width, f.height), (256, 256));
        assert_eq!(f.pixels.len(), 256 * 256 * 3, "RGB8 buffer");
        assert!(
            f.pixels.iter().any(|&p| p != 0),
            "rendered frame should not be all-black"
        );
    }
}

/// LoRA (MoE): both experts train + converge, write a bare-dotted PEFT high/low pair, reload + render.
#[test]
#[ignore = "needs real Wan2.2-T2V-A14B weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn wan_t2v_14b_trainer_lora_trains_reloads_and_renders() {
    let tmp = std::env::temp_dir().join("candle_wan_trainer_lora_e2e");
    // 96 steps ⇒ 48 micro-steps per expert — enough for a clear median fall on each band.
    let out = run(&tmp, "swatch_lora.safetensors", NetworkType::Lora, 96);
    assert_converged_per_expert("wan-lora", &out.losses);

    for p in [&out.high_path, &out.low_path] {
        assert_eq!(
            read_meta(p).get("networkType").map(String::as_str),
            Some("lora")
        );
        let tensors = candle_gen::candle_core::safetensors::load(p, &Device::Cpu).unwrap();
        assert!(
            tensors
                .keys()
                .any(|k| k.ends_with(".attn1.to_q.lora_A.weight")),
            "adapter should carry bare-dotted attention LoRA keys: {}",
            p.display()
        );
    }
    let n_high = count_targets(&out.high_path, AdapterKind::Lora);
    let n_low = count_targets(&out.low_path, AdapterKind::Lora);
    assert!(
        n_high > 0 && n_high == n_low,
        "both experts adapt the same target set"
    );

    assert_reloads_pair(&out.high_path, &out.low_path, AdapterKind::Lora);
    render_finite_with_pair(&out.high_path, &out.low_path, AdapterKind::Lora);
    println!("[wan-lora] e2e OK — {n_high} targets/expert reload + merge, video renders finite");
}

/// LoKr (MoE): both experts train + converge, write a LoKr high/low pair (with `decomposeFactor`),
/// reload + render.
#[test]
#[ignore = "needs real Wan2.2-T2V-A14B weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn wan_t2v_14b_trainer_lokr_trains_reloads_and_renders() {
    let tmp = std::env::temp_dir().join("candle_wan_trainer_lokr_e2e");
    // LoKr's Kronecker reparam descends slower than LoRA, so give each expert more micro-steps.
    let out = run(&tmp, "swatch_lokr.safetensors", NetworkType::Lokr, 160);
    assert_converged_per_expert("wan-lokr", &out.losses);

    for p in [&out.high_path, &out.low_path] {
        let meta = read_meta(p);
        assert_eq!(meta.get("networkType").map(String::as_str), Some("lokr"));
        assert!(meta.contains_key("decomposeFactor"));
    }
    let n_high = count_targets(&out.high_path, AdapterKind::Lokr);
    assert!(n_high > 0, "adapter should contain LoKr factor keys");

    assert_reloads_pair(&out.high_path, &out.low_path, AdapterKind::Lokr);
    render_finite_with_pair(&out.high_path, &out.low_path, AdapterKind::Lokr);
    println!("[wan-lokr] e2e OK — {n_high} targets/expert reload + merge, video renders finite");
}

/// Launch-portable determinism: the same seed produces the same adapter pair, run to run.
#[test]
#[ignore = "needs real Wan2.2-T2V-A14B weights + a CUDA GPU; run with --features cuda --release --ignored"]
fn wan_t2v_14b_trainer_same_seed_is_reproducible() {
    let a = run(
        &std::env::temp_dir().join("candle_wan_trainer_det_a"),
        "det_a.safetensors",
        NetworkType::Lora,
        4,
    );
    let b = run(
        &std::env::temp_dir().join("candle_wan_trainer_det_b"),
        "det_b.safetensors",
        NetworkType::Lora,
        4,
    );
    assert_eq!(
        a.losses, b.losses,
        "same-seed runs should give identical losses"
    );
    for (pa, pb) in [(&a.high_path, &b.high_path), (&a.low_path, &b.low_path)] {
        let ta = candle_gen::candle_core::safetensors::load(pa, &Device::Cpu).unwrap();
        let tb = candle_gen::candle_core::safetensors::load(pb, &Device::Cpu).unwrap();
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
    }
    println!("[wan-det] e2e OK — same seed reproduces the adapter pair bit-for-bit");
}

/// sc-8650: with a sample cadence set, the dual-expert MoE trainer renders preview images from the
/// **in-progress** experts (both carrying their live adapters) and emits them as
/// [`TrainingProgress::Sample`] — each a valid, non-empty RGB8 bitmap. Wan T2V's preview is a single
/// still frame (`F = 1`, already squeezed to RGB8), so a Sample is a normal [`Image`] just like the
/// image families. The candle twin of the MLX sc-5637 `*_emits_preview_samples` smokes.
///
/// CFG-free: the Wan trainer pre-encodes only the positive caption, so `sample_guidance_scale` is
/// ignored — guidance 1.0. The trainer's main loop runs `for step in 1..=cfg.steps` (one iteration
/// per overall step, experts alternating *within* the loop), and the preview block fires once per step
/// when `step % sample_every == 0`, iterating over **all** prompts. So with `steps = 4`,
/// `sample_every = 2`, 2 prompts → cadence steps {2, 4} × 2 prompts = **exactly 4** Sample events. We
/// assert that exact count (the loop is deterministic — one preview block per cadence step, prompts
/// iterated fully).
#[test]
#[ignore = "needs real Wan A14B T2V weights + a GPU; run with --features cuda --release --ignored"]
fn wan_trainer_emits_preview_samples() {
    if !snapshot().exists() {
        eprintln!("skipping: set WAN_T2V_14B_SNAPSHOT (or populate the HF cache)");
        return;
    }
    let tmp = std::env::temp_dir().join("candle_wan_trainer_samples_e2e");
    let items = make_dataset(&tmp);
    // Reference the provider crate so its `inventory::submit!` trainer registration is linked in.
    candle_gen_wan::force_link();
    let mut trainer = gen_core::load_trainer(
        "wan2_2_t2v_14b",
        &LoadSpec::new(WeightsSource::Dir(snapshot())),
    )
    .expect("wan2_2_t2v_14b candle trainer should be registered");

    // 4 steps, render a preview every 2 steps over 2 prompts → 2 cadences × 2 prompts = 4 Sample events.
    let mut cfg = config(NetworkType::Lora, 4);
    cfg.sample_every = 2;
    cfg.sample_steps = 4;
    cfg.sample_guidance_scale = 1.0; // CFG-free preview (only the positive caption is cached)
    cfg.sample_prompts = vec![
        "a solid red swatch".to_string(),
        "a solid blue swatch".to_string(),
    ];

    let req = TrainingRequest {
        items,
        config: cfg,
        output_dir: tmp.join("out"),
        file_name: "wan_samples.safetensors".to_string(),
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
        // A Wan T2V preview is a single still frame, already squeezed to an RGB8 `Image`.
        assert!(img.width > 0 && img.height > 0, "non-empty preview dims");
        assert_eq!(
            img.pixels.len(),
            (img.width * img.height * 3) as usize,
            "RGB8 row-major still frame: pixels.len() == w*h*3"
        );
        assert!(
            img.pixels.iter().any(|&b| b != 0),
            "preview from the in-progress dual-expert MoE is not all-black (prompt {prompt:?})"
        );
        println!(
            "[wan sample] step {step} {index}/{total} {}x{} prompt {prompt:?}",
            img.width, img.height
        );
    }
}
