//! Real-weight gen-core **Trainer contract** conformance for the candle `z_image_turbo` trainer
//! (sc-5166, epic 3720 / sc-4895) — the candle twin of `mlx-gen-z-image/tests/trainer_conformance.rs`.
//!
//! Drives the actual [`ZImageTrainer`](candle_gen_z_image) through the backend-neutral checks
//! (capability honesty, `TrainingProgress` monotonicity, typed cancellation before any step, registry
//! round-trip) — the same guarantees the MLX trainer is held to. `#[ignore]` + `cfg(feature = "cuda")`
//! because it needs the real `Tongyi-MAI/Z-Image-Turbo` weights (`Z_IMAGE_SNAPSHOT` or the HF cache)
//! and a CUDA GPU. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set Z_IMAGE_SNAPSHOT=C:\Users\…\models--Tongyi-MAI--Z-Image-Turbo\snapshots\<hash>
//! cargo test -p candle-gen-z-image --features cuda --release --test trainer_conformance -- --ignored --nocapture
//! ```
//!
//! `trainer_conformance` constructs a fresh trainer per `train()`-invoking check (the cancellation
//! paths + the progress run), because `train` is `&mut self`. The cheap profile keeps each cheap: a
//! 2-item 64px dataset, 2 steps.
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

// Force-link the provider so its `inventory::submit!` trainer registration survives into the test
// binary (the registry round-trip check resolves `z_image_turbo` through it).
use candle_gen_z_image as _;

use candle_gen::gen_core::{self, LoadSpec, TrainingItem, WeightsSource};
use gen_core_testkit::TrainerProfile;

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

/// Two solid-colour swatch PNGs + captions in `dir` (mirrors the trainer e2e dataset).
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

#[test]
#[ignore = "needs real Z-Image-Turbo weights (Z_IMAGE_SNAPSHOT or HF cache) + a CUDA GPU; run with --features cuda --ignored"]
fn z_image_turbo_trainer_satisfies_gen_core_contract() {
    assert_eq!(candle_gen_z_image::MODEL_ID, "z_image_turbo");
    let tmp = std::env::temp_dir().join("candle_z_image_trainer_conformance");
    let items = make_dataset(&tmp.join("data"));
    let profile = TrainerProfile::cheap(items, tmp.join("out"));
    let snap = snapshot();

    gen_core_testkit::trainer_conformance(
        || {
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            gen_core::load_trainer("z_image_turbo", &spec).expect("load z_image_turbo trainer")
        },
        &profile,
    );
}
