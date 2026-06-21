//! gen-core contract conformance for the candle SDXL provider (sc-4481, epic 3720).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator.
//! This is the suite whose **seed-determinism** check is the regression guard for the spike's
//! repro defect (sc-3498) that sc-3673 fixed.
//!
//! It drives a real `generate`, so it needs the CUDA backend + a local SDXL snapshot and is
//! `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set SDXL_SNAPSHOT=C:\Users\…\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\<hash>
//! cargo test -p candle-gen-sdxl --features cuda --release --test conformance -- --ignored
//! ```
//!
//! ## SDXL parity (sc-3677)
//!
//! Both [`sdxl_conformance`] and [`realvisxl_conformance`] run the SAME suite against the SAME
//! `candle_gen_sdxl::load` — the worker maps both ids onto one engine and RealVisXL_V5.0 shares the
//! SDXL architecture + diffusers component layout, so the only input that varies is the snapshot dir.
//! Passing the suite *is* the parity evidence the story asks for (AC: realvisxl generates a correct
//! image on the Candle lane; parity tests pass): it locks **output dims** (the request's WxH is the
//! emitted image size), **seed semantics** (same request+seed ⇒ byte-identical output;
//! [`check_seed_determinism`]), **scheduler/steps/guidance defaults** (`Step.total` == resolved
//! steps; [`check_progress`]), **contract/sidecar field shape** (validate-honesty +
//! [`gen_core::GenerationOutput::Images`]), and **cancellation/progress** (typed `Canceled`, monotone
//! `Step`).
//!
//! **Accepted differences vs the Python `SdxlDiffusersAdapter` (documented, not bugs):**
//! - **Sampler:** the candle lane runs **DDIM (eta=0)** and advertises only `ddim`; the Python/MLX
//!   default is `euler_ancestral`. sc-3673 chose DDIM for launch-portable determinism (the spike's
//!   ancestral path was non-reproducible across launches, sc-3498). Both are SDXL-correct solvers;
//!   cross-backend *pixel* equality is explicitly NOT a goal (RNG algorithms differ).
//! - **Surface:** txt2img only — conditioning and the lcm/hyper accel samplers are not advertised, so
//!   the worker keeps those shapes on the Python fallback (sc-3678) rather than the backend silently
//!   dropping a control. (sc-6128 DID wire the few-step `lightning` sampler; the testkit's
//!   validate-honesty check therefore now also exercises `validate(sampler="lightning")`, and
//!   [`realvisxl_lightning_render`] is the real-weight non-degeneracy guard for the render itself.)
//! - **dtype:** CLIP + UNet + VAE load f16 with the `madebyollin/sdxl-vae-fp16-fix` VAE; the VAE
//!   un-scale is the diffusers-correct 0.13025. These match diffusers' fp16 path, not a deviation.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use gen_core_testkit::{conformance, Profile};

#[test]
#[ignore = "needs SDXL_SNAPSHOT (a diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn sdxl_conformance() {
    let snap = std::env::var("SDXL_SNAPSHOT")
        .expect("set SDXL_SNAPSHOT to a stabilityai/stable-diffusion-xl-base-1.0 snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 512² (≥ the descriptor's min_size 512) at a small step count keeps the suite's ~4 generate()
    // calls cheap — it verifies contract behavior, not image quality. `steps` must equal what the
    // model resolves req.steps to (check_progress asserts Step.total == profile.steps); the pipeline
    // uses req.steps verbatim, so 4 → 4.
    let profile = Profile {
        width: 512,
        height: 512,
        steps: 4,
        ..Profile::cheap()
    };

    // Resolve through THIS crate's `load` (its inventory registration is linked into the test binary,
    // so the suite's registry round-trip check also passes). Panics with aggregated failures.
    conformance(|| candle_gen_sdxl::load(&spec).unwrap(), &profile);
}

/// sc-3677: the same conformance suite against a **RealVisXL_V5.0** snapshot — the parity evidence
/// that `realvisxl` generates a correct image on the Candle lane. RealVisXL ships the standard
/// diffusers tree with the SAME `.fp16.safetensors` component filenames this pipeline loads
/// (`unet/diffusion_pytorch_model.fp16.safetensors`, `text_encoder{,_2}/model.fp16.safetensors`), so
/// it resolves through the identical `candle_gen_sdxl::load` path — no single-file loader is needed
/// (the diffusers component layout is present, not absent). Only the snapshot env var differs from
/// [`sdxl_conformance`]. See the module header for the accepted differences vs the Python adapter.
///
/// ```text
/// set REALVISXL_SNAPSHOT=C:\Users\…\models--SG161222--RealVisXL_V5.0\snapshots\<hash>
/// cargo test -p candle-gen-sdxl --features cuda --release --test conformance -- --ignored
/// ```
#[test]
#[ignore = "needs REALVISXL_SNAPSHOT (a RealVisXL_V5.0 diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn realvisxl_conformance() {
    let snap = std::env::var("REALVISXL_SNAPSHOT")
        .expect("set REALVISXL_SNAPSHOT to an SG161222/RealVisXL_V5.0 snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // Same cheap 512²/4-step profile as sdxl_conformance — this verifies contract parity, not image
    // quality; the human-eyeball check is the txt2img example pointed at a RealVisXL snapshot.
    let profile = Profile {
        width: 512,
        height: 512,
        steps: 4,
        ..Profile::cheap()
    };

    conformance(|| candle_gen_sdxl::load(&spec).unwrap(), &profile);
}

/// sc-6128 acceptance: the candle SDXL lightning path renders a **non-degenerate** image at ~5 steps
/// via `sampler="lightning"` — the automatable half of "RealVisXL Lightning renders correctly on
/// Windows" (image *quality* is the human eyeball via `examples/sdxl-txt2img.rs --sampler lightning`).
///
/// Gated on `REALVISXL_LIGHTNING_SNAPSHOT` (a distilled RealVisXL Lightning / SDXL-Lightning diffusers
/// snapshot dir) — base SDXL through this sampler at 5 steps would render undertrained mush, which is
/// exactly the failure this story guards against, so the test demands a Lightning checkpoint. It
/// asserts the output dims, that progress reaches the 5th step, and that the pixels are not a flat
/// constant (a collapsed/blank decode), i.e. the few-step schedule actually produced structure.
///
/// ```text
/// set REALVISXL_LIGHTNING_SNAPSHOT=C:\Users\…\models--…--RealVisXL-Lightning\snapshots\<hash>
/// cargo test -p candle-gen-sdxl --features cuda --release --test conformance -- --ignored realvisxl_lightning_render
/// ```
#[test]
#[ignore = "needs REALVISXL_LIGHTNING_SNAPSHOT (a distilled Lightning snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn realvisxl_lightning_render() {
    let snap = std::env::var("REALVISXL_LIGHTNING_SNAPSHOT").expect(
        "set REALVISXL_LIGHTNING_SNAPSHOT to a distilled RealVisXL Lightning / SDXL-Lightning \
         diffusers snapshot dir",
    );
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));
    let gen = candle_gen_sdxl::load(&spec).unwrap();

    let req = GenerationRequest {
        prompt: "a photo of a rusty robot holding a lit candle, cinematic lighting".into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(42),
        steps: Some(5),
        // The worker forces this for `realvisxl_lightning`; CFG is off in the engine regardless.
        sampler: Some("lightning".into()),
        guidance: Some(1.0),
        ..Default::default()
    };

    let mut last_step = (0u32, 0u32);
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            last_step = (current, total);
        }
    };
    let out = gen
        .generate(&req, &mut on_progress)
        .expect("lightning render");

    let images = match out {
        GenerationOutput::Images(imgs) => imgs,
        GenerationOutput::Video { .. } => panic!("expected images, got video"),
    };
    assert_eq!(images.len(), 1, "count=1 ⇒ one image");
    let img = &images[0];
    assert_eq!((img.width, img.height), (512, 512), "output dims = request");
    assert_eq!(img.pixels.len(), 512 * 512 * 3, "RGB8 buffer = W·H·3");
    // Progress reached the final (5th) step — the 5-step schedule actually ran.
    assert_eq!(last_step, (5, 5), "Step progress should end at 5/5");
    // Non-degenerate: a collapsed/blank decode is a flat constant. Real structure has spread.
    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    assert!(
        max - min > 16,
        "lightning render looks degenerate (flat): min={min} max={max}"
    );
}
