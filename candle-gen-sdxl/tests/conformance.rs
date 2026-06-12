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
//! - **Surface:** txt2img only — conditioning / LoRA / accel samplers are not advertised, so the
//!   worker keeps those shapes on the Python fallback (sc-3678) rather than the backend silently
//!   dropping a control.
//! - **dtype:** CLIP + UNet + VAE load f16 with the `madebyollin/sdxl-vae-fp16-fix` VAE; the VAE
//!   un-scale is the diffusers-correct 0.13025. These match diffusers' fp16 path, not a deviation.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
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
