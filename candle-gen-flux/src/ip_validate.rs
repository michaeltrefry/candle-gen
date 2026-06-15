//! FLUX XLabs IP-Adapter real-weight GPU validation (sc-5872, epic 5480) — an env-driven, `#[ignore]`d
//! integration test that drives the REAL [`IpAdapterFlux`] stack on the deployed hardware (a FLUX.1
//! snapshot + `XLabs-AI/flux-ip-adapter` + the CLIP ViT-L image encoder + a reference image). The
//! analog of the SDXL/Kolors IP-Adapter Phase-5 harnesses.
//!
//! **Quantitative gate (no extra deps).** The XLabs adapter conditions on CLIP image features, so the
//! metric is the CLIP-feature cosine between the reference and the generated output (using a ViT-L
//! tower, the one the adapter conditions on). We generate twice at one seed: **with** IP
//! (`ip_adapter_scale > 0`) and **without** (`ip_adapter_scale = 0` → the forked DiT is byte-identical
//! to plain FLUX), and assert the IP run's reference-cosine is meaningfully higher. Plus the cancel
//! contract (pre + mid-denoise). Outputs are written as PPM for eyeballing.
//!
//! Run (after deploying weights into a local dir):
//! ```text
//! set IP_FLUX_BASE=...\FLUX.1-dev               # BFL snapshot (flux1-dev.safetensors, ae.safetensors, …)
//! set IP_FLUX_ADAPTER=...\ip_adapter.safetensors
//! set IP_FLUX_IMAGE_ENCODER=...\clip-vit-large-patch14   # dir with model.safetensors (or the file)
//! set IP_FLUX_REF=...\ref.ppm                   # a reference image (P6 PPM)
//! set IP_FLUX_OUT=...\out                        # output dir
//! cargo test -p candle-gen-flux --features cuda --release ip_validate::real_weight -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_core::DType;

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, Progress};
use candle_gen_sdxl::ip_adapter::preprocess_clip_image_sized;
use candle_gen_sdxl::vision_encoder::{ClipVisionEncoder, VisionConfig};
use candle_gen_sdxl::weights::Weights;

use crate::ip_provider::{IpAdapterFlux, IpAdapterFluxPaths, IpAdapterFluxRequest};

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
}

/// Minimal P6 PPM reader, tolerant of a single comment line — enough for hand-prepared reference images.
fn read_ppm(path: &Path) -> Image {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut i = 0usize;
    let mut tok = || -> String {
        loop {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            } else {
                break;
            }
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        String::from_utf8_lossy(&bytes[start..i]).to_string()
    };
    assert_eq!(tok(), "P6", "not a binary PPM");
    let w: usize = tok().parse().expect("ppm width");
    let h: usize = tok().parse().expect("ppm height");
    let _max: usize = tok().parse().expect("ppm maxval");
    i += 1; // single whitespace after maxval
    let pixels = bytes[i..i + w * h * 3].to_vec();
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

fn write_ppm(path: &Path, img: &Image) {
    let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    out.extend_from_slice(&img.pixels);
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// A standalone CLIP ViT-L feature extractor for the cosine metric: preprocess → penultimate →
/// mean-pool over tokens → L2-normalize. Returns a 1024-vec.
struct ClipMetric {
    encoder: ClipVisionEncoder,
    size: usize,
    device: candle_core::Device,
}

impl ClipMetric {
    fn load(image_encoder: &Path) -> Self {
        let device = candle_gen::default_device().unwrap();
        let cfg = VisionConfig::vit_l_14();
        let path = if image_encoder.is_file() {
            image_encoder.to_path_buf()
        } else {
            ["model.safetensors", "model.fp16.safetensors"]
                .iter()
                .map(|n| image_encoder.join(n))
                .find(|p| p.is_file())
                .unwrap_or_else(|| panic!("no model.safetensors under {}", image_encoder.display()))
        };
        let w = Weights::from_file(&path, &device, DType::F32).unwrap();
        let encoder = ClipVisionEncoder::from_weights(&w, &cfg).unwrap();
        Self {
            encoder,
            size: cfg.image_size,
            device,
        }
    }

    fn feature(&self, img: &Image) -> Vec<f32> {
        let px = preprocess_clip_image_sized(img, self.size, &self.device).unwrap();
        let penult = self.encoder.penultimate(&px).unwrap(); // [1, N, 1024]
        let pooled = penult.mean(1).unwrap().flatten_all().unwrap(); // [1024]
        let v = pooled
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / norm).collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Drive the real FLUX IP-Adapter stack: a with-IP vs no-IP ablation (the IP run must score a higher
/// reference cosine) + the cancel contract. Visually-inspectable PPMs land in `IP_FLUX_OUT`.
#[test]
#[ignore = "real-weight GPU validation; set IP_FLUX_BASE/IP_FLUX_ADAPTER/IP_FLUX_IMAGE_ENCODER/IP_FLUX_REF/IP_FLUX_OUT"]
fn real_weight_ip_adapter() {
    let out_dir = env_path("IP_FLUX_OUT");
    std::fs::create_dir_all(&out_dir).ok();
    let image_encoder = env_path("IP_FLUX_IMAGE_ENCODER");

    let paths = IpAdapterFluxPaths {
        flux_base: env_path("IP_FLUX_BASE"),
        ip_adapter: env_path("IP_FLUX_ADAPTER"),
        image_encoder: image_encoder.clone(),
    };
    let reference = read_ppm(&env_path("IP_FLUX_REF"));
    println!(
        "reference {}x{}; loading IpAdapterFlux …",
        reference.width, reference.height
    );

    let t0 = std::time::Instant::now();
    let model = IpAdapterFlux::load(&paths).expect("load IpAdapterFlux");
    println!("loaded in {:?}", t0.elapsed());

    let base = IpAdapterFluxRequest {
        prompt: "a cinematic portrait photo, soft natural light, photorealistic, sharp focus"
            .into(),
        width: 1024,
        height: 1024,
        steps: 20,
        guidance: 3.5,
        ip_adapter_scale: 0.7,
        seed: 12345,
        cancel: CancelFlag::new(),
    };

    let mut noop = |_p: Progress| {};

    // With IP.
    let t = std::time::Instant::now();
    let out_ip = model
        .generate(&base, &reference, &mut noop)
        .expect("generate (ip)");
    println!("[ip] {:?}", t.elapsed());
    write_ppm(&out_dir.join("flux_ip.ppm"), &out_ip);

    // Without IP (scale 0 → forked DiT byte-identical to plain FLUX at the same seed/prompt).
    let plain_req = IpAdapterFluxRequest {
        ip_adapter_scale: 0.0,
        ..base.clone()
    };
    let t = std::time::Instant::now();
    let out_plain = model
        .generate(&plain_req, &reference, &mut noop)
        .expect("generate (no ip)");
    println!("[no-ip] {:?}", t.elapsed());
    write_ppm(&out_dir.join("flux_no_ip.ppm"), &out_plain);

    // CLIP-feature cosine to the reference: the IP run must pull meaningfully closer.
    let metric = ClipMetric::load(&image_encoder);
    let ref_feat = metric.feature(&reference);
    let cos_ip = cosine(&ref_feat, &metric.feature(&out_ip));
    let cos_plain = cosine(&ref_feat, &metric.feature(&out_plain));
    println!("=== FLUX XLabs IP-Adapter validation ===");
    println!("  clip cosine (ip)    : {cos_ip:.4}");
    println!("  clip cosine (no-ip) : {cos_plain:.4}");
    println!("  delta               : {:.4}", cos_ip - cos_plain);
    println!("  outputs: {}", out_dir.display());

    // Pre-cancel.
    let cancelled = IpAdapterFluxRequest {
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..base.clone()
    };
    let pre = model.generate(&cancelled, &reference, &mut noop);
    assert!(
        matches!(pre, Err(candle_gen::CandleError::Canceled)),
        "pre-cancel must return Canceled"
    );
    println!("[cancel:pre] Err(Canceled) ✓");

    // Mid-denoise cancel: flip the flag from the progress callback on the 3rd step.
    let mid = CancelFlag::new();
    let mid_req = IpAdapterFluxRequest {
        cancel: mid.clone(),
        ..base.clone()
    };
    let seen = Arc::new(AtomicUsize::new(0));
    let seen_cb = seen.clone();
    let mut cancel_at_3 = move |p: Progress| {
        if let Progress::Step { current, .. } = p {
            seen_cb.store(current as usize, Ordering::SeqCst);
            if current >= 3 {
                mid.cancel();
            }
        }
    };
    let res = model.generate(&mid_req, &reference, &mut cancel_at_3);
    assert!(
        matches!(res, Err(candle_gen::CandleError::Canceled)),
        "mid-cancel must return Canceled"
    );
    let steps_seen = seen.load(Ordering::SeqCst);
    assert!(
        (3..=4).contains(&steps_seen),
        "mid-cancel should stop right after step 3 (saw {steps_seen})"
    );
    println!("[cancel:mid] Err(Canceled) after {steps_seen} steps ✓");

    // The gate: IP conditioning pulls the output toward the reference in CLIP space.
    assert!(
        cos_ip > cos_plain + 0.02,
        "IP run cosine {cos_ip:.4} not meaningfully above no-IP {cos_plain:.4}"
    );
    assert!(out_ip.width == 1024 && out_ip.height == 1024);
    println!("FLUX XLabs IP-Adapter validation PASS ✅");
}
