//! SD3.5 memory profiling → `minMemoryGb` (sc-7879, epic 7982).
//!
//! The worker's model-eligibility gate routes a request to a backend only when the device has at
//! least `minMemoryGb` of free VRAM. C5 (sc-7880) wires the candle SD3.5 manifest; this module is the
//! single source of truth it consumes — a **principled** estimate per variant × precision, built from
//! the actual quantized-tensor parameter counts and then **calibrated against the C6 (sc-7881)
//! measured CUDA peaks** so the gate is a safe ceiling, not a guess.
//!
//! ## Methodology
//!
//! ```text
//! minMemoryGb  =  ( quantized DiT weight bytes        // the projections C4 folds to Q4/Q8
//!               +   dense-kept DiT leaf bytes          // AdaLN/timestep/patch-embed/pos_embed — always bf16
//!               +   text-encoder bytes                 // CLIP-L + CLIP-bigG + T5-XXL, kept DENSE (bf16)
//!               +   VAE bytes                          // 16-ch AutoencoderKL, kept DENSE (bf16)
//!               +   ACTIVATION_OVERHEAD_GIB )          // FIXED dual-CFG working set + CUDA context (additive!)
//!               *   (1 + FRAG_MARGIN)                  // allocator-fragmentation slack on the whole footprint
//! ```
//!
//! - **Quantized DiT weights** use [`crate::quant::bytes_per_param`] (block-scale-inclusive: Q4 ≈
//!   0.5625 B/param, Q8 ≈ 1.0625, bf16 = 2.0).
//! - **Text encoders + VAE are kept dense (bf16).** This matches the established candle precedent —
//!   Lens/FLUX.2 quantize the DiT (and, for FLUX.2-dev, the giant TE) but the SD3.5 encoders/VAE are
//!   small relative to the DiT and chaos-sensitive, so C4 quantizes the **DiT only**; the TE/VAE
//!   bytes are a fixed dense addend across all three precisions.
//! - **Working set is ADDITIVE, not multiplicative.** C6 (sc-7881), re-measured under sc-8504's
//!   CPU-stage load, measured the true CUDA peak of a real 1024² dual-CFG render (`nvidia-smi`, idle
//!   baseline, RTX PRO 6000 Blackwell). The peak minus the resident weights is a near-**fixed**
//!   ~12.0-13.6 GiB **regardless of variant or precision** — it is the two CFG forwards' activations +
//!   attention scores at 1MP + the CUDA context, which scale with *resolution*, not model size. A
//!   multiplicative headroom (the original
//!   `HEADROOM = 0.30`) therefore had the wrong shape: 0.30 over-covered Large (which needed ~0.49)
//!   but badly **under**-covered Medium (which needed ~0.81 — Medium's smaller weights make the same
//!   fixed ~12.3 GiB a far larger *fraction*), so the gate could admit a Medium request that then
//!   OOMs. We model it as a fixed [`ACTIVATION_OVERHEAD_GIB`] addend plus a small [`FRAG_MARGIN`].
//!
//! ## Measured peaks vs the gate (sc-8504 CPU-stage re-measurement; real weights)
//!
//! ```text
//! variant / precision   resident weights   measured peak   this gate (minMemoryGb)
//! Large  / bf16             25.7 GiB          38.45 GiB          43.0 GiB
//! Large  / Q8               21.0 GiB          34.69 GiB          37.9 GiB
//! Large  / Q4               18.6 GiB          31.60 GiB          35.2 GiB
//! Medium / bf16             15.2 GiB          27.41 GiB          31.5 GiB
//! Medium / Q8               13.9 GiB          25.97 GiB          30.1 GiB
//! Medium / Q4               13.2 GiB          25.22 GiB          29.4 GiB
//! ```
//!
//! Every gate is >= the measured peak with >= 3.2 GiB / >= 8.4 % margin — a safe ceiling, re-validated
//! on the CUDA box, not silently raised. The [`tests::min_memory_gate_exceeds_measured_peaks`] test
//! pins this so a future tweak to the constants can't quietly drop the gate below a measured peak.
//!
//! **NOTE — the load transient is no longer the binding peak (and never was).** `Pipeline::load_components`
//! now **CPU-stages** the quantize (sc-8504, the FLUX.2-dev pattern): it builds the dense DiT on a
//! *CPU* VarBuilder and `Sd3Transformer::quantize_onto`s the GPU, so the dense projection weights
//! never land on the GPU — only the (small) `Q4_0`/`Q8_0` blocks do. Before this (the sc-7879 in-place
//! path) the dense bf16 DiT was built on-device and folded in place, so dense + quantized briefly
//! coexisted, leaving a high-water residue in the caching allocator. The sc-8504 re-measurement
//! confirms removing it: the **bf16** peaks are unchanged (38.45 / 27.41 — that path is untouched)
//! while the **quantized** peaks dropped ~0.5-1.0 GiB (Large Q4 32.57→31.60, Large Q8 35.22→34.69,
//! Medium Q4 25.97→25.22, Medium Q8 26.60→25.97). The implied activation overhead (peak − resident
//! weights) is now at most ~13.6 GiB (Large Q8), still under the fixed [`ACTIVATION_OVERHEAD_GIB`] =
//! 14.0 budget — so the gate constants are **unchanged**: the binding peak was always the
//! resolution-bound render activation set, not the load, and the residue removed (~0.5-1.0 GiB) is
//! below the ~1.5 GiB the budget already absorbed. The gate is left intact (a conservative ceiling)
//! rather than tightened by a fraction of a GiB.
//!
//! The figures are exposed as [`min_memory_gb`] (and the per-component breakdown [`MemoryProfile`])
//! so C5 can read them in code; the same numbers are tabulated in the PR.

use candle_gen::gen_core::Quant;

use crate::config::Sd3Config;
use crate::pipeline::Variant;
use crate::quant::bytes_per_param;

/// bf16 is 2 bytes per parameter — the dense baseline precision SD3.5 loads at.
const BF16_BYTES_PER_PARAM: f64 = 2.0;

/// **Fixed** activation working set + CUDA context, in GiB — an *additive* term, not a fraction of
/// weights. Measured `peak - resident_weights ≈ 12.0-13.6 GiB` across every variant × precision on a
/// real 1024² dual-CFG render (see the module doc table); it is the two CFG forwards' activations +
/// attention scores at 1MP + the CUDA context, which track *resolution*, not model size. 14.0 covers
/// the worst observed (Large Q8, ~13.6 GiB on the sc-8504 CPU-stage re-measurement; ~14.2 on the old
/// in-place path, which was inflated by the dense-build residue CPU-staging now removes). Left at 14.0
/// — the binding peak was always the render activation set, not the load, and the residue removed is
/// below the headroom 14.0 already carries, so there is no measured justification to tighten it.
/// Raising this is how you would cover a higher default render resolution.
pub const ACTIVATION_OVERHEAD_GIB: f64 = 14.0;

/// Allocator-fragmentation slack applied to the *whole* footprint (weights + activation overhead).
/// Small and multiplicative: fragmentation scales with the number/size of live allocations, so a
/// fraction of the total is the right shape (whereas the activation working set is fixed — hence the
/// additive [`ACTIVATION_OVERHEAD_GIB`]). 0.08 keeps every gate >= 7.5 % above the measured peak.
pub const FRAG_MARGIN: f64 = 0.08;

/// 1 GiB in bytes (the `minMemoryGb` unit is gibibytes, matching the worker's free-VRAM probe).
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The precision a variant is loaded at — dense bf16, or DiT-quantized Q8/Q4. The text encoders and
/// VAE stay dense (bf16) in every case; only the MMDiT projections quantize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Dense bf16 (no quantization).
    Bf16,
    /// DiT projections folded to GGUF `Q8_0`.
    Q8,
    /// DiT projections folded to GGUF `Q4_0`.
    Q4,
}

impl Precision {
    /// The optional [`Quant`] this precision applies to the DiT (`None` for dense bf16).
    pub fn quant(self) -> Option<Quant> {
        match self {
            Precision::Bf16 => None,
            Precision::Q8 => Some(Quant::Q8),
            Precision::Q4 => Some(Quant::Q4),
        }
    }

    /// Bytes-per-parameter for the **quantized** DiT projections at this precision.
    fn dit_bytes_per_param(self) -> f64 {
        match self.quant() {
            Some(q) => bytes_per_param(q),
            None => BF16_BYTES_PER_PARAM,
        }
    }
}

/// A per-component parameter-count breakdown of an SD3.5 variant, in **parameters** (not bytes).
/// Built from [`Sd3Config`] geometry ([`dit_param_counts`]) plus the fixed text-encoder / VAE counts
/// (the published SD3.5 component sizes — they are architecture constants, identical across variants).
#[derive(Debug, Clone, Copy)]
pub struct ParamCounts {
    /// DiT projection params that C4 **quantizes** (attention q/k/v/out, joint add_*, GELU MLP,
    /// `attn2`, `ff_context`, `context_embedder`, `proj_out`).
    pub dit_quantized: u64,
    /// DiT leaf params kept **dense** (AdaLN modulation linears, timestep/text embedders,
    /// patch-embed conv, the learned `pos_embed` table, norms).
    pub dit_dense: u64,
    /// Triple text-encoder params (CLIP-L ≈ 123M + OpenCLIP bigG ≈ 695M + T5-XXL ≈ 4.76B), kept dense.
    pub text_encoders: u64,
    /// 16-channel AutoencoderKL params (≈ 84M), kept dense.
    pub vae: u64,
}

/// Published dense parameter counts for the SD3.5 conditioning + VAE — architecture constants shared
/// across Large / Turbo / Medium (the same triple-TE stack and 16-ch VAE). Turbo shares Large's
/// encoders/VAE exactly. Sourced from the public component config sizes.
const CLIP_L_PARAMS: u64 = 123_000_000; // text_encoder (CLIP ViT-L/14)
const CLIP_BIGG_PARAMS: u64 = 695_000_000; // text_encoder_2 (OpenCLIP bigG/14)
const T5_XXL_PARAMS: u64 = 4_762_000_000; // text_encoder_3 (T5-XXL encoder)
const VAE_PARAMS: u64 = 84_000_000; // 16-channel AutoencoderKL

impl ParamCounts {
    /// All-dense total (used only for sanity / reporting).
    pub fn total(&self) -> u64 {
        self.dit_quantized + self.dit_dense + self.text_encoders + self.vae
    }
}

/// Count the DiT parameters, split into the **quantized** projections and the **dense** leaves, from
/// the [`Sd3Config`] geometry. Pure arithmetic over the config — no weights, no device.
///
/// Per joint block (inner `d`, ff hidden `f = mlp_ratio·d`):
///  - joint attention: image q/k/v/out (4·d²) + text add_q/k/v (3·d²) + to_add_out (d², absent on the
///    final `context_pre_only` block) + biases (≈ 8·d, negligible but counted);
///  - GELU MLP: proj (d·f) + out (f·d) + biases;
///  - `ff_context` (absent on the final block): another d·f + f·d;
///  - dual (`attn2`) blocks: + image-only q/k/v/out (4·d²);
///  - AdaLN modulation linears (dense leaves): norm1 (6 or 9 chunks · d², +bias) + norm1_context
///    (6 or 2 · d², +bias).
///
/// Plus the model-level pieces: `context_embedder` (joint_attention_dim·d, quantized),
/// `proj_out` (d·patch_dim, quantized), the time/text embedders + patch-embed conv + the learned
/// `pos_embed` table (dense leaves).
pub fn dit_param_counts(cfg: &Sd3Config) -> (u64, u64) {
    let d = cfg.inner_dim as u64;
    let f = cfg.ff_hidden() as u64;
    let d2 = d * d;

    let mut quantized: u64 = 0;
    let mut dense: u64 = 0;

    for i in 0..cfg.num_layers {
        let pre_only = cfg.context_pre_only_last && i == cfg.num_layers - 1;
        let dual = cfg.is_dual_block(i);

        // --- quantized projections ---
        // Joint attention: image q/k/v (3·d²) + to_out (d²) = 4·d², each with a d-wide bias.
        quantized += 4 * (d2 + d);
        // Text add_q/k/v (3·d²) with biases.
        quantized += 3 * (d2 + d);
        // to_add_out (d²+bias) — present on every non-final block.
        if !pre_only {
            quantized += d2 + d;
        }
        // GELU MLP: proj (d·f + f bias) + out (f·d + d bias).
        quantized += (d * f + f) + (f * d + d);
        // ff_context on non-final blocks (same shape as the image MLP).
        if !pre_only {
            quantized += (d * f + f) + (f * d + d);
        }
        // Dual (attn2) blocks: image-only q/k/v/out = 4·d² + biases.
        if dual {
            quantized += 4 * (d2 + d);
        }

        // --- dense leaves: the AdaLN modulation linears ---
        // Image norm1: 6 chunks (standard) or 9 (dual) → n·d² weight + n·d bias.
        let img_chunks: u64 = if dual { 9 } else { 6 };
        dense += img_chunks * d2 + img_chunks * d;
        // Context norm1: 6 chunks (standard) or 2 (context_pre_only) → n·d² + n·d.
        let ctx_chunks: u64 = if pre_only { 2 } else { 6 };
        dense += ctx_chunks * d2 + ctx_chunks * d;
    }

    // Model-level quantized projections.
    let joint = cfg.joint_attention_dim as u64;
    let patch_dim = cfg.patch_dim() as u64;
    quantized += joint * d + d; // context_embedder (joint→d) + bias
    quantized += d * patch_dim + patch_dim; // proj_out (d→patch_dim) + bias

    // Model-level dense leaves.
    let ts = cfg.timestep_channels as u64;
    let pooled = cfg.pooled_dim as u64;
    let pos_max = cfg.pos_embed_max_size as u64;
    let in_ch = cfg.in_channels as u64;
    let patch = cfg.patch_size as u64;
    // CombinedTimestepTextEmbed: timestep MLP (ts→d, d→d) + text MLP (pooled→d, d→d), each linear +bias.
    dense += (ts * d + d) + (d * d + d); // timestep_embedder
    dense += (pooled * d + d) + (d * d + d); // text_embedder
                                             // norm_out AdaLN-continuous: linear(d → 2d) + bias.
    dense += d * (2 * d) + 2 * d;
    // Patch-embed conv2d: [d, in_ch, patch, patch] weight + d bias.
    dense += d * in_ch * patch * patch + d;
    // Learned positional embedding table: [1, pos_max², d].
    dense += pos_max * pos_max * d;

    (quantized, dense)
}

/// The full parameter breakdown for a [`Variant`] (DiT from its config; TE/VAE the fixed constants).
pub fn param_counts(variant: Variant) -> ParamCounts {
    let cfg = variant.config();
    let (dit_quantized, dit_dense) = dit_param_counts(&cfg);
    ParamCounts {
        dit_quantized,
        dit_dense,
        text_encoders: CLIP_L_PARAMS + CLIP_BIGG_PARAMS + T5_XXL_PARAMS,
        vae: VAE_PARAMS,
    }
}

/// A byte-level memory profile for a variant × precision, in **bytes**, with the `minMemoryGb`
/// rollup. Built by [`memory_profile`]; the headline number is [`Self::min_memory_gb`].
#[derive(Debug, Clone, Copy)]
pub struct MemoryProfile {
    pub variant: Variant,
    pub precision: Precision,
    /// Quantized (or dense, at bf16) DiT projection bytes.
    pub dit_quantized_bytes: f64,
    /// Always-dense DiT leaf bytes (bf16).
    pub dit_dense_bytes: f64,
    /// Text-encoder bytes (dense bf16).
    pub text_encoder_bytes: f64,
    /// VAE bytes (dense bf16).
    pub vae_bytes: f64,
}

impl MemoryProfile {
    /// Total resident **weight** bytes (no activation overhead, no fragmentation margin).
    pub fn weight_bytes(&self) -> f64 {
        self.dit_quantized_bytes + self.dit_dense_bytes + self.text_encoder_bytes + self.vae_bytes
    }

    /// `minMemoryGb`: `(resident weights + fixed activation overhead) × (1 + frag margin)`, in GiB,
    /// rounded up to one decimal so the worker gate has a clean, conservative threshold. The additive
    /// [`ACTIVATION_OVERHEAD_GIB`] models the resolution-bound dual-CFG working set (which does NOT
    /// scale with weights); [`FRAG_MARGIN`] is the allocator slack on the whole footprint. Validated
    /// >= the C6 measured peaks (sc-7881) by [`tests::min_memory_gate_exceeds_measured_peaks`].
    pub fn min_memory_gb(&self) -> f64 {
        let footprint = self.weight_bytes() + ACTIVATION_OVERHEAD_GIB * GIB;
        let gib = footprint * (1.0 + FRAG_MARGIN) / GIB;
        (gib * 10.0).ceil() / 10.0
    }
}

/// Build the [`MemoryProfile`] for a variant × precision from the parameter counts and the
/// per-precision bytes-per-param.
pub fn memory_profile(variant: Variant, precision: Precision) -> MemoryProfile {
    let counts = param_counts(variant);
    MemoryProfile {
        variant,
        precision,
        dit_quantized_bytes: counts.dit_quantized as f64 * precision.dit_bytes_per_param(),
        dit_dense_bytes: counts.dit_dense as f64 * BF16_BYTES_PER_PARAM,
        text_encoder_bytes: counts.text_encoders as f64 * BF16_BYTES_PER_PARAM,
        vae_bytes: counts.vae as f64 * BF16_BYTES_PER_PARAM,
    }
}

/// The headline **`minMemoryGb`** the worker eligibility gate consumes for `variant` at `precision`.
/// This is the function C5 (sc-7880) wires into the candle SD3.5 manifest.
pub fn min_memory_gb(variant: Variant, precision: Precision) -> f64 {
    memory_profile(variant, precision).min_memory_gb()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Large DiT param counts reproduce the published ~8.05B-param SD3.5-Large MMDiT total: the
    /// quantized projections (attn + MLP) are the bulk (~5.4B); the dense leaves are dominated by the
    /// AdaLN modulation linears (each block's `norm1`/`norm1_context` is a full d→6d linear, ~2.6B
    /// across 38 blocks at d=2432), so they are a sizeable — but kept-dense, chaos-sensitive — share.
    #[test]
    fn large_dit_param_counts_match_published_total() {
        let cfg = Sd3Config::large();
        let (q, dense) = dit_param_counts(&cfg);
        // Quantized projections are the largest single bucket.
        assert!(
            (4_500_000_000..6_500_000_000).contains(&q),
            "Large quantized DiT params {q} outside the ~5.4B expected band"
        );
        // The full DiT (quantized + dense) reproduces the published ~8.05B total within 5%.
        let total = q + dense;
        assert!(
            (7_600_000_000..8_500_000_000).contains(&total),
            "Large DiT total {total} not near the published ~8.05B"
        );
    }

    /// Medium's DiT is smaller than Large's (24×1536 vs 38×2432) yet non-trivial.
    #[test]
    fn medium_dit_smaller_than_large() {
        let (lq, _) = dit_param_counts(&Sd3Config::large());
        let (mq, _) = dit_param_counts(&Sd3Config::medium());
        assert!(mq < lq, "Medium DiT {mq} must be smaller than Large {lq}");
        assert!(mq > 1_000_000_000, "Medium DiT {mq} unexpectedly tiny");
    }

    /// Turbo shares Large's geometry exactly, so its param counts (and thus minMemoryGb) match Large.
    #[test]
    fn turbo_matches_large_counts() {
        let l = param_counts(Variant::Large);
        let t = param_counts(Variant::LargeTurbo);
        assert_eq!(l.dit_quantized, t.dit_quantized);
        assert_eq!(l.dit_dense, t.dit_dense);
        assert_eq!(l.total(), t.total());
    }

    /// **minMemoryGb is strictly ordered Q4 < Q8 < bf16** for every variant (the headline assertion:
    /// quantization lowers the gate threshold monotonically).
    #[test]
    fn min_memory_is_ordered_q4_lt_q8_lt_bf16() {
        for variant in [Variant::Large, Variant::LargeTurbo, Variant::Medium] {
            let bf16 = min_memory_gb(variant, Precision::Bf16);
            let q8 = min_memory_gb(variant, Precision::Q8);
            let q4 = min_memory_gb(variant, Precision::Q4);
            assert!(
                q4 < q8 && q8 < bf16,
                "{variant:?}: expected Q4 ({q4}) < Q8 ({q8}) < bf16 ({bf16})"
            );
        }
    }

    /// The estimates are positive, finite, and in a plausible VRAM band (low-tens of GiB for an
    /// 8B-param diffusion model with a 4.7B T5 encoder kept dense, plus the fixed ~14 GiB dual-CFG
    /// working set).
    #[test]
    fn estimates_are_finite_and_plausible() {
        for variant in [Variant::Large, Variant::LargeTurbo, Variant::Medium] {
            for precision in [Precision::Bf16, Precision::Q8, Precision::Q4] {
                let gb = min_memory_gb(variant, precision);
                assert!(
                    gb.is_finite() && gb > 0.0,
                    "{variant:?}/{precision:?} = {gb}"
                );
                assert!(
                    (16.0..64.0).contains(&gb),
                    "{variant:?}/{precision:?} minMemoryGb {gb} outside a plausible band"
                );
            }
        }
    }

    /// Medium's minMemoryGb ≤ Large's at every precision (smaller DiT, same TE/VAE).
    #[test]
    fn medium_threshold_not_above_large() {
        for precision in [Precision::Bf16, Precision::Q8, Precision::Q4] {
            assert!(
                min_memory_gb(Variant::Medium, precision)
                    <= min_memory_gb(Variant::Large, precision)
            );
        }
    }

    /// **The gate is a safe ceiling over the C6 measured peaks (sc-7881).** These are the true CUDA
    /// peaks of a real 1024² dual-CFG render (`nvidia-smi`, idle baseline, RTX PRO 6000 Blackwell);
    /// the gate must stay strictly above each so the worker never admits a request the card can't
    /// hold. Pinning them here means a future tweak to [`ACTIVATION_OVERHEAD_GIB`] / [`FRAG_MARGIN`]
    /// that drops the gate below a measured peak fails the build instead of silently shipping an
    /// OOM-prone threshold. Re-measure (the `sd3-txt2img` example + `nvidia-smi` peak sampling) and
    /// update these if the load/render path changes the footprint.
    #[test]
    fn min_memory_gate_exceeds_measured_peaks() {
        // (variant, precision, measured peak GiB) — sc-8504 CPU-stage re-measurement (real weights,
        // D:\sd35\large + D:\sd35\medium, 1024² dual-CFG, 28 steps, nvidia-smi GPU-0 peak sampling,
        // 4 MiB idle baseline, RTX PRO 6000 Blackwell). The bf16 peaks are unchanged from C6 (that
        // path is untouched); the quantized peaks dropped ~0.5-1.0 GiB now that the dense-build
        // transient no longer lands on the GPU (the load high-water residue is gone — see the module
        // doc). The gate constants are unchanged (the binding peak was already the render
        // steady-state, not the load transient), so every gate still clears its NEW peak with margin.
        let peaks = [
            (Variant::Large, Precision::Bf16, 38.45),
            (Variant::Large, Precision::Q8, 34.69),
            (Variant::Large, Precision::Q4, 31.60),
            (Variant::Medium, Precision::Bf16, 27.41),
            (Variant::Medium, Precision::Q8, 25.97),
            (Variant::Medium, Precision::Q4, 25.22),
        ];
        for (variant, precision, peak) in peaks {
            let gate = min_memory_gb(variant, precision);
            assert!(
                gate >= peak,
                "{variant:?}/{precision:?}: gate {gate} GiB must be >= measured peak {peak} GiB"
            );
            // And not absurdly over-provisioned (the safe-but-not-wasteful band: < peak + 6 GiB).
            assert!(
                gate <= peak + 6.0,
                "{variant:?}/{precision:?}: gate {gate} GiB is > 6 GiB above peak {peak} — too loose"
            );
        }
    }

    /// Print the full minMemoryGb table (run with `--nocapture` to read it) — the source of the PR
    /// table. Also re-asserts the ordering so the printed values are the asserted ones.
    #[test]
    fn print_min_memory_table() {
        println!(
            "\nSD3.5 minMemoryGb (DiT-only quant; TE+VAE dense bf16; \
             +{ACTIVATION_OVERHEAD_GIB}GiB activation, x{:.2} frag):",
            1.0 + FRAG_MARGIN
        );
        println!("{:<20} {:>8} {:>8} {:>8}", "variant", "bf16", "Q8", "Q4");
        for (name, variant) in [
            ("Large", Variant::Large),
            ("Large-Turbo", Variant::LargeTurbo),
            ("Medium", Variant::Medium),
        ] {
            let bf16 = min_memory_gb(variant, Precision::Bf16);
            let q8 = min_memory_gb(variant, Precision::Q8);
            let q4 = min_memory_gb(variant, Precision::Q4);
            println!("{name:<20} {bf16:>8.1} {q8:>8.1} {q4:>8.1}");
        }
        // The detailed Large breakdown (bytes → GiB) for the PR methodology section.
        for precision in [Precision::Bf16, Precision::Q8, Precision::Q4] {
            let p = memory_profile(Variant::Large, precision);
            println!(
                "Large/{:?}: dit_q={:.2}G dit_dense={:.2}G te={:.2}G vae={:.2}G weights={:.2}G min={:.1}G",
                precision,
                p.dit_quantized_bytes / GIB,
                p.dit_dense_bytes / GIB,
                p.text_encoder_bytes / GIB,
                p.vae_bytes / GIB,
                p.weight_bytes() / GIB,
                p.min_memory_gb(),
            );
        }
    }
}
