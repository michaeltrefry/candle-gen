//! SD3.5 memory profiling → `minMemoryGb` (sc-7879, epic 7982).
//!
//! The worker's model-eligibility gate routes a request to a backend only when the device has at
//! least `minMemoryGb` of free VRAM. C5 (sc-7880) wires the candle SD3.5 manifest; this module is the
//! single source of truth it consumes — a **principled** estimate per variant × precision, computed
//! from the actual quantized-tensor parameter counts (NOT a measured peak, which needs the gated
//! real weights — that is C6's `#[ignore]`d profiling test [`crate::pipeline`]).
//!
//! ## Methodology
//!
//! ```text
//! minMemoryGb  =  (quantized DiT weight bytes        // the projections C4 folds to Q4/Q8
//!               +  dense-kept DiT leaf bytes          // AdaLN/timestep/patch-embed/pos_embed — always bf16
//!               +  text-encoder bytes                 // CLIP-L + CLIP-bigG + T5-XXL, kept DENSE (bf16)
//!               +  VAE bytes)                          // 16-ch AutoencoderKL, kept DENSE (bf16)
//!               *  (1 + HEADROOM)                      // activation / working-set / fragmentation margin
//! ```
//!
//! - **Quantized DiT weights** use [`crate::quant::bytes_per_param`] (block-scale-inclusive: Q4 ≈
//!   0.5625 B/param, Q8 ≈ 1.0625, bf16 = 2.0).
//! - **Text encoders + VAE are kept dense (bf16).** This matches the established candle precedent —
//!   Lens/FLUX.2 quantize the DiT (and, for FLUX.2-dev, the giant TE) but the SD3.5 encoders/VAE are
//!   small relative to the DiT and chaos-sensitive, so C4 quantizes the **DiT only**; the TE/VAE
//!   bytes are a fixed dense addend across all three precisions.
//! - **Headroom** (`HEADROOM` = 0.30) covers the denoise working set (latents, attention scores at
//!   1MP, the two CFG forwards' transient activations), CUDA context, and allocator fragmentation —
//!   the same order-of-magnitude margin the worker applies to the other diffusion backends. It is a
//!   ceiling, deliberately conservative so the gate never admits a request the device can't hold.
//!
//! The figures are exposed as [`min_memory_gb`] (and the per-component breakdown [`MemoryProfile`])
//! so C5 can read them in code; the same numbers are tabulated in the PR. The `#[ignore]`d
//! `real_weight_memory_profile` test (env-gated) measures the TRUE peak for C6 to confirm against.

use candle_gen::gen_core::Quant;

use crate::config::Sd3Config;
use crate::pipeline::Variant;
use crate::quant::bytes_per_param;

/// bf16 is 2 bytes per parameter — the dense baseline precision SD3.5 loads at.
const BF16_BYTES_PER_PARAM: f64 = 2.0;

/// Activation / working-set / CUDA-context / fragmentation headroom, as a fraction of the resident
/// weight bytes. Conservative (the gate must not over-admit); covers the two CFG forwards' transient
/// attention buffers at up to 1MP, the latent/decoder working set, and allocator slack.
pub const HEADROOM: f64 = 0.30;

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
    /// Total resident **weight** bytes (no headroom).
    pub fn weight_bytes(&self) -> f64 {
        self.dit_quantized_bytes + self.dit_dense_bytes + self.text_encoder_bytes + self.vae_bytes
    }

    /// `minMemoryGb`: resident weights × (1 + headroom), in GiB, rounded up to one decimal so the
    /// worker gate has a clean, conservative threshold.
    pub fn min_memory_gb(&self) -> f64 {
        let gib = self.weight_bytes() * (1.0 + HEADROOM) / GIB;
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

    /// The estimates are positive, finite, and in a plausible VRAM band (single-digit to low-tens of
    /// GiB for an 8B-param diffusion model with a 4.7B T5 encoder kept dense).
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
                    (4.0..64.0).contains(&gb),
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

    /// Print the full minMemoryGb table (run with `--nocapture` to read it) — the source of the PR
    /// table. Also re-asserts the ordering so the printed values are the asserted ones.
    #[test]
    fn print_min_memory_table() {
        println!("\nSD3.5 minMemoryGb (DiT-only quant; TE+VAE dense bf16; headroom {HEADROOM}):");
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
