//! # candle-gen-lens
//!
//! The **Lens / Lens-Turbo** text-to-image provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of the `mlx-gen` Lens port (epic 3164). Lens is a three-component model:
//!
//! 1. a **gpt-oss-20b** MoE LLM used **encoder-only** ([`text_encoder`]) — 24-layer / 32-expert /
//!    top-4, attention sinks, alternating sliding/full attention, YaRN RoPE, clamped-SwiGLU experts,
//!    MXFP4-native expert weights; run forward capturing hidden states at `[5, 11, 17, 23]`;
//! 2. a **48-layer dual-stream MMDiT** ([`transformer`], `LensTransformer2DModel`, sc-5112) —
//!    fused-QKV joint attention over `[img, txt]`, complex axial RoPE ([`rope`]), AdaLN dual
//!    modulation, SwiGLU MLPs, multi-layer text front-end;
//! 3. the **Flux.2 VAE** ([`vae`], `AutoencoderKLFlux2`, sc-5113) — reused from `candle-gen-flux2`
//!    via a thin decode shim (reshape the DiT output into the packed NCHW grid → `decode_packed`).
//!
//! This crate is being built story-by-story under epic **5107**. The first landed piece is the
//! gpt-oss encoder decoder block ([`text_encoder`], sc-5108): a from-scratch port — candle-transformers
//! ships no `gpt_oss` model (the Gate-0 spike found upstream PRs #3129/#3581/#3391 all unmerged), so
//! the decoder is adapted from the verified-parity reference in candle PR #3581 onto `candle_nn`.
//!
//! **Dtype:** the encoder runs **bf16** (the checkpoint's native non-expert dtype); the MXFP4 expert
//! weights are dequantized to bf16 at load (sc-5108 bring-up). The eventual MXFP4 → GGUF Q4 `QMatMul`
//! transcode that keeps the ~12 GB footprint is sc-5111.

pub mod rope;
pub mod text;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
