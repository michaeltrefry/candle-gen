//! SD3.5 triple text-encoder **aggregator** (sc-7876, epic 7982).
//!
//! SD3.5 conditions on three text encoders — CLIP-L, OpenCLIP bigG, and T5-XXL — combined into two
//! tensors fed to the MMDiT:
//!
//! - **pooled** `[B, 2048]` = `cat(CLIP-L pooled [768], CLIP-bigG pooled [1280])`. This is added to
//!   the timestep embedding (NOT to the token sequence) — it conditions the AdaLN modulation.
//! - **context** `[B, 333, 4096]` (at the SD3.5 defaults) = the token sequence the joint blocks
//!   attend over. Built in two steps, exactly as the public diffusers `StableDiffusion3Pipeline`:
//!   1. CLIP context = `cat(CLIP-L penultimate [77, 768], CLIP-bigG penultimate [77, 1280])` →
//!      `[77, 2048]`, then **zero-padded on the hidden axis** to `[77, 4096]`
//!      (`joint_attention_dim`). The pad is on the *trailing* hidden dims (diffusers
//!      `F.pad(clip, (0, t5_dim - clip_concat_dim))`).
//!   2. context = `cat([clip_padded [77, 4096], t5 [t5_len, 4096]], dim=seq)` → `[77 + t5_len, 4096]`.
//!
//! This module owns the **aggregation** — the parity-critical concat/pad/order that the spike
//! flagged. The actual CLIP/T5 forward (loading the encoders, penultimate-layer extraction, EOS
//! pooling) is wired in C2's pipeline; keeping the aggregator a pure tensor transform lets the
//! ordering be unit-tested on CPU with synthetic encoder outputs (no weights/GPU needed), the same
//! correctness bar epic 7841 used.

use std::path::Path;

use candle_gen::candle_core::IndexOp;
use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::{CandleError, Result as CandleResult};
use candle_transformers::models::stable_diffusion::{self, clip};
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use tokenizers::Tokenizer;

use crate::config::Sd3Config;

/// The raw per-encoder outputs the aggregator combines. Produced by the encoders in C2; here they
/// are the inputs to the pure aggregation so the ordering is testable in isolation.
///
/// All tensors carry a leading batch axis `B`.
pub struct EncoderOutputs {
    /// CLIP-L penultimate hidden state `[B, clip_seq_len, clip_l_dim]` (768-wide).
    pub clip_l_hidden: Tensor,
    /// CLIP-bigG penultimate hidden state `[B, clip_seq_len, clip_g_dim]` (1280-wide).
    pub clip_g_hidden: Tensor,
    /// CLIP-L pooled/projected output `[B, clip_l_dim]` (768-wide).
    pub clip_l_pooled: Tensor,
    /// CLIP-bigG pooled/projected output `[B, clip_g_dim]` (1280-wide).
    pub clip_g_pooled: Tensor,
    /// T5-XXL encoder sequence `[B, t5_seq_len, t5_dim]` (4096-wide).
    pub t5_hidden: Tensor,
}

/// The two SD3.5 conditioning tensors fed to the MMDiT.
pub struct Sd3Conditioning {
    /// `[B, pooled_dim]` (2048) — added to the timestep embedding.
    pub pooled: Tensor,
    /// `[B, context_seq_len, joint_attention_dim]` (333 × 4096 at defaults) — the joint token
    /// sequence.
    pub context: Tensor,
}

/// Build the SD3.5 pooled + context conditioning from the three encoders' outputs.
///
/// Order and padding match the public diffusers `StableDiffusion3Pipeline._get_clip_prompt_embeds`
/// + `encode_prompt`:
/// - pooled = `cat([clip_l_pooled, clip_g_pooled], dim=-1)`;
/// - clip_context = `cat([clip_l_hidden, clip_g_hidden], dim=-1)` then right-pad the hidden axis to
///   `joint_attention_dim` with zeros;
/// - context = `cat([clip_context, t5_hidden], dim=seq)`.
pub fn aggregate(cfg: &Sd3Config, enc: &EncoderOutputs) -> Result<Sd3Conditioning> {
    // ---- pooled [B, 2048] ----
    let pooled = Tensor::cat(&[&enc.clip_l_pooled, &enc.clip_g_pooled], D::Minus1)?;

    // ---- CLIP context [B, 77, 2048] -> zero-pad hidden axis to [B, 77, 4096] ----
    let clip_context = Tensor::cat(&[&enc.clip_l_hidden, &enc.clip_g_hidden], D::Minus1)?;
    // The concatenated CLIP width must be the configured `clip_concat_dim` (768 + 1280 = 2048); a
    // mismatch means a mis-shaped encoder output, caught here before the pad rather than producing a
    // silently wrong context.
    let clip_w = clip_context.dim(D::Minus1)?;
    if clip_w != cfg.clip_concat_dim {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "sd3 aggregator: concatenated CLIP context width {clip_w} != configured \
             clip_concat_dim {} (clip_l_dim {} + clip_g_dim {})",
            cfg.clip_concat_dim, cfg.clip_l_dim, cfg.clip_g_dim
        )));
    }
    let clip_padded = pad_hidden_to(&clip_context, cfg.joint_attention_dim)?;

    // ---- context = cat([clip_padded, t5], seq) -> [B, 333, 4096] ----
    let context = Tensor::cat(&[&clip_padded, &enc.t5_hidden], 1)?;

    Ok(Sd3Conditioning { pooled, context })
}

/// Right-pad the LAST (hidden) axis of `x` `[..., h]` to width `target` with zeros (`F.pad(x, (0,
/// target - h))`). Errors if `x` is already wider than `target`.
fn pad_hidden_to(x: &Tensor, target: usize) -> Result<Tensor> {
    let h = x.dim(D::Minus1)?;
    if h == target {
        return Ok(x.clone());
    }
    if h > target {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "sd3 aggregator: clip context hidden {h} exceeds joint_attention_dim {target}"
        )));
    }
    let mut shape = x.dims().to_vec();
    *shape.last_mut().unwrap() = target - h;
    let pad = Tensor::zeros(shape, x.dtype(), x.device())?;
    Tensor::cat(&[x, &pad], D::Minus1)
}

/// Build zeroed encoder outputs at the config's shapes for a given batch (the "empty/unconditioned"
/// path and a test fixture). C2's CFG uses this to assemble the negative branch when a T5/CLIP empty
/// encode is degenerate; here it doubles as the structural-test fixture.
pub fn zeroed_outputs(
    cfg: &Sd3Config,
    batch: usize,
    dtype: DType,
    device: &Device,
) -> Result<EncoderOutputs> {
    Ok(EncoderOutputs {
        clip_l_hidden: Tensor::zeros((batch, cfg.clip_seq_len, cfg.clip_l_dim), dtype, device)?,
        clip_g_hidden: Tensor::zeros((batch, cfg.clip_seq_len, cfg.clip_g_dim), dtype, device)?,
        clip_l_pooled: Tensor::zeros((batch, cfg.clip_l_dim), dtype, device)?,
        clip_g_pooled: Tensor::zeros((batch, cfg.clip_g_dim), dtype, device)?,
        t5_hidden: Tensor::zeros((batch, cfg.t5_seq_len, cfg.t5_dim), dtype, device)?,
    })
}

/// CLIP token cap (both encoders). SD3.5 truncates/pads the prompt to 77 tokens.
const CLIP_MAX_LEN: usize = 77;

/// T5 pad token id (`<pad>`). SD3.5 pads the T5 sequence to `t5_seq_len` with this id and attends
/// every position (no T5 attention mask), so the padded length is parity-critical.
const T5_PAD_TOKEN_ID: u32 = 0;

/// Right-pad / hard-truncate a CLIP token row to exactly `CLIP_MAX_LEN`. SD3.5's diffusers pipeline
/// truncates to the model max (77) with a warning; we truncate (keeping BOS + the leading content)
/// and re-append the EOS so the pooled EOS lookup still finds it. Pads with the encoder's pad id.
fn fit_clip_tokens(mut ids: Vec<u32>, pad_id: u32, eos_id: u32) -> Vec<u32> {
    if ids.len() > CLIP_MAX_LEN {
        ids.truncate(CLIP_MAX_LEN);
        // Force the last slot to EOS so `eos_position` (arg-max) still selects a real EOS.
        *ids.last_mut().unwrap() = eos_id;
    } else {
        ids.resize(CLIP_MAX_LEN, pad_id);
    }
    ids
}

/// The EOS position of a CLIP token row = the arg-max token id (EOS = `<|endoftext|>` = 49407 is the
/// highest id). diffusers pools the CLIP final hidden state here for the pooled `text_embeds`.
fn eos_position(ids: &[u32]) -> usize {
    ids.iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// The three loaded SD3.5 text encoders + their tokenizers and pooled-projection heads. Built once
/// per model (held resident by the pipeline); [`encode`](Self::encode) is called per request and
/// produces the [`EncoderOutputs`] the [`aggregate`] step combines.
///
/// - CLIP-L (`text_encoder/`, embed 768) and CLIP-bigG (`text_encoder_2/`, embed 1280): the
///   **penultimate** hidden state (`hidden_states[-2]`, pre-final-norm — diffusers
///   `output_hidden_states=True`) feeds the joint context; the **final**-norm hidden at the EOS
///   position, projected through each encoder's `text_projection`, feeds the pooled vector.
/// - T5-XXL (`text_encoder_3/`, hidden 4096): the full encoder sequence, padded to `t5_seq_len`.
pub struct Sd3TextEncoders {
    tok_l: Tokenizer,
    tok_g: Tokenizer,
    tok_t5: Tokenizer,
    clip_l: clip::ClipTextTransformer,
    clip_g: clip::ClipTextTransformer,
    /// CLIP-L `text_projection.weight` (`[768, 768]`, no bias).
    proj_l: Linear,
    /// CLIP-bigG `text_projection.weight` (`[1280, 1280]`, no bias).
    proj_g: Linear,
    t5: T5EncoderModel,
    t5_seq_len: usize,
    device: Device,
    dtype: DType,
}

impl Sd3TextEncoders {
    /// Load the three encoders from a `stabilityai/stable-diffusion-3.5-*` diffusers snapshot:
    /// `text_encoder/` (CLIP-L), `text_encoder_2/` (CLIP-bigG), `text_encoder_3/` (T5-XXL). The two
    /// CLIP tokenizers load from `tokenizer.json` when present and otherwise are **synthesized** from
    /// the stock `vocab.json` + `merges.txt` (sc-8500; see [`crate::clip_tokenizer`]); T5 ships its
    /// own `tokenizer.json`. `t5_seq_len` is the configured T5 length (256 default).
    pub fn load(
        root: &Path,
        t5_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let cfg_l = clip::Config::sdxl(); // CLIP-L (openai/clip-vit-large-patch14, embed 768)
        let cfg_g = clip::Config::sdxl2(); // OpenCLIP bigG (embed 1280)

        // CLIP-L / CLIP-bigG: load `tokenizer.json` if present, else SYNTHESIZE the CLIP
        // byte-level BPE from `vocab.json` + `merges.txt` (a stock gated diffusers SD3.5
        // download ships no `tokenizer.json` for the CLIP encoders — sc-8500).
        let tok_l = crate::clip_tokenizer::load_clip_tokenizer(&root.join("tokenizer"), "CLIP-L")?;
        let tok_g =
            crate::clip_tokenizer::load_clip_tokenizer(&root.join("tokenizer_2"), "CLIP-bigG")?;
        // T5 ships its own `tokenizer.json` in a stock snapshot (out of scope for sc-8500).
        let tok_t5 = Tokenizer::from_file(root.join("tokenizer_3/tokenizer.json"))
            .map_err(|e| CandleError::Msg(format!("sd3: load T5 tokenizer: {e}")))?;

        let l_file = single_safetensors(root, "text_encoder")?;
        let g_file = single_safetensors(root, "text_encoder_2")?;
        let clip_l = stable_diffusion::build_clip_transformer(&cfg_l, &l_file, device, dtype)?;
        let clip_g = stable_diffusion::build_clip_transformer(&cfg_g, &g_file, device, dtype)?;
        let proj_l = load_text_projection(&l_file, "text_encoder", device, dtype)?;
        let proj_g = load_text_projection(&g_file, "text_encoder_2", device, dtype)?;

        // T5-XXL (`text_encoder_3/`, sharded; config.json alongside).
        let t5_dir = root.join("text_encoder_3");
        let t5_cfg: T5Config = {
            let cfg = std::fs::read_to_string(t5_dir.join("config.json")).map_err(|e| {
                CandleError::Msg(format!("sd3: read text_encoder_3/config.json: {e}"))
            })?;
            serde_json::from_str(&cfg)
                .map_err(|e| CandleError::Msg(format!("sd3: parse T5 config.json: {e}")))?
        };
        let t5_files = safetensors_in(&t5_dir)?;
        // SAFETY: mmap of read-only weight files; standard candle loading path.
        let t5_vb = unsafe {
            candle_gen::candle_nn::VarBuilder::from_mmaped_safetensors(&t5_files, dtype, device)?
        };
        let t5 = T5EncoderModel::load(t5_vb, &t5_cfg)?;

        Ok(Self {
            tok_l,
            tok_g,
            tok_t5,
            clip_l,
            clip_g,
            proj_l,
            proj_g,
            t5,
            t5_seq_len,
            device: device.clone(),
            dtype,
        })
    }

    /// Run one CLIP encoder for `prompt`: returns `(penultimate [1, 77, embed], pooled [1, embed])`.
    /// The penultimate hidden is the pre-final-norm `hidden_states[-2]`; the pooled is the EOS-position
    /// final-norm hidden projected through that encoder's `text_projection`.
    fn encode_clip(
        &self,
        tok: &Tokenizer,
        clip: &clip::ClipTextTransformer,
        proj: &Linear,
        prompt: &str,
    ) -> CandleResult<(Tensor, Tensor)> {
        let vocab = tok.get_vocab(true);
        let eos_id = *vocab
            .get("<|endoftext|>")
            .ok_or_else(|| CandleError::Msg("sd3: CLIP tokenizer missing <|endoftext|>".into()))?;
        // SD3.5 CLIP pads with the EOS token (diffusers `pad_token_id` = eos for these encoders).
        let ids = tok
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("sd3: CLIP tokenize: {e}")))?
            .get_ids()
            .to_vec();
        let ids = fit_clip_tokens(ids, eos_id, eos_id);
        let eos = eos_position(&ids);
        let input = Tensor::new(ids.as_slice(), &self.device)?.reshape((1, CLIP_MAX_LEN))?;
        // `forward_until_encoder_layer(.., -2)` → (final-norm hidden, penultimate hidden).
        let (final_hidden, penult) = clip.forward_until_encoder_layer(&input, usize::MAX, -2)?;
        let pooled_eos = final_hidden.i((0, eos))?.unsqueeze(0)?; // [1, embed]
        let pooled = proj.forward(&pooled_eos)?;
        Ok((penult.to_dtype(self.dtype)?, pooled.to_dtype(self.dtype)?))
    }

    /// Encode `prompt` into the per-encoder [`EncoderOutputs`] (batch 1). T5 is tokenized and padded
    /// to `t5_seq_len` with the pad id; every position is attended (no T5 mask), matching diffusers.
    pub fn encode(&mut self, prompt: &str) -> CandleResult<EncoderOutputs> {
        let (clip_l_hidden, clip_l_pooled) =
            self.encode_clip(&self.tok_l, &self.clip_l, &self.proj_l, prompt)?;
        let (clip_g_hidden, clip_g_pooled) =
            self.encode_clip(&self.tok_g, &self.clip_g, &self.proj_g, prompt)?;

        // T5 sequence, padded/truncated to t5_seq_len.
        let mut t5_ids: Vec<u32> = self
            .tok_t5
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("sd3: T5 tokenize: {e}")))?
            .get_ids()
            .to_vec();
        t5_ids.truncate(self.t5_seq_len);
        t5_ids.resize(self.t5_seq_len, T5_PAD_TOKEN_ID);
        let t5_input = Tensor::new(t5_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let t5_hidden = self.t5.forward(&t5_input)?.to_dtype(self.dtype)?; // [1, t5_seq_len, 4096]

        Ok(EncoderOutputs {
            clip_l_hidden,
            clip_g_hidden,
            clip_l_pooled,
            clip_g_pooled,
            t5_hidden,
        })
    }
}

/// Resolve the single `model.safetensors` (or first sorted shard) in a snapshot component subdir.
fn single_safetensors(root: &Path, sub: &str) -> CandleResult<std::path::PathBuf> {
    let files = safetensors_in(&root.join(sub))?;
    Ok(files.into_iter().next().unwrap())
}

/// Sorted list of every `.safetensors` in `dir` (single-file or sharded), erroring if absent.
fn safetensors_in(dir: &Path) -> CandleResult<Vec<std::path::PathBuf>> {
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "sd3 snapshot is missing the {} component directory",
            dir.display()
        )));
    }
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("sd3: read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "sd3: no .safetensors found in {}",
            dir.display()
        )));
    }
    Ok(files)
}

/// Load a CLIP `text_projection.weight` (no bias) from a CLIP checkpoint into a [`Linear`]. SD3.5's
/// CLIP-L and bigG are `CLIPTextModelWithProjection`s — `build_clip_transformer` reads only the
/// `text_model.*`; the pooled head's projection lives at the top level as `text_projection.weight`.
fn load_text_projection(
    file: &Path,
    sub: &str,
    device: &Device,
    dtype: DType,
) -> CandleResult<Linear> {
    let tensors = candle_gen::candle_core::safetensors::load(file, device)?;
    let w = tensors
        .get("text_projection.weight")
        .ok_or_else(|| {
            CandleError::Msg(format!(
                "sd3 conditioning: text_projection.weight missing from {sub}/ checkpoint ({})",
                file.display()
            ))
        })?
        .to_dtype(dtype)?;
    Ok(Linear::new(w, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn fixture(cfg: &Sd3Config, batch: usize) -> EncoderOutputs {
        let dev = Device::Cpu;
        // Distinctive fill values per source so the concat ORDER is observable in the output.
        EncoderOutputs {
            clip_l_hidden: Tensor::full(1f32, (batch, cfg.clip_seq_len, cfg.clip_l_dim), &dev)
                .unwrap(),
            clip_g_hidden: Tensor::full(2f32, (batch, cfg.clip_seq_len, cfg.clip_g_dim), &dev)
                .unwrap(),
            clip_l_pooled: Tensor::full(3f32, (batch, cfg.clip_l_dim), &dev).unwrap(),
            clip_g_pooled: Tensor::full(4f32, (batch, cfg.clip_g_dim), &dev).unwrap(),
            t5_hidden: Tensor::full(5f32, (batch, cfg.t5_seq_len, cfg.t5_dim), &dev).unwrap(),
        }
    }

    #[test]
    fn aggregate_shapes_match_sd35_defaults() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        // pooled = 768 + 1280 = 2048.
        assert_eq!(out.pooled.dims(), &[1, 2048]);
        // context = (77 + 256) x 4096 = 333 x 4096.
        assert_eq!(out.context.dims(), &[1, 333, 4096]);
    }

    #[test]
    fn pooled_concat_order_is_l_then_g() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        let v = out.pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // First 768 from CLIP-L (filled 3), next 1280 from bigG (filled 4).
        assert!(
            v[..768].iter().all(|&x| x == 3.0),
            "CLIP-L pooled goes first"
        );
        assert!(
            v[768..2048].iter().all(|&x| x == 4.0),
            "bigG pooled goes second"
        );
    }

    #[test]
    fn context_layout_is_clip_padded_then_t5() {
        let cfg = Sd3Config::large();
        let enc = fixture(&cfg, 1);
        let out = aggregate(&cfg, &enc).unwrap();
        // Token 0 (a CLIP token): hidden = [CLIP-L 768 = 1, bigG 1280 = 2, zero-pad 2048 = 0].
        let tok0 = out.context.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        assert!(tok0[..768].iter().all(|&x| x == 1.0), "clip-l region");
        assert!(tok0[768..2048].iter().all(|&x| x == 2.0), "bigg region");
        assert!(
            tok0[2048..4096].iter().all(|&x| x == 0.0),
            "zero-pad region"
        );
        // Token 77 (the first T5 token): all 5 across the full 4096 width.
        let tok_t5 = out.context.i((0, 77)).unwrap().to_vec1::<f32>().unwrap();
        assert!(
            tok_t5.iter().all(|&x| x == 5.0),
            "t5 region is full-width 4096"
        );
    }

    #[test]
    fn t5_length_drives_context_seq() {
        let mut cfg = Sd3Config::large();
        cfg.t5_seq_len = 512;
        let enc = fixture(&cfg, 2);
        let out = aggregate(&cfg, &enc).unwrap();
        assert_eq!(out.context.dims(), &[2, 77 + 512, 4096]);
        assert_eq!(out.pooled.dims(), &[2, 2048]);
    }

    #[test]
    fn aggregate_rejects_misshaped_clip_width() {
        // A config whose clip_concat_dim disagrees with clip_l_dim + clip_g_dim trips the guard.
        let mut cfg = Sd3Config::large();
        cfg.clip_concat_dim = 999; // != 768 + 1280
        let enc = fixture(&cfg, 1);
        assert!(aggregate(&cfg, &enc).is_err());
    }

    /// `fit_clip_tokens` pads short rows to 77 and hard-truncates long rows, keeping an EOS in the
    /// last slot so the pooled EOS lookup still lands on a real EOS (not a content token).
    #[test]
    fn fit_clip_tokens_pads_and_truncates_with_eos() {
        let eos = 49407u32;
        // Short row pads to 77 with the pad id.
        let short = fit_clip_tokens(vec![49406, 320, eos], 9, eos);
        assert_eq!(short.len(), 77);
        assert_eq!(&short[..3], &[49406, 320, eos]);
        assert!(short[3..].iter().all(|&x| x == 9));
        // Over-long row truncates to 77 and forces the last slot to EOS.
        let long: Vec<u32> = (0..100).collect();
        let fit = fit_clip_tokens(long, 9, eos);
        assert_eq!(fit.len(), 77);
        assert_eq!(*fit.last().unwrap(), eos);
    }

    /// `eos_position` finds the arg-max id (EOS is the highest CLIP id), even with padding after it.
    #[test]
    fn eos_position_is_argmax() {
        assert_eq!(eos_position(&[49406, 320, 49407, 9, 9]), 2);
        assert_eq!(eos_position(&[49406, 1, 2, 3, 49407]), 4);
    }

    #[test]
    fn zeroed_outputs_aggregate_to_correct_shape() {
        let cfg = Sd3Config::large();
        let enc = zeroed_outputs(&cfg, 1, DType::F32, &Device::Cpu).unwrap();
        let out = aggregate(&cfg, &enc).unwrap();
        assert_eq!(out.context.dims(), &[1, 333, 4096]);
        assert_eq!(out.pooled.dims(), &[1, 2048]);
    }
}
