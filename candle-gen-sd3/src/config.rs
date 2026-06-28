//! SD3.5 architecture configuration (sc-7876, epic 7982).
//!
//! All layer counts, dims, and the T5 sequence length are config-driven so the later stories (C2
//! Large+Turbo pipeline, C3 Medium MMDiT-X) can reuse the same `Sd3Config` with different presets
//! rather than re-deriving the geometry. The defaults here are the **Large** preset
//! (`stabilityai/stable-diffusion-3.5-large`), spike-confirmed against the public diffusers
//! `SD3Transformer2DModel` config.json and the SD3 paper "Scaling Rectified Flow Transformers".

/// SD3.5 model + MMDiT geometry. Constructed via the named presets ([`Sd3Config::large`]); the
/// fields are public so C2/C3 can tweak the T5 length or block count without a new constructor.
#[derive(Debug, Clone)]
pub struct Sd3Config {
    // ---- latent / patchify ----
    /// VAE latent channel count (and the DiT `in_channels`). SD3.5 = 16.
    pub in_channels: usize,
    /// Patchify patch size on each spatial axis. SD3.5 = 2.
    pub patch_size: usize,
    /// Max latent grid side the learned positional embedding table covers (the diffusers
    /// `pos_embed_max_size`). The patchified token grid is cropped from the centre of this table.
    pub pos_embed_max_size: usize,

    // ---- MMDiT core ----
    /// Joint attention hidden width (`num_attention_heads * attention_head_dim`). Large = 2432.
    pub inner_dim: usize,
    /// Attention heads. Large = 38.
    pub num_heads: usize,
    /// Per-head dim (`inner_dim / num_heads`). Large = 64.
    pub head_dim: usize,
    /// Joint/double-stream block count. Large = 38.
    pub num_layers: usize,
    /// FFN hidden expansion ratio (diffusers `mlp_ratio` = 4.0).
    pub mlp_ratio: f32,
    /// Whether per-head QK-RMSNorm is applied (SD3.5 Large/Medium = true; vanilla SD3 = false).
    pub qk_norm: bool,
    /// Whether the LAST joint block drops its context (text) stream output (`context_pre_only`):
    /// the final block only needs to update the image tokens, so its `ff_context`/`add_*_out` are
    /// absent in the checkpoint. diffusers sets this on the last block.
    pub context_pre_only_last: bool,

    // ---- conditioning aggregator ----
    /// Pooled projection width = CLIP-L pooled (768) + CLIP-bigG pooled (1280) = 2048. Added to the
    /// timestep embedding (NOT the token sequence).
    pub pooled_dim: usize,
    /// Joint attention context width the DiT's `context_embedder` consumes. SD3.5 = 4096 (the T5
    /// hidden; the concatenated CLIP context is zero-padded up to this on the hidden axis).
    pub joint_attention_dim: usize,
    /// CLIP-L penultimate hidden width (768).
    pub clip_l_dim: usize,
    /// CLIP-bigG penultimate hidden width (1280).
    pub clip_g_dim: usize,
    /// Combined CLIP context width before zero-pad to `joint_attention_dim` (clip_l_dim + clip_g_dim
    /// = 2048).
    pub clip_concat_dim: usize,
    /// CLIP token length (both encoders). SD3.5 = 77.
    pub clip_seq_len: usize,
    /// T5-XXL token length — **configurable** (SD3.5 default 256; 77/512 are also valid). The full
    /// context sequence is `clip_seq_len + t5_seq_len` (333 at the defaults).
    pub t5_seq_len: usize,
    /// T5-XXL hidden width (4096).
    pub t5_dim: usize,

    // ---- timestep embedding ----
    /// Sinusoidal timestep embedding width before the MLP (diffusers `time_embed` in dim = 256).
    pub timestep_channels: usize,
}

impl Sd3Config {
    /// The full joint context sequence length: CLIP (77) ++ T5 (`t5_seq_len`). 333 at the SD3.5
    /// defaults (77 + 256).
    pub fn context_seq_len(&self) -> usize {
        self.clip_seq_len + self.t5_seq_len
    }

    /// The patchified-latent channel count the `proj_out`/unpatchify head produces:
    /// `patch_size^2 * in_channels`.
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.in_channels
    }

    /// FFN hidden width for a joint block (`mlp_ratio * inner_dim`).
    pub fn ff_hidden(&self) -> usize {
        (self.mlp_ratio * self.inner_dim as f32) as usize
    }

    /// **SD3.5 Large** preset (`stabilityai/stable-diffusion-3.5-large`).
    pub fn large() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            pos_embed_max_size: 192,
            inner_dim: 2432,
            num_heads: 38,
            head_dim: 64,
            num_layers: 38,
            mlp_ratio: 4.0,
            qk_norm: true,
            context_pre_only_last: true,
            pooled_dim: 2048,
            joint_attention_dim: 4096,
            clip_l_dim: 768,
            clip_g_dim: 1280,
            clip_concat_dim: 2048,
            clip_seq_len: 77,
            t5_seq_len: 256,
            t5_dim: 4096,
            timestep_channels: 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_preset_geometry_matches_diffusers() {
        let c = Sd3Config::large();
        // inner_dim factors cleanly into heads × head_dim.
        assert_eq!(c.inner_dim, c.num_heads * c.head_dim);
        assert_eq!(c.inner_dim, 2432);
        assert_eq!(c.num_heads, 38);
        assert_eq!(c.num_layers, 38);
        // pooled = CLIP-L (768) + bigG (1280) = 2048.
        assert_eq!(c.clip_l_dim + c.clip_g_dim, c.pooled_dim);
        assert_eq!(c.clip_concat_dim, 2048);
        // context sequence = 77 CLIP + 256 T5 = 333 at the defaults.
        assert_eq!(c.context_seq_len(), 333);
        // T5 hidden is the joint-attention width.
        assert_eq!(c.t5_dim, c.joint_attention_dim);
        // patch dim = 2*2*16 = 64.
        assert_eq!(c.patch_dim(), 64);
    }

    #[test]
    fn t5_length_is_configurable() {
        let mut c = Sd3Config::large();
        assert_eq!(c.context_seq_len(), 333);
        c.t5_seq_len = 512;
        assert_eq!(c.context_seq_len(), 77 + 512);
        c.t5_seq_len = 77;
        assert_eq!(c.context_seq_len(), 154);
    }
}
