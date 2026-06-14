//! The NEO-Unify **vision embedder** — the candle port of `mlx-gen-sensenova`'s `vision.rs`
//! (`modeling_neo_vit.py`). The T2I slice only ever builds the **generation-path** instance
//! (`fm_modules.vision_model_mot_gen.embeddings`); the understanding-path tower (it2i/VQA) is Phase 6.
//!
//! For the 8B-MoT checkpoint the "vision tower" has no transformer blocks: a full-kernel
//! `patch_embedding` (Conv2d 3→`hidden_size`, kernel=stride=`patch_size`) + erf-GELU, an
//! **interleaved** 2D RoPE over the patch grid, then a `factor×factor`-strided `dense_embedding`
//! (Conv2d `hidden_size`→`llm_hidden_size`) that merges each 2×2 block of patches into one LLM token.
//!
//! The full-kernel `patch_embedding` is computed as a Linear over the flattened patch (an exact
//! equivalent). candle conv2d is **NCHW** (mlx is NHWC), so `dense_embedding` keeps the torch
//! `[llm, embed, f, f]` kernel layout unchanged and the roped patch grid is fed as `[1, embed, h, w]`.

use candle_gen::candle_core::{Result as CResult, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::Result;

use crate::config::NeoChatConfig;

/// A NEO vision embedder (Conv patch-embed → 2D RoPE → Conv patch-merge).
pub struct NeoVisionEmbedder {
    /// `patch_embedding` as a Linear over the flattened `[ch·ps·ps]` patch.
    patch: Linear,
    /// `dense_embedding` conv kernel `[llm, embed, factor, factor]` (torch/NCHW layout) + bias.
    dense_w: Tensor,
    dense_b: Tensor,
    embed_dim: usize,
    downsample_factor: usize,
    rope_theta: f32,
}

impl NeoVisionEmbedder {
    /// Build from a checkpoint. `prefix` = the embeddings namespace, e.g.
    /// `"fm_modules.vision_model_mot_gen.embeddings"` (generation path).
    pub fn from_weights(vb: &VarBuilder, cfg: &NeoChatConfig, prefix: &str) -> Result<Self> {
        let factor = (1.0 / cfg.vision.downsample_ratio).round() as usize;
        // patch_embedding: torch Conv2d weight [embed, ch, ps, ps] -> flat [embed, ch·ps·ps].
        let patch_w = vb.get_unchecked(&format!("{prefix}.patch_embedding.weight"))?;
        let (embed_dim, ch, ps, _) = patch_w.dims4()?;
        let patch_w = patch_w.reshape((embed_dim, ch * ps * ps))?;
        let patch_b = vb.get_unchecked(&format!("{prefix}.patch_embedding.bias"))?;
        let dense_w = vb.get_unchecked(&format!("{prefix}.dense_embedding.weight"))?; // [llm, embed, f, f]
        let dense_b = vb.get_unchecked(&format!("{prefix}.dense_embedding.bias"))?; // [llm]
        Ok(Self {
            patch: Linear::new(patch_w, Some(patch_b)),
            dense_w,
            dense_b,
            embed_dim,
            downsample_factor: factor,
            rope_theta: cfg.vision.rope_theta_vision,
        })
    }

    /// Embed `pixel_values` `[N, ch·ps·ps]` (row-major patch list, channel-first patch layout) for the
    /// images described by `grid` (each `(h, w)` patch-grid). Returns `[Σ (h/f)·(w/f), llm_hidden]`
    /// tokens in row-major order, concatenated across images.
    pub fn forward(&self, pixel_values: &Tensor, grid: &[(usize, usize)]) -> CResult<Tensor> {
        let embed = self.embed_dim;
        // patch_embedding (full-kernel conv == linear over the flat patch) + erf-GELU.
        let pe = self.patch.forward(pixel_values)?.gelu_erf()?; // [N, embed]

        // Interleaved 2D RoPE (f32). Split the embedding in half: first half rotates by abs-x, second
        // by abs-y. The narrowed halves are made contiguous so the per-pair reshape inside the RoPE
        // helper is valid.
        let (abs_x, abs_y) = abs_positions(grid);
        let half = embed / 2;
        let h0 = pe.narrow(1, 0, half)?.contiguous()?;
        let h1 = pe.narrow(1, half, half)?.contiguous()?;
        let p1 = rope_1d_interleaved(&h0, &abs_x, self.rope_theta)?;
        let p2 = rope_1d_interleaved(&h1, &abs_y, self.rope_theta)?;
        let roped = Tensor::cat(&[&p1, &p2], 1)?; // [N, embed]

        // dense_embedding (factor×factor patch merge) per image, as an NCHW strided conv.
        let f = self.downsample_factor;
        let llm_dim = self.dense_b.dim(0)?;
        let dense_b = self.dense_b.reshape((1, llm_dim, 1, 1))?; // [1, llm, 1, 1]
        let mut outs: Vec<Tensor> = Vec::with_capacity(grid.len());
        let mut cur = 0usize;
        for &(h, w) in grid {
            let n = h * w;
            let block = roped
                .narrow(0, cur, n)? // [n, embed]
                .reshape((h, w, embed))?
                .permute((2, 0, 1))? // [embed, h, w]
                .unsqueeze(0)? // [1, embed, h, w]
                .contiguous()?;
            let merged = block.conv2d(&self.dense_w, 0, f, 1, 1)?; // [1, llm, h/f, w/f]
            let merged = merged.broadcast_add(&dense_b)?;
            let (_, llm, hf, wf) = merged.dims4()?;
            // [1, llm, h/f, w/f] -> [(h/f)·(w/f), llm], row-major.
            let tokens = merged
                .squeeze(0)?
                .permute((1, 2, 0))? // [h/f, w/f, llm]
                .contiguous()?
                .reshape((hf * wf, llm))?;
            outs.push(tokens);
            cur += n;
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        Tensor::cat(&refs, 0)
    }
}

/// Row-major patch coordinates `(abs_x, abs_y)` for the concatenated images: `abs_x = i % w`,
/// `abs_y = i / w` within each image.
fn abs_positions(grid: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for &(h, w) in grid {
        for i in 0..(h * w) {
            xs.push((i % w) as i32);
            ys.push((i / w) as i32);
        }
    }
    (xs, ys)
}

/// Interleaved 1D RoPE on `x` `[N, part]` at integer `positions` (length N), base `theta`. Pairs
/// `(x[2j], x[2j+1])` rotate by `positions · theta^(-2j/part)`. Mirrors `apply_rotary_emb_1d`.
fn rope_1d_interleaved(x: &Tensor, positions: &[i32], theta: f32) -> CResult<Tensor> {
    let (n, part) = x.dims2()?;
    let half = part / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / part as f32))
        .collect();
    let mut freqs = Vec::with_capacity(n * half);
    for &p in positions {
        for &f in &inv_freq {
            freqs.push(p as f32 * f);
        }
    }
    let freqs = Tensor::from_vec(freqs, (n, half), x.device())?;
    let cos = freqs.cos()?.reshape((n, half, 1))?;
    let sin = freqs.sin()?.reshape((n, half, 1))?;
    let xr = x.reshape((n, half, 2))?;
    let x1 = xr.narrow(2, 0, 1)?; // even lane [n, half, 1]
    let x2 = xr.narrow(2, 1, 1)?; // odd lane
    let rot1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rot2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    let out = Tensor::cat(&[&rot1, &rot2], 2)?; // [n, half, 2] interleaved
    out.reshape((n, part))
}
