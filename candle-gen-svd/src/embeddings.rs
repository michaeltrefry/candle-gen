//! SVD time/frame embeddings — the diffusers `Timesteps` sinusoidal encoder + the 2-layer
//! `TimestepEmbedding` MLP, shared by the UNet (timestep + `added_time_ids`) and each
//! `TransformerSpatioTemporalModel` (per-frame `time_pos_embed`). candle port of `mlx-gen-svd`'s
//! `embeddings.rs`.

use candle_gen::candle_core::{Device, Result, Tensor, D};
use candle_gen::candle_nn::{linear, Linear, Module, VarBuilder};

/// diffusers `get_timestep_embedding(x, dim, flip_sin_to_cos=True, downscale_freq_shift=0,
/// max_period=10000)`: `freq_i = 10000^(−i/half)` (`i∈[0,half)`), `emb = x[:,None]·freq`, output
/// `concat([cos(emb), sin(emb)], -1)` (cos first). `x` is `[N]` → returns `[N, dim]`. The frequencies
/// are computed on the host in the same f32 op order as the MLX/diffusers path so the rounding matches.
pub fn sinusoidal_timestep(x: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    assert!(
        dim.is_multiple_of(2),
        "svd sinusoidal_timestep: dim must be even, got {dim}"
    );
    let half = dim / 2;
    let neg_ln = -(10000f64.ln());
    // freq_i = exp((i · neg_ln) / half) — host f32, matching diffusers `(−ln·arange)/half`.
    let freqs: Vec<f32> = (0..half)
        .map(|i| (((i as f32) * (neg_ln as f32)) / (half as f32)).exp())
        .collect();
    let freqs = Tensor::from_vec(freqs, half, device)?; // [half]
                                                        // emb = x[:, None] · freqs → [N, half].
    let emb = x.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[emb.cos()?, emb.sin()?], D::Minus1) // [N, dim], cos first
}

/// The 2-layer time-embedding MLP (`linear_1 → SiLU → linear_2`). `out_dim` differs from the input
/// only for the transformer `time_pos_embed` (C→C·4→C); the UNet's `time_embedding`/`add_embedding`
/// map into the 1280-wide embedding.
pub struct TimestepEmbedding {
    lin1: Linear,
    lin2: Linear,
}

impl TimestepEmbedding {
    pub fn load(in_dim: usize, mid_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin1: linear(in_dim, mid_dim, vb.pp("linear_1"))?,
            lin2: linear(mid_dim, out_dim, vb.pp("linear_2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.lin1.forward(x)?.silu()?;
        self.lin2.forward(&x)
    }
}
