//! `Embeddings1DConnector` — the LTX-2.3 video text-feature connector. Port of mlx-gen-ltx
//! `connector.rs`. An 8-layer pre-norm transformer over the Gemma feature-extractor output (dim
//! 4096 = 32×128): per-block unit-weight RMSNorm → **gated** attention (q/k RMSNorm + 1-D split
//! RoPE, per-head sigmoid gate) → unit-RMSNorm → exact-gelu MLP (inner 16384), then a final
//! unit-RMSNorm. **128 learnable registers** replace the left-padding slots. Runs bf16; attention
//! computes in f32.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{ops::rms_norm, ops::softmax_last_dim, Linear, Module, VarBuilder};

use crate::config::ConnectorConfig;
use crate::rope::{apply_split_rope, precompute_connector_freqs};

const EPS: f32 = 1e-6;

fn linear(vb: &VarBuilder, key: &str) -> Result<Linear> {
    let w = vb
        .get_unchecked(&format!("{key}.weight"))?
        .to_dtype(DType::BF16)?;
    let b = vb
        .get_unchecked(&format!("{key}.bias"))?
        .to_dtype(DType::BF16)?;
    Ok(Linear::new(w, Some(b)))
}

struct ConnectorBlock {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Linear,
    ff_in: Linear,
    ff_out: Linear,
}

pub struct Connector {
    blocks: Vec<ConnectorBlock>,
    registers: Tensor, // [num_registers, dim]
    ones: Tensor,      // unit RMSNorm weight [dim]
    cfg: ConnectorConfig,
    device: Device,
}

impl Connector {
    /// Build from a VarBuilder rooted at the DiT prefix, under `video_embeddings_connector`.
    pub fn new(vb: VarBuilder, cfg: &ConnectorConfig) -> Result<Self> {
        let device = vb.device().clone();
        let cvb = vb.pp("video_embeddings_connector");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let b = cvb.pp(format!("transformer_1d_blocks.{i}"));
            let a = b.pp("attn1");
            blocks.push(ConnectorBlock {
                to_q: linear(&a, "to_q")?,
                to_k: linear(&a, "to_k")?,
                to_v: linear(&a, "to_v")?,
                to_out: linear(&a, "to_out.0")?,
                q_norm: a.get_unchecked("q_norm.weight")?.to_dtype(DType::BF16)?,
                k_norm: a.get_unchecked("k_norm.weight")?.to_dtype(DType::BF16)?,
                gate: linear(&a, "to_gate_logits")?,
                ff_in: linear(&b.pp("ff.net.0"), "proj")?,
                ff_out: linear(&b.pp("ff.net"), "2")?,
            });
        }
        let registers = cvb
            .get_unchecked("learnable_registers")?
            .to_dtype(DType::BF16)?;
        let ones = Tensor::ones(cfg.inner_dim(), DType::BF16, &device)?;
        Ok(Self {
            blocks,
            registers,
            ones,
            cfg: cfg.clone(),
            device,
        })
    }

    /// Replace left-padding with learnable registers (batch 1): the trailing `nv` valid tokens move
    /// to the front; registers tile-fill the tail to length `s`.
    fn replace_with_registers(&self, x: &Tensor, nv: usize) -> Result<Tensor> {
        let (_b, s, dim) = x.dims3()?;
        if nv >= s {
            return Ok(x.clone());
        }
        let num_reg = self.registers.dim(0)?;
        let num_tiles = s / num_reg;
        // tile registers to (s, dim) then (1, s, dim).
        let reg_full = Tensor::cat(&vec![&self.registers; num_tiles], 0)?.reshape((1, s, dim))?;
        let valid = x.narrow(1, s - nv, nv)?; // trailing nv real tokens
        let reg_tail = reg_full.narrow(1, nv, s - nv)?;
        Tensor::cat(&[&valid, &reg_tail], 1)
    }

    fn attn(&self, blk: &ConnectorBlock, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, d) = (self.cfg.num_heads, self.cfg.head_dim);
        let q = rms_norm(&blk.to_q.forward(x)?.contiguous()?, &blk.q_norm, EPS)?;
        let k = rms_norm(&blk.to_k.forward(x)?.contiguous()?, &blk.k_norm, EPS)?;
        let v = blk.to_v.forward(x)?;
        let q = q.reshape((b, s, h, d))?.transpose(1, 2)?;
        let k = k.reshape((b, s, h, d))?.transpose(1, 2)?;
        let v = v.reshape((b, s, h, d))?.transpose(1, 2)?;
        let q = apply_split_rope(&q, cos, sin)?;
        let k = apply_split_rope(&k, cos, sin)?;
        // Attention in f32, no mask (full attention over real tokens + registers).
        let qf = q.to_dtype(DType::F32)?.contiguous()?;
        let kf = k.to_dtype(DType::F32)?.contiguous()?;
        let vf = v.to_dtype(DType::F32)?.contiguous()?;
        let scale = 1.0 / (d as f64).sqrt();
        let scores = (qf.matmul(&kf.transpose(2, 3)?)? * scale)?;
        let out = softmax_last_dim(&scores)?.matmul(&vf)?; // (b,h,s,d)
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, h * d))?
            .to_dtype(DType::BF16)?;
        // Per-head gate: out *= sigmoid(gate(x)).
        let gates =
            candle_gen::candle_nn::ops::sigmoid(&blk.gate.forward(x)?)?.reshape((b, s, h, 1))?;
        let out = out
            .reshape((b, s, h, d))?
            .broadcast_mul(&gates)?
            .reshape((b, s, h * d))?;
        blk.to_out.forward(&out)
    }

    fn block(
        &self,
        blk: &ConnectorBlock,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let n = rms_norm(&x.contiguous()?, &self.ones, EPS)?;
        let x = (x + self.attn(blk, &n, cos, sin)?)?;
        let n = rms_norm(&x.contiguous()?, &self.ones, EPS)?;
        let ff = blk.ff_out.forward(&blk.ff_in.forward(&n)?.gelu_erf()?)?;
        &x + ff
    }

    /// Run the connector. `x` = `[1, seq, 4096]` feature-extractor output (bf16); `nv` = number of
    /// valid (non-padding) tokens. Returns video embeddings `[1, seq, 4096]`.
    pub fn forward(&self, x: &Tensor, nv: usize) -> Result<Tensor> {
        let mut h = self.replace_with_registers(&x.to_dtype(DType::BF16)?, nv)?;
        let seq = h.dim(1)?;
        let (cos, sin) = precompute_connector_freqs(
            seq,
            self.cfg.inner_dim(),
            self.cfg.rope_theta,
            self.cfg.max_pos,
            self.cfg.num_heads,
            &self.device,
        )?;
        for blk in &self.blocks {
            h = self.block(blk, &h, &cos, &sin)?;
        }
        rms_norm(&h.contiguous()?, &self.ones, EPS)
    }
}
