//! Boogu's **Qwen3-VL vision tower** — candle (Windows/CUDA) port of `mlx-gen-boogu`'s `vision/`.
//! The ViT that turns a reference image into the merged vision tokens (and 3 deepstack features) the
//! MLLM consumes for image-conditioned editing (the Edit path; sc-7523).
//!
//! Port of `Qwen3VLVisionModel` (transformers `models/qwen3_vl/modeling_qwen3_vl.py`). Structure:
//!   - **Patch embed** — a `Conv3d` with kernel == stride == `[temporal 2, 16, 16]`; the full-window
//!     kernel is folded to a per-patch matmul (`[embed, in·t·ph·pw]`).
//!   - **Learned `pos_embed`** — an `nn.Embedding(num_position_embeddings, hidden)` (a `√n × √n` grid)
//!     **bilinearly interpolated** to the image grid (merge-grouped order) and added.
//!   - **`depth` blocks** — pre-`LayerNorm` (eps 1e-6) → full attention (fused-QKV + bias, 2-D NeoX
//!     half-split rotary, single-image ⇒ full unmasked) → `proj`; pre-LayerNorm → **GELU-tanh** MLP
//!     (`linear_fc1`/`linear_fc2`, bias). No windowing (unlike Qwen2.5-VL).
//!   - **Patch merger** — pre-shuffle `LayerNorm` → concat `merge²` (=4) group → `linear_fc1 →
//!     GELU(exact) → linear_fc2` → `out_hidden`.
//!   - **Deepstack** — at vision layers `deepstack_visual_indexes` ([8,16,24]), a post-shuffle-norm
//!     merger produces a feature the LM later injects into its early layers.
//!
//! The grid-derived host-side math (rope table, bilinear pos-embed indices/weights) mirrors the
//! reference `get_vision_position_ids` / `get_vision_bilinear_indices_and_weights`. Runs in **f32**
//! (parity-grade; image-embeds cosine 0.9998 vs the reference) — the DiT casts the features → bf16.

pub mod preprocess;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::rotary_emb::rope;
use candle_gen::candle_nn::{LayerNorm, Linear, Module};

use crate::loader::{linear, Weights};

const LN_EPS: f64 = 1e-6;
const ROPE_THETA: f32 = 10000.0;

/// Qwen3-VL vision-tower config (the `vision_config` block of `mllm/config.json`).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

impl VisionConfig {
    /// Boogu's Qwen3-VL-8B vision tower (`mllm/config.json::vision_config`).
    pub fn qwen3_vl() -> Self {
        Self {
            hidden_size: 1152,
            num_heads: 16,
            depth: 27,
            out_hidden_size: 4096,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            num_position_embeddings: 2304,
            deepstack_visual_indexes: vec![8, 16, 24],
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// `spatial_merge_size²` — patches per merged token.
    fn merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// `√num_position_embeddings` — the learned pos-embed grid side.
    fn num_grid_per_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }
}

/// Affine LayerNorm over the last dim (eps 1e-6), built from a `Weights` `{prefix}.weight`/`.bias`.
fn layer_norm(w: &Weights, prefix: &str) -> Result<LayerNorm> {
    let weight = w.get(&format!("{prefix}.weight"))?;
    let bias = w.get(&format!("{prefix}.bias"))?;
    Ok(LayerNorm::new(weight, bias, LN_EPS))
}

/// One vision block: pre-LayerNorm full attention + pre-LayerNorm GELU-tanh MLP, both residual.
struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Block {
    fn load(w: &Weights, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let head_dim = cfg.head_dim();
        Ok(Self {
            norm1: layer_norm(w, &format!("{prefix}.norm1"))?,
            norm2: layer_norm(w, &format!("{prefix}.norm2"))?,
            qkv: linear(w, &format!("{prefix}.attn.qkv"), true)?,
            proj: linear(w, &format!("{prefix}.attn.proj"), true)?,
            fc1: linear(w, &format!("{prefix}.mlp.linear_fc1"), true)?,
            fc2: linear(w, &format!("{prefix}.mlp.linear_fc2"), true)?,
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Full attention over `x` `[seq, hidden]` with precomputed `cos`/`sin` `[seq, head_dim/2]` (f32).
    /// Single-image ⇒ unmasked. NeoX half-split rope ([`rope`]) then `matmul → softmax → matmul`.
    fn attention(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let seq = x.dim(0)?;
        let (h, hd) = (self.num_heads, self.head_dim);

        let qkv = self.qkv.forward(x)?.reshape((seq, 3, h, hd))?;
        // Each → [1, h, seq, hd] for candle's [b, h, s, d] rope/attention layout.
        let to_heads = |idx: usize| -> Result<Tensor> {
            qkv.narrow(1, idx, 1)?
                .squeeze(1)? // [seq, h, hd]
                .transpose(0, 1)? // [h, seq, hd]
                .unsqueeze(0)? // [1, h, seq, hd]
                .contiguous()
        };
        let q = rope(&to_heads(0)?, cos, sin)?;
        let k = rope(&to_heads(1)?, cos, sin)?;
        let v = to_heads(2)?;

        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?; // [1, h, seq, hd]
        let o = o
            .squeeze(0)? // [h, seq, hd]
            .transpose(0, 1)? // [seq, h, hd]
            .contiguous()?
            .reshape((seq, h * hd))?;
        self.proj.forward(&o)
    }

    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = (x + self.attention(&self.norm1.forward(x)?, cos, sin)?)?;
        &x + self.mlp(&self.norm2.forward(&x)?)?
    }
}

/// Patch merger: `LayerNorm` → concat `merge²` group → `linear_fc1 → GELU(exact) → linear_fc2`.
/// The main merger norms **pre-shuffle** (over `hidden`); the deepstack mergers norm **post-shuffle**
/// (over `hidden·merge²`).
struct Merger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    postshuffle: bool,
    merged_dim: usize, // hidden · merge²
}

impl Merger {
    fn load(w: &Weights, prefix: &str, postshuffle: bool, merged_dim: usize) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(w, &format!("{prefix}.norm"))?,
            fc1: linear(w, &format!("{prefix}.linear_fc1"), true)?,
            fc2: linear(w, &format!("{prefix}.linear_fc2"), true)?,
            postshuffle,
            merged_dim,
        })
    }

    /// `x` `[seq, hidden]` → `[merged, out_hidden]` (`merged = seq / merge²`).
    fn forward(&self, x: &Tensor, merged: usize) -> Result<Tensor> {
        let x = if self.postshuffle {
            // group merge-units first, then norm over hidden·merge².
            let g = x.reshape((merged, self.merged_dim))?;
            self.norm.forward(&g)?
        } else {
            // norm over hidden per-patch, then group merge-units.
            let n = self.norm.forward(x)?;
            n.reshape((merged, self.merged_dim))?
        };
        self.fc2.forward(&self.fc1.forward(&x)?.gelu_erf()?)
    }
}

/// Host-side `grid_thw`-derived plan: the rope `cos`/`sin` (f32 `[seq, head_dim/2]`, merge-grouped
/// order) and the 4 bilinear corner indices + weights for the learned pos-embed interpolation.
struct Plan {
    merged: usize,
    cos: Tensor,               // f32 [seq, head_dim/2]
    sin: Tensor,               // f32 [seq, head_dim/2]
    bilinear_idx: [Tensor; 4], // u32 [seq]
    bilinear_w: [Tensor; 4],   // f32 [seq, 1]
}

/// The native Qwen3-VL vision tower.
pub struct VisionTower {
    patch_embed: Linear,
    pos_embed: Tensor, // [num_position_embeddings, hidden]
    blocks: Vec<Block>,
    merger: Merger,
    deepstack_mergers: Vec<Merger>,
    cfg: VisionConfig,
    device: Device,
}

impl VisionTower {
    /// Build from the mllm weight set (`{prefix}.*`, e.g. `"model.visual"`), loaded f32.
    pub fn load(w: &Weights, cfg: VisionConfig, prefix: &str) -> Result<Self> {
        // Fold the Conv3d patch-embed weight `[embed, in, t, ph, pw]` → `[embed, in·t·ph·pw]` so the
        // full-kernel conv runs as a per-patch matmul; keep its bias.
        let conv = w.get(&format!("{prefix}.patch_embed.proj.weight"))?;
        let dims = conv.dims();
        let embed = dims[0];
        let in_dim: usize = dims[1..].iter().product();
        let bias = w.get(&format!("{prefix}.patch_embed.proj.bias"))?;
        let patch_embed = Linear::new(conv.reshape((embed, in_dim))?, Some(bias));

        let blocks = (0..cfg.depth)
            .map(|i| Block::load(w, &format!("{prefix}.blocks.{i}"), &cfg))
            .collect::<Result<Vec<_>>>()?;

        let merged_dim = cfg.hidden_size * cfg.merge_unit();
        let merger = Merger::load(w, &format!("{prefix}.merger"), false, merged_dim)?;
        let deepstack_mergers = (0..cfg.deepstack_visual_indexes.len())
            .map(|i| {
                Merger::load(
                    w,
                    &format!("{prefix}.deepstack_merger_list.{i}"),
                    true,
                    merged_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            patch_embed,
            pos_embed: w.get(&format!("{prefix}.pos_embed.weight"))?,
            blocks,
            merger,
            deepstack_mergers,
            cfg,
            device: w.device().clone(),
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Host-side plan from `grid_thw` (rows `[t, h, w]` in patches), merge-grouped order — mirrors
    /// `get_vision_position_ids` (rope) + `get_vision_bilinear_indices_and_weights` (pos-embed).
    fn build_plan(&self, grid: &[[i32; 3]]) -> Result<Plan> {
        let c = &self.cfg;
        let m = c.spatial_merge_size as i32;
        let hd = c.head_dim();
        let rd = hd / 2; // rope width per token (= head_dim/2)
        let nfreq = rd / 2; // inv_freq length (= head_dim/4)
        let side = c.num_grid_per_side() as i32;
        let inv: Vec<f32> = (0..nfreq)
            .map(|j| ROPE_THETA.powf(-((2 * j) as f32) / rd as f32))
            .collect();

        let mut seq = 0usize;
        let mut merged = 0usize;
        let mut rope_rows: Vec<f32> = Vec::new(); // [seq, rd]
        let mut idx: [Vec<u32>; 4] = [vec![], vec![], vec![], vec![]];
        let mut wts: [Vec<f32>; 4] = [vec![], vec![], vec![], vec![]];

        for g in grid {
            let (t, h, w) = (g[0], g[1], g[2]);
            seq += (t * h * w) as usize;
            merged += (t * (h / m) * (w / m)) as usize;

            // linspace(0, side-1, n): value at index i.
            let lin = |i: i32, n: i32| -> f64 {
                if n <= 1 {
                    0.0
                } else {
                    (side - 1) as f64 * i as f64 / (n - 1) as f64
                }
            };

            for _f in 0..t {
                for bh in 0..(h / m) {
                    for bw in 0..(w / m) {
                        for ih in 0..m {
                            for iw in 0..m {
                                let hpos = bh * m + ih;
                                let wpos = bw * m + iw;
                                // rope row: [hpos·inv(nfreq), wpos·inv(nfreq)] → rd.
                                for &fq in &inv {
                                    rope_rows.push(hpos as f32 * fq);
                                }
                                for &fq in &inv {
                                    rope_rows.push(wpos as f32 * fq);
                                }
                                // bilinear pos-embed interpolation corners (into the side×side grid).
                                let hc = lin(hpos, h);
                                let wc = lin(wpos, w);
                                let hf = hc.floor();
                                let wf = wc.floor();
                                let h0 = hf as i32;
                                let w0 = wf as i32;
                                let h1 = (h0 + 1).min(side - 1);
                                let w1 = (w0 + 1).min(side - 1);
                                let hfr = (hc - hf) as f32;
                                let wfr = (wc - wf) as f32;
                                idx[0].push((h0 * side + w0) as u32);
                                idx[1].push((h0 * side + w1) as u32);
                                idx[2].push((h1 * side + w0) as u32);
                                idx[3].push((h1 * side + w1) as u32);
                                wts[0].push((1.0 - hfr) * (1.0 - wfr));
                                wts[1].push((1.0 - hfr) * wfr);
                                wts[2].push(hfr * (1.0 - wfr));
                                wts[3].push(hfr * wfr);
                            }
                        }
                    }
                }
            }
        }

        let rope = Tensor::from_vec(rope_rows, (seq, rd), &self.device)?;
        let cos = rope.cos()?;
        let sin = rope.sin()?;
        let mk_i = |v: &[u32]| Tensor::from_vec(v.to_vec(), (seq,), &self.device);
        let mk_w = |v: &[f32]| Tensor::from_vec(v.to_vec(), (seq, 1), &self.device);
        Ok(Plan {
            merged,
            cos,
            sin,
            bilinear_idx: [
                mk_i(&idx[0])?,
                mk_i(&idx[1])?,
                mk_i(&idx[2])?,
                mk_i(&idx[3])?,
            ],
            bilinear_w: [
                mk_w(&wts[0])?,
                mk_w(&wts[1])?,
                mk_w(&wts[2])?,
                mk_w(&wts[3])?,
            ],
        })
    }

    /// Bilinearly-interpolated learned pos-embed `[seq, hidden]` (f32) for the plan.
    fn pos_embeds(&self, plan: &Plan) -> Result<Tensor> {
        let pe = self.pos_embed.to_dtype(DType::F32)?;
        let mut acc: Option<Tensor> = None;
        for k in 0..4 {
            let gathered = pe.index_select(&plan.bilinear_idx[k], 0)?; // [seq, hidden]
            let term = gathered.broadcast_mul(&plan.bilinear_w[k])?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        Ok(acc.unwrap())
    }

    /// Encode packed patches → (merged image embeds `[merged, out_hidden]`, deepstack features —
    /// one `[merged, out_hidden]` per `deepstack_visual_indexes` entry).
    ///
    /// `pixel_values` is `[seq, in·t·ph·pw]` (f32); `grid_thw` rows are `[t, h, w]` (patches).
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[[i32; 3]],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let plan = self.build_plan(grid_thw)?;
        let merged = plan.merged;

        // Patch embed + learned (interpolated) position embedding (all f32).
        let h = self.patch_embed.forward(pixel_values)?;
        let pos = self.pos_embeds(&plan)?;
        let mut h = (h + pos)?;

        let mut deepstack = Vec::with_capacity(self.cfg.deepstack_visual_indexes.len());
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h, &plan.cos, &plan.sin)?;
            if let Some(di) = self
                .cfg
                .deepstack_visual_indexes
                .iter()
                .position(|&x| x == i)
            {
                deepstack.push(self.deepstack_mergers[di].forward(&h, merged)?);
            }
        }

        let embeds = self.merger.forward(&h, merged)?;
        Ok((embeds, deepstack))
    }
}
