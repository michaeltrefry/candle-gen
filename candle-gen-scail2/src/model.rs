//! The SCAIL-2 DiT forward (`SCAIL2Model.forward`, upstream `wan/modules/model_scail2.py`).
//!
//! SCAIL-2 is a Wan2.1-14B **I2V** diffusion backbone with Bernini-family **packed-token**
//! conditioning. Each denoise step assembles one self-attention sequence from up to four token chunks
//! — `[additional_ref | ref | video | pose]` — embeds them through three Conv3d patch stems (latent /
//! pose / 28-channel color-coded mask, the mask & pose embeds *added* onto the latent embeds), applies
//! a per-chunk 3-axis RoPE (the [`crate::rope::ScailRope`] shifts; `replace_flag` toggles the
//! reference H-shift between animation and cross-identity replacement), runs the Wan blocks with
//! **I2V image cross-attention** (CLIP image tokens via `k_img`/`v_img` alongside the UMT5 text
//! tokens), and finally keeps only the video tokens (`unpatchify` at `offset = additional_ref + ref`).
//!
//! The DiT runs in **f32** end-to-end: bf16 overflows to NaN at high token length (sc-5446 finding), and
//! the f32 14B params (~28 GiB) fit the 96 GiB `minMemoryGb` budget. The `Conv3d` patch weights
//! `[out, in, 1, 2, 2]` are read as `[out, in·4]` Linears via [`crate::common::conv_as_linear`].

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen_wan::rope::apply_rope;

use crate::common::{
    conv_as_linear, linear, ln_affine, ln_no_affine, patchify, rms, sdpa, unpatchify,
};
use crate::config::Scail2Config;
use crate::rope::ScailRope;

/// `nn.LayerNorm` default eps (the `img_emb` MLPProj LayerNorms). The DiT's own `WanLayerNorm` uses
/// `cfg.eps` (1e-6) instead.
const IMG_LN_EPS: f64 = 1e-5;

/// adaLN affine `m·(1+scale)+shift` (broadcasting `scale`/`shift` `[1,1,dim]` over the token axis).
fn modulate(m: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    m.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// Gated residual `x + y·gate`.
fn gated(x: &Tensor, y: &Tensor, gate: &Tensor) -> Result<Tensor> {
    x.broadcast_add(&y.broadcast_mul(gate)?)
}

/// Wan self-attention with qk-RMSNorm and 3-axis RoPE, over the full packed sequence.
struct SelfAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    n: usize,
    d: usize,
    scale: f64,
    eps: f64,
}

impl SelfAttn {
    fn new(vb: &VarBuilder, cfg: &Scail2Config) -> Result<Self> {
        let head_dim = cfg.head_dim();
        Ok(Self {
            q: linear(cfg.dim, cfg.dim, vb.pp("q"))?,
            k: linear(cfg.dim, cfg.dim, vb.pp("k"))?,
            v: linear(cfg.dim, cfg.dim, vb.pp("v"))?,
            o: linear(cfg.dim, cfg.dim, vb.pp("o"))?,
            norm_q: vb.pp("norm_q").get(cfg.dim, "weight")?,
            norm_k: vb.pp("norm_k").get(cfg.dim, "weight")?,
            n: cfg.num_heads,
            d: head_dim,
            scale: (head_dim as f64).powf(-0.5),
            eps: cfg.eps,
        })
    }

    /// `x`: `[1, L, dim]` (f32, already adaLN-modulated). `cos`/`sin`: `[L, half_d]` (f32).
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (n, d) = (self.n, self.d);
        let q = rms(&self.q.forward(x)?, &self.norm_q, self.eps)?;
        let k = rms(&self.k.forward(x)?, &self.norm_k, self.eps)?;
        let v = self.v.forward(x)?;
        let to_heads = |t: &Tensor| -> Result<Tensor> {
            t.reshape((b, s, n, d))?.transpose(1, 2)?.contiguous()
        };
        let q = apply_rope(&to_heads(&q)?, cos, sin)?;
        let k = apply_rope(&to_heads(&k)?, cos, sin)?;
        let v = to_heads(&v)?;
        let out = sdpa(&q, &k, &v, self.scale)?;
        let out = out.transpose(1, 2)?.reshape((b, s, n * d))?;
        self.o.forward(&out)
    }
}

/// Wan **I2V** cross-attention: text tokens through `k`/`v`, CLIP image tokens through `k_img`/`v_img`;
/// the two attention outputs are summed before the output projection.
struct CrossAttnI2V {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    k_img: Linear,
    v_img: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    norm_k_img: Tensor,
    n: usize,
    d: usize,
    scale: f64,
    eps: f64,
}

impl CrossAttnI2V {
    fn new(vb: &VarBuilder, cfg: &Scail2Config) -> Result<Self> {
        let head_dim = cfg.head_dim();
        Ok(Self {
            q: linear(cfg.dim, cfg.dim, vb.pp("q"))?,
            k: linear(cfg.dim, cfg.dim, vb.pp("k"))?,
            v: linear(cfg.dim, cfg.dim, vb.pp("v"))?,
            o: linear(cfg.dim, cfg.dim, vb.pp("o"))?,
            k_img: linear(cfg.dim, cfg.dim, vb.pp("k_img"))?,
            v_img: linear(cfg.dim, cfg.dim, vb.pp("v_img"))?,
            norm_q: vb.pp("norm_q").get(cfg.dim, "weight")?,
            norm_k: vb.pp("norm_k").get(cfg.dim, "weight")?,
            norm_k_img: vb.pp("norm_k_img").get(cfg.dim, "weight")?,
            n: cfg.num_heads,
            d: head_dim,
            scale: (head_dim as f64).powf(-0.5),
            eps: cfg.eps,
        })
    }

    /// `x`: `[1, L, dim]` (f32). `text_ctx`: `[1, L_text, dim]`. `img_ctx`: `[1, L_img, dim]`.
    fn forward(&self, x: &Tensor, text_ctx: &Tensor, img_ctx: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (n, d) = (self.n, self.d);
        let to_heads = |t: &Tensor, len: usize| -> Result<Tensor> {
            t.reshape((b, len, n, d))?.transpose(1, 2)?.contiguous()
        };
        let q = to_heads(&rms(&self.q.forward(x)?, &self.norm_q, self.eps)?, s)?;
        let lt = text_ctx.dim(1)?;
        let li = img_ctx.dim(1)?;
        let k = to_heads(
            &rms(&self.k.forward(text_ctx)?, &self.norm_k, self.eps)?,
            lt,
        )?;
        let v = to_heads(&self.v.forward(text_ctx)?, lt)?;
        let k_img = to_heads(
            &rms(&self.k_img.forward(img_ctx)?, &self.norm_k_img, self.eps)?,
            li,
        )?;
        let v_img = to_heads(&self.v_img.forward(img_ctx)?, li)?;
        let flat = |o: Tensor| -> Result<Tensor> { o.transpose(1, 2)?.reshape((b, s, n * d)) };
        let x_txt = flat(sdpa(&q, &k, &v, self.scale)?)?;
        let x_img = flat(sdpa(&q, &k_img, &v_img, self.scale)?)?;
        self.o.forward(&(x_txt + x_img)?)
    }
}

/// One Wan attention block: adaLN-6vec modulation → self-attn (gated residual) → affine-LN +
/// I2V cross-attn → adaLN FFN (gated residual).
struct Block {
    modulation: Tensor, // [1, 6, dim] f32
    self_attn: SelfAttn,
    cross: CrossAttnI2V,
    norm3_w: Tensor,
    norm3_b: Tensor,
    ffn0: Linear,
    ffn2: Linear,
    eps: f64,
}

impl Block {
    fn new(vb: &VarBuilder, cfg: &Scail2Config) -> Result<Self> {
        Ok(Self {
            modulation: vb.get((1, 6, cfg.dim), "modulation")?,
            self_attn: SelfAttn::new(&vb.pp("self_attn"), cfg)?,
            cross: CrossAttnI2V::new(&vb.pp("cross_attn"), cfg)?,
            norm3_w: vb.pp("norm3").get(cfg.dim, "weight")?,
            norm3_b: vb.pp("norm3").get(cfg.dim, "bias")?,
            ffn0: linear(cfg.dim, cfg.ffn_dim, vb.pp("ffn").pp("0"))?,
            ffn2: linear(cfg.ffn_dim, cfg.dim, vb.pp("ffn").pp("2"))?,
            eps: cfg.eps,
        })
    }

    /// `x`: `[1, L, dim]` (f32). `e0`: `[1, 6, dim]` (f32, time modulation).
    fn forward(
        &self,
        x: &Tensor,
        e0: &Tensor,
        text_ctx: &Tensor,
        img_ctx: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let m = self.modulation.broadcast_add(e0)?; // [1, 6, dim]
        let p = |i: usize| -> Result<Tensor> { m.narrow(1, i, 1) }; // [1, 1, dim]

        // self-attention
        let x_mod = modulate(&ln_no_affine(x, self.eps)?, &p(1)?, &p(0)?)?;
        let y = self.self_attn.forward(&x_mod, cos, sin)?;
        let x = gated(x, &y, &p(2)?)?;
        // cross-attention (affine LN, ungated)
        let x_cross = ln_affine(&x, &self.norm3_w, &self.norm3_b, self.eps)?;
        let cx = self.cross.forward(&x_cross, text_ctx, img_ctx)?;
        let x = (x + cx)?;
        // feed-forward
        let x_mod = modulate(&ln_no_affine(&x, self.eps)?, &p(4)?, &p(3)?)?;
        let y = self.ffn2.forward(&self.ffn0.forward(&x_mod)?.gelu()?)?;
        gated(&x, &y, &p(5)?)
    }
}

/// All conditioning tensors for one denoise-step forward. Spatial dims are latent (`vae_stride`-down)
/// dims; channel counts are `vae_z_dim` (16) for latents and `mask_dim` (28) for masks. All f32.
pub struct Scail2Inputs<'a> {
    /// Noisy video latent `[16, T, H, W]`.
    pub x: &'a Tensor,
    /// Reference-character latent `[16, 1, H, W]`.
    pub ref_latent: &'a Tensor,
    /// Reference mask latent `[28, 1+T, H, W]`.
    pub ref_masks: &'a Tensor,
    /// Driving-pose latent `[16, T, H/2, W/2]` (half spatial res).
    pub pose_latent: &'a Tensor,
    /// Driving-mask latent `[28, T, H/2, W/2]`.
    pub driving_masks: &'a Tensor,
    /// Clean-history mask `[4, T, H, W]` (segment > 0); `None` appends the i2v zero-mask.
    pub history_mask: Option<&'a Tensor>,
    /// Extra-character latents `[16, n, H, W]` (multi-reference); `None` for single reference.
    pub additional_ref_latent: Option<&'a Tensor>,
    /// Extra-character mask latents `[28, n, H, W]`; required iff `additional_ref_latent` is set.
    pub additional_ref_masks: Option<&'a Tensor>,
    /// CLIP image features `[1, 257, 1280]` from the open-CLIP XLM-RoBERTa ViT-H/14 visual tower.
    pub clip_fea: &'a Tensor,
    /// UMT5 text embeddings `[L_text, 4096]`.
    pub context: &'a Tensor,
    /// Diffusion timestep.
    pub t: f64,
    /// `true` = cross-identity replacement (ref H-shift 120), `false` = animation (H-shift 0).
    pub replace_flag: bool,
}

/// The loaded SCAIL-2 DiT.
pub struct Scail2Dit {
    patch_embedding: Linear,
    patch_embedding_pose: Linear,
    patch_embedding_mask: Linear,
    text_embedding_0: Linear,
    text_embedding_2: Linear,
    time_embedding_0: Linear,
    time_embedding_2: Linear,
    time_projection: Linear,
    img_ln0_w: Tensor,
    img_ln0_b: Tensor,
    img_emb_1: Linear,
    img_emb_3: Linear,
    img_ln4_w: Tensor,
    img_ln4_b: Tensor,
    blocks: Vec<Block>,
    head_modulation: Tensor, // [1, 2, dim] f32
    head: Linear,
    rope: ScailRope,
    cfg: Scail2Config,
    device: Device,
}

impl Scail2Dit {
    /// Load the DiT from a `VarBuilder` over the converted `SCAIL2Model` parameters (raw param names:
    /// `patch_embedding{,_pose,_mask}`, `blocks.{i}.{self_attn,cross_attn,...}`, `img_emb.proj.*`,
    /// `time_*`, `text_embedding.*`, `head.*`). The `VarBuilder` should be f32.
    pub fn new(vb: VarBuilder, cfg: &Scail2Config) -> Result<Self> {
        let p3 = [cfg.patch.0, cfg.patch.1, cfg.patch.2];
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::new(&vb.pp("blocks").pp(i), cfg)?);
        }
        Ok(Self {
            patch_embedding: conv_as_linear(
                cfg.dim,
                cfg.in_dim,
                &p3,
                "patch_embedding.weight",
                Some("patch_embedding.bias"),
                &vb,
            )?,
            patch_embedding_pose: conv_as_linear(
                cfg.dim,
                cfg.in_dim,
                &p3,
                "patch_embedding_pose.weight",
                Some("patch_embedding_pose.bias"),
                &vb,
            )?,
            patch_embedding_mask: conv_as_linear(
                cfg.dim,
                cfg.mask_dim,
                &p3,
                "patch_embedding_mask.weight",
                Some("patch_embedding_mask.bias"),
                &vb,
            )?,
            text_embedding_0: linear(cfg.text_dim, cfg.dim, vb.pp("text_embedding").pp("0"))?,
            text_embedding_2: linear(cfg.dim, cfg.dim, vb.pp("text_embedding").pp("2"))?,
            time_embedding_0: linear(cfg.freq_dim, cfg.dim, vb.pp("time_embedding").pp("0"))?,
            time_embedding_2: linear(cfg.dim, cfg.dim, vb.pp("time_embedding").pp("2"))?,
            time_projection: linear(cfg.dim, 6 * cfg.dim, vb.pp("time_projection").pp("1"))?,
            img_ln0_w: vb.pp("img_emb").pp("proj").pp("0").get(1280, "weight")?,
            img_ln0_b: vb.pp("img_emb").pp("proj").pp("0").get(1280, "bias")?,
            // Wan-I2V `img_emb` MLPProj (`proj.0` LN → `proj.1` Linear → GELU → `proj.3` Linear →
            // `proj.4` LN): the intermediate Linear keeps the CLIP width (1280→1280); only `proj.3`
            // projects up to the DiT `dim` (1280→5120). (Both were mis-sized to `dim` before, which the
            // real-weight load rejected — the checkpoint has `proj.1 = [1280,1280]`, `proj.3 = [5120,1280]`.)
            img_emb_1: linear(1280, 1280, vb.pp("img_emb").pp("proj").pp("1"))?,
            img_emb_3: linear(1280, cfg.dim, vb.pp("img_emb").pp("proj").pp("3"))?,
            img_ln4_w: vb.pp("img_emb").pp("proj").pp("4").get(cfg.dim, "weight")?,
            img_ln4_b: vb.pp("img_emb").pp("proj").pp("4").get(cfg.dim, "bias")?,
            blocks,
            head_modulation: vb.pp("head").get((1, 2, cfg.dim), "modulation")?,
            head: linear(
                cfg.dim,
                cfg.out_dim * cfg.patch.0 * cfg.patch.1 * cfg.patch.2,
                vb.pp("head").pp("head"),
            )?,
            rope: ScailRope::new(cfg.head_dim()),
            cfg: cfg.clone(),
            device: vb.device().clone(),
        })
    }

    /// Number of transformer blocks (40 for the 14B).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The device the DiT weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Sinusoidal timestep embedding → `(e, e0)`: `e` `[1, dim]` (head modulation), `e0` `[1, 6, dim]`
    /// (block modulation). Built in f64 then cast to f32, matching upstream `sinusoidal_embedding_1d`.
    fn time_embed(
        &self,
        t: f64,
        dev: &candle_gen::candle_core::Device,
    ) -> Result<(Tensor, Tensor)> {
        let freq_dim = self.cfg.freq_dim;
        let half = freq_dim / 2;
        let mut emb = vec![0f32; freq_dim];
        for j in 0..half {
            let ang = t * 10000f64.powf(-(j as f64) / half as f64);
            emb[j] = ang.cos() as f32;
            emb[half + j] = ang.sin() as f32;
        }
        let sin_emb = Tensor::from_vec(emb, (1, freq_dim), dev)?;
        let e = self
            .time_embedding_2
            .forward(&self.time_embedding_0.forward(&sin_emb)?.silu()?)?;
        let e0 = self
            .time_projection
            .forward(&e.silu()?)?
            .reshape((1, 6, self.cfg.dim))?;
        Ok((e, e0))
    }

    /// UMT5 text embeddings `[L, 4096]` → `[1, text_len, dim]` (zero-padded to `text_len`).
    fn embed_text(&self, context: &Tensor) -> Result<Tensor> {
        let text_len = self.cfg.text_len;
        let (l, td) = context.dims2()?;
        let ctx = if l < text_len {
            let pad = Tensor::zeros((text_len - l, td), context.dtype(), context.device())?;
            Tensor::cat(&[context, &pad], 0)?
        } else if l > text_len {
            context.narrow(0, 0, text_len)?
        } else {
            context.clone()
        };
        let ctx = ctx.reshape((1, text_len, td))?;
        let h = self.text_embedding_0.forward(&ctx)?.gelu()?;
        self.text_embedding_2.forward(&h)
    }

    /// CLIP features `[1, 257, 1280]` → image context `[1, 257, dim]` via the MLPProj
    /// (LayerNorm → Linear → exact-GELU → Linear → LayerNorm).
    fn embed_img(&self, clip_fea: &Tensor) -> Result<Tensor> {
        let h = ln_affine(clip_fea, &self.img_ln0_w, &self.img_ln0_b, IMG_LN_EPS)?;
        let h = self.img_emb_1.forward(&h)?.gelu_erf()?;
        let h = self.img_emb_3.forward(&h)?;
        ln_affine(&h, &self.img_ln4_w, &self.img_ln4_b, IMG_LN_EPS)
    }

    /// Modulated output head: `[1, L, dim]` → `[1, L, out_dim·∏patch]`.
    fn apply_head(&self, x: &Tensor, e: &Tensor) -> Result<Tensor> {
        let m = self
            .head_modulation
            .broadcast_add(&e.reshape((1, 1, self.cfg.dim))?)?; // [1, 2, dim]
        let shift = m.narrow(1, 0, 1)?;
        let scale = m.narrow(1, 1, 1)?;
        let x_mod = modulate(&ln_no_affine(x, self.cfg.eps)?, &scale, &shift)?;
        self.head.forward(&x_mod)
    }

    /// One denoise-step velocity prediction → `[16, T, H, W]` (single sample).
    pub fn forward(&self, inp: &Scail2Inputs) -> Result<Tensor> {
        let cfg = &self.cfg;
        let dev = inp.x.device();
        let ps = cfg.patch;
        let i2v = cfg.i2v_mask_dim;
        let dim = cfg.dim;

        let (_xc, tt, hh, ww) = inp.x.dims4()?;

        // --- append the i2v binary-mask channels (in_dim 20 = 16 + 4) ---
        let x20 = match inp.history_mask {
            Some(hm) => Tensor::cat(&[inp.x, hm], 0)?,
            None => Tensor::cat(
                &[inp.x, &Tensor::zeros((i2v, tt, hh, ww), DType::F32, dev)?],
                0,
            )?,
        };
        let ref20 = Tensor::cat(
            &[
                inp.ref_latent,
                &Tensor::ones((i2v, 1, hh, ww), DType::F32, dev)?,
            ],
            0,
        )?;
        let (_pc, pose_t, pose_h, pose_w) = inp.pose_latent.dims4()?;
        let pose20 = Tensor::cat(
            &[
                inp.pose_latent,
                &Tensor::ones((i2v, pose_t, pose_h, pose_w), DType::F32, dev)?,
            ],
            0,
        )?;

        // --- patch grids / chunk lengths (patch (1,2,2)) ---
        let rope_t = tt / ps.0;
        let rope_h = hh / ps.1;
        let rope_w = ww / ps.2;
        let ref_length = rope_h * rope_w;
        let seq_length = rope_t * rope_h * rope_w;
        let h_shift = if inp.replace_flag {
            cfg.replace_h_shift
        } else {
            0
        };
        let base_video_shift = 1usize;

        // --- patch-embed stems (ref+video share patch_embedding; mask/pose added) ---
        let refvid = Tensor::cat(&[&ref20, &x20], 1)?; // [20, 1+T, H, W]
        let (rv_tok, _) = patchify(&refvid, ps)?;
        let (rm_tok, _) = patchify(inp.ref_masks, ps)?;
        let refvid_emb = (self.patch_embedding.forward(&rv_tok)?
            + self.patch_embedding_mask.forward(&rm_tok)?)?;
        let (pose_tok, _) = patchify(&pose20, ps)?;
        let (dm_tok, _) = patchify(inp.driving_masks, ps)?;
        let pose_emb = (self.patch_embedding_pose.forward(&pose_tok)?
            + self.patch_embedding_mask.forward(&dm_tok)?)?;

        // --- assemble packed tokens + per-chunk RoPE: [additional_ref | ref | video | pose] ---
        let mut tok_list: Vec<Tensor> = Vec::new();
        let mut cos_list: Vec<Tensor> = Vec::new();
        let mut sin_list: Vec<Tensor> = Vec::new();
        let mut addref_count = 0usize;

        if let Some(ar) = inp.additional_ref_latent {
            let arm = inp.additional_ref_masks.ok_or_else(|| {
                candle_gen::candle_core::Error::Msg(
                    "scail2: additional_ref_masks required with additional_ref_latent".into(),
                )
            })?;
            let (_arc, ar_n, _arh, _arw) = ar.dims4()?;
            let ar20 = Tensor::cat(
                &[ar, &Tensor::ones((i2v, ar_n, hh, ww), DType::F32, dev)?],
                0,
            )?;
            let (ar_tok, _) = patchify(&ar20, ps)?;
            let (arm_tok, _) = patchify(arm, ps)?;
            let ar_emb = (self.patch_embedding.forward(&ar_tok)?
                + self.patch_embedding_mask.forward(&arm_tok)?)?;
            addref_count = ar_n;
            let (c, s) =
                self.rope
                    .chunk((addref_count, rope_h, rope_w), (0, h_shift, 0), false, dev)?;
            tok_list.push(ar_emb);
            cos_list.push(c);
            sin_list.push(s);
        }

        // ref+video tokens (one block); RoPE splits ref (1 frame) and video (rope_t frames).
        tok_list.push(refvid_emb);
        let (rc, rs) =
            self.rope
                .chunk((1, rope_h, rope_w), (addref_count, h_shift, 0), false, dev)?;
        let (vc, vs) = self.rope.chunk(
            (rope_t, rope_h, rope_w),
            (base_video_shift + addref_count, 0, 0),
            false,
            dev,
        )?;
        cos_list.push(rc);
        cos_list.push(vc);
        sin_list.push(rs);
        sin_list.push(vs);

        // pose tokens (W-shifted, freq avg-pool downsampled).
        tok_list.push(pose_emb);
        let (pc, psn) = self.rope.chunk(
            (rope_t, rope_h, rope_w),
            (base_video_shift + addref_count, 0, cfg.pose_w_shift),
            true,
            dev,
        )?;
        cos_list.push(pc);
        sin_list.push(psn);

        let tok_refs: Vec<&Tensor> = tok_list.iter().collect();
        let cos_refs: Vec<&Tensor> = cos_list.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_list.iter().collect();
        let tokens = Tensor::cat(&tok_refs, 0)?; // [L_total, dim]
        let l_total = tokens.dim(0)?;
        let tokens = tokens.reshape((1, l_total, dim))?;
        let cos = Tensor::cat(&cos_refs, 0)?; // [L_total, half_d]
        let sin = Tensor::cat(&sin_refs, 0)?;

        // --- time / text / image conditioning ---
        let (e, e0) = self.time_embed(inp.t, dev)?;
        let text_ctx = self.embed_text(inp.context)?;
        let img_ctx = self.embed_img(inp.clip_fea)?;

        // --- transformer blocks (f32 activations) ---
        let mut x = tokens;
        for block in &self.blocks {
            x = block.forward(&x, &e0, &text_ctx, &img_ctx, &cos, &sin)?;
        }
        let xh = self.apply_head(&x, &e)?; // [1, L_total, out_dim·∏patch]

        // --- keep only the video tokens, unpatchify back to [16, T, H, W] ---
        let addref_length = addref_count * ref_length;
        let offset = addref_length + ref_length;
        let op = xh.dim(2)?;
        let vid_tok = xh
            .narrow(1, offset, seq_length)?
            .reshape((seq_length, op))?;
        unpatchify(&vid_tok, (rope_t, rope_h, rope_w), cfg.out_dim, ps)
    }
}
