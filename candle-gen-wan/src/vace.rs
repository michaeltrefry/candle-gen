//! The **`WanVACETransformer3DModel`** DiT + the host-side VACE conditioning construction — the candle
//! (Windows/CUDA) sibling of `mlx-gen-wan`'s `vace.rs`. Ported from the diffusers
//! `transformer_wan_vace.py` + `WanVACEPipeline` (Wan2.1-VACE-14B).
//!
//! **VACE is purely additive on the base Wan DiT.** The base block math ([`crate::transformer::Block`]
//! — AdaLN self-attn with 3-axis RoPE + qk-RMSNorm, cross-attn to UMT5, gated-GELU FFN) is unchanged;
//! VACE adds (a) a `vace_patch_embedding` (a 96-ch patchify→linear), (b) `len(vace_layers)`
//! [`VaceBlock`]s that produce per-layer "hints" from the control latent, and (c) hint injection
//! `hidden += proj_out(control)·scale` at each main layer in `vace_layers`. The control latent is the
//! 96-channel `cat([video_latents(32), mask_latents(64)], C)` the host builds from the masked control
//! clip + reference images ([`prepare_video_latents`] / [`prepare_masks`] / [`build_vace_control`]).
//!
//! **Dtypes:** the DiT runs **bf16** (norms / modulation / RoPE upcast to f32 inside [`Block`]) — the
//! candle Wan regime; the z16 VAE runs f32. This mirrors the validated candle base Wan 14B path
//! (`wan2_2_t2v_14b`), not the mlx f32-residual stream (the candle base keeps the inter-block stream in
//! bf16; VACE follows it).

use candle_gen::candle_core::{DType, Device, Error as CoreError, Result, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::gen_core::CancelFlag;
use candle_gen::{CandleError, Result as CResult};

use crate::config::{WanVaceConfig, VAE16_STRIDE_SPATIAL, VAE16_STRIDE_TEMPORAL};
use crate::pipeline::cfg;
use crate::scheduler::{FlowScheduler, Sampler};
use crate::transformer::{linear, ln_no_affine, timestep_sinusoid, Block};
use crate::vae16::WanVae16;

/// The z16 VAE temporal/spatial strides (Wan2.1; VACE is Wan2.1-based).
const VAE_T: usize = VAE16_STRIDE_TEMPORAL as usize;
const VAE_S: usize = VAE16_STRIDE_SPATIAL as usize;

/// Squeeze a diffusers Conv3d patch-embedding weight `[dim, in, pt, ph, pw]` → a per-frame conv2d
/// `[dim, in, ph, pw]` (the patch temporal kernel `pt` is 1 for Wan). Mirrors
/// [`crate::transformer::WanTransformer`]'s patch handling.
fn load_patch_conv(
    vb: &VarBuilder,
    name: &str,
    dim: usize,
    in_c: usize,
    patch: (usize, usize, usize),
) -> Result<(Tensor, Tensor)> {
    let (pt, ph, pw) = patch;
    let w = vb
        .get((dim, in_c, pt, ph, pw), &format!("{name}.weight"))?
        .narrow(2, 0, 1)?
        .squeeze(2)?
        .contiguous()?;
    let b = vb
        .get(dim, &format!("{name}.bias"))?
        .reshape((1, dim, 1, 1))?;
    Ok((w, b))
}

/// A VACE block: a [`Block`] over the **control** stream, plus `proj_in` (block 0 only — injects the
/// main noisy-latent tokens into the control stream once) and `proj_out` (every block — emits the
/// per-layer hint). Diffusers `WanVACETransformerBlock`.
struct VaceBlock {
    proj_in: Option<Linear>,
    core: Block,
    proj_out: Linear,
}

impl VaceBlock {
    fn new(cfg: &WanVaceConfig, vb: VarBuilder, has_proj_in: bool) -> Result<Self> {
        let dim = cfg.base.dim;
        let proj_in = if has_proj_in {
            Some(linear(dim, dim, vb.pp("proj_in"))?)
        } else {
            None
        };
        Ok(Self {
            proj_in,
            core: Block::new(&cfg.base, vb.clone())?,
            proj_out: linear(dim, dim, vb.pp("proj_out"))?,
        })
    }

    /// `control`/`hidden_tokens`: `[B,L,dim]` (bf16). Returns `(hint, new_control)` both bf16: the hint
    /// added to the main stream at the matching vace layer, and the control stream threaded forward.
    /// `proj_in` (block 0 only) injects the main noisy-latent tokens into the control stream once.
    fn forward(
        &self,
        control: &Tensor,
        hidden_tokens: &Tensor,
        temb6: &Tensor,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let control = match &self.proj_in {
            Some(p) => p.forward(control)?.broadcast_add(hidden_tokens)?,
            None => control.clone(),
        };
        let new_control = self.core.forward(&control, temb6, context, cos, sin)?;
        let hint = self.proj_out.forward(&new_control)?;
        Ok((hint, new_control))
    }
}

/// The Wan-VACE DiT (Wan2.1-VACE-14B): the base Wan transformer plus the VACE control path. Reads
/// diffusers tensor names directly (the VACE checkpoint ships diffusers layout). Runs bf16.
pub struct WanVaceTransformer {
    patch_w: Tensor, // patch_embedding (16→dim), conv2d weight
    patch_b: Tensor,
    vace_patch_w: Tensor, // vace_patch_embedding (96→dim)
    vace_patch_b: Tensor,
    text_l1: Linear,
    text_l2: Linear,
    time_l1: Linear,
    time_l2: Linear,
    time_proj: Linear,
    blocks: Vec<Block>,
    vace_blocks: Vec<VaceBlock>,
    scale_shift_table: Tensor, // head [1,2,dim] f32
    proj_out: Linear,          // head proj
    cfg: WanVaceConfig,
    device: Device,
    dtype: DType,
}

impl WanVaceTransformer {
    pub fn new(cfg: &WanVaceConfig, vb: VarBuilder) -> Result<Self> {
        let base = &cfg.base;
        let dim = base.dim;
        let (pt, ph, pw) = base.patch;

        let (patch_w, patch_b) =
            load_patch_conv(&vb, "patch_embedding", dim, base.in_channels, base.patch)?;
        let (vace_patch_w, vace_patch_b) = load_patch_conv(
            &vb,
            "vace_patch_embedding",
            dim,
            cfg.vace_in_channels,
            base.patch,
        )?;

        let ce = vb.pp("condition_embedder");
        let text_l1 = linear(base.text_dim, dim, ce.pp("text_embedder").pp("linear_1"))?;
        let text_l2 = linear(dim, dim, ce.pp("text_embedder").pp("linear_2"))?;
        let time_l1 = linear(base.freq_dim, dim, ce.pp("time_embedder").pp("linear_1"))?;
        let time_l2 = linear(dim, dim, ce.pp("time_embedder").pp("linear_2"))?;
        let time_proj = linear(dim, 6 * dim, ce.pp("time_proj"))?;

        let mut blocks = Vec::with_capacity(base.num_layers);
        for i in 0..base.num_layers {
            blocks.push(Block::new(base, vb.pp("blocks").pp(i))?);
        }

        // VACE blocks: `vace_blocks.0` carries `proj_in` (injects the main tokens into the control
        // stream once); every block carries `proj_out` (emits its per-layer hint).
        let mut vace_blocks = Vec::with_capacity(cfg.vace_layers.len());
        for j in 0..cfg.vace_layers.len() {
            vace_blocks.push(VaceBlock::new(cfg, vb.pp("vace_blocks").pp(j), j == 0)?);
        }

        let proj_out = linear(dim, base.out_channels * pt * ph * pw, vb.pp("proj_out"))?;
        let scale_shift_table = vb
            .get((1, 2, dim), "scale_shift_table")?
            .to_dtype(DType::F32)?;

        Ok(Self {
            patch_w,
            patch_b,
            vace_patch_w,
            vace_patch_b,
            text_l1,
            text_l2,
            time_l1,
            time_l2,
            time_proj,
            blocks,
            vace_blocks,
            scale_shift_table,
            proj_out,
            cfg: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Project UMT5 prompt embeds `[B,S,4096]` → cross-attn context `[B,S,dim]` (bf16, constant across
    /// the denoise loop), `gelu_tanh` between the two linears (PixArtAlphaTextProjection).
    pub fn embed_text(&self, prompt_embeds: &Tensor) -> Result<Tensor> {
        let x = prompt_embeds.to_dtype(self.dtype)?;
        self.text_l2.forward(&self.text_l1.forward(&x)?.gelu()?)
    }

    /// Per-frame strided conv2d patchify of a `[B,C,F,Hl,Wl]` latent → tokens `[B, L, dim]` (bf16).
    fn patchify(&self, latent: &Tensor, w: &Tensor, b: &Tensor, in_c: usize) -> Result<Tensor> {
        let (bb, _c, f, hl, wl) = latent.dims5()?;
        let (_pt, ph, _pw) = self.cfg.base.patch;
        let dim = self.cfg.base.dim;
        let merged = latent
            .permute((0, 2, 1, 3, 4))?
            .reshape((bb * f, in_c, hl, wl))?
            .contiguous()?
            .to_dtype(self.dtype)?;
        let y = merged.conv2d(w, 0, ph, 1, 1)?.broadcast_add(b)?; // [B*F,dim,pph,ppw]
        let pph = hl / ph;
        let ppw = wl / ph;
        y.reshape((bb, f, dim, pph, ppw))?
            .permute((0, 1, 3, 4, 2))? // [B,F,pph,ppw,dim]
            .reshape((bb, f * pph * ppw, dim))?
            .contiguous()
    }

    /// Embed the 96-ch control latent `[1,96,F,Hl,Wl]` via `vace_patch_embedding`, zero-padded to the
    /// main token length `l`. Computed once per generate (constant across denoise steps).
    pub fn embed_control(&self, control: &Tensor, l: usize) -> Result<Tensor> {
        let emb = self.patchify(
            control,
            &self.vace_patch_w,
            &self.vace_patch_b,
            self.cfg.vace_in_channels,
        )?;
        let lc = emb.dim(1)?;
        let dim = self.cfg.base.dim;
        match lc.cmp(&l) {
            std::cmp::Ordering::Less => {
                let pad = Tensor::zeros((emb.dim(0)?, l - lc, dim), self.dtype, &self.device)?;
                Tensor::cat(&[&emb, &pad], 1)
            }
            std::cmp::Ordering::Greater => Err(CoreError::Msg(format!(
                "wan-vace: control token count {lc} exceeds latent token count {l}; the control clip \
                 resolution must match the generation resolution"
            ))),
            std::cmp::Ordering::Equal => Ok(emb),
        }
    }

    /// Time embedding: `temb [B,dim]` (head modulation) + `temb6 [B,6,dim]` (per-block modulation, f32).
    fn time_embed(&self, t: f64, b: usize) -> Result<(Tensor, Tensor)> {
        let dim = self.cfg.base.dim;
        let sinus =
            timestep_sinusoid(t, self.cfg.base.freq_dim, b, &self.device)?.to_dtype(self.dtype)?;
        let temb = self
            .time_l2
            .forward(&self.time_l1.forward(&sinus)?.silu()?)?; // [B,dim]
        let temb6 = self
            .time_proj
            .forward(&temb.silu()?)?
            .reshape((b, 6, dim))?
            .to_dtype(DType::F32)?;
        Ok((temb, temb6))
    }

    /// Output head: modulated non-affine norm + the `proj_out` projection, then unpatchify → velocity
    /// `[B, out_c, F, Hl, Wl]` (f32).
    fn head(&self, hidden: &Tensor, temb: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (b, _l, _dim) = hidden.dims3()?;
        let (ppf, pph, ppw) = grid;
        let (pt, ph, pw) = self.cfg.base.patch;
        let oc = self.cfg.base.out_channels;
        let head_mod = self
            .scale_shift_table
            .broadcast_add(&temb.unsqueeze(1)?.to_dtype(DType::F32)?)?; // [B,2,dim]
        let shift = head_mod.narrow(1, 0, 1)?;
        let scale = head_mod.narrow(1, 1, 1)?;
        let hf = hidden.to_dtype(DType::F32)?;
        let normed = ln_no_affine(&hf, self.cfg.base.eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?
            .to_dtype(self.dtype)?;
        let out = self.proj_out.forward(&normed)?; // [B,L,out_c*patch]
        out.reshape(&[b, ppf, pph, ppw, pt, ph, pw, oc][..])?
            .permute(&[0usize, 7, 1, 4, 2, 5, 3, 6][..])?
            .reshape((b, oc, ppf * pt, pph * ph, ppw * pw))?
            .to_dtype(DType::F32)
    }

    /// VACE forward with a pre-embedded control latent (`embed_control`, computed once). `latents`:
    /// `[1,16,F,Hl,Wl]` (f32). `control_emb`: `[1,L,dim]` (bf16). `context`: `[1,S,dim]` (bf16,
    /// projected). `scales`: one `control_hidden_states_scale` per `vace_layers` entry. Returns the
    /// predicted velocity `[1,16,F,Hl,Wl]` (f32).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        latents: &Tensor,
        t: f64,
        control_emb: &Tensor,
        context: &Tensor,
        scales: &[f32],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        if scales.len() != self.cfg.vace_layers.len() {
            return Err(CoreError::Msg(format!(
                "wan-vace: control scales len {} != vace_layers len {}",
                scales.len(),
                self.cfg.vace_layers.len()
            )));
        }
        let (b, _c, f, hl, wl) = latents.dims5()?;
        let (pt, ph, _pw) = self.cfg.base.patch;
        let grid = (f / pt, hl / ph, wl / ph);

        // Patch-embed the noisy latent → [B,L,dim] (bf16).
        let x_tokens = self.patchify(
            latents,
            &self.patch_w,
            &self.patch_b,
            self.cfg.base.in_channels,
        )?;
        let (temb, temb6) = self.time_embed(t, b)?;

        // VACE hint prep: thread the control stream through every vace block, collect (hint, scale).
        let mut control_hs = control_emb.clone();
        let mut hints: Vec<(Tensor, f32)> = Vec::with_capacity(self.vace_blocks.len());
        for (vb, &scale) in self.vace_blocks.iter().zip(scales.iter()) {
            let (hint, new_control) =
                vb.forward(&control_hs, &x_tokens, &temb6, context, cos, sin)?;
            hints.push((hint, scale));
            control_hs = new_control;
        }
        hints.reverse();

        // Main blocks with hint injection at each layer in vace_layers.
        let mut x = x_tokens;
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(&x, &temb6, context, cos, sin)?;
            if self.cfg.vace_layers.contains(&i) {
                let (hint, scale) = hints
                    .pop()
                    .expect("one hint per vace layer (vace_layers.len() == vace_blocks.len())");
                x = (x + hint.affine(scale as f64, 0.0)?)?;
            }
        }
        self.head(&x, &temb, grid)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ============================================================================================
// VACE conditioning construction (the host / VAE side) — builds the 96-ch control latent.
//
// `control = cat([video_latents(32), mask_latents(64)], C)`. Mirrors diffusers `WanVACEPipeline`'s
// `prepare_video_latents` + `prepare_masks` + the `__call__` concat (single batch). The VAE-encode +
// normalize is the validated [`WanVae16::encode`]; the new pieces are the mask 8×8-unfold + nearest-
// exact temporal resample, the inactive/reactive split, and the reference-frame prepend.
// ============================================================================================

/// Binarize a soft control mask: `where(mask > 0.5, 1.0, 0.0)` (diffusers `prepare_video_latents`).
pub fn binarize_mask(mask: &Tensor) -> Result<Tensor> {
    mask.gt(0.5f64)?.to_dtype(DType::F32)
}

/// Nearest-exact temporal resample along the frame axis (`axis`) → `out_t` frames. torch
/// `mode="nearest-exact"`: `src = floor((i + 0.5)·F / out_t)`, clamped to `[0, F−1]`.
fn nearest_exact_temporal(x: &Tensor, axis: usize, out_t: usize) -> Result<Tensor> {
    let f = x.dim(axis)?;
    let idx: Vec<u32> = (0..out_t)
        .map(|i| {
            let s = (((i as f64) + 0.5) * (f as f64) / (out_t as f64)).floor() as i64;
            s.clamp(0, f as i64 - 1) as u32
        })
        .collect();
    let idx = Tensor::from_vec(idx, out_t, x.device())?;
    x.index_select(&idx, axis)
}

/// `prepare_masks`: a soft control mask `[1,3,F,H,W]` (channel 0 used) → the 64-ch mask latent
/// `[1, 64, new_t (+ num_ref), H/8, W/8]`, where `64 = vae_s²`, `new_t = ⌈F / vae_t⌉`. The mask is
/// unfolded `view(F, new_h, vae_s, new_w, vae_s).permute(2,4,0,1,3).flatten(0,1)`, nearest-exact
/// resampled in time, then `num_ref` zero frames prepended. Pure host op (diffusers
/// `WanVACEPipeline.prepare_masks`, single batch).
pub fn prepare_masks(mask: &Tensor, patch: usize, num_ref: usize) -> Result<Tensor> {
    let (_b, _c, f, h, w) = mask.dims5()?;
    let new_t = f.div_ceil(VAE_T);
    let new_h = h / (VAE_S * patch) * patch;
    let new_w = w / (VAE_S * patch) * patch;
    let dev = mask.device();

    // Channel 0 → [F,H,W].
    let ch0 = mask.narrow(1, 0, 1)?.reshape((f, h, w))?;
    // [F, new_h, vae_s, new_w, vae_s] → permute → [vae_s, vae_s, F, new_h, new_w] → flatten → [vae_s²,...].
    let m = ch0
        .reshape((f, new_h, VAE_S, new_w, VAE_S))?
        .permute((2, 4, 0, 1, 3))?
        .reshape((VAE_S * VAE_S, f, new_h, new_w))?
        .contiguous()?;
    let m = nearest_exact_temporal(&m, 1, new_t)?; // [64, new_t, new_h, new_w]
    let m = if num_ref > 0 {
        let pad = Tensor::zeros((VAE_S * VAE_S, num_ref, new_h, new_w), DType::F32, dev)?;
        Tensor::cat(&[&pad, &m], 1)?
    } else {
        m
    };
    m.unsqueeze(0) // [1, 64, new_t(+num_ref), new_h, new_w]
}

/// `prepare_video_latents`: the control video `[1,3,F,H,W]` (+ binarized mask + reference images) → the
/// 32-ch video-latent `[1, 32, T_lat (+ num_ref), H/8, W/8]`. `inactive = video·(1−mask)`,
/// `reactive = video·mask`, each z16-VAE-encoded + normalized, concatenated along channels. Each
/// reference `[1,3,1,H,W]` is encoded to one latent frame, `cat([ref, zeros])` to 32 ch, and prepended
/// along the frame axis. Mirrors diffusers `WanVACEPipeline.prepare_video_latents` (single batch).
pub fn prepare_video_latents(
    vae: &WanVae16,
    video: &Tensor,
    mask: &Tensor,
    references: &[Tensor],
) -> Result<Tensor> {
    let m = binarize_mask(mask)?; // [1,3,F,H,W]
    let one_minus_m = m.affine(-1.0, 1.0)?; // 1 − mask
    let inactive = vae.encode(&video.broadcast_mul(&one_minus_m)?)?; // [1,16,T_lat,h,w]
    let reactive = vae.encode(&video.broadcast_mul(&m)?)?;
    let mut latents = Tensor::cat(&[&inactive, &reactive], 1)?; // [1,32,T_lat,h,w]

    for reference in references {
        let ref_lat = vae.encode(reference)?; // [1,16,1,h,w]
        let zeros = Tensor::zeros(ref_lat.shape(), ref_lat.dtype(), ref_lat.device())?;
        let ref_lat = Tensor::cat(&[&ref_lat, &zeros], 1)?; // [1,32,1,h,w]
        latents = Tensor::cat(&[&ref_lat, &latents], 2)?; // prepend along frames
    }
    Ok(latents)
}

/// Assemble the 96-ch `control = cat([video_latents(32), mask_latents(64)], C)` (diffusers `__call__`'s
/// `conditioning_latents = cat([conditioning_latents, mask], dim=1)`).
pub fn build_vace_control(video_latents: &Tensor, mask_latents: &Tensor) -> Result<Tensor> {
    Tensor::cat(&[video_latents, mask_latents], 1)
}

/// VACE CFG denoise loop — mirrors the candle base Wan denoise (`FlowScheduler` per step), but each step
/// runs [`WanVaceTransformer::forward_cached`] with the constant 96-ch control + per-vace-layer
/// `scales`, classifier-free-guided against the (optional) unconditional context. The control latent is
/// embedded once (`embed_control`) and reused across every step + both CFG branches.
///
/// `init_noise`: `[1, 16, T, h, w]` (f32). `ctx_pos` / `ctx_neg`: projected contexts (`embed_text`).
#[allow(clippy::too_many_arguments)]
pub fn denoise_vace(
    transformer: &WanVaceTransformer,
    control: &Tensor,
    scales: &[f32],
    sampler: Sampler,
    steps: usize,
    shift: f64,
    guidance: f64,
    ctx_pos: &Tensor,
    ctx_neg: Option<&Tensor>,
    init_noise: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> CResult<Tensor> {
    let (_b, _c, f, hl, wl) = init_noise.dims5()?;
    let (pt, ph, _pw) = transformer.cfg.base.patch;
    let l = (f / pt) * (hl / ph) * (wl / ph);
    let control_emb = transformer.embed_control(control, l)?;

    let mut latents = init_noise.clone();
    let mut sched = FlowScheduler::new(sampler, steps, shift);
    for i in 0..steps {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let t = sched.timestep(i);
        let cond =
            transformer.forward_cached(&latents, t, &control_emb, ctx_pos, scales, cos, sin)?;
        let v = match ctx_neg {
            Some(neg) => {
                let uncond =
                    transformer.forward_cached(&latents, t, &control_emb, neg, scales, cos, sin)?;
                cfg(&cond, &uncond, guidance)?
            }
            None => cond,
        };
        latents = sched.step(&v, &latents)?;
        on_step(i + 1);
    }
    Ok(latents)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::Cpu
    }

    #[test]
    fn binarize_thresholds_at_half() {
        let m = Tensor::from_vec(vec![0.0f32, 0.4, 0.6, 1.0], (1, 1, 1, 2, 2), &dev()).unwrap();
        let b = binarize_mask(&m)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(b, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn nearest_exact_matches_torch_indices() {
        // F=5 → out_t=2: src = floor((i+0.5)*5/2) = floor(1.25)=1, floor(3.75)=3.
        let f = 5usize;
        let x = Tensor::arange(0u32, f as u32, &dev())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((1, f, 1, 1))
            .unwrap();
        let y = nearest_exact_temporal(&x, 1, 2).unwrap();
        let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![1.0, 3.0]);
    }

    #[test]
    fn prepare_masks_unfold_shape() {
        // F=5 (1+4·1) → new_t=2; H=W=32, vae_s=8, patch=2 → new_h=new_w=4; num_ref=1.
        let mask = Tensor::ones((1, 3, 5, 32, 32), DType::F32, &dev()).unwrap();
        let out = prepare_masks(&mask, 2, 1).unwrap();
        // [1, 64, new_t(2)+num_ref(1)=3, 4, 4].
        assert_eq!(out.dims(), &[1, 64, 3, 4, 4]);
        // The prepended reference frame is zero; the rest are the unfolded (all-ones) mask.
        let frame0 = out
            .narrow(2, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(frame0.iter().all(|&v| v == 0.0));
        let frame1 = out
            .narrow(2, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(frame1.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn build_control_concats_channels() {
        let v = Tensor::zeros((1, 32, 2, 4, 4), DType::F32, &dev()).unwrap();
        let m = Tensor::ones((1, 64, 2, 4, 4), DType::F32, &dev()).unwrap();
        let c = build_vace_control(&v, &m).unwrap();
        assert_eq!(c.dims(), &[1, 96, 2, 4, 4]);
    }
}
