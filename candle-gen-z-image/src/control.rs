//! Z-Image **Fun-ControlNet (strict pose)** provider (sc-5489, epic 5480) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-z-image`'s `ZImageControlTransformer` / `ZImageTurboControl`. The LAST family of
//! the sc-5489 3-family ControlNet port (after Qwen + Kolors). Unlike those (a separate ControlNet
//! model producing per-block residuals), Z-Image is a **VACE-style dual-injection** control:
//! `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` ships a parallel control stack (2 refiner + 15
//! main blocks) that threads a control state seeded from the VAE-encoded pose, each block emitting an
//! `after_proj(c)` hint added (scaled) into the base DiT at fixed places — refiner `[0,1]`, main
//! `[0,2,…,28]`.
//!
//! Built on the **vendored** [`crate::dit::ZImageTransformer2DModel`] (NOT the stock candle-transformers
//! DiT the txt2img pipeline uses): the vendored copy exposes the embed → refiner → main phases + the
//! `pub` block forward, so the wrapper can interleave the base loops with the control stack. With no
//! control context the base forward is bit-identical to stock (the `dit.rs` parity gate); with
//! `control_scale = 0` the hints contribute zero — both reproduce plain Z-Image.
//!
//! **Two modes (sc-8680):**
//!
//! * **Turbo** (`base = false`, the original path): distilled, **no CFG** (Z-Image-Turbo) — single
//!   forward per step, no negative prompt / guidance, the control branch runs once per step (NOT twice,
//!   unlike the Qwen/Kolors true-CFG paths), over the distilled 4-step shift-3.0 schedule. **Byte-unchanged**
//!   by sc-8680 — Z-Image-Turbo is guidance-distilled, so touching it would break the distillation.
//! * **Base** (`base = true`, sc-8680): the **undistilled** `Tongyi-MAI/Z-Image` + the base
//!   `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` overlay — the candle sibling of the MLX base control
//!   variant (`mlx-gen-z-image::model_base_control`, sc-8251). It reuses the identical control transformer
//!   but drives it with the **base** treatment (mirroring [`crate::pipeline::Pipeline::render_base`]):
//!   the static **shift = 6.0** scheduler ([`crate::pipeline::base_scheduler_config`]), a ~**50-step**
//!   default, and **real classifier-free guidance** (`v = v_uncond + guidance·(v_cond − v_uncond)`,
//!   guidance 3–5 default 4, + a negative prompt). Following the MLX base control loop
//!   (`denoise_control_cfg_with_progress`), the constant control context threads through **both** the
//!   cond and uncond forward of the CFG combine — the control residuals inject identically on each pass.
//!
//! The whole control transformer + the control context run at **bf16** (the candle Z-Image native dtype,
//! matching the validated txt2img path): candle requires explicit dtype matching, so rather than the MLX
//! fork's mixed-precision (bf16 base + f32 control branch) the candle port runs uniform bf16 — coherence
//! over fork-bit-parity (the VAE-encode runs f32, then casts the control latents to bf16). The provider is
//! a plain struct driven **directly** by the worker (a bespoke pose stream, like the Qwen/Kolors control
//! providers), NOT gen-core-registered — the registered `z_image_turbo` descriptor stays txt2img-only.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{self as nn, Linear, Module, VarBuilder};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{Image, Progress};
use candle_gen::{CandleError, Result};
use candle_transformers::models::z_image::preprocess::prepare_inputs;
use candle_transformers::models::z_image::sampling::postprocess_image;
use candle_transformers::models::z_image::scheduler::{
    calculate_shift, FlowMatchEulerDiscreteScheduler, SchedulerConfig, BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT,
};
use candle_transformers::models::z_image::text_encoder::{TextEncoderConfig, ZImageTextEncoder};
use candle_transformers::models::z_image::transformer::{
    create_coordinate_grid, patchify, unpatchify, Config as DitConfig,
};
use candle_transformers::models::z_image::vae::{AutoEncoderKL, Encoder as VaeEncoder, VaeConfig};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::dit::{ZImageTransformer2DModel, ZImageTransformerBlock};

/// The control transformer + context run bf16 (Z-Image native, candle txt2img dtype); the VAE encoder
/// runs f32 (the encode path's dtype) and its output is cast to bf16 for the control context.
const DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;

/// VAE spatial downscale (latent = image/8 per side) + DiT patch size — shared with the txt2img pipeline.
const SPATIAL_SCALE: u32 = 8;
const PATCH_SIZE: u32 = 2;
const LATENT_CHANNELS: usize = 16;

/// Channel count of the VAE-encoded control context: 16 control latent + 1 zero mask + 16 zero inpaint
/// (the Fun-Controlnet-Union `control_all_x_embedder`'s 33ch input). Pure-pose control zeroes the mask +
/// inpaint groups.
const CONTROL_IN_DIM: usize = 33;
/// Base `layers` indices the 15 control layers inject into (the fork's `CONTROL_LAYERS_PLACES`).
const CONTROL_LAYERS_PLACES: [usize; 15] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
/// Base `noise_refiner` indices the 2 control refiner blocks inject into.
const CONTROL_REFINER_PLACES: [usize; 2] = [0, 1];

/// Z-Image-Turbo is guidance-distilled to a fixed 4-step schedule (the txt2img default).
const DEFAULT_STEPS: usize = 4;
/// **Base** (non-Turbo) control default steps — the undistilled foundation model wants ~50 (sc-8680).
/// Mirrors [`crate::pipeline::BASE_DEFAULT_STEPS`] and `mlx-gen-z-image::model_base::DEFAULT_STEPS`
/// (sc-8251). Used when a base-mode request omits `steps`.
const BASE_DEFAULT_STEPS: usize = 50;
/// **Base** default classifier-free guidance scale — the card recommends 3.0–5.0; 4.0 matches the
/// reference `ZImagePipeline` example and `mlx-gen-z-image::model_base::DEFAULT_GUIDANCE` (sc-8251).
/// Used when a base-mode request omits `guidance`; ignored in Turbo mode (guidance-distilled).
const BASE_DEFAULT_GUIDANCE: f32 = 4.0;
/// Qwen3 pad token id (`<|endoftext|>`) — matches the txt2img tokenizer config.
const QWEN_PAD_TOKEN_ID: i32 = 151643;
const TOKENIZER_MAX_LEN: usize = 512;

/// Default ControlNet conditioning scale (the strict-pose tier — parity with the Qwen/Kolors slices).
pub const DEFAULT_CONTROL_SCALE: f32 = 1.0;

/// `x + hint·scale` (same dtype — the whole control branch runs bf16, see the module header).
fn add_hint(x: &Tensor, hint: &Tensor, scale: f64) -> Result<Tensor> {
    Ok((x + (hint * scale)?)?)
}

/// The control-stack hint index for base block `i`, or `None` when no control block injects there.
fn hint_index(places: &[usize], i: usize) -> Option<usize> {
    places.iter().position(|&p| p == i)
}

/// A VACE control block: a base [`ZImageTransformerBlock`] (its `forward` runs on the threaded control
/// state) + `after_proj` (every block, the per-block hint) + `before_proj` (block 0 only, the seed
/// projection `before_proj(c) + x_base`).
struct ZImageControlBlock {
    base: ZImageTransformerBlock,
    before_proj: Option<Linear>,
    after_proj: Linear,
}

impl ZImageControlBlock {
    /// Load a control block from the Fun-Controlnet-Union checkpoint under `prefix` (e.g.
    /// `"control_layers.0"`). The base-block keys (`attention.*`, `feed_forward.*`, the four norms,
    /// `adaLN_modulation.0`) map 1:1 onto [`ZImageTransformerBlock::new`] (modulation on); `after_proj`
    /// is on every block, `before_proj` only on block 0.
    fn from_weights(
        cfg: &DitConfig,
        vb: VarBuilder,
        dim: usize,
        has_before_proj: bool,
    ) -> Result<Self> {
        let base = ZImageTransformerBlock::new(cfg, true, vb.clone())?;
        let after_proj = nn::linear(dim, dim, vb.pp("after_proj"))?;
        let before_proj = if has_before_proj {
            Some(nn::linear(dim, dim, vb.pp("before_proj"))?)
        } else {
            None
        };
        Ok(Self {
            base,
            before_proj,
            after_proj,
        })
    }
}

/// Paths to the Z-Image control checkpoints.
pub struct ZImageControlPaths {
    /// The base snapshot dir (`tokenizer/`, `text_encoder/`, `transformer/`, `vae/`) — a
    /// `Tongyi-MAI/Z-Image-Turbo` (Turbo mode) or `Tongyi-MAI/Z-Image` (base mode) tree.
    pub snapshot: PathBuf,
    /// The Fun-Controlnet-Union checkpoint — a single `.safetensors` file or a dir containing it
    /// (`Z-Image-Turbo-Fun-Controlnet-Union-2.1` for Turbo, `Z-Image-Fun-Controlnet-Union-2.1` for base).
    pub control: PathBuf,
    /// Select the **base** (undistilled, full-CFG) treatment (sc-8680): shift-6.0 scheduler,
    /// ~50-step default, and real classifier-free guidance in the control denoise (the candle sibling of
    /// `mlx-gen-z-image::model_base_control`). `false` = the original distilled Turbo path (no CFG,
    /// 4-step shift-3.0), byte-unchanged. The control transformer architecture is identical either way.
    pub base: bool,
}

/// One Z-Image Fun-ControlNet generation request.
///
/// `guidance` + `negative_prompt` are consumed **only in base mode** (sc-8680): the undistilled base
/// runs real classifier-free guidance. In Turbo mode they are ignored (Z-Image-Turbo is
/// guidance-distilled — single cond forward, no negative prompt).
#[derive(Clone)]
pub struct ZImageControlRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// ControlNet conditioning scale on the pose hints.
    pub control_scale: f32,
    /// **Base mode only** (sc-8680): the classifier-free guidance scale. `None` → the base default
    /// (4.0); `Some(1.0)` collapses CFG to a single cond forward (Turbo-equivalent cost). Ignored in
    /// Turbo mode.
    pub guidance: Option<f32>,
    /// **Base mode only** (sc-8680): the negative-prompt text for the uncond CFG branch. `None`/empty →
    /// the unconditional embedding (empty string, still wrapped by the Qwen chat template). Ignored in
    /// Turbo mode.
    pub negative_prompt: Option<String>,
    pub seed: u64,
    pub cancel: CancelFlag,
}

impl Default for ZImageControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: DEFAULT_STEPS,
            control_scale: DEFAULT_CONTROL_SCALE,
            guidance: None,
            negative_prompt: None,
            seed: 0,
            cancel: CancelFlag::default(),
        }
    }
}

/// The VACE control transformer: the vendored base DiT + the Fun-Controlnet-Union control stack
/// (`control_all_x_embedder` + 15 `control_layers` + 2 `control_noise_refiner`).
struct ZImageControlTransformer {
    base: ZImageTransformer2DModel,
    control_x_embedder: Linear,
    control_layers: Vec<ZImageControlBlock>,
    control_noise_refiner: Vec<ZImageControlBlock>,
}

impl ZImageControlTransformer {
    /// Build from an already-loaded base transformer + the Fun-Controlnet-Union checkpoint VarBuilder.
    fn from_weights(
        base: ZImageTransformer2DModel,
        cfg: &DitConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let key = format!("{}-{}", cfg.all_patch_size[0], cfg.all_f_patch_size[0]);
        let control_in = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * CONTROL_IN_DIM;
        let control_x_embedder =
            nn::linear(control_in, dim, vb.pp("control_all_x_embedder").pp(key))?;

        let control_layers = (0..CONTROL_LAYERS_PLACES.len())
            .map(|i| {
                ZImageControlBlock::from_weights(cfg, vb.pp("control_layers").pp(i), dim, i == 0)
            })
            .collect::<Result<Vec<_>>>()?;
        let control_noise_refiner = (0..CONTROL_REFINER_PLACES.len())
            .map(|i| {
                ZImageControlBlock::from_weights(
                    cfg,
                    vb.pp("control_noise_refiner").pp(i),
                    dim,
                    i == 0,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            base,
            control_x_embedder,
            control_layers,
            control_noise_refiner,
        })
    }

    /// Run a parallel control stack, returning `(per-block hints, threaded control state)`. Block 0
    /// seeds the branch via `before_proj(c) + x_base`; each block runs the base-block forward on the
    /// threaded state `c` and emits `after_proj(c)` as its hint.
    #[allow(clippy::too_many_arguments)]
    fn run_control_blocks(
        &self,
        blocks: &[ZImageControlBlock],
        c: Tensor,
        x_base: &Tensor,
        attn_mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln: &Tensor,
    ) -> Result<(Vec<Tensor>, Tensor)> {
        let mut c = c;
        let mut hints = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            if i == 0 {
                let bp = block.before_proj.as_ref().ok_or_else(|| {
                    CandleError::Msg("z-image control block 0 is missing before_proj".into())
                })?;
                c = (bp.forward(&c)? + x_base)?;
            }
            c = block
                .base
                .forward(&c, Some(attn_mask), cos, sin, Some(adaln))?;
            hints.push(block.after_proj.forward(&c)?);
        }
        Ok((hints, c))
    }

    /// Dual-injection control forward — re-walks the base DiT's embed → refiner → main phases (the
    /// vendored [`ZImageTransformer2DModel`] internals) while interleaving the parallel control stack
    /// and adding its scaled hints. Returns the **raw** velocity `(B, C, F, H, W)` (the caller negates,
    /// the Z-Image sign convention). `control_context`: the `(B, 33, F, H/8, W/8)` VAE-encoded control;
    /// `scale`: `control_scale`.
    fn forward_control(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
        control_context: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        let base = &self.base;
        let cfg = &base.cfg;
        let device = x.device();
        let (b, _c, f, h, w) = x.dims5()?;
        let patch_size = cfg.all_patch_size[0];
        let f_patch_size = cfg.all_f_patch_size[0];

        // 1. Timestep embedding.
        let t_scaled = (t * cfg.t_scale)?;
        let adaln = base.t_embedder.forward(&t_scaled)?;

        // 2. Patchify + embed the image latent.
        let (x_patches, orig_size) = patchify(x, patch_size, f_patch_size)?;
        let mut x_emb = x_patches.apply(&base.x_embedder)?;
        let img_seq_len = x_emb.dim(1)?;

        // 3. Image position ids (offset past the caption block) + RoPE + an all-valid image mask.
        let f_tokens = f / f_patch_size;
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let text_len = cap_feats.dim(1)?;
        let x_pos_ids =
            create_coordinate_grid((f_tokens, h_tokens, w_tokens), (text_len + 1, 0, 0), device)?;
        let (x_cos, x_sin) = base.rope_embedder.forward(&x_pos_ids)?;
        let x_attn_mask = Tensor::ones((b, img_seq_len), DType::U8, device)?;

        // 4. Embed the control context (same patchify geometry as the image → aligns 1:1).
        let (c_patches, _) = patchify(control_context, patch_size, f_patch_size)?;
        let c_emb = c_patches.apply(&self.control_x_embedder)?;

        // 5. Control refiner: seed + thread through the 2 control refiner blocks (image-length stage).
        let (refiner_hints, threaded) = self.run_control_blocks(
            &self.control_noise_refiner,
            c_emb,
            &x_emb,
            &x_attn_mask,
            &x_cos,
            &x_sin,
            &adaln,
        )?;

        // 6. Base noise refiner, injecting the control refiner hints.
        for (i, layer) in base.noise_refiner.iter().enumerate() {
            x_emb = layer.forward(&x_emb, Some(&x_attn_mask), &x_cos, &x_sin, Some(&adaln))?;
            if let Some(n) = hint_index(&CONTROL_REFINER_PLACES, i) {
                x_emb = add_hint(&x_emb, &refiner_hints[n], scale)?;
            }
        }

        // 7. Caption stream: RMSNorm → linear → context refiner.
        let cap_normed = base.cap_embedder_norm.forward_diff(cap_feats)?;
        let mut cap_emb = cap_normed.apply(&base.cap_embedder_linear)?;
        let cap_pos_ids = create_coordinate_grid((text_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = base.rope_embedder.forward(&cap_pos_ids)?;
        let cap_attn_mask = cap_mask.to_dtype(DType::U8)?;
        for layer in &base.context_refiner {
            cap_emb = layer.forward(&cap_emb, Some(&cap_attn_mask), &cap_cos, &cap_sin, None)?;
        }

        // 8. Unify [image, caption].
        let mut unified = Tensor::cat(&[&x_emb, &cap_emb], 1)?;
        let unified_pos_ids = Tensor::cat(&[&x_pos_ids, &cap_pos_ids], 0)?;
        let (unified_cos, unified_sin) = base.rope_embedder.forward(&unified_pos_ids)?;
        let unified_attn_mask = Tensor::cat(&[&x_attn_mask, &cap_attn_mask], 1)?;

        // 9. Main control pass: thread the (refined) control state + caption through the 15 control
        // layers → the per-block hints for the unified main loop.
        let control_unified = Tensor::cat(&[&threaded, &cap_emb], 1)?;
        let (main_hints, _) = self.run_control_blocks(
            &self.control_layers,
            control_unified,
            &unified,
            &unified_attn_mask,
            &unified_cos,
            &unified_sin,
            &adaln,
        )?;

        // 10. Base main layers, injecting the control hints at CONTROL_LAYERS_PLACES.
        for (i, layer) in base.layers.iter().enumerate() {
            unified = layer.forward(
                &unified,
                Some(&unified_attn_mask),
                &unified_cos,
                &unified_sin,
                Some(&adaln),
            )?;
            if let Some(n) = hint_index(&CONTROL_LAYERS_PLACES, i) {
                unified = add_hint(&unified, &main_hints[n], scale)?;
            }
        }

        // 11. Head: image tokens → final AdaLN layer → unpatchify to the raw velocity.
        let x_out = unified.narrow(1, 0, img_seq_len)?;
        let x_out = base.final_layer.forward(&x_out, &adaln)?;
        Ok(unpatchify(
            &x_out,
            orig_size,
            patch_size,
            f_patch_size,
            cfg.in_channels,
        )?)
    }
}

/// Loaded Z-Image Fun-ControlNet model: the Qwen3 tokenizer + text encoder, the VACE control
/// transformer (vendored base DiT + control stack), the decode VAE, and the VAE encoder (to encode the
/// pose skeleton into the control context).
pub struct ZImageControl {
    root: PathBuf,
    device: Device,
    /// Base (undistilled, full-CFG) vs Turbo (distilled, no-CFG) treatment (sc-8680). Selected at load
    /// from [`ZImageControlPaths::base`]; drives the scheduler (shift 6.0 vs 3.0), the default step
    /// count, and whether the denoise runs real CFG.
    base: bool,
    text_encoder: ZImageTextEncoder,
    transformer: ZImageControlTransformer,
    vae: AutoEncoderKL,
    vae_encoder: VaeEncoder,
    vae_shift: f64,
    vae_scale: f64,
}

impl ZImageControl {
    /// Load the base Z-Image components (Qwen3 encoder + vendored DiT + VAE) + the Fun-Controlnet-Union
    /// control overlay + a VAE encoder for the pose. The control transformer runs bf16; the VAE encoder
    /// runs f32.
    pub fn load(paths: &ZImageControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.snapshot.clone();

        let text_encoder = ZImageTextEncoder::new(
            &TextEncoderConfig::z_image(),
            component_vb(&root, "text_encoder", DTYPE, &device)?,
        )?;

        let dit_cfg = DitConfig::z_image_turbo();
        let base = ZImageTransformer2DModel::new(
            &dit_cfg,
            component_vb(&root, "transformer", DTYPE, &device)?,
        )?;
        let control_file = resolve_control_file(&paths.control)?;
        // SAFETY: mmap of a read-only weight file.
        let control_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[control_file], DTYPE, &device)? };
        let transformer = ZImageControlTransformer::from_weights(base, &dit_cfg, control_vb)?;

        let vae_cfg = VaeConfig::z_image();
        let vae = AutoEncoderKL::new(&vae_cfg, component_vb(&root, "vae", DTYPE, &device)?)?;
        let vae_encoder = VaeEncoder::new(
            &vae_cfg,
            component_vb(&root, "vae", ENC_DTYPE, &device)?.pp("encoder"),
        )?;

        Ok(Self {
            root,
            device,
            base: paths.base,
            text_encoder,
            transformer,
            vae,
            vae_encoder,
            vae_shift: vae_cfg.shift_factor,
            vae_scale: vae_cfg.scaling_factor,
        })
    }

    /// Strict-pose generation: condition the Z-Image generation on `skeleton` (a rendered OpenPose /
    /// canny / depth image at the request size) via the Fun-ControlNet. The worker renders the control
    /// image; this VAE-encodes it into the 33ch control context once, then runs the dual-injection
    /// denoise. Dispatches on the load-time [`base`](ZImageControl::base) flag (sc-8680):
    ///
    /// * **Turbo** ([`generate_turbo`](Self::generate_turbo)): distilled 4-step shift-3.0 schedule, no
    ///   CFG — byte-unchanged from the original path.
    /// * **Base** ([`generate_base`](Self::generate_base)): undistilled shift-6.0 ~50-step schedule with
    ///   real classifier-free guidance (the candle sibling of `mlx-gen-z-image::model_base_control`).
    pub fn generate(
        &self,
        req: &ZImageControlRequest,
        skeleton: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if self.base {
            self.generate_base(req, skeleton, on_progress)
        } else {
            self.generate_turbo(req, skeleton, on_progress)
        }
    }

    /// The original **Turbo** (distilled, no-CFG) control denoise — condition Z-Image-Turbo on the
    /// control image via the Fun-ControlNet, denoising with the distilled flow-match Euler schedule
    /// (4-step shift-3.0, single cond forward per step). **Byte-unchanged by sc-8680** — Z-Image-Turbo is
    /// guidance-distilled, so this stays exactly as validated (`req.guidance`/`req.negative_prompt` are
    /// ignored here).
    fn generate_turbo(
        &self,
        req: &ZImageControlRequest,
        skeleton: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let steps = req.steps.max(1);
        let total = steps as u32;
        let lat_h = (req.height / SPATIAL_SCALE) as usize;
        let lat_w = (req.width / SPATIAL_SCALE) as usize;

        // Text embeddings (bf16, like the txt2img path).
        let cap = self.text_embeddings(&req.prompt)?;

        // VAE-encode the pose → the 33ch control context, constant across steps. Built once.
        let control_context = self.encode_control_context(skeleton, req.width, req.height)?;

        // Deterministic, launch-portable initial noise (sc-3673), CPU RNG → device.
        let noise = self.seed_noise(req.seed, lat_h, lat_w)?;

        // Flow-match Euler schedule (the txt2img construction: Some(mu), no static shift).
        let image_seq_len = ((lat_h as u32 / PATCH_SIZE) * (lat_w as u32 / PATCH_SIZE)) as usize;
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        scheduler.set_timesteps(steps, Some(mu));

        // prepare_inputs pads cap_feats (+ mask) and adds the frame axis → latents (1,16,1,lat_h,lat_w).
        let prepared = prepare_inputs(&noise, std::slice::from_ref(&cap), &self.device)?;
        let cap_feats = prepared.cap_feats;
        let cap_mask = prepared.cap_mask;
        let mut latents = prepared.latents;
        let scale = req.control_scale as f64;

        for step_i in 0..steps {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let t_norm = scheduler.current_timestep_normalized();
            let t = Tensor::from_vec(vec![t_norm as f32], (1,), &self.device)?;
            // Dual-injection control forward; the velocity is negated (Z-Image sign convention).
            let velocity = self
                .transformer
                .forward_control(&latents, &t, &cap_feats, &cap_mask, &control_context, scale)?
                .neg()?;
            latents = scheduler.step(&velocity, &latents)?;
            on_progress(Progress::Step {
                current: step_i as u32 + 1,
                total,
            });
        }

        on_progress(Progress::Decoding);
        self.decode(&latents)
    }

    /// The **base** (undistilled, full-CFG) control denoise (sc-8680) — the candle sibling of
    /// `mlx-gen-z-image::model_base_control` / `pipeline::denoise_control_cfg_with_progress`. It reuses
    /// the identical dual-injection control transformer but mirrors [`crate::pipeline::render_base`]:
    ///
    /// * **Static shift = 6.0** schedule ([`crate::pipeline::base_scheduler_config`], `set_timesteps(N,
    ///   None)`), default ~**50 steps** ([`BASE_DEFAULT_STEPS`]). We feed that σ table to
    ///   [`run_flow_sampler`](candle_gen::run_flow_sampler) with [`TimestepConvention::OneMinusSigma`],
    ///   which derives the DiT timestep `t = 1−σ` from the σ schedule itself — structurally free of the
    ///   Turbo `Some(mu)`-vs-`None` timesteps-desync speckle (we never read `current_timestep_normalized`).
    /// * **Real CFG**: each step runs the control DiT **twice** — once on the prompt conditioning and
    ///   once on the negative/uncond conditioning — both threading the **same** constant control context +
    ///   scale (control residuals inject identically on each pass, per the MLX base control loop), and
    ///   combines `v = v_uncond + guidance·(v_cond − v_uncond)`. `guidance == 1.0` (or no CFG) collapses
    ///   to a single cond forward, byte-identical cost to a Turbo step.
    fn generate_base(
        &self,
        req: &ZImageControlRequest,
        skeleton: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let steps = if req.steps == 0 {
            BASE_DEFAULT_STEPS
        } else {
            req.steps
        };
        let lat_h = (req.height / SPATIAL_SCALE) as usize;
        let lat_w = (req.width / SPATIAL_SCALE) as usize;
        let scale = req.control_scale as f64;

        // Real CFG: default 4.0 (the base card); 1.0 turns CFG off (single cond forward).
        let guidance = req.guidance.unwrap_or(BASE_DEFAULT_GUIDANCE);
        let cfg_on = guidance != 1.0;

        // Text embeddings (bf16). The uncond branch (negative prompt, empty when unset — the
        // unconditional embedding) is only encoded when CFG is active.
        let cap = self.text_embeddings(&req.prompt)?;
        let neg_cap = if cfg_on {
            let neg = req.negative_prompt.as_deref().unwrap_or("");
            Some(self.uncond_embeddings(neg)?)
        } else {
            None
        };

        // VAE-encode the control image → the constant 33ch control context (shared by BOTH CFG branches).
        let control_context = self.encode_control_context(skeleton, req.width, req.height)?;

        // Deterministic, launch-portable initial noise (sc-3673).
        let noise = self.seed_noise(req.seed, lat_h, lat_w)?;

        // Static shift=6.0 schedule (the base model's scheduler_config.json). Unlike the Turbo path's
        // `Some(mu)` no-op, `None` fires the static-shift branch; `run_flow_sampler`'s `OneMinusSigma`
        // reads t = 1−σ from these σ directly, so there is no timesteps desync to guard against.
        let mut scheduler =
            FlowMatchEulerDiscreteScheduler::new(crate::pipeline::base_scheduler_config());
        scheduler.set_timesteps(steps, None);
        let sigmas: Vec<f32> = scheduler.sigmas.iter().map(|&s| s as f32).collect();

        // prepare_inputs pads cap_feats (+ mask) and adds the frame axis for the cond and (when CFG is
        // on) the uncond branch.
        let prepared = prepare_inputs(&noise, std::slice::from_ref(&cap), &self.device)?;
        let cap_feats = prepared.cap_feats;
        let cap_mask = prepared.cap_mask;
        let uncond = match neg_cap.as_ref() {
            Some(neg) => {
                let p = prepare_inputs(&noise, std::slice::from_ref(neg), &self.device)?;
                Some((p.cap_feats, p.cap_mask))
            }
            None => None,
        };

        let latents = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::OneMinusSigma,
            &sigmas,
            prepared.latents,
            req.seed,
            &req.cancel,
            on_progress,
            |latents, t| -> Result<Tensor> {
                let t_tensor = Tensor::from_vec(vec![t], (1,), &self.device)?;
                // Conditional velocity (Z-Image sign convention: the DiT output is negated before the
                // flow-match step). The control context + scale thread through this forward.
                let v_cond = self
                    .transformer
                    .forward_control(
                        latents,
                        &t_tensor,
                        &cap_feats,
                        &cap_mask,
                        &control_context,
                        scale,
                    )?
                    .neg()?;
                let velocity = match uncond.as_ref() {
                    Some((neg_feats, neg_mask)) => {
                        // The uncond branch threads the SAME constant control context + scale (residuals
                        // inject identically on both passes — the MLX base control loop's behaviour).
                        let v_uncond = self
                            .transformer
                            .forward_control(
                                latents,
                                &t_tensor,
                                neg_feats,
                                neg_mask,
                                &control_context,
                                scale,
                            )?
                            .neg()?;
                        // v = v_uncond + guidance·(v_cond − v_uncond). Combining the negated velocities is
                        // linear, so it equals combining-then-negating.
                        let delta = (&v_cond - &v_uncond)?;
                        (v_uncond + (delta * guidance as f64)?)?
                    }
                    None => v_cond,
                };
                Ok(velocity)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode(&latents)
    }

    /// Deterministic, launch-portable initial latent noise (sc-3673): N(0,1) from a fixed-algorithm CPU
    /// RNG seeded by `seed`, built on CPU then moved to the device at the model dtype. Shared by the
    /// Turbo + base control loops so both are pure functions of `(seed, request)`.
    fn seed_noise(&self, seed: u64, lat_h: usize, lat_w: usize) -> Result<Tensor> {
        let n = LATENT_CHANNELS * lat_h * lat_w;
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        Ok(
            Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
                .to_device(&self.device)?
                .to_dtype(DTYPE)?,
        )
    }

    /// Build the Z-Image Qwen tokenizer (chat template + max-length policy). Shared by the conditional
    /// ([`text_embeddings`](Self::text_embeddings)) and unconditional
    /// ([`uncond_embeddings`](Self::uncond_embeddings)) encode paths.
    fn tokenizer(&self) -> Result<TextTokenizer> {
        TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: TOKENIZER_MAX_LEN,
                pad_token_id: QWEN_PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenInstruct,
                pad_to_max_length: false,
            },
        )
        .map_err(|e| CandleError::Msg(format!("z-image control: load tokenizer: {e}")))
    }

    /// Token `ids` → `cap_feats` `(seq, 2560)` at bf16 via the Qwen3 encoder.
    fn encode_cap(&self, ids: &[i32]) -> Result<Tensor> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let enc = self.text_encoder.forward(&input_ids)?; // (1, L, 2560)
        Ok(enc.squeeze(0)?.to_dtype(DTYPE)?)
    }

    /// Prompt → `cap_feats` `(seq, 2560)` at bf16 via the Qwen3 encoder + the Qwen chat template (the
    /// txt2img path's tokenizer config).
    fn text_embeddings(&self, prompt: &str) -> Result<Tensor> {
        let out = self
            .tokenizer()?
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("z-image control: tokenize: {e}")))?;
        if out.ids.is_empty() {
            return Err(CandleError::Msg("z-image control: empty prompt".into()));
        }
        self.encode_cap(&out.ids)
    }

    /// Negative prompt → `cap_feats` for the **unconditional** CFG branch of the base control path
    /// (sc-8680). Identical encoding to [`text_embeddings`](Self::text_embeddings), but the negative
    /// prompt may be the **empty string** (the unconditional embedding).
    ///
    /// gen-core's [`TextTokenizer::tokenize`] short-circuits an empty prompt to a `(1, 0)` sequence
    /// **before** the chat template is applied (`pad_to_max_length = false`), so an empty negative
    /// prompt must be encoded via [`encode_chat_ids`], which renders the QwenInstruct scaffolding
    /// around `""` and tokenizes it to the non-empty role-marker sequence — matching the reference
    /// `mlx-gen-z-image::model_base_control` uncond branch and mirroring
    /// [`crate::pipeline::Pipeline::uncond_embeddings`]. See sc-8646.
    fn uncond_embeddings(&self, negative_prompt: &str) -> Result<Tensor> {
        if !negative_prompt.is_empty() {
            return self.text_embeddings(negative_prompt);
        }
        let ids = self
            .tokenizer()?
            .encode_chat_ids("", true)
            .map_err(|e| CandleError::Msg(format!("z-image control: tokenize uncond: {e}")))?;
        if ids.is_empty() {
            return Err(CandleError::Msg(
                "z-image control: unconditional embedding tokenized to an empty sequence".into(),
            ));
        }
        self.encode_cap(&ids)
    }

    /// Build the 33ch VAE-encoded control context `(1, 33, 1, H/8, W/8)` (bf16): VAE-encode the pose to
    /// 16ch latents + a zero mask (1ch) + a zero inpaint latent (16ch) — the Fun-Controlnet-Union
    /// channel layout. Pure-pose control has no init/mask, so those groups are zeros.
    fn encode_control_context(&self, skeleton: &Image, width: u32, height: u32) -> Result<Tensor> {
        let img = preprocess_control_image(skeleton, width, height, &self.device)?; // f32 (1,3,H,W) [-1,1]
        let moments = img.apply(&self.vae_encoder)?; // (1, 32, H/8, W/8)
        let mean = moments.chunk(2, 1)?[0].clone(); // (1, 16, H/8, W/8)
        let control_latents = ((mean - self.vae_shift)? * self.vae_scale)?;
        let (b, c, lh, lw) = control_latents.dims4()?;
        // Add the singleton frame axis → (1, 16, 1, H/8, W/8).
        let control_latents = control_latents.reshape((b, c, 1, lh, lw))?;
        let mask = Tensor::zeros((b, 1, 1, lh, lw), ENC_DTYPE, &self.device)?;
        let inpaint = Tensor::zeros((b, c, 1, lh, lw), ENC_DTYPE, &self.device)?;
        let context = Tensor::cat(&[&control_latents, &mask, &inpaint], 1)?; // (1, 33, 1, H/8, W/8)
        Ok(context.to_dtype(DTYPE)?)
    }

    /// VAE-decode the final latents `(1, 16, 1, h, w)` → an RGB8 [`Image`] (the txt2img decode).
    fn decode(&self, latents: &Tensor) -> Result<Image> {
        let latents = latents.squeeze(2)?; // (1, 16, h, w)
        let decoded = self.vae.decode(&latents)?.to_dtype(DType::F32)?; // (1, 3, H, W) in [-1,1]
        let img = postprocess_image(&decoded)?.i(0)?.to_device(&Device::Cpu)?;
        let (c, h, w) = img.dims3()?;
        if c != 3 {
            return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
        }
        let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        Ok(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        })
    }
}

/// A pre-rendered RGB8 control image (the OpenPose skeleton, at the request size) → `[1, 3, H, W]` f32
/// in `[-1, 1]` (the VAE encoder's input range). Requires `image` already at `width × height` (the
/// worker renders at the target size — no silent stretch).
fn preprocess_control_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    if image.width != width || image.height != height {
        return Err(CandleError::Msg(format!(
            "z-image control: control image {}x{} must match the request {width}x{height}",
            image.width, image.height
        )));
    }
    let (w, h) = (width as usize, height as usize);
    if image.pixels.len() != w * h * 3 {
        return Err(CandleError::Msg(format!(
            "z-image control: control image buffer {} != {w}x{h}x3",
            image.pixels.len()
        )));
    }
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                data[c * h * w + y * w + x] =
                    image.pixels[(y * w + x) * 3 + c] as f32 / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, h, w), device)?)
}

/// Deterministic overlay-file resolution (sc-8680): pick the intended Fun-Controlnet-**Union** weight
/// **file** from a dir-or-file path, and NOT a Tile / `-lite` sibling.
///
/// A direct `File` path is used verbatim. For a `Dir`, the base + Turbo Fun-Controlnet-Union repos each
/// ship **multiple** `.safetensors` in one snapshot (e.g. the base repo ships `…-Tile-2.1.safetensors`,
/// `…-Tile-2.1-lite.safetensors`, `…-Union-2.1-lite.safetensors`, `…-Union-2.1.safetensors`), so a plain
/// alphabetical "first file" grabs the wrong overlay (`Tile-…-lite` sorts first). We score candidates so
/// the **full Union** checkpoint wins deterministically:
///
/// * exact well-known names first (base/Turbo Union-2.1 + the diffusers default),
/// * else prefer files whose stem contains `union` over those that do not,
/// * penalise `tile` and `lite` variants,
/// * ties broken by the sorted path (stable/deterministic).
///
/// A dir with **no** Union-ish file still resolves the best-scoring `.safetensors` (a single-file control
/// checkpoint dir), so a hand-placed non-standard name is not rejected.
fn resolve_control_file(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    // Exact well-known filenames (base + Turbo Union-2.1, then the diffusers default) — an unambiguous fast
    // path that also pins the base repo's `Z-Image-Fun-Controlnet-Union-2.1.safetensors` (sc-8680).
    for name in [
        "Z-Image-Fun-Controlnet-Union-2.1.safetensors",
        "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
        "diffusion_pytorch_model.safetensors",
    ] {
        let p = path.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| {
                CandleError::Msg(format!("z-image control: read {}: {e}", path.display()))
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort(); // deterministic tie-break
                      // Prefer the full Union checkpoint over Tile / `-lite` siblings (sc-8680).
        if let Some(f) = files.iter().max_by_key(|p| control_file_score(p)).cloned() {
            return Ok(f);
        }
    }
    Err(CandleError::Msg(format!(
        "z-image control: Fun-Controlnet-Union weights not found under {} (expected a \
         .safetensors file or a dir containing one)",
        path.display()
    )))
}

/// Preference score for a candidate control `.safetensors` (sc-8680): higher = more preferred. Prefers a
/// `union` stem, penalises `tile` and `lite` variants, so the full Fun-Controlnet-Union checkpoint is
/// selected over the Tile-lite siblings the base repo ships alongside it. Pure (stem string only) so the
/// resolution policy is unit-testable without real files.
fn control_file_score(path: &Path) -> i32 {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut score = 0;
    if stem.contains("union") {
        score += 10;
    }
    if stem.contains("tile") {
        score -= 10;
    }
    if stem.contains("lite") {
        score -= 5;
    }
    score
}

/// mmap a [`VarBuilder`] over every `.safetensors` in `root/sub` at `dtype` (the txt2img loader).
fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "z-image control: snapshot is missing the {sub}/ dir (at {})",
            root.display()
        )));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| CandleError::Msg(format!("z-image control: read {sub}/: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "z-image control: no .safetensors in {sub}/ (at {})",
            dir.display()
        )));
    }
    // SAFETY: mmap of read-only weight files; standard candle loading path.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = ZImageControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 4);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert_eq!(DEFAULT_CONTROL_SCALE, 1.0);
        // Base-mode CFG fields default to None (unset → base default 4.0 / unconditional in base mode;
        // ignored in Turbo mode).
        assert!(r.guidance.is_none());
        assert!(r.negative_prompt.is_none());
        assert!(!r.cancel.is_cancelled());
    }

    /// Base-mode constants (sc-8680) mirror the base txt2img pipeline + the mlx base control provider
    /// (sc-8251): ~50-step default + CFG default 4.0, distinct from the Turbo distilled 4-step CFG-free
    /// path. The base scheduler config lifts the static shift 3.0 → 6.0. GPU-free.
    #[test]
    fn base_mode_constants_and_scheduler() {
        // Base vs Turbo defaults.
        assert_eq!(BASE_DEFAULT_STEPS, 50);
        assert_eq!(BASE_DEFAULT_GUIDANCE, 4.0);
        assert_eq!(DEFAULT_STEPS, 4);
        // The base control path drives shift=6.0 (reused from the base txt2img pipeline), Turbo 3.0.
        let base_cfg = crate::pipeline::base_scheduler_config();
        let turbo_cfg = SchedulerConfig::z_image_turbo();
        assert_eq!(base_cfg.shift, 6.0);
        assert_eq!(turbo_cfg.shift, 3.0);
        assert!(!base_cfg.use_dynamic_shifting);

        // Base mode builds a shift-6.0 σ table (set_timesteps(N, None) → the static-shift branch fires):
        // N+1 sigmas, start 1.0, strictly decreasing to 0, and lifted above BOTH the linear ramp and the
        // Turbo shift-3.0 table. This is what feeds the OneMinusSigma DiT timestep in `generate_base`.
        let steps = BASE_DEFAULT_STEPS;
        let mut base = FlowMatchEulerDiscreteScheduler::new(base_cfg);
        base.set_timesteps(steps, None);
        let mut turbo = FlowMatchEulerDiscreteScheduler::new(turbo_cfg);
        turbo.set_timesteps(steps, None);
        assert_eq!(base.sigmas.len(), steps + 1);
        assert!((base.sigmas[0] - 1.0).abs() < 1e-9);
        assert!(base.sigmas[steps].abs() < 1e-9);
        for w in base.sigmas.windows(2) {
            assert!(w[0] > w[1], "base sigmas must strictly decrease");
        }
        for i in 1..steps {
            let linear = 1.0 - (i as f64) / (steps as f64);
            assert!(base.sigmas[i] > linear + 1e-9, "shift lifts σ above linear");
            assert!(
                base.sigmas[i] > turbo.sigmas[i] + 1e-9,
                "base shift 6.0 σ must exceed Turbo shift 3.0 σ at step {i}"
            );
        }
    }

    /// The base-mode step-count / guidance resolution logic (the head of `generate_base`, GPU-free):
    /// `steps == 0` → the 50-step default (else the explicit request); `guidance` unset → 4.0, `Some(1.0)`
    /// turns CFG off (single cond forward), any other value turns it on.
    #[test]
    fn base_mode_step_and_guidance_resolution() {
        // Step default: 0 → BASE_DEFAULT_STEPS; explicit honored.
        let resolve_steps = |s: usize| if s == 0 { BASE_DEFAULT_STEPS } else { s };
        assert_eq!(resolve_steps(0), 50);
        assert_eq!(resolve_steps(28), 28);
        // Guidance default + CFG-on gate.
        let resolve_g = |g: Option<f32>| g.unwrap_or(BASE_DEFAULT_GUIDANCE);
        assert_eq!(resolve_g(None), 4.0);
        assert_eq!(resolve_g(Some(3.5)), 3.5);
        assert!(resolve_g(None) != 1.0, "default CFG is on");
        assert!(resolve_g(Some(1.0)) == 1.0, "guidance 1.0 turns CFG off");
    }

    /// The VACE injection geometry: 15 main control layers at base places [0,2,…,28] + 2 refiner blocks
    /// at [0,1] (the Fun-Controlnet-Union-2.1 structure).
    #[test]
    fn vace_injection_places() {
        assert_eq!(CONTROL_LAYERS_PLACES.len(), 15);
        assert_eq!(CONTROL_REFINER_PLACES.len(), 2);
        assert_eq!(*CONTROL_LAYERS_PLACES.last().unwrap(), 28);
        assert_eq!(CONTROL_IN_DIM, 33);
        // control_x_embedder in-features = f_patch·patch²·33 = 1·4·33 = 132.
        let cfg = DitConfig::z_image_turbo();
        let control_in = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * CONTROL_IN_DIM;
        assert_eq!(control_in, 132);
        // The base must have enough main layers for the injection places.
        assert!(cfg.n_layers > *CONTROL_LAYERS_PLACES.last().unwrap());
        assert!(cfg.n_refiner_layers >= CONTROL_REFINER_PLACES.len());
    }

    #[test]
    fn control_preprocess_shape_and_range() {
        let img = Image {
            width: 16,
            height: 8,
            pixels: vec![255u8; 16 * 8 * 3],
        };
        let t = preprocess_control_image(&img, 16, 8, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 8, 16]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-4)); // 255 → 1.0
        assert!(preprocess_control_image(&img, 32, 8, &Device::Cpu).is_err());
    }

    /// The exact-name fast path + `File` passthrough (Turbo repo's single-file layout).
    #[test]
    fn control_file_resolution() {
        let dir = std::env::temp_dir().join(format!("zimg_cn_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(resolve_control_file(&dir).is_err());
        let f = dir.join("Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_control_file(&dir).unwrap(), f);
        assert_eq!(resolve_control_file(&f).unwrap(), f);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Deterministic overlay resolution (sc-8680): a **base** Fun-Controlnet-Union snapshot ships the full
    /// Union checkpoint alongside Tile / `-lite` siblings; resolution must pick the intended Union file,
    /// NOT a Tile-lite variant (which sorts first alphabetically → the pre-fix bug that forced a manual
    /// override during validation).
    #[test]
    fn control_file_resolution_prefers_union_over_tile_lite() {
        let dir = std::env::temp_dir().join(format!("zimg_cn_union_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // The four files the real base repo ships (alphabetical order: Tile-lite sorts FIRST).
        let tile_lite = dir.join("Z-Image-Fun-Controlnet-Tile-2.1-lite.safetensors");
        let tile = dir.join("Z-Image-Fun-Controlnet-Tile-2.1.safetensors");
        let union_lite = dir.join("Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors");
        let union = dir.join("Z-Image-Fun-Controlnet-Union-2.1.safetensors");
        for p in [&tile_lite, &tile, &union_lite, &union] {
            std::fs::write(p, b"x").unwrap();
        }
        // Must select the full Union-2.1 file (the exact-name fast path), never a Tile/lite sibling.
        assert_eq!(resolve_control_file(&dir).unwrap(), union);

        // Even with the exact name removed, the scored fallback prefers Union-lite over the Tile files
        // (union stem beats tile, and lite penalty is smaller than the tile penalty).
        std::fs::remove_file(&union).unwrap();
        assert_eq!(resolve_control_file(&dir).unwrap(), union_lite);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The scoring policy directly (GPU-/fs-free): union > union-lite > (no keyword) > tile-lite > tile.
    #[test]
    fn control_file_score_orders_union_first() {
        let s = |n: &str| control_file_score(Path::new(n));
        assert!(s("Z-Image-Fun-Controlnet-Union-2.1.safetensors") > s("plain.safetensors"));
        assert!(
            s("Z-Image-Fun-Controlnet-Union-2.1.safetensors")
                > s("Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors")
        );
        assert!(
            s("Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors")
                > s("Z-Image-Fun-Controlnet-Tile-2.1.safetensors")
        );
        assert!(
            s("Z-Image-Fun-Controlnet-Tile-2.1.safetensors")
                > s("Z-Image-Fun-Controlnet-Tile-2.1-lite.safetensors"),
            "among Tile files, the full one beats the -lite (lite penalty)"
        );
    }
}
