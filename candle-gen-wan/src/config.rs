//! Static configuration for the **Wan2.2 TI2V-5B** text-to-video model, read from the diffusers
//! checkpoint (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`): `transformer/config.json` (`WanTransformer3DModel`),
//! `vae/config.json` (`AutoencoderKLWan`), `text_encoder/config.json` (`UMT5EncoderModel`), and
//! `scheduler/scheduler_config.json` (`UniPCMultistepScheduler`, flow-match).

/// Registry id — matches the mlx-gen-wan descriptor so a consumer resolves the same engine across
/// backends.
pub const MODEL_ID: &str = "wan2_2_ti2v_5b";

/// Registry id for the Wan2.2 **T2V-A14B** dual-expert MoE (text→video). Matches the mlx-gen-wan
/// descriptor so a consumer resolves the same engine across backends.
pub const MODEL_ID_T2V_14B: &str = "wan2_2_t2v_14b";
/// Registry id for the Wan2.2 **I2V-A14B** dual-expert MoE (channel-concat image→video).
pub const MODEL_ID_I2V_14B: &str = "wan2_2_i2v_14b";

/// Default denoise steps (diffusers `sample_steps` / the UniPC default for the 5B).
pub const DEFAULT_STEPS: u32 = 40;
/// Default classifier-free guidance scale (`sample_guide_scale`).
pub const DEFAULT_GUIDANCE: f32 = 5.0;
/// Default output frame count. Must satisfy `frames % 4 == 1` (one latent frame + groups of 4).
pub const DEFAULT_FRAMES: u32 = 81;
/// Default playback / muxing cadence (`sample_fps`).
pub const DEFAULT_FPS: u32 = 24;
/// Flow-match time-shift applied to the sigma schedule (`flow_shift`).
pub const FLOW_SHIFT: f64 = 5.0;
/// Diffusion training horizon (`num_train_timesteps`).
pub const NUM_TRAIN_TIMESTEPS: usize = 1000;

/// Wan's default negative prompt (the reference anti-artifact string) used when CFG is on and the
/// request supplies none.
pub const NEGATIVE_FALLBACK: &str =
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，\
     低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，\
     毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

/// Spatial size must be a multiple of `vae_stride_spatial (16) × patch (2) = 32` so the latent
/// (`H/16`) is even for the DiT 2×2 spatial patch.
pub const SIZE_MULTIPLE: u32 = 32;
/// VAE spatial downsample factor (latent `H = height / 16`).
pub const VAE_STRIDE_SPATIAL: u32 = 16;
/// VAE temporal downsample factor (latent `T = (frames - 1) / 4 + 1`).
pub const VAE_STRIDE_TEMPORAL: u32 = 4;

/// `WanTransformer3DModel` dims (TI2V-5B, dense — no MoE).
#[derive(Clone, Copy, Debug)]
pub struct TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// `num_heads × head_dim` = 3072.
    pub dim: usize,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub text_dim: usize,
    /// `(p_t, p_h, p_w)` patch (`(1, 2, 2)`).
    pub patch: (usize, usize, usize),
    pub eps: f64,
    pub rope_theta: f64,
    pub rope_max_seq_len: usize,
}

impl TransformerConfig {
    pub fn ti2v_5b() -> Self {
        Self {
            in_channels: 48,
            out_channels: 48,
            num_layers: 30,
            num_heads: 24,
            head_dim: 128,
            dim: 3072,
            ffn_dim: 14336,
            freq_dim: 256,
            text_dim: 4096,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }

    /// `WanTransformer3DModel` dims for **one A14B expert** (dim 5120, 40 layers, 40 heads, z16 in/out).
    /// Both the `transformer/` (high-noise) and `transformer_2/` (low-noise) experts share these dims;
    /// only the loaded weights differ. From `Wan-AI/Wan2.2-T2V-A14B-Diffusers/transformer/config.json`.
    pub fn t2v_14b() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            num_layers: 40,
            num_heads: 40,
            head_dim: 128,
            dim: 5120,
            ffn_dim: 13824,
            freq_dim: 256,
            text_dim: 4096,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }

    /// A14B I2V expert dims — identical to [`Self::t2v_14b`] but **`in_channels = 36`**: the 16-channel
    /// noise latent channel-concatenated with the 20-channel image conditioning `y` (4 mask + 16 image
    /// latent). The patch embedding consumes 36 channels; the prediction stays `out_channels = 16`.
    pub fn i2v_14b() -> Self {
        Self {
            in_channels: 36,
            ..Self::t2v_14b()
        }
    }
}

/// `AutoencoderKLWan` (z48, `is_residual`) decoder dims.
#[derive(Clone, Copy, Debug)]
pub struct VaeConfig {
    pub z_dim: usize,
    /// Decoder base width (`decoder_base_dim`).
    pub base_dim: usize,
    pub num_res_blocks: usize,
    /// Final spatial unpatchify factor (`patch_size`).
    pub patch_size: usize,
    /// Channels emitted by `conv_out` before unpatchify (= `out_channels × patch²` = 12).
    pub conv_out_channels: usize,
    pub out_channels: usize,
}

impl VaeConfig {
    pub fn ti2v_5b() -> Self {
        Self {
            z_dim: 48,
            base_dim: 256,
            num_res_blocks: 2,
            patch_size: 2,
            conv_out_channels: 12,
            out_channels: 3,
        }
    }
}

/// Per-channel latent de-normalization (`z = z·std + mean` before decode), from `vae/config.json`.
pub const LATENTS_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
pub const LATENTS_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

// ===========================================================================================
// Wan2.2 A14B (MoE) — z16 VAE + dual-expert inference knobs
// ===========================================================================================

/// `AutoencoderKLWan` (z16, Wan2.1 VAE) dims, used by **both** A14B variants. From
/// `Wan2.2-T2V-A14B-Diffusers/vae/config.json`: `base_dim 96`, `dim_mult [1,2,4,4]`, `z_dim 16`,
/// `num_res_blocks 2`, `temperal_downsample [false, true, true]`, **non-residual, no patchify** (unlike
/// the 5B's z48 [`VaeConfig`]). Spatial stride 8 (3 spatial up/down stages), temporal stride 4.
#[derive(Clone, Copy, Debug)]
pub struct Vae16Config {
    pub z_dim: usize,
    pub base_dim: usize,
    pub num_res_blocks: usize,
    pub out_channels: usize,
}

impl Vae16Config {
    pub fn wan21() -> Self {
        Self {
            z_dim: 16,
            base_dim: 96,
            num_res_blocks: 2,
            out_channels: 3,
        }
    }
}

/// Per-channel z16 latent de-normalization (`z = z·std + mean` before decode), from the z16
/// `vae/config.json` (`latents_mean`/`latents_std`). Distinct from the z48 [`LATENTS_MEAN`].
pub const LATENTS16_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
pub const LATENTS16_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

/// z16 VAE spatial downsample factor (latent `H = height / 8`).
pub const VAE16_STRIDE_SPATIAL: u32 = 8;
/// z16 VAE temporal downsample factor (latent `T = (frames - 1) / 4 + 1`).
pub const VAE16_STRIDE_TEMPORAL: u32 = 4;
/// Spatial size must be a multiple of `vae_stride_spatial (8) × patch (2) = 16` (vs 32 for the 5B).
pub const SIZE_MULTIPLE_14B: u32 = 16;

/// A14B defaults (the reference `WanModelConfig` MoE presets / the diffusers `model_index.json`).
pub const DEFAULT_STEPS_14B: u32 = 40;
pub const DEFAULT_FRAMES_14B: u32 = 81;
/// A14B playback cadence (`sample_fps`; 16 for both variants, vs the 5B's 24).
pub const DEFAULT_FPS_14B: u32 = 16;

/// T2V-A14B MoE knobs: timestep boundary `0.875·1000` selects high (≥) vs low (<) expert; flow-shift
/// 12.0; per-expert CFG (low 3.0, high 4.0).
pub const T2V_14B_BOUNDARY: f64 = 0.875;
pub const T2V_14B_FLOW_SHIFT: f64 = 12.0;
pub const T2V_14B_GUIDANCE_LOW: f32 = 3.0;
pub const T2V_14B_GUIDANCE_HIGH: f32 = 4.0;

/// I2V-A14B MoE knobs: boundary `0.900·1000`; flow-shift 5.0; per-expert CFG (both 3.5). Max-area cap
/// 704×1280 (aspect-preserving grid-aligned fit), like the 5B.
pub const I2V_14B_BOUNDARY: f64 = 0.900;
pub const I2V_14B_FLOW_SHIFT: f64 = 5.0;
pub const I2V_14B_GUIDANCE_LOW: f32 = 3.5;
pub const I2V_14B_GUIDANCE_HIGH: f32 = 3.5;
/// Resolution cap for I2V (and the 5B): the long edge × short edge must fit `704·1280`.
pub const MAX_AREA_14B: usize = 704 * 1280;

// ===========================================================================================
// Wan-VACE (Wan2.1-VACE-14B) — controllable video (replace_person / extend / bridge)
// ===========================================================================================

/// Registry id for the Wan-VACE controllable-video model. Matches the `mlx-gen-wan` `wan_vace`
/// descriptor so a consumer resolves the same engine across backends. VACE is **mode-agnostic** at
/// the engine boundary (diffusers `WanVACEPipeline`): the worker builds the per-mode control video +
/// mask (replace_person / extend / bridge) and passes one `ControlClip`; the provider VAE-encodes the
/// 96-channel control latent and runs the CFG VACE denoise.
pub const MODEL_ID_VACE: &str = "wan_vace";

/// VACE defaults (Wan2.1-VACE-14B: UniPC, flow-shift 5.0, guidance 5.0, 50 steps, 16 fps). The worker
/// overrides steps / guidance / shift / fps per request.
pub const DEFAULT_STEPS_VACE: u32 = 50;
pub const DEFAULT_GUIDANCE_VACE: f32 = 5.0;
pub const VACE_FLOW_SHIFT: f64 = 5.0;
pub const DEFAULT_FPS_VACE: u32 = 16;
pub const DEFAULT_FRAMES_VACE: u32 = 81;

/// `WanVACETransformer3DModel` config: the dimension-parametric Wan DiT ([`TransformerConfig`]) plus the
/// two VACE-only fields. The base dims are the Wan2.1-14B preset (z16 VAE, stride 4×8×8, dim 5120, 40
/// layers); the noisy latent enters the patch embedding as 16 channels (VACE conditioning rides the
/// separate control path, **not** a channel-concat like I2V). Mirrors `mlx-gen-wan`'s `WanVaceConfig`.
#[derive(Clone, Debug)]
pub struct WanVaceConfig {
    /// The base Wan DiT dims (dim, heads, layers, ffn, patch, eps, rope …) — Wan2.1-14B.
    pub base: TransformerConfig,
    /// Which main-block indices receive a VACE control hint (diffusers default
    /// `[0, 5, 10, 15, 20, 25, 30, 35]`; must include 0 so `vace_blocks.0` carries `proj_in`).
    pub vace_layers: Vec<usize>,
    /// The control-latent channel count (diffusers default 96 = 32 video latent + 64 mask unfold).
    pub vace_in_channels: usize,
}

impl WanVaceConfig {
    /// Wan2.1-VACE-14B defaults (the real checkpoint: 40 layers, dim 5120, 8 vace blocks, 96 control
    /// channels). The base preset is the 14B T2V DiT (in/out 16) — VACE adds the control path on top.
    pub fn vace_14b() -> Self {
        Self {
            base: TransformerConfig::t2v_14b(),
            vace_layers: vec![0, 5, 10, 15, 20, 25, 30, 35],
            vace_in_channels: 96,
        }
    }
}

/// `UMT5EncoderModel` (`google/umt5-xxl`) dims.
#[derive(Clone, Copy, Debug)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub num_buckets: usize,
    pub max_distance: usize,
    pub eps: f64,
    pub max_length: usize,
    pub pad_token_id: i32,
}

impl TextEncoderConfig {
    pub fn umt5_xxl() -> Self {
        Self {
            vocab_size: 256384,
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            num_buckets: 32,
            max_distance: 128,
            eps: 1e-6,
            max_length: 512,
            pad_token_id: 0,
        }
    }
}
