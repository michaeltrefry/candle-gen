//! SVD (img2vid-xt) component configs — transcribed from the checkpoint JSON
//! (`stabilityai/stable-video-diffusion-img2vid-xt`): `unet/config.json`, `vae/config.json`,
//! `image_encoder/config.json`, `scheduler/scheduler_config.json`. Static defaults (no JSON parse on
//! the hot path). A direct transcription of `mlx-gen-svd`'s config (backend-neutral data).

/// `UNetSpatioTemporalConditionModel` config.
#[derive(Clone, Debug)]
pub struct UnetConfig {
    /// 8 = 4 noise latent + 4 image-conditioning latent (channel-concat).
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub cross_attention_dim: usize,
    pub num_attention_heads: Vec<usize>,
    pub transformer_layers_per_block: usize,
    /// Each fps/motion_bucket/noise_aug id → a 256-dim sinusoid; 3 of them concat → 768.
    pub addition_time_embed_dim: usize,
    /// 3 · `addition_time_embed_dim` = 768 (the `add_embedding` MLP input).
    pub projection_class_embeddings_input_dim: usize,
    /// SVD-XT default frame count.
    pub num_frames: usize,
}

impl Default for UnetConfig {
    fn default() -> Self {
        Self {
            in_channels: 8,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280, 1280],
            layers_per_block: 2,
            cross_attention_dim: 1024,
            num_attention_heads: vec![5, 10, 20, 20],
            transformer_layers_per_block: 1,
            addition_time_embed_dim: 256,
            projection_class_embeddings_input_dim: 768,
            num_frames: 25,
        }
    }
}

/// `AutoencoderKLTemporalDecoder` config (2D encoder + temporal decoder).
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub scaling_factor: f32,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.18215,
        }
    }
}

/// `CLIPVisionModelWithProjection` (OpenCLIP ViT-H/14) config — image conditioning encoder.
#[derive(Clone, Debug)]
pub struct ImageEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub image_size: usize,
    /// Output dim of the visual projection (the image embed fed to UNet cross-attn).
    pub projection_dim: usize,
    pub layer_norm_eps: f64,
}

impl Default for ImageEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            patch_size: 14,
            image_size: 224,
            projection_dim: 1024,
            layer_norm_eps: 1e-5,
        }
    }
}

/// `EulerDiscreteScheduler` (EDM) config for SVD (`use_karras_sigmas`, `timestep_type="continuous"`,
/// `prediction_type="v_prediction"`). The sigma schedule is pure Karras over the **config**
/// `sigma_min`/`sigma_max` (the betas/alphas path is unused) and the model timestep is `0.25·ln(σ)`.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub sigma_min: f32,
    pub sigma_max: f32,
    /// Karras rho (paper default 7).
    pub rho: f32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 700.0,
            rho: 7.0,
        }
    }
}

/// Public registry id: `candle_gen::load("svd_xt", spec)`.
pub const MODEL_ID: &str = "svd_xt";

/// SVD-XT default denoise steps / frame count (the request overrides each).
pub const DEFAULT_STEPS: u32 = 25;
pub const DEFAULT_FRAMES: u32 = 25;
/// Default motion-conditioning cadence (the `fps_id` SVD was trained on).
pub const DEFAULT_CONDITIONING_FPS: u32 = 7;
/// Default playback cadence when the request omits `fps`.
pub const DEFAULT_FPS: u32 = 7;
/// VAE spatial compression (8×).
pub const VAE_SCALE: u32 = 8;
/// Output `width`/`height` must be divisible by this: VAE 8× × UNet 8× (3 stride-2 stages).
pub const SIZE_ALIGN: u32 = VAE_SCALE * 8;
