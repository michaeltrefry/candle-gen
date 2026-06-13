#!/usr/bin/env python
"""Dump a Lens VAE decode golden (candle-gen sc-5113).

Runs the cached `microsoft/Lens-Turbo` VAE (a diffusers `AutoencoderKLFlux2`) through the vendor
`LensImagePipeline._decode` math, in **float32**, over a synthetic DiT-shaped latent. The Rust side
loads the same `vae/` checkpoint into the shared `candle_gen_flux2::Flux2Vae` and decodes via the Lens
shim; this golden lets it check the decoded pixels match the reference.

The reference `_decode` (verbatim):
  rearrange(b (h w) (c p1 p2) -> b c (h p1) (w p2)) → _patchify → x·std + mean (bn de-normalize,
  std = sqrt(running_var + batch_norm_eps)) → _unpatchify → vae.decode(x).sample.
The rearrange+patchify pair is an identity (the DiT's 128 channels already pack as c·4 + p1·2 + p2),
so the candle shim is just a reshape to the packed NCHW grid before `decode_packed`.

Run (from the worktree root) with the transformers-5.8 lens-venv:
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-lens\\scripts\\dump_lens_vae_golden.py [out_dir]

Default out_dir: .scratch/lens-vae-goldens/  (not committed — regenerable).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file

# Packed latent grid (post-DiT): image is latent_h·16 × latent_w·16 (2× unpatchify × 8× VAE upsample).
LATENT_H, LATENT_W = 16, 16


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def find_vae() -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    m = sorted(glob.glob(str(hub / "models--microsoft--Lens-Turbo" / "snapshots" / "*" / "vae")))
    if not m:
        sys.exit("no microsoft/Lens-Turbo vae snapshot found")
    return m[-1]


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/lens-vae-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)

    vdir = find_vae()
    print(f"vae: {vdir}\nloading (f32, CPU)…", flush=True)
    vae = AutoencoderKLFlux2.from_pretrained(vdir, torch_dtype=torch.float32).to("cpu").eval()

    img_len = LATENT_H * LATENT_W
    torch.manual_seed(0)
    # DiT packed patch-space output: [1, h·w, 128].
    latents = torch.randn(1, img_len, 128, dtype=torch.float32)

    with torch.no_grad():
        # rearrange "b (h w) (c p1 p2) -> b c (h p1) (w p2)" via plain torch (no einops dep).
        x0 = (
            latents.view(1, LATENT_H, LATENT_W, 32, 2, 2)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(1, 32, LATENT_H * 2, LATENT_W * 2)
        )
        bn = vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1)
        var = bn.running_var.view(1, -1, 1, 1)
        std = torch.sqrt(var + vae.config.batch_norm_eps)
        shift = -mean
        scale = 1.0 / std
        x = _patchify(x0)
        x = x / scale - shift  # = x·std + mean
        x = _unpatchify(x)
        out = vae.decode(x).sample  # [1, 3, H, W] in [-1, 1]

    tensors = {
        "latents": latents.contiguous(),  # [1, h·w, 128] DiT-shaped input
        "out": out.contiguous(),  # [1, 3, H, W] decoded pixels in [-1, 1]
        "grid_hw": torch.tensor([LATENT_H, LATENT_W], dtype=torch.int64),
    }
    dst = out_dir / "lens_vae_golden.safetensors"
    save_file(tensors, str(dst))
    print(f"wrote {dst}  (latents={tuple(latents.shape)}, out={tuple(out.shape)})")


if __name__ == "__main__":
    main()
