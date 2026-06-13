#!/usr/bin/env python
"""Dump a schedule + CFG golden for the Lens sampler (candle-gen sc-5114). Weight-free.

Builds the diffusers `FlowMatchEulerDiscreteScheduler` exactly as `LensImagePipeline` does
(empirical-mu + custom `linspace(1, 1/n, n)` sigmas + dynamic shift) for the Turbo (4-step) and base
(20-step) counts, and records the resulting sigmas/timesteps, a single denoise `step`, and the
norm-rescaled CFG output, so the Rust `candle_gen_lens::schedule` can be checked near-bit (f32, CPU).

Golden contents (per `n` in {4, 20}):
  - `sigmas_{n}`     [n+1] — `scheduler.sigmas` (shifted, trailing 0);
  - `timesteps_{n}`  [n]   — `scheduler.timesteps` (= shifted_sigma · 1000);
  - `step_in_{n}` / `step_noise_{n}` / `step_out_{n}` — one `scheduler.step(noise, t0, latents)`.
Plus CFG: `cfg_cond` / `cfg_uncond` / `cfg_out` (norm-rescaled, guidance 5.0).

Run (from the worktree root) with the transformers-5.8 lens-venv:
  & "C:\\Users\\Michael\\AppData\\Roaming\\SceneWorks\\python\\lens-venv\\Scripts\\python.exe" \\
      candle-gen-lens\\scripts\\dump_lens_schedule_golden.py [out_dir]

Default out_dir: .scratch/lens-schedule-goldens/  (not committed — regenerable).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file

SEQ_LEN = 4096  # 64×64 latent grid (1024px) — exercises the ≤4300 interpolation branch
GUIDANCE = 5.0
STEP_COUNTS = [4, 20]


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def find_scheduler() -> str:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    m = sorted(glob.glob(str(hub / "models--microsoft--Lens-Turbo" / "snapshots" / "*" / "scheduler")))
    if not m:
        sys.exit("no microsoft/Lens-Turbo scheduler snapshot found")
    return m[-1]


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".scratch/lens-schedule-goldens")
    out_dir.mkdir(parents=True, exist_ok=True)
    sched_dir = find_scheduler()
    print(f"scheduler: {sched_dir}")

    tensors: dict[str, torch.Tensor] = {}
    torch.manual_seed(0)

    for n in STEP_COUNTS:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(sched_dir)
        mu = compute_empirical_mu(SEQ_LEN, n)
        sigmas = np.linspace(1.0, 1.0 / n, n)
        sched.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)

        tensors[f"sigmas_{n}"] = sched.sigmas.to(torch.float32).cpu().contiguous()
        tensors[f"timesteps_{n}"] = sched.timesteps.to(torch.float32).cpu().contiguous()

        latents = torch.randn(1, SEQ_LEN, 128, dtype=torch.float32)
        noise = torch.randn(1, SEQ_LEN, 128, dtype=torch.float32)
        out = sched.step(noise, sched.timesteps[0], latents, return_dict=False)[0]
        tensors[f"step_in_{n}"] = latents.contiguous()
        tensors[f"step_noise_{n}"] = noise.contiguous()
        tensors[f"step_out_{n}"] = out.to(torch.float32).contiguous()

    # CFG (norm-rescaled), guidance 5.0
    cond = torch.randn(1, SEQ_LEN, 128, dtype=torch.float32)
    uncond = torch.randn(1, SEQ_LEN, 128, dtype=torch.float32)
    comb = uncond + GUIDANCE * (cond - uncond)
    cond_norm = torch.norm(cond, dim=-1, keepdim=True)
    comb_norm = torch.norm(comb, dim=-1, keepdim=True)
    scale = torch.where(
        comb_norm > 0, cond_norm / comb_norm.clamp_min(1e-12), torch.ones_like(comb_norm)
    )
    cfg_out = comb * scale
    tensors["cfg_cond"] = cond.contiguous()
    tensors["cfg_uncond"] = uncond.contiguous()
    tensors["cfg_out"] = cfg_out.contiguous()

    dst = out_dir / "lens_schedule_golden.safetensors"
    save_file(tensors, str(dst))
    print(f"wrote {dst}  (steps {STEP_COUNTS}, seq_len {SEQ_LEN}, guidance {GUIDANCE})")


if __name__ == "__main__":
    main()
