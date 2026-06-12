# candle-gen

Rust-native generative image (and, later, video) model inference on
[candle](https://github.com/huggingface/candle) — the **Windows/CUDA sibling** of
[`mlx-gen`](https://github.com/michaeltrefry/mlx-gen) (Apple MLX). Both crates implement the **same**
backend-neutral [`gen_core`](https://github.com/michaeltrefry/mlx-gen/tree/main/gen-core) contract
(SceneWorks epic 3720), so a consumer pins one backend by git SHA, links its provider crates, and
calls the identical `Generator` / registry API regardless of which tensor backend is underneath.

> **Status: SDXL txt2img implemented on the Candle/CUDA lane.** `SdxlGenerator::generate` runs the
> full pipeline — dual CLIP → UNet (real CFG) → f16 VAE — for both `sdxl` and `realvisxl`
> (sc-3675, RealVisXL + parity tests sc-3677). Output is deterministic and launch-portable per seed
> (CPU-seeded noise + non-ancestral DDIM, sc-3673). Perf/VRAM work has landed: f16 CLIP + optional
> flash-attention (sc-3674), VAE tiling + staged CLIP free for torch-parity peak VRAM at 1024²
> (sc-4987), and UNet/VAE component caching across `generate` calls (sc-5037). The provider still
> self-registers into the shared `gen_core` inventory registry, with the
> `CandleError → gen_core::Error` bridge + device plumbing wired (scaffold sc-4946).
>
> **Z-Image txt2img** is the first model-family expansion beyond SDXL (epic 3692, sc-3693):
> `ZImageGenerator::generate` adapts the `candle-transformers` `z_image` reference (Qwen3 text
> encoder → DiT flow-match Euler, distilled 4-step, **no CFG** → AutoencoderKL VAE), registered under
> `"z_image_turbo"`. Same deterministic CPU-seeded-noise contract; the Qwen chat-template tokenization
> is reused from gen-core (`TextTokenizer` / `ChatTemplate::QwenInstruct`). txt2img-only first slice
> (img2img / LoRA / quantization are rejected, not silently dropped). **GPU-verified** on RTX PRO 6000
> (sm_120): real 1024² renders + the conformance suite pass.
>
> **FLUX.1 [schnell] + [dev] txt2img** is the second model-family expansion (epic 3692, sc-3694):
> `FluxGenerator::generate` adapts the `candle-transformers` `flux` reference (dual **CLIP-L + T5-XXL**
> text encoders → FLUX DiT flow-match Euler → FLUX AutoencoderKL VAE), registered under both
> `"flux1_schnell"` (Apache-2.0, timestep-distilled: 4-step, **no guidance**) and `"flux1_dev"`
> (gated, guidance-distilled: 25-step time-shifted schedule, embedded guidance ~3.5). The DiT + VAE
> load directly from the black-forest-labs **root** checkpoints (`flux1-*.safetensors`,
> `ae.safetensors`) — candle's `flux` speaks the BFL key layout, so no diffusers→BFL remap is needed —
> while the text encoders come from the `text_encoder/` (CLIP) and `text_encoder_2/` (T5) subdirs. The
> CLIP `tokenizer.json` is **vendored** (the snapshot ships CLIP only as `vocab.json`+`merges.txt`,
> which a byte-level BPE mis-tokenizes; sc-2787 parity). Same deterministic CPU-seeded-noise contract;
> FLUX.1[dev] license/credential gating stays upstream in the worker (no descriptor gating flag,
> consistent with the mlx provider). txt2img-only first slice (Reference/IP-adapter, LoRA,
> quantization rejected). **GPU-verified** on RTX PRO 6000 (sm_120): real 1024² schnell + dev renders
> + both conformance suites pass.
>
> **candle pinned to git main (post-0.10.2)** — REQUIRED for Blackwell sm_120. The crates.io 0.10.2
> release throws `CUDA_ERROR_INVALID_PTX` at the first candle-kernels kernel whenever
> candle-transformers is linked (SDXL + Z-Image both; plain candle-core works). The git rev clears it
> and is source-compatible. See `[workspace.dependencies]`.

## Layout

```
candle-gen/                 # workspace root
  candle-gen/               # core crate: re-exports gen_core + candle; device/dtype helpers;
                            #   CandleError -> gen_core::Error bridge
  candle-gen-sdxl/          # SDXL provider crate: Generator impl + descriptor + inventory::submit!
  candle-gen-z-image/       # Z-Image (Z-Image-Turbo) provider crate: txt2img via candle-transformers
  candle-gen-flux/          # FLUX.1 [schnell]+[dev] provider crate: txt2img via candle-transformers `flux`
  scripts/
    check-gen-core-skew.sh  # version-skew gate: fails if >1 sceneworks-gen-core resolves
    check-cuda.ps1          # local cuda gate: vcvars + cargo build/test --features cuda (run pre-push)
    package-cuda.ps1        # bundle a CUDA build + redist DLLs into dist/ (sc-3676; see Packaging)
  .github/workflows/ci.yml  # macOS/Linux fmt+clippy+check+test + skew self-test; manual Windows/CUDA lane
```

A provider crate self-registers just by being linked (`inventory::submit!`), so adding a model is
purely additive — there is no central match statement to edit. `candle-gen-sdxl` registers a single
descriptor under the id `"sdxl"` (the SceneWorks worker maps both `sdxl` and `realvisxl` to engine
id `"sdxl"`), with `backend: "candle"`.

## Backends / features

The default build is **CPU** (`candle-core`'s default) and works on macOS with no extra features.

| feature      | backend                | platform        | in `default`? |
|--------------|------------------------|-----------------|---------------|
| *(none)*     | CPU                    | all (Mac dev)   | yes           |
| `metal`      | Apple Metal GPU        | macOS           | no            |
| `cuda`       | NVIDIA CUDA            | Windows/Linux   | no            |
| `flash-attn` | implies `cuda` (TODO)  | Windows/CUDA    | no            |

`cuda` / `flash-attn` need the CUDA toolkit and **do not build on Mac**; all CUDA-only code is gated
behind `#[cfg(feature = "cuda")]`. `flash-attn` currently just implies `cuda` — the fused kernels
need the separate `candle-flash-attn` crate, wired in a later slice on the Windows box.

## Packaging (Windows / CUDA) — sc-3676

The goal is **one distributable CUDA worker that runs on every NVIDIA GPU we support, not just the
build box's Blackwell** — the "central fat binary, like torch" model.

### How portability actually works here: baseline PTX, not a fatbin

The spike (sc-3495) assumed candle compiles a multi-arch **SASS fatbin** at build time and that we
would feed it a multi-cap list (`CUDA_COMPUTE_CAP=80;86;89;90;120`). **That is not how candle 0.10.2
works**, and this was verified against the vendored sources:

- candle-kernels 0.10.2 builds via **cudaforge 0.1.5** `.build_ptx()` → `nvcc --ptx`, emitting **one
  PTX (virtual ISA) per kernel**, embedded in the binary. No `.cubin`/fatbin is produced.
- cudaforge parses `CUDA_COMPUTE_CAP` as a **single** value (`GpuArch::parse` runs `parse::<usize>()`
  on the whole string). A `;`-separated list **fails to parse** — candle does not accept a cap list.

So portability comes from **PTX forward-compatibility** instead: we build at a **baseline virtual
arch, `CUDA_COMPUTE_CAP=80`** (Ampere). The embedded `compute_80` PTX is **JIT-compiled by the
driver** to the runtime GPU's native SASS at first load, so a **single binary runs on every NVIDIA
arch ≥ sm_80** — Ampere (sm_80/86) → Ada (sm_89) → Hopper (sm_90) → Blackwell (sm_120). This is
broader coverage than any fixed fatbin cap list, with no candle fork.

Tradeoffs (acceptable for SDXL): `compute_80` PTX does **not** use sm_90a/sm_120a arch-accelerated
tensor features, and **first run is slower** while the driver JIT-compiles the PTX. The driver caches
the result per-GPU under `%APPDATA%\NVIDIA\ComputeCache`, so subsequent launches load cached SASS.

> If we ever need true per-arch SASS (e.g. to use sm_90a/sm_120a features), the path is to **fork
> candle-kernels' `build.rs`** to emit `-gencode` for a cap list (a real fatbin) — deliberately out
> of scope here; baseline PTX is the lighter, more portable default.

### Build

Build-time needs the **CUDA 12.9 toolkit (nvcc)** + **VS 2022 v143 (MSVC 14.4x) Build Tools**; the
build is driven through `vcvars64.bat`. From a `cmd` shell that has sourced vcvars:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_COMPUTE_CAP=80
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
cargo build --release -p candle-gen-sdxl --example sdxl-txt2img --features cuda
```

The scripted, reproducible form of this — sources vcvars, sets the env, runs `cargo build/test
--workspace --features cuda` — is `scripts/check-cuda.ps1`. **Run it before pushing CUDA-touching
changes**: the CPU/Metal CI lanes are blind to `#[cfg(feature = "cuda")]` code, so this is the real
cuda gate.

```powershell
pwsh scripts/check-cuda.ps1            # build + test
pwsh scripts/check-cuda.ps1 -SkipTests # build-only smoke check
```

The `windows-cuda` lane in `.github/workflows/ci.yml` runs the same recipe but is **manual-only**
(`workflow_dispatch`) — it needs no standing runner. To run it in CI you must first register a
self-hosted `[self-hosted, windows, cuda]` runner, then dispatch the workflow by hand. (GitHub's
hosted GPU larger-runners are Tesla T4 / sm_75, below our sm_80 baseline, so they can't run it.)

### Bundle the runtime DLLs

The target machine needs the CUDA **runtime** libraries but should **not** require a CUDA Toolkit
install. `scripts/package-cuda.ps1` copies the binary plus the redistributable DLLs (which cudarc
dynamic-links, resolved from the exe's own directory) into `dist/`:

```powershell
pwsh scripts/package-cuda.ps1 -BinaryPath target\release\examples\sdxl-txt2img.exe
```

Bundled redist DLLs (CUDA 12.9; the script globs the version suffixes):

| DLL                          | provides            |
|------------------------------|---------------------|
| `cudart64_12.dll`            | CUDA runtime        |
| `cublas64_12.dll`            | cuBLAS              |
| `cublasLt64_12.dll`          | cuBLAS-Lt           |
| `curand64_10.dll`            | cuRAND              |
| `nvrtc64_120_0.dll`          | NVRTC               |
| `nvrtc-builtins64_129.dll`   | NVRTC builtins      |

The script also writes a `RUNTIME.txt` manifest into the bundle. Verified: with the bundle's DLLs
present and the **CUDA toolkit removed from `PATH`**, the binary runs end-to-end (DLLs resolve from
the exe's directory).

### Minimum driver

The **NVIDIA driver is not bundled** (it is not redistributable) and is what JIT-compiles the PTX +
provides `libcuda`. For the bundled **CUDA 12.9** runtime the floor is:

- **Windows: driver ≥ 576.02** (CUDA 12.9 GA).
- GPU compute capability **≥ 8.0** (Ampere / RTX 30-series or newer).

Older drivers should be updated from nvidia.com; the CUDA runtime DLLs in the bundle do **not** lift
the driver requirement.

## gen-core pinning (read before bumping)

`sceneworks-gen-core` is pinned by **git SHA** in the root `Cargo.toml`
(`[workspace.dependencies]`) to the **same rev the SceneWorks worker pins**. Everything is
SHA-pinned: if candle-gen resolves gen-core at rev A while the worker resolves rev B, cargo silently
builds **both**, the provider crate registers into one `inventory` registry while the worker queries
the other, and the symptom is **"engine not found" at runtime** (not a compile error). Run the gate:

```bash
bash scripts/check-gen-core-skew.sh            # checks candle-gen's build graph
bash scripts/check-gen-core-skew.sh --self-test  # proves the gate fires on canned skew
```

When bumping the gen-core pin, bump it in lockstep with the worker's `mlx-gen` + `sceneworks-gen-core`
pins.

## Develop

```bash
cargo fmt --all
cargo check --workspace                 # CPU (Mac default)
cargo check --workspace --features metal  # Metal backend builds
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                  # registry-resolution + bridge tests
```

The candle version this scaffold settled on is recorded in `[workspace.dependencies]`
(`candle-core` / `candle-nn`).
