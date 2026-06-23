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
> **FLUX.2-klein-9B txt2img** is the third model-family expansion (epic 3692, sc-3695) and the first
> **from-scratch** port — `candle-transformers` has no FLUX.2, so the whole architecture is ported from
> `mlx-gen-flux2` on candle-core/candle-nn: a **Qwen3** text encoder (36-layer dense LM; the hidden
> states of layers 9/18/27 concatenate into a 12288-wide `prompt_embeds`), the **MMDiT** transformer
> (8 joint + 24 fused-parallel single blocks, **4-axis interleaved RoPE**, global per-stream
> modulation), and the **AutoencoderKL-Flux2** VAE (32-ch latent, a 2×2 pack into 128-ch transformer
> space, BatchNorm-stats latent normalization). Registered under `"flux2_klein_9b"`, distilled 4-step
> flow-match Euler with the empirical-mu sigma shift, guidance 1.0 (>1.0 runs a CFG negative pass).
> Same deterministic CPU-seeded-noise contract; tokenization reuses gen-core's `TextTokenizer`
> (`ChatTemplate::QwenInstructNoThink`). Runs the reference math in f32 (~59 GB resident on the 96 GB
> Blackwell; a bf16 pass is a follow-up). txt2img-only first slice — the edit variants
> (`flux2_klein_9b_edit` / `_kv_edit`, single/multi Reference + reference-KV cache), LoRA, and
> quantization are deferred. **GPU-verified** on RTX PRO 6000 (sm_120): real 1024² render + conformance
> suite pass.
>
> **Qwen-Image txt2img** is the fourth model-family expansion (epic 3692, sc-3696) — the largest
> from-scratch port, the ~20B 60-layer dual-stream MMDiT. Ported from `mlx-gen-qwen-image`: a
> **Qwen2.5-VL** text encoder (28-layer LM; the last normed hidden state with the 34-token system
> prefix dropped → 3584-wide `prompt_embeds`), the **dual-stream MMDiT** (60 blocks, joint `[txt,img]`
> attention, **3-axis interleaved RoPE**, per-stream AdaLN modulation, timestep-only conditioning),
> and the **AutoencoderKLQwenImage** VAE (a causal-Conv3d VAE that, for a single image, reduces to
> conv2d on the last depth tap; **channel-L2** normalization; per-channel latent mean/std). Registered
> under `"qwen_image"`, dynamic-μ flow-match Euler with **true CFG** (norm-rescaled) and a negative
> prompt. The encoder runs **f32** and the MMDiT **bf16** (~74 GB resident; an all-f32 load would not
> fit). txt2img-only first slice — img2img / Edit / ControlNet / Lightning / LoRA / quantization are
> deferred. **GPU-verified** on RTX PRO 6000 (sm_120): real 1024² render + conformance suite pass.
> (The snapshot's `tokenizer/tokenizer.json` is built by the worker from `vocab.json`+`merges.txt`;
> the provider requires it, matching the mlx provider.)
>
> **Wan2.2 TI2V-5B txt2video** is the fifth model-family expansion (epic 3692, sc-3697) — the first
> **video** family (modality `Video`), emitting `GenerationOutput::Video`. Ported from the diffusers
> checkpoint (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`): a **UMT5-XXL** encoder (24-layer `UMT5EncoderModel`
> — per-layer relative-position bias, gated-GELU, no attention scaling), the 30-layer
> **`WanTransformer3DModel`** DiT (**3-axis interleaved RoPE**, AdaLN modulation, cross-attention to
> text, classifier-free guidance), and the **`AutoencoderKLWan`** temporal VAE. Since candle ships **no
> conv3d**, the causal Conv3d is implemented as a left-pad-in-time + summed conv2d taps
> (`candle-gen-wan/src/conv3d.rs`); temporal upsampling reproduces the reference `time_conv` doubling +
> `DupUp3D` residual in one pass. **UniPC** flow-match scheduler (order-2 bh2, default) with a Euler
> fallback. UMT5 + VAE run **f32**, the 5B DiT **bf16** (~33 GB resident). The text context is
> **zero-padded to 512 tokens** before the DiT (the model trained that way — feeding only the real
> tokens silently collapses the latent). txt2video-only first slice — image/keyframe conditioning
> (TI2V/I2V), VACE, LoRA, quantization, and tiling are deferred. **GPU-verified** on RTX PRO 6000
> (sm_120): real 512² cat-walking clip + conformance suite pass; UMT5 / DiT / VAE forward passes are
> **bit-exact** vs diffusers.
>
> **LTX-2.3 (distilled 22B) txt2video** is the sixth model-family expansion (epic 3692, sc-3698) — the
> heaviest port: a **Gemma-3-12B** text encoder (48-layer GQA, alternating local/global RoPE, q/k-norm;
> all 49 hidden states extracted) feeding a **per-token-RMS aggregation** (3840×49 → 188160) → text
> projection → an 8-layer **learnable-register connector** (128 registers replace the left-pad), then
> the 48-layer **`AVTransformer3DModel`** video DiT (**split 3-D RoPE** with per-head float64 freqs,
> per-head **2·sigmoid gated** attention, adaLN-single 9-row modulation, prompt-adaLN text conditioning),
> and the **`CausalVideoAutoencoder`** temporal VAE (latent 128-ch, patch 4, 8× temporal / 32× spatial;
> pixel-norm; depth-to-space upsampling). Single flat 22B safetensors bundles DiT + VAE + projection +
> connector; the Gemma encoder is a separate snapshot (`LTX_GEMMA_DIR`). Since candle ships **no conv3d**,
> the VAE causal Conv3d is summed conv2d taps with frame-replication temporal pad
> (`candle-gen-ltx/src/conv3d.rs`). **Rectified-flow** distilled scheduler (fixed 8-step σ schedule, no
> CFG — guidance is distilled in). DiT + connector + projection + Gemma run **bf16** (22B+12B doesn't fit
> f32 on one 96 GB GPU), the VAE **f32**, attention/norms upcast to f32. txt2video-only first slice — the
> **audio stack** (audio-VAE + vocoder + AV-joint DiT), the 2-stage latent upsampler, I2V conditioning,
> prompt-enhance, LoRA/IC-LoRA, and fp8/quant are deferred. **GPU-verified** on RTX PRO 6000 (sm_120):
> real cat-walking clip renders coherently on the first visual try.
>
> **JoyCaption (Llama-JoyCaption-beta-one) image captioning** is the seventh model-family expansion
> (epic 3692, sc-3699) and the first **`Captioner`** (image → text, not `Generator`): a
> `LlavaForConditionalGeneration` ported from scratch (no candle-transformers — the contract needs the
> SigLIP **`-2`** hidden state and a Llama fed pre-spliced `inputs_embeds`, neither of which it exposes).
> A **SigLIP-so400m/14@384** vision tower (27 layers, 1152-d; returns the penultimate hidden state, all
> 729 patch tokens) → a **gelu-MLP** multimodal projector (1152→4096) → the single `<|image|>` marker is
> expanded to 729 placeholders and the projected rows are spliced over them → a **Llama-3.1-8B** decoder
> (GQA 32/8, head-dim 128, **llama3 RoPE scaling**, KV-cache) generates the caption autoregressively
> (greedy or temperature/top-p with a small CTRL-style repetition penalty; stops at the eot/eom/eos set).
> The whole assembly runs **bf16** (native checkpoint dtype), logits upcast to f32 for sampling; the
> SceneWorks caption prompt map (12 caption types × length templates) and the Llama-3 chat wrapper port
> verbatim. Single snapshot dir (4 shards + `tokenizer.json`). **GPU-verified** on RTX PRO 6000 (sm_120):
> a real photo captions coherently and on-subject. `backend = "candle"`, `mac_only = false`.
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
  candle-gen-flux2/         # FLUX.2-klein-9B provider crate: from-scratch MMDiT + Qwen3 + AutoencoderKL-Flux2
  candle-gen-qwen-image/    # Qwen-Image provider crate: from-scratch 60-layer MMDiT + Qwen2.5-VL + causal-Conv3d VAE
  candle-gen-wan/           # Wan2.2 TI2V-5B video provider crate: WanTransformer3DModel + UMT5-XXL + temporal AutoencoderKLWan (from-scratch conv3d)
  candle-gen-ltx/           # LTX-2.3 (distilled 22B) video provider crate: AVTransformer3DModel DiT + Gemma-3-12B encoder + connector + CausalVideoAutoencoder (from-scratch conv3d)
  candle-gen-joycaption/    # JoyCaption image-captioning provider crate (first Captioner): SigLIP-so400m tower + gelu-MLP projector + Llama-3.1-8B decoder (from-scratch LLaVA)
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

### How portability works: PTX-JIT for dense kernels, a multi-arch fatbin for quantized kernels

candle-kernels has **two** compile paths, and they need different portability treatments (verified
against the vendored sources):

- **Dense kernels** build via cudaforge `.build_ptx()` → `nvcc --ptx`, emitting one **`compute_80`
  PTX** (virtual ISA) per kernel. The driver JIT-compiles that PTX to the runtime GPU's native SASS
  at first load, so it runs on **every NVIDIA arch ≥ sm_80** — Ampere (sm_80/86) → Ada (sm_89) →
  Hopper (sm_90) → Blackwell (sm_120) — from a single embedded PTX. (Tradeoff: it does not use
  sm_90a/sm_120a arch-accelerated tensor features, and the first run is slower while the driver JITs;
  the result caches per-GPU under `%APPDATA%\NVIDIA\ComputeCache`.)
- **Quantized + MoE kernels** (`mmq_gguf/*`, `moe/*`, `mmvq_gguf` — the GGUF `QMatMul`) build via
  cudaforge `.build_lib()` → `nvcc -c`: a **static `libmoe.a` of SASS, _not_ PTX**. cudaforge emits
  one `-gencode` from `CUDA_COMPUTE_CAP` (`GpuArch::parse` runs `parse::<usize>()` on the whole
  string, so a `;`-list does **not** parse — there is no multi-cap support). At the `=80` baseline the
  archive held only an **sm_80 cubin**; SASS is not forward-compatible across major arches and there
  is no PTX to JIT, so on **Blackwell sm_120 every quant matmul silently returned zeros** — dense
  models rendered but quantized models came out black (**sc-7544**; the dense PTX path masked it).

**The fix (sc-7544): a multi-arch fatbin for the quant path.** cudaforge can't emit a cap list and the
candle pin is upstream (not a fork), so candle-kernels is **locally forked** in `vendor/candle-kernels`
(identical to the pinned rev except three `-gencode` lines in `build.rs`) and patched in via the
workspace `[patch]`. nvcc accumulates `-gencode` flags, so `libmoe.a` becomes a real fatbin embedding
**sm_80 + sm_90 + sm_120 SASS + `compute_120` PTX** — one binary that runs natively Ampere → Ada →
Hopper → Blackwell and JITs forward to newer archs. Keep `CUDA_COMPUTE_CAP=80` in the recipes (it
seeds the sm_80 baseline for both paths). Verified on RTX PRO 6000 (sm_120): `cuobjdump --list-elf`
shows sm_80/sm_90/sm_120 cubin per kernel, and `candle-gen`'s `cuda_quant_smoke` test has the Q4/Q8
`QMatMul` matching the CPU reference (cos ≈ 1.0, vs cos ≈ 0 / all-zeros before). That smoke runs in
the CUDA gate so the regression can't return silently. **Re-vendor on every candle pin bump** — see
`vendor/candle-kernels/VENDORED.md`.

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
