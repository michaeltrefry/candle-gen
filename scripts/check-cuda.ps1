<#
.SYNOPSIS
  Local CUDA gate for candle-gen — the reproducible build/test recipe in a script, not shell history.

.DESCRIPTION
  The default CI lanes (CPU / Metal) are structurally blind to anything behind
  `#[cfg(feature = "cuda")]`: cuda-only code can break the build and CI stays green because no default
  lane compiles that path. Run this before pushing CUDA-touching changes — it sources the VS2022 v143
  toolset via vcvars and builds + tests the workspace with `--features cuda` at the baseline virtual
  arch CUDA_COMPUTE_CAP=80 (see README "Packaging"). This is the same recipe the (manual) windows-cuda
  workflow runs; keep the three in sync (here, .github/workflows/ci.yml, scripts/package-cuda.ps1).

  The `--test` run includes candle-gen's `cuda_quant_smoke` (sc-7544): a weightless Q4/Q8 `QMatMul`
  CPU-vs-CUDA check that guards the multi-arch fatbin so the Blackwell quant regression (quant kernels
  silently no-op'ing to zeros at the sm_80-only baseline) can't return silently. Don't run with
  -SkipTests when validating a CUDA/packaging change.

.PARAMETER ComputeCap
  Baseline virtual arch. Default 80 (Ampere). Seeds the dense kernels' embedded `compute_80` PTX (the
  driver JIT-compiles it up to the runtime GPU) AND the sm_80 baseline of the quantized kernels'
  multi-arch fatbin (vendor/candle-kernels adds native sm_90/sm_120 SASS + forward PTX on top). Keep
  =80; NOT a hardware pin.

.PARAMETER CudaPath
  CUDA Toolkit root. Default: $env:CUDA_PATH, else the v12.9 default install path.

.PARAMETER Vcvars
  Path to vcvars64.bat (VS2022 v143 Build Tools). Default: the 2022 BuildTools install.

.PARAMETER SkipTests
  Build only, skip `cargo test` (faster smoke check).

.EXAMPLE
  pwsh scripts/check-cuda.ps1
#>
[CmdletBinding()]
param(
    [int]$ComputeCap = 80,
    [string]$CudaPath = $(if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" }),
    [string]$Vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Vcvars -PathType Leaf)) {
    throw "vcvars64.bat not found: $Vcvars (need VS2022 v143 Build Tools; override with -Vcvars). nvcc 12.9 rejects VS18."
}
if (-not (Test-Path -LiteralPath (Join-Path $CudaPath "bin\nvcc.exe") -PathType Leaf)) {
    throw "nvcc not found under $CudaPath (need the CUDA 12.9 toolkit; override with -CudaPath or `$env:CUDA_PATH)."
}

# Repo root is this script's parent's parent.
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$cargo = if ($SkipTests) { "cargo build --workspace --features cuda" } else { "cargo build --workspace --features cuda && cargo test --workspace --features cuda" }

Write-Host "[check-cuda] cap=$ComputeCap  toolkit=$CudaPath"
Write-Host "[check-cuda] $cargo"

# cmd (not pwsh) sources vcvars in-process and runs cargo in the same environment — avoids the MSYS
# `cmd /c` path-mangling gotcha. `call ... && ...` chains so a vcvars/cargo failure propagates.
$inner = "call `"$Vcvars`" && set CUDA_COMPUTE_CAP=$ComputeCap && set `"CUDA_PATH=$CudaPath`" && cd /d `"$repo`" && $cargo"
& cmd /c $inner
$code = $LASTEXITCODE

if ($code -ne 0) {
    Write-Host "[check-cuda] FAILED (exit $code)" -ForegroundColor Red
    exit $code
}
Write-Host "[check-cuda] OK" -ForegroundColor Green
