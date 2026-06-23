<#
.SYNOPSIS
  Bundle a CUDA-built candle-gen binary with the CUDA 12.9 runtime redistributable DLLs into a
  self-contained dist/ folder — the "fat binary, like torch" distribution model (sc-3676).

.DESCRIPTION
  The worker is built at the baseline virtual arch CUDA_COMPUTE_CAP=80 (see README "Packaging"), and
  ONE binary runs on every NVIDIA arch >= sm_80 (Ampere -> Ada -> Hopper -> Blackwell). candle's two
  kernel families get there differently: the DENSE kernels embed `compute_80` PTX that the driver
  JIT-compiles to the runtime GPU's SASS, while the QUANTIZED + MoE kernels (GGUF QMatMul) ship a
  multi-arch fatbin with native sm_80/sm_90/sm_120 SASS + forward PTX (sc-7544, via vendor/candle-
  kernels — a single-arch sm_80-only build silently no-ops the quant matmuls on Blackwell). What the
  target machine still needs at runtime is the CUDA runtime libraries (cudart/cublas/cublasLt/curand/
  nvrtc). Rather than require a CUDA Toolkit install on every target, we copy the redistributable DLLs
  next to the binary (cudarc links them dynamically, found via the exe's directory on the default DLL
  search path).

  This does NOT bundle the NVIDIA *driver* — the user must have a driver new enough for the CUDA 12.9
  runtime: Windows >= 576.02 (CUDA 12.9 GA). The driver is what JIT-compiles the PTX and provides
  libcuda; it is never redistributable.

.PARAMETER BinaryPath
  Path to the CUDA-built .exe to package (e.g. the SceneWorks worker, or this repo's txt2img example
  at target\release\examples\sdxl-txt2img.exe).

.PARAMETER OutDir
  Output directory for the bundle. Created if absent; existing contents are NOT cleared. Default: dist

.PARAMETER CudaPath
  CUDA Toolkit root holding bin\*.dll. Default: $env:CUDA_PATH, else the v12.9 default install path.

.EXAMPLE
  pwsh scripts/package-cuda.ps1 -BinaryPath target\release\examples\sdxl-txt2img.exe
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BinaryPath,

    [string]$OutDir = "dist",

    [string]$CudaPath = $(if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" })
)

$ErrorActionPreference = "Stop"

# Minimum NVIDIA driver for the bundled CUDA 12.9 runtime (Windows). Kept here and in README in sync.
$MinDriverWindows = "576.02"

# Redist DLLs the worker needs at runtime. cudarc dynamic-links these; they ship in the CUDA Toolkit's
# bin\ and are redistributable (unlike the driver). Patterns tolerate the per-minor-version filename
# suffixes (cudart64_12, curand64_10, nvrtc64_120_0, nvrtc-builtins64_129, ...). The nvrtc *_0.alt.dll
# is a driver-fallback variant we deliberately skip to keep the bundle lean.
$DllPatterns = @(
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "curand64_*.dll",
    "nvrtc64_*_0.dll",
    "nvrtc-builtins64_*.dll"
)

if (-not (Test-Path -LiteralPath $BinaryPath -PathType Leaf)) {
    throw "Binary not found: $BinaryPath"
}
$cudaBin = Join-Path $CudaPath "bin"
if (-not (Test-Path -LiteralPath $cudaBin -PathType Container)) {
    throw "CUDA bin\ not found: $cudaBin (set -CudaPath or `$env:CUDA_PATH to the CUDA 12.9 toolkit root)"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Copy the worker binary.
$binName = Split-Path -Leaf $BinaryPath
Copy-Item -LiteralPath $BinaryPath -Destination (Join-Path $OutDir $binName) -Force
Write-Host "[package-cuda] worker : $binName"

# Resolve + copy each redist DLL. Exclude the .alt fallback from the nvrtc glob explicitly.
$copied = New-Object System.Collections.Generic.List[string]
foreach ($pattern in $DllPatterns) {
    $matches = Get-ChildItem -LiteralPath $cudaBin -Filter $pattern -File |
        Where-Object { $_.Name -notlike "*.alt.dll" }
    if ($matches.Count -eq 0) {
        throw "Required redist DLL '$pattern' not found in $cudaBin. Is this a CUDA 12.9 toolkit install?"
    }
    foreach ($dll in $matches) {
        Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $OutDir $dll.Name) -Force
        $copied.Add($dll.Name)
        Write-Host "[package-cuda] redist : $($dll.Name)"
    }
}

# Drop a RUNTIME.txt manifest documenting the driver floor next to the bundle.
$manifest = @"
candle-gen CUDA bundle (sc-3676, sc-7544)
=========================================
Built at baseline virtual arch CUDA_COMPUTE_CAP=80, this single bundle runs on any NVIDIA GPU of
compute capability 8.0 (Ampere) or newer, through 12.x (Blackwell). Dense kernels embed compute_80
PTX that the NVIDIA driver JIT-compiles to your GPU's native SASS at first load; the quantized + MoE
kernels ship a multi-arch fatbin with native sm_80/sm_90/sm_120 SASS + forward PTX (sc-7544), so Q4/Q8
models run correctly on Blackwell sm_120 instead of silently producing black/NaN output.

Requirements on the target machine:
  * NVIDIA GPU, compute capability >= 8.0 (Ampere / RTX 30-series or newer).
  * NVIDIA driver >= $MinDriverWindows (Windows) for the bundled CUDA 12.9 runtime.
The CUDA runtime DLLs are bundled here; do NOT install a separate CUDA Toolkit. The driver is NOT
bundled (it is not redistributable) — install/update it from nvidia.com if older than the above.

First run is slower while the driver JIT-compiles + caches the dense PTX (per-GPU, under
%APPDATA%\NVIDIA\ComputeCache). Subsequent runs load the cached SASS. The quant kernels' native cubin
is used directly on sm_80/sm_90/sm_120 (no JIT).

Bundled redist DLLs:
$($copied | ForEach-Object { "  - $_" } | Out-String)
"@
$manifest | Set-Content -Path (Join-Path $OutDir "RUNTIME.txt") -Encoding utf8

Write-Host ""
Write-Host "[package-cuda] wrote bundle -> $OutDir  ($($copied.Count) DLLs + $binName + RUNTIME.txt)"
Write-Host "[package-cuda] min driver (Windows): >= $MinDriverWindows"
