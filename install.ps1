# AI FaceMat — Installer for Windows 10/11
# Run from an elevated PowerShell:  .\install.ps1
# Or right-click → "Run with PowerShell"

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       AI FaceMat — Installer for Scratch         ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ────────────────────────────────────────────────────────────
# 1. Ask where to install
# ────────────────────────────────────────────────────────────
$defaultInstall = "V:\PROGRAMING\Scratch-Scripts\AI_FaceMat"
$installDir = Read-Host "Install directory [$defaultInstall]"
if ([string]::IsNullOrWhiteSpace($installDir)) { $installDir = $defaultInstall }

# ────────────────────────────────────────────────────────────
# 2. Ask for UV cache location (optional)
# ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "The UV cache stores Python environments and model checkpoints." -ForegroundColor DarkGray
Write-Host "On a fast SSD (G:) it speeds up subsequent runs significantly." -ForegroundColor DarkGray
$defaultCache = if (Test-Path "G:\") { "G:\Scratch-AI-matte-cache" } else { $installDir }
$cacheDir = Read-Host "UV cache directory [$defaultCache]"
if ([string]::IsNullOrWhiteSpace($cacheDir)) { $cacheDir = $defaultCache }

# ────────────────────────────────────────────────────────────
# 2b. Ask for Scratch REST API port and access key
# ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Assimilate Scratch REST API connection." -ForegroundColor DarkGray
$defaultPort = "8080"
$scratchPort = Read-Host "Scratch REST API port [$defaultPort]"
if ([string]::IsNullOrWhiteSpace($scratchPort)) { $scratchPort = $defaultPort }

$scratchApiKey = Read-Host "Scratch API access key (optional, press Enter to skip)"

# ────────────────────────────────────────────────────────────
# 3. Check / install uv
# ────────────────────────────────────────────────────────────
Write-Host ""
$uvPath = $null

# Check common locations
$candidates = @(
    (Get-Command uv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    "$env:USERPROFILE\.local\bin\uv.exe",
    "$env:USERPROFILE\.cargo\bin\uv.exe",
    "$env:LOCALAPPDATA\uv\uv.exe"
) | Where-Object { $_ -and (Test-Path $_) }

if ($candidates) {
    $uvPath = $candidates[0]
    Write-Host "✓ uv found: $uvPath" -ForegroundColor Green
}
else {
    Write-Host "uv not found. Installing..." -ForegroundColor Yellow
    try {
        irm https://astral.sh/uv/install.ps1 | iex
        # Re-check after install
        $candidates = @(
            (Get-Command uv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
            "$env:USERPROFILE\.local\bin\uv.exe",
            "$env:USERPROFILE\.cargo\bin\uv.exe"
        ) | Where-Object { $_ -and (Test-Path $_) }
        if ($candidates) {
            $uvPath = $candidates[0]
            Write-Host "✓ uv installed: $uvPath" -ForegroundColor Green
        }
        else {
            Write-Host "✗ uv installation failed. Install manually from https://docs.astral.sh/uv/" -ForegroundColor Red
            exit 1
        }
    }
    catch {
        Write-Host "✗ Failed to install uv: $_" -ForegroundColor Red
        Write-Host "  Install manually: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor DarkGray
        exit 1
    }
}

# ────────────────────────────────────────────────────────────
# 4. Detect GPU and select CUDA / PyTorch versions
# ────────────────────────────────────────────────────────────
Write-Host ""
$cudaVersion = "cu124"
$torchVer = "2.5.1"
$torchvisionVer = "0.20.1"
$gpuName = "Unknown"
$gpuComputeCap = "0.0"

try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $smiOutput = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1
        if ($smiOutput -and $smiOutput -notmatch "error") {
            $parts = ($smiOutput | Select-Object -First 1) -split ","
            $gpuName = $parts[0].Trim()
            $gpuComputeCap = $parts[1].Trim()
            $major = [int]($gpuComputeCap.Split(".")[0])

            Write-Host "  GPU: $gpuName (compute capability $gpuComputeCap)" -ForegroundColor White

            if ($major -ge 10) {
                # Blackwell (sm_100+) requires CUDA 12.8+
                $cudaVersion = "cu128"
                $torchVer = "2.6.0"
                $torchvisionVer = "0.21.0"
                Write-Host "  → Blackwell detected, using CUDA 12.8 build" -ForegroundColor Yellow
            }
            else {
                Write-Host "  → Using CUDA 12.4 build (compatible)" -ForegroundColor DarkGray
            }
        }
        else {
            Write-Host "  ⚠ Could not query GPU, defaulting to CUDA 12.4" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  ⚠ nvidia-smi not found, defaulting to CUDA 12.4" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  ⚠ GPU detection failed: $_" -ForegroundColor Yellow
    Write-Host "  Defaulting to CUDA 12.4" -ForegroundColor DarkGray
}

$torchSpec = "torch==$torchVer+$cudaVersion"
$torchvisionSpec = "torchvision==$torchvisionVer+$cudaVersion"
$pytorchIndex = "https://download.pytorch.org/whl/$cudaVersion"

Write-Host "  PyTorch: $torchSpec" -ForegroundColor White
Write-Host "  Index:   $pytorchIndex" -ForegroundColor White

# ────────────────────────────────────────────────────────────
# 5. Copy files to install directory
# ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Installing to: $installDir" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if ([string]::IsNullOrWhiteSpace($scriptDir)) { $scriptDir = Get-Location }

# Files to copy
$files = @(
    "AI_FaceMat.py",
    "scratch_api.py",
    "run_AIFaceMat.bat",
    "Ai_Facemat.acc",
    "README.md"
)

# Create install dir
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    Write-Host "  Created: $installDir"
}

foreach ($file in $files) {
    $src = Join-Path $scriptDir $file
    $dst = Join-Path $installDir $file
    if (Test-Path $src) {
        Copy-Item $src $dst -Force
        Write-Host "  Copied: $file" -ForegroundColor DarkGray
    }
    else {
        Write-Host "  ⚠ Not found in source: $file" -ForegroundColor Yellow
    }
}

# ────────────────────────────────────────────────────────────
# 5. Patch run_AIFaceMat.bat with correct paths
# ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Configuring batch file..." -ForegroundColor Cyan

$batPath = Join-Path $installDir "run_AIFaceMat.bat"
$batContent = @"
@echo off
SETLOCAL

chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "$installDir"

set "UV_CACHE_BASE=$cacheDir"
if not exist "%UV_CACHE_BASE%\.uv-cache" mkdir "%UV_CACHE_BASE%\.uv-cache"
if not exist "%UV_CACHE_BASE%\.uv-tmp" mkdir "%UV_CACHE_BASE%\.uv-tmp"
set "UV_CACHE_DIR=%UV_CACHE_BASE%\.uv-cache"
set "TMP=%UV_CACHE_BASE%\.uv-tmp"
set "TEMP=%UV_CACHE_BASE%\.uv-tmp"
set "UV_LINK_MODE=copy"

set "SCRATCH_PORT=$scratchPort"
set "SCRATCH_API_KEY=$scratchApiKey"

"$uvPath" run ^
	--verbose ^
	--default-index https://pypi.org/simple ^
	--index $pytorchIndex ^
	--index-strategy unsafe-best-match ^
	AI_FaceMat.py %*
pause

ENDLOCAL
"@
Set-Content -Path $batPath -Value $batContent -Encoding UTF8
Write-Host "  ✓ run_AIFaceMat.bat configured" -ForegroundColor Green

# ────────────────────────────────────────────────────────────
# 6. Patch AI_FaceMat.py inline metadata with correct torch/torchvision
# ────────────────────────────────────────────────────────────
Write-Host "Patching PyTorch versions in script..." -ForegroundColor Cyan

$scriptPath = Join-Path $installDir "AI_FaceMat.py"
if (Test-Path $scriptPath) {
    $scriptContent = Get-Content $scriptPath -Raw
    # Replace pinned torch version
    $scriptContent = $scriptContent -replace 'torch==[\d.]+\+cu\d+', $torchSpec
    # Replace pinned torchvision version
    $scriptContent = $scriptContent -replace 'torchvision==[\d.]+\+cu\d+', $torchvisionSpec
    Set-Content -Path $scriptPath -Value $scriptContent -Encoding UTF8
    Write-Host "  ✓ AI_FaceMat.py patched: $torchSpec, $torchvisionSpec" -ForegroundColor Green
}

# ────────────────────────────────────────────────────────────
# 7. Patch Ai_Facemat.acc with correct cmdline path
# ────────────────────────────────────────────────────────────
Write-Host "Configuring Scratch custom command..." -ForegroundColor Cyan

$accPath = Join-Path $installDir "Ai_Facemat.acc"
$batPathEsc = "$installDir\run_AIFaceMat.bat" -replace '\\', '\\'

$accContent = @"
<?xml version="1.0" encoding="UTF-8"?>
<!--Generator: Assimilate Product Suite-->
<custom_commands version="9.9">
	<command uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="app" blocking="N" minimize_on_exe="N" player_menu="Y" shot_selection="Y">
		<title>Ai_FaceMat</title>
		<cmdline>$installDir\run_AIFaceMat.bat</cmdline>
		<inputs>
			<input uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="3" store="0">
				<label>Processing Mode</label>
				<value>grayscale,color</value>
			</input>
			<input uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="5" store="0">
				<label>Clean up cache</label>
				<value>yes</value>
			</input>
			<input uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="3" store="0">
				<label>Timeline destination</label>
				<value>Add layer, Add version</value>
			</input>
			<input uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="3" store="0">
				<label>SegFace Model</label>
				<value>ConvNeXt CelebA (accurate),SwinB CelebA (fast),SwinV2B CelebA (best),SwinB LaPa (alt dataset)</value>
			</input>
		</inputs>
	</command>
</custom_commands>
"@
Set-Content -Path $accPath -Value $accContent -Encoding UTF8
Write-Host "  ✓ Ai_Facemat.acc configured" -ForegroundColor Green

# ────────────────────────────────────────────────────────────
# 8. Create uninstaller
# ────────────────────────────────────────────────────────────
Write-Host "Creating uninstaller..." -ForegroundColor Cyan

$uninstallPath = Join-Path $installDir "uninstall.ps1"
$uninstallContent = @"
# AI FaceMat — Uninstaller
`$ErrorActionPreference = "Stop"
Write-Host ""
Write-Host "AI FaceMat — Uninstaller" -ForegroundColor Cyan
Write-Host ""

`$installDir = "$installDir"
`$cacheDir = "$cacheDir"

# Remove install directory
if (Test-Path `$installDir) {
    `$confirm = Read-Host "Remove `$installDir ? [y/N]"
    if (`$confirm -eq 'y' -or `$confirm -eq 'Y') {
        Remove-Item -Path `$installDir -Recurse -Force
        Write-Host "✓ Removed: `$installDir" -ForegroundColor Green
    }
    else {
        Write-Host "Skipped install directory removal." -ForegroundColor Yellow
    }
}

# Optionally remove UV cache
if (Test-Path `$cacheDir) {
    `$confirmCache = Read-Host "Remove UV cache at `$cacheDir ? (contains envs + model checkpoints) [y/N]"
    if (`$confirmCache -eq 'y' -or `$confirmCache -eq 'Y') {
        Remove-Item -Path `$cacheDir -Recurse -Force
        Write-Host "✓ Removed UV cache: `$cacheDir" -ForegroundColor Green
    }
    else {
        Write-Host "Skipped UV cache removal." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Uninstall complete." -ForegroundColor Cyan
Write-Host "Note: Model checkpoints in ~/.cache/huggingface/hub/ were preserved." -ForegroundColor DarkGray
"@
Set-Content -Path $uninstallPath -Value $uninstallContent -Encoding UTF8
Write-Host "  ✓ uninstall.ps1 created" -ForegroundColor Green

# ────────────────────────────────────────────────────────────
# 8. Summary
# ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║            Installation Complete!                 ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  Install dir : $installDir" -ForegroundColor White
Write-Host "  UV cache    : $cacheDir" -ForegroundColor White
Write-Host "  uv binary   : $uvPath" -ForegroundColor White
Write-Host "  GPU         : $gpuName (sm_$gpuComputeCap)" -ForegroundColor White
Write-Host "  PyTorch     : $torchSpec" -ForegroundColor White
Write-Host ""
Write-Host "  To load in Scratch:" -ForegroundColor Cyan
Write-Host "    Menu → Custom Commands → Load → $installDir\Ai_Facemat.acc" -ForegroundColor White
Write-Host ""
Write-Host "  To uninstall:" -ForegroundColor Cyan
Write-Host "    powershell -File $installDir\uninstall.ps1" -ForegroundColor White
Write-Host ""
