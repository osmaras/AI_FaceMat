# AI FaceMat - Installer for Windows 10/11
# Run from an elevated PowerShell:  .\install.ps1
# Or right-click -> "Run with PowerShell"

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "       AI FaceMat - Installer for Scratch         " -ForegroundColor Cyan
Write-Host "==================================================`n" -ForegroundColor Cyan

# ────────────────────────────────────────────────────────────
# 1. Ask where to install
# ────────────────────────────────────────────────────────────
$defaultInstall = "V:\PROGRAMING\Scratch-Scripts\AI_FaceMat"
$installDir = Read-Host -Prompt "Install directory [$defaultInstall]"
if ([string]::IsNullOrWhiteSpace($installDir)) { $installDir = $defaultInstall }

# ────────────────────────────────────────────────────────────
# 2. Ask for UV cache location (optional)
# ────────────────────────────────────────────────────────────
Write-Host "`nThe UV cache stores Python environments and model checkpoints." -ForegroundColor DarkGray
Write-Host "On a fast SSD (G:) it speeds up subsequent runs significantly." -ForegroundColor DarkGray
$defaultCache = if (Test-Path "G:\") { "G:\Scratch-AI-matte-cache" } else { $installDir }
$cacheDir = Read-Host -Prompt "UV cache directory [$defaultCache]"
if ([string]::IsNullOrWhiteSpace($cacheDir)) { $cacheDir = $defaultCache }

# ────────────────────────────────────────────────────────────
# 2b. Ask for Scratch REST API port and access key
# ────────────────────────────────────────────────────────────
Write-Host "`nAssimilate Scratch REST API connection." -ForegroundColor DarkGray
$defaultPort = "8080"
$scratchPort = Read-Host -Prompt "Scratch REST API port [$defaultPort]"
if ([string]::IsNullOrWhiteSpace($scratchPort)) { $scratchPort = $defaultPort }

$scratchApiKey = Read-Host -Prompt "Scratch API access key (optional, press Enter to skip)"

# ────────────────────────────────────────────────────────────
# 3. Check / install uv
# ────────────────────────────────────────────────────────────
Write-Host ""
$uvPath = $null

$candidates = @(
    (Get-Command uv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    "$env:USERPROFILE\.local\bin\uv.exe",
    "$env:USERPROFILE\.cargo\bin\uv.exe",
    "$env:LOCALAPPDATA\uv\uv.exe"
) | Where-Object { $_ -and (Test-Path $_) }

if ($candidates) {
    $uvPath = $candidates[0]
    Write-Host "UV found: $uvPath" -ForegroundColor Green
}
else {
    Write-Host "UV not found. Installing..." -ForegroundColor Yellow
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $candidates = @(
            (Get-Command uv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
            "$env:USERPROFILE\.local\bin\uv.exe",
            "$env:USERPROFILE\.cargo\bin\uv.exe",
            "$env:LOCALAPPDATA\uv\uv.exe"
        ) | Where-Object { $_ -and (Test-Path $_) }
        
        if ($candidates) {
            $uvPath = $candidates[0]
            Write-Host "UV installed: $uvPath" -ForegroundColor Green
        }
        else {
            Write-Host "UV installation failed. Install manually from https://docs.astral.sh/uv/" -ForegroundColor Red
            exit 1
        }
    }
    catch {
        Write-Host "Failed to install uv: $_" -ForegroundColor Red
        Write-Host "Install manually: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor DarkGray
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
        $smiOutput = & nvidia-smi --query-gpu="name,compute_cap" --format="csv,noheader" 2>&1
        if ($smiOutput -and $smiOutput -notmatch "error") {
            $parts = ($smiOutput | Select-Object -First 1) -split ','
            if ($parts.Count -ge 2) {
                $gpuName = $parts[0].Trim()
                $gpuComputeCap = $parts[1].Trim()
                $major = [int]($gpuComputeCap.Split('.')[0])

                Write-Host "  GPU: $gpuName (compute capability $gpuComputeCap)" -ForegroundColor White

                if ($major -ge 10) {
                    $cudaVersion = "cu128"
                    $torchVer = "2.7.0"
                    $torchvisionVer = "0.22.0"
                    Write-Host "  -> Blackwell detected, using PyTorch 2.7.0 (cu128)" -ForegroundColor Yellow
                }
                else {
                    Write-Host "  -> Using CUDA 12.4 build (compatible)" -ForegroundColor DarkGray
                }
            }
        }
        else {
            Write-Host "  Could not query GPU, defaulting to CUDA 12.4" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  nvidia-smi not found, defaulting to CUDA 12.4" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  GPU detection failed: $_" -ForegroundColor Yellow
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
Write-Host "`nInstalling to: $installDir" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if ([string]::IsNullOrWhiteSpace($scriptDir)) { $scriptDir = Get-Location }

$files = @(
    "AI_FaceMat.py",
    "scratch_api.py",
    "run_AIFaceMat.bat",
    "Ai_Facemat.acc",
    "README.md"
)

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
        Write-Host "  Not found in source: $file" -ForegroundColor Yellow
    }
}

# ────────────────────────────────────────────────────────────
# 6. Patch run_AIFaceMat.bat with correct paths
# ────────────────────────────────────────────────────────────
Write-Host "`nConfiguring batch file..." -ForegroundColor Cyan

$batPath = Join-Path $installDir "run_AIFaceMat.bat"
$batLines = @(
    '@echo off'
    'SETLOCAL'
    ''
    'chcp 65001 >nul'
    'set "PYTHONUTF8=1"'
    "cd /d `"$installDir`""
    ''
    "set `"UV_CACHE_BASE=$cacheDir`""
    'if not exist "%UV_CACHE_BASE%\.uv-cache" mkdir "%UV_CACHE_BASE%\.uv-cache"'
    'if not exist "%UV_CACHE_BASE%\.uv-tmp" mkdir "%UV_CACHE_BASE%\.uv-tmp"'
    'set "UV_CACHE_DIR=%UV_CACHE_BASE%\.uv-cache"'
    'set "TMP=%UV_CACHE_BASE%\.uv-tmp"'
    'set "TEMP=%UV_CACHE_BASE%\.uv-tmp"'
    'set "UV_LINK_MODE=copy"'
    ''
    "set `"SCRATCH_PORT=$scratchPort`""
    "set `"SCRATCH_API_KEY=$scratchApiKey`""
    ''
    "`"$uvPath`" run ^"
    "   --verbose ^"
    "   --default-index https://pypi.org/simple ^"
    "   --index $pytorchIndex ^"
    "   --index-strategy unsafe-best-match ^"
    "   AI_FaceMat.py %*"
    'pause'
    ''
    'ENDLOCAL'
)
[System.IO.File]::WriteAllLines($batPath, $batLines, [System.Text.Encoding]::UTF8)
Write-Host "  run_AIFaceMat.bat configured" -ForegroundColor Green

# ────────────────────────────────────────────────────────────
# 7. Patch AI_FaceMat.py inline metadata
# ────────────────────────────────────────────────────────────
Write-Host "Patching PyTorch versions in script..." -ForegroundColor Cyan

$scriptDestPath = Join-Path $installDir "AI_FaceMat.py"
if (Test-Path $scriptDestPath) {
    $scriptContent = Get-Content $scriptDestPath -Raw
    $scriptContent = $scriptContent -replace 'torch==[\d.]+\+cu\d+', $torchSpec
    $scriptContent = $scriptContent -replace 'torchvision==[\d.]+\+cu\d+', $torchvisionSpec
    Set-Content -Path $scriptDestPath -Value $scriptContent -Encoding UTF8
    Write-Host "  AI_FaceMat.py patched: $torchSpec, $torchvisionSpec" -ForegroundColor Green
}

# ────────────────────────────────────────────────────────────
# 8. Patch Ai_Facemat.acc with correct cmdline path
# ────────────────────────────────────────────────────────────
Write-Host "Configuring Scratch custom command..." -ForegroundColor Cyan

$accPath = Join-Path $installDir "Ai_Facemat.acc"
$batPathEsc = "$installDir\run_AIFaceMat.bat" -replace '\\', '\\'

$accLines = @(
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!--Generator: Assimilate Product Suite-->'
    '<custom_commands version="9.9">'
    '   <command uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="app" blocking="N" minimize_on_exe="N" player_menu="Y" shot_selection="Y">'
    '       <title>Ai_FaceMat</title>'
    "       <cmdline>$batPathEsc</cmdline>"
    '       <inputs>'
    '           <input uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="3" store="0">'
    '               <label>Processing Mode</label>'
    '               <value>grayscale,color</value>'
    '           </input>'
    '           <input uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="5" store="0">'
    '               <label>Clean up cache</label>'
    '               <value>yes</value>'
    '           </input>'
    '           <input uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="3" store="0">'
    '               <label>Timeline destination</label>'
    '               <value>Add layer, Add version</value>'
    '           </input>'
    '           <input uuid="072206de-9b40-40c3-9ca4-a7ab72d158ab" type="3" store="0">'
    '               <label>SegFace Model</label>'
    '               <value>ConvNeXt CelebA (accurate),SwinB CelebA (fast),SwinV2B CelebA (best),SwinB LaPa (alt dataset)</value>'
    '           </input>'
    '       </inputs>'
    '   </command>'
    '</custom_commands>'
)
[System.IO.File]::WriteAllLines($accPath, $accLines, [System.Text.Encoding]::UTF8)
Write-Host "  Ai_Facemat.acc configured" -ForegroundColor Green

# ────────────────────────────────────────────────────────────
# 9. Create uninstaller
# ────────────────────────────────────────────────────────────
Write-Host "Creating uninstaller..." -ForegroundColor Cyan

$uninstallPath = Join-Path $installDir "uninstall.ps1"
$uninstallLines = @(
    '# AI FaceMat - Uninstaller'
    '$ErrorActionPreference = "Stop"'
    'Write-Host ""'
    'Write-Host "AI FaceMat - Uninstaller" -ForegroundColor Cyan'
    'Write-Host ""'
    ''
    "`$installDir = `"$installDir`""
    "`$cacheDir = `"$cacheDir`""
    ''
    'if (Test-Path $installDir) {'
    '    $confirm = Read-Host "Remove $installDir ? [y/N]"'
    '    if ($confirm -eq "y" -or $confirm -eq "Y") {'
    '        Remove-Item -Path $installDir -Recurse -Force'
    '        Write-Host "Removed: $installDir" -ForegroundColor Green'
    '    }'
    '}'
    ''
    'if (Test-Path $cacheDir) {'
    '    $confirmCache = Read-Host "Remove UV cache at $cacheDir ? [y/N]"'
    '    if ($confirmCache -eq "y" -or $confirmCache -eq "Y") {'
    '        Remove-Item -Path $cacheDir -Recurse -Force'
    '        Write-Host "Removed UV cache: $cacheDir" -ForegroundColor Green'
    '    }'
    '}'
    ''
    'Write-Host ""'
    'Write-Host "Uninstall complete." -ForegroundColor Cyan'
)
[System.IO.File]::WriteAllLines($uninstallPath, $uninstallLines, [System.Text.Encoding]::UTF8)
Write-Host "  uninstall.ps1 created" -ForegroundColor Green

# ────────────────────────────────────────────────────────────
# 10. Summary
# ────────────────────────────────────────────────────────────
Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "              Installation Complete!              " -ForegroundColor Green
Write-Host "==================================================`n" -ForegroundColor Green
Write-Host "  Install dir : $installDir" -ForegroundColor White
Write-Host "  UV cache    : $cacheDir" -ForegroundColor White
Write-Host "  uv binary   : $uvPath" -ForegroundColor White
Write-Host "  GPU         : $gpuName (sm_$gpuComputeCap)" -ForegroundColor White
Write-Host "  PyTorch     : $torchSpec`n" -ForegroundColor White

Write-Host "  To load in Scratch:" -ForegroundColor Cyan
Write-Host "    Menu -> Custom Commands -> Load -> $installDir\Ai_Facemat.acc`n" -ForegroundColor White

Write-Host "  To uninstall:" -ForegroundColor Cyan
Write-Host "    powershell -File `"$installDir\uninstall.ps1`"`n" -ForegroundColor White

# SIG # Begin signature block
# MIIFcQYJKoZIhvcNAQcCoIIFYjCCBV4CAQExCzAJBgUrDgMCGgUAMGkGCisGAQQB
# gjcCAQSgWzBZMDQGCisGAQQBgjcCAR4wJgIDAQAABBAfzDtgWUsITrck0sYpfvNR
# AgEAAgEAAgEAAgEAAgEAMCEwCQYFKw4DAhoFAAQUshnRYIGgK6ApO7yJaqQok0x6
# zZ6gggMSMIIDDjCCAfagAwIBAgIQGfGZsMjlDYxGUSgIyQP9iTANBgkqhkiG9w0B
# AQsFADAUMRIwEAYDVQQDDAlBSUZhY2VNYXQwHhcNMjYwNjE1MTkwODAwWhcNMjcw
# NjE1MTkyODAwWjAUMRIwEAYDVQQDDAlBSUZhY2VNYXQwggEiMA0GCSqGSIb3DQEB
# AQUAA4IBDwAwggEKAoIBAQDY5/wiYR8HzrNtLrgSMFsSOcm98PFAToOfBp4a7zz8
# koQeQMSX/9zSSGiDT3caTS45WadEYbLWJ81alUUsWwhuQL7sYUSWev43KHPwyTyu
# 4a3ezglji5RTAAJpOHmsGSto6hadO7/ZRMxskw/XeA7UyyMftWkGZpBJI6QKOxx5
# fn4OpnV1U+2YYvGQHVxh9BL8qFfXxIw+EEag1CEgtv7sym3lAAAUC5whJSIDfWCm
# r/D71Le4ihr/jeWRTidO7zqGIyFfg+tr1LVCbJmXKV0pDzIlpC3xgGUmw8KxGfLH
# vjsGB9Qq7nCddenB5PO0gaz1wyBxYvcV+93OxS56xjnBAgMBAAGjXDBaMA4GA1Ud
# DwEB/wQEAwIHgDATBgNVHSUEDDAKBggrBgEFBQcDAzAUBgNVHREEDTALgglBSUZh
# Y2VNYXQwHQYDVR0OBBYEFEfa/fz150BOhyvs1ri0LvWpKXpxMA0GCSqGSIb3DQEB
# CwUAA4IBAQBB19+Ra6bdx6BhCrG34axKyKt+E1SyK5JtBNdpg1LhB0SDfd8kVpHE
# i/b1MBtdfpPGHz7/quRULfUYe1V1mFq+5iIebPIRXLm/qR0yqKkVwe7pVbfBg7PO
# rJ0ViY5emOTN2B0nqEQ84bTGZsZGpupbSWZm6/nzZowmuGNfn/663g26rsbU1EHx
# HVKnA9vjbLrRlLmm8IO3W6gflSxeQzmVleTnOjBaDNydh9B9vmwwdPvJpD2lWC9Z
# Uod1WN5JH+EuJ3UjuiepLXN4aiYoNrrN2c8T98YejW8xMTQeSdmgAsTZAdu9xH3i
# DJQyukDtathgw4kZl3OnR2bjNxtJWpLUMYIByTCCAcUCAQEwKDAUMRIwEAYDVQQD
# DAlBSUZhY2VNYXQCEBnxmbDI5Q2MRlEoCMkD/YkwCQYFKw4DAhoFAKB4MBgGCisG
# AQQBgjcCAQwxCjAIoAKAAKECgAAwGQYJKoZIhvcNAQkDMQwGCisGAQQBgjcCAQQw
# HAYKKwYBBAGCNwIBCzEOMAwGCisGAQQBgjcCARUwIwYJKoZIhvcNAQkEMRYEFG/V
# N19GKN5ol2jLlzJpgJ9buJCZMA0GCSqGSIb3DQEBAQUABIIBAHkzn4c2jJjyFWjZ
# qdUzN+lcYOiI9jBnhdrv/Jf35nWywMa8+ehe+e7CrIhAdaP5T7wy6vwv94WnZ2RE
# T2OwWOVvCf2VV4UcoHZfi64k7a42tIuVUXiOk8F3xR5MDBm8qP4M3SRycNziJRvR
# WQv93OCZTT6/3i3bQLWh3Jws/zzoXsfeMVpZwcgBoO8hzZr534d1XFmz9nyljf5+
# kuII6BkUjVA1hC/d8RXabz/QfK19cD45aoS0Gs8i0U/kOHK8+gVuFoBTzM+oZDKH
# +OxSTbSHTDa0fUXISMUbhywRPUScWZ09sdv4r1HUELFnn5Dm2jry5dY5992fs41J
# tTVTGrI=
# SIG # End signature block
