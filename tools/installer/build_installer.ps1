<#
.SYNOPSIS
  Build the Baja MV Windows installer: dist\BajaMV-Setup-<version>.exe

.DESCRIPTION
  Run on a Windows PC that has the real models in deploy\models (they are not in git). Steps:
    1. download (or take from build\cache) the pinned Python runtime and ffmpeg, checking their sha256 (pins.json)
    2. unpack the runtime into build\stage\python and install the locked, hashed packages into it
       (requirements-runtime.txt), plus the Visual C++ runtime DLLs next to python.exe (ONNX Runtime and OpenVINO
       need them; a fresh Windows may not have them)
    3. put ffmpeg.exe / ffprobe.exe into build\stage\ffmpeg\bin
    4. stage the code + models (stage_app.py: checks every model against deploy\models\manifest.json)
    5. compile BajaMV.iss with Inno Setup (winget install JRSoftware.InnoSetup)
  Re-runnable: the stage is rebuilt from scratch each time; downloads are cached in build\cache.

.PARAMETER ModelsDir
  Folder with car_n.onnx, car_s.onnx, panel.onnx, recog.onnx (and optionally yolo11n_coco.onnx).
  Default: <repo>\deploy\models

.PARAMETER AllowDummyModels
  Accept models whose sha256 does not match the manifest (CI's stand-in models). Never for a real installer.

.PARAMETER FfmpegZip
  Use this local ffmpeg zip (or a folder with bin\ffmpeg.exe) instead of the pinned download.

.PARAMETER Iscc
  Path to ISCC.exe if Inno Setup is not in its usual place.

.PARAMETER SkipCompile
  Stop after staging (build\stage), without Inno Setup.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\installer\build_installer.ps1
#>
[CmdletBinding()]
param(
    [string]$ModelsDir = "",
    [switch]$AllowDummyModels,
    [string]$FfmpegZip = "",
    [string]$Iscc = "",
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"      # Invoke-WebRequest is many times faster without the progress bar

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $ModelsDir) { $ModelsDir = Join-Path $Repo "deploy\models" }
$ModelsDir = (Resolve-Path $ModelsDir).Path
$Build = Join-Path $Repo "build"
$Cache = Join-Path $Build "cache"
$Stage = Join-Path $Build "stage"
$Dist = Join-Path $Repo "dist"
$Pins = Get-Content (Join-Path $PSScriptRoot "pins.json") -Raw | ConvertFrom-Json

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    # Run a program and stop the build if it fails (PowerShell does not do that by itself for .exe files).
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

function Get-VerifiedDownload([string]$Url, [string]$Sha256, [string]$FileName) {
    # Download once into build\cache; any file whose sha256 differs from the pin is refused.
    New-Item -ItemType Directory -Force $Cache | Out-Null
    $target = Join-Path $Cache $FileName
    if (Test-Path $target) {
        if ((Get-FileHash $target -Algorithm SHA256).Hash -eq $Sha256) {
            Write-Host "cached: $FileName"
            return $target
        }
        Remove-Item $target
    }
    Write-Host "downloading $Url"
    $partial = "$target.part"
    Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
    $actual = (Get-FileHash $partial -Algorithm SHA256).Hash
    if ($actual -ne $Sha256) {
        Remove-Item $partial
        throw "sha256 mismatch for $FileName`n  expected $Sha256`n  got      $actual"
    }
    Move-Item $partial $target
    return $target
}

function Find-Iscc {
    if ($Iscc) { return $Iscc }
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    )
    foreach ($candidate in $candidates) { if ($candidate -and (Test-Path $candidate)) { return $candidate } }
    $onPath = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw "Inno Setup not found. Install it with: winget install JRSoftware.InnoSetup (or pass -Iscc <path to ISCC.exe>)"
}

# ------------------------------------------------------------------------------------------------ 1. clean stage
Write-Step "Stage folder $Stage"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force $Stage | Out-Null

# ------------------------------------------------------------------------------------------------ 2. Python
Write-Step "Python $($Pins.python.version) (python-build-standalone)"
$pyArchive = Get-VerifiedDownload $Pins.python.url $Pins.python.sha256 $Pins.python.file
Invoke-Native "tar.exe" @("-xzf", $pyArchive, "-C", $Stage)          # the archive's top folder is python\
$Py = Join-Path $Stage "python\python.exe"
if (-not (Test-Path $Py)) { throw "python.exe not found after unpacking $pyArchive" }
& $Py -m pip --version
if ($LASTEXITCODE -ne 0) { Invoke-Native $Py @("-m", "ensurepip", "--default-pip") }

Write-Step "Runtime packages (requirements-runtime.txt, hashes required)"
Invoke-Native $Py @("-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location",
    "--require-hashes", "--only-binary=:all:", "--cache-dir", (Join-Path $Cache "pip"),
    "-r", (Join-Path $PSScriptRoot "requirements-runtime.txt"))

Write-Step "Visual C++ runtime DLLs next to python.exe (app-local)"
$vcDlls = @("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
            "msvcp140_codecvt_ids.dll", "concrt140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
foreach ($dll in $vcDlls) {
    $source = Join-Path $env:SystemRoot "System32\$dll"
    $target = Join-Path $Stage "python\$dll"
    if (Test-Path $target) { continue }                    # python's own copy wins
    if (Test-Path $source) { Copy-Item $source $target; Write-Host "copied $dll" }
    else { Write-Warning "$dll not found in System32: install the VC++ 2015-2022 x64 redistributable on this PC" }
}

# ------------------------------------------------------------------------------------------------ 3. ffmpeg
Write-Step "ffmpeg"
$ffmpegBin = Join-Path $Stage "ffmpeg\bin"
New-Item -ItemType Directory -Force $ffmpegBin | Out-Null
$ffmpegVersion = $Pins.ffmpeg.version
if ($FfmpegZip -and (Test-Path $FfmpegZip -PathType Container)) {
    $ffmpegRoot = (Resolve-Path $FfmpegZip).Path
    $ffmpegVersion = "local folder $ffmpegRoot"
} else {
    if ($FfmpegZip) {
        $zip = (Resolve-Path $FfmpegZip).Path
        $ffmpegVersion = "local zip $(Split-Path $zip -Leaf)"
    } else {
        $zip = Get-VerifiedDownload $Pins.ffmpeg.url $Pins.ffmpeg.sha256 $Pins.ffmpeg.file
    }
    $extract = Join-Path $Build "ffmpeg-extract"
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    New-Item -ItemType Directory -Force $extract | Out-Null
    Invoke-Native "tar.exe" @("-xf", $zip, "-C", $extract)
    $found = Get-ChildItem $extract -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $found) { throw "no ffmpeg.exe inside $zip" }
    $ffmpegRoot = Split-Path (Split-Path $found.FullName -Parent) -Parent
}
foreach ($exe in @("ffmpeg.exe", "ffprobe.exe")) {
    $source = Get-ChildItem $ffmpegRoot -Recurse -Filter $exe | Select-Object -First 1
    if (-not $source) { throw "$exe not found under $ffmpegRoot" }
    Copy-Item $source.FullName $ffmpegBin
}
Get-ChildItem $ffmpegRoot -Filter "LICENSE*" | ForEach-Object { Copy-Item $_.FullName (Join-Path $Stage "ffmpeg") }
Invoke-Native (Join-Path $ffmpegBin "ffmpeg.exe") @("-hide_banner", "-version")

# ------------------------------------------------------------------------------------------------ 4. app + models
Write-Step "App code and models (models from $ModelsDir)"
$stageArgs = @((Join-Path $PSScriptRoot "stage_app.py"), "--repo", $Repo, "--stage", $Stage, "--models", $ModelsDir,
               "--python-dir", (Join-Path $Stage "python"), "--ffmpeg-version", $ffmpegVersion)
if ($AllowDummyModels) { $stageArgs += "--allow-dummy-models" }
$stageOutput = & $Py @stageArgs
$stageCode = $LASTEXITCODE
$stageOutput | ForEach-Object { Write-Host $_ }
if ($stageCode -ne 0) { throw "staging failed (see above)" }
$Version = ($stageOutput | Where-Object { $_ -like "VERSION=*" } | Select-Object -Last 1).Substring(8)
Copy-Item (Join-Path $PSScriptRoot "BajaMV.cmd") $Stage
# bytecode now, so the read-only Program Files copy never needs to compile (hash-based: install time stamps don't matter)
Invoke-Native $Py @("-m", "compileall", "-q", "--invalidation-mode", "unchecked-hash", (Join-Path $Stage "app"))

if ($SkipCompile) {
    Write-Step "Staged (version $Version); -SkipCompile: no installer built"
    exit 0
}

# ------------------------------------------------------------------------------------------------ 5. Inno Setup
Write-Step "Inno Setup"
$isccExe = Find-Iscc
New-Item -ItemType Directory -Force $Dist | Out-Null
Invoke-Native $isccExe @("/Q", "/DAppVersion=$Version", "/DStageDir=$Stage", "/DOutputDir=$Dist",
                          (Join-Path $PSScriptRoot "BajaMV.iss"))
$setup = Join-Path $Dist "BajaMV-Setup-$Version.exe"
if (-not (Test-Path $setup)) { throw "Inno Setup did not produce $setup" }
Write-Step "Done: $setup ($([math]::Round((Get-Item $setup).Length / 1MB)) MB)"
