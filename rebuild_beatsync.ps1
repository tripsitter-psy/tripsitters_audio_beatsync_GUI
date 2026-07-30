# rebuild_beatsync.ps1
# Full rebuild of BeatSync backend for Windows

param(
    [switch]$GPU,
    [switch]$AudioFlux,
    [switch]$Deploy,
    [switch]$CPUOnly  # Force CPU-only build, skip GPU
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== BeatSync Backend Rebuild ===" -ForegroundColor Cyan

# Get script directory
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ProjectRoot "build"

Write-Host "Project root: $ProjectRoot" -ForegroundColor Gray
Write-Host "Build dir: $BuildDir" -ForegroundColor Gray

# Step 1: Clean old build files
Write-Host "`n[1/4] Cleaning old CMake cache..." -ForegroundColor Yellow
if (Test-Path $BuildDir) {
    # Try to remove build directory; if locked, clear contents instead
    try {
        Write-Host "  Removing entire build directory..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $BuildDir -ErrorAction Stop
        Write-Host "  Build directory removed" -ForegroundColor Green
    } catch {
        Write-Host "  Directory locked, clearing contents instead..." -ForegroundColor Yellow
        Get-ChildItem -Path $BuildDir -Recurse -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
        Write-Host "  Build directory contents cleared" -ForegroundColor Green
    }
} else {
    Write-Host "  Build directory doesn't exist yet" -ForegroundColor Gray
}

# Ensure build directory exists
if (-not (Test-Path $BuildDir)) {
    mkdir $BuildDir | Out-Null
    Write-Host "  Created build directory" -ForegroundColor Green
}

# Also clean vcpkg buildtrees to remove stale nvidia-cutlass artifacts
$VcpkgBuildtrees = Join-Path $ProjectRoot "vcpkg\buildtrees"
if (Test-Path "$VcpkgBuildtrees\nvidia-cutlass") {
    Remove-Item -Recurse -Force "$VcpkgBuildtrees\nvidia-cutlass" -ErrorAction SilentlyContinue
    Write-Host "  Removed nvidia-cutlass build artifacts" -ForegroundColor Green
}

# Step 2: Configure with CMake
Write-Host "`n[2/4] Configuring with CMake..." -ForegroundColor Yellow

$ConfigArgs = @(
    "-S", ".",
    "-B", "build",
    "-DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake",
    "-DTRIPSITTER_EXE_PATH=D:/UnrealEngine/Engine/Binaries/Win64/TripSitter.exe",
    "-DVCPKG_MANIFEST_FEATURES=cpu"
)

if ($GPU -and -not $CPUOnly) {
    $ConfigArgs += "-DVCPKG_OVERLAY_TRIPLETS=triplets"
    # Remove cpu feature, add gpu feature
    $ConfigArgs = $ConfigArgs | Where-Object { $_ -ne "-DVCPKG_MANIFEST_FEATURES=cpu" }
    $ConfigArgs += "-DVCPKG_MANIFEST_FEATURES=gpu"
    Write-Host "  GPU acceleration enabled" -ForegroundColor Cyan
} else {
    Write-Host "  CPU-only build (use -GPU to enable CUDA/TensorRT support)" -ForegroundColor Cyan
}

if ($AudioFlux) {
    $ConfigArgs += "-DAUDIOFLUX_ROOT=C:/audioFlux"
    Write-Host "  AudioFlux enabled" -ForegroundColor Cyan
}

cd $ProjectRoot
try {
    cmake @ConfigArgs 2>&1 | Tee-Object -Variable ConfigOutput | ForEach-Object {
        if ($_ -match "error|Error|ERROR") {
            Write-Host $_ -ForegroundColor Red
        } elseif ($_ -match "warning|Warning|WARNING") {
            Write-Host $_ -ForegroundColor Yellow
        } else {
            Write-Host $_
        }
    }
    Write-Host "  Configuration successful" -ForegroundColor Green
} catch {
    Write-Host "  Configuration FAILED: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Build
Write-Host "`n[3/4] Building beatsync_backend_shared..." -ForegroundColor Yellow

$BuildArgs = @(
    "--build", "build",
    "--config", "Release",
    "--target", "beatsync_backend_shared"
)

try {
    cmake @BuildArgs 2>&1 | Tee-Object -Variable BuildOutput | ForEach-Object {
        if ($_ -match "error|Error|ERROR|failed") {
            Write-Host $_ -ForegroundColor Red
        } elseif ($_ -match "warning|Warning|WARNING") {
            Write-Host $_ -ForegroundColor Yellow
        } else {
            Write-Host $_
        }
    }
    
    # Check if build succeeded
    if (Test-Path "$BuildDir\bin\Release\beatsync_backend_shared.dll") {
        $DllSize = [math]::Round((Get-Item "$BuildDir\bin\Release\beatsync_backend_shared.dll").Length / 1MB, 2)
        Write-Host "  Build successful ($DllSize MB)" -ForegroundColor Green
    } else {
        Write-Host "  Build FAILED: DLL not found at expected location" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  Build FAILED: $_" -ForegroundColor Red
    exit 1
}

# Step 4: Deploy (optional)
if ($Deploy) {
    Write-Host "`n[4/4] Deploying DLLs to TripSitter..." -ForegroundColor Yellow
    
    $DeployScript = Join-Path $ProjectRoot "scripts\deploy_tripsitter.ps1"
    if (Test-Path $DeployScript) {
        try {
            & $DeployScript
            Write-Host "  Deployment successful" -ForegroundColor Green
        } catch {
            Write-Host "  Deployment FAILED: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  Deploy script not found: $DeployScript" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[4/4] Skipping deployment (use -Deploy flag to auto-deploy)" -ForegroundColor Gray
}

Write-Host "`n=== Rebuild Complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Verify DLLs: .\scripts\deploy_tripsitter.ps1 -Verify" -ForegroundColor Gray
Write-Host "  2. Deploy DLLs: .\scripts\deploy_tripsitter.ps1" -ForegroundColor Gray
Write-Host "  3. Run TripSitter: D:\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe" -ForegroundColor Gray
