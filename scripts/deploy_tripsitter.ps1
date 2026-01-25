# deploy_tripsitter.ps1
# Deploys all required DLLs to TripSitter.exe directory with verification
#
# IMPORTANT: This script must be run AFTER building beatsync_backend_shared
# Run from BeatSyncEditor root directory

param(
    [string]$UEPath = "C:\UE5_Source\UnrealEngine",
    [switch]$Verify,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$UE_BIN = "$UEPath\Engine\Binaries\Win64"
$BUILD_RELEASE = "build\Release"
$VCPKG_BIN = "build\vcpkg_installed\x64-windows\bin"
$THIRDPARTY = "unreal-prototype\ThirdParty\beatsync\lib\x64"
$AUDIOFLUX = "C:\audioFlux\build\windowBuild\Release"

# Define expected DLLs with minimum sizes (to catch wrong versions)
$RequiredDLLs = @{
    # From build/Release (Project Output)
    "beatsync_backend_shared.dll" = @{ Source = $BUILD_RELEASE; MinSize = 300KB }

    # From vcpkg bin (ONNX and dependencies - explicit path to avoid build/Release mixing)
    "onnxruntime.dll"             = @{ Source = $VCPKG_BIN; MinSize = 13MB }
    "abseil_dll.dll"              = @{ Source = $VCPKG_BIN; MinSize = 1MB }
    "libprotobuf.dll"             = @{ Source = $VCPKG_BIN; MinSize = 10MB }
    "libprotobuf-lite.dll"        = @{ Source = $VCPKG_BIN; MinSize = 1MB }
    "re2.dll"                     = @{ Source = $VCPKG_BIN; MinSize = 1MB }

    # From vcpkg bin (GPU providers)
    "onnxruntime_providers_shared.dll" = @{ Source = $VCPKG_BIN; MinSize = 10KB }
    "onnxruntime_providers_cuda.dll"   = @{ Source = $VCPKG_BIN; MinSize = 300MB }

    # From ThirdParty (FFmpeg - MUST use these, not build/Release versions!)
    "avcodec-62.dll"   = @{ Source = $THIRDPARTY; MinSize = 100MB }  # ~106MB, NOT ~13MB!
    "avformat-62.dll"  = @{ Source = $THIRDPARTY; MinSize = 20MB }
    "avutil-60.dll"    = @{ Source = $THIRDPARTY; MinSize = 2MB }
    "avfilter-11.dll"  = @{ Source = $THIRDPARTY; MinSize = 80MB }
    "avdevice-62.dll"  = @{ Source = $THIRDPARTY; MinSize = 3MB }
    "swresample-6.dll" = @{ Source = $THIRDPARTY; MinSize = 500KB }
    "swscale-9.dll"    = @{ Source = $THIRDPARTY; MinSize = 2MB }
}

# Optional DLLs (app works without these but with reduced functionality)
$OptionalDLLs = @{
    # AudioFlux (for spectral flux beat detection - falls back to energy without these)
    "audioflux.dll"    = @{ Source = $AUDIOFLUX; MinSize = 1MB }
    "libfftw3f-3.dll"  = @{ Source = $AUDIOFLUX; MinSize = 1MB }
}

function Write-Header($text) {
    Write-Host "`n=== $text ===" -ForegroundColor Cyan
}

function Test-DLLSize($path, $minSize) {
    if (-not (Test-Path $path)) {
        return $false
    }
    $actualSize = (Get-Item $path).Length
    return $actualSize -ge $minSize
}

# Verify mode - just check existing DLLs
if ($Verify) {
    Write-Header "Verifying DLLs in $UE_BIN"
    $errors = @()

    foreach ($dll in $RequiredDLLs.Keys) {
        $targetPath = Join-Path $UE_BIN $dll
        $minSize = $RequiredDLLs[$dll].MinSize

        if (-not (Test-Path $targetPath)) {
            $errors += "MISSING: $dll"
            Write-Host "  [MISSING] $dll" -ForegroundColor Red
        }
        elseif (-not (Test-DLLSize $targetPath $minSize)) {
            $actualSize = (Get-Item $targetPath).Length
            $errors += "WRONG SIZE: $dll (expected >=$minSize, got $actualSize)"
            Write-Host "  [WRONG SIZE] $dll - Expected >=$minSize, Got $([math]::Round($actualSize/1MB, 2))MB" -ForegroundColor Yellow
        }
        else {
            $actualSize = (Get-Item $targetPath).Length
            Write-Host "  [OK] $dll ($([math]::Round($actualSize/1MB, 2))MB)" -ForegroundColor Green
        }
    }

    if ($errors.Count -gt 0) {
        Write-Host "`n!!! VERIFICATION FAILED !!!" -ForegroundColor Red
        Write-Host "Run this script without -Verify to deploy correct DLLs" -ForegroundColor Yellow
        exit 1
    }
    else {
        Write-Host "`nAll DLLs verified successfully!" -ForegroundColor Green
        exit 0
    }
}

# Deploy mode
Write-Header "Deploying DLLs to TripSitter"
Write-Host "Target: $UE_BIN"

if (-not (Test-Path $UE_BIN)) {
    Write-Host "ERROR: Target directory does not exist: $UE_BIN" -ForegroundColor Red
    exit 1
}

# Check source directories exist
$sourceDirs = @($BUILD_RELEASE, $VCPKG_BIN, $THIRDPARTY) | Select-Object -Unique
foreach ($dir in $sourceDirs) {
    if (-not (Test-Path $dir)) {
        Write-Host "ERROR: Source directory does not exist: $dir" -ForegroundColor Red
        Write-Host "Have you built the project? Run: cmake --build build --config Release" -ForegroundColor Yellow
        exit 1
    }
}

# Copy DLLs in correct order (ThirdParty FFmpeg LAST to avoid overwrite)
$copyOrder = @(
    @{ Name = "ONNX Runtime & Dependencies"; Pattern = "build/Release DLLs"; DLLs = @(
        "beatsync_backend_shared.dll", "onnxruntime.dll", "abseil_dll.dll",
        "libprotobuf.dll", "libprotobuf-lite.dll", "re2.dll"
    )},
    @{ Name = "ONNX GPU Providers"; Pattern = "vcpkg bin"; DLLs = @(
        "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"
    )},
    @{ Name = "FFmpeg (ThirdParty)"; Pattern = "ThirdParty"; DLLs = @(
        "avcodec-62.dll", "avformat-62.dll", "avutil-60.dll", "avfilter-11.dll",
        "avdevice-62.dll", "swresample-6.dll", "swscale-9.dll"
    )},
    @{ Name = "AudioFlux (Optional)"; Pattern = "AudioFlux"; Optional = $true; DLLs = @(
        "audioflux.dll", "libfftw3f-3.dll"
    )}
)

    $allOk = $true
foreach ($group in $copyOrder) {
    Write-Header $group.Name
    $isOptional = $group.Optional -eq $true
    foreach ($dll in $group.DLLs) {
        # Look up in RequiredDLLs first, then OptionalDLLs
        $dllInfo = $RequiredDLLs[$dll]
        if (-not $dllInfo) {
            $dllInfo = $OptionalDLLs[$dll]
        }
        if (-not $dllInfo) {
            Write-Host "  [SKIP] $dll - Not found in DLL definitions" -ForegroundColor Yellow
            continue
        }

        $source = $dllInfo.Source
        $sourcePath = Join-Path $source $dll
        $targetPath = Join-Path $UE_BIN $dll
        $minSize = $dllInfo.MinSize

        if (-not (Test-Path $sourcePath)) {
            if ($isOptional) {
                Write-Host "  [SKIP] $dll - Optional, source not found" -ForegroundColor Gray
            } else {
                Write-Host "  [ERROR] $dll - Required DLL source not found: $sourcePath" -ForegroundColor Red
                $allOk = $false
            }
            continue
        }

        # Verify source file size
        if (-not (Test-DLLSize $sourcePath $minSize)) {
            $actualSize = (Get-Item $sourcePath).Length
            if ($isOptional) {
                Write-Host "  [WARN] $dll - Source file smaller than expected ($([math]::Round($actualSize/1MB, 2))MB < $minSize)" -ForegroundColor Yellow
            } else {
                 Write-Host "  [ERROR] $dll - Source file smaller than expected ($([math]::Round($actualSize/1MB, 2))MB < $minSize). Check if the correct library is installed." -ForegroundColor Red
                 $allOk = $false
                 continue
            }
        }

        if ($DryRun) {
            Write-Host "  [DRY-RUN] Would copy: $sourcePath -> $targetPath" -ForegroundColor Cyan
        }
        else {
            Copy-Item -Path $sourcePath -Destination $targetPath -Force
            $size = [math]::Round((Get-Item $targetPath).Length / 1MB, 2)
            Write-Host "  [COPIED] $dll ($size MB)" -ForegroundColor Green
        }
    }
}

# Abort deployment if any required DLLs were missing
if (-not $allOk) {
    Write-Host "Deployment failed: One or more required DLLs were missing." -ForegroundColor Red
    exit 1
}

# Final verification
Write-Header "Final Verification"
$allOk = $true
foreach ($dll in $RequiredDLLs.Keys) {
    $targetPath = Join-Path $UE_BIN $dll
    $minSize = $RequiredDLLs[$dll].MinSize

    if ($DryRun) {
        continue
    }

    if (-not (Test-DLLSize $targetPath $minSize)) {
        Write-Host "  [FAIL] $dll" -ForegroundColor Red
        $allOk = $false
    }
}

if ($DryRun) {
    Write-Host "`nDry run complete. No files were copied." -ForegroundColor Cyan
}
elseif ($allOk) {
    Write-Host "`nDeployment successful! TripSitter.exe is ready to run." -ForegroundColor Green
}
else {
    Write-Host "`nDeployment completed with warnings. Check DLL sizes above." -ForegroundColor Yellow
}
