param([switch]$Verify)

$UE_BIN = "D:\UnrealEngine\Engine\Binaries\Win64"

# Define expected DLLs with minimum sizes (to catch wrong versions)
$RequiredDLLs = @{
    # From build/Release (Project Output)
    "beatsync_backend_shared.dll" = 300KB

    # From vcpkg bin (ONNX and dependencies)
    "abseil_dll.dll"              = 1MB
    "libprotobuf.dll"             = 10MB
    "libprotobuf-lite.dll"        = 1MB
    "re2.dll"                     = 1MB
    "onnxruntime.dll"             = 14MB
    "onnxruntime_providers_shared.dll"   = 10KB
    "onnxruntime_providers_cuda.dll"     = 5MB

    # From ThirdParty (FFmpeg)
    "avcodec-62.dll"   = 100MB
    "avformat-62.dll"  = 20MB
    "avutil-60.dll"    = 2MB
    "avfilter-11.dll"  = 80MB
    "avdevice-62.dll"  = 3MB
    "swresample-6.dll" = 500KB
    "swscale-9.dll"    = 2MB
}

Write-Host "`n=== Checking DLLs in $UE_BIN ===" -ForegroundColor Cyan

if (-not (Test-Path $UE_BIN)) {
    Write-Host "ERROR: Directory does not exist: $UE_BIN" -ForegroundColor Red
    exit 1
}

$missing = @()
$wrongSize = @()
$ok = @()

foreach ($dll in $RequiredDLLs.Keys) {
    $targetPath = Join-Path $UE_BIN $dll
    $minSize = $RequiredDLLs[$dll]

    if (-not (Test-Path $targetPath)) {
        $missing += $dll
        Write-Host "  [MISSING] $dll" -ForegroundColor Red
    }
    else {
        $actualSize = (Get-Item $targetPath).Length
        if ($actualSize -lt $minSize) {
            $wrongSize += @{dll=$dll; expected=[math]::Round($minSize/1MB,2); actual=[math]::Round($actualSize/1MB,2)}
            Write-Host "  [WRONG SIZE] $dll - Expected >=$([math]::Round($minSize/1MB,2))MB, Got $([math]::Round($actualSize/1MB,2))MB" -ForegroundColor Yellow
        }
        else {
            $ok += $dll
            $sizeMB = [math]::Round($actualSize/1MB,2)
            Write-Host "  [OK] $dll ($sizeMB MB)" -ForegroundColor Green
        }
    }
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "OK: $($ok.Count) / $($RequiredDLLs.Count)" -ForegroundColor Green
if ($missing.Count -gt 0) {
    Write-Host "MISSING: $($missing.Count)" -ForegroundColor Red
    Write-Host "  $($missing -join ', ')"
}
if ($wrongSize.Count -gt 0) {
    Write-Host "WRONG SIZE: $($wrongSize.Count)" -ForegroundColor Yellow
}

if ($missing.Count -eq 0 -and $wrongSize.Count -eq 0) {
    Write-Host "`nAll DLLs verified successfully! TripSitter.exe can run." -ForegroundColor Green
    exit 0
}
else {
    Write-Host "`nDeploy using: ..\scripts\deploy_tripsitter.ps1" -ForegroundColor Yellow
    exit 1
}
