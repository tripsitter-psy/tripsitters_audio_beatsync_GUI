param(
    [string]$ReleaseDir = "build/Release"
)

$ErrorActionPreference = "Stop"
Write-Host "Checking for prohibited FFmpeg DLLs in $ReleaseDir..."

if (-not (Test-Path $ReleaseDir)) {
    Write-Warning "Release directory '$ReleaseDir' does not exist yet. Skipping check."
    exit 0
}

# Patterns for FFmpeg DLLs (including versioned ones like avcodec-61.dll)
$forbiddenPatterns = @("avcodec*.dll", "avformat*.dll", "avutil*.dll", "swresample*.dll", "swscale*.dll", "avdevice*.dll", "avfilter*.dll")
$found = $false

foreach ($pattern in $forbiddenPatterns) {
    $foundFiles = Get-ChildItem -Path $ReleaseDir -Filter $pattern -Recurse -ErrorAction SilentlyContinue
    if ($foundFiles) {
        foreach ($file in $foundFiles) {
            Write-Warning "Removing prohibited FFmpeg DLL from build output: $($file.Name). This implies vcpkg or another process copied the wrong FFmpeg version."
            Remove-Item $file.FullName -Force
            $found = $true
        }
    }
}

if ($found) {
    Write-Error "Build validation failed: FFmpeg DLLs detected in output directory. Check VCPKG_APPLOCAL_DEPS setting."
    exit 1
}

Write-Host "Clean. No FFmpeg DLLs found in $ReleaseDir." -ForegroundColor Green
exit 0
