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
$removalFailed = $false

foreach ($pattern in $forbiddenPatterns) {
    $foundFiles = Get-ChildItem -Path $ReleaseDir -Filter $pattern -Recurse -ErrorAction SilentlyContinue
    if ($foundFiles) {
        foreach ($file in $foundFiles) {
            Write-Warning "Removing prohibited FFmpeg DLL from build output: $($file.Name). This implies vcpkg or another process copied the wrong FFmpeg version."
            try {
                Remove-Item $file.FullName -Force -ErrorAction Stop
            } catch {
                # Use Write-Warning instead of Write-Error to avoid terminating under $ErrorActionPreference = "Stop"
                Write-Warning "Failed to remove $($file.FullName): $_"
                $removalFailed = $true
            }
            $found = $true
        }
    }
}

if ($removalFailed) {
    Write-Host "Build validation failed: FFmpeg DLLs were detected and some could not be removed. Check VCPKG_APPLOCAL_DEPS setting." -ForegroundColor Red
    exit 1
}

if ($found) {
    Write-Host "Build validation warning: FFmpeg DLLs were detected and removed from output directory. Check VCPKG_APPLOCAL_DEPS setting." -ForegroundColor Yellow
    exit 1
}

Write-Host "Clean. No FFmpeg DLLs found in $ReleaseDir." -ForegroundColor Green
exit 0
