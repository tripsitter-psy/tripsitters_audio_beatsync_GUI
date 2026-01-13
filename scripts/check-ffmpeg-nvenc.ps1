# Check for ffmpeg and NVENC encoder support
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpeg) {
    Write-Host 'FFmpeg found at:' $ffmpeg.Source
    & $ffmpeg.Source -hide_banner -encoders 2>&1 | Select-String -Pattern 'nvenc|h264_nvenc|hevc_nvenc'
} else {
    Write-Host 'FFmpeg not in PATH. Checking common locations...'
    $paths = @(
        'C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe',
        'C:\ffmpeg\bin\ffmpeg.exe'
    )
    foreach ($p in $paths) {
        if (Test-Path $p) {
            Write-Host 'Found at:' $p
            & $p -hide_banner -encoders 2>&1 | Select-String -Pattern 'nvenc'
        }
    }
}