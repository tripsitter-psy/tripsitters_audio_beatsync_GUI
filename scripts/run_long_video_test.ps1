<#
Run a long end-to-end video creation test and capture GPU metrics.
Usage (PowerShell):
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_long_video_test.ps1 \
    -OutputVideo build/tmp/out_gpu_test.mp4 -Audio build/tmp/gpu_stress.wav -DurationSeconds 600
#>
param(
    [string]$OutputVideo = "build/tmp/out_gpu_test.mp4",
    [string]$Audio = "build/tmp/gpu_stress.wav",
    [int]$DurationSeconds = 600
)

# Ensure output directories
New-Item -ItemType Directory -Force -Path (Split-Path -Path $OutputVideo) | Out-Null
New-Item -ItemType Directory -Force -Path build | Out-Null

Write-Host "Starting long video test: $Audio -> $OutputVideo (duration $DurationSeconds s)"

# Generate audio if missing
if (-not (Test-Path $Audio)) {
    Write-Host "Audio file missing: $Audio. Generating..."
    python tools/generate_long_test_wav.py $Audio $DurationSeconds
}

# Start background GPU logging (every 10s)

$gpuLogPath = Join-Path (Resolve-Path .) "build/gpu_log.csv"
if (Test-Path $gpuLogPath) { Remove-Item $gpuLogPath }
$job = Start-Job -Name NvidiaLog -ScriptBlock {
    param($gpuLogPath)
    if (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue) {
        while ($true) {
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits >> $gpuLogPath
            Start-Sleep -Seconds 10
        }
    }
} -ArgumentList $gpuLogPath

# Add FFmpeg to PATH if we have a local build (optionally)
$ffmpegDir = "C:\ffmpeg-dev"
if (Test-Path $ffmpegDir) {
    Write-Host "Adding FFmpeg to PATH from $ffmpegDir"
    $env:PATH = "$ffmpegDir\bin;" + $env:PATH
}

# Run beatsync create video (capture output)

$stdout = "build/long_video_stdout.log"
$stderr = "build/long_video_stderr.log"
if (Test-Path $stdout) { Remove-Item $stdout }
if (Test-Path $stderr) { Remove-Item $stderr }

$exe = "./build/bin/Release/beatsync.exe"
# Try a few variants if CLI supports different flags
$cmds = @(
    "$exe create `"$OutputVideo`" `"$Audio`" --strategy downbeat --gpu",
    "$exe create `"$OutputVideo`" `"$Audio`" --strategy downbeat",
    "$exe create `"$OutputVideo`" `"$Audio`""
)

$exitCode = 1

$proc = $null
foreach ($cmd in $cmds) {
    Write-Host "Running: $cmd"
    Add-Content -Path $stdout -Value "----- BEGIN OUTPUT -----"

    $tempOut = [System.IO.Path]::GetTempFileName()
    try {
        # Using -FilePath to executable allows direct exit code capture. 
        # But since $cmd has args and escaping, nested powershell is used.
        # We append "; exit `$LASTEXITCODE" to ensure the wrapper returns the real code.
        $proc = Start-Process -FilePath powershell -ArgumentList "-NoProfile -Command $cmd; exit `$LASTEXITCODE" -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tempOut -RedirectStandardError $stderr
        $exitCode = $proc.ExitCode
    } catch {
        if ($proc -and $proc.ExitCode -ne $null) {
            $exitCode = $proc.ExitCode
        } else {
            $exitCode = 1
        }
    }
    if (Test-Path $tempOut) {
        Add-Content -Path $stdout -Value (Get-Content $tempOut)
        Remove-Item $tempOut -Force
    }

    # Append any stderr content instead of overwriting, if needed by the test logic (stderr file already redirected above)
    # The redirection flags on Start-Process handle the file writing.

    Write-Host "Command exit code: $exitCode"
    if ($exitCode -eq 0 -and (Test-Path $OutputVideo)) {
        Write-Host "Video created successfully: $OutputVideo"
        break
    }
    Write-Host "Command failed or produced no output video; trying next form..."
}

# Stop GPU logging job
if (Get-Job -Name NvidiaLog -ErrorAction SilentlyContinue) {
    Stop-Job -Name NvidiaLog -Force
    Receive-Job -Name NvidiaLog -Keep | Out-Null
    Remove-Job -Name NvidiaLog -Force
}

# Summarize
Write-Host "Test finished with exit code $exitCode"
Write-Host "GPU log: build/gpu_log.csv"
Write-Host "Stdout log: $stdout"
Write-Host "Stderr log: $stderr"

exit $exitCode
