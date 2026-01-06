<#
Simple sanity check for evaluation pipeline (Windows PowerShell)
Generates a synthetic corrupted test, probes files with ffprobe, runs beatsync, and prints a short verdict.
#>
param(
    [string]$Id = "corrupt_audio",
    [int]$AudioDuration = 5,
    [int]$VideoDuration = 5,
    [string]$GeneratorArgs = "--corrupt-audio-header --corrupt-bytes 256",
    [switch]$EnsureVenv,
    [switch]$ExpectFailure
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

function CheckCmd($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    return $null -ne $cmd
}

if (-not (CheckCmd ffmpeg) -or -not (CheckCmd ffprobe)) {
    Write-Error "ffmpeg/ffprobe not found on PATH. Install (choco/scoop/brew/apt) or add to PATH and re-run."
    exit 2
}

if ($EnsureVenv) {
    Write-Host "Creating project .venv and installing requirements..."
    & .\scripts\setup_venv.ps1
}

$tmp = Join-Path $root "evaluation/tmp/$Id"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }

$gen = Join-Path $root "evaluation/generate_synthetic.py"
$cmd = "python `"$gen`" --outdir `"$tmp`" --id $Id --audio-duration $AudioDuration --video-duration $VideoDuration $GeneratorArgs"
Write-Host "Running generator:`n$cmd"
$genOut = cmd /c $cmd 2>&1
Write-Host $genOut

# Find generated audio/video
$audio = Get-ChildItem -Path $tmp -File -Include "${Id}.*" -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in '.wav','.mp3','.m4a' } | Select-Object -First 1
$video = Get-ChildItem -Path $tmp -File -Include "${Id}.*" -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in '.mp4','.mov','.m4a' } | Select-Object -First 1

Write-Host "Generated files:"; Get-ChildItem -Path $tmp -File | Format-Table Name, Length -AutoSize

if ($audio) {
    Write-Host "\nffprobe (audio):"
    & ffprobe -v error -show_streams $audio.FullName 2>&1 | Write-Host
} else {
    Write-Host "No generated audio file found"
}

if ($video) {
    Write-Host "\nffprobe (video):"
    & ffprobe -v error -show_streams $video.FullName 2>&1 | Write-Host
} else {
    Write-Host "No generated video file found"
}

$out = Join-Path $tmp "${Id}.out.mp4"
$beatsyncPath = Join-Path $root "build\bin\Release\beatsync.exe"
$beatsyncExists = Test-Path $beatsyncPath

$beatsyncErr = $null
$beatsyncRC = -1
if ($beatsyncExists -and $video -and $audio) {
    Write-Host "\nRunning beatsync sync..."
    $proc = Start-Process -FilePath $beatsyncPath -ArgumentList @('sync', $video.FullName, $audio.FullName, '-o', $out) -NoNewWindow -PassThru -Wait -RedirectStandardError "evaluation/tmp/beatsync.err.txt" -RedirectStandardOutput "evaluation/tmp/beatsync.out.txt"
    $beatsyncRC = $proc.ExitCode
    if (Test-Path "evaluation/tmp/beatsync.err.txt") { $beatsyncErr = Get-Content "evaluation/tmp/beatsync.err.txt" -Raw }
    Write-Host "beatsync exit code: $beatsyncRC"
    if ($beatsyncErr) { Write-Host "beatsync stderr:"; Write-Host $beatsyncErr }
} else {
    Write-Host "beatsync.exe not found or media missing; skipping beatsync run"
}

if (Test-Path $out) {
    Write-Host "\nffprobe (output):"
    & ffprobe -v error -show_streams $out 2>&1 | Write-Host
} else {
    Write-Host "No output file produced"
}

# Tail traces.jsonl if present
if (Test-Path "$root\traces.jsonl") {
    Write-Host "\nLast traces.jsonl lines:"
    Get-Content "$root\traces.jsonl" -Tail 40 | Write-Host
} else {
    Write-Host "No traces.jsonl found"
}

# Basic verdict
$playable = $false
if (Test-Path $out) {
    $probe = & ffprobe -v error -show_streams $out 2>&1
    if ($probe -and $probe -notmatch 'Error') { $playable = $true }
}

if ($ExpectFailure) {
    $passed = ($beatsyncRC -ne 0) -or (-not (Test-Path $out)) -or (-not $playable)
} else {
    $passed = (Test-Path $out) -and $playable
}

Write-Host "\nSanity check verdict:"; if ($passed) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL" -ForegroundColor Red }

exit ([int](-not $passed))
