# TripSitter UE Source Sync Script
# Syncs source files between BeatSyncEditor repo and UE Engine Programs folder
#
# Usage:
#   .\sync_tripsitter_ue.ps1 -ToEngine    # Copy from repo to Engine (for building)
#   .\sync_tripsitter_ue.ps1 -ToRepo      # Copy from Engine to repo (after editing in Engine)
#
# Source locations:
#   Repo:   BeatSyncEditor/unreal-prototype/Source/TripSitter/
#   Engine: C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\

param(
    [switch]$ToEngine,
    [switch]$ToRepo
)

$RepoRoot = $PSScriptRoot | Split-Path -Parent
$RepoSource = Join-Path $RepoRoot "unreal-prototype\Source\TripSitter"

function Get-EngineSourcePath {
    $engineSource = $env:TRIPSITTER_ENGINE_PATH
    if (-not $engineSource) {
        $engineSource = Read-Host "Enter the path to the Unreal Engine source directory (e.g., C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter)"
    }
    if (-not (Test-Path $engineSource)) {
        Write-Error "Engine source path '$engineSource' does not exist. Set TRIPSITTER_ENGINE_PATH environment variable or provide a valid path."
        exit 1
    }
    return $engineSource
}

# Files to sync (relative to Private folder)
$SourceFiles = @(
    "BeatsyncLoader.h",
    "BeatsyncLoader.cpp",
    "BeatsyncProcessingTask.h",
    "BeatsyncProcessingTask.cpp",
    "STripSitterMainWidget.h",
    "STripSitterMainWidget.cpp",
    "SWaveformViewer.h",
    "SWaveformViewer.cpp"
)

# Additional engine-only files (not synced to repo)
$EngineOnlyFiles = @(
    "TripSitterApp.h",
    "TripSitterApp.cpp",
    "TripSitter.h",
    "Windows\TripSitterMainWindows.cpp"
)

function Sync-ToEngine {
    $EngineSource = Get-EngineSourcePath
    Write-Host "Syncing from Repo to Engine..." -ForegroundColor Cyan
    Write-Host "  From: $RepoSource\Private\" -ForegroundColor Gray
    Write-Host "  To:   $EngineSource\Private\" -ForegroundColor Gray

    foreach ($file in $SourceFiles) {
        $srcFile = Join-Path $RepoSource "Private\$file"
        $dstFile = Join-Path $EngineSource "Private\$file"

        if (Test-Path $srcFile) {
            # Strip TRIPSITTERUE_API from class declarations
            $content = Get-Content $srcFile -Raw
            $content = $content -replace 'class\s+TRIPSITTERUE_API\s+', 'class '
            $content = $content -replace 'struct\s+TRIPSITTERUE_API\s+', 'struct '

            # Create directory if needed
            $dstDir = Split-Path $dstFile -Parent
            if (-not (Test-Path $dstDir)) {
                New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
            }

            try {
                Set-Content $dstFile -Value $content -NoNewline -ErrorAction Stop
                Write-Host "  [OK] $file" -ForegroundColor Green
            } catch {
                Write-Host "  [FAIL] $file: $_" -ForegroundColor Red
            }
        } else {
            Write-Host "  [SKIP] $file (not found in repo)" -ForegroundColor Yellow
        }
    }

    # Also sync Resources folder
    $repoResources = Join-Path $RepoSource "Resources"
    $engineResources = Join-Path $EngineSource "Resources"
    if (Test-Path $repoResources) {
        if (-not (Test-Path $engineResources)) {
            New-Item -ItemType Directory -Path $engineResources -Force | Out-Null
        }
        Copy-Item "$repoResources\*" $engineResources -Force -Recurse
        Write-Host "  [OK] Resources folder" -ForegroundColor Green
    }

    # Use Engine root for cd, and Build.bat path relative to that
    $EngineRoot = Split-Path $EngineSource -Parent
    Write-Host "`nSync complete! Now run:" -ForegroundColor Cyan
    Write-Host "  cd '$EngineRoot'" -ForegroundColor White
    Write-Host "  .\Engine\Build\BatchFiles\Build.bat TripSitter Win64 Shipping" -ForegroundColor White
}

function Sync-ToRepo {
    $EngineSource = Get-EngineSourcePath
    Write-Host "Syncing from Engine to Repo..." -ForegroundColor Cyan
    Write-Host "  From: $EngineSource\Private\" -ForegroundColor Gray
    Write-Host "  To:   $RepoSource\Private\" -ForegroundColor Gray

    foreach ($file in $SourceFiles) {
        $srcFile = Join-Path $EngineSource "Private\$file"
        $dstFile = Join-Path $RepoSource "Private\$file"

        if (Test-Path $srcFile) {
            $content = Get-Content $srcFile -Raw

            # Add TRIPSITTERUE_API back to class/struct declarations that don't have it
            $ApiMarkedClasses = @(
                'STripSitterMainWidget',
                'SWaveformViewer',
                'FBeatsyncLoader',
                'FBeatsyncProcessingTask',
                'FBeatsyncProcessingResult',
                'FEffectsConfig'
            )
            $ApiPattern = ($ApiMarkedClasses | ForEach-Object { [Regex]::Escape($_) }) -join '|'
            $content = $content -replace "class\s+(?!TRIPSITTERUE_API\\b)($ApiPattern)", 'class TRIPSITTERUE_API $1'
            $content = $content -replace "struct\s+(?!TRIPSITTERUE_API\\b)($ApiPattern)", 'struct TRIPSITTERUE_API $1'

            # Create directory if needed
            $dstDir = Split-Path $dstFile -Parent
            if (-not (Test-Path $dstDir)) {
                New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
            }

            try {
                Set-Content $dstFile -Value $content -NoNewline -ErrorAction Stop
                Write-Host "  [OK] $file" -ForegroundColor Green
            } catch {
                Write-Host "  [FAIL] $file: $_" -ForegroundColor Red
            }
        } else {
            Write-Host "  [SKIP] $file (not found in engine)" -ForegroundColor Yellow
        }
    }

    Write-Host "`nSync complete!" -ForegroundColor Cyan
}

# Main
if (-not $ToEngine -and -not $ToRepo) {
    Write-Host @"
TripSitter UE Source Sync Script
================================

Usage:
  .\sync_tripsitter_ue.ps1 -ToEngine    Copy from repo to Engine (for building)
  .\sync_tripsitter_ue.ps1 -ToRepo      Copy from Engine to repo (after editing)

Workflow:
  1. Edit files in either location
  2. Run -ToEngine to prepare for build
  3. Build with: .\Engine\Build\BatchFiles\Build.bat TripSitter Win64 Shipping
  4. Run -ToRepo to save changes back to git repo

Files synced:
"@ -ForegroundColor Cyan
    foreach ($f in $SourceFiles) { Write-Host "  - $f" -ForegroundColor White }

    Write-Host "`nEngine-only files (not synced):" -ForegroundColor Gray
    foreach ($f in $EngineOnlyFiles) { Write-Host "  - $f" -ForegroundColor Gray }
    exit 0
}

# Check for mutual exclusivity of switches

if ($ToEngine -and $ToRepo) {
    Write-Error "Cannot specify both -ToEngine and -ToRepo switches. Choose one direction for sync: use -ToEngine to copy from repo to Engine (Sync-ToEngine function) or -ToRepo to copy from Engine to repo (Sync-ToRepo function)."
    exit 1
}

if ($ToEngine) {
    Sync-ToEngine
} elseif ($ToRepo) {
    Sync-ToRepo
}
