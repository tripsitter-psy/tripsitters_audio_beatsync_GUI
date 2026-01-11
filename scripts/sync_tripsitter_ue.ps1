# TripSitter UE Source Sync Script
# Syncs source files between BeatSyncEditor repo and UE Engine Programs folder
#
# Usage:
#   .\sync_tripsitter_ue.ps1 -ToEngine    # Copy from repo to Engine (for building)
#   .\sync_tripsitter_ue.ps1 -ToRepo      # Copy from Engine to repo (after editing in Engine)
#
# Source locations:
#   Repo:   BeatSyncEditor/unreal-prototype/Source/TripSitterUE/
#   Engine: C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\

param(
    [switch]$ToEngine,
    [switch]$ToRepo,
    [switch]$Force
)

$RepoRoot = $PSScriptRoot | Split-Path -Parent
$RepoSource = Join-Path $RepoRoot "unreal-prototype\Source\TripSitterUE"
$EngineSource = "C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter"

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

            Set-Content $dstFile -Value $content -NoNewline
            Write-Host "  [OK] $file" -ForegroundColor Green
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

    Write-Host "`nSync complete! Now run:" -ForegroundColor Cyan
    Write-Host "  cd 'C:\UE5_Source\UnrealEngine'" -ForegroundColor White
    Write-Host "  .\Engine\Build\BatchFiles\Build.bat TripSitter Win64 Shipping" -ForegroundColor White
}

function Sync-ToRepo {
    Write-Host "Syncing from Engine to Repo..." -ForegroundColor Cyan
    Write-Host "  From: $EngineSource\Private\" -ForegroundColor Gray
    Write-Host "  To:   $RepoSource\Private\" -ForegroundColor Gray

    foreach ($file in $SourceFiles) {
        $srcFile = Join-Path $EngineSource "Private\$file"
        $dstFile = Join-Path $RepoSource "Private\$file"

        if (Test-Path $srcFile) {
            $content = Get-Content $srcFile -Raw

            # Add TRIPSITTERUE_API back to class declarations that don't have it
            # This is a simple heuristic - classes starting with S (Slate), F (struct-like), or specific names
            $content = $content -replace 'class\s+(STripSitterMainWidget|SWaveformViewer|FBeatsyncLoader|FBeatsyncProcessingTask|FBeatsyncProcessingResult|FEffectsConfig)', 'class TRIPSITTERUE_API $1'

            Set-Content $dstFile -Value $content -NoNewline
            Write-Host "  [OK] $file" -ForegroundColor Green
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

if ($ToEngine) {
    Sync-ToEngine
} elseif ($ToRepo) {
    Sync-ToRepo
}
