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
    [switch]$ToRepo,
    [switch]$NonInteractive
)


$RepoRoot = $PSScriptRoot | Split-Path -Parent
$RepoSource = Join-Path $RepoRoot "unreal-prototype\Source\TripSitter"
# Validate $RepoSource exists
if (-not (Test-Path $RepoSource)) {
    Write-Error "Repo source path '$RepoSource' does not exist. Please check your repository structure."
    exit 1
}

function Get-EngineSourcePath {
    $engineSource = $env:TRIPSITTER_ENGINE_PATH
    if (-not $engineSource) {
        $isInteractive = -not $NonInteractive -and [Environment]::UserInteractive
        if ($isInteractive) {
            $engineSource = Read-Host "Enter the path to the Unreal Engine source directory (e.g., C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter)"
        } else {
            Write-Warning "Engine source path not set and cannot prompt in non-interactive mode. Set TRIPSITTER_ENGINE_PATH or pass interactively."
            return $null
        }
    }
    if (-not (Test-Path $engineSource)) {
        Write-Error "Engine source path '$engineSource' does not exist. Set TRIPSITTER_ENGINE_PATH environment variable or provide a valid path."
        return $null
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
    try {
        $EngineSource = Get-EngineSourcePath
        if (-not $EngineSource) {
            Write-Error "Engine source path could not be resolved. Aborting sync."
            return
        }
        Write-Host "Syncing from Repo to Engine..." -ForegroundColor Cyan
        Write-Host "  From: $RepoSource\Private\" -ForegroundColor Gray
        Write-Host "  To:   $EngineSource\Private\" -ForegroundColor Gray

        $failedWrites = 0
        foreach ($file in $SourceFiles) {
            $srcFile = Join-Path $RepoSource "Private\$file"
            $dstFile = Join-Path $EngineSource "Private\$file"

            if (Test-Path $srcFile) {
                # Strip TRIPSITTERUE_API from class declarations
                $content = [System.IO.File]::ReadAllText($srcFile, [System.Text.UTF8Encoding]::new($false))
                $content = $content -replace 'class\s+TRIPSITTERUE_API\s+', 'class '
                $content = $content -replace 'struct\s+TRIPSITTERUE_API\s+', 'struct '

                # Create directory if needed
                $dstDir = Split-Path $dstFile -Parent
                if (-not (Test-Path $dstDir)) {
                    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
                }

                try {
                    [System.IO.File]::WriteAllText($dstFile, $content, [System.Text.UTF8Encoding]::new($false))
                    Write-Host "  [OK] $file" -ForegroundColor Green
                } catch {
                    Write-Host "  [FAIL] $file: $_" -ForegroundColor Red
                    $failedWrites++
                }
            } else {
                Write-Host "  [SKIP] $file (not found in repo)" -ForegroundColor Yellow
            }
        }
        if ($failedWrites -gt 0) {
            Write-Host "One or more file writes failed. Exiting with error." -ForegroundColor Red
            exit 1
        }

        # Also sync Resources folder
        $repoResources = Join-Path $RepoSource "Resources"
        $engineResources = Join-Path $EngineSource "Resources"
        if (Test-Path $repoResources) {
            $children = Get-ChildItem $repoResources -Recurse -File -ErrorAction SilentlyContinue
            if ($children.Count -eq 0) {
                Write-Warning "Resources folder exists but is empty. Skipping copy."
            } else {
                if (-not (Test-Path $engineResources)) {
                    New-Item -ItemType Directory -Path $engineResources -Force | Out-Null
                }
                try {
                    $sourcePath = Join-Path $repoResources '*'
                    Copy-Item $sourcePath $engineResources -Force -Recurse
                    Write-Host "  [OK] Resources folder" -ForegroundColor Green
                } catch {
                    Write-Error "Failed to copy Resources folder: $_"
                    exit 1
                }
            }
        }

        # Resolve Engine root by searching upward for a directory named 'Engine'
        $EngineRoot = $null
        $candidate = $EngineSource
        while ($candidate) {
            if ((Split-Path $candidate -Leaf) -eq 'Engine') {
                $EngineRoot = Split-Path $candidate -Parent
                break
            }
            $parent = Split-Path $candidate -Parent
            if ($parent -eq $candidate -or [string]::IsNullOrEmpty($parent)) { break }
            $candidate = $parent
        }

        # Fallback: climb up 4 levels from EngineSource to approximate root
        if (-not $EngineRoot) {
            try {
                $EngineRoot = Split-Path (Split-Path (Split-Path (Split-Path $EngineSource -Parent) -Parent) -Parent) -Parent
            } catch {
                $EngineRoot = Split-Path $EngineSource -Parent
            }
        }

        if (-not (Test-Path $EngineRoot)) {
            Write-Warning "Engine root directory '$EngineRoot' does not exist."
            return
        }
        Write-Host "`nSync complete! Now run:" -ForegroundColor Cyan
        Write-Host "  cd '$EngineRoot'" -ForegroundColor White
        Write-Host "  .\Engine\Build\BatchFiles\Build.bat TripSitter Win64 Shipping" -ForegroundColor White
    } catch {
        Write-Error "Sync-ToEngine failed: $_"
        exit 1
    }
}

function Sync-ToRepo {
    $EngineSource = Get-EngineSourcePath
    if (-not $EngineSource) {
        Write-Error "Engine source path could not be resolved. Aborting sync."
        return
    }
    $failedWrites = 0
    Write-Host "Syncing from Engine to Repo..." -ForegroundColor Cyan
    Write-Host "  From: $EngineSource\Private\" -ForegroundColor Gray
    Write-Host "  To:   $RepoSource\Private\" -ForegroundColor Gray

    foreach ($file in $SourceFiles) {
        $srcFile = Join-Path $EngineSource "Private\$file"
        $dstFile = Join-Path $RepoSource "Private\$file"

        if (Test-Path $srcFile) {
            # Read file as UTF-8 (no BOM)
            $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
            $content = [System.IO.File]::ReadAllText($srcFile, $utf8NoBom)

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
            # Anchor word boundaries correctly in the regex (use single '\b')
            $content = $content -replace "class\s+(?!TRIPSITTERUE_API\b)($ApiPattern)\b", 'class TRIPSITTERUE_API $1'
            $content = $content -replace "struct\s+(?!TRIPSITTERUE_API\b)($ApiPattern)\b", 'struct TRIPSITTERUE_API $1'

            # Create directory if needed
            $dstDir = Split-Path $dstFile -Parent
            if (-not (Test-Path $dstDir)) {
                New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
            }

            try {
                [System.IO.File]::WriteAllText($dstFile, $content, $utf8NoBom)
                Write-Host "  [OK] $file" -ForegroundColor Green
            } catch {
                Write-Host "  [FAIL] $file: $_" -ForegroundColor Red
                $failedWrites++
            }
        } else {
            Write-Host "  [SKIP] $file (not found in engine)" -ForegroundColor Yellow
        }
    }

    if ($failedWrites -eq 0) {
        Write-Host "`nSync complete!" -ForegroundColor Cyan
    } else {
        Write-Error "Sync failed: $failedWrites file(s) could not be written."
        exit 1
    }
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
