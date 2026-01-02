<#
import_assets.ps1

Copies GUI asset files from a source folder into the project's `assets/` folder.
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\import_assets.ps1 -Source "C:\path\to\assets for GUI aesthetics" -Force

Parameters:
  -Source <path>      Path to the folder containing image files (PNG, ICO, JPG, JPEG)
  -Force              Overwrite existing files without prompting
  -DryRun             Show what would be copied without performing the copy
#>

param(
    [Parameter(Mandatory=$true)][string]$Source,
    [switch]$Force,
    [switch]$DryRun
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$assetsDir = Join-Path $projectRoot "..\assets" | Resolve-Path -ErrorAction Stop
$assetsDir = $assetsDir.Path

if (-not (Test-Path -Path $Source)) {
    Write-Error "Source path '$Source' does not exist."
    exit 1
}

# Supported extensions
$exts = @('*.png','*.jpg','*.jpeg','*.ico')

# Find files
$files = @()
foreach ($e in $exts) {
    $files += Get-ChildItem -Path $Source -Filter $e -File -Recurse -ErrorAction SilentlyContinue
}

if ($files.Count -eq 0) {
    Write-Host "No supported image files found in: $Source"
    exit 0
}

# Backup existing assets if any
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = Join-Path $assetsDir "backup-$timestamp"
New-Item -ItemType Directory -Path $backupDir | Out-Null

foreach ($file in $files) {
    $dest = Join-Path $assetsDir $file.Name
    if (Test-Path $dest) {
        Write-Host "Backing up existing: $($file.Name) -> $backupDir"
        Copy-Item -Path $dest -Destination $backupDir -Force
    }
}

# Copy (or show on dry-run)
foreach ($file in $files) {
    $dest = Join-Path $assetsDir $file.Name
    Write-Host "Copying: $($file.FullName) -> $dest"
    if (-not $DryRun) {
        Copy-Item -Path $file.FullName -Destination $dest -Force:$Force.IsPresent
    }
}

Write-Host "Imported $($files.Count) file(s) into $assetsDir"
Write-Host "If you need to embed an icon, ensure there is a valid 'assets/icon.ico' and 'assets/icon.rc' is present."