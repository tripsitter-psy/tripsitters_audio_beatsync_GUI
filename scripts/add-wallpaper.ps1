<#
add-wallpaper.ps1

Copies a wallpaper image into the project's `assets/wallpaper.png`, commits it to git, and optionally pushes.
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\add-wallpaper.ps1 -Source "C:\path\to\tripsitter MTV GUI WALLPAPER_final_ (2).png" -Commit -Push

Parameters:
  -Source <path>   Path to the source image to copy (PNG recommended)
  -Commit          Create a git commit after copying
  -Push            Push the commit to origin (will fail if remote not configured)
  -Message <text>  Commit message (default: "assets: add wallpaper.png")
#>
param(
    [Parameter(Mandatory=$true)][string]$Source,
    [switch]$Commit,
    [switch]$Push,
    [string]$Message = "assets: add wallpaper.png"
)

if (-not (Test-Path $Source)) {
    Write-Error "Source file '$Source' not found"
    exit 1
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$assetsDir = Join-Path $projectRoot "..\assets" | Resolve-Path -ErrorAction Stop
$assetsDir = $assetsDir.Path
$dest = Join-Path $assetsDir "wallpaper.png"

# Backup existing wallpaper if present
if (Test-Path $dest) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $bak = Join-Path $assetsDir "wallpaper.$timestamp.png"
    Copy-Item -Path $dest -Destination $bak -Force
    Write-Host "Backed up existing wallpaper to $bak"
}

Copy-Item -Path $Source -Destination $dest -Force
Write-Host "Copied wallpaper to $dest"

if ($Commit.IsPresent) {
    Push-Location $projectRoot
    git add "$dest"
    git commit -m "$Message"
    if ($Push.IsPresent) {
        git push
    }
    Pop-Location
}
