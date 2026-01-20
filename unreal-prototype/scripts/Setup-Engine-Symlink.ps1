# Engine Symlink Setup for TripSitter Program Target
# Run this script as Administrator to create the required Engine symlink

param(
    [string]$EnginePath = "C:\Program Files\Epic Games\UE_5.7\Engine"
)

Write-Host "Setting up Engine symlink for TripSitter Program target..." -ForegroundColor Cyan

$ProjectEnginePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\Engine"))


# Check if Engine directory or symlink already exists
if (Test-Path $ProjectEnginePath -Force) {
    $item = Get-Item $ProjectEnginePath -Force
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        # It's a symlink or junction
        $target = $null
        try {
            $target = (Get-Item $ProjectEnginePath -Force).Target
        } catch {}
        if ($target -and [System.IO.Directory]::Exists($target)) {
            Write-Host "Engine symlink already exists and target is valid: $ProjectEnginePath -> $target" -ForegroundColor Yellow
            exit 0
        } else {
            Write-Warning "Broken symlink detected at $ProjectEnginePath. Removing..."
            Remove-Item $ProjectEnginePath -Force
        }
    } else {
        Write-Error "A real directory exists at '$ProjectEnginePath' instead of a symlink. Remove or rename this directory before running this script."
        exit 1
    }
}

# Check if source Engine exists
if (!(Test-Path $EnginePath)) {
    Write-Error "Installed Engine not found at: $EnginePath. Please verify your Unreal Engine 5.7 installation path or provide a custom path using -EnginePath parameter."
    exit 1
}


# Check for Administrator privileges before attempting mklink
$currentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object System.Security.Principal.WindowsPrincipal($currentIdentity)
if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: This script must be run as Administrator to create a symlink." -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as administrator', then re-run this script." -ForegroundColor Yellow
    exit 1
}


# Create the symlink (requires Administrator privileges)
try {
    cmd /c "mklink /d \"$ProjectEnginePath\" \"$EnginePath\""
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Engine symlink created successfully!" -ForegroundColor Green
        Write-Host "Symlink: $ProjectEnginePath -> $EnginePath" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to create Engine symlink. Please run this script as Administrator." -ForegroundColor Red
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "ERROR: Exception occurred while creating symlink: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`nSetup complete! You can now build the TripSitter Program target." -ForegroundColor Green