# Engine Symlink Setup for TripSitter Program Target
# Run this script as Administrator to create the required Engine symlink

Write-Host "Setting up Engine symlink for TripSitter Program target..." -ForegroundColor Cyan

$EnginePath = "C:\Program Files\Epic Games\UE_5.7\Engine"
$ProjectEnginePath = "$PSScriptRoot\Engine"

# Check if Engine directory already exists
if (Test-Path $ProjectEnginePath) {
    Write-Host "Engine directory/symlink already exists at: $ProjectEnginePath" -ForegroundColor Yellow
    exit 0
}

# Check if source Engine exists
if (!(Test-Path $EnginePath)) {
    Write-Host "ERROR: Installed Engine not found at: $EnginePath" -ForegroundColor Red
    Write-Host "Please verify your Unreal Engine 5.7 installation path." -ForegroundColor Red
    exit 1
}

# Create the symlink (requires Administrator privileges)
try {
    cmd /c mklink /d "`"$ProjectEnginePath`"" "`"$EnginePath`""
    Write-Host "Engine symlink created successfully!" -ForegroundColor Green
    Write-Host "Symlink: $ProjectEnginePath -> $EnginePath" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to create Engine symlink. Please run this script as Administrator." -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`nSetup complete! You can now build the TripSitter Program target." -ForegroundColor Green