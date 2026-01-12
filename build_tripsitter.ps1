# Build and package TripSitter standalone app
param(
    [string]$UERoot = "C:\Program Files\Epic Games\UE_5.7",
    [string]$ProjectPath = (Join-Path $env:USERPROFILE "OneDrive\Documents\Unreal Projects\MyProject\MyProject.uproject"),
    [string]$ArchiveDir = (Join-Path $PSScriptRoot "packaged")
)

Write-Host "Building and packaging TripSitter..." -ForegroundColor Cyan

# Clean intermediate files to force rebuild
$IntermediatePath = Join-Path $env:USERPROFILE "OneDrive\Documents\Unreal Projects\MyProject\Intermediate"
if (Test-Path "$IntermediatePath\Build") {
    Write-Host "Cleaning intermediate build files..."
    Remove-Item -Recurse -Force "$IntermediatePath\Build" -ErrorAction SilentlyContinue
}

# Run BuildCookRun
& "$UERoot\Engine\Build\BatchFiles\RunUAT.bat" BuildCookRun `
    -project="$ProjectPath" `
    -noP4 `
    -platform=Win64 `
    -clientconfig=Shipping `
    -cook `
    -build `
    -stage `
    -pak `
    -archive `
    -archivedirectory="$ArchiveDir"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done! Build completed successfully." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Build failed with exit code $LASTEXITCODE. Check above for errors." -ForegroundColor Red
    exit $LASTEXITCODE
}
Read-Host "Press Enter to close"
