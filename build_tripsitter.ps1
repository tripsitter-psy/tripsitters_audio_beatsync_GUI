# Build and package TripSitter standalone app
Write-Host "Building and packaging TripSitter..." -ForegroundColor Cyan

# Clean intermediate files to force rebuild
$IntermediatePath = "C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Intermediate"
if (Test-Path "$IntermediatePath\Build") {
    Write-Host "Cleaning intermediate build files..."
    Remove-Item -Recurse -Force "$IntermediatePath\Build" -ErrorAction SilentlyContinue
}

# Run BuildCookRun
& "C:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\RunUAT.bat" BuildCookRun `
    -project="C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\MyProject.uproject" `
    -noP4 `
    -platform=Win64 `
    -clientconfig=Shipping `
    -cook `
    -build `
    -stage `
    -pak `
    -archive `
    -archivedirectory="C:\Users\samue\Desktop\BeatSyncEditor\packaged"

Write-Host ""
Write-Host "Done! Check above for any errors." -ForegroundColor Green
Read-Host "Press Enter to close"
