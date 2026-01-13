# Build and package TripSitter standalone app
# WARNING: Avoid using OneDrive-synced folders for UERoot, ProjectPath, or ArchiveDir
# as file-locking during builds can cause failures.
param(
    [Parameter(Mandatory=$false)]
    [string]$UERoot = "C:\UE5_Source\UnrealEngine",
    [Parameter(Mandatory=$true, HelpMessage="Path to the .uproject file (avoid OneDrive paths)")]
    [string]$ProjectPath,
    [Parameter(Mandatory=$false)]
    [string]$ArchiveDir = (Join-Path $PSScriptRoot "packaged")
)

Write-Host "Building and packaging TripSitter..." -ForegroundColor Cyan

# Clean intermediate files to force rebuild (derive from ProjectPath)
$ProjectDir = Split-Path -Parent $ProjectPath
$IntermediatePath = Join-Path $ProjectDir "Intermediate"
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
