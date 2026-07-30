# Create packaged build from installed version
$source = 'C:\Program Files\MTV TripSitter'
$dest = 'C:\Users\samue\Desktop\TripSitterBuild\Windows'

if (-not (Test-Path $source)) {
    Write-Error "Installation not found at $source"
    exit 1
}

Write-Host "Creating packaged build at $dest..."

# Create destination directory
New-Item -ItemType Directory -Path $dest -Force | Out-Null

# Copy TripSitter.exe as MyProject.exe (the launcher)
Copy-Item "$source\TripSitter.exe" "$dest\MyProject.exe" -Force
Write-Host "  Copied launcher as MyProject.exe"

# Copy manifest files
Copy-Item "$source\Manifest_*.txt" $dest -Force
Write-Host "  Copied manifests"

# Copy Engine folder
if (Test-Path "$source\Engine") {
    Copy-Item "$source\Engine" $dest -Recurse -Force
    Write-Host "  Copied Engine folder"
}

# Copy MyProject folder (game content)
if (Test-Path "$source\MyProject") {
    Copy-Item "$source\MyProject" $dest -Recurse -Force
    Write-Host "  Copied MyProject folder"
}

Write-Host "`nPackaged build created at: $dest"
Write-Host "Set UE_PACKAGED_BUILD=$dest in cmake"
