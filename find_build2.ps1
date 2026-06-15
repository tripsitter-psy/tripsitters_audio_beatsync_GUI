# Check common locations for UE packaged builds
$paths = @(
    'D:\UnrealEngine\Engine\Programs\TripSitter\Saved\StagedBuilds',
    'C:\Users\samue\Documents',
    'C:\Users\samue\OneDrive\Desktop'
)

foreach ($p in $paths) {
    if (Test-Path $p) {
        Write-Host "Checking $p..."
        $found = Get-ChildItem $p -Filter 'MyProject.exe' -Recurse -Depth 5 -ErrorAction SilentlyContinue | Select-Object -First 2
        if ($found) {
            $found | ForEach-Object { Write-Host "  FOUND: $($_.FullName)" }
        }
    }
}

# Also check the installed location to copy it back
$installed = 'C:\Program Files\MTV TripSitter'
if (Test-Path $installed) {
    Write-Host "`nInstalled version at: $installed"
    Write-Host "We can create a packaged build from this installation"
}
