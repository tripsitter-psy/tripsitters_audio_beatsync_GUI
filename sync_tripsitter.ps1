param(
    [string]$UE = 'D:\UnrealEngine',
    [string]$RepoRoot = $PSScriptRoot
)

$Src = Join-Path $RepoRoot 'unreal-prototype\Source\TripSitter'
$Dst = Join-Path $UE 'Engine\Source\Programs\TripSitter'

Write-Host "Creating destination if needed: $Dst"
New-Item -ItemType Directory -Path $Dst -Force | Out-Null
Write-Host "Copying from $Src to $Dst"
Copy-Item -Path "$Src\*" -Destination $Dst -Recurse -Force -ErrorAction Stop
Write-Host "SUCCESS: Synced TripSitter source. Files copied:"
Get-ChildItem $Dst -Recurse | ForEach-Object { Write-Host $_.FullName }
Write-Host "Sync complete. Source count: $((Get-ChildItem $Src -Recurse | Measure-Object).Count) files/directories."