
param(
    [string]$ProjectRoot
)


$desktopPath = [Environment]::GetFolderPath('Desktop')
$root = $ProjectRoot
if (-not $root) { $root = $PSScriptRoot }
if (-not $root) {
    Write-Error "ProjectRoot and PSScriptRoot are both unset. Cannot determine project root."
    Write-Error "Set -ProjectRoot or run from a valid script location."
    exit 1
}

$targetPath = Join-Path $root "build\tests\Release\test_backend_api.exe"
$workingDir = $root

# Verify target exists before creating shortcut
if (-not (Test-Path $targetPath)) {
    Write-Error "Target executable not found: $targetPath"
    Write-Error "Make sure you have built the project first (cmake --build build --config Release)"
    exit 1
}

$shortcutPath = Join-Path $desktopPath "BeatSync Backend Tests.lnk"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcutPath)
$s.TargetPath = $targetPath
$s.WorkingDirectory = $workingDir
$s.Save()
Write-Host "Shortcut created successfully at: $shortcutPath"
