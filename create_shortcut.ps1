
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
$s = $null
$shell = $null
try {
    $shell = New-Object -ComObject WScript.Shell
    $s = $shell.CreateShortcut($shortcutPath)
    $s.TargetPath = $targetPath
    $s.WorkingDirectory = $workingDir
    $s.Description = "Shortcut to run BeatSync backend tests."
    $s.Save()
    Write-Host "Shortcut created successfully at: $shortcutPath"
} catch {
    Write-Error "Failed to create shortcut at $shortcutPath: $($_.Exception.Message)"
    exit 1
} finally {
    # Cleanup COM object if needed
    if ($null -ne $s) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($s) | Out-Null }
    if ($null -ne $shell) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($shell) | Out-Null }
}
