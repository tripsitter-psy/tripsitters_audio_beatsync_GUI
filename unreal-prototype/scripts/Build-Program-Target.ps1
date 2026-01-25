
# Build script for TripSitter Program target
#
# $UEPath can be overridden via the -UEPath parameter. Unreal Engine installs may live on other drives or under Epic Launcher GUID folders.
# If not provided or not found, this script will attempt to auto-detect the Unreal Engine installation from common registry locations and folders.
# If detection fails, you must pass -UEPath explicitly.
param(
    [string]$Configuration = "Development",
    [string]$Platform = "Win64",
    [string]$UEPath = "C:\Program Files\Epic Games\UE_5.7"
)

# Auto-detect UEPath if not passed or not found
if (-not (Test-Path $UEPath)) {
    $detectedUEPath = $null
    # Try registry (Epic Launcher)
    try {
        $regPaths = @(
            'HKLM:\SOFTWARE\WOW6432Node\Epic Games\Unreal Engine',
            'HKLM:\SOFTWARE\Epic Games\Unreal Engine'
        )
        foreach ($reg in $regPaths) {
            if (Test-Path $reg) {
                $keys = Get-ChildItem -Path $reg -ErrorAction SilentlyContinue
                foreach ($key in $keys) {
                    $InstallDir = (Get-ItemProperty -Path $key.PSPath -Name InstallLocation -ErrorAction SilentlyContinue).InstallLocation
                    if ($InstallDir -and (Test-Path $InstallDir)) {
                        $detectedUEPath = $InstallDir
                        break
                    }
                }
            }
            if ($detectedUEPath) { break }
        }
    } catch {
        Write-Verbose "Failed to query registry for Unreal Engine installations: $($_.Exception.Message)"
        Write-Debug "Full exception details: $($_.Exception | Out-String)"
    }
    # Try common folders
    if (-not $detectedUEPath) {
        $parentDirs = @(
            'C:\Program Files\Epic Games',
            'D:\Program Files\Epic Games'
        )
        $commonFolders = @()
        foreach ($parent in $parentDirs) {
            if (Test-Path $parent) {
                # Sort by numeric version (e.g., UE_5.10 > UE_5.9) instead of lexical
                $found = Get-ChildItem -Path $parent -Directory -Filter 'UE_*' | Sort-Object {
                    $versionStr = $_.Name -replace '^UE_', ''
                    try {
                        [System.Version]::Parse($versionStr)
                    } catch {
                        # Fallback for non-standard version formats
                        [System.Version]::new(0, 0)
                    }
                } -Descending | Select-Object -ExpandProperty FullName
                if ($found) {
                    $commonFolders += $found
                }
            }
        }
        if ($commonFolders.Count -gt 0) {
            $detectedUEPath = $commonFolders[0]
        }
    }
    if ($detectedUEPath) {
        $UEPath = $detectedUEPath
        Write-Host "Auto-detected Unreal Engine path: $UEPath" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Unreal Engine path not found. Please pass -UEPath with the correct install location." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Building TripSitter Program Target" -ForegroundColor Cyan
Write-Host "Configuration: $Configuration" -ForegroundColor Cyan
Write-Host "Platform: $Platform" -ForegroundColor Cyan
Write-Host "UE Path: $UEPath" -ForegroundColor Cyan
Write-Host ""

# Check if Engine symlink exists
$EngineSymlink = Join-Path $PSScriptRoot "..\Engine"
if (!(Test-Path $EngineSymlink)) {
    Write-Host "ERROR: Engine symlink not found. Please run Setup-Engine-Symlink.ps1 as Administrator first." -ForegroundColor Red
    exit 1
}

# Verify UBT exists
$UBTPath = Join-Path $UEPath "Engine\Binaries\DotNET\UnrealBuildTool\UnrealBuildTool.exe"
if (!(Test-Path $UBTPath)) {
    $UBTPath = Join-Path $UEPath "Engine\Binaries\Win64\UnrealBuildTool.exe"
    if (!(Test-Path $UBTPath)) {
        Write-Host "ERROR: UnrealBuildTool not found at expected locations." -ForegroundColor Red
        exit 1
    }
}

# Get project path
$ProjectPath = Join-Path $PSScriptRoot "..\TripSitter.uproject"
if (!(Test-Path $ProjectPath)) {
    Write-Host "ERROR: Project file not found: $ProjectPath" -ForegroundColor Red
    exit 1
}

Write-Host "Building Program target..." -ForegroundColor Yellow

# Build the program

# Use named -Project flag and add -log argument
$LogPath = Join-Path $PSScriptRoot "UBT-TripSitter.log"
$BuildArgs = @(
    "TripSitter",
    $Platform,
    $Configuration,
    "-Project=`"$ProjectPath`"",
    "-WaitMutex",
    "-FromMsBuild",
    "-log=`"$LogPath`""
)

Write-Host "Running: $UBTPath $BuildArgs" -ForegroundColor Gray

# Capture stdout/stderr and exit code
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $UBTPath
$psi.Arguments = [string]::Join(' ', $BuildArgs)
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$script:stdout = ""
$script:stderr = ""
$proc.add_OutputDataReceived({ if ($_.Data) { $script:stdout += $_.Data + "`n" } })
$proc.add_ErrorDataReceived({ if ($_.Data) { $script:stderr += $_.Data + "`n" } })
$proc.Start() | Out-Null
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

# Wait for process with timeout (e.g., 30 min = 1800000 ms)
$timeoutMillis = 1800000
if (-not $proc.WaitForExit($timeoutMillis)) {
    Write-Host "ERROR: Build process timed out after $($timeoutMillis/60000) minutes. Killing process..." -ForegroundColor Red
    Write-Host "See log: $LogPath" -ForegroundColor Red
    $proc.Kill()
    $proc.WaitForExit()
    # Exit immediately with non-zero code - don't rely on $proc.ExitCode after kill
    exit 1
}

# Wait indefinitely for remaining output streams to flush
$proc.WaitForExit()

Write-Host $script:stdout
if ($script:stderr) { Write-Host $script:stderr -ForegroundColor Yellow }

if ($proc.ExitCode -ne 0) {
    Write-Host "ERROR: Build failed with exit code $($proc.ExitCode)" -ForegroundColor Red
    Write-Host "See log: $LogPath" -ForegroundColor Red
    exit $proc.ExitCode
}

Write-Host "Build completed successfully!" -ForegroundColor Green

# Check if executable was created
$ExePath = Join-Path $PSScriptRoot "..\Binaries\$Platform\TripSitter.exe"
if (Test-Path $ExePath) {
    Write-Host "Executable created: $ExePath" -ForegroundColor Green
} else {
    Write-Host "WARNING: Executable not found at expected location: $ExePath" -ForegroundColor Yellow
}

Write-Host "Build script completed." -ForegroundColor Green