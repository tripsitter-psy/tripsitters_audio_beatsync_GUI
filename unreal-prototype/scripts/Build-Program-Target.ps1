
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
    } catch {}
    # Try common folders
    if (-not $detectedUEPath) {
        $commonFolders = @(
            'C:\Program Files\Epic Games\UE_5.7',
            'C:\Program Files\Epic Games\UE_5.3',
            'D:\Program Files\Epic Games\UE_5.7',
            'D:\Program Files\Epic Games\UE_5.3'
        )
        foreach ($folder in $commonFolders) {
            if (Test-Path $folder) {
                $detectedUEPath = $folder
                break
            }
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
$BuildArgs = @(
    "TripSitter",
    $Platform,
    $Configuration,
    "`"$ProjectPath`"",
    "-WaitMutex",
    "-FromMsBuild"
)

Write-Host "Running: $UBTPath $BuildArgs" -ForegroundColor Gray

$proc = Start-Process -FilePath $UBTPath -ArgumentList $BuildArgs -Wait -PassThru -NoNewWindow

if ($proc.ExitCode -ne 0) {
    Write-Host "ERROR: Build failed with exit code $($proc.ExitCode)" -ForegroundColor Red
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