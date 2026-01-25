<#
.SYNOPSIS
    Unified release build script for MTV TripSitter.

.DESCRIPTION
    Builds the complete MTV TripSitter release package including:
    1. Backend DLL (beatsync_backend_shared.dll)
    2. TripSitter.exe (Unreal Engine GUI)
    3. NSIS installer
    4. Portable ZIP archive

.PARAMETER SkipBackend
    Skip building the backend DLL (use existing build).

.PARAMETER SkipTripSitter
    Skip building TripSitter.exe (use existing executable).

.PARAMETER SkipInstaller
    Skip creating the NSIS installer.

.PARAMETER SkipZip
    Skip creating the portable ZIP.

.PARAMETER Configuration
    Build configuration: Release or Debug. Default is Release.

.PARAMETER Version
    Version string for the installer. Default reads from CMakeLists.txt.

.PARAMETER DryRun
    Show what would be done without executing.

.EXAMPLE
    .\build_release.ps1
    Build everything with default settings.

.EXAMPLE
    .\build_release.ps1 -SkipTripSitter -Version "0.2.0-beta"
    Build backend and installer only, skip UE build, use custom version.

.EXAMPLE
    .\build_release.ps1 -DryRun
    Preview what would be built.
#>

[CmdletBinding()]
param(
    [switch]$SkipBackend,
    [switch]$SkipTripSitter,
    [switch]$SkipInstaller,
    [switch]$SkipZip,
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$Version = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$BuildDir = Join-Path $ProjectRoot "build"
$UE5Root = "C:\UE5_Source\UnrealEngine"
$TripSitterSource = Join-Path $ProjectRoot "unreal-prototype\Source\TripSitter"
$TripSitterDest = Join-Path $UE5Root "Engine\Source\Programs\TripSitter"
$TripSitterExe = Join-Path $UE5Root "Engine\Binaries\Win64\TripSitter.exe"
$VcpkgToolchain = Join-Path $ProjectRoot "vcpkg\scripts\buildsystems\vcpkg.cmake"
$AudioFluxRoot = "C:\audioFlux"

# Colors for output
function Write-Step { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Banner
Write-Host @"

  __  __ _______     __  _______     _       _____ _ _   _
 |  \/  |__   __|   / / |__   __|   (_)     / ____(_) | | |
 | \  / |  | |_   _/ /     | |_ __ _ _ _ __ | (___  _| |_| |_ ___ _ __
 | |\/| |  | \ \ / /      | | '__| | | '_ \ \___ \| | __| __/ _ \ '__|
 | |  | |  | |\ V /       | | |  | | | |_) |____) | | |_| ||  __/ |
 |_|  |_|  |_| \_/        |_|_|  |_|_| .__/|_____/|_|\__|\__\___|_|
                                     | |
                                     |_|  Release Build Script

"@ -ForegroundColor Magenta

Write-Host "Project Root: $ProjectRoot" -ForegroundColor DarkGray
Write-Host "Configuration: $Configuration" -ForegroundColor DarkGray
Write-Host "Dry Run: $DryRun" -ForegroundColor DarkGray
Write-Host ""

# Validate environment
Write-Step "Validating Environment"

if (-not (Test-Path $VcpkgToolchain)) {
    Write-Err "vcpkg toolchain not found at $VcpkgToolchain"
    Write-Host "Run: git submodule update --init --recursive"
    exit 1
}
Write-Success "vcpkg toolchain found"

if (-not $SkipTripSitter) {
    if (-not (Test-Path $UE5Root)) {
        Write-Err "Unreal Engine not found at $UE5Root"
        Write-Host "Set UE5Root or use -SkipTripSitter"
        exit 1
    }
    Write-Success "Unreal Engine found at $UE5Root"
}

# Check for NSIS
$makensis = Get-Command makensis -ErrorAction SilentlyContinue
if (-not $makensis -and -not $SkipInstaller) {
    Write-Warn "NSIS not found - installer will not be created"
    $SkipInstaller = $true
}

# Stage 1: Build Backend
if (-not $SkipBackend) {
    Write-Step "Stage 1: Building Backend DLL"

    $cmakeArgs = @(
        "-S", $ProjectRoot,
        "-B", $BuildDir,
        "-DCMAKE_BUILD_TYPE=$Configuration",
        "-DCMAKE_TOOLCHAIN_FILE=$VcpkgToolchain",
        "-DVCPKG_OVERLAY_TRIPLETS=$ProjectRoot\triplets"
    )

    # Add AudioFlux if available
    if (Test-Path $AudioFluxRoot) {
        $cmakeArgs += "-DAUDIOFLUX_ROOT=$AudioFluxRoot"
        Write-Host "AudioFlux: $AudioFluxRoot" -ForegroundColor DarkGray
    }

    # Add version if specified
    if ($Version) {
        $cmakeArgs += "-DBEATSYNC_VERSION=$Version"
    }

    if ($DryRun) {
        Write-Host "Would run: cmake $($cmakeArgs -join ' ')" -ForegroundColor Yellow
    } else {
        Write-Host "Configuring CMake..." -ForegroundColor DarkGray
        & cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Err "CMake configuration failed"
            exit 1
        }

        Write-Host "Building backend..." -ForegroundColor DarkGray
        & cmake --build $BuildDir --config $Configuration --target beatsync_backend_shared -- /m
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Backend build failed"
            exit 1
        }
        Write-Success "Backend DLL built successfully"
    }
} else {
    Write-Step "Stage 1: Skipping Backend Build"
}

# Stage 2: Deploy DLLs
Write-Step "Stage 2: Deploying DLLs"

$ContinueOnDeployFailure = $false
$deployScript = Join-Path $ScriptDir "deploy_tripsitter.ps1"
if (Test-Path $deployScript) {
    if ($DryRun) {
        Write-Host "Would run: $deployScript -DryRun" -ForegroundColor Yellow
    } else {
        $deployArgs = @()
        if ($DryRun) { $deployArgs += "-DryRun" }
        & $deployScript @deployArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "DLL deployment had issues - check output above"
            exit $LASTEXITCODE
        } else {
            Write-Success "DLLs deployed successfully"
        }
    }
} else {
    Write-Warn "deploy_tripsitter.ps1 not found - skipping DLL deployment"
}

# Stage 3: Build TripSitter.exe
if (-not $SkipTripSitter) {
    Write-Step "Stage 3: Building TripSitter.exe"

    # Sync source files
    Write-Host "Syncing source files to UE5..." -ForegroundColor DarkGray
    if ($DryRun) {
        Write-Host "Would copy: $TripSitterSource\Private\* -> $TripSitterDest\Private\" -ForegroundColor Yellow
    } else {
        if (-not (Test-Path "$TripSitterDest\Private")) {
            New-Item -ItemType Directory -Path "$TripSitterDest\Private" -Force | Out-Null
        }
        Copy-Item -Path "$TripSitterSource\Private\*" -Destination "$TripSitterDest\Private\" -Recurse -Force
        Write-Success "Source files synced"
    }

    # Build with UE5
    $ueBuildBat = Join-Path $UE5Root "Engine\Build\BatchFiles\Build.bat"
    if ($DryRun) {
        Write-Host "Would run: $ueBuildBat TripSitter Win64 $Configuration" -ForegroundColor Yellow
    } else {
        Write-Host "Building TripSitter with Unreal Engine (this may take a while)..." -ForegroundColor DarkGray
        & $ueBuildBat TripSitter Win64 $Configuration
        if ($LASTEXITCODE -ne 0) {
            Write-Err "TripSitter build failed"
            exit 1
        }

        if (Test-Path $TripSitterExe) {
            Write-Success "TripSitter.exe built successfully"
            $exeSize = (Get-Item $TripSitterExe).Length / 1MB
            Write-Host "  Size: $([math]::Round($exeSize, 2)) MB" -ForegroundColor DarkGray
        } else {
            Write-Err "TripSitter.exe not found after build"
            exit 1
        }
    }
} else {
    Write-Step "Stage 3: Skipping TripSitter Build"
    if (Test-Path $TripSitterExe) {
        Write-Success "Using existing TripSitter.exe"
    } else {
        Write-Warning "TripSitter.exe not found - installer will be incomplete"
    }
}

# Stage 4: Patch Application Icon
Write-Step "Stage 4: Patching Application Icon"

$IconFile = Join-Path $ProjectRoot "unreal-prototype\Source\TripSitter\Resources\TripSitter.ico"
$PackagedExe = "$env:USERPROFILE\Desktop\TripSitterBuild\Windows\TripSitter\Binaries\Win64\MyProject.exe"

if (Test-Path $IconFile) {
    if (Test-Path $PackagedExe) {
        # Check for rcedit
        $rcedit = Get-Command rcedit -ErrorAction SilentlyContinue
        if (-not $rcedit) {
            $rcedit = Get-Command "rcedit.exe" -ErrorAction SilentlyContinue
        }
        if (-not $rcedit -and (Test-Path "$ProjectRoot\tools\rcedit.exe")) {
            $rcedit = "$ProjectRoot\tools\rcedit.exe"
        }

        if ($rcedit) {
            if ($DryRun) {
                Write-Host "Would run: rcedit `"$PackagedExe`" --set-icon `"$IconFile`"" -ForegroundColor Yellow
            } else {
                Write-Host "Patching icon in packaged exe..." -ForegroundColor DarkGray
                & $rcedit $PackagedExe --set-icon $IconFile
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Application icon patched successfully"
                } else {
                    Write-Warn "Icon patching failed - app will have default UE icon"
                }
            }
        } else {
            Write-Warn "rcedit not found - install via 'npm install -g rcedit' or place rcedit.exe in tools/"
            Write-Host "  App will have default UE icon until rcedit is available" -ForegroundColor DarkGray
        }
    } else {
        Write-Warn "Packaged exe not found at $PackagedExe - skipping icon patch"
    }
} else {
    Write-Warn "Icon file not found at $IconFile"
}

# Stage 5: Create Installer
if (-not $SkipInstaller) {
    Write-Step "Stage 5: Creating NSIS Installer"

    if ($DryRun) {
        Write-Host "Would run: cpack -C $Configuration -G NSIS (in $BuildDir)" -ForegroundColor Yellow
    } else {
        Push-Location $BuildDir
        try {
            Write-Host "Running CPack NSIS..." -ForegroundColor DarkGray
            & cpack -C $Configuration -G NSIS
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "NSIS installer generation had issues"
            } else {
                $installer = Get-ChildItem -Path $BuildDir -Filter "*.exe" |
                    Where-Object { $_.Name -like "*TripSitter*" -or $_.Name -like "*MTV*" } |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1
                if ($installer) {
                    Write-Success "Installer created: $($installer.Name)"
                    $installerSize = $installer.Length / 1MB
                    Write-Host "  Size: $([math]::Round($installerSize, 2)) MB" -ForegroundColor DarkGray
                }
            }
        } finally {
            Pop-Location
        }
    }
} else {
    Write-Step "Stage 5: Skipping Installer"
}

# Stage 6: Create Portable ZIP
if (-not $SkipZip) {
    Write-Step "Stage 6: Creating Portable ZIP"

    if ($DryRun) {
        Write-Host "Would run: cpack -C $Configuration -G ZIP (in $BuildDir)" -ForegroundColor Yellow
    } else {
        Push-Location $BuildDir
        try {
            Write-Host "Running CPack ZIP..." -ForegroundColor DarkGray
            & cpack -C $Configuration -G ZIP
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "ZIP creation had issues"
            } else {
                $zipFile = Get-ChildItem -Path $BuildDir -Filter "*.zip" |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1
                if ($zipFile) {
                    Write-Success "ZIP created: $($zipFile.Name)"
                    $zipSize = $zipFile.Length / 1MB
                    Write-Host "  Size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor DarkGray
                }
            }
        } finally {
            Pop-Location
        }
    }
} else {
    Write-Step "Stage 6: Skipping ZIP"
}

# Summary
Write-Step "Build Complete"

Write-Host "`nOutput files:" -ForegroundColor Cyan
if (Test-Path $BuildDir) {
    $outputs = Get-ChildItem -Path $BuildDir -File |
        Where-Object { $_.Extension -in '.exe', '.zip' -and $_.Name -notlike "*uninstall*" } |
        Sort-Object Extension, Name

    if ($outputs) {
        foreach ($file in $outputs) {
            $size = $file.Length / 1MB
            Write-Host "  $($file.Name) ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
        }
    } else {
        Write-Host "  (no output files found)" -ForegroundColor Yellow
    }
}

Write-Host "`n" -NoNewline
Write-Host "Done!" -ForegroundColor Green
