# FFmpeg Setup Script for BeatSync Editor
# This script downloads and sets up FFmpeg development libraries

param(
    [string]$FFmpegArchive = "C:\Users\samue\Desktop\ffmpeg-dev.7z",
    [string]$ExtractPath = "C:\ffmpeg-dev"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BeatSync Editor - FFmpeg Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if 7-Zip is installed
$7zipPaths = @(
    "C:\Program Files\7-Zip\7z.exe",
    "C:\Program Files (x86)\7-Zip\7z.exe"
)

$7zExe = $null
foreach ($path in $7zipPaths) {
    if (Test-Path $path) {
        $7zExe = $path
        break
    }
}

if (-not $7zExe) {
    Write-Host "7-Zip not found. Installing 7-Zip..." -ForegroundColor Yellow

    # Try to install via winget
    try {
        winget install --id 7zip.7zip --silent --accept-package-agreements --accept-source-agreements
        Start-Sleep -Seconds 5

        # Check again
        foreach ($path in $7zipPaths) {
            if (Test-Path $path) {
                $7zExe = $path
                break
            }
        }
    } catch {
        Write-Host "Could not install 7-Zip automatically." -ForegroundColor Red
        Write-Host "Please download and install 7-Zip from: https://www.7-zip.org/" -ForegroundColor Yellow
        Write-Host "Then run this script again." -ForegroundColor Yellow
        exit 1
    }
}

if (-not $7zExe) {
    Write-Host "ERROR: 7-Zip is required but not found." -ForegroundColor Red
    Write-Host "Please install from: https://www.7-zip.org/" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found 7-Zip at: $7zExe`n" -ForegroundColor Green

# Check if archive exists
if (-not (Test-Path $FFmpegArchive)) {
    Write-Host "Downloading FFmpeg..." -ForegroundColor Yellow
    $url = "https://github.com/GyanD/codexffmpeg/releases/download/2025.12.29/ffmpeg-2025.12.29-full_build-shared.7z"

    try {
        Invoke-WebRequest -Uri $url -OutFile $FFmpegArchive -UseBasicParsing
        Write-Host "Download complete!`n" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Could not download FFmpeg." -ForegroundColor Red
        Write-Host "Please download manually from:" -ForegroundColor Yellow
        Write-Host $url -ForegroundColor Cyan
        Write-Host "Save to: $FFmpegArchive" -ForegroundColor Cyan
        exit 1
    }
} else {
    Write-Host "Found FFmpeg archive: $FFmpegArchive`n" -ForegroundColor Green
}

# Extract archive
Write-Host "Extracting FFmpeg to $ExtractPath..." -ForegroundColor Yellow

if (Test-Path $ExtractPath) {
    Write-Host "Removing existing directory..." -ForegroundColor Yellow
    Remove-Item -Path $ExtractPath -Recurse -Force
}

New-Item -ItemType Directory -Path $ExtractPath -Force | Out-Null

# Extract using 7-Zip
& $7zExe x $FFmpegArchive -o"$ExtractPath" -y

# Find the extracted directory (it's usually nested)
$extractedDirs = Get-ChildItem -Path $ExtractPath -Directory
if ($extractedDirs.Count -eq 1) {
    $nestedDir = $extractedDirs[0].FullName

    # Check if it has the expected structure
    if ((Test-Path "$nestedDir\include") -and (Test-Path "$nestedDir\lib")) {
        Write-Host "`nFFmpeg extracted successfully!" -ForegroundColor Green
        Write-Host "Location: $nestedDir`n" -ForegroundColor Green

        # Configure CMake
        Write-Host "Configuring CMake..." -ForegroundColor Yellow
        Set-Location "$PSScriptRoot\build"

        cmake .. -DFFMPEG_DIR="$nestedDir"

        if ($LASTEXITCODE -eq 0) {
            Write-Host "`nCMake configuration successful!" -ForegroundColor Green
            Write-Host "`nNext steps:" -ForegroundColor Cyan
            Write-Host "1. Build the project:" -ForegroundColor White
            Write-Host "   cd build" -ForegroundColor Gray
            Write-Host "   cmake --build . --config Release" -ForegroundColor Gray
            Write-Host "`n2. Run the analyzer:" -ForegroundColor White
            Write-Host "   cd bin\Release" -ForegroundColor Gray
            Write-Host "   beatsync.exe analyze song.mp3" -ForegroundColor Gray
        } else {
            Write-Host "`nCMake configuration failed. Please check the errors above." -ForegroundColor Red
        }
    } else {
        Write-Host "ERROR: Unexpected archive structure." -ForegroundColor Red
        Write-Host "Expected to find 'include' and 'lib' directories." -ForegroundColor Yellow
    }
} else {
    Write-Host "ERROR: Could not determine extracted directory structure." -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
