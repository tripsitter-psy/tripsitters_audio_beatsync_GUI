<#
Downloads specified ONNX Runtime ZIP release for Windows x64, extracts it to a temp folder, updates PATH, and runs the helper test.
Usage:
  .\scripts\run_official_onnx_test.ps1 -Version 1.18.1

Note: Requires PowerShell 7+, `Expand-Archive` and that you have network access.
#>
param(
    [string]$Version = "1.23.2"
)

$baseUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/"
# Common file name pattern - may need adjustment per release
$zipName = "onnxruntime-win-x64-$Version.zip"
$url = $baseUrl + $zipName
$outDir = Join-Path $env:TEMP ("onnxruntime_$Version")
$zipPath = Join-Path $env:TEMP $zipName

Write-Host "Downloading $url..."
try {
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
} catch {
    Write-Error "Failed to download $url : $($_.Exception.Message)"
    exit 1
}

# Verify download succeeded
if (!(Test-Path $zipPath) -or (Get-Item $zipPath).Length -eq 0) {
    Write-Error "Downloaded file $zipPath is missing or empty"
    exit 1
}

if (Test-Path $outDir) {
    try {
        Remove-Item $outDir -Recurse -Force -ErrorAction Stop
    } catch {
        Write-Warning "Failed to remove existing directory $outDir : $($_.Exception.Message)"
        # Continue anyway - extraction will overwrite
    }
}

Write-Host "Extracting to $outDir"
try {
    Expand-Archive -LiteralPath $zipPath -DestinationPath $outDir
    # Cleanup: remove the downloaded ZIP after extraction
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
} catch {
    Write-Error "Failed to extract archive: $zipPath -> $outDir"
    Write-Error $_
    if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force -ErrorAction SilentlyContinue }
    exit 2
}

# Wrap all remaining work in try/finally to ensure $outDir cleanup
$scriptExitCode = 0
try {
    # Find the runtime bin folder (search for onnxruntime.dll)
    $bin = Get-ChildItem -Path $outDir -Recurse -Filter onnxruntime.dll | Select-Object -First 1
    if (-not $bin) {
        Write-Error "Could not find onnxruntime.dll in extracted archive."
        $scriptExitCode = 1
        throw "onnxruntime.dll not found"
    }
    $binDir = $bin.DirectoryName
    Write-Host "Found runtime in: $binDir"

    # Add to PATH for this session
    $env:Path = $binDir + ";" + $env:Path

    # Build & run helper test

    Write-Host "Configuring & building (with tracing)"
    cmake -S . -B build -DUSE_TRACING=ON
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configuration failed with exit code $LASTEXITCODE"
        $scriptExitCode = $LASTEXITCODE
        throw "CMake configuration failed"
    }
    cmake --build build --config Debug -- /m
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake build failed with exit code $LASTEXITCODE"
        $scriptExitCode = $LASTEXITCODE
        throw "CMake build failed"
    }

    Write-Host "Running helper test (official ONNX $Version)"
    Push-Location build
    try {
        ctest -C Debug -R onnx_inference_helper -V --output-on-failure
        $scriptExitCode = $LASTEXITCODE
        if ($scriptExitCode -ne 0) {
            Write-Error "ctest failed with exit code: $scriptExitCode"
        }
    } finally {
        Pop-Location
    }

    if ($scriptExitCode -ne 0) {
        Write-Host "If the helper crashed, consider running ProcDump to capture a full .dmp file."
    }
} finally {
    # Always cleanup the extracted temp directory
    if (Test-Path $outDir) {
        Write-Host "Cleaning up temp directory: $outDir"
        Remove-Item $outDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($scriptExitCode -ne 0) {
        exit $scriptExitCode
    }
}
