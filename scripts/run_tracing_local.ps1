# Run tracing smoke test on Windows (PowerShell)
$ErrorActionPreference = 'Stop'
$buildDir = "build"
if (!(Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
Push-Location $buildDir
try {
    cmake .. -G "Visual Studio 17 2022" -A x64 -DBEATSYNC_ENABLE_TRACING=ON
    $configureExitCode = $LASTEXITCODE
    if ($configureExitCode -ne 0) {
        Write-Error "CMake configure failed with exit code: $configureExitCode"
        exit $configureExitCode
    }
    cmake --build . --config Debug -- /m
    $buildExitCode = $LASTEXITCODE
    if ($buildExitCode -ne 0) {
        Write-Error "CMake build failed with exit code: $buildExitCode"
        exit $buildExitCode
    }
    $traceOut = Join-Path $PWD 'beatsync-trace.log'
    $env:BEATSYNC_TRACE_OUT = $traceOut
    ctest -C Debug -R tracing -V --output-on-failure
    $ctestExitCode = $LASTEXITCODE
    if ($ctestExitCode -ne 0) {
        Write-Error "ctest failed with exit code: $ctestExitCode"
        exit $ctestExitCode
    }
    if (Test-Path $traceOut) {
        Write-Host "Tracing output written to: $traceOut"
    } else {
        Write-Host "No trace file found. Tracing may not have been enabled for the test run."
    }
} finally {
    Pop-Location
}
