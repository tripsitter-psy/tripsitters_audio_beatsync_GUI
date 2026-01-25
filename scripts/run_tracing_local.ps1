# Run tracing smoke test on Windows (PowerShell)
$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$buildDir = Join-Path $projectRoot "build"
if (!(Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
Push-Location $buildDir
try {
    cmake $projectRoot -G "Visual Studio 17 2022" -A x64 -DBEATSYNC_ENABLE_TRACING=ON -DBEATSYNC_ENABLE_TESTS=ON
    $configureExitCode = $LASTEXITCODE
    if ($configureExitCode -ne 0) {
        throw "CMake configure failed with exit code: $configureExitCode"
    }
    cmake --build . --config Debug -- /m
    $buildExitCode = $LASTEXITCODE
    if ($buildExitCode -ne 0) {
        throw "CMake build failed with exit code: $buildExitCode"
    }
    $traceOut = Join-Path $PWD 'beatsync-trace.log'
    $env:BEATSYNC_TRACE_OUT = $traceOut
    try {
        ctest -C Debug -R tracing -V --output-on-failure
        $ctestExitCode = $LASTEXITCODE
        if ($ctestExitCode -ne 0) {
            Write-Error "ctest failed with exit code: $ctestExitCode"
            exit $ctestExitCode
        }
        if (Test-Path $traceOut) {
            Write-Host "Tracing output written to: $traceOut"
        } else {
            Write-Error "No trace file found. Tracing may not have been enabled for the test run."
            exit 1
        }
    } finally {
        # Cleanup: remove the environment variable so it doesn't affect future runs
        Remove-Item Env:BEATSYNC_TRACE_OUT -ErrorAction SilentlyContinue
    }
} finally {
    Pop-Location
}
