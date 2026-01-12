# Run tracing smoke test on Windows (PowerShell)
$ErrorActionPreference = 'Stop'
$buildDir = "build"
if (!(Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
Push-Location $buildDir
try {
    cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_TRACING=ON
    cmake --build . --config Debug -- /m
    $traceOut = Join-Path $PWD 'beatsync-trace.log'
    $env:BEATSYNC_TRACE_OUT = $traceOut
    ctest -C Debug -R tracing -V --output-on-failure
    if (Test-Path $traceOut) { Write-Host "Tracing output written to: $traceOut" } else { Write-Host "No trace file found. Tracing may not have been enabled for the test run." }
} finally {
    Pop-Location
}
