# Start local OTLP collector + Jaeger (Windows PowerShell)
$cwd = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $cwd
Write-Host "Starting tracing collector (docker-compose up -d)..."
docker-compose up -d
Write-Host "Collector and Jaeger should be available. Jaeger UI: http://localhost:16686"
Pop-Location
