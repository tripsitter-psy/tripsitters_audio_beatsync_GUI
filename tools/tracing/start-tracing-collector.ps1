
# Start local OTLP collector + Jaeger (Windows PowerShell)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$cwd = Split-Path -Parent $MyInvocation.MyCommand.Path
try {
	Push-Location $cwd
	Write-Host "Starting tracing collector (docker compose up -d)..."
	docker compose up -d
	if ($LASTEXITCODE -ne 0) {
		throw "docker compose failed with exit code $LASTEXITCODE"
	}
	Write-Host "Collector and Jaeger should be available. Jaeger UI: http://localhost:16686"
} finally {
	Pop-Location
}
