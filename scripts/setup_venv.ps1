# Create and populate project-local .venv for development
param(
    [string]$Requirements = "scripts/requirements.txt"
)

Write-Host "Creating .venv in project root..."
python -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv. Ensure python is on PATH."; exit 1 }

$py = Join-Path -Path (Get-Location) -ChildPath ".venv\Scripts\python.exe"
Write-Host "Upgrading pip..."
& $py -m pip install -U pip setuptools wheel
Write-Host "Installing requirements from $Requirements..."
& $py -m pip install -r $Requirements

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"