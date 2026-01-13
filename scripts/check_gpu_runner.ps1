<#
checks a self-hosted GPU runner for common issues: NVIDIA visibility, Docker GPU access, nvcc presence, disk space in TEMP, and optional GitHub runner label check.
Usage:
  powershell -File scripts\check_gpu_runner.ps1

Requires: PowerShell 7+, optionally Docker and `nvidia-smi` in PATH.
#>

Write-Host "== GPU Runner Validation Script =="
Write-Host "Checking NVIDIA GPU visibility..."
if (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue) {
    & nvidia-smi | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "nvidia-smi found but failed with exit code $LASTEXITCODE. Check NVIDIA driver installation."
    }
} else {
    Write-Warning "nvidia-smi not found. Make sure NVIDIA driver is installed and nvidia-smi is on PATH."
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "Docker found — testing container GPU access..."
    docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Docker GPU test failed with exit code $LASTEXITCODE. Docker may not be configured for GPUs (nvidia container toolkit)."
    }
} else {
    Write-Host "Docker not found — skipping Docker GPU test."
}

Write-Host "Checking CUDA compiler (nvcc) presence..."
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    nvcc --version | Out-Host
} else {
    Write-Warning "nvcc not found on PATH. If you plan to build ONNX Runtime with CUDA, ensure nvcc is installed and on PATH."
}

Write-Host "Checking TEMP disk free space..."
$temp = [System.IO.Path]::GetTempPath()
$drive = [System.IO.DriveInfo]::new((Get-Item $temp).PSDrive.Root)
Write-Host "Temp path: $temp (Free: $([Math]::Round($drive.AvailableFreeSpace / 1GB, 2)) GB)"
if ($drive.AvailableFreeSpace -lt 10GB) { Write-Warning "Less than 10GB free in TEMP. nvcc and builds may fail with 'ptxas' or 'No space left' errors." }

if ($env:GITHUB_REPOSITORY) {
    if (-not $env:GITHUB_TOKEN) {
        Write-Warning "GITHUB_TOKEN not set — skipping GitHub runner label check."
    } else {
        Write-Host "Checking GitHub self-hosted runners for repo: $env:GITHUB_REPOSITORY (requires repo-level token)..."
        $uri = "https://api.github.com/repos/$env:GITHUB_REPOSITORY/actions/runners"
        try {
            $headers = @{ Authorization = "token $env:GITHUB_TOKEN"; 'User-Agent' = 'check-gpu-runner-script' }
            $resp = Invoke-RestMethod -Uri $uri -Headers $headers -ErrorAction Stop
            $runners = $resp.runners
            if (-not $runners) { Write-Warning "No self-hosted runners found for $env:GITHUB_REPOSITORY"; exit 0 }
            foreach ($r in $runners) {
                $labels = ($r.labels | ForEach-Object { $_.name }) -join ','
                Write-Host "Runner: $($r.name) — Labels: $labels"
            }
            Write-Host "Looking for a runner with label 'gpu'..."
            $hasGpu = $runners | Where-Object { $_.labels.name -contains 'gpu' }
            if ($hasGpu) { Write-Host "Found runner(s) with 'gpu' label." } else { Write-Warning "No runner with 'gpu' label found in repo runners." }
        } catch {
            $errorMsg = $_.Exception.Message
            $statusCode = $null
            if ($_.Exception.Response) {
                $statusCode = $_.Exception.Response.StatusCode
                $statusDesc = $_.Exception.Response.StatusDescription
                if ($statusCode) {
                    $errorMsg = "$errorMsg (status: $statusCode $statusDesc)"
                }
            }
            Write-Warning "GitHub API request failed: $errorMsg"
        }
    }
} else {
    Write-Host "No GITHUB_REPOSITORY provided — skipping GitHub runner label check. Set GITHUB_REPOSITORY='owner/repo' to enable."
}

Write-Host "Validation done. Review warnings above and fix as needed."