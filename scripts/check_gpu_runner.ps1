<#
checks a self-hosted GPU runner for common issues: NVIDIA visibility, Docker GPU access, nvcc presence, disk space in TEMP, and optional GitHub runner label check.
Usage:
  powershell -File scripts\check_gpu_runner.ps1 [-GithubRepo "owner/repo"]

Requires: PowerShell 7+, optionally Docker and `nvidia-smi` in PATH.
#>
param(
    [string]$GithubRepo = $env:GITHUB_REPOSITORY
)

Write-Host "== GPU Runner Validation Script =="
Write-Host "Checking NVIDIA GPU visibility..."
try {
    & nvidia-smi | Out-Host
} catch {
    Write-Warning "nvidia-smi not found or failed. Make sure NVIDIA driver is installed and nvidia-smi is on PATH."
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "Docker found — testing container GPU access..."
    try {
        docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi | Out-Host
    } catch {
        Write-Warning "Docker GPU test failed or Docker is not configured for GPUs (nvidia container toolkit)."
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

if ($GithubRepo) {
    if (-not $env:GITHUB_TOKEN) {
        Write-Warning "GITHUB_TOKEN not set — skipping GitHub runner label check."
    } else {
        Write-Host "Checking GitHub self-hosted runners for repo: $GithubRepo (requires repo-level token)..."
        $uri = "https://api.github.com/repos/$GithubRepo/actions/runners"
        try {
            $headers = @{ Authorization = "token $env:GITHUB_TOKEN"; 'User-Agent' = 'check-gpu-runner-script' }
            $resp = Invoke-RestMethod -Uri $uri -Headers $headers -ErrorAction Stop
            $runners = $resp.runners
            if (-not $runners) { Write-Warning "No self-hosted runners found for $GithubRepo"; exit 0 }
            foreach ($r in $runners) {
                $labels = ($r.labels | ForEach-Object { $_.name }) -join ','
                Write-Host "Runner: $($r.name) — Labels: $labels"
            }
            Write-Host "Looking for a runner with label 'gpu'..."
            $hasGpu = $runners | Where-Object { $_.labels.name -contains 'gpu' }
            if ($hasGpu) { Write-Host "Found runner(s) with 'gpu' label." } else { Write-Warning "No runner with 'gpu' label found in repo runners." }
        } catch {
            Write-Warning "GitHub API request failed: $_"
        }
    }
} else {
    Write-Host "No GITHUB_REPOSITORY provided — skipping GitHub runner label check. Set GITHUB_REPOSITORY or pass -GithubRepo 'owner/repo' to enable."
}

Write-Host "Validation done. Review warnings above and fix as needed."