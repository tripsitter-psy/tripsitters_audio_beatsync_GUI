<#
checks a self-hosted GPU runner for common issues: NVIDIA visibility, Docker GPU access, nvcc presence, disk space in TEMP, and optional GitHub runner label check.
Usage:
  powershell -File scripts\check_gpu_runner.ps1

Requires: PowerShell 7+, optionally Docker and `nvidia-smi` in PATH.
#>


# Allow CUDA image to be set via parameter or environment variable, fallback to default
param(
    [string]$CudaImage = $env:CUDA_IMAGE
)
if (-not $CudaImage) {
    $CudaImage = 'nvidia/cuda:12.0-base'
}

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
    Write-Host "Docker found: testing container GPU access..."
    # Timeout after 120s to avoid indefinite hangs
    $job = Start-Job {
        try {
            $tempFile = [System.IO.Path]::GetTempFileName()
            $dockerArgs = @("run", "--rm", "--gpus", "all", $using:CudaImage, "nvidia-smi")
            $process = Start-Process -FilePath "docker" -ArgumentList $dockerArgs -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tempFile
            $output = Get-Content $tempFile -Raw
            Remove-Item $tempFile -ErrorAction SilentlyContinue
            return @{ ExitCode = $process.ExitCode; Output = $output }
        } catch {
            return @{ ExitCode = 1; Output = $_.Exception.Message }
        }
    }
    $completed = Wait-Job $job -Timeout 120
    $dockerGpuResult = @{ ExitCode = 1; Output = "" } # Default to failure
    $dockerGpuExitCode = 1
    if ($completed) {
        $dockerGpuResult = Receive-Job $job
        $dockerGpuExitCode = $dockerGpuResult.ExitCode
        if ($job.State -eq 'Completed' -and $dockerGpuResult.ExitCode -eq 0) {
            Write-Host "Docker GPU test succeeded (exit code 0)."
            Write-Host $dockerGpuResult.Output
        } else {
            Write-Host "Docker GPU test output:" -ForegroundColor Yellow
            Write-Host $dockerGpuResult.Output
            # Run again to show output for debugging
            try {
                $debugJob = Start-Job -ScriptBlock { param($img) docker run --rm --gpus all $img nvidia-smi 2>&1 } -ArgumentList $CudaImage
                $debugCompleted = Wait-Job $debugJob -Timeout 30
                if ($debugCompleted) {
                    $debugOutput = Receive-Job $debugJob
                    if ($debugOutput) {
                        Write-Host $debugOutput
                    }
                } else {
                    Stop-Job $debugJob
                    Write-Host "Docker debug run timed out after 30 seconds." -ForegroundColor Red
                }
                Remove-Job $debugJob -Force
            } catch {
                Write-Host "Failed to get Docker output for debugging: $_" -ForegroundColor Red
            }
        }
    } else {
        Stop-Job $job
        $dockerGpuExitCode = 1
        Write-Warning "Docker GPU test timed out after 120 seconds."
    }
    Remove-Job $job -Force
    if ($dockerGpuExitCode -ne 0) {
        Write-Warning "Docker GPU test failed (exit code $dockerGpuExitCode). Docker may not be configured for GPUs (nvidia container toolkit)."
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
$drive = $null
$freeGB = $null
try {
    $drive = [System.IO.DriveInfo]::new((Get-Item $temp).PSDrive.Root)
    $freeGB = [Math]::Round($drive.AvailableFreeSpace / 1GB, 2)
    Write-Host "Temp path: $temp (Free: $freeGB GB)"
    if ($drive.AvailableFreeSpace -lt 10GB) {
        Write-Warning "Less than 10GB free in TEMP. nvcc and builds may fail with 'ptxas' or 'No space left' errors."
    }
} catch {
    # Handle UNC/network path or other errors
    $psDrive = Get-PSDrive -Name ((Get-Item $temp).PSDrive.Name) -ErrorAction SilentlyContinue
    if ($psDrive -and $psDrive.Free) {
        $freeGB = [Math]::Round($psDrive.Free / 1GB, 2)
        Write-Host "Temp path: $temp (Free: $freeGB GB via PSDrive)"
        if ($psDrive.Free -lt 10GB) {
            Write-Warning "Less than 10GB free in TEMP. nvcc and builds may fail with 'ptxas' or 'No space left' errors."
        }
    } else {
        Write-Warning "Temp path: $temp is a network/UNC path or free space could not be determined. Skipping disk space check."
    }
}

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
            if (-not $runners) { Write-Warning "No self-hosted runners found for $env:GITHUB_REPOSITORY" }
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
            $script:GitHubApiFailed = $true
        }
    }
} else {
    Write-Host "No GITHUB_REPOSITORY provided — skipping GitHub runner label check. Set GITHUB_REPOSITORY='owner/repo' to enable."
}

# Final error check for GitHub API failure
if ($script:GitHubApiFailed) {
    Write-Error "GitHub API check failed. Exiting with error."
    exit 1
}

Write-Host "Validation done. Review warnings above and fix as needed."
