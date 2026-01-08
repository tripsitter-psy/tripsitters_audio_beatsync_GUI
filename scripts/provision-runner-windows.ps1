<#
Safe helper to provision a self-hosted GitHub Actions runner on Windows.
This script installs common dev packages (via Chocolatey) and downloads the GitHub Actions runner.

IMPORTANT: Runner registration requires a repository/org-specific token; the script will print the registration command for you to run manually with a token.
#>
param(
    [Parameter(Mandatory=$true)][string]$RepoUrl,
    [string]$RunnerName = "ue5-runner",
    [string]$Labels = "self-hosted,windows,ue5-5.3",
    [string]$WorkDir = "C:\actions-runner",
    [string]$UE5Root,
    [string]$GithubPat = $env:GITHUB_PAT,
    [switch]$AutoRegister,
    [switch]$RunSmoke,
    [switch]$UploadGist
) 

function Ensure-Admin {
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        Write-Error "This script must be run as Administrator. Please re-run in elevated PowerShell."; exit 2
    }
}

Ensure-Admin

# Install Chocolatey if missing (non-interactive)
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Common packages
$pkgs = @('git','7zip','cmake','ninja','pwsh','ffmpeg')
Write-Host "Installing packages: $($pkgs -join ', ')"
choco install -y $pkgs

# Visual Studio Build Tools (optional but recommended for UE builds)
Write-Host "Installing Visual Studio 2022 Build Tools (this may take a while)..."
choco install -y visualstudio2022buildtools --ignore-checksums || Write-Warning "visualstudio2022buildtools install failed or interactive prompts required; install manually via Visual Studio installer."

# Create runner work dir
if (-not (Test-Path $WorkDir)) { New-Item -ItemType Directory -Path $WorkDir | Out-Null }
Set-Location $WorkDir

# Download latest runner package
$arch = if ([Environment]::Is64BitOperatingSystem) { 'x64' } else { 'x86' }
$runnerUrl = "https://api.github.com/repos/actions/runner/releases/latest" | Invoke-RestMethod | Select-Object -ExpandProperty assets | Where-Object { $_.name -like "actions-runner-windows-$arch-*" } | Select-Object -First 1 -ExpandProperty browser_download_url

Write-Host "Downloading runner package..."
Invoke-WebRequest -Uri $runnerUrl -OutFile "$WorkDir\actions-runner.zip"

# Extract
Expand-Archive -Path "$WorkDir\actions-runner.zip" -DestinationPath $WorkDir -Force

# Auto-register if requested and a PAT is provided
if ($AutoRegister.IsPresent -and ($GithubPat -or $env:GITHUB_PAT)) {
    $pat = if ($GithubPat) { $GithubPat } else { $env:GITHUB_PAT }
    if ($RepoUrl -match "github.com/([^/]+)/([^/]+)(\.git)?") {
        $owner = $matches[1]; $repo = $matches[2]
        try {
            Write-Host "Requesting ephemeral registration token from GitHub..."
            $resp = Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$owner/$repo/actions/runners/registration-token" -Headers @{ Authorization = "token $pat"; 'User-Agent' = 'provision-runner-script' }
            $regToken = $resp.token
            if (-not $regToken) { throw "No token returned" }
            Write-Host "Configuring runner (unattended)..."
            & .\config.cmd --url $RepoUrl --token $regToken --name $RunnerName --labels $Labels --work _work --unattended --replace
            Write-Host "Installing runner as service..."
            & .\svc.sh install
            & .\svc.sh start

            # Verify runner appears online in GitHub (poll API)
            try {
                $maxRetries = 12
                $sleepSec = 5
                $found = $false
                for ($i=0; $i -lt $maxRetries; $i++) {
                    Start-Sleep -Seconds $sleepSec
                    $runners = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$owner/$repo/actions/runners" -Headers @{ Authorization = "token $pat"; 'User-Agent' = 'provision-runner-script' }
                    if ($runners.runners) {
                        foreach ($r in $runners.runners) {
                            if ($r.name -eq $RunnerName -and $r.status -eq 'online') { $found = $true; break }
                        }
                    }
                    if ($found) { break }
                }
                if ($found) { Write-Host "Runner '$RunnerName' is online on GitHub." } else { Write-Warning "Runner did not appear online within expected time. Check runner service logs and GitHub UI." }

                # Optionally dispatch a smoke workflow to verify the runner executes jobs
                if ($RunSmoke.IsPresent -and $pat) {
                    try {
                        Write-Host "Dispatching runner smoke workflow..."
                        $dispatchBody = @{ ref = 'ci/nsis-smoke-test'; inputs = @{ expected_runner_name = $RunnerName } } | ConvertTo-Json -Depth 4
                        Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$owner/$repo/actions/workflows/runner-smoke.yml/dispatches" -Headers @{ Authorization = "token $pat"; 'User-Agent' = 'provision-runner-script'; 'Accept' = 'application/vnd.github+json' } -Body $dispatchBody
                        Write-Host "Smoke workflow dispatched; checking run status and attempting to download artifact..."

                        # Poll for the workflow run completion and download artifact if present
                        try {
                            $runId = $null
                            $maxAttempts = 30
                            for ($k=0; $k -lt $maxAttempts; $k++) {
                                Start-Sleep -Seconds 5
                                $runs = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$owner/$repo/actions/workflows/runner-smoke.yml/runs?per_page=10" -Headers @{ Authorization = "token $pat"; 'User-Agent' = 'provision-runner-script' }
                                foreach ($r in $runs.workflow_runs) {
                                    if ($r.created_at -gt (Get-Date).AddMinutes(-10)) {
                                        if ($r.status -eq 'completed') { $runId = $r.id; break }
                                    }
                                }
                                if ($runId) { break }
                            }

                            if ($runId) {
                                Write-Host "Found completed smoke run: $runId. Fetching artifacts..."
                                $arts = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$owner/$repo/actions/runs/$runId/artifacts" -Headers @{ Authorization = "token $pat"; 'User-Agent' = 'provision-runner-script' }
                                $zipUrl = $null
                                foreach ($a in $arts.artifacts) { if ($a.name -eq 'runner-smoke') { $zipUrl = $a.archive_download_url; break } }
                                if ($zipUrl) {
                                    $outZip = Join-Path $WorkDir "runner-smoke.zip"
                                    Write-Host "Downloading artifact to $outZip"
                                    Invoke-WebRequest -Uri $zipUrl -Headers @{ Authorization = "token $pat"; 'User-Agent' = 'provision-runner-script' } -OutFile $outZip -UseBasicParsing
                                    $dest = Join-Path $WorkDir "runner-smoke"
                                    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
                                    Expand-Archive -Path $outZip -DestinationPath $dest -Force
                                    Write-Host "Downloaded and extracted runner-smoke to $dest"

                                    # Discover and record interesting artifacts extracted from the runner-smoke artifact
                                    $artifactList = @()
                                    $wallpaperFile = Join-Path $dest 'wallpaper_check.txt'
                                    if (Test-Path $wallpaperFile) { $artifactList += $wallpaperFile; Write-Host "Found artifact: $wallpaperFile" }
                                    $provisionGist = Join-Path $WorkDir 'provision-gist-url.txt'
                                    if (Test-Path $provisionGist) { $artifactList += $provisionGist; Write-Host "Found artifact: $provisionGist" }
                                    if ($artifactList.Count -gt 0) { $artifactList | Out-File -FilePath (Join-Path $WorkDir 'provision-artifacts.txt'); Write-Host "Provision artifacts recorded in: $(Join-Path $WorkDir 'provision-artifacts.txt')" }

                                    # Validate artifact contents (smoke.txt must contain 'Smoke OK')
                                    $smokeFile = Join-Path $dest "smoke.txt"
                                    if (Test-Path $smokeFile) {
                                        $content = Get-Content $smokeFile -Raw
                                        if ($content -match "Smoke OK") {
                                            Write-Host "Smoke artifact validation passed."
                                            $result = @{ status = 'success'; timestamp = (Get-Date).ToString('o'); runner = $RunnerName; artifact = $dest; message = 'Smoke artifact validated' }
                                            $result | ConvertTo-Json | Out-File -FilePath (Join-Path $WorkDir 'provision-result.json') -Encoding UTF8
                                            # If we discovered additional artifacts, attach them and rewrite result
                                            if ($artifactList -and $artifactList.Count -gt 0) {
                                                $result.artifacts = $artifactList
                                                $result | ConvertTo-Json | Out-File -FilePath (Join-Path $WorkDir 'provision-result.json') -Encoding UTF8
                                                Write-Host "Updated provision-result.json with artifacts list"
                                            }
                                            # Optionally upload provision-result.json as a private Gist
                                            if ($UploadGist.IsPresent) {
                                                try {
                                                    $token = if ($GithubPat) { $GithubPat } elseif ($env:GITHUB_PAT) { $env:GITHUB_PAT } elseif ($pat) { $pat } else { $null }
                                                    if (-not $token) { Write-Warning "No PAT available for Gist upload; set -GithubPat or GITHUB_PAT env var" } else {
                                                        $content = Get-Content (Join-Path $WorkDir 'provision-result.json') -Raw
                                                        $body = @{ description = "Provision result: $RunnerName"; public = $false; files = @{ 'provision-result.json' = @{ content = $content } } } | ConvertTo-Json -Depth 6
                                                        $resp = Invoke-RestMethod -Method Post -Uri "https://api.github.com/gists" -Headers @{ Authorization = "token $token"; 'User-Agent' = 'provision-runner-script' } -Body $body -ContentType 'application/json'
                                                        if ($resp.html_url) { $resp.html_url | Out-File -FilePath (Join-Path $WorkDir 'provision-gist-url.txt'); Write-Host "Provision result uploaded as Gist: $($resp.html_url)" }
                                                    }
                                                } catch { Write-Warning "Gist upload failed: $_" }
                                            }
                                        } else {
                                            $result = @{ status = 'failure'; timestamp = (Get-Date).ToString('o'); runner = $RunnerName; artifact = $smokeFile; message = "Smoke artifact validation failed: 'Smoke OK' not found" }
                                            $result | ConvertTo-Json | Out-File -FilePath (Join-Path $WorkDir 'provision-result.json') -Encoding UTF8
                                            Write-Error $result.message
                                            exit 1
                                        }
                                    } else {
                                        $result = @{ status = 'failure'; timestamp = (Get-Date).ToString('o'); runner = $RunnerName; artifact = $smokeFile; message = 'Smoke artifact validation failed: file not found' }
                                        $result | ConvertTo-Json | Out-File -FilePath (Join-Path $WorkDir 'provision-result.json') -Encoding UTF8
                                        if ($UploadGist.IsPresent) {
                                            try {
                                                $token = if ($GithubPat) { $GithubPat } elseif ($env:GITHUB_PAT) { $env:GITHUB_PAT } elseif ($pat) { $pat } else { $null }
                                                if ($token) {
                                                    $content = Get-Content (Join-Path $WorkDir 'provision-result.json') -Raw
                                                    $body = @{ description = "Provision result: $RunnerName"; public = $false; files = @{ 'provision-result.json' = @{ content = $content } } } | ConvertTo-Json -Depth 6
                                                    $resp = Invoke-RestMethod -Method Post -Uri "https://api.github.com/gists" -Headers @{ Authorization = "token $token"; 'User-Agent' = 'provision-runner-script' } -Body $body -ContentType 'application/json'
                                                    if ($resp.html_url) { $resp.html_url | Out-File -FilePath (Join-Path $WorkDir 'provision-gist-url.txt'); Write-Host "Provision result uploaded as Gist: $($resp.html_url)" }
                                                } else {
                                                    Write-Warning "No PAT available for Gist upload; set -GithubPat or GITHUB_PAT env var"
                                                }
                                            } catch { Write-Warning "Gist upload failed: $_" }
                                        }
                                        Write-Error $result.message
                                        exit 1
                                    }
                                } else {
                                    Write-Warning "runner-smoke artifact not found for run $runId"
                                }
                            } else {
                                Write-Warning "No completed smoke run found within timeout."
                            }
                        } catch {
                            Write-Warning "Smoke workflow polling/download failed: $_"
                        }
                    } catch {
                        Write-Warning "Failed to dispatch smoke workflow: $_"
                    }
                }

            } catch {
                Write-Warning "Runner verification failed: $_"
            }

            # Clear token variable
            $regToken = $null
        } catch {
            Write-Warning "Auto-registration failed: $_"
        }
    } else {
        Write-Warning "Could not parse owner/repo from RepoUrl: $RepoUrl"
    }
}

# Print next steps
Write-Host "Runner package downloaded to $WorkDir. Next steps (manual if not auto-registered):" -ForegroundColor Green
Write-Host "1) Create a registration token on GitHub (Repo Settings -> Actions -> Runners -> New self-hosted runner or use GH API)."
Write-Host "2) From $WorkDir, run the configure command shown below (replace <TOKEN>):"
Write-Host "   .\config.cmd --url $RepoUrl --token <TOKEN> --name $RunnerName --labels $Labels --work _work"
Write-Host "3) Install runner service as Administrator: .\svc.sh install ; Start: .\svc.sh start"
Write-Host "4) (Optional) Set UE5_ROOT as a System Environment variable if UE is installed in a custom location."

if ($UE5Root) {
    Write-Host "Setting UE5_ROOT = $UE5Root (machine-level env)"
    setx UE5_ROOT $UE5Root /M | Out-Null
}

Write-Host "Provisioning helper finished. If you used --AutoRegister, the runner should be configured and the service started (check GitHub Repo Actions -> Runners)." -ForegroundColor Green
