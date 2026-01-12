# Windows Code Signing Script for MTV TripSitter
# This script signs Windows executables and DLLs using signtool.exe
#
# Prerequisites:
# - Windows SDK installed (provides signtool.exe)
# - Code signing certificate (PFX file or Windows cert store)
#
# Environment Variables (set in CI):
# - CODESIGN_CERTIFICATE_BASE64: Base64-encoded PFX file
# - CODESIGN_CERTIFICATE_PASSWORD: Password for the PFX file
# - CODESIGN_TIMESTAMP_URL: Timestamp server URL (optional, defaults to DigiCert)

param(
    [Parameter(Mandatory=$true)]
    [string]$BuildDir,

    [Parameter(Mandatory=$false)]
    [string]$CertificatePath,

    [Parameter(Mandatory=$false)]
    [string]$CertificatePassword,

    [Parameter(Mandatory=$false)]
    [string]$TimestampUrl = "http://timestamp.digicert.com",

    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=== MTV TripSitter Windows Code Signing ===" -ForegroundColor Cyan

# Find signtool.exe
function Find-SignTool {
    $programFiles = @(
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)}
    )

    foreach ($pf in $programFiles) {
        $sdkPath = Join-Path $pf "Windows Kits\10\bin"
        if (Test-Path $sdkPath) {
            $versions = Get-ChildItem $sdkPath -Directory | Sort-Object Name -Descending
            foreach ($version in $versions) {
                $signtool = Join-Path $version.FullName "x64\signtool.exe"
                if (Test-Path $signtool) {
                    return $signtool
                }
            }
        }
    }

    # Try PATH
    $inPath = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($inPath) {
        return $inPath.Source
    }

    return $null
}

# Decode base64 certificate from environment
function Get-CertificateFromEnv {
    if ($env:CODESIGN_CERTIFICATE_BASE64) {
        $tempCert = Join-Path $env:TEMP "codesign_cert.pfx"
        [System.IO.File]::WriteAllBytes($tempCert, [System.Convert]::FromBase64String($env:CODESIGN_CERTIFICATE_BASE64))
        
        # Set restrictive permissions: current user only
        $acl = Get-Acl $tempCert
        $acl.SetAccessRuleProtection($true, $false)  # Remove inherited permissions
        
        # Get current Windows identity, fallback to USERNAME env var if it fails
        try {
            $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        } catch {
            $currentUser = $env:USERNAME
        }
        
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($currentUser, "FullControl", "Allow")
        $acl.SetAccessRule($rule)
        Set-Acl $tempCert $acl
        
        return $tempCert
    }
    return $null
}

# Main signing logic
$signtool = Find-SignTool
if (-not $signtool) {
    Write-Warning "signtool.exe not found. Skipping code signing."
    Write-Host "Install Windows SDK to enable code signing."
    exit 0
}

Write-Host "Found signtool: $signtool"

# Get certificate
$cert = $CertificatePath
$password = $CertificatePassword

if (-not $cert) {
    $cert = Get-CertificateFromEnv
    $password = $env:CODESIGN_CERTIFICATE_PASSWORD
}

if (-not $cert) {
    Write-Warning "No code signing certificate provided. Skipping signing."
    Write-Host "Set CODESIGN_CERTIFICATE_BASE64 and CODESIGN_CERTIFICATE_PASSWORD in CI secrets."
    exit 0
}

if (-not (Test-Path $cert)) {
    Write-Error "Certificate file not found: $cert"
    exit 1
}

if (-not $password -or $password -eq '') {
    Write-Error "Certificate password is missing. Set CODESIGN_CERTIFICATE_PASSWORD in CI secrets."
    exit 1
}

# Import certificate to store securely
$securePassword = ConvertTo-SecureString $password -AsPlainText -Force

try {
    $certObject = Import-PfxCertificate -FilePath $cert -CertStoreLocation Cert:\CurrentUser\My -Password $securePassword
    $thumbprint = $certObject.Thumbprint

    Write-Host "Imported certificate with thumbprint: $thumbprint"

    # Find files to sign
    $filesToSign = @()
    $extensions = @("*.exe", "*.dll")

    foreach ($ext in $extensions) {
        $files = Get-ChildItem -Path $BuildDir -Recurse -Filter $ext -File
        $filesToSign += $files
    }

    if ($filesToSign.Count -eq 0) {
        Write-Host "No files found to sign in $BuildDir"
        exit 0
    }

    Write-Host "Found $($filesToSign.Count) files to sign:"
    $filesToSign | ForEach-Object { Write-Host "  - $($_.Name)" }

    # Sign each file
    $signedCount = 0
    $failedCount = 0

    foreach ($file in $filesToSign) {
        Write-Host "`nSigning: $($file.FullName)" -ForegroundColor Yellow

        if ($DryRun) {
            Write-Host "  [DRY RUN] Would sign this file"
            $signedCount++
            continue
        }

        $args = @(
            "sign",
            "/sha1", $thumbprint,
            "/fd", "SHA256",
            "/tr", $TimestampUrl,
            "/td", "SHA256",
            "/d", "MTV TripSitter",
            "/du", "https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI",
            $file.FullName
        )

        try {
            $result = & $signtool @args 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  Signed successfully" -ForegroundColor Green
                $signedCount++
            } else {
                Write-Warning "  Failed to sign: $result"
                $failedCount++
            }
        } catch {
            Write-Warning "  Error signing: $_"
            $failedCount++
        }
    }

    Write-Host "`n=== Signing Complete ===" -ForegroundColor Cyan
    Write-Host "Signed: $signedCount"
    Write-Host "Failed: $failedCount"

    if ($failedCount -gt 0) {
        exit 1
    }
} finally {
    # Clean up certificate from store and temp file
    if ($thumbprint -and (Test-Path "Cert:\CurrentUser\My\$thumbprint")) {
        Remove-Item "Cert:\CurrentUser\My\$thumbprint" -ErrorAction SilentlyContinue
    }
    if ($cert -and (Test-Path $cert) -and ($cert -eq (Join-Path $env:TEMP "codesign_cert.pfx"))) {
        Remove-Item $cert -Force -ErrorAction SilentlyContinue
    }
}
