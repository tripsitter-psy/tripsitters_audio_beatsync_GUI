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
    [Parameter(Mandatory=$false)]
    [string]$BuildDir,

    [Parameter(Mandatory=$false)]
    [string]$InstallerPath,

    [Parameter(Mandatory=$false)]
    [string]$CertificatePath,

    [Parameter(Mandatory=$false)]
    [System.Security.SecureString]$CertificatePassword,

    [Parameter(Mandatory=$false)]
    [string]$TimestampUrl = "http://timestamp.digicert.com",

    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

# Validate that at least one of BuildDir or InstallerPath is provided
if (-not $BuildDir -and -not $InstallerPath) {
    Write-Error "Either -BuildDir or -InstallerPath must be specified."
    exit 1
}

$ErrorActionPreference = "Stop"

Write-Host "=== MTV TripSitter Windows Code Signing ===" -ForegroundColor Cyan

# Initialize cleanup tracking at script scope
# These variables track the temp cert file so it can be cleaned up on any exit
$script:createdTempCert = $false
$script:tempCertPath = $null

# Register cleanup function that runs on script termination (including early exits)
# This ensures the temp cert file is always removed, even if the script exits before
# reaching the main try/finally block
function Remove-TempCertIfExists {
    if ($script:createdTempCert -and $script:tempCertPath -and (Test-Path $script:tempCertPath)) {
        Remove-Item $script:tempCertPath -Force -ErrorAction SilentlyContinue
        Write-Host "Cleaned up temp certificate file" -ForegroundColor Gray
    }
}

# Trap to handle terminating errors and ensure cleanup
trap {
    Remove-TempCertIfExists
    break
}

# Find signtool.exe
function Find-SignTool {
    $programFiles = @(
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)}
    )

    foreach ($pf in $programFiles) {
        $sdkPath = Join-Path $pf "Windows Kits\10\bin"
        if (Test-Path $sdkPath) {
            $versions = Get-ChildItem $sdkPath -Directory | Where-Object { [version]::TryParse($_.Name, [ref]([version]::new())) } | Sort-Object { [version]$_.Name } -Descending
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
        $guid = [guid]::NewGuid().ToString()
        $tempCert = Join-Path $env:TEMP ("codesign_cert_" + $guid + ".pfx")

        # Store path in script scope BEFORE writing, so cleanup can find it even if write fails partway
        $script:tempCertPath = $tempCert


        # Create file with restricted ACL before writing bytes
        $fs = $null
        try {
            $acl = New-Object System.Security.AccessControl.FileSecurity
            # Get current Windows identity, fallback to USERNAME env var if it fails
            try {
                $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
            } catch {
                $domain = $env:USERDOMAIN
                if (-not $domain) { $domain = $env:COMPUTERNAME }
                $currentUser = "$domain\$($env:USERNAME)"
            }
            $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($currentUser, "FullControl", "Allow")
            $acl.SetAccessRule($rule)
            $acl.SetAccessRuleProtection($true, $false)
            $fs = [System.IO.File]::Open($tempCert, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            $fs.SetAccessControl($acl)
            $script:createdTempCert = $true
            $bytes = [System.Convert]::FromBase64String($env:CODESIGN_CERTIFICATE_BASE64)
            $fs.Write($bytes, 0, $bytes.Length)
        } finally {
            if ($fs) { $fs.Close() }
        }

        # Mark that we created this temp cert so cleanup knows to delete it
        $script:createdTempCert = $true

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

# Get certificate and password, resolve and validate before use
$cert = $CertificatePath
$securePassword = $CertificatePassword
if (-not $cert) {
    $cert = Get-CertificateFromEnv
}
if (-not $securePassword) {
    $envPwd = $env:CODESIGN_CERTIFICATE_PASSWORD
    if ($envPwd) {
        $securePassword = ConvertTo-SecureString $envPwd -AsPlainText -Force
    }
}

# Validate certificate and password presence
if (-not $cert) {
    Write-Warning "No code signing certificate provided. Skipping signing."
    Write-Host "Set CODESIGN_CERTIFICATE_BASE64 and CODESIGN_CERTIFICATE_PASSWORD in CI secrets."
    exit 0
}
if (-not (Test-Path $cert)) {
    Write-Error "Certificate file not found: $cert"
    exit 1
}
if (-not $securePassword) {
    Write-Error "Certificate password is missing. Set CODESIGN_CERTIFICATE_PASSWORD in CI secrets."
    exit 1
}

# Ensure password is a SecureString
if ($securePassword -isnot [System.Security.SecureString]) {
    $securePassword = ConvertTo-SecureString $securePassword -AsPlainText -Force
}

$thumbprint = $null
$global:ExitCode = 1  # Default to failure
try {
    $certObject = Import-PfxCertificate -FilePath $cert -CertStoreLocation Cert:\CurrentUser\My -Password $securePassword
    $thumbprint = $certObject.Thumbprint


    # Find files to sign
    $filesToSign = @()

    if ($InstallerPath) {
        # Single file mode - sign only the specified installer
        if (-not (Test-Path $InstallerPath)) {
            Write-Error "Installer file not found: $InstallerPath"
            $global:ExitCode = 1
            return
        }
        $filesToSign += Get-Item $InstallerPath
    } else {
        # Directory mode - find all exe/dll files
        if (-not (Test-Path -Path $BuildDir -PathType Container)) {
            Write-Error "Build directory invalid or missing: $BuildDir"
            $global:ExitCode = 1
            return
        }
        $extensions = @("*.exe", "*.dll")
        foreach ($ext in $extensions) {
            $files = Get-ChildItem -Path $BuildDir -Recurse -Filter $ext -File
            $filesToSign += $files
        }
    }

    if ($filesToSign.Count -eq 0) {
        $location = if ($InstallerPath) { $InstallerPath } else { $BuildDir }
        Write-Host "No files found to sign in $location"
        $global:ExitCode = 0
        return
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


        $signArgs = @(
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
            $result = & $signtool @signArgs 2>&1
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
        $global:ExitCode = 1
        return
    }
    $global:ExitCode = 0
} finally {
    # Clean up certificate from store and temp file
    if ($thumbprint -and (Test-Path "Cert:\CurrentUser\My\$thumbprint")) {
        Remove-Item "Cert:\CurrentUser\My\$thumbprint" -ErrorAction SilentlyContinue
    }
    # Use the script-scoped cleanup function to remove temp cert file
    Remove-TempCertIfExists
}

# Exit with the appropriate code after cleanup completes
if ($global:ExitCode -ne $null) {
    exit $global:ExitCode
}
