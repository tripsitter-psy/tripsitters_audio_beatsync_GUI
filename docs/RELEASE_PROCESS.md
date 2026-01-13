# MTV TripSitter Release Process

This document describes how to create and publish releases of MTV TripSitter.

## Overview

The release pipeline automates:
1. Building the application
2. Code signing (when certificates are configured)
3. Creating NSIS installer
4. Creating portable ZIP
5. Running smoke tests
6. Publishing to GitHub Releases

## Quick Start

### Creating a Release

1. **Tag the release**:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

2. The `windows-release.yml` workflow automatically:
   - Builds the application
   - Signs binaries (if configured)
   - Creates installer and ZIP
   - Runs smoke tests
   - Creates a draft GitHub Release

3. **Review and publish** the draft release on GitHub

### Manual Release (Workflow Dispatch)

1. Go to Actions > "Windows Release Pipeline"
2. Click "Run workflow"
3. Optionally set version string and enable release creation
4. Click "Run workflow"

## Code Signing Setup

Code signing is optional but recommended for production releases.

### Obtaining a Code Signing Certificate

Options:
- **Commercial**: DigiCert, Sectigo, GlobalSign ($200-500/year)
- **Open Source**: SignPath.io (free for open source)
- **Self-signed**: For testing only (users will see warnings)

### Configuring CI for Code Signing

1. Export your certificate as a PFX file with a password

2. Base64-encode the PFX:
   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("certificate.pfx")) | Set-Clipboard
   ```

3. Add GitHub repository secrets:
   - `CODESIGN_CERTIFICATE_BASE64`: The base64-encoded PFX
   - `CODESIGN_CERTIFICATE_PASSWORD`: The PFX password

4. The workflow automatically uses these when available

### Local Signing (Development)


```powershell
# Sign a single file (do NOT pass -CertificatePassword in plaintext)
# Use secure alternatives: prompt for password interactively, read from a protected secret store or key vault, or pass via a secure environment variable.
# Example (prompt for password):
$certPassword = Read-Host -AsSecureString "Enter certificate password"
.\scripts\sign-windows-binaries.ps1 -BuildDir build\bin\Release `
   -CertificatePath "path\to\cert.pfx" `
   -CertificatePassword $certPassword

# Use -DryRun to verify what would be signed without performing actual signing:
.\scripts\sign-windows-binaries.ps1 -BuildDir build\bin\Release -DryRun
```

> **Security Note:**
> Never expose sensitive values like -CertificatePassword on the command line or in scripts. Always use secure input methods or secret management tools. The sign-windows-binaries.ps1 script supports -BuildDir, -CertificatePassword, and -DryRun flags; substitute secure handling as appropriate for your environment.

## Installer Configuration

### NSIS Settings

The installer is configured in `CMakeLists.txt`:

```cmake
# Key settings
set(CPACK_NSIS_DISPLAY_NAME "MTV Trip Sitter")
set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/assets/icon.ico")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
```

### Customization

To customize the installer appearance:
1. Create `assets/installer-header.bmp` (150x57 pixels)
2. Create `assets/installer-welcome.bmp` (164x314 pixels)
3. Uncomment the header/welcome image lines in `installer/nsis_template.nsi.in`

## Smoke Tests

The release workflow includes automated smoke tests:

1. **Installation test**: Silent install to Program Files
2. **Verification**: Check installed files exist
3. **Launch test**: Brief application launch
4. **Uninstallation**: Clean uninstall

### Running Smoke Tests Locally

```powershell
# Build installer
cmake --build build --config Release
cd build && cpack -C Release -G NSIS

# Install silently
.\MTVTripSitter-0.1.0-Windows-AMD64.exe /S

# Verify
Test-Path "$env:ProgramFiles\MTV TripSitter\bin\TripSitter.exe"

# Uninstall
& "$env:ProgramFiles\MTV TripSitter\Uninstall.exe" /S
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `1.0.0`)
- Pre-release: `1.0.0-alpha`, `1.0.0-beta.1`

Update version in `CMakeLists.txt`:
```cmake
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
```

## Troubleshooting

### Installer fails to build

Check NSIS is installed:
```powershell
makensis -VERSION
```

Install with Chocolatey:
```powershell
choco install -y nsis
```

### Code signing fails

1. Verify certificate is valid and not expired
2. Check timestamp server is reachable
3. Ensure signtool.exe is in PATH (install Windows SDK)

### Smoke test fails

Common causes:
- Missing Visual C++ Redistributable
- Missing FFmpeg DLLs
- Antivirus blocking execution

Check the workflow logs for specific error messages.

## Release Checklist

Before tagging a release:

- [ ] Update version in `CMakeLists.txt`
- [ ] Update `CHANGELOG.md`
- [ ] Run tests locally: `cmake --build build && ctest --test-dir build -C Release`
- [ ] Build and test installer locally
- [ ] Commit all changes
- [ ] Create and push tag
- [ ] Review draft release on GitHub
- [ ] Publish release
