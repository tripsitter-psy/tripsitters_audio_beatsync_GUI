Runner provisioning — scripts and notes

This document describes the helper scripts that assist in provisioning self-hosted GitHub Actions runners used for Unreal Editor automation.

Files added
- `scripts/provision-runner-windows.ps1` — PowerShell helper to install common packages (Chocolatey, Git, CMake, FFmpeg, Visual Studio Build Tools) and download the GitHub Actions runner. It prints the registration command you must run with a token.
- `scripts/provision-runner-macos.sh` — Bash helper to install Homebrew packages and download the GitHub Actions runner for macOS. It prints the registration command you must run with a token.

Security & tokens
- Runner registration requires a one-time token from GitHub (Repo Settings -> Actions -> Runners -> New self-hosted runner), or use the GitHub REST API to generate tokens.
- Tokens are short-lived; the scripts *do not* attempt to register the runner automatically without user confirmation.
- Do not share your token or embed it in public workflows.

Usage examples
- Windows (run as Administrator):
  PowerShell:
  ```powershell
  Set-ExecutionPolicy Bypass -Scope Process -Force
  .\scripts\provision-runner-windows.ps1 -RepoUrl "https://github.com/<owner>/<repo>" -RunnerName "ue5-runner" -Labels "self-hosted,windows,ue5-5.7" -UE5Root "C:\Program Files\Epic Games\UE_5.7.1"
  ```

- macOS:
  ```bash
  ./scripts/provision-runner-macos.sh "https://github.com/<owner>/<repo>" ue5-mac-runner "self-hosted,macos,ue5-5.3" /opt/actions-runner
  ```

After running the helper

Interactive/manual registration (original flow)
1) Create the registration token in GitHub, then run the configure command printed by the helper (it includes `--url` and `--name` and `--labels`).
2) Install and start the service (`svc.sh install` / `svc.sh start`).
3) Confirm in the repository Settings -> Actions -> Runners that your runner is online and labelled correctly.

Unattended registration (optional, automated)
- Both helper scripts now support unattended registration using a GitHub Personal Access Token (PAT) with appropriate scope. The script will request a short-lived runner registration token and configure the runner automatically. After auto-registering, the script polls GitHub for up to ~1 minute to confirm the runner appears online and will warn if it does not.

Windows example (PowerShell, requires admin):
```powershell
# Uses env var GITHUB_PAT or pass -GithubPat
Set-ExecutionPolicy Bypass -Scope Process -Force
.\scripts\provision-runner-windows.ps1 -RepoUrl "https://github.com/<owner>/<repo>" -GithubPat "<YOUR_PAT>" -AutoRegister
```

macOS example (Bash):
```bash
# Pass PAT as 5th arg or set GITHUB_PAT env var
./scripts/provision-runner-macos.sh "https://github.com/<owner>/<repo>" ue5-mac-runner "self-hosted,macos,ue5-5.7" /opt/actions-runner "<YOUR_PAT>"
```

Verification
- After unattended registration the script will poll the GitHub REST API to verify that the runner is online. If the runner does not appear online within the timeout window the script will print a warning and you should check the runner service logs and the GitHub UI.
- If you request the optional smoke workflow (`-RunSmoke` on Windows or `RUN_SMOKE=1` on macOS), the scripts will dispatch the `runner-smoke.yml` workflow, poll for completion, download the `runner-smoke` artifact, and validate that `smoke.txt` contains the string `Smoke OK`.
  - If the validation fails (missing artifact or content mismatch), the provisioning script will exit with a non-zero status so you can detect a failed provision step automatically.
- On success or failure the scripts write a small JSON summary file to the runner workdir: `provision-result.json` with keys `status` ("success"|"failure"), `timestamp` (ISO 8601), `runner`, `artifact`, and `message`. This is helpful for programmatic validation or integration with provisioning systems.
- Runtime artifact: when the GUI runtime check (`TripSitter --check-wallpaper`) is run under CI it will also write `wallpaper_check.txt` in the test working dir containing `WALLPAPER_FOUND` or `WALLPAPER_MISSING` for easier debugging (this is produced when `GITHUB_ACTIONS` or `CI` is present, or when `WRITE_WALLPAPER_ARTIFACT` is set).
- Provisioning artifact index: after running the smoke workflow the provisioning scripts will collect discovered artifacts (for example `wallpaper_check.txt`, `provision-gist-url.txt`) and record their paths in `provision-artifacts.txt` in the runner workdir. Check this file (and `provision-result.json`) for a quick inventory of what the provisioning run produced.
- Optional: upload the result as a private GitHub Gist for remote auditing:
  - Windows: pass `-UploadGist` to `provision-runner-windows.ps1` (requires a PAT via `-GithubPat` or `GITHUB_PAT` env var).
  - macOS: pass `RUN_UPLOAD_GIST=1` as the 7th arg to `provision-runner-macos.sh` (requires `GITHUB_PAT`).
  - On success the script writes the resulting Gist URL to `provision-gist-url.txt` in the runner workdir.
  - The Gist is created as private (`public: false`).
  - Be careful with PAT scope and handling; prefer environment variables over inline tokens.

Security & PAT scope
- The PAT used for unattended registration must have repository administration scope (repo level admin) or organization admin permissions to create runner registration tokens. Avoid using a PAT with broader scopes than necessary.
- Treat the PAT as a secret: prefer passing it as an environment variable (`GITHUB_PAT`) rather than embedding it in commands or files.
- The unattended flow does not store your PAT; it only uses it to request a short-lived registration token from GitHub and then discards it.

Security & PAT scope
- The PAT used for unattended registration must have repository administration scope (repo level admin) or organization admin permissions to create runner registration tokens. Avoid using a PAT with broader scopes than necessary.
- Treat the PAT as a secret: prefer passing it as an environment variable (`GITHUB_PAT`) rather than embedding it in scripts or commit history.
- The unattended flow does not store your PAT; it only uses it to request a short-lived registration token from GitHub and then discards it.

Notes & caveats
- Visual Studio installation (Windows) is large and may require interactive components; the script attempts an automated Chocolatey install of Build Tools but you should verify the VC++ workload is present.
- UE installation is manual via Epic Games Launcher; ensure UE is installed and `UE5_ROOT` set for the runner environment.
- These scripts are opinionated convenience helpers. Review them before running on any production host.

If you'd like, I can add an optional step to generate the PAT via the GitHub REST API using a machine token (requires a secure service token) but that adds complexity and security considerations — I can add it on request.