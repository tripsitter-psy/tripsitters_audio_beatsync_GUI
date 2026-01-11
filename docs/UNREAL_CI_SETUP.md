Unreal Editor Automation — Self-hosted runner setup

Overview
--------
This document describes configuring a self-hosted runner to execute the `Unreal Editor Automation` workflow added to this repository. The workflow is designed to run the Editor automation test `TripSitter.Beatsync.EditorSmoke` on a machine that has Unreal Engine 5.7.1 installed.

Important security note
-----------------------
Self-hosted runners run arbitrary code from workflows. Only attach runners you fully control and *do not* use them to run untrusted pull requests unless your policy explicitly allows it. Use dedicated labels and restrict runner access to trusted teams.

Runner labels recommended
- Windows: self-hosted, windows, ue5-5.7, x64
- macOS:   self-hosted, macos, ue5-5.7, x64

Minimum software requirements (per platform)
- Windows:
  - Windows Server 2019 / Windows 10+ with latest updates
  - Visual Studio 2022 (C++ workload, Desktop dev)
  - Unreal Engine 5.7.1 installed and working (Editor + command-line tools)
  - PowerShell (core) available (the repo includes a PowerShell helper)
  - GitHub Actions Runner (self-hosted runner service) installed and configured
- macOS:
  - macOS 12+ supported by UE5.7.1
  - Xcode + command-line tools
  - Unreal Engine 5.3 installed (Editor + command line)
  - Optional: Homebrew (for CMake / FFmpeg if building locally)
  - GitHub Actions Runner (self-hosted runner service) installed and configured

Setting up the self-hosted runner
---------------------------------
1. On a machine with the required software, create a folder for the runner and download the GitHub Actions runner package for your OS:
   - https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners
2. Register the runner with your repository or organization and assign the labels you will use, e.g.: `self-hosted, windows, ue5-5.7`.
3. Configure the runner to run as a service (Windows) or a systemd/launchd service (macOS) so it starts on boot.

Install Unreal Engine and prerequisites
---------------------------------------
- Windows: Install UE through Epic Games Launcher and install the Editor. Ensure the Editor's command-line binary is present at `C:\Program Files\Epic Games\UE_5.7.1\Engine\Binaries\Win64\UE5Editor-Cmd.exe` or set `UE5_ROOT` on the runner to your install root.
- macOS: Install UE (App/Editor). The command-line binary is typically at `/Applications/Epic Games/UE_5.7.1/Engine/Binaries/Mac/UE5Editor-Cmd.app/Contents/MacOS/UE5Editor-Cmd`.

Environment variables
---------------------
Set `UE5_ROOT` environment variable on the runner to the Engine root (`C:\Program Files\Epic Games\UE_5.7.1` or `/Applications/Epic Games/UE_5.3`). The workflow will read this variable if the `ue_root` input is not provided.

Using the workflow
------------------
- In GitHub, go to Actions -> Workflows -> "Unreal Editor Automation (self-hosted ready)" and choose "Run workflow".
- Supply `uproject_path` (relative path to the `.uproject` file within the checked-out repo) and optionally `ue_root`.

Example (Windows)
- `uproject_path`: `unreal-prototype/YourProject.uproject`
- Runner labels: `self-hosted, windows, ue5-5.7`

Artifacts and logs
------------------
- Automation logs are uploaded as workflow artifacts (BeatsyncAutomation.log). Collect these and the UE logs for debugging.
- The Editor will write an `.abslog` as specified by the command; the `BeatsyncAutomation.log` should contain test results.

CI best practices
-----------------
- Use dedicated, patched, and monitored standalone runners for UE automation.
- Use single-purpose labels and restrict repository/organization access where feasible.
- Keep the runner up-to-date with UE and Visual Studio/Xcode patches and store the runner as an image snapshot (Windows) so it can be re-provisioned quickly.
- If you need to run tests for many branches/PRs, consider ephemeral runners (or VMs) that are started per job and then torn down.

Next steps / Troubleshooting
---------------------------
- If the run fails with "library not found" in the plugin loader: ensure the shared backend (`libbeatsync_backend.dylib` or `beatsync_backend.dll`) is present under `unreal-prototype/ThirdParty/beatsync/lib/<Platform>`.
- If FFmpeg isn't found: ensure the host has FFmpeg installed or set `FFMPEG_ROOT` when building or package FFmpeg dylibs in the app bundle.

Contact
-------
If you'd like, I can prepare a maintenance checklist and a sample runner provisioning script (Windows PowerShell/Chocolatey or macOS Bash/Homebrew) — say the word and I'll add it.