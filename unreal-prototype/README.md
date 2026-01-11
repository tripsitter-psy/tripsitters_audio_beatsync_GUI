TripSitter — Unreal Prototype (UE5.7.1)

Overview
--------
This folder contains a small Unreal Engine prototype skeleton and plugin scaffold to migrate the TripSitter GUI from wxWidgets to Unreal (recommended UE5.7.1).

Goals
- Provide a minimal, easy-to-open UE plugin that demonstrates invoking the existing processing pipeline (initially via launching FFmpeg / VideoWriter subprocesses).
- Document FFmpeg licensing and distribution choices (we will bundle LGPL-only FFmpeg shared DLLs on Windows by default).
- Provide a clear path to integrate the existing C++ backend as a library or plugin later.

What I added
- Plugin skeleton: `Source/TripSitterUE` with minimal module files and a small FFmpeg process wrapper stub.
- `TripSitterUE.uplugin` — plugin descriptor (drop into a UE project's Plugins folder).
- `README` with instructions and FFmpeg licensing guidance.
- `LICENSES/FFmpeg-LGPL-README.md` — short summary of LGPL vs GPL choices and distribution notes.

Prerequisites

Windows (prototype)
- Unreal Engine 5.7.1 (recommended). Install via Epic Games Launcher.
- Visual Studio 2022 (Desktop development with C++ workload).
- FFmpeg: For the prototype we will bundle an LGPL shared build (DLLs) in package stage. See license notes below.

macOS (prototype)
- Unreal Engine 5.7.1 (recommended). Install via Epic Games Launcher or the Epic Games App (ensure Editor + Mac support installed).
- Xcode + Command Line Tools (install via `xcode-select --install`).
- Homebrew (optional, for dependencies): https://brew.sh/
- Recommended packages: `brew install cmake ninja pkg-config ffmpeg`
  - Note: If you prefer vcpkg, bootstrap vcpkg on mac and pass `-DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake` to CMake.
- When building the shared backend on macOS the library will be copied to `unreal-prototype/ThirdParty/beatsync/lib/Mac/libbeatsync_backend.dylib` for the plugin to load.

Quick start
1. Create a new UE5.7.1 project (Blank C++ or Blueprint).
2. Copy the `unreal-prototype/` folder into your project root, or copy the `Source/TripSitterUE` folder into `<YourProject>/Plugins/TripSitterUE/Source/TripSitterUE` and add `TripSitterUE.uplugin` under `<YourProject>/Plugins/TripSitterUE/`.
3. In the Editor, open the project, build plugin (Editor -> Plugins -> locate TripSitterUE), then restart the editor if required.
4. The plugin includes a small FFmpeg wrapper stub (currently invokes an external ffmpeg process) — use for early E2E testing.

FFmpeg licensing notes
- Recommended for bundled distribution: **LGPL shared build** (dynamically linked DLLs) so that you may distribute a closed-source application. You must:
  - Ship the FFmpeg DLLs alongside your app (e.g. place them in the app bundle / same folder),
  - Include an LGPL license copy and a README describing the bundled FFmpeg build and how to replace it.
- If you require GPL-only codecs/flags, you must comply with the GPL (disclose source under GPL). See `LICENSES/FFmpeg-LGPL-README.md` for more.

Running Editor automation tests

You can run the editor automation smoke test (implemented as `TripSitter.Beatsync.EditorSmoke`) locally to validate the plugin UI and backend loader logic.

- From the Editor: open the Automation window (Window -> Developer Tools -> Session Frontend -> Automation), find the "TripSitter" category and run **Beatsync.EditorSmoke**.

- From the command line (headless, Windows example):

  ```powershell
  # Replace paths below with your UE installation and project paths
  "C:\Program Files\Epic Games\UE_5.7.1\Engine\Binaries\Win64\UE5Editor-Cmd.exe" "C:\Path\To\YourProject\YourProject.uproject" -ExecCmds="Automation RunTests TripSitter.Beatsync.EditorSmoke; Quit" -unattended -nopause -nullrhi -abslog="BeatsyncAutomation.log"
  ```

- Helper script: there's a small PowerShell helper `unreal-prototype/scripts/run-unreal-automation.ps1` to simplify invoking the Editor automation runner. Example usage:

  ```powershell
  .\unreal-prototype\scripts\run-unreal-automation.ps1 -Project "C:\Path\To\YourProject\YourProject.uproject"
  ```

CI note (important)

- Running Unreal Editor automation tests in CI requires a self-hosted Windows runner with Unreal Engine 5.7.1 installed (GitHub-hosted runners do not provide UE). See the CI template at `.github/workflows/unreal-editor-automation-template.yml` for a commented example job that you can copy into a self-hosted runner environment.

Windows DLL smoke test (GitHub Actions)

- We added a lightweight workflow (`.github/workflows/dll-smoke.yml`) that runs on `windows-latest` for quick validation of the Beatsync shared library.
  - The job clones & bootstraps `vcpkg`, downloads a small FFmpeg build, configures CMake with the vcpkg toolchain, builds the `dll_loader_smoke` target, and runs it.
  - `dll_loader_smoke` attempts to load the copied DLL from `unreal-prototype/ThirdParty/beatsync/lib/x64/` and verifies the exported C API (`bs_resolve_ffmpeg_path`, `bs_create_audio_analyzer`, `bs_destroy_audio_analyzer`).
  - This provides a fast, deterministic guard against regressions that break the plugin loader.

Local run (developer)

- To reproduce locally on Windows:
  - Ensure you have Visual Studio 2022 + CMake installed.
  - Install or download a portable FFmpeg build and set `FFMPEG_ROOT` to its root (the folder that contains `bin/` and `lib/`).
  - Configure & build the smoke target:

  ```powershell
  cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT="C:\path\to\ffmpeg"
  cmake --build build --config Release --target dll_loader_smoke
  $env:PATH = "C:\path\to\ffmpeg\bin;$env:PATH"
  ./build/bin/Release/dll_loader_smoke.exe
  ```

- The job is intended to be fast and small; add an editor automation job later when you have self-hosted runners with UE installed.

Next steps / TODOs
- Implement plugin bindings to call AudioAnalyzer/VideoWriter directly (as a library) rather than shelling out.
- Add UI prototype (UMG) with File Pickers, Start/Cancel, and Progress bar.
- Add automated smoke test that runs a short FFmpeg concat via the plugin.

If you want, I can now:
- Add the UMG skeleton and a minimal widget that triggers a test FFmpeg command, or
- Open a branch and push this scaffold as the first commit to review.

— GitHub Copilot
