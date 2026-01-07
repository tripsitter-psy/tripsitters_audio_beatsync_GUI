# PR #5 — chore: centralize process runner, add FFmpeg logging + GUI fixes

Short summary
-------------
Centralize subprocess execution into `src/utils/ProcessUtils.{h,cpp}`, capture FFmpeg stdout/stderr into persistent logs, improve FFmpeg input probing and concat fallbacks, fix tests and Windows test runtime DLL availability, and make GUI visual improvements to reduce white input boxes and overlay artifacts.

Checklist
---------
- [x] Centralize process runner into `src/utils/ProcessUtils.{h,cpp}` and remove duplicate implementations
- [x] Capture FFmpeg output and errors into per-operation logs (`beatsync_ffmpeg_extract.log`, `beatsync_ffmpeg_concat.log`, `beatsync_ffmpeg_preview.log`)
- [x] Improve FFmpeg probing for audio presence/duration and add re-encode fallback for concat edge-cases
- [x] Fix tests (add missing sources) and copy FFmpeg DLLs into test runtime directory so tests run on Windows
- [x] GUI visual fixes: blended control backgrounds, plain-controls toggle, waveform/effect-region UX and BeatVisualizer improvements
- [x] Polish BeatNet/Demucs Python wrapper scripts for clearer JSON output and error handling
- [ ] Add CI job to run build & tests and upload FFmpeg logs on failure (follow-up)

Testing instructions
--------------------
1. Build and run unit tests locally (Windows example):

   ```powershell
   cmake -S . -B build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release --parallel
   ctest --test-dir build -C Release -V
   ```

   Expected: configure/build succeed and tests pass. If tests fail with a 0xc0000135 error, ensure FFmpeg DLLs were copied into the test output dir (CMake already attempts this on Windows).

2. Trigger an FFmpeg operation (e.g., extract a segment, concatenate segments or preview a transition) and inspect logs:

   - `beatsync_ffmpeg_extract.log` — segment extraction runs
   - `beatsync_ffmpeg_concat.log` — concatenation & transition runs
   - `beatsync_ffmpeg_preview.log` — preview runs

   Logs contain: timestamp, label, command, exit code, and truncated output tail for easier triage.

3. Manual GUI checks (TripSitter application):

   - Verify file pickers / small text inputs blend with the artwork (no plain white boxes) and that the "Plain Controls" toggle enforces a consistent dark UI.
   - Open the BeatVisualizer and confirm effect region selection, effect handles, zoom/pan, and effect-beat overlay behave as expected.
   - Run a preview transition and check `beatsync_ffmpeg_preview.log` for any failures and command details.

4. AI helpers (optional):

   - `python scripts/beatnet_analyze.py <path/to/audio>` — should return JSON with `beats` and `bpm` or an `error` field
   - `python scripts/demucs_separate.py <path/to/audio> <outdir>` — should return JSON with `stems` or an `error` field

Notes & follow-ups
------------------
- I squashed the branch into a single commit for a cleaner history; this commit includes the consolidated changes and conflict resolutions.
- If you'd like, I can add a GitHub Actions workflow that runs the build/tests and uploads the FFmpeg logs on failure (recommended next step).