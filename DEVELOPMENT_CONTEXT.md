# BeatSyncEditor — Development Context 📄

## Project Overview 💡
**BeatSyncEditor** is a C++ command-line tool that synchronizes video clips to audio beats using FFmpeg. It analyzes audio files to detect beats, then cuts and arranges video clips to match those beats.

**Location:** `C:\Users\samue\Desktop\BeatSyncEditor`

---

## Architecture 🔧

src/
- `main.cpp` — CLI entry point, command handlers  
- `audio/`
  - `AudioAnalyzer.cpp` — Beat detection & audio metadata (duration)
  - `BeatGrid.cpp` / `BeatGrid.h` — Beat timing data structure (now stores audio duration)
- `video/`
  - `VideoProcessor.cpp` — FFmpeg-based video reading & info
  - `VideoWriter.cpp` — Segment extraction, normalization, concatenation, audio muxing

---

## Commands (CLI) ▶️

- `analyze <audio>` — Detect beats in audio file  
- `sync <video> <audio>` — Sync a single video to beats  
- `multiclip <folder> <audio>` — Create beat-synced video from multiple clips (cycles through clips)  
- `split <video> <audio>` — Split video at beat timestamps

---

## Build System 🛠️
- CMake-based; builds with MSVC on Windows  
- FFmpeg dev libs expected at `C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared` (or set `BEATSYNC_FFMPEG_PATH`)  
- Build command:
```bash
cmake --build build --config Release
```
- Output: `build\bin\Release\beatsync.exe`

---

## Session Log (2026-01-02) 🧾

### Issue 1: Video Freezing and DTS Desync ❌
- **Problem:** Video intermittently froze while audio continued.  
- **Root Cause:** Source clips had mixed resolutions (e.g., `1920x1080` vs `3808x2176`) → FFmpeg reconfigured filter graph frequently → duplicated/dropped frames and timestamp errors.  
- **Fix (in `VideoWriter.cpp`):** Normalize all segments on extraction with a consistent filter and encoding:
```text
-vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24"
-c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p
-c:a aac -b:a 192k -ar 44100
-video_track_timescale 90000
```
- **Status:** ✅ FIXED — No more freezing

---

### Issue 2: Output Duration Shorter Than Audio ⚠️
- **Problem:** Output video shorter than audio (e.g., output 5:45 vs audio 6:01).  
- **Root Cause:** Beat detection ended before song end (fade-outs lack strong beats); last segment used fixed 2s duration and `-shortest` behavior sometimes trimmed audio.  
- **Fixes Applied:**
  - `BeatGrid.h/.cpp` — Added audio duration tracking:
    - `void setAudioDuration(double duration);`
    - `double getAudioDuration() const;`
    - `double m_audioDuration;`
  - `AudioAnalyzer.cpp` — Populate beat grid with `audio.duration`.
  - `main.cpp` (multiclip) — Extend last beat segment to audio end; if still short, add padding segments (cycle clips). Changed behavior so audio is not cut (do not use `-shortest`).
- **Status:** ✅ FIXED — Build successful, awaiting test

---

## Current Status ✅
- Video freezing — **Fixed and tested**
- Duration padding — **Fixed, build successful, awaiting test**
- GUI (TripSitter) — **Implemented and builds successfully** (`build/bin/Release/TripSitter.exe`). Added a wxWidgets-based GUI with a **PREVIEW FRAME** button and a timestamp input; implemented `VideoPreview::LoadFrame` (uses `VideoProcessor` + `libswscale`) and added `BeatVisualizer` for visualizing beats.
- Packaging & CI — **CPack configured (ZIP + NSIS)** and `assets/` placeholders + `scripts/import_assets.ps1` added. A GitHub Actions workflow (`.github/workflows/windows-build.yml`) was added to build and upload artifacts.
- Build issues resolved — Fixed syntax/namespace errors, missing includes, and VideoWriter API mismatches; disabled an invalid `assets/icon.ico` resource to avoid RC failures (re-enable when a valid `.ico` is supplied).

**Recent Packaging Run (2026-01-02):**
- Ran `cpack -C Release` locally. Result: `build/TripSitter--Windows-AMD64.zip` generated and moved to `build/artifacts/TripSitter--Windows-AMD64.zip` (83,296 KB).
- NSIS packaging failed locally: CPack reported `makensis` not found. Attempts to install NSIS locally failed due to the downloaded NSIS installer not running (error: "file or directory is corrupted and unreadable").
- To address this, the Windows CI workflow (`.github/workflows/windows-build.yml`) was updated to install NSIS via Chocolatey and to run `cpack` on the runner so the NSIS installer can be built in CI.
- I created and pushed a branch `ci/nsis-smoke-test` with these workflow changes to the repo: https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI/tree/ci/nsis-smoke-test. Attempting to create a PR from this environment failed because the GitHub CLI (`gh`) is not available here; you can open the PR in the GitHub UI or I can open it after you confirm.
- Local binary behavior: `TripSitter.exe --version`/`--help` exited with code 128 and produced no stdout/stderr, but launching `TripSitter.exe` without arguments starts the GUI and it remained running for several seconds during a smoke launch (local GUI launch appears to work).
- The CI smoke test was updated to launch `TripSitter.exe` for 6 seconds and fail if it exits immediately; this will help validate the GUI in a fresh runner environment.

**Next steps (updated):**
1. Merge the `ci/nsis-smoke-test` branch (or open a PR) and trigger the GitHub Actions workflow to build packages and produce the NSIS installer in CI (recommended).
2. If you want NSIS locally, retry installing NSIS after reboot (verify installer integrity or install via Chocolatey if available), then re-run `cpack -C Release` to produce the `.exe` locally.
3. Verify GUI preview at runtime (use PREVIEW FRAME with timestamp) — still not fully verified end-to-end; I can run this after CI artifacts are available or after a local GUI run.
4. Import user GUI assets (`C:\Users\samue\Downloads\assets for GUI aesthetics`) using `scripts/import_assets.ps1`, re-enable the icon resource when a valid `assets/icon.ico` is present, and re-run packaging.

**Note:** You mentioned restarting your PC — after reboot, please either merge the PR or let me know and I can open it for you; I’ll continue work (CI monitoring / NSIS retry / GUI verification) once you’re back.

---

## Test Data Locations 🔍
- Video clips: `C:\Users\samue\Downloads\midjourny\` (42 `.mp4`, mixed resolutions)  
- Audio file: `C:\Users\samue\Downloads\we're the imagination-01-01.wav` (6:01 duration)

---

## Quick Start — Next Session 🚀
```bash
cd C:\Users\samue\Desktop\BeatSyncEditor
cmake --build build --config Release

# Test multiclip sync
.\build\bin\Release\beatsync.exe multiclip "C:\Users\samue\Downloads\midjourny" "C:\Users\samue\Downloads\we're the imagination-01-01.wav" -o output.mp4

# Verify output duration
ffprobe output.mp4
```
Goal: Output video should be ~6:01 (matches audio).

---

## Future Enhancements (Ideas) ✨
- Configurable output resolution (not just 1920x1080)  
- Configurable frame rate (not just 24fps)  
- Progress bar with ETA  
- Additional beat detection algorithms  
- GUI interface & Preview mode (process only first N beats)  
- Transition effects (crossfade), random/sequential clip selection  
- Export/import beat grid; support variable clip durations

---

## Known Dependencies 📦
- FFmpeg CLI + dev libraries  
- Windows 10/11  
- `BEATSYNC_FFMPEG_PATH` optional override for detection

---

## Technical Notes 📝

FFmpeg filter chain explanation:
- `scale=1920:1080:force_original_aspect_ratio=decrease` — fit preserving aspect ratio  
- `pad=1920:1080:(ow-iw)/2:(oh-ih)/2` — letterbox/pillarbox to exact output size  
- `setsar=1` — square pixels  
- `fps=24` — consistent framerate

Concatenation rationale:
- Normalize & re-encode during extraction so concatenation can often use `-c copy` safely (fast, no quality loss) when segment properties match.

Audio duration padding logic:
1. Track actual audio duration from `AudioAnalyzer`.  
2. Last beat segment duration = `audioDuration - lastBeatTime`.  
3. If total video < audio duration, add padding segments (2s each, cycling clips).  
4. Avoid `-shortest` so full audio plays.
