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
- Video freezing — **✅ FIXED and tested**
- Duration padding — **✅ FIXED and tested**
- GUI (TripSitter) — **✅ COMPLETE** (`build/bin/Release/TripSitter.exe`). GUI implemented (no longer uses wxWidgets).
- CLI (beatsync) — **✅ COMPLETE** (`build/bin/Release/beatsync.exe`). Full command-line interface for analyze/sync/multiclip/split operations.
- Packaging & CI — **✅ CONFIGURED**. CPack setup for ZIP + NSIS. GitHub Actions workflow in `.github/workflows/windows-build.yml` builds and uploads artifacts.
- Assets — **✅ IMPORTED**. High-quality TripSitter psychedelic fractal backgrounds imported from user's Downloads folder and committed.
- Repository — **✅ INITIALIZED**. Full source code committed to `main` branch at https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI

**Session Completion (2026-01-02):**
✅ All source code committed and pushed to GitHub repository
✅ Main branch created with full project history
✅ CI branch `ci/nsis-smoke-test` updated with latest code and workflow
✅ GUI assets imported (2 splash screen variants, background.png set)
✅ Both executables built and tested locally (beatsync.exe CLI works perfectly)
✅ PNG image loading fixed - wxInitAllImageHandlers() added to GUI initialization
✅ TripSitter.exe rebuilt and verified - background images load correctly
✅ GUI transparency working - psychedelic background fully visible
✅ Window aspect ratio set to 16:9 (1344x768) matching background image
✅ Static boxes removed in favor of section labels for full transparency
✅ Quick rebuild scripts added (quick_rebuild.bat, rebuild_gui.bat)
✅ Project is ready for production use and distribution

**Known Limitations:**
- Native file dialogs remain system-styled (expected behavior)


---

## Session Log (2026-01-03) — UI investigation
wxUniversal (a wxWidgets port) was investigated but ultimately not adopted; the project no longer uses wxWidgets for the GUI. The GUI uses the current UI stack and build configuration.

---

---

## Session Log (2026-01-04) — GUI Background Scrolling & Video Processing Issues 🐛

### Issue 1: Background Image Scrolling ❌
- **Problem:** Background image scrolled with content instead of staying static.
- **Fix Applied:**
  - Used `wxBufferedPaintDC` for double buffering
  - Drew background at fixed position (0,0) in paint handler
  - Set device origin for children to scroll properly
  - Added `wxFULL_REPAINT_ON_RESIZE` style flag
  - Set `SetDoubleBuffered(true)` on scroll window
  - Cached scaled bitmap for efficiency
- **Status:** ✅ FIXED — Background now stays static while content scrolls over it

### Issue 2: Video Segment Extraction Failing ⚠️
- **Problem:** "Error extracting segment: Segment extraction failed"
  - FFmpeg commands work perfectly when run directly from PowerShell
  - `_popen()` calls not capturing FFmpeg stderr output (FFmpeg writes progress/errors to stderr)
  - Concat operations show exit code 0 but empty output in logs

- **Investigation:**
  - Tested FFmpeg commands manually — all work and create valid 10MB+ segment files
  - Issue is in how the application captures FFmpeg output via `_popen()`
  - FFmpeg writes most output to stderr, not stdout

- **Attempted Fix:**
  - Added `2>&1` stderr redirect to ALL FFmpeg `_popen()` calls in `VideoWriter.cpp`:
    1. Line 260 — `copySegmentFast()`
    2. Line 338 — `copySegmentPrecise()`
    3. Line 448 — `concatenateVideos()` main concat
    4. Line 503 — `concatenateVideos()` re-encode fallback
    5. Line 570 — `addAudioTrack()`
    6. Line 680 — `applyEffects()` copy mode
    7. Line 728 — `applyEffects()` with effects

- **Current Status:** ❌ STILL FAILING — Same error after stderr redirect fixes
  - Build completed successfully with all redirects in place
  - Video processing still reports "segment extraction failed"
  - User testing on their own to troubleshoot further

### Issue 3: Selection-Trimmed Exports (TripSitter) 🎯
- **Problem:** When selecting a smaller audio range with the beat visualizer sliders, the exported video sometimes outlasted the trimmed audio or audio ignored the selection.
- **Fixes (2026-01-04):**
  - Clamp per-beat segment end times to the selection end; skip zero-length segments.
  - Pass the selection window to audio mux and seek/trim audio with `-ss/-t` before muxing.
  - Use `-shortest` during mux so output stops at the shorter of trimmed audio/video.
  - Wrap audio mux FFmpeg calls with `cmd /C` and log mux output to `beatsync_ffmpeg_concat.log` for diagnostics.
- **Status:** ✅ Tested — audio and video now end together for partial selections.

### Files Modified (2026-01-04):
- `src/GUI/MainWindow.cpp` — Multiple scroll/paint handling improvements for static background
- `src/video/VideoWriter.cpp` — Added stderr redirect (`2>&1`) to all 7 FFmpeg `_popen()` calls; added audio mux trimming and logging; command wrapping with `cmd /C`

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

# For GUI selection trims
Run TripSitter GUI, drag waveform handles to select a subrange, export, and confirm audio/video end together (mux uses `-shortest` with trimmed audio).

---

## Future Enhancements (Ideas) ✨
- Configurable output resolution (not just 1920x1080)  
- Configurable frame rate (not just 24fps)  
- Progress bar with ETA  
- Additional beat detection algorithms  
- GUI interface & Preview mode (process only first N beats)  
- Transition effects (crossfade), random/sequential clip selection  
- Export/import beat grid; support variable clip durations

- DEFLATE compression for log archives — currently the GUI saves log ZIPs using the fast "store" method with no compression. To implement DEFLATE later:
  1. Add zlib to the build: in `CMakeLists.txt` add `find_package(ZLIB REQUIRED)` and link `${ZLIB_LIBRARIES}` to the target that builds `src/utils/LogArchiver.cpp` (e.g., `target_link_libraries(beatsync PRIVATE ${ZLIB_LIBRARIES})`).
  2. Implement DEFLATE in `BeatSync::createZip` (in `src/utils/LogArchiver.cpp`) using zlib (`deflateInit`, `deflate`, `deflateEnd`) to compress entry data and write appropriate ZIP local/central headers (update sizes and CRC accordingly).
  3. Add a small unit test to verify compressed archive integrity and that the GUI `Save Logs...` honors the `ZipUseDeflate` setting. (There is a hidden Catch2 test stub at `tests/test_deflate_catch2.cpp` tagged `[.deflate]` — replace this placeholder with active assertions when DEFLATE is implemented.)
  4. Update docs and the Logs dialog note to remove the "not implemented" warning.

  Note: The current default remains the fast store method to avoid adding new runtime deps.

---

## Apple Silicon / macOS Tasks (pending) 🍎
- Refresh CMake toolchain/deps for macOS arm64 (Homebrew FFmpeg or vcpkg `ffmpeg:arm64-osx`); ensure `BEATSYNC_FFMPEG_PATH` detection works on macOS.
- Validate builds with `cmake -B build -DCMAKE_OSX_ARCHITECTURES=arm64` (and universal if needed).
- Recreate installers: `.app` bundle + `dmg` via CPack (Bundle/DragNDrop); update `assets/Info.plist.in` for signing/notarization if required.
- Add a GitHub Actions macOS workflow to produce arm64 artifacts (zip + dmg) with caching for Homebrew/vcpkg.
- Manual QA on Apple Silicon: run TripSitter GUI, verify FFmpeg resolution, and export a trimmed selection to confirm audio/video alignment.

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
