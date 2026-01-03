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
- GUI (TripSitter) — **✅ COMPLETE** (`build/bin/Release/TripSitter.exe`). wxWidgets-based GUI with PREVIEW FRAME button, timestamp input, `VideoPreview::LoadFrame` implementation, and `BeatVisualizer` for beat visualization.
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
- wxUniversal is less tested than native wxWidgets ports

---

## Session Log (2026-01-03) — wxUniversal Integration 🎨

### Goal: Static Background + Fully Custom Theme
**Problem:** Background image scrolled with content; native controls couldn't be fully styled.
**Solution:** wxUniversal — a wxWidgets port that renders all controls itself, enabling:
- Static background that stays fixed while UI scrolls over it
- Complete control over all widget rendering via custom theme

### Phase 1: Build wxWidgets with wxUniversal ✅

**Build Script Created:** `C:\Users\samue\Desktop\build_wxuniv.bat`
```batch
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\wxWidgets-3.2.4\build\msw
nmake /f makefile.vc BUILD=release SHARED=1 WXUNIV=1 UNICODE=1 TARGET_CPU=X64
```

**Output Libraries:** `C:\wxWidgets-3.2.4\lib\vc_x64_dll\`
- `wxmswuniv32u_core_vc_x64.dll`
- `wxmswuniv32u_base_vc_x64.dll`
- `wxmswuniv32u_adv_vc_x64.dll`

### Phase 2: PsychedelicTheme Implementation ✅

**New Files:**
- `src/gui/PsychedelicTheme.h` — Theme header with color palette
- `src/gui/PsychedelicTheme.cpp` — Full theme implementation

**Color Palette:**
| Element | Color | Hex |
|---------|-------|-----|
| Primary | Neon Cyan | #00D9FF |
| Secondary | Neon Purple | #8B00FF |
| Background | Dark Blue-Black | #0A0A1A |
| Surface | Dark Gray-Blue | #141428 |
| Text | Light Blue-White | #C8DCFF |
| Accent | Hot Pink | #FF0080 |

**Theme Features:**
- Custom button rendering with gradients
- Glow effects on hover/focus
- Custom checkbox and radio button drawing
- Styled scrollbars and progress bars
- Transparent control backgrounds

### Phase 3: CMakeLists.txt Updates ✅

**New Option Added:**
```cmake
option(USE_WXUNIVERSAL "Use wxUniversal build for custom theming" OFF)

if(WIN32)
    if(USE_WXUNIVERSAL)
        set(wxWidgets_CONFIGURATION mswunivu)
        add_definitions(-D__WXUNIVERSAL__)
    else()
        set(wxWidgets_CONFIGURATION mswu)
    endif()
endif()
```

### Phase 4: MainWindow Updates ✅

**Changes to `src/gui_main.cpp`:**
- Added theme registration: `WX_USE_THEME(psychedelic);`
- Theme initialization in `OnInit()`: `wxTheme::Set(wxTheme::Create("psychedelic"));`

**Changes to `src/gui/MainWindow.cpp`:**
- Conditional code with `#ifdef __WXUNIVERSAL__`
- Simplified background handling for wxUniversal (transparent scrolled panel)
- Frame-level paint handler for static background

### Build Commands

**Standard Build (Native Controls):**
```bash
cmake -B build
cmake --build build --config Release
```

**wxUniversal Build (Custom Theme):**
```bash
cmake -B build -DUSE_WXUNIVERSAL=ON
cmake --build build --config Release
```

### Status
- ✅ wxWidgets wxUniversal libraries built
- ✅ PsychedelicTheme files created
- ✅ CMakeLists.txt updated with USE_WXUNIVERSAL option
- ✅ MainWindow updated for layered panel approach
- ⏳ Rebuild and test with USE_WXUNIVERSAL=ON

**Repository Structure:**
- `main` branch — stable release code with assets
- `ci/nsis-smoke-test` branch — includes CI workflow for NSIS packaging
- GitHub Actions will build packages (ZIP + NSIS) automatically on push
- Artifacts available at: https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI/actions

**Ready for:**
1. ✅ Local testing — Build and run immediately
2. ✅ CI packaging — Push triggers GitHub Actions workflow
3. ✅ Distribution — Download artifacts from Actions or run `cpack -C Release` locally
4. ⏭️ Optional: Merge `ci/nsis-smoke-test` to `main` to enable CI on main branch

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

- DEFLATE compression for log archives — currently the GUI saves log ZIPs using the fast "store" method with no compression. To implement DEFLATE later:
  1. Add zlib to the build: in `CMakeLists.txt` add `find_package(ZLIB REQUIRED)` and link `${ZLIB_LIBRARIES}` to the target that builds `src/utils/LogArchiver.cpp` (e.g., `target_link_libraries(beatsync PRIVATE ${ZLIB_LIBRARIES})`).
  2. Implement DEFLATE in `BeatSync::createZip` (in `src/utils/LogArchiver.cpp`) using zlib (`deflateInit`, `deflate`, `deflateEnd`) to compress entry data and write appropriate ZIP local/central headers (update sizes and CRC accordingly).
  3. Add a small unit test to verify compressed archive integrity and that the GUI `Save Logs...` honors the `ZipUseDeflate` setting. (There is a hidden Catch2 test stub at `tests/test_deflate_catch2.cpp` tagged `[.deflate]` — replace this placeholder with active assertions when DEFLATE is implemented.)
  4. Update docs and the Logs dialog note to remove the "not implemented" warning.

  Note: The current default remains the fast store method to avoid adding new runtime deps.

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
