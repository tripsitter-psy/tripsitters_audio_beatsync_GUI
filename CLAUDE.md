# Claude Code Instructions for BeatSyncEditor

## Critical Build Instructions

**NEVER mention or attempt to use wxWidgets.** The GUI has been fully migrated to Unreal Engine.

Always build with:
```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DBEATSYNC_SKIP_GUI=ON
cmake --build build --config Release
```

The `BEATSYNC_SKIP_GUI=ON` flag is the standard build configuration.

## Project Architecture

```
BeatSyncEditor/
├── src/
│   ├── audio/           # Audio analysis (AudioAnalyzer, BeatGrid, ONNX detectors)
│   ├── video/           # Video processing (VideoProcessor, VideoWriter, TransitionLibrary)
│   ├── backend/         # C API wrapper (beatsync_capi.h/.cpp) + tracing
│   └── tracing/         # OpenTelemetry tracing support
├── tests/               # Catch2 unit tests
├── unreal-prototype/    # Unreal Engine 5 plugin source (synced to MyProject)
│   ├── Source/TripSitterUE/
│   │   ├── Private/
│   │   │   ├── BeatsyncLoader.cpp      # DLL loading + C API bindings
│   │   │   ├── BeatsyncProcessingTask.cpp  # Async background processing
│   │   │   ├── STripSitterMainWidget.cpp   # Main Slate UI widget
│   │   │   ├── SWaveformViewer.cpp     # Waveform visualization widget
│   │   │   └── TripSitterUEModule.cpp  # Module startup/shutdown
│   │   ├── Public/
│   │   │   ├── BeatsyncLoader.h
│   │   │   ├── STripSitterMainWidget.h
│   │   │   ├── SWaveformViewer.h
│   │   │   └── TripSitterUEModule.h
│   │   └── Resources/
│   │       ├── Corpta.otf              # Custom display font
│   │       ├── wallpaper.png           # Background image
│   │       └── TitleHeader.png         # Title banner
│   └── ThirdParty/beatsync/  # Built DLLs copied here automatically
└── vcpkg/               # Package manager submodule
```

## Deployed UE Plugin Location

**IMPORTANT**: The actual running Unreal project is at:
```
C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\
```

Changes to `unreal-prototype/` in this repo need to be synced/copied there, OR edit files directly in the MyProject location. The UE Editor compiles from the MyProject location, not from this repo.

## Key Build Targets

| Target | Description |
|--------|-------------|
| `beatsync_backend_shared` | DLL for Unreal plugin (auto-copies to ThirdParty) |
| `beatsync_backend_static` | Static lib for tests and internal linking |
| `test_backend_api` | C API unit tests (`build/tests/Release/test_backend_api.exe`) |

## C API (`beatsync_capi.h`)

### Core Functions
- `bs_init()` / `bs_shutdown()` - Library lifecycle
- `bs_get_version()` - Returns version string

### Audio Analysis
- `bs_create_audio_analyzer()` / `bs_destroy_audio_analyzer()`
- `bs_analyze_audio()` - Detect beats, returns `bs_beatgrid_t`
- `bs_get_waveform()` - Get downsampled peaks for visualization
- `bs_free_beatgrid()` / `bs_free_waveform()` - Memory cleanup

### Video Processing
- `bs_create_video_writer()` / `bs_destroy_video_writer()`
- `bs_video_cut_at_beats()` - Cut single video at beat times
- `bs_video_cut_at_beats_multi()` - Cut multiple videos, cycling through
- `bs_video_concatenate()` - Join video files
- `bs_video_add_audio_track()` - Mux audio into video
- `bs_video_set_progress_callback()` - Progress reporting

### Effects
- `bs_video_set_effects_config()` - Configure transitions, color grading, vignette, beat flash/zoom
- `bs_video_apply_effects()` - Apply configured effects with beat times

### Frame Extraction
- `bs_video_extract_frame()` - Extract RGB frame at timestamp (for preview)
- `bs_free_frame_data()` - Free extracted frame

### Tracing (optional)
- `bs_initialize_tracing()` / `bs_shutdown_tracing()`
- `bs_start_span()` / `bs_end_span()` / `bs_span_set_error()` / `bs_span_add_event()`

## Unreal Plugin Components

### BeatsyncLoader (DLL Interface)
- Dynamically loads `beatsync_backend_shared.dll`
- Wraps all C API functions with UE-friendly types
- `FEffectsConfig` struct mirrors `bs_effects_config_t`

### FBeatsyncProcessingTask (Async Processing)
- `FAsyncTask` subclass for non-blocking video processing
- Progress callback updates UI via game thread delegate
- Stages: Analyze Audio → Cut Video → Apply Effects → Add Audio
- **IMPORTANT**: Header must be included (not forward declared) in STripSitterMainWidget.h because `FAsyncTask<T>` requires complete type

### STripSitterMainWidget (UI)
- Main Slate widget with file selection, beat visualization, effects controls
- Preview texture from `bs_video_extract_frame()`
- Async processing with progress bar
- Custom Corpta font for headings (loaded from Resources/Corpta.otf)
- **Note**: Help text uses default system font (Corpta lacks some glyphs like `|`)

### SWaveformViewer
- Custom Slate widget for audio waveform display
- Supports selection handles, zoom, pan
- Beat marker overlay

## Custom Font (Corpta)

The UI uses a custom display font "Corpta" for headings. Located at:
- Source: `unreal-prototype/Source/TripSitterUE/Resources/Corpta.otf`
- Deployed: `C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\Resources\Corpta.otf`

Font is loaded at runtime via `FSlateFontInfo(AbsolutePath, Size)`. Falls back to `FCoreStyle::GetDefaultFontStyle()` if not found.

**Corpta font limitations**: Does not include all ASCII glyphs (missing `|` pipe, some punctuation). Use default system font for body text and help strings.

## vcpkg Dependencies

Defined in `vcpkg.json`:
- `ffmpeg` (avcodec, avformat, swresample, swscale, avfilter)
  - **Note**: `avutil` removed from features list - now included in core as of FFmpeg 8.0.1
- `onnxruntime` (AI beat detection)

Baseline: `25b458671af03578e6a34edd8f0d1ac85e084df4` (vcpkg submodule HEAD)

## Current Status (January 2026)

### Completed
- [x] C API with effects config and frame extraction
- [x] BeatsyncLoader updated with all new functions
- [x] FBeatsyncProcessingTask for async processing
- [x] STripSitterMainWidget with preview support
- [x] vcpkg baseline fix (onnxruntime issue)
- [x] FFmpeg 8.0.1 feature fix (avutil now in core)
- [x] BEATSYNC_SKIP_GUI CMake option
- [x] Backend DLL builds successfully
- [x] test_backend_api passes
- [x] Custom Corpta font integration
- [x] VideoWriter separator check fix (empty string .back() issue)

### Pending / Future Work
- [ ] Test effects pipeline end-to-end with real video
- [ ] Test frame extraction in UE preview widget
- [ ] Verify async task completion and UI updates
- [ ] Add more comprehensive C API tests
- [ ] ONNX beat detection model integration

## Common Issues & Fixes

### UE Plugin won't compile after source changes
The UE Editor compiles from `C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\`, not from this repo. Either:
1. Edit files directly in the MyProject location, OR
2. Copy changed files from `unreal-prototype/` to the MyProject plugin folder

To force recompile: Delete `Intermediate/` and `Binaries/` folders in the plugin directory, then reopen UE Editor.

### "FAsyncTask uses undefined class" error
The `FBeatsyncProcessingTask` class must be fully defined (not forward declared) when used with `FAsyncTask<T>`. Ensure `#include "BeatsyncProcessingTask.h"` is in STripSitterMainWidget.h.

### Font shows garbled characters
The Corpta font is missing some ASCII glyphs. Use `FCoreStyle::GetDefaultFontStyle()` for body text and help strings that contain punctuation like `|`.

### vcpkg baseline errors
Use the vcpkg submodule HEAD commit as baseline: `25b458671af03578e6a34edd8f0d1ac85e084df4`

### "avutil feature not found" error
FFmpeg 8.0.1 moved avutil to core. Remove "avutil" from the features list in vcpkg.json.

## Quick Reference

```powershell
# Build backend DLL
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DBEATSYNC_SKIP_GUI=ON
cmake --build build --config Release --target beatsync_backend_shared

# Build and run tests
cmake --build build --config Release --target test_backend_api
./build/tests/Release/test_backend_api.exe

# DLL output location
build/Release/beatsync_backend_shared.dll
unreal-prototype/ThirdParty/beatsync/lib/x64/beatsync_backend_shared.dll

# Force UE plugin recompile (run in PowerShell)
Remove-Item -Recurse -Force 'C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\Intermediate' -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force 'C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\Binaries' -ErrorAction SilentlyContinue

# Copy font to deployed plugin
Copy-Item 'unreal-prototype\Source\TripSitterUE\Resources\Corpta.otf' 'C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\Resources\'
```

## Git Workflow

Current branch: `ci/nsis-smoke-test`
Remote: `https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI.git`

Commit style: `type: description` (e.g., `feat:`, `fix:`, `ci:`)
