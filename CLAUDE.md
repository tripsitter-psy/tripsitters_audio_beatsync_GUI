# Claude Code Instructions for BeatSyncEditor

> **Path Convention**: All user-specific paths use `%USERPROFILE%` (for documentation/display) or `$env:USERPROFILE` (for PowerShell commands) instead of hardcoded usernames.

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
%USERPROFILE%\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\
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
- `const char* bs_get_version()` - Returns version string
- `int bs_init()` / `void bs_shutdown()` - Library lifecycle

### Audio Analysis
- `void* bs_create_audio_analyzer()` / `void bs_destroy_audio_analyzer(void* analyzer)`
- `int bs_analyze_audio(void* analyzer, const char* filepath, bs_beatgrid_t* outGrid)` - Detect beats, returns `bs_beatgrid_t`
- `int bs_get_waveform(void* analyzer, const char* filepath, float** outPeaks, size_t* outCount, double* outDuration)` - Get downsampled peaks for visualization
- `void bs_free_beatgrid(bs_beatgrid_t* grid)` / `void bs_free_waveform(float* peaks)` - Memory cleanup

### Video Processing
- `void* bs_create_video_writer()` / `void bs_destroy_video_writer(void* writer)`
- `const char* bs_video_get_last_error(void* writer)` - Get last error message
- `const char* bs_resolve_ffmpeg_path()` - Get resolved FFmpeg path
- `void bs_video_set_progress_callback(void* writer, bs_progress_cb cb, void* user_data)` - Set progress callback
- `int bs_video_cut_at_beats(void* writer, const char* inputVideo, const double* beatTimes, size_t count, const char* outputVideo, double clipDuration)` - Cut single video at beat times
- `int bs_video_cut_at_beats_multi(void* writer, const char** inputVideos, size_t videoCount, const double* beatTimes, size_t beatCount, const char* outputVideo, double clipDuration)` - Cut multiple videos, cycling through
- `int bs_video_concatenate(const char** inputs, size_t count, const char* outputVideo)` - Join video files
- `int bs_video_add_audio_track(void* writer, const char* inputVideo, const char* audioFile, const char* outputVideo, int trimToShortest, double audioStart, double audioEnd)` - Mux audio into video

### Effects
- `void bs_video_set_effects_config(void* writer, const bs_effects_config_t* config)` - Configure transitions, color grading, vignette, beat flash/zoom
- `int bs_video_apply_effects(void* writer, const char* inputVideo, const char* outputVideo, const double* beatTimes, size_t beatCount)` - Apply configured effects with beat times

### Frame Extraction
- `int bs_video_extract_frame(const char* videoPath, double timestamp, unsigned char** outData, int* outWidth, int* outHeight)` - Extract RGB frame at timestamp (for preview)
- `void bs_free_frame_data(unsigned char* data)` - Free extracted frame

### Tracing (optional)
- `int bs_initialize_tracing(const char* service_name)` / `void bs_shutdown_tracing()` - Tracing lifecycle
- `bs_span_t bs_start_span(const char* name)` / `void bs_end_span(bs_span_t span)` - Span management
- `void bs_span_set_error(bs_span_t span, const char* msg)` / `void bs_span_add_event(bs_span_t span, const char* event)` - Span operations

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
- Deployed: `%USERPROFILE%\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\Resources\Corpta.otf`

Font is loaded at runtime via `FSlateFontInfo(AbsolutePath, Size)`. Falls back to `FCoreStyle::GetDefaultFontStyle()` if not found.

**Corpta font limitations**: Does not include all ASCII glyphs (missing `|` pipe, some punctuation). Use default system font for body text and help strings.

## vcpkg Dependencies

Defined in `vcpkg.json`:
- `ffmpeg` (avcodec, avformat, swresample, swscale, avfilter)
  - **Note**: `avutil` removed from features list - now included in core as of FFmpeg 8.0.1
- `onnxruntime` (AI beat detection)

Baseline: [configured in vcpkg.json](vcpkg.json) (vcpkg submodule HEAD)

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
The UE Editor compiles from `%USERPROFILE%\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\`, not from this repo. Either:
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
Remove-Item -Recurse -Force "$env:USERPROFILE\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\Intermediate" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:USERPROFILE\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\Binaries" -ErrorAction SilentlyContinue

# Copy font to deployed plugin
Copy-Item 'unreal-prototype\Source\TripSitterUE\Resources\Corpta.otf' "$env:USERPROFILE\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\Resources\"
```

## Git Workflow

Current branch: run `git branch --show-current` to see active branch
Remote: `https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI.git`

Commit style: `type: description` (e.g., `feat:`, `fix:`, `ci:`)
