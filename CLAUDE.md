# Claude Code Instructions for BeatSyncEditor

> **Path Convention**: All user-specific paths use `%USERPROFILE%` (for documentation/display) or `$env:USERPROFILE` (for PowerShell commands) instead of hardcoded usernames.
> Shared/infrastructure paths (such as the Unreal Engine source location, e.g., C:\UE5_Source\UnrealEngine) may remain hardcoded for clarity. Only user-specific locations (home, Documents, OneDrive, etc.) are replaced with environment variables. Exempt paths: UE source, vcpkg root, build output directories.

## Critical Build Instructions

The GUI is implemented in Unreal Engine (TripSitter standalone app). The C++ backend provides the core audio/video processing with ONNX Runtime for AI-powered beat detection.

**Unreal Engine**: Source-built at `C:\UE5_Source\UnrealEngine` (NOT Epic Games Launcher install).

### Backend Build (with CUDA + TensorRT GPU Acceleration)

```powershell
# Configure with vcpkg and overlay triplet for TensorRT
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake --overlay-triplets=triplets

# Build
cmake --build build --config Release
```

**Important**: The `triplets/x64-windows.cmake` overlay sets `TENSORRT_HOME` for ONNX Runtime GPU acceleration. TensorRT must be installed at `C:\TensorRT-10.9.0.34`.

### TripSitter UE Build

```powershell
# Copy source files to engine
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination 'C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\Private\' -Recurse -Force

# Build
& "C:\UE5_Source\UnrealEngine\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
```

**Output**: `C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe`

## Project Architecture

```
BeatSyncEditor/
├── src/
│   ├── audio/           # Audio analysis (AudioAnalyzer, BeatGrid, ONNX detectors)
│   │   ├── OnnxBeatDetector.cpp/h  # ONNX Runtime neural network inference
│   │   ├── OnnxMusicAnalyzer.cpp/h # High-level AI music analysis
│   │   └── SpectralFlux.cpp/h      # Spectral analysis
│   ├── video/           # Video processing (VideoProcessor, VideoWriter, TransitionLibrary)
│   ├── backend/         # C API wrapper (beatsync_capi.h/.cpp) + tracing
│   └── tracing/         # OpenTelemetry tracing support
├── tests/               # Catch2 unit tests
├── triplets/            # vcpkg overlay triplets (for TensorRT)
│   └── x64-windows.cmake
├── unreal-prototype/    # Unreal Engine 5 standalone program source
│   ├── Source/TripSitter/
│   │   ├── Private/
│   │   │   ├── BeatsyncLoader.cpp      # DLL loading + C API bindings
│   │   │   ├── BeatsyncProcessingTask.cpp  # Async background processing
│   │   │   ├── STripSitterMainWidget.cpp   # Main Slate UI widget
│   │   │   ├── SWaveformViewer.cpp     # Waveform visualization widget
│   │   └── Resources/
│   │       ├── Corpta.otf              # Custom display font
│   │       ├── wallpaper.png           # Background image
│   │       └── TitleHeader.png         # Title banner
│   └── ThirdParty/beatsync/  # Built DLLs copied here automatically
└── vcpkg/               # Package manager submodule
```

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

### Audio Analysis (Basic)
- `void* bs_create_audio_analyzer()` / `void bs_destroy_audio_analyzer(void* analyzer)`
- `int bs_analyze_audio(void* analyzer, const char* filepath, bs_beatgrid_t* outGrid)` - Detect beats, returns `bs_beatgrid_t`
- `int bs_get_waveform(void* analyzer, const char* filepath, float** outPeaks, size_t* outCount, double* outDuration)` - Get downsampled peaks for visualization
- `void bs_free_beatgrid(bs_beatgrid_t* grid)` / `void bs_free_waveform(float* peaks)` - Memory cleanup

### AI Analysis (ONNX Runtime)
- `void* bs_create_ai_analyzer(const bs_ai_config_t* config)` - Create AI analyzer with model paths
- `void bs_destroy_ai_analyzer(void* analyzer)` - Destroy AI analyzer
- `int bs_ai_analyze_file(void* analyzer, const char* audio_path, bs_ai_result_t* out_result, bs_ai_progress_cb cb, void* user_data)` - Full AI analysis with stem separation
- `int bs_ai_analyze_quick(void* analyzer, const char* audio_path, bs_ai_result_t* out_result, bs_ai_progress_cb cb, void* user_data)` - Quick analysis without stems
- `void bs_free_ai_result(bs_ai_result_t* result)` - Free AI result data
- `int bs_ai_is_available()` - Check if ONNX Runtime is available
- `const char* bs_ai_get_providers()` - Get available execution providers (CPU, CUDA, TensorRT)

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

## vcpkg Dependencies

Defined in `vcpkg.json`:
- `ffmpeg` (avcodec, avformat, swresample, swscale, avfilter)
- `onnxruntime` (AI beat detection with CUDA/TensorRT GPU acceleration)

**GPU Acceleration Requirements**:
- CUDA Toolkit 12.x
- TensorRT 10.9.0.34 installed at `C:\TensorRT-10.9.0.34`
- Overlay triplet `triplets/x64-windows.cmake` sets `TENSORRT_HOME`

Note: `avutil` is included in FFmpeg core and is not a selectable vcpkg feature.

Baseline: [configured in vcpkg.json](vcpkg.json) (vcpkg submodule HEAD)

## Current Status (January 2026)

### Completed
- [x] C API with effects config and frame extraction
- [x] BeatsyncLoader updated with all new functions
- [x] FBeatsyncProcessingTask for async processing
- [x] STripSitterMainWidget with preview support
- [x] vcpkg baseline fix (onnxruntime issue)
- [x] FFmpeg 8.0.1 feature fix (avutil now in core)
- [x] Deprecated wxWidgets GUI removed (src/GUI deleted)
- [x] Backend DLL builds successfully
- [x] test_backend_api passes
- [x] Custom Corpta font integration
- [x] VideoWriter separator check fix (empty string .back() issue)
- [x] ONNX Runtime 1.23.2 with CUDA + TensorRT support
- [x] TensorRT 10.9.0.34 integration via overlay triplet
- [x] Fixed bs_ai_result_t struct definition (removed nested typedef)
- [x] Fixed std::numbers::pi C++20 issue (replaced with constexpr PI)
- [x] Fixed missing brace in bs_ai_analyze_quick function
- [x] Fixed IDesktopPlatform preprocessor condition for standalone builds
- [x] TripSitter.exe builds successfully

### Pending / Future Work
- [ ] Test effects pipeline end-to-end with real video
- [ ] Test frame extraction in UE preview widget
- [ ] Verify async task completion and UI updates
- [ ] Add more comprehensive C API tests
- [ ] Train/integrate ONNX beat detection models (BeatNet, All-In-One, TCN)

## Common Issues & Fixes

### Backend DLL won't compile - bs_ai_result_t redefinition
The `bs_beatgrid_t` typedef was incorrectly using `struct bs_ai_result_t` as its tag name. Fixed by using `struct bs_beatgrid_t` instead.

### std::numbers::pi not found (C++20 required)
Replaced `std::numbers::pi` with `constexpr double PI = 3.14159265358979323846;` in OnnxBeatDetector.cpp for C++17 compatibility.

### TripSitter compile error - IDesktopPlatform undeclared
The file dialog code used `#if WITH_EDITOR || PLATFORM_DESKTOP` but `IDesktopPlatform` is only available in editor builds. Fixed by changing to `#if WITH_EDITOR` so standalone builds use native Windows file dialogs.

### TensorRT not found during vcpkg build
ONNX Runtime's TensorRT support requires `TENSORRT_HOME` environment variable. Solved using overlay triplet at `triplets/x64-windows.cmake` that sets the environment variable during vcpkg builds.

### UE Plugin won't compile after source changes
The UE Editor compiles from `C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\`, not from this repo. Copy files from `unreal-prototype/Source/TripSitter/` to the engine Programs folder.

### "FAsyncTask uses undefined class" error
The `FBeatsyncProcessingTask` class must be fully defined (not forward declared) when used with `FAsyncTask<T>`. Ensure `#include "BeatsyncProcessingTask.h"` is in STripSitterMainWidget.h.

### Font shows garbled characters
The Corpta font is missing some ASCII glyphs. Use `FCoreStyle::GetDefaultFontStyle()` for body text and help strings that contain punctuation like `|`.

### vcpkg baseline errors
Check the `"builtin-baseline"` value in vcpkg.json for the current baseline commit hash to use with the vcpkg submodule.

### "avutil feature not found" error
FFmpeg 8.0.1 moved avutil to core. Remove "avutil" from the features list in vcpkg.json.

## Quick Reference

```powershell
# Build backend DLL (with TensorRT support)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake --overlay-triplets=triplets
cmake --build build --config Release --target beatsync_backend_shared

# Build and run tests
cmake --build build --config Release --target test_backend_api
./build/tests/Release/test_backend_api.exe

# DLL output location
build/Release/beatsync_backend_shared.dll
unreal-prototype/ThirdParty/beatsync/lib/x64/beatsync_backend_shared.dll

# Copy DLL to ThirdParty
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'unreal-prototype\ThirdParty\beatsync\lib\x64\' -Force

# Build TripSitter UE
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination 'C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\Private\' -Recurse -Force
& "C:\UE5_Source\UnrealEngine\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development

# TripSitter executable location
C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe
```

## Git Workflow

Current branch: run `git branch --show-current` to see active branch
Remote: `https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI.git`

Commit style: `type: description` (e.g., `feat:`, `fix:`, `ci:`)
