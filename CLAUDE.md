# Claude Code Instructions for BeatSyncEditor

## Critical Build Instructions

**NEVER mention or attempt to use wxWidgets.** The GUI has been fully migrated to Unreal Engine.

Always build with:
```
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
├── unreal-prototype/    # Unreal Engine 5 plugin (THE ONLY GUI)
│   ├── Source/TripSitterUE/
│   │   ├── Private/
│   │   │   ├── BeatsyncLoader.cpp      # DLL loading + C API bindings
│   │   │   ├── BeatsyncProcessingTask.cpp  # Async background processing
│   │   │   └── STripSitterMainWidget.cpp   # Main Slate UI widget
│   │   └── Public/
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

### STripSitterMainWidget (UI)
- Main Slate widget with file selection, beat visualization, effects controls
- Preview texture from `bs_video_extract_frame()`
- Async processing with progress bar

## vcpkg Dependencies

Defined in `vcpkg.json`:
- `ffmpeg` (avcodec, avformat, swresample, swscale, avfilter)
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

### Pending / Future Work
- [ ] Test effects pipeline end-to-end with real video
- [ ] Test frame extraction in UE preview widget
- [ ] Verify async task completion and UI updates
- [ ] Add more comprehensive C API tests
- [ ] ONNX beat detection model integration

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
```
