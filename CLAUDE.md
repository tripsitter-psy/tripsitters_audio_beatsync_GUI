# Claude Code Instructions for BeatSyncEditor

> **Path Convention**: All user-specific paths use `%USERPROFILE%` (for documentation/display) or `$env:USERPROFILE` (for PowerShell commands) instead of hardcoded usernames.
> Shared/infrastructure paths (such as the Unreal Engine source location, e.g., C:\UE5_Source\UnrealEngine) may remain hardcoded for clarity. Only user-specific locations (home, Documents, OneDrive, etc.) are replaced with environment variables. Exempt paths: UE source, vcpkg root, build output directories.

## Critical Build Instructions

The GUI is implemented in Unreal Engine (TripSitter standalone app). The C++ backend provides the core audio/video processing with ONNX Runtime for AI-powered beat detection.

**Unreal Engine**: Source-built at `C:\UE5_Source\UnrealEngine` (NOT Epic Games Launcher install).

### Backend Build (with CUDA + TensorRT GPU Acceleration + AudioFlux)

```powershell
# Configure with vcpkg, overlay triplet for TensorRT, and AudioFlux
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DAUDIOFLUX_ROOT="C:/audioFlux"

# Build
cmake --build build --config Release
```

**Important**:
- The `triplets/x64-windows.cmake` overlay sets `TENSORRT_HOME` for ONNX Runtime GPU acceleration
- **AudioFlux** is required for spectral flux beat detection. Set `AUDIOFLUX_ROOT` to the AudioFlux installation directory (e.g., `C:/audioFlux`)
- If AudioFlux is not found, the "Flux" and "Stems + Flux" analysis modes will be unavailable and fall back to energy-based detection

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
│   │   ├── AudioFluxBeatDetector.cpp/h # AudioFlux spectral analysis (requires AudioFlux library)
│   │   └── SpectralFlux.cpp/h      # Basic spectral flux analysis
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

### AudioFlux Analysis (Spectral Processing)
- `int bs_audioflux_is_available()` - Check if AudioFlux library is available (returns 0 if not built with USE_AUDIOFLUX)
- `int bs_audioflux_analyze(const char* audio_path, bs_ai_result_t* out_result, bs_ai_progress_cb cb, void* user_data)` - Spectral flux beat detection
- `int bs_audioflux_analyze_with_stems(const char* audio_path, const char* stem_model_path, bs_ai_result_t* out_result, bs_ai_progress_cb cb, void* user_data)` - Stems + Flux hybrid analysis (best quality)

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
- TensorRT 10.9.0.34 (location set via `$env:TENSORRT_HOME`)
- Overlay triplet `triplets/x64-windows.cmake` sets `TENSORRT_HOME`

**GPU Execution Provider Fallback Chain**:

The `OnnxBeatDetector` automatically selects the best available execution provider:

1. **TensorRT** (RTX GPUs) - Best performance on RTX cards with Tensor Cores, FP16 enabled
2. **CUDA** (GTX/RTX GPUs) - Falls back if TensorRT unavailable or fails
3. **CPU** - Final fallback if no GPU acceleration available

This ensures the app works on any system while maximizing performance on NVIDIA GPUs.

Note: `avutil` is included in FFmpeg core and is not a selectable vcpkg feature.

**vcpkg baseline:** The required vcpkg baseline commit is `25b458671af03578e6a34edd8f0d1ac85e084df4` (see `vcpkg.json`).

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
- [x] TensorRT → CUDA → CPU fallback chain in OnnxBeatDetector
- [x] TensorRT DLL bundling in NSIS installer
- [x] CMake post-build TensorRT DLL copying
- [x] Beat effects using chained eq filters (FFmpeg expression limit fix)
- [x] CodeRabbit fixes (thread safety, memory leaks, bounds checks)
- [x] AudioFlux integration for spectral flux beat detection
- [x] Stems + Flux hybrid analysis mode

### Pending / Future Work
- [ ] Test effects pipeline end-to-end with real video
- [ ] Test frame extraction in UE preview widget
- [ ] Verify async task completion and UI updates
- [ ] Add more comprehensive C API tests
- [ ] Train/integrate ONNX beat detection models (BeatNet, All-In-One, TCN)

## Required DLLs for TripSitter.exe

**CRITICAL**: TripSitter.exe requires all these DLLs in the same directory (`C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\`). Missing or mismatched DLLs will cause silent failures or crashes.

### Complete DLL List

| DLL | Source | Purpose |
|-----|--------|---------|
| `beatsync_backend_shared.dll` | `build/Release/` | Main backend library |
| `onnxruntime.dll` | `build/Release/` | ONNX Runtime (must be v1.23.x) |
| `onnxruntime_providers_shared.dll` | `build/vcpkg_installed/x64-windows/bin/` | GPU provider interface |
| `onnxruntime_providers_cuda.dll` | `build/vcpkg_installed/x64-windows/bin/` | CUDA execution provider |
| `abseil_dll.dll` | `build/Release/` | ONNX dependency |
| `libprotobuf.dll` | `build/Release/` | ONNX dependency |
| `libprotobuf-lite.dll` | `build/Release/` | ONNX dependency |
| `re2.dll` | `build/Release/` | ONNX dependency |
| `avcodec-62.dll` | `unreal-prototype/ThirdParty/beatsync/lib/x64/` | FFmpeg codec (106MB) |
| `avformat-62.dll` | `unreal-prototype/ThirdParty/beatsync/lib/x64/` | FFmpeg format |
| `avutil-60.dll` | `unreal-prototype/ThirdParty/beatsync/lib/x64/` | FFmpeg utils |
| `avfilter-11.dll` | `unreal-prototype/ThirdParty/beatsync/lib/x64/` | FFmpeg filters |
| `avdevice-62.dll` | `unreal-prototype/ThirdParty/beatsync/lib/x64/` | FFmpeg device |
| `swresample-6.dll` | `unreal-prototype/ThirdParty/beatsync/lib/x64/` | FFmpeg resampling |
| `swscale-9.dll` | `unreal-prototype/ThirdParty/beatsync/lib/x64/` | FFmpeg scaling |

### AudioFlux DLLs (Required for Flux/Stems+Flux modes)

These DLLs enable spectral flux beat detection. Without them, "Flux" and "Stems + Flux" modes will fall back to energy-based detection.

| DLL | Source | Purpose |
|-----|--------|---------|
| `audioflux.dll` | `C:/audioFlux/build/windowBuild/Release/` | AudioFlux spectral analysis library |
| `libfftw3f-3.dll` | `C:/audioFlux/python/audioflux/lib/` | FFTW3 (Fast Fourier Transform) dependency |

**AudioFlux Installation**: `C:\audioFlux` (set via `-DAUDIOFLUX_ROOT="C:/audioFlux"` during CMake configure)

### TensorRT DLLs (Optional - for RTX GPU Acceleration)

These DLLs enable TensorRT acceleration on NVIDIA RTX GPUs. They are optional - the app will fall back to CUDA or CPU if not present.

| DLL | Size | Purpose |
|-----|------|---------|
| `nvinfer_10.dll` | 420 MB | TensorRT inference engine |
| `nvinfer_lean_10.dll` | 42 MB | Lightweight inference runtime |
| `nvinfer_plugin_10.dll` | 49 MB | TensorRT plugins |
| `nvinfer_dispatch_10.dll` | 0.6 MB | Provider dispatch |
| `nvonnxparser_10.dll` | 2.9 MB | ONNX model parser |

**Note**: The `nvinfer_builder_resource_10.dll` (1.88 GB) is NOT needed at runtime - it's only for building TensorRT engines.

### Deployment Script (Recommended)

Use the deployment script which verifies DLL sizes and copies in correct order:

```powershell
# Deploy all DLLs with verification
.\scripts\deploy_tripsitter.ps1

# Verify existing DLLs without copying
.\scripts\deploy_tripsitter.ps1 -Verify

# Dry run to see what would be copied
.\scripts\deploy_tripsitter.ps1 -DryRun
```

### Manual Copy Command

**WARNING**: Order matters! FFmpeg DLLs from ThirdParty MUST be copied LAST because `build/Release/*.dll` contains smaller vcpkg FFmpeg DLLs that will break the app.

```powershell
# Step 1: ONNX Runtime and dependencies (from build)
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force
Copy-Item 'build\Release\onnxruntime.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force
Copy-Item 'build\Release\abseil_dll.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force
Copy-Item 'build\Release\libprotobuf*.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force
Copy-Item 'build\Release\re2.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force

# Step 2: ONNX GPU providers
Copy-Item 'build\vcpkg_installed\x64-windows\bin\onnxruntime_providers_*.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force

# Step 3: FFmpeg from ThirdParty (MUST be last - these are ~106MB not ~13MB!)
Copy-Item 'unreal-prototype\ThirdParty\beatsync\lib\x64\av*.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force
Copy-Item 'unreal-prototype\ThirdParty\beatsync\lib\x64\sw*.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force
```

### DLL Size Verification

If TripSitter crashes on startup or waveform doesn't load, verify FFmpeg DLL sizes:

```powershell
# avcodec-62.dll should be ~106MB, NOT ~13MB
(Get-Item 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\avcodec-62.dll').Length / 1MB
```

If it shows ~13MB, the wrong FFmpeg DLLs were copied. Re-run the deployment script or copy ThirdParty FFmpeg DLLs again.

## Common Issues & Fixes

### DLL fails to load - "Failed to load Beatsync library"
Missing dependency DLLs. Copy ALL DLLs listed above to UE5 Binaries folder.

### ONNX Runtime version mismatch - "API version [23] is not available"
The `onnxruntime.dll` in UE5 Binaries is wrong version. Must use v1.23.x from `build/Release/` (14.6MB), not an older version.

### GPU not being used - silent fallback to CPU
Missing `onnxruntime_providers_shared.dll` and `onnxruntime_providers_cuda.dll`. Copy from `build/vcpkg_installed/x64-windows/bin/`.

### Waveform not loading - "No audio loaded"
Either DLL failed to load (check above) or FFmpeg DLLs are wrong version. The `avcodec-62.dll` should be ~106MB from ThirdParty, not ~13MB.

### App crashes immediately on startup
**Most likely cause**: Wrong FFmpeg DLLs. The `build/Release/` folder contains vcpkg FFmpeg DLLs (~13MB avcodec) which are incompatible. The ThirdParty FFmpeg DLLs (~106MB avcodec) must be used. This happens when `Copy-Item 'build\Release\*.dll'` overwrites the correct FFmpeg DLLs. **Solution**: Always copy ThirdParty FFmpeg DLLs LAST, or use `scripts/deploy_tripsitter.ps1`.

### Backend DLL won't compile - bs_ai_result_t redefinition
The `bs_beatgrid_t` typedef was incorrectly using `struct bs_ai_result_t` as its tag name. Fixed by using `struct bs_beatgrid_t` instead.

### std::numbers::pi not found (C++20 required)
Replaced `std::numbers::pi` with `constexpr double PI = 3.14159265358979323846;` in OnnxBeatDetector.cpp for C++17 compatibility.

### TripSitter compile error - IDesktopPlatform undeclared
The file dialog code used `#if WITH_EDITOR || PLATFORM_DESKTOP` but `IDesktopPlatform` is only available in editor builds. Fixed by changing to `#if WITH_EDITOR` so standalone builds use native Windows file dialogs.

### TensorRT not found during vcpkg build
ONNX Runtime's TensorRT support requires `TENSORRT_HOME` environment variable. Solved using overlay triplet at `triplets/x64-windows.cmake` that sets the environment variable during vcpkg builds.

### AudioFlux/Flux mode falls back to Energy - "AudioFlux not available"
**Symptom**: Selecting "Flux" or "Stems + Flux" analysis mode shows "AudioFlux not available - falling back to Energy mode".

**Root cause**: The backend DLL was not built with AudioFlux support. This happens when CMake cannot find the AudioFlux library during configuration.

**Solution**: Rebuild with `-DAUDIOFLUX_ROOT="C:/audioFlux"`:
```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DAUDIOFLUX_ROOT="C:/audioFlux"
cmake --build build --config Release --target beatsync_backend_shared
```

**Verification**: CMake output should show `AudioFlux found: C:/audioFlux/include, C:/audioFlux/build/windowBuild/Release/audioflux.lib - enabling spectral analysis`

**DLLs required**: `audioflux.dll` and `libfftw3f-3.dll` must be in `C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\`

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

### CRITICAL: Slate threading crash - "IsInGameThread() || IsInSlateThread()"
**Symptom**: App crashes during video processing with assertion `Slate can only be accessed from the GameThread or the SlateLoadingThread!`

**Root cause**: The backend DLL's progress callback (`bs_video_set_progress_callback`) is called from a worker thread, but the UE code was directly updating Slate UI widgets (progress bar, status text) from that callback.

**Solution**: All UI updates from backend callbacks MUST be marshaled to the GameThread:
```cpp
// WRONG - crashes!
FBeatsyncLoader::SetProgressCallback(Writer, [OnProgress](double Prog) {
    OnProgress.ExecuteIfBound(Prog, TEXT("Processing..."));  // Direct call from worker thread
});

// CORRECT - marshal to GameThread
FBeatsyncLoader::SetProgressCallback(Writer, [OnProgress](double Prog) {
    float Progress = static_cast<float>(Prog);
    AsyncTask(ENamedThreads::GameThread, [OnProgress, Progress]() {
        OnProgress.ExecuteIfBound(Progress, TEXT("Processing..."));
    });
});
```

**Files affected**: `BeatsyncProcessingTask.cpp` - the progress callback lambda at ~line 221.

### CRITICAL: FAsyncTask deletion crash - "!QueuedPool" assertion
**Symptom**: App crashes at the END of successful processing with assertion `!QueuedPool` in `FAsyncTaskBase::CheckIdle()`.

**Root cause**: The completion callback is fired from within `DoWork()`, but the async task system doesn't consider the task "done" until `DoWork()` returns. Calling `ProcessingTask.Reset()` in the completion callback deletes the task while it's still technically queued.

**Solution**: Defer task cleanup to after the current execution completes:
```cpp
// WRONG - crashes!
void OnProcessingComplete(const FBeatsyncProcessingResult& Result) {
    // ... handle result ...
    ProcessingTask.Reset();  // Task still "running" - assertion fails!
}

// CORRECT - defer cleanup
void OnProcessingComplete(const FBeatsyncProcessingResult& Result) {
    // ... handle result ...
    if (ProcessingTask.IsValid()) {
        TUniquePtr<FAsyncTask<FBeatsyncProcessingTask>> TaskToCleanup = MoveTemp(ProcessingTask);
        AsyncTask(ENamedThreads::GameThread, [Task = MoveTemp(TaskToCleanup)]() mutable {
            if (Task.IsValid()) {
                Task->EnsureCompletion();
                Task.Reset();
            }
        });
    }
}
```

**Files affected**: `STripSitterMainWidget.cpp` - `OnProcessingComplete()` at ~line 1661.

### FFmpeg scientific notation crash - "Invalid duration for option ss"
**Symptom**: Video processing fails with FFmpeg error `Invalid duration for option ss: 2e-05`.

**Root cause**: C++ `std::ostringstream` uses scientific notation for very small doubles (e.g., `2e-05` instead of `0.000020`). FFmpeg's `-ss` and `-t` parameters reject scientific notation.

**Solution**: Use `std::fixed << std::setprecision(6)` when building FFmpeg command strings:
```cpp
std::ostringstream cmd;
cmd << std::fixed << std::setprecision(6);  // Force decimal notation
cmd << " -ss " << startTime << " -t " << duration;
cmd << std::defaultfloat;  // Reset for other values
```

Also clamp near-zero values from `fmod()` floating-point precision issues:
```cpp
sourceStart = fmod(startTime, cachedDuration);
if (sourceStart < 0.001) {  // Sub-millisecond = essentially zero
    sourceStart = 0.0;
}
```

**Files affected**: `VideoWriter.cpp` (multiple locations), `beatsync_capi.cpp`.

### CRITICAL: Installer shows white blank window - missing Slate content
**Symptom**: After installing the app, TripSitter.exe shows a window but it's completely white/blank. The taskbar icon appears and the window can be moved, but no UI renders.

**Root cause**: The CMakeLists.txt was only installing `Engine/Content/Slate/Fonts` but NOT the essential `Slate/Common` folder which contains all UI brushes, textures, and visual elements required for Slate rendering.

**Solution**: The CMakeLists.txt install section must include ALL essential Slate folders:
```cmake
# Install Slate content (required by UE5 Slate UI rendering)
set(UE_SLATE_DIR "C:/UE5_Source/UnrealEngine/Engine/Content/Slate")
if(EXISTS "${UE_SLATE_DIR}")
    install(DIRECTORY "${UE_SLATE_DIR}/Common" DESTINATION "Engine/Content/Slate")
    install(DIRECTORY "${UE_SLATE_DIR}/Fonts" DESTINATION "Engine/Content/Slate")
    install(DIRECTORY "${UE_SLATE_DIR}/Cursor" DESTINATION "Engine/Content/Slate")
    install(DIRECTORY "${UE_SLATE_DIR}/Old" DESTINATION "Engine/Content/Slate")
    if(EXISTS "${UE_SLATE_DIR}/Checkerboard.png")
        install(FILES "${UE_SLATE_DIR}/Checkerboard.png" DESTINATION "Engine/Content/Slate")
    endif()
endif()
```

**Required Slate folders**:
- `Common` - UI brushes, backgrounds, borders (CRITICAL)
- `Fonts` - Roboto, DroidSans, NotoSans fonts
- `Cursor` - Mouse cursor textures
- `Old` - Legacy brushes some widgets may reference
- `Checkerboard.png` - Transparency indicator texture

**Files affected**: `CMakeLists.txt` (install section around line 480).

### CRITICAL: Black screen after install - Resources at wrong path
**Symptom**: After installing the app, TripSitter.exe shows a black window with no UI content. Slate renders (window exists) but background/fonts/images are missing.

**Root cause**: The CMakeLists.txt was installing resources (wallpaper.png, Corpta.otf, etc.) to `resources/` at the install root, but STripSitterMainWidget.cpp looks for them relative to the executable at `Engine/Binaries/Win64/Resources/`.

**Widget code path resolution** (STripSitterMainWidget.cpp lines 87-115):

```cpp
FString ExeDir = FPaths::GetPath(FPlatformProcess::ExecutablePath());
FString ResourceDir = FPaths::Combine(ExeDir, TEXT("Resources"));
FString WallpaperPath = FPaths::Combine(ResourceDir, TEXT("wallpaper.png"));
```

**Solution**: In CMakeLists.txt, the resource install destination MUST match where the widget looks:

```cmake
# WRONG - installs to C:\Program Files\MTV TripSitter\resources\
install(FILES ... DESTINATION resources)

# CORRECT - installs to C:\Program Files\MTV TripSitter\Engine\Binaries\Win64\Resources\
install(FILES ... DESTINATION Engine/Binaries/Win64/Resources)
```

**Files affected**: `CMakeLists.txt` (install section around line 508).

**Important**: After changing CMakeLists.txt install destinations, you MUST run `cmake -S . -B build ...` to reconfigure before running `cpack`. Otherwise the old cached install rules will be used.

### CRITICAL: Use pre-packaged UE build for installer
**Symptom**: Installed app shows black screen, missing UI, missing fonts, or crashes.

**Root cause**: UE5 Program targets (like TripSitter.exe built from source) require extensive engine runtime files (725+ DLLs, shaders, cooked content) that are impractical to assemble manually.

**Solution**: Use a pre-packaged UE build created by the UE packaging system. The CMakeLists.txt now installs from:

```
%USERPROFILE%\Desktop\TripSitterBuild\Windows\
├── MyProject.exe         -> installed as TripSitter.exe (launcher)
├── Engine/               -> Engine/ (runtime, ThirdParty DLLs, Slate, shaders)
├── MyProject/            -> TripSitter/ (game binaries and cooked content)
│   ├── Binaries/Win64/   -> Contains actual game exe and all DLLs
│   └── Content/Paks/     -> Cooked game assets
└── Manifest_*.txt        -> UE manifest files
```

**To create the pre-packaged build**:
1. Open the TripSitter UE project in Unreal Editor
2. Package for Windows (File > Package Project > Windows)
3. Output to `%USERPROFILE%\Desktop\TripSitterBuild\Windows`

**Files affected**: `CMakeLists.txt` (install section starting around line 379).

## Streamlined Release Workflow

The `scripts/build_release.ps1` script automates the entire build and package process.

### Full Release Build (One Command)

```powershell
# Build everything: backend DLL, TripSitter.exe, installer, and ZIP
.\scripts\build_release.ps1
```

### Stages

The script runs these stages in order:

1. **Stage 1: Build Backend DLL** - Compiles beatsync_backend_shared.dll with CUDA/TensorRT/AudioFlux support
2. **Stage 2: Deploy DLLs** - Runs deploy_tripsitter.ps1 to copy all required DLLs to UE5 Binaries
3. **Stage 3: Build TripSitter.exe** - Syncs source files and builds with UE5
4. **Stage 4: Patch Application Icon** - Uses rcedit to replace the default UE icon with TripSitter.ico
5. **Stage 5: Create NSIS Installer** - Generates the Windows installer
6. **Stage 6: Create Portable ZIP** - Creates a portable archive

### Quick Update Workflow

When making code changes, use these flags to skip unchanged stages:

```powershell
# Backend code changed - rebuild DLL and repackage
.\scripts\build_release.ps1 -SkipTripSitter

# UI code changed - rebuild TripSitter.exe and repackage
.\scripts\build_release.ps1 -SkipBackend

# Just repackage (no code changes)
.\scripts\build_release.ps1 -SkipBackend -SkipTripSitter

# Preview what would be built without executing
.\scripts\build_release.ps1 -DryRun
```

### Icon Patching

The packaged UE build has the default Unreal Engine icon. Stage 4 uses **rcedit** to replace it with the custom TripSitter icon.

**Install rcedit** (one-time setup):
```powershell
# Option 1: Via npm (recommended)
npm install -g rcedit

# Option 2: Download binary to tools/ folder
# Download from: https://github.com/electron/rcedit/releases
# Place rcedit.exe in BeatSyncEditor/tools/
```

If rcedit is not available, the build will continue but the app will have the default UE icon.

### Prerequisites Checklist

Before running the release script, ensure:

- [ ] vcpkg submodule initialized (`git submodule update --init --recursive`)
- [ ] UE5 Source build at `C:\UE5_Source\UnrealEngine`
- [ ] Pre-packaged UE build at `%USERPROFILE%\Desktop\TripSitterBuild\Windows\`
- [ ] AudioFlux installed at `C:\audioFlux` (optional, for Flux mode)
- [ ] NSIS installed and in PATH (for installer creation)
- [ ] rcedit installed (for custom icon)

### Output Files

After a successful build, find the outputs in `build/`:

| File                                | Description              |
|-------------------------------------|--------------------------|
| `MTVTripSitter-*-Windows-AMD64.exe` | NSIS Installer (~1.5 GB) |
| `MTVTripSitter-*-Windows-AMD64.zip` | Portable ZIP (~1.8 GB)   |

## Quick Reference

```powershell
# Build backend DLL (with TensorRT + AudioFlux support)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DAUDIOFLUX_ROOT="C:/audioFlux"
cmake --build build --config Release --target beatsync_backend_shared

# Build and run tests
cmake --build build --config Release --target test_backend_api
./build/tests/Release/test_backend_api.exe

# DLL output location
build/Release/beatsync_backend_shared.dll
unreal-prototype/ThirdParty/beatsync/lib/x64/beatsync_backend_shared.dll

# Copy DLL to ThirdParty
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'unreal-prototype\ThirdParty\beatsync\lib\x64\' -Force

# Deploy to UE (includes AudioFlux DLLs)
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force
Copy-Item 'C:\audioFlux\build\windowBuild\Release\audioflux.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force
Copy-Item 'C:\audioFlux\python\audioflux\lib\libfftw3f-3.dll' 'C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\' -Force

# Build TripSitter UE
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination 'C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\Private\' -Recurse -Force
& "C:\UE5_Source\UnrealEngine\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development

# TripSitter executable location
C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe

# AudioFlux installation
C:\audioFlux  # Set via -DAUDIOFLUX_ROOT during CMake configure
```

## Git Workflow

Current branch: run `git branch --show-current` to see active branch
Remote: `https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI.git`

Commit style: `type: description` (e.g., `feat:`, `fix:`, `ci:`)
