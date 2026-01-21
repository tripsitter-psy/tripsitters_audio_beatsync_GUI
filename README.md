# BeatSync Editor

A C++ desktop application for automatically synchronizing video clips with music beats using audio analysis, AI-powered beat detection, and intelligent cutting algorithms.

## Features

- **AI-Powered Beat Detection**: ONNX Runtime with CUDA/TensorRT GPU acceleration
- **Spectral Flux Beat Detection**: AudioFlux-based onset detection with stem separation support
- **Video Processing**: FFmpeg-based cutting, concatenation, and effects
- **Modern GUI**: Unreal Engine 5 Slate UI (TripSitter standalone app)
- **Waveform Visualization**: Interactive audio waveform with beat markers
- **Effects Pipeline**: Transitions, color grading, beat-synced flash/zoom

## Architecture

The project consists of two main components:

1. **Backend DLL** (`beatsync_backend.dll`)
   - C++ library with C API for cross-language compatibility
   - FFmpeg for audio/video processing
   - ONNX Runtime for neural network inference
   - GPU acceleration via CUDA and TensorRT

2. **TripSitter GUI** (Unreal Engine 5)
   - Standalone program (not a game)
   - Slate UI for native look and feel
   - Async processing with progress feedback
   - Native Windows file dialogs

## Requirements

### Build Tools

- CMake 3.20+
- Visual Studio 2022 (MSVC - Microsoft Visual C++ compiler)
- Unreal Engine 5 (UE5 - source build)

### Dependencies (via vcpkg)

- FFmpeg (avcodec, avformat, swresample, swscale, avfilter)
- ONNX Runtime 1.23.2

vcpkg baseline

- The repository pins a vcpkg "builtin-baseline" in `vcpkg.json` to ensure reproducible dependency versions. The expected baseline commit in this repo is `25b458671af03578e6a34edd8f0d1ac85e084df4` — please ensure your checked-out `vcpkg` submodule is at (or compatible with) that commit. If you update `vcpkg.json`'s `builtin-baseline`, also update the submodule reference and document the change in this README or `CONTRIBUTING.md` so contributors use the same vcpkg state.

### GPU Acceleration (Optional)

For optimal performance on NVIDIA GPUs:

- **CUDA Toolkit 12.x** - Required for any GPU acceleration
- **TensorRT 10.9.0.34** - Optional, provides best performance on RTX GPUs

The application automatically detects available GPU capabilities and uses the best option:

1. **TensorRT** (RTX GPUs with Tensor Cores) - Fastest, FP16 inference
2. **CUDA** (Any NVIDIA GPU) - Good performance on GTX and RTX
3. **CPU** - Fallback when no GPU available

## Quick Start

### Build Backend


```powershell
# Change to your local clone location
cd <path-to-BeatSyncEditor>

# Configure (first run installs dependencies)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --config Release --target beatsync_backend_shared
```

### Build with GPU Acceleration

```powershell
# Install TensorRT to C:\TensorRT-10.9.0.34

# Configure with overlay triplet
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake --overlay-triplets=triplets

# Build
cmake --build build --config Release --target beatsync_backend_shared
```

### Build TripSitter GUI

```powershell
# Set your Unreal Engine path (Windows example)
$Env:UE_ENGINE_PATH = 'C:\UE5_Source\UnrealEngine'

# Copy source to UE engine
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination "$Env:UE_ENGINE_PATH\Engine\Source\Programs\TripSitter\Private\" -Recurse -Force

# Build
& "$Env:UE_ENGINE_PATH\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
```

### Deploy DLLs to TripSitter

**CRITICAL**: TripSitter.exe requires all dependency DLLs in its directory. Use the deployment script:

```powershell
# Recommended: Use the deployment script (verifies DLL sizes)
.\scripts\deploy_tripsitter.ps1

# Verify DLLs are correct
.\scripts\deploy_tripsitter.ps1 -Verify
```

**WARNING**: Do NOT use `Copy-Item 'build\Release\*.dll'` as this copies wrong FFmpeg versions and causes crashes. The `build/Release/` folder contains vcpkg FFmpeg DLLs (~13MB) which are incompatible with the backend. ThirdParty FFmpeg DLLs (~106MB) must be used instead.

#### Required DLLs

| DLL | Size | Purpose |
|-----|------|---------|
| `beatsync_backend_shared.dll` | ~350KB | Main backend |
| `onnxruntime.dll` | ~14MB | ONNX Runtime v1.23.x |
| `onnxruntime_providers_shared.dll` | ~11KB | GPU provider interface |
| `onnxruntime_providers_cuda.dll` | ~335MB | CUDA execution provider |
| `abseil_dll.dll` | ~2MB | ONNX dependency |
| `libprotobuf.dll` | ~12MB | ONNX dependency |
| `libprotobuf-lite.dll` | ~1.5MB | ONNX dependency |
| `re2.dll` | ~1.2MB | ONNX dependency |
| `avcodec-62.dll` | ~106MB | FFmpeg codec |
| `avformat-62.dll` | ~22MB | FFmpeg format |
| `avutil-60.dll` | ~3MB | FFmpeg utils |
| `avfilter-11.dll` | ~89MB | FFmpeg filters |
| `avdevice-62.dll` | ~4MB | FFmpeg device |
| `swresample-6.dll` | ~700KB | FFmpeg resampling |
| `swscale-9.dll` | ~2MB | FFmpeg scaling |

#### AudioFlux DLLs (Optional - Spectral Flux Beat Detection)

These are only needed for Flux mode beat detection. The app falls back to energy mode without them.

| DLL | Size | Purpose |
|-----|------|---------|
| `audioflux.dll` | ~2MB | AudioFlux spectral analysis |
| `libfftw3f-3.dll` | ~1.5MB | FFTW dependency |

#### TensorRT DLLs (Optional - RTX Acceleration)

These are only needed for TensorRT acceleration on RTX GPUs. The app works without them.

| DLL | Size | Purpose |
|-----|------|---------|
| `nvinfer_10.dll` | ~420MB | TensorRT inference engine |
| `nvinfer_lean_10.dll` | ~42MB | Lightweight runtime |
| `nvinfer_plugin_10.dll` | ~49MB | TensorRT plugins |
| `nvinfer_dispatch_10.dll` | ~0.6MB | Provider dispatch |
| `nvonnxparser_10.dll` | ~2.9MB | ONNX model parser |

**Note**: `nvinfer_builder_resource_10.dll` (1.88 GB) is NOT needed at runtime.

## C API Overview

The backend exposes a C API for easy integration:

```c
// Initialize
bs_init();

// Audio analysis
void* analyzer = bs_create_audio_analyzer();
bs_beatgrid_t grid;
bs_analyze_audio(analyzer, "song.mp3", &grid);

// AI analysis (with GPU)
bs_ai_config_t config = { .beat_model_path = "beatnet.onnx" };
void* ai = bs_create_ai_analyzer(&config);
bs_ai_result_t result;
bs_ai_analyze_file(ai, "song.mp3", &result, progress_cb, NULL);

// Video processing
void* writer = bs_create_video_writer();
bs_video_cut_at_beats(writer, "video.mp4", grid.beats, grid.count, "output.mp4", 2.0);

// Cleanup
bs_free_beatgrid(&grid);
bs_free_ai_result(&result);
bs_destroy_audio_analyzer(analyzer);
bs_destroy_ai_analyzer(ai);
bs_destroy_video_writer(writer);
bs_shutdown();
```

## Project Structure

```text
BeatSyncEditor/
├── src/
│   ├── audio/           # Audio analysis and beat detection
│   ├── video/           # Video processing and effects
│   ├── backend/         # C API wrapper
│   └── tracing/         # OpenTelemetry support
├── tests/               # Catch2 unit tests
├── triplets/            # vcpkg overlay for TensorRT
├── unreal-prototype/    # UE5 TripSitter source
└── vcpkg/               # Package manager submodule
```

## Documentation

- [BUILD.md](BUILD.md) - Detailed build instructions
- [TODO.md](TODO.md) - Development roadmap
- [DEVELOPMENT_CONTEXT.md](DEVELOPMENT_CONTEXT.md) - Technical details
- [ROADMAP.md](ROADMAP.md) - Feature roadmap

## Current Status

### Completed

- Backend DLL with FFmpeg and ONNX Runtime
- CUDA + TensorRT GPU acceleration with automatic fallback
- TripSitter UE5 standalone app
- C API for all major functions
- Waveform visualization
- Beat-synced effects (flash, zoom) using chained FFmpeg filters
- NSIS installer with TensorRT DLL bundling
- AudioFlux spectral flux beat detection integration
- Stem separation with Demucs (via ONNX Runtime)

### In Progress

- AI beat detection model integration (BeatNet, TCN models)
- End-to-end effects testing

### Planned

- GLSL transition library
- Additional trained ONNX models

## Algorithm Details

### Beat Detection

The current implementation supports:

1. **Energy-based detection**: FFmpeg audio decoding, energy calculation, peak detection
2. **Spectral flux detection** (AudioFlux): Onset detection using spectral flux analysis
3. **AI-based detection**: ONNX models for neural network inference (BeatNet, All-In-One, TCN)

**Detection modes in TripSitter**:
- **Energy**: Basic energy-based beat detection (always available)
- **Flux**: Spectral flux onset detection (requires AudioFlux)
- **AI**: Neural network beat detection (requires ONNX Runtime)
- **Stems + Flux**: Demucs stem separation + flux on drums stem
- **Stems + AI**: Demucs stem separation + AI on drums stem

### Video Processing

All segments are normalized during extraction:

```text
scale=1920:1080:force_original_aspect_ratio=decrease
pad=1920:1080:(ow-iw)/2:(oh-ih)/2
setsar=1
fps=24
```

This ensures consistent properties for fast concatenation.

## Performance

- **Analysis speed**: ~10-20x real-time (CPU), faster with GPU
- **Memory usage**: ~40MB per minute of audio
- **GPU acceleration**: TensorRT provides 2-5x speedup over CPU

## License

[To be determined]

## Contributing

[To be determined]

---

Last updated: January 21, 2026
