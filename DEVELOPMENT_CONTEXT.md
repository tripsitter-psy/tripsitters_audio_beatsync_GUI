# BeatSyncEditor - Development Context

## Project Overview

**BeatSyncEditor** is a C++ desktop application for synchronizing video clips to audio beats. It features:

- **Backend**: C++ library with FFmpeg for video processing and ONNX Runtime for AI-powered beat detection
- **Frontend**: Unreal Engine 5 standalone program (TripSitter) with Slate UI
- **GPU Acceleration**: CUDA + TensorRT support for neural network inference

**Location**: `<project-root>`

---

## Architecture

```text
BeatSyncEditor/
├── src/
│   ├── audio/
│   │   ├── AudioAnalyzer.cpp/h        # FFmpeg audio loading & basic beat detection
│   │   ├── AudioFluxBeatDetector.cpp/h # AudioFlux spectral flux beat detection
│   │   ├── BeatGrid.cpp/h             # Beat timing data structure
│   │   ├── OnnxBeatDetector.cpp/h     # ONNX Runtime neural network inference
│   │   ├── OnnxMusicAnalyzer.cpp/h    # High-level AI music analysis
│   │   └── SpectralFlux.cpp/h         # Spectral analysis utilities
│   ├── video/
│   │   ├── VideoProcessor.cpp/h    # FFmpeg video reading & info
│   │   ├── VideoWriter.cpp/h       # Segment extraction, concat, audio mux
│   │   └── TransitionLibrary.cpp/h # Video transitions & effects
│   ├── backend/
│   │   ├── beatsync_capi.cpp/h     # C API wrapper for DLL export
│   │   └── tracing.cpp/h           # OpenTelemetry implementation & DLL exports (Internal)
│   └── tracing/
│       └── Tracing.h               # Tracing macros and utilities (Public API)
├── tests/                          # Catch2 unit tests
├── triplets/
│   └── x64-windows.cmake           # vcpkg overlay triplet for TensorRT
├── unreal-prototype/
│   └── Source/TripSitter/Private/  # UE5 standalone app source
└── vcpkg/                          # Package manager submodule
```

---

## Build System

### Backend (C++ DLL)

- **CMake** with vcpkg for dependencies
- **MSVC 2022** on Windows
- **Dependencies**: FFmpeg, ONNX Runtime (with CUDA/TensorRT)
- **Optional**: AudioFlux (for spectral flux beat detection)

```powershell
# Configure with TensorRT support
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets

# Configure with TensorRT + AudioFlux
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DAUDIOFLUX_ROOT="C:/audioFlux"

# Build
cmake --build build --config Release --target beatsync_backend_shared
```

### Frontend (Unreal Engine)

- **UE5 Source Build** at `$env:UE_ENGINE_PATH` (set this environment variable to your local UE5 source location; e.g., in PowerShell: `$env:UE_ENGINE_PATH = 'C:\Path\To\UnrealEngine'`, in bash: `export UE_ENGINE_PATH=/path/to/UnrealEngine`. If unset, a typical default might be `C:\UnrealEngine` or `/home/user/UnrealEngine`)
- **TripSitter Program Target** (standalone executable, not game)

```powershell


# Copy source to engine
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination "$env:UE_ENGINE_PATH\Engine\Source\Programs\TripSitter\Private\" -Recurse -Force

# Build
& "$env:UE_ENGINE_PATH\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
```

**Output**: `$env:UE_ENGINE_PATH\Engine\Binaries\Win64\TripSitter.exe`

---

## GPU Acceleration Setup

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- TensorRT 10.9.0.34

### TensorRT Installation

1. Download TensorRT from NVIDIA Developer
2. Extract to `C:\TensorRT-10.9.0.34`
3. The overlay triplet `triplets/x64-windows.cmake` sets `TENSORRT_HOME` automatically

### ONNX Runtime Configuration

The project uses ONNX Runtime 1.23.2 with:

- **CUDA Execution Provider** - GPU acceleration via CUDA
- **TensorRT Execution Provider** - Optimized inference via TensorRT

Available execution providers can be queried via `bs_ai_get_providers()`.

---

## Current Status (January 2026)

### Completed

- Backend DLL with ONNX Runtime + CUDA + TensorRT
- TripSitter UE5 standalone app builds successfully
- C API for audio analysis, video processing, AI inference
- vcpkg manifest with FFmpeg and ONNX Runtime
- Overlay triplet for TensorRT environment setup
- AudioFlux spectral flux beat detection integration
- Stem separation with Demucs (via ONNX Runtime)

### Recent Fixes (January 13, 2026)

1. **bs_ai_result_t redefinition** - Fixed struct tag name conflict in beatsync_capi.h
2. **std::numbers::pi** - Replaced C++20 constant with C++17-compatible `constexpr double PI`
3. **Missing brace** - Fixed syntax error in bs_ai_analyze_quick function
4. **IDesktopPlatform** - Fixed preprocessor condition for standalone builds (use native Windows dialogs)

### Pending

- Train/integrate ONNX beat detection models (BeatNet, All-In-One, TCN)
- End-to-end testing of effects pipeline
- Frame extraction testing in UE preview widget

---

## Key Files

### C API Header

`src/backend/beatsync_capi.h` - Defines the public DLL interface:

- `bs_beatgrid_t` - Simple beat grid structure
- `bs_ai_result_t` - Extended AI analysis result with downbeats and segments
- `bs_ai_config_t` - AI analyzer configuration
- `bs_effects_config_t` - Video effects configuration

### ONNX Beat Detector

`src/audio/OnnxBeatDetector.cpp` - Neural network inference:

- Mel spectrogram extraction
- In-place FFT implementation
- CUDA/TensorRT execution provider setup
- Model loading and inference

### UE Widget

`unreal-prototype/Source/TripSitter/Private/STripSitterMainWidget.cpp`:

- Slate UI with file selection
- Waveform visualization
- Progress tracking
- Native Windows file dialogs (standalone builds)

---

## Quick Reference

```powershell
# Full build sequence (with TensorRT + AudioFlux)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DAUDIOFLUX_ROOT="C:/audioFlux"
cmake --build build --config Release --target beatsync_backend_shared
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'unreal-prototype\ThirdParty\beatsync\lib\x64\' -Force

Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination "$env:UE_ENGINE_PATH\Engine\Source\Programs\TripSitter\Private\" -Recurse -Force
& "$env:UE_ENGINE_PATH\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development

# Run TripSitter
& "$env:UE_ENGINE_PATH\Engine\Binaries\Win64\TripSitter.exe"

# Run tests
cmake --build build --config Release --target test_backend_api
./build/tests/Release/test_backend_api.exe
```

---

## Technical Notes

### FFmpeg Filter Chain

```text
scale=1920:1080:force_original_aspect_ratio=decrease  # Fit within bounds
pad=1920:1080:(ow-iw)/2:(oh-ih)/2                     # Letterbox/pillarbox
setsar=1                                               # Square pixels
fps=24                                                 # Consistent framerate
```

### ONNX Runtime Execution Providers

Priority order for GPU acceleration:

1. **TensorRT** - Fastest, requires TensorRT installation
2. **CUDA** - Good performance, requires CUDA toolkit
3. **CPU** - Fallback, always available

### Pi Constant (C++17 Compatible)

```cpp
// In OnnxBeatDetector.cpp
constexpr double PI = 3.14159265358979323846;
```

Used instead of `std::numbers::pi` for C++17 compatibility.

---

*Last updated: January 21, 2026*
