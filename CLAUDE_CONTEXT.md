# BeatSyncEditor - Quick Context for Claude

## What This Project Is

A beat-synced video editor with:

- **Backend**: C++ DLL (FFmpeg + ONNX Runtime with CUDA/TensorRT)
- **Frontend**: Unreal Engine 5 standalone app (TripSitter)

## Key Locations

| Item | Path |
| ---- | ---- |
| Project Root | `<PROJECT_ROOT>` |
| Backend DLL | `<BACKEND_DLL_PATH>` |
| UE Source | `<UE_SOURCE>` |
| TripSitter EXE | `<TRIPSITTER_EXE>` |
| TensorRT | `<TENSORRT_PATH>` |

> **Note:** Set the following environment variables to configure your local paths:
> - PROJECT_ROOT
> - BACKEND_DLL_PATH
> - UE_SOURCE
> - TRIPSITTER_EXE
> - TENSORRT_PATH

## Build Commands

```powershell
# Backend (with TensorRT)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake --overlay-triplets=triplets
cmake --build build --config Release --target beatsync_backend_shared

# TripSitter
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination 'C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\Private\' -Recurse -Force
& "C:\UE5_Source\UnrealEngine\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
```

## Recent Fixes (January 14, 2026)

1. **bs_ai_result_t redefinition** - Fixed in beatsync_capi.h (wrong struct tag name)
2. **std::numbers::pi** - Replaced with `constexpr double PI` for C++17
3. **Missing brace** - Fixed in bs_ai_analyze_quick function
4. **IDesktopPlatform** - Changed `#if WITH_EDITOR || PLATFORM_DESKTOP` to `#if WITH_EDITOR`

## Key Files

| File | Purpose |
| ---- | ------- |
| `src/backend/beatsync_capi.h` | C API definitions |
| `src/audio/OnnxBeatDetector.cpp` | ONNX inference |
| `unreal-prototype/Source/TripSitter/Private/STripSitterMainWidget.cpp` | Main UI |
| `triplets/x64-windows.cmake` | TensorRT environment setup |
| `vcpkg.json` | Dependencies (FFmpeg, ONNX Runtime) |

## Current State

- Backend DLL: Builds successfully with ONNX Runtime + CUDA + TensorRT
- TripSitter: Builds successfully (some deprecation warnings)
- Tests: `test_backend_api` available

## Pending Work

- Train/integrate ONNX beat detection models
- End-to-end testing with real media
- NSIS installer packaging

## vcpkg Configuration

```json
{
  "dependencies": [
    { "name": "ffmpeg", "features": ["avcodec", "avformat", "swresample", "swscale", "avfilter"] },
    { "name": "onnxruntime", "platform": "windows" }
  ]
}
```

The overlay triplet `triplets/x64-windows.cmake` sets `TENSORRT_HOME` for GPU acceleration.

---

See [CLAUDE.md](CLAUDE.md) for full instructions.
