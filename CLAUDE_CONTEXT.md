# BeatSyncEditor - Quick Context for Claude

## What This Project Is

A beat-synced video editor with:

- **Backend**: C++ DLL (FFmpeg + ONNX Runtime with CUDA/TensorRT)
- **Frontend**: Unreal Engine 5 standalone app (TripSitter)

## Key Locations

| Item | Path |
| ---- | ---- |
| Project Root | Current working directory |
| Backend DLL | `build\Release\beatsync_backend_shared.dll` |
| UE Source | `C:\UE5_Source\UnrealEngine` |
| TripSitter EXE | `C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe` |
| TensorRT | `C:\TensorRT-10.9.0.34` (set via `$env:TENSORRT_HOME`) |
| AudioFlux | `C:\audioFlux` (set via `-DAUDIOFLUX_ROOT` CMake flag) |

> **Note:** UE Source and TensorRT paths are hardcoded infrastructure paths. User-specific paths (home, Documents) should use `$env:USERPROFILE`.

## Build Commands

```powershell
# Backend (with TensorRT + AudioFlux)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DAUDIOFLUX_ROOT="C:/audioFlux"
cmake --build build --config Release --target beatsync_backend_shared

# TripSitter (UE_SOURCE path from Key Locations table above)
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination 'C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\Private\' -Recurse -Force
& "C:\UE5_Source\UnrealEngine\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
```

## Recent Fixes (January 13, 2026)

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
- ~~NSIS installer packaging~~ — Template added (`installer/nsis_template.nsi.in`); finalize and test installer generation

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
