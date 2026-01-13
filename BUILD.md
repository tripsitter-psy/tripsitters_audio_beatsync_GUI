# Build Instructions

## Prerequisites

### Windows

1. **Visual Studio 2022** (Build Tools or Community Edition)
   - Include "Desktop development with C++"
   - C++ CMake tools for Windows

2. **CMake 3.20+**
   - Usually included with Visual Studio
   - Or download from <https://cmake.org/download/>

3. **vcpkg** (included as submodule)
   - Already configured in this repository
   - Uses manifest mode (`vcpkg.json`)

4. **NVIDIA GPU Support** (optional, for AI acceleration)
   - CUDA Toolkit 12.x
   - TensorRT 10.9.0.34

5. **Unreal Engine 5** (for TripSitter GUI)
   - Source build at `C:\UE5_Source\UnrealEngine`

## Quick Start

### Backend DLL Only

```powershell
# Navigate to project
cd C:\Users\samue\Desktop\BeatSyncEditor

# Configure with vcpkg (first run installs dependencies, ~30-60 min)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --config Release --target beatsync_backend_shared
```

Output: `build/Release/beatsync_backend_shared.dll`

### With GPU Acceleration (CUDA + TensorRT)

```powershell
# Install TensorRT to C:\TensorRT-10.9.0.34

# Configure with overlay triplet (sets TENSORRT_HOME)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake --overlay-triplets=triplets

# Build (first run with TensorRT takes ~2 hours for ONNX Runtime)
cmake --build build --config Release --target beatsync_backend_shared
```

### TripSitter GUI (Unreal Engine)

```powershell
# Copy DLL to ThirdParty
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'unreal-prototype\ThirdParty\beatsync\lib\x64\' -Force

# Copy source files to engine
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination 'C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\Private\' -Recurse -Force

# Build TripSitter
& "C:\UE5_Source\UnrealEngine\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
```

Output: `C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe`

## Dependencies

### Managed by vcpkg (automatic)

Defined in `vcpkg.json`:

- **FFmpeg** - avcodec, avformat, swresample, swscale, avfilter
- **ONNX Runtime** - Neural network inference (with optional CUDA/TensorRT)

### External (manual installation)

- **TensorRT 10.9.0.34** - For GPU-accelerated inference
  - Download from NVIDIA Developer
  - Extract to `C:\TensorRT-10.9.0.34`
  - The overlay triplet handles environment setup

- **CUDA Toolkit 12.x** - Required for GPU acceleration
  - Download from NVIDIA Developer
  - Install to default location

## Build Configurations

### Debug

```powershell
cmake --build build --config Debug
```

- Debug symbols included
- No optimizations
- Easier debugging

### Release

```powershell
cmake --build build --config Release
```

- Optimizations enabled
- Faster execution
- Recommended for normal use

### RelWithDebInfo

```powershell
cmake --build build --config RelWithDebInfo
```

- Optimizations + debug symbols
- Good for profiling

## Build Targets

| Target | Description |
|--------|-------------|
| `beatsync_backend_shared` | Main DLL for Unreal plugin |
| `beatsync_backend_static` | Static lib for tests |
| `test_backend_api` | C API unit tests |

## Testing

```powershell
# Build and run tests
cmake --build build --config Release --target test_backend_api
./build/tests/Release/test_backend_api.exe
```

## Troubleshooting

### vcpkg Dependencies Not Installing

Ensure you're using the toolchain file:

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
```

### TensorRT Not Found

Use the overlay triplet:

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake --overlay-triplets=triplets
```

The triplet at `triplets/x64-windows.cmake` sets `TENSORRT_HOME`.

### ONNX Runtime Build Takes Too Long

First build with CUDA/TensorRT takes ~2 hours. Subsequent builds are fast.

To build without GPU support:

```powershell
# Edit vcpkg.json to remove cuda/tensorrt features, then:
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Missing DLLs at Runtime

Copy required DLLs to executable directory:

```powershell
# FFmpeg DLLs
Copy-Item 'build\vcpkg_installed\x64-windows\bin\*.dll' 'build\Release\' -Force

# ONNX Runtime DLLs
Copy-Item 'build\vcpkg_installed\x64-windows\bin\onnxruntime*.dll' 'build\Release\' -Force
```

### TripSitter Won't Compile

Ensure source files are copied to the engine:

```powershell
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination 'C:\UE5_Source\UnrealEngine\Engine\Source\Programs\TripSitter\Private\' -Recurse -Force
```

### std::numbers::pi Error

This was fixed by using `constexpr double PI` instead. If you see this error, ensure you have the latest code.

### bs_ai_result_t Redefinition Error

This was fixed in beatsync_capi.h. Ensure you have the latest code.

## IDE Integration

### Visual Studio 2022

1. Open Visual Studio 2022
2. File > Open > CMake...
3. Select `CMakeLists.txt` from project root
4. Configure vcpkg integration in CMakeSettings.json

### Visual Studio Code

1. Install extensions: C/C++, CMake Tools
2. Open folder in VS Code
3. Select kit (MSVC 2022)
4. Configure and build via CMake Tools

## Clean Build

```powershell
# Remove build directory
Remove-Item -Recurse -Force build

# Reconfigure and build
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake --overlay-triplets=triplets
cmake --build build --config Release
```

## Performance Notes

### Compile Time

- First build with vcpkg: 30-60 minutes (FFmpeg)
- First build with TensorRT: ~2 hours (ONNX Runtime)
- Subsequent builds: 1-3 minutes
- Incremental builds: less than 30 seconds

### Build Size

- FFmpeg libraries: ~100-200 MB
- ONNX Runtime: ~50-100 MB
- Project executable: ~2-5 MB (Release)
- Total with dependencies: ~300-400 MB

---

Last updated: January 14, 2026
