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
cd path\to\BeatSyncEditor

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
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets

# Build (first run with TensorRT takes ~2 hours for ONNX Runtime)
cmake --build build --config Release --target beatsync_backend_shared
```

### With AudioFlux (Spectral Flux Beat Detection)

```powershell
# Install AudioFlux to C:\audioFlux

# Configure with AudioFlux support
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DAUDIOFLUX_ROOT="C:/audioFlux"

# Or combine with GPU acceleration
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DAUDIOFLUX_ROOT="C:/audioFlux"

# Build
cmake --build build --config Release --target beatsync_backend_shared
```

**Note**: AudioFlux enables the "Flux" beat detection mode in TripSitter. Without it, the app falls back to energy-based detection.


### TripSitter GUI (Unreal Engine)

> **Before building the TripSitter GUI:**
> - **PowerShell:** `$env:UE_ENGINE_PATH = 'C:\UE5_Source\UnrealEngine'`
> - **cmd.exe:** `set UE_ENGINE_PATH=C:\UE5_Source\UnrealEngine`
> - **bash:** `export UE_ENGINE_PATH=/mnt/c/UE5_Source/UnrealEngine`



```powershell
# Copy DLL to ThirdParty
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'unreal-prototype\ThirdParty\beatsync\lib\x64\' -Force

# Copy source files to engine
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination "$env:UE_ENGINE_PATH\Engine\Source\Programs\TripSitter\Private\" -Recurse -Force

# Build TripSitter
& "$env:UE_ENGINE_PATH\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
```

Output: `$env:UE_ENGINE_PATH\Engine\Binaries\Win64\TripSitter.exe`

### Deploy All DLLs to TripSitter

**CRITICAL**: TripSitter.exe requires all dependency DLLs in its directory.

```powershell
# Recommended: Use the deployment script
.\scripts\deploy_tripsitter.ps1

# Or verify existing deployment
.\scripts\deploy_tripsitter.ps1 -Verify
```

**IMPORTANT**: The build is configured to strictly separate project artifacts from dependencies. Dependency DLLs (FFmpeg, ONNX Runtime) are **NOT** copied to `build/Release/` to prevent mixing incompatible versions.
A build-time validation step ensures `build/Release` contains only project binaries.
Always use the deployment script to collect DLLs from their dedicated directories (`vcpkg_installed` and `ThirdParty`).

## Dependencies

### Managed by vcpkg (automatic)

Defined in `vcpkg.json`:

- **FFmpeg** - avcodec, avformat, swresample, swscale, avfilter
- **ONNX Runtime** - Neural network inference (with optional CUDA/TensorRT)

### External (manual installation)

- **TensorRT 10.9.0.34** - For GPU-accelerated inference (RTX GPUs)
  - Download from NVIDIA Developer
  - Extract to `C:\TensorRT-10.9.0.34`
  - The overlay triplet handles environment setup
  - Runtime DLLs (~515 MB) are copied automatically during build

- **CUDA Toolkit 12.x** - Required for GPU acceleration
  - Download from NVIDIA Developer
  - Install to default location

- **AudioFlux** (optional) - For spectral flux beat detection
  - Build or download to `C:\audioFlux`
  - Headers: `C:\audioFlux\include\`
  - Library: `C:\audioFlux\build\windowBuild\Release\audioflux.lib`
  - Runtime DLLs: `audioflux.dll`, `libfftw3f-3.dll`

### GPU Execution Provider Fallback

The application automatically selects the best GPU provider:

1. **TensorRT** - Best performance on RTX GPUs with Tensor Cores (FP16 enabled)
2. **CUDA** - Good performance on any NVIDIA GPU (GTX or RTX)
3. **CPU** - Fallback when no GPU available

No configuration needed - detection is automatic at runtime.

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
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets
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

Last updated: January 21, 2026
