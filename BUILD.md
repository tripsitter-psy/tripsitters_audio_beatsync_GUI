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
   - Source build at `D:\UnrealEngine`

## Quick Start

### Initial Setup

Before starting, initialize the vcpkg submodule:

```powershell
git submodule update --init --recursive
```

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

# Configure with overlay triplet (sets TENSORRT_HOME) and GPU vcpkg feature
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DVCPKG_MANIFEST_FEATURES="gpu"

# Build (first run with TensorRT takes ~2 hours for ONNX Runtime)
cmake --build build --config Release --target beatsync_backend_shared
```

### With AudioFlux (Spectral Flux Beat Detection)

```powershell
# Install AudioFlux to C:\audioFlux

# Configure with AudioFlux support
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DAUDIOFLUX_ROOT="C:/audioFlux"

# Or combine with GPU acceleration
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets -DVCPKG_MANIFEST_FEATURES="gpu" -DAUDIOFLUX_ROOT="C:/audioFlux"

# Build
cmake --build build --config Release --target beatsync_backend_shared
```

**Note**: AudioFlux enables the "Flux" beat detection mode in TripSitter. Without it, the app falls back to energy-based detection.


### TripSitter GUI (Unreal Engine)

> **Before building the TripSitter GUI:**
> - **PowerShell:** `$env:UE_ENGINE_PATH = 'D:\UnrealEngine'`
> - **cmd.exe:** `set UE_ENGINE_PATH=D:\UnrealEngine`
> - **bash:** `export UE_ENGINE_PATH=/mnt/d/UnrealEngine`



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

**CRITICAL**: TripSitter.exe requires all dependency DLLs in its directory with correct file sizes.

Target location: `D:\UnrealEngine\Engine\Binaries\Win64\`

The TripSitter executable cannot run without:
- **beatsync_backend_shared.dll** (300KB+) — Backend library
- **FFmpeg DLLs** (~400MB total) — Audio/video processing
- **ONNX Runtime DLLs** (~15MB+) — AI neural network inference
- **ONNX provider DLLs** — GPU acceleration (CUDA, TensorRT)

**Deploy the DLLs:**

```powershell
# Recommended: Use the deployment script
.\scripts\deploy_tripsitter.ps1

# Verify DLL deployment and sizes
.\scripts\deploy_tripsitter.ps1 -Verify

# Or check DLLs manually
.\check_dlls.ps1
```

See [DLL_VERIFICATION_GUIDE.md](DLL_VERIFICATION_GUIDE.md) for detailed troubleshooting if verification fails.

**If deployment fails**, ensure dependencies are built:
```powershell
cmake --build build --config Release --target beatsync_backend_shared
```

**IMPORTANT**: The build is configured to strictly separate project artifacts from dependencies. Dependency DLLs (FFmpeg, ONNX Runtime) are **NOT** copied to `build/Release/` to prevent mixing incompatible versions.
A build-time validation step ensures `build/Release` contains only project binaries.
Always use the deployment script to collect DLLs from their dedicated directories (`vcpkg_installed` and `ThirdParty`).

## Dependencies

### Managed by vcpkg (automatic)

Defined in `vcpkg.json`:

- **libsamplerate** - High-quality audio resampling (always installed as a top-level dependency)

The following dependencies are installed via vcpkg **features** (not top-level):

- **FFmpeg** (avcodec, avformat, swresample, swscale, avfilter) - Installed via the `cpu` feature (default) or `gpu` feature (adds nvcodec)
- **ONNX Runtime** - Neural network inference; installed via `cpu` feature (CPU-only) or `gpu` feature (with CUDA/TensorRT support)

To select a feature during CMake configure:

- CPU (default): No extra flags needed, or `-DVCPKG_MANIFEST_FEATURES="cpu"`
- GPU: `-DVCPKG_MANIFEST_FEATURES="gpu"`

### External (manual installation)

- **TensorRT 10.9.0.34** - For GPU-accelerated inference (RTX GPUs)
  - Download from NVIDIA Developer
  - Extract to `C:\TensorRT-10.9.0.34`
  - The overlay triplet handles environment setup
  - Runtime DLLs (~515 MB) are staged during deployment only (NOT copied to `build/Release/`)
  - Per the separation policy, TensorRT DLLs are copied to the UE5 Binaries directory or installer staging area at deployment time via `scripts/deploy_tripsitter.ps1`, keeping `build/Release/` clean for project-only binaries

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

## Linux Build (Backend + CLI)

The backend library and `beatsync` CLI build natively on Linux (tested on Fedora 44, GCC 16). The Unreal Engine GUI is not ported yet.

### Prerequisites

```bash
# Fedora (FFmpeg from RPM Fusion)
sudo dnf install cmake ninja-build gcc-c++ ffmpeg-devel libsamplerate-devel
```

ONNX Runtime is not taken from the distro; download the official GPU build:

```bash
mkdir -p thirdparty && cd thirdparty
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz
tar xzf onnxruntime-linux-x64-gpu-1.23.2.tgz
# The bundled CMake config expects these two paths to exist:
ln -s lib onnxruntime-linux-x64-gpu-1.23.2/lib64
mkdir -p onnxruntime-linux-x64-gpu-1.23.2/include/onnxruntime
(cd onnxruntime-linux-x64-gpu-1.23.2/include/onnxruntime && for f in ../*.h; do ln -sf "$f" .; done)
```

Optional, for the Flux / Stems+Flux analysis modes, build AudioFlux (on Linux it uses its
built-in FFT — no FFTW, so no GPL implications):

```bash
cd thirdparty
git clone --depth 1 https://github.com/libaudioflux/audioFlux.git
# Its CMakeLists hardcodes clang and "omp"; with GCC change those to the default
# compiler and "gomp", then:
cmake -S audioFlux/src -B audioFlux/build/linuxBuild -DCMAKE_SYSTEM_NAME=linux -DCMAKE_BUILD_TYPE=Release
cmake --build audioFlux/build/linuxBuild -j$(nproc)
```

### Configure and Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$PWD/thirdparty/onnxruntime-linux-x64-gpu-1.23.2 \
    -DAUDIOFLUX_ROOT=$PWD/thirdparty/audioFlux
cmake --build build
ctest --test-dir build
```

Outputs: `build/libbeatsync_backend_shared.so` and `build/bin/Release/beatsync`.

### GPU Acceleration (CUDA)

ONNX Runtime's CUDA execution provider (`libonnxruntime_providers_cuda.so`) is dlopen'd at
runtime and ships without a RUNPATH, so cuBLAS/cuDNN/cuFFT/cuRAND must be resolvable — via
ldconfig (system install), `LD_LIBRARY_PATH`, or by stamping a RUNPATH into the provider:

```bash
patchelf --set-rpath '$ORIGIN:/path/to/cuda/lib64' \
    thirdparty/onnxruntime-linux-x64-gpu-1.23.2/lib/libonnxruntime_providers_cuda.so
```

Alternatively pass `-DBEATSYNC_CUDA_LIB_DIRS="/path/to/cuda/lib64;/path/to/cudnn/lib"` at
configure time to bake a DT_RPATH into the CLI and shared library. Without any of this the
app silently falls back to CPU (the TensorRT → CUDA → CPU chain still works).

## Performance Notes

### Compile Time

- First build with vcpkg: 30-60 minutes (FFmpeg)
- First build with TensorRT: ~2 hours (ONNX Runtime)
- Subsequent builds: 1-3 minutes
- Incremental builds: less than 30 seconds

### Build Size

- FFmpeg libraries: ~100-200 MB
- ONNX Runtime: ~50-100 MB
- TensorRT runtime DLLs: ~515 MB (optional)
- AudioFlux runtime: ~40 MB (optional)
- Project executable: ~2-5 MB (Release)
- Total with minimal dependencies: ~300-400 MB
- Total with full GPU support: ~900 MB - 1 GB

---

Last updated: May 5, 2026
