# Build Instructions

## Platform-Specific Guides

This project supports multiple platforms. For detailed platform-specific instructions:

- **[macOS Build Guide](BUILD_MACOS.md)** - Comprehensive macOS build instructions (Homebrew & vcpkg)
- **Windows** - See below for Windows build instructions

---

## Prerequisites

### Windows

1. **Visual Studio 2022** (Build Tools or Community Edition)
   - Include "Desktop development with C++"
   - C++ CMake tools for Windows

2. **CMake 3.20+**
   - Usually included with Visual Studio
   - Or download from https://cmake.org/download/

3. **vcpkg** (Package Manager)
   - Clone: `git clone https://github.com/Microsoft/vcpkg.git`
   - Bootstrap: `cd vcpkg && bootstrap-vcpkg.bat`
   - Add to PATH or note the installation directory

## Quick Start (vcpkg method - Recommended)

### Step 1: Clone/Download Project

```bash
# If from git
git clone [repository-url]
cd BeatSyncEditor

# Or navigate to the project directory
cd C:\Users\samue\Desktop\BeatSyncEditor
```

### Step 2: Configure with vcpkg

The project includes a `vcpkg.json` manifest that automatically installs dependencies.

```bash
# Create build directory
mkdir build
cd build

# Configure with vcpkg toolchain
cmake .. -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake

# For example:
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

This will automatically:
- Install FFmpeg (libavcodec, libavformat, libavutil, libswresample)
- Configure the project
- May take 10-30 minutes on first run

### Step 3: Build

```bash
# Build Release configuration
cmake --build . --config Release

# Or Debug
cmake --build . --config Debug
```

### Step 4: Run

```bash
cd bin\Release
beatsync.exe analyze path\to\audio.mp3
```

## Alternative: Manual FFmpeg Installation

If you prefer not to use vcpkg or already have FFmpeg installed:

### Step 1: Install FFmpeg

Download pre-built FFmpeg libraries from:
- https://www.gyan.dev/ffmpeg/builds/ (Windows builds)
- Extract to a known location (e.g., `C:\ffmpeg`)

### Step 2: Configure with FFmpeg Path

```bash
mkdir build
cd build

cmake .. -DFFMPEG_DIR="C:/ffmpeg"
```

### Step 3: Build

```bash
cmake --build . --config Release
```

### Step 4: Copy DLLs

Copy required DLLs to the executable directory:
- avcodec-XX.dll
- avformat-XX.dll
- avutil-XX.dll
- swresample-XX.dll

```bash
copy C:\ffmpeg\bin\*.dll bin\Release\
```

## Troubleshooting

### FFmpeg Not Found

**Issue**: `CMake Warning: FFmpeg not found`

**Solutions**:
1. Use vcpkg toolchain (recommended)
2. Set `FFMPEG_DIR` to your FFmpeg installation:
   ```bash
   cmake .. -DFFMPEG_DIR="C:/path/to/ffmpeg"
   ```
3. Install via vcpkg manually:
   ```bash
   vcpkg install ffmpeg:x64-windows
   ```

### OpenCV Not Found (Warning)

This is normal for Phase 1. OpenCV is optional and only needed for Phase 2 (video processing).

To install OpenCV for future phases:
```bash
vcpkg install opencv:x64-windows
```

### Build Errors with MSVC

**Issue**: Compiler errors or warnings

**Solutions**:
1. Ensure Visual Studio 2022 with C++ tools is installed
2. Open "Developer Command Prompt for VS 2022"
3. Re-run cmake and build from that prompt

### Missing DLLs at Runtime

**Issue**: `The program can't start because avcodec-XX.dll is missing`

**Solutions**:

With vcpkg:
```bash
# DLLs are automatically copied to bin directory
# If not, they're in: [vcpkg-root]/installed/x64-windows/bin/
```

With manual FFmpeg:
```bash
# Copy DLLs from FFmpeg bin directory to executable directory
copy "C:\ffmpeg\bin\*.dll" "build\bin\Release\"
```

### vcpkg Manifest Error

**Issue**: `Could not locate a manifest (vcpkg.json)`

**Solution**: Ensure `vcpkg.json` exists in project root:
```json
{
  "name": "beatsynceditor",
  "version": "1.0.0",
  "dependencies": [
    {
      "name": "ffmpeg",
      "default-features": false,
      "features": ["avcodec", "avformat", "avutil", "swresample"]
    }
  ],
  "builtin-baseline": "2024-12-09"
}
```

## Build Configurations

### Debug Build
- Includes debug symbols
- No optimizations
- Easier debugging
```bash
cmake --build . --config Debug
```

### Release Build
- Optimizations enabled
- Faster execution
- Recommended for normal use
```bash
cmake --build . --config Release
```

### RelWithDebInfo
- Optimizations + debug symbols
- Good for profiling
```bash
cmake --build . --config RelWithDebInfo
```

## Testing the Build

After building, test with a sample audio file:

```bash
cd bin\Release

# Download a test audio file or use your own
beatsync.exe analyze test.mp3

# Try different sensitivity
beatsync.exe analyze test.mp3 --sensitivity 0.7
```

Expected output:
```
========================================
BeatSync Audio Analyzer
========================================

Analyzing: test.mp3
Sensitivity: 0.5

Audio loaded: 120.5s, 44100 Hz, 5308800 samples
Detected 245 beats

========================================
Analysis Results
========================================

BeatGrid Information:
  Number of beats: 245
  BPM: 122.3
  ...
```

## Clean Build

To start fresh:

```bash
# Remove build directory
rm -r build

# Or on Windows
rmdir /s build

# Then rebuild from step 2
```

## IDE Integration

### Visual Studio 2022

VS 2022 has native CMake support:

1. Open Visual Studio 2022
2. File > Open > CMake...
3. Select `CMakeLists.txt` from project root
4. VS will automatically configure
5. Build using the toolbar

Make sure to configure vcpkg integration:
- Tools > Options > CMake > General
- Set CMake toolchain file to vcpkg's script

### Visual Studio Code

1. Install extensions:
   - C/C++
   - CMake Tools
2. Open folder in VS Code
3. Select kit (MSVC 2022)
4. Configure and build via CMake Tools

## Performance Notes

### Compile Time
- First build with vcpkg: 15-45 minutes (building FFmpeg)
- Subsequent builds: 1-3 minutes
- Incremental builds: <30 seconds

### Build Size
- FFmpeg libraries: ~100-200 MB
- Project executable: ~2-5 MB (Release)
- Total with dependencies: ~200-300 MB

## Next Steps

After successful build:
1. Test with various audio files (MP3, WAV, FLAC)
2. Experiment with sensitivity settings
3. Move to Phase 2 development (video processing)

For Phase 2, you'll need to install OpenCV:
```bash
# In vcpkg.json, add:
"opencv"

# Then reconfigure
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg-path]/scripts/buildsystems/vcpkg.cmake
```
