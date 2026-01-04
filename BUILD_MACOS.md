# macOS Build Instructions

This guide provides comprehensive instructions for building TripSitter BeatSync on macOS, covering both Homebrew and vcpkg dependency management approaches.

## Prerequisites

- **macOS 11.0 (Big Sur) or later**
- **Xcode Command Line Tools**: `xcode-select --install`
- **CMake 3.15+**: Install via Homebrew or from https://cmake.org/download/

## Quick Start (Homebrew - Recommended for macOS)

### Step 1: Install Dependencies via Homebrew

```bash
# Install FFmpeg, CMake, Ninja, and wxWidgets
brew install ffmpeg cmake ninja wxwidgets
```

### Step 2: Configure the Build (arm64 architecture, Release)

```bash
# Navigate to project directory
cd /path/to/tripsitters_audio_beatsync_GUI

# Configure for arm64 (Apple Silicon) Release build
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DFFMPEG_ROOT=$(brew --prefix ffmpeg) \
  -DwxWidgets_ROOT_DIR=$(brew --prefix wxwidgets)
```

**For Intel Macs (x86_64):**
```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DFFMPEG_ROOT=$(brew --prefix ffmpeg) \
  -DwxWidgets_ROOT_DIR=$(brew --prefix wxwidgets)
```

**For Universal Binary (both arm64 and x86_64):**
```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DFFMPEG_ROOT=$(brew --prefix ffmpeg) \
  -DwxWidgets_ROOT_DIR=$(brew --prefix wxwidgets)
```

### Step 3: Build

```bash
cmake --build build --config Release
```

Build output will be in:
- CLI: `build/bin/Release/beatsync`
- GUI: `build/bin/Release/TripSitter.app`

### Step 4: Run

**CLI:**
```bash
./build/bin/Release/beatsync --help
```

**GUI:**
```bash
./build/bin/Release/TripSitter.app/Contents/MacOS/TripSitter
```

Or double-click `build/bin/Release/TripSitter.app` in Finder.

## Alternative: vcpkg Method

If you prefer vcpkg for dependency management:

### Step 1: Install vcpkg

```bash
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
./bootstrap-vcpkg.sh

# Add vcpkg to PATH (optional)
export PATH="$(pwd):$PATH"

# Return to project directory
cd /path/to/tripsitters_audio_beatsync_GUI
```

### Step 2: Install Dependencies via vcpkg

For arm64 (Apple Silicon):
```bash
/path/to/vcpkg/vcpkg install ffmpeg:arm64-osx wxwidgets:arm64-osx
```

For x86_64 (Intel):
```bash
/path/to/vcpkg/vcpkg install ffmpeg:x64-osx wxwidgets:x64-osx
```

### Step 3: Configure with vcpkg

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

Replace `/path/to/vcpkg` with the actual path to your vcpkg installation.

### Step 4: Build and Run

Same as Homebrew method (Steps 3-4 above).

## Asset Verification

After building, verify that required assets are copied correctly:

```bash
# For GUI app bundle (macOS)
ls -la build/bin/Release/TripSitter.app/Contents/Resources/assets/

# You should see:
# - alpha_2.png (top header transparency asset)
# - alpha.png (button transparency asset)
# - background.png (main background)
# - ComfyUI_03324_.png (upscaled background, optional)
# - Other asset files
```

If assets are missing, CMake should have copied them from the `assets/` directory in the project root during the build. If they're still missing, manually copy:

```bash
cp -r assets/* build/bin/Release/TripSitter.app/Contents/Resources/assets/
```

## Font Installation

The GUI uses the **Corpta** font family. Install it system-wide:

1. Download or locate the Corpta font files (.ttf or .otf)
2. Install via Font Book:
   - Open **Font Book** (Applications > Font Book)
   - Click **File > Add Fonts...**
   - Select the Corpta font files
   - Click **Open**

3. Verify installation:
   ```bash
   fc-list | grep -i corpta
   ```
   
   Or check in Font Book that Corpta appears in the font list.

If the font doesn't render in the GUI:
- Restart the TripSitter application
- Verify the font is enabled in Font Book
- Check that the font name matches what's used in the code (check `src/GUI/PsychedelicTheme.cpp`)

## Packaging as DMG

Create a distributable DMG (Drag-and-Drop installer):

### Step 1: Build Release

```bash
cmake --build build --config Release
```

### Step 2: Create DMG Package

```bash
pushd build
cpack -C Release
popd
```

This generates:
- `build/TripSitter-<version>-Darwin-<arch>.dmg` - Drag-and-Drop DMG installer
- `build/TripSitter-<version>-Darwin-<arch>.zip` - ZIP archive

The DMG file contains the `TripSitter.app` bundle and can be distributed to users. Users can drag the app to their Applications folder.

### DMG Configuration

The DMG packaging is configured in `CMakeLists.txt` with:
- **CPACK_GENERATOR**: DragNDrop + ZIP
- **CPACK_DMG_VOLUME_NAME**: TripSitter
- **CPACK_DMG_FORMAT**: UDZO (compressed)

## Smoke Tests

### CLI Smoke Test

Test the command-line interface:

```bash
./build/bin/Release/beatsync --help
```

Expected output should show usage information and available commands.

### GUI Smoke Test

1. **Launch the GUI:**
   ```bash
   ./build/bin/Release/TripSitter.app/Contents/MacOS/TripSitter
   ```
   
   Or open via Finder.

2. **Verify visual elements:**
   - Check that the **header transparency** (alpha_2.png) renders correctly at the top
   - Verify **button artwork** (alpha.png) displays with proper transparency
   - Ensure the **background image** appears without distortion

3. **Load sample media:**
   - Load a sample audio file (MP3, WAV, FLAC)
   - Load a sample video file (MP4, MOV, AVI)
   - Verify no crashes occur during file loading

4. **Check basic functionality:**
   - Ensure beat detection works on the audio file
   - Verify the waveform/beat visualizer displays correctly
   - Test basic UI interactions (buttons, sliders, etc.)

## Troubleshooting

### FFmpeg Not Found

**Issue**: `CMake Error: Could not find FFmpeg`

**Solution**:
1. Verify FFmpeg is installed:
   ```bash
   brew list ffmpeg
   # or for vcpkg:
   /path/to/vcpkg/vcpkg list | grep ffmpeg
   ```

2. Explicitly pass FFMPEG_ROOT:
   ```bash
   cmake -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DFFMPEG_ROOT=$(brew --prefix ffmpeg)
   ```

3. For vcpkg, ensure the toolchain file is specified:
   ```bash
   cmake -B build -G Ninja \
     -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
   ```

### wxWidgets Not Found

**Issue**: `CMake Error: Could not find wxWidgets`

**Solution**:
1. Verify wxWidgets is installed:
   ```bash
   brew list wxwidgets
   # or for vcpkg:
   /path/to/vcpkg/vcpkg list | grep wxwidgets
   ```

2. Explicitly pass wxWidgets_ROOT_DIR:
   ```bash
   cmake -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DwxWidgets_ROOT_DIR=$(brew --prefix wxwidgets)
   ```

### Architecture Mismatch

**Issue**: Binary won't run or reports architecture errors

**Solution**:
1. Verify your Mac's architecture:
   ```bash
   uname -m
   # arm64 = Apple Silicon (M1/M2/M3)
   # x86_64 = Intel
   ```

2. Rebuild with correct architecture:
   ```bash
   cmake -B build -G Ninja \
     -DCMAKE_OSX_ARCHITECTURES=arm64  # or x86_64
   ```

### Missing Assets

**Issue**: GUI shows missing images or broken transparency

**Solution**:
1. Verify assets exist in source:
   ```bash
   ls -la assets/
   ```

2. Check if assets were copied to app bundle:
   ```bash
   ls -la build/bin/Release/TripSitter.app/Contents/Resources/assets/
   ```

3. Manually copy if needed:
   ```bash
   cp -r assets/* build/bin/Release/TripSitter.app/Contents/Resources/assets/
   ```

4. Rebuild to trigger CMake's copy command:
   ```bash
   cmake --build build --config Release --clean-first
   ```

### Font Not Rendering

**Issue**: Corpta font doesn't appear in the GUI

**Solution**:
1. List installed fonts:
   ```bash
   fc-list | grep -i corpta
   ```

2. Check Font Book:
   - Open Font Book
   - Search for "Corpta"
   - Ensure it's enabled (not grayed out)

3. Verify font name in code matches installed font:
   - Check `src/GUI/PsychedelicTheme.cpp` for font references
   - Compare with the exact font name in Font Book

4. Restart the application after installing fonts

### Build Errors with Ninja

**Issue**: Ninja build fails or CMake can't find Ninja

**Solution**:
1. Verify Ninja is installed:
   ```bash
   which ninja
   ninja --version
   ```

2. Install if missing:
   ```bash
   brew install ninja
   ```

3. Use Xcode build instead (slower but more compatible):
   ```bash
   cmake -B build -G Xcode \
     -DCMAKE_BUILD_TYPE=Release
   ```

### Code Signing Issues

**Issue**: macOS blocks the application ("developer cannot be verified")

**Solution**:
1. For local development, disable Gatekeeper temporarily:
   ```bash
   sudo spctl --master-disable
   ```
   
   (Re-enable after: `sudo spctl --master-enable`)

2. Or allow specific app:
   ```bash
   xattr -cr build/bin/Release/TripSitter.app
   ```

3. For distribution, sign the app (requires Apple Developer ID):
   ```bash
   codesign --deep --force --sign "Developer ID Application: Your Name" \
     build/bin/Release/TripSitter.app
   ```

## Build Configurations

### Debug Build

For development with debugging symbols:
```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_OSX_ARCHITECTURES=arm64

cmake --build build --config Debug
```

Debug build includes:
- Full debugging symbols
- No optimizations
- Easier to debug with lldb/Xcode

### RelWithDebInfo Build

Optimized build with debugging symbols:
```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build build --config RelWithDebInfo
```

Useful for profiling and performance analysis.

## Clean Build

To start from scratch:

```bash
# Remove build directory
rm -rf build

# Reconfigure and rebuild
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DFFMPEG_ROOT=$(brew --prefix ffmpeg) \
  -DwxWidgets_ROOT_DIR=$(brew --prefix wxwidgets)

cmake --build build --config Release
```

## Performance Notes

- **First build**: 5-15 minutes (depending on Mac speed)
- **Incremental builds**: 30 seconds - 2 minutes
- **Clean rebuild**: 3-10 minutes

Ninja builds are typically faster than Xcode builds for this project.

## Next Steps

After successful build:
1. Run smoke tests (CLI and GUI)
2. Test with various audio/video files
3. Verify all UI elements render correctly
4. Create DMG package for distribution
5. Test on different macOS versions if possible

## Additional Resources

- **FFmpeg Documentation**: https://ffmpeg.org/documentation.html
- **wxWidgets macOS Guide**: https://docs.wxwidgets.org/latest/plat_osx_install.html
- **CMake macOS Bundle Guide**: https://cmake.org/cmake/help/latest/prop_tgt/MACOSX_BUNDLE.html
- **vcpkg Documentation**: https://vcpkg.io/en/getting-started.html

## Contact & Support

For build issues specific to this project, please file an issue on the GitHub repository.
