# BeatSync Editor

A C++ desktop application for automatically synchronizing video clips with music beats using audio analysis and intelligent cutting algorithms.

## Current Status: Phase 1 - Audio Analysis ✓

Phase 1 implements core beat detection functionality with a command-line interface.

### Features

- **Audio File Loading**: Supports MP3, WAV, FLAC, and other formats via FFmpeg
- **Beat Detection**: Energy-based beat detection algorithm
- **BPM Estimation**: Automatic tempo detection
- **Configurable Sensitivity**: Adjust beat detection sensitivity (0.0-1.0)
- **Detailed Output**: Beat timestamps, BPM, and analysis statistics

## Architecture

```
BeatSyncEditor/
├── src/
│   ├── audio/
│   │   ├── AudioAnalyzer.h/cpp    # FFmpeg integration & beat detection
│   │   └── BeatGrid.h/cpp         # Beat timestamp storage
│   └── main.cpp                   # CLI entry point
├── build/                         # Build output directory
└── CMakeLists.txt                 # Build configuration
```

## Requirements

### Build Tools
- CMake 3.20+
- C++17 or C++20 compiler (MSVC 2022 on Windows)

### Libraries
- **FFmpeg** (libavcodec, libavformat, libavutil, libswresample) - Audio decoding
- **OpenCV** (optional, for Phase 2+) - Video processing

## Building

For detailed build instructions:
- **[macOS Build Guide](BUILD_MACOS.md)** - Comprehensive macOS instructions (includes automated script)
- **[Windows Build Guide](BUILD.md)** - Full Windows build instructions

### Quick Start - macOS

```bash
# Automated build (recommended)
./build_macos.sh

# Or manual steps:
brew install ffmpeg cmake ninja wxwidgets
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFFMPEG_ROOT=$(brew --prefix ffmpeg) \
  -DwxWidgets_ROOT_DIR=$(brew --prefix wxwidgets)
cmake --build build --config Release
```

### Quick Start - Windows

### 1. Install Dependencies

Using vcpkg (recommended on Windows):

```bash
# Install FFmpeg
vcpkg install ffmpeg:x64-windows

# OpenCV (optional, for Phase 2+)
vcpkg install opencv:x64-windows
```

### 2. Configure and Build

```bash
cd BeatSyncEditor
mkdir build
cd build

# Configure with vcpkg
cmake .. -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake

# Or configure with manual FFmpeg location
cmake .. -DFFMPEG_DIR="C:/path/to/ffmpeg"

# Build
cmake --build . --config Release
```

### 3. Run

```bash
# The executable will be in build/bin/
cd bin
beatsync analyze path/to/audio.mp3
```

## Usage

### Basic Beat Analysis

```bash
beatsync analyze song.mp3
```

### With Custom Sensitivity

```bash
beatsync analyze track.wav --sensitivity 0.7
```

Higher sensitivity (0.8-1.0) detects more beats, lower (0.2-0.4) is more conservative.

### Output Example

```
========================================
BeatSync Audio Analyzer
========================================

Analyzing: song.mp3
Sensitivity: 0.5

Audio loaded: 180.5s, 44100 Hz, 7958400 samples
Detected 458 beats

========================================
Analysis Results
========================================

BeatGrid Information:
  Number of beats: 458
  BPM: 120.5
  Duration: 180.245 seconds
  Average interval: 0.498 seconds

Beat Timestamps (first 20):
  Beat   1:  0:  0.523 (0.523s)
  Beat   2:  0:  1.021 (1.021s)
  Beat   3:  0:  1.519 (1.519s)
  ...
```

## Algorithm Details

### Beat Detection

The current implementation uses an energy-based beat detection algorithm:

1. **Audio Decoding**: FFmpeg decodes audio to mono PCM float32
2. **Energy Calculation**: Compute energy in 30ms frames with 50% overlap
3. **Smoothing**: Apply moving average filter to energy envelope
4. **Peak Detection**: Find local maxima above dynamic threshold
5. **Beat Filtering**: Remove beats too close together (minimum 300ms gap)

### BPM Estimation

- Calculates intervals between consecutive beats
- Uses median interval (more robust than mean)
- Converts to beats per minute: BPM = 60 / median_interval

## Upcoming Phases

### Phase 2: Video Processing
- FFmpeg video decoding
- Frame extraction at specific timestamps
- Video splitting and re-encoding
- CLI: `beatsync split video.mp4 timestamps.txt`

### Phase 3: Synchronization Engine
- Automatic video-beat alignment
- Multiple cut strategies (every-beat, downbeat, custom)
- CLI: `beatsync create video.mp4 audio.mp3 --strategy downbeat`

### Phase 4+: GUI Application
- Qt 6 desktop interface
- Visual timeline with beat markers
- Waveform visualization
- Drag-and-drop editing
- Real-time preview

### GUI Assets
To customize the GUI appearance, place image files into the `assets/` directory at the project root:

- `background.png` — Recommended: 1920x1080 PNG for the main background
- `icon.ico` — Windows application icon (used when building the GUI executable)

A helper script is provided to import assets from another location (for example `C:\Users\samue\Downloads\assets for GUI aesthetics`):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\import_assets.ps1 -Source "C:\Users\samue\Downloads\assets for GUI aesthetics" -Force
```

The script backs up existing assets into `assets/backup-YYYYMMDD-HHMMSS` before copying. Replace placeholder files with real images before creating release builds.

## Technical Notes

### Why Not Essentia?

While Essentia is a comprehensive audio analysis library, it has complex build requirements on Windows. For Phase 1, we implemented a lightweight energy-based beat detector that:
- Has no external dependencies beyond FFmpeg
- Builds easily on Windows with MSVC
- Provides good accuracy for most music
- Can be upgraded to Essentia or other libraries later

### Performance

- Typical analysis speed: ~10-20x real-time (analyze 3min song in <20s)
- Memory usage: ~40MB per minute of audio
- Supports audio files up to several hours

## Development

### Project Structure

- `src/audio/BeatGrid.*` - Data structure for beat timestamps
- `src/audio/AudioAnalyzer.*` - Audio loading and beat detection
- `src/main.cpp` - CLI application entry point

### Adding New Features

To add new beat detection algorithms:
1. Modify `AudioAnalyzer::detectBeats()` method
2. Add new parameters via `AudioAnalyzer` interface
3. Expose via CLI arguments in `main.cpp`

## License

[To be determined]

## Contributing

[To be determined]

## Contact

[To be determined]
