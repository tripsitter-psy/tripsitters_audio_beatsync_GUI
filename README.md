# BeatSync Editor

A C++ desktop application for automatically synchronizing video clips with music beats using audio analysis, AI-powered beat detection, and intelligent cutting algorithms.

## Features

- **AI-Powered Beat Detection**: ONNX Runtime with CUDA/TensorRT GPU acceleration
- **Video Processing**: FFmpeg-based cutting, concatenation, and effects
- **Modern GUI**: Unreal Engine 5 Slate UI (TripSitter standalone app)
- **Waveform Visualization**: Interactive audio waveform with beat markers
- **Effects Pipeline**: Transitions, color grading, beat-synced flash/zoom

## Architecture

The project consists of two main components:

1. **Backend DLL** (`beatsync_backend_shared.dll`)
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
- Visual Studio 2022 (MSVC)
- Unreal Engine 5 (source build)

### Dependencies (via vcpkg)

- FFmpeg (avcodec, avformat, swresample, swscale, avfilter)
- ONNX Runtime 1.23.2

### GPU Acceleration (optional)

- CUDA Toolkit 12.x
- TensorRT 10.9.0.34

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
- CUDA + TensorRT GPU acceleration
- TripSitter UE5 standalone app
- C API for all major functions
- Waveform visualization

### In Progress

- AI beat detection model integration
- End-to-end testing

### Planned

- Stem separation (Demucs)
- Additional beat detection algorithms
- GLSL transition library
- NSIS installer

## Algorithm Details

### Beat Detection

The current implementation supports:

1. **Energy-based detection**: FFmpeg audio decoding, energy calculation, peak detection
2. **AI-based detection**: ONNX models for neural network inference (BeatNet, All-In-One, TCN)

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

Last updated: January 14, 2026
