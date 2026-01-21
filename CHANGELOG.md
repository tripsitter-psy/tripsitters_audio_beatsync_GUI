# Changelog

All notable changes to MTV TripSitter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Windows NSIS installer with proper TripSitter.exe integration
- ONNX models bundled in installer for AI beat detection
- UI resources (fonts, images) bundled in installer
- Third-party license documentation
- Unified release build script (`scripts/build_release.ps1`)

### Changed
- Installer now references TripSitter.exe instead of placeholder beatsync.exe
- Improved uninstaller cleanup for models, resources, and licenses directories

### Fixed
- NSIS template executable references corrected

## [0.1.0] - 2026-01-21

### Added
- **TripSitter GUI**: Unreal Engine 5 standalone application with Slate UI
  - Waveform visualization with beat markers
  - Video preview with frame extraction
  - Multiple analysis modes: Energy, ONNX AI, Flux, Stems + Flux
  - Effect timeline for beat-synchronized video effects
  - Custom Corpta font integration

- **C++ Backend** (`beatsync_backend_shared.dll`)
  - Audio analysis with automatic BPM detection
  - Beat grid generation with downbeat tracking
  - Waveform peak extraction for visualization

- **AI Beat Detection** (ONNX Runtime)
  - BeatNet neural network integration
  - TCN (Temporal Convolutional Network) models
  - Demucs stem separation for enhanced accuracy
  - TensorRT acceleration on NVIDIA RTX GPUs
  - CUDA fallback for GTX GPUs
  - CPU fallback for systems without NVIDIA GPUs

- **AudioFlux Integration**
  - Spectral flux onset detection
  - Stems + Flux hybrid analysis mode
  - No GPU required for spectral analysis

- **Video Processing** (FFmpeg)
  - Beat-synchronized video cutting
  - Multi-video cycling at beat times
  - Audio track muxing
  - Video concatenation
  - Frame extraction for preview

- **Beat Effects**
  - Flash effect on beat
  - Zoom pulse effect
  - Color grading
  - Vignette effect
  - Configurable transitions between clips

- **Build System**
  - CMake with vcpkg integration
  - NSIS installer generation
  - Code signing support
  - TensorRT DLL bundling (optional)
  - AudioFlux DLL bundling (optional)

- **CI/CD**
  - GitHub Actions release workflow
  - Automated installer smoke tests
  - Code signing integration (when certificate available)

### Technical Details
- C++17 standard
- ONNX Runtime 1.23.x with GPU acceleration
- FFmpeg 8.x for audio/video processing
- TensorRT 10.9.x support (optional)
- AudioFlux for spectral analysis (optional)
- Unreal Engine 5.x (source build) for GUI

### Known Issues
- TripSitter.exe must be built separately using Unreal Engine
- Models (~294 MB) significantly increase installer size
- TensorRT DLLs (~515 MB) are optional but add to installer size

## [0.0.1] - 2024-12-01

### Added
- Initial project structure
- Basic audio analysis prototype
- FFmpeg integration for video processing
- C API design (`beatsync_capi.h`)

---

[Unreleased]: https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI/releases/tag/v0.0.1
