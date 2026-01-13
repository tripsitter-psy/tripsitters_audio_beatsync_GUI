# TripSitter BeatSync Editor - TODO List

## Overview

This project is a desktop application for beat-syncing videos to audio. It consists of:

- **Backend**: C++ DLL with FFmpeg + ONNX Runtime (CUDA/TensorRT GPU acceleration)
- **Frontend**: Unreal Engine 5 standalone program (TripSitter)

## Completed

### Build System

- [x] CMake configuration with vcpkg
- [x] ONNX Runtime 1.23.2 with CUDA + TensorRT support
- [x] TensorRT 10.9.0.34 integration via overlay triplet
- [x] Backend DLL builds successfully
- [x] TripSitter UE5 program builds successfully

### Code Fixes (January 2026)

- [x] Fixed bs_ai_result_t struct redefinition in beatsync_capi.h
- [x] Fixed std::numbers::pi C++20 issue (replaced with constexpr PI)
- [x] Fixed missing brace in bs_ai_analyze_quick function
- [x] Fixed IDesktopPlatform preprocessor condition for standalone builds
- [x] Fixed memory leaks in AudioAnalyzer (RAII for buffers)
- [x] Added thread safety for bCancelRequested (FThreadSafeBool)
- [x] Fixed callback storage leaks (proper cleanup)
- [x] Updated error handling in C API (catch exceptions, set s_lastError)

### C API

- [x] Core functions (init, shutdown, version)
- [x] Audio analysis (basic beat detection, waveform extraction)
- [x] AI analysis (ONNX Runtime inference with progress callbacks)
- [x] Video processing (cut, concatenate, add audio, effects)
- [x] Frame extraction for preview
- [x] Tracing support (OpenTelemetry)

### UE Integration

- [x] BeatsyncLoader DLL wrapper
- [x] FBeatsyncProcessingTask async processing
- [x] STripSitterMainWidget Slate UI
- [x] SWaveformViewer visualization
- [x] Native Windows file dialogs for standalone builds

## In Progress

### AI Beat Detection Models

- [ ] Convert BeatNet model to ONNX format
- [ ] Convert All-In-One model to ONNX format
- [ ] Convert TCN model to ONNX format
- [ ] Test inference with real audio files
- [ ] Benchmark GPU vs CPU performance

## Pending

### Testing

- [ ] End-to-end test of effects pipeline with real video
- [ ] Test frame extraction in UE preview widget
- [ ] Verify async task completion and UI updates
- [ ] Add comprehensive C API unit tests
- [ ] Performance benchmarks for audio analysis

### Features

- [ ] Stem separation (Demucs) for drums-first beat detection
- [ ] Additional beat detection algorithms (Essentia)
- [ ] GLSL transition library for beat-synced cuts
- [ ] Audio-reactive visual effects
- [ ] Export/import beat grid files

### Documentation

- [ ] User guide for TripSitter app
- [ ] API documentation for C interface
- [ ] Model training guide for custom beat detectors

### Packaging

- [ ] NSIS installer for Windows
- [ ] Include TensorRT runtime DLLs
- [ ] Include CUDA runtime DLLs
- [ ] Code signing for distribution

## Known Issues

### Build Issues

- FSlateFontInfo deprecation warnings (use FCompositeFont constructor)
- vcpkg ONNX Runtime build takes ~2 hours with TensorRT

### Runtime Issues

- TensorRT requires specific CUDA version compatibility
- GPU memory usage needs monitoring for large audio files

## Priority Order

1. **HIGH**: Complete AI model integration and testing
2. **MEDIUM**: End-to-end testing with real media files
3. **LOW**: Documentation, packaging, additional features

## Quick Reference

```powershell
# Build backend
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake --overlay-triplets=triplets
cmake --build build --config Release --target beatsync_backend_shared


# Build TripSitter
# Set the UE5 root directory as an environment variable (e.g., $Env:UE5_ROOT in PowerShell or %UE5_ROOT% in cmd), or adjust the path below as needed.
# Example (PowerShell): $Env:UE5_ROOT="C:\UE5_Source\UnrealEngine"
# Example (cmd): set UE5_ROOT=C:\UE5_Source\UnrealEngine
Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination "$Env:UE5_ROOT\Engine\Source\Programs\TripSitter\Private\" -Recurse -Force
& "$Env:UE5_ROOT\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development

# Run tests
cmake --build build --config Release --target test_backend_api
./build/tests/Release/test_backend_api.exe
```

---

Last updated: January 14, 2026
