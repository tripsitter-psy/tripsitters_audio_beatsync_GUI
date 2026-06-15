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
- [x] Fixed models install path for packaged version (CMakeLists.txt)
- [x] Fixed Start Menu shortcuts not removed by uninstaller (SetShellVarContext)
- [x] Fixed TripSitter.Build.cs duplicate symbol errors (removed TripSitterUE dependency)

### Beat Detection Tuning (January 2026)

- [x] Tuned OnnxBeatDetector parameters for psytrance (hopLength 256, thresholds 0.5)
- [x] Tuned AudioFluxBeatDetector parameters (hopLength 256, threshold 0.2)
- [x] Added low-frequency focus (30-200Hz) for kick drum isolation
- [x] Fixed hardcoded thresholds in BeatsyncProcessingTask.cpp (0.66→0.5)
- [x] Created Demucs kick training setup (training/demucs_kick/)

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

### Beat Detection Quality (Psytrance Focus)

- [x] Parameter tuning for EDM/psytrance (thresholds, hop length)
- [x] Low-frequency focus for kick drum isolation (30-200Hz)
- [ ] Test tuned parameters with psytrance tracks
- [ ] Fine-tune thresholds based on test results
- [ ] Train custom kick-only Demucs model (dataset: training/demucs_kick/)

### AI Beat Detection Models

- [x] BeatNet ONNX model integrated
- [ ] Convert All-In-One model to ONNX format
- [ ] Convert TCN model to ONNX format
- [ ] Benchmark GPU vs CPU performance

## Pending

### Testing

- [ ] End-to-end test of effects pipeline with real video
- [ ] Test frame extraction in UE preview widget
- [ ] Verify async task completion and UI updates
- [ ] Add comprehensive C API unit tests
- [ ] Performance benchmarks for audio analysis

### Features

- [x] Stem separation (Demucs) for drums-first beat detection
- [ ] Train custom kick-only Demucs model for psytrance
- [ ] Additional beat detection algorithms (Essentia)
- [ ] GLSL transition library for beat-synced cuts
- [ ] Audio-reactive visual effects
- [ ] Export/import beat grid files

### Documentation

- [ ] User guide for TripSitter app
- [ ] API documentation for C interface
- [ ] Model training guide for custom beat detectors

### Packaging

- [x] NSIS installer for Windows
- [x] Include TensorRT runtime DLLs
- [x] Include AudioFlux DLLs
- [x] build_release.ps1 automated workflow
- [ ] Code signing for distribution

## Known Issues

### Build Issues

- FSlateFontInfo deprecation warnings (use FCompositeFont constructor)
- vcpkg ONNX Runtime build takes ~2 hours with TensorRT

### Runtime Issues

- TensorRT requires specific CUDA version compatibility
- GPU memory usage needs monitoring for large audio files
- **INVESTIGATING (Feb 2026)**: Crash when selecting audio file in TripSitter
  - Added debug logging to `%TEMP%\beatsync_ue_debug.log`
  - Added bounds checks for waveform band counts
  - Added AudioPathBox.IsValid() guard before SetText
  - Added memory limit check for STFT buffer allocation

### Code Sync Issues

- TripSitter standalone uses raw `void*` handles (not type-safe FAnalyzerHandle)
- Must manually copy source to UE5 Engine/Source/Programs/TripSitter/Private/
- Always rebuild TripSitter.exe after source changes

## Priority Order

1. **HIGH**: Beat detection quality for psytrance (parameter tuning, testing)
2. **HIGH**: Train custom kick-only Demucs model
3. **MEDIUM**: End-to-end testing with real media files
4. **LOW**: Documentation, additional features

## Quick Reference

```powershell
# Build backend
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_OVERLAY_TRIPLETS=triplets
cmake --build build --config Release --target beatsync_backend_shared


# Build TripSitter
# Set the UE5 root directory as an environment variable (e.g., $Env:UE5_ROOT in PowerShell or %UE5_ROOT% in cmd).
# Example (PowerShell): $Env:UE5_ROOT="D:\UnrealEngine"
if (-not (Test-Path "$Env:UE5_ROOT\Engine\Build\BatchFiles\Build.bat")) {
    Write-Error "UE5_ROOT not valid or Build.bat missing."
} else {
    $destIdx = "$Env:UE5_ROOT\Engine\Source\Programs\TripSitter\Private\"
    Write-Warning "Deploying to: $destIdx"
    $conf = Read-Host "Proceed with overwrite? (y/n)"
    if ($conf -eq 'y') {
        Copy-Item -Path 'unreal-prototype\Source\TripSitter\Private\*' -Destination $destIdx -Recurse -Force
        & "$Env:UE5_ROOT\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
    }
}

# Run tests
cmake --build build --config Release --target test_backend_api
./build/tests/Release/test_backend_api.exe
```

---

Last updated: February 7, 2026

## Session Notes (Feb 7, 2026)

### Current Issue: Audio File Selection Crash

The application crashes when selecting an audio file. Investigation in progress:

1. **Source Code Sync Issue**: The UE5 engine source was out of sync with the repo.
   - Fixed by copying updated source files to `D:\UnrealEngine\Engine\Source\Programs\TripSitter\Private\`
   - Rebuilt TripSitter.exe

2. **Fixes Applied**:
   - Added `AudioPathBox.IsValid()` guard before calling SetText (line 1781)
   - Added bounds check for waveform band count (max 10 million peaks)
   - Added memory limit check for STFT buffer allocation (max 500MB)
   - Added debug logging to GetWaveformBands function

3. **Debug Log Location**: `%TEMP%\beatsync_ue_debug.log`

4. **Files Modified** (uncommitted):
   - `unreal-prototype/Source/TripSitter/Private/STripSitterMainWidget.cpp`
   - `unreal-prototype/Source/TripSitter/Private/BeatsyncLoader.cpp`
   - `src/backend/beatsync_capi.cpp`

5. **Next Steps**:
   - Check debug log after crash to identify exact failure point
   - Rebuild installer once crash is fixed
   - Commit fixes
