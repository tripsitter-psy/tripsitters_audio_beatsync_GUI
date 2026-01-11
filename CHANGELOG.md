# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed heap corruption in OnnxBeatDetector caused by dangling pointers during output name collection
- Removed erroneous `allocator.Free()` call that caused double-free during ONNX inference cleanup

### Added
- Added `tests/test_onnx_detector_regression.cpp` with 200-iteration stress test to catch allocator/heap corruption regressions
- Added `tests/onnx_inference_helper.cpp` and `test_onnx_inference_helper` to run ONNX inference in an isolated process to contain runtime crashes
- Updated `tests/CMakeLists.txt` with regression test configuration and DLL copy helper

### Changed
- Improved output name handling in OnnxBeatDetector to prevent vector reallocation invalidating pointers
- ONNX tests now use official Microsoft binaries instead of vcpkg-built ones (workaround for vcpkg heap corruption bug, see [vcpkg#49349](https://github.com/microsoft/vcpkg/issues/49349))
- Added `cmake/FetchOnnxRuntime.cmake` to automatically download official ONNX Runtime binaries during configure
- Added optional SHA256 verification for official ONNX Runtime download and CI caching for the downloaded archive to improve supply-chain safety and speed CI runs
- Temporarily pinned `onnxruntime` to `>= 1.18.1` in `vcpkg.json` to avoid regressions introduced in vcpkg's 1.23.2 build (see https://github.com/microsoft/vcpkg/issues/49349)
