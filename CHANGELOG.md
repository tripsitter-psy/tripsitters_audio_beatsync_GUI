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
- Updated `tests/CMakeLists.txt` with regression test configuration and DLL copy helper

### Changed
- Improved output name handling in OnnxBeatDetector to prevent vector reallocation invalidating pointers
