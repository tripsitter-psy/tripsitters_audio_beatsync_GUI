# TripSitter BeatSync Editor - Comprehensive TODO List

## Overview
This project is a desktop application for beat-syncing videos to audio using Unreal Engine. The current implementation has architectural issues (built as Game target instead of Program target) that need to be resolved for proper desktop app behavior.

## Immediate Fixes Completed ✅
- [x] Fixed memory leaks in AudioAnalyzer (RAII for buffers)
- [x] Added thread safety for bCancelRequested (FThreadSafeBool)
- [x] Fixed callback storage leaks (proper cleanup)
- [x] Updated error handling in C API (catch exceptions, set s_lastError)
- [x] Fixed NSIS installer paths (consistent bin/ directory)
- [x] Updated GitHub Actions (valid checkout SHA)
- [x] Fixed vcpkg baseline (removed invalid hash)
- [x] Improved string handling (FTCHARToANSI/UTF8 converters)
- [x] Fixed weak_ptr lifetime issues in processing tasks
- [x] Corrected typedef usage (FProgressCb consistency)

## Major Architectural Restructure 🚧

### Phase 1: Convert to Program Target
- [ ] Create engine symlink for installed UE build
  ```powershell
  mklink /d "C:\Path\To\Project\Engine" "C:\Program Files\Epic Games\UE_5.7\Engine"
  ```
- [ ] Create TripSitter.Target.cs (TargetType.Program)
- [ ] Create TripSitter.Build.cs (proper module dependencies)
- [ ] Implement TripSitter.cpp main entry point
- [ ] Restructure Source/ folder (remove game modules)

### Phase 2: Remove Game Framework Dependencies
- [ ] Remove TripSitterGameMode
- [ ] Remove TripSitterPlayerController
- [ ] Remove viewport injection code
- [ ] Remove cursor workaround code
- [ ] Clean up DefaultEngine.ini (remove game settings)

### Phase 3: Preserve and Adapt Core Functionality
- [ ] Keep STripSitterMainWidget (Slate UI)
- [ ] Keep SWaveformViewer (audio visualization)
- [ ] Adapt FBeatsyncLoader (DLL loading)
- [ ] Adapt FBeatsyncProcessingTask (async processing)
- [ ] Preserve resources (fonts, images, assets)

## Build and Packaging 📦
- [ ] Update build scripts for Program target
- [ ] Test packaging with NSIS installer
- [ ] Verify backend DLL distribution
- [ ] Test mouse cursor functionality
- [ ] Validate UI responsiveness

## Testing and Validation 🧪
- [ ] Unit tests for backend API
- [ ] Integration tests for video processing
- [ ] UI tests for Slate widgets
- [ ] Performance tests for audio analysis
- [ ] Cross-platform compatibility (if needed)

## Documentation 📚
- [ ] Update README.md with new architecture
- [ ] Document build process for Program target
- [ ] Add troubleshooting guide
- [ ] Update developer setup instructions

## Future Enhancements 🔮
- [ ] GPU acceleration for video processing
- [ ] Additional audio analysis algorithms
- [ ] Plugin system for effects
- [ ] Multi-language support
- [ ] Advanced beat detection options

## Risk Mitigation 🛡️
- [ ] Backup current working Game target
- [ ] Incremental testing during restructure
- [ ] Rollback plan if Program target fails
- [ ] Compatibility testing with existing user data

## Priority Order
1. **HIGH**: Complete Program target conversion (Phases 1-3)
2. **MEDIUM**: Build and packaging fixes
3. **LOW**: Testing, documentation, enhancements

## Notes
- Current Game target works but has workarounds
- Program target will provide clean, maintainable architecture
- All Slate UI code is reusable without changes
- Backend C API integration remains stable

---

*Last updated: January 12, 2026*
*Maintained by: Development Team*