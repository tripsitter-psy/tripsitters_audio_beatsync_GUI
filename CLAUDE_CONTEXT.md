# BeatSyncEditor - Development Context

## Project Overview
**BeatSyncEditor** is a C++ command-line application that synchronizes video clips to audio beats using FFmpeg. It analyzes audio files to detect beats, then cuts and arranges video clips to match those beats.

**Location**: `C:\Users\samue\Desktop\BeatSyncEditor`

## Architecture

```
src/
├── main.cpp              # CLI entry point, command handlers
├── audio/
│   ├── AudioAnalyzer.cpp # Beat detection from audio files
│   └── BeatGrid.cpp      # Beat timing data structure
└── video/
    ├── VideoProcessor.cpp # FFmpeg-based video reading/info
    └── VideoWriter.cpp    # Segment extraction, concatenation, audio muxing
```

## Commands
- `analyze <audio>` - Detect beats in audio file
- `sync <video> <audio>` - Sync single video to beats
- `multiclip <folder> <audio>` - Create beat-synced video from multiple clips (cycles through clips)
- `split <video> <audio>` - Split video at beat timestamps

## Build System
- CMake-based, builds with MSVC on Windows
- FFmpeg libraries from `C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared`
- Build command: `cmake --build build --config Release`
- Output: `build\bin\Release\beatsync.exe`

---

## Session Log (2026-01-02)

### Issue 1: Video Freezing and Desync
**Problem**: Video freezes intermittently while audio continues

**Root Cause**: Source video clips had **MIXED RESOLUTIONS** (1920x1080 vs 3808x2176)
- FFmpeg constantly reconfigured filter graph
- Resulted in 19,000 duplicated frames and 7,500 dropped frames
- DTS timestamp errors

**Fix Applied** (VideoWriter.cpp):
All segments now normalized to consistent format:
```cpp
-vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24"
-c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p
-c:a aac -b:a 192k -ar 44100
-video_track_timescale 90000
```

**Status**: FIXED - No more freezing

---

### Issue 2: Output Duration Shorter Than Audio
**Problem**: Output video was 5:45 when audio was 6:01 (15 seconds short)

**Root Cause**:
- Beat detection ends before song ends (fade-out sections have no strong beats)
- Last segment used fixed 2-second duration instead of extending to audio end

**Fix Applied**:

1. **BeatGrid.h/cpp** - Added audio duration tracking:
   ```cpp
   void setAudioDuration(double duration);
   double getAudioDuration() const;
   double m_audioDuration;  // Actual audio file duration
   ```

2. **AudioAnalyzer.cpp** - Store actual audio duration:
   ```cpp
   beatGrid.setAudioDuration(audio.duration);
   ```

3. **main.cpp (multiclip command)** - Pad video to match audio:
   - Last beat segment extends to audio end (not fixed 2 seconds)
   - If still short, adds padding segments cycling through clips
   - Changed `trimToShortest` to `false` so audio isn't cut

**Status**: FIXED - Build successful, awaiting test

---

## Current Status
- **Freezing issue**: FIXED and tested
- **Duration padding**: FIXED, build successful, AWAITING TEST

---

## Test Data Locations
- Video clips: `C:\Users\samue\Downloads\midjourny\` (42 .mp4 files, MIXED RESOLUTIONS)
- Audio file: `C:\Users\samue\Downloads\we're the imagination-01-01.wav` (6:01 duration)

## Quick Start for Next Session

```bash
# Navigate to project
cd C:\Users\samue\Desktop\BeatSyncEditor

# Build
cmake --build build --config Release

# Test multiclip sync
.\build\bin\Release\beatsync.exe multiclip "C:\Users\samue\Downloads\midjourny" "C:\Users\samue\Downloads\we're the imagination-01-01.wav" -o output.mp4

# Verify output matches audio duration (should be ~6:01)
ffprobe output.mp4
```

---

## Future Enhancements (Ideas)
- [ ] Configurable output resolution (not just 1920x1080)
- [ ] Configurable frame rate (not just 24fps)
- [ ] Add progress bar with ETA
- [ ] Support for different beat detection algorithms
- [ ] GUI interface
- [ ] Preview mode (process only first N beats)
- [ ] Transition effects between clips (crossfade, etc.)
- [ ] Random vs sequential clip selection option
- [ ] Export beat grid to file for reuse
- [ ] Support for variable clip durations

## Known Dependencies
- FFmpeg (path auto-detected or set via `BEATSYNC_FFMPEG_PATH` env var)
- FFmpeg dev libraries for compilation
- Windows 10/11

---

## Technical Notes

### FFmpeg Filter Chain Explanation
```
scale=1920:1080:force_original_aspect_ratio=decrease
  → Scale to fit within 1920x1080, maintaining aspect ratio

pad=1920:1080:(ow-iw)/2:(oh-ih)/2
  → Add black bars (letterbox/pillarbox) to reach exactly 1920x1080

setsar=1
  → Set sample aspect ratio to 1:1 (square pixels)

fps=24
  → Force consistent 24fps frame rate
```

### Why Stream Copy Works for Concatenation
Previously, concatenating required re-encoding because segments had different properties.
Now that all segments are normalized during extraction, we use `-c copy` for
fast concatenation without quality loss.

### Audio Duration Padding Logic
```
1. Track actual audio duration from AudioAnalyzer
2. Last beat segment: duration = audioDuration - lastBeatTime
3. If total video < audio duration, add padding segments (2s each)
4. Don't use -shortest flag so full audio plays
```
