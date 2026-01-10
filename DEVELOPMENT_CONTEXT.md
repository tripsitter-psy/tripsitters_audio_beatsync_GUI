# MTV TripSitter — Development Context

## What Is This Project?

**MTV TripSitter** is a video editing tool that automatically cuts and syncs video clips to the beat of music. Think of it like an automated music video maker with a psychedelic neon visual theme.

**In simple terms:** You give it a song and some video clips, it detects the beats in the music, then cuts up your video to switch clips on every beat (or every 2nd beat, 4th beat, etc.).

---

## Project Structure (Where Everything Lives)

```
Two main locations:

1. Backend Code (the "engine"):
   C:\Users\samue\Desktop\BeatSyncEditor\
   └── This is where the C++ code lives that does the actual work
       (audio analysis, video cutting, effects)

2. Unreal Engine Plugin (the "interface"):
   C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\
   └── This is the GUI that users interact with
       (buttons, waveform display, file selection)

3. Standalone Build (packaged app):
   C:\Users\samue\Desktop\TripSitterBuild\Windows\
   └── Ready-to-run version that doesn't need UE Editor
```

---

## How The Pieces Fit Together

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (UE Plugin)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  STripSitterMainWidget.cpp                                │  │
│  │  - File selection (audio, video, output)                  │  │
│  │  - Effect checkboxes and sliders                          │  │
│  │  - Start/Cancel buttons                                   │  │
│  │  - Progress bar                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  SWaveformViewer.cpp                                      │  │
│  │  - Shows audio waveform visually                          │  │
│  │  - Beat markers (yellow lines)                            │  │
│  │  - Selection handles (pink) for choosing video portion    │  │
│  │  - Effect regions you can draw                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  BeatsyncLoader.cpp                                       │  │
│  │  - Loads the DLL at runtime                               │  │
│  │  - Converts between UE types and C types                  │  │
│  │  - Acts as a "translator" between UE and the backend      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Calls functions via DLL
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 C API LAYER (beatsync_capi.cpp)                  │
│  - Exposes simple C functions that UE can call                  │
│  - Handles path conversion (Windows vs Unix style)              │
│  - Wraps the C++ classes in C-compatible functions              │
│                                                                  │
│  Key functions:                                                  │
│  • bs_analyze_audio()      → Detect beats in audio file         │
│  • bs_get_waveform()       → Get waveform data for display      │
│  • bs_video_cut_at_beats() → Process single video               │
│  • bs_video_cut_at_beats_multi() → Process multiple videos      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Uses C++ classes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKEND ENGINE (C++ Classes)                   │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │  AudioAnalyzer.cpp  │  │  VideoWriter.cpp                 │  │
│  │  - Load audio file  │  │  - Extract video segments        │  │
│  │  - Detect BPM       │  │  - Apply effects (flash, zoom)   │  │
│  │  - Find beat times  │  │  - Concatenate clips together    │  │
│  │  - Generate waveform│  │  - Add audio track               │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Calls FFmpeg commands
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          FFMPEG                                  │
│  External program that does the actual video manipulation        │
│  Location: C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared\ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files Explained

### Backend (C:\Users\samue\Desktop\BeatSyncEditor\src\)

| File | What It Does |
|------|--------------|
| `backend/beatsync_capi.cpp` | The "bridge" - exposes C functions that UE can call. Handles Windows path issues. |
| `backend/beatsync_capi.h` | Header file listing all the functions the DLL exports |
| `audio/AudioAnalyzer.cpp` | Loads audio files, detects beats, calculates BPM |
| `audio/BeatGrid.cpp` | Simple data structure holding beat times and BPM |
| `video/VideoWriter.cpp` | The big one - handles all video cutting, effects, and concatenation |
| `video/VideoProcessor.cpp` | Gets video info (duration, resolution, fps) |

### UE Plugin (Plugins\TripSitterUE\Source\TripSitterUE\)

| File | What It Does |
|------|--------------|
| `Public/BeatsyncLoader.h` | Declares the DLL loading functions |
| `Private/BeatsyncLoader.cpp` | Loads the DLL, gets function pointers, wraps them for UE |
| `Public/STripSitterMainWidget.h` | Declares the main UI widget and all its member variables |
| `Private/STripSitterMainWidget.cpp` | The entire UI - file selection, buttons, progress, all the controls |
| `Public/SWaveformViewer.h` | Declares the waveform widget |
| `Private/SWaveformViewer.cpp` | Custom widget that draws waveform, handles mouse input, zoom/pan |

---

## How Video Processing Works (Step by Step)

When you click "START SYNC", here's what happens:

```
1. AUDIO ANALYSIS
   ┌─────────────────────────────────────────┐
   │ Load audio file (MP3/WAV/FLAC)          │
   │ ↓                                        │
   │ Analyze energy levels over time         │
   │ ↓                                        │
   │ Detect peaks = beat locations           │
   │ ↓                                        │
   │ Calculate BPM from beat spacing         │
   │ ↓                                        │
   │ Return list of beat times in seconds    │
   │ Example: [0.5, 1.0, 1.5, 2.0, 2.5, ...] │
   └─────────────────────────────────────────┘

2. BEAT FILTERING (based on user selection)
   ┌─────────────────────────────────────────┐
   │ "Every Beat"     → use all beats        │
   │ "Every 2nd Beat" → use beats 0,2,4,6... │
   │ "Every 4th Beat" → use beats 0,4,8,12...│
   └─────────────────────────────────────────┘

3. SEGMENT EXTRACTION (for each beat)
   ┌─────────────────────────────────────────┐
   │ For each beat time:                     │
   │   - Calculate segment duration          │
   │   - Pick source video (cycles through   │
   │     all videos in folder mode)          │
   │   - Call FFmpeg to extract that segment │
   │   - Normalize to 1920x1080, 24fps       │
   │   - Save to temp file                   │
   └─────────────────────────────────────────┘

4. CONCATENATION
   ┌─────────────────────────────────────────┐
   │ Create a list of all temp segment files │
   │ Call FFmpeg to join them into one video │
   │ Clean up temp files                     │
   └─────────────────────────────────────────┘

5. OUTPUT
   ┌─────────────────────────────────────────┐
   │ Final video saved to user's chosen path │
   └─────────────────────────────────────────┘
```

---

## Multi-Video Mode (Folder Selection)

When user clicks "Folder..." instead of "File...":

```
1. Scan folder for video files (*.mp4, *.mov, *.avi, *.mkv, *.webm)
2. Sort alphabetically
3. Store list in VideoPaths array
4. Set bIsMultiClip = true

During processing:
   Beat 0  → Use video 0
   Beat 1  → Use video 1
   Beat 2  → Use video 2
   Beat 3  → Use video 0 (cycles back)
   Beat 4  → Use video 1
   ... and so on
```

This creates a music video that switches between different clips on each beat!

---

## Building The Project

### Step 1: Build the Backend DLL

```powershell
# Open PowerShell and run:
cd "C:\Users\samue\Desktop\BeatSyncEditor"

# Build command (one line):
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" "build\TripSitterBeatSync.sln" -t:beatsync_backend_shared -p:Configuration=Release -p:Platform=x64

# If successful, you'll see:
#   beatsync_backend_shared.vcxproj -> ...\build\Release\beatsync_backend_shared.dll
```

### Step 2: Copy DLL to UE Plugin

```powershell
# Copy the built DLL to where UE expects it:
Copy-Item "build\Release\beatsync_backend_shared.dll" `
  "C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\ThirdParty\beatsync\lib\x64\"
```

### Step 3: Test in UE Editor

```powershell
# Open the UE project:
Start-Process "C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe" `
  -ArgumentList "C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\MyProject.uproject"
```

### Step 4: Package Standalone Build

```powershell
# Run from Epic Games folder (needs admin for some machines):
& "C:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\RunUAT.bat" `
  BuildCookRun `
  -project="C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\MyProject.uproject" `
  -noP4 -platform=Win64 -clientconfig=Development `
  -cook -allmaps -build -stage -pak -archive `
  -archivedirectory="C:\Users\samue\Desktop\TripSitterBuild"

# After packaging, copy DLLs to the build:
Copy-Item "C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\ThirdParty\beatsync\lib\x64\*.dll" `
  "C:\Users\samue\Desktop\TripSitterBuild\Windows\MyProject\Binaries\Win64\"
```

---

## Common Issues & Solutions

### "Video processing failed"

**Check the log files:**
```
C:\Users\samue\AppData\Local\Temp\beatsync_ffmpeg_extract.log
C:\Users\samue\AppData\Local\Temp\beatsync_ffmpeg_concat.log
```

**Common causes:**
1. **Input video too short** - If beat times exceed video duration, extraction fails
2. **Path issues** - UE uses forward slashes, FFmpeg needs backslashes on Windows
3. **FFmpeg not found** - Make sure FFmpeg is installed at the expected path
4. **Permissions** - Can't write to temp directory

### "Backend not loaded"

The DLL didn't load. Check:
1. Is `beatsync_backend_shared.dll` in the plugin's ThirdParty folder?
2. Are all dependency DLLs there? (avcodec, avformat, etc.)
3. Check UE's Output Log for specific errors

### UE Plugin Changes Not Showing

Delete these folders and restart UE:
```
Plugins\TripSitterUE\Intermediate\
Plugins\TripSitterUE\Binaries\
```

### Standalone Build Crashes on Start

Missing DLLs. Make sure these are in `Binaries\Win64\`:
- beatsync_backend_shared.dll
- avcodec-62.dll, avformat-62.dll, avutil-60.dll (FFmpeg)
- onnxruntime.dll (if using AI beat detection)

---

## The Waveform Viewer Explained

The waveform viewer is a custom Slate widget that:

```
Visual Elements:
┌──────────────────────────────────────────────────────────────┐
│ ▲                                                            │
│ │    ║  ║║║  ║║  ║║║║  ║║  ║║║  ║║  ║║║║  ║║  ║║║           │
│ │   ║║ ║║║║ ║║║ ║║║║║ ║║║ ║║║║ ║║║ ║║║║║ ║║║ ║║║║          │
│ 0  ─────────────────────────────────────────────────────────  │
│ │   ║║ ║║║║ ║║║ ║║║║║ ║║║ ║║║║ ║║║ ║║║║║ ║║║ ║║║║          │
│ │    ║  ║║║  ║║  ║║║║  ║║  ║║║  ║║  ║║║║  ║║  ║║║           │
│ ▼                                                            │
│      │     │     │     │     │     │     │  ← Beat markers   │
│   ┌──┐                                   ┌──┐                │
│   │  │ ← Start handle                    │  │ ← End handle   │
│   └──┘                                   └──┘                │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ← Selection region │
└──────────────────────────────────────────────────────────────┘

Mouse Controls:
• Scroll wheel = Zoom in/out (centered on cursor)
• Middle-click + drag = Pan left/right
• Left-click on handle = Drag to move
• Left-click elsewhere = Move nearest handle to that position
• Right-click = Context menu (add effect region)
```

---

## Effect Regions

Users can right-click on the waveform to add effect regions:

```
Effect Types:
┌─────────────────────────────────────────────────────────────┐
│ Vignette    - Dark edges around the video (purple overlay)  │
│ Beat Flash  - White flash on each beat (yellow overlay)     │
│ Beat Zoom   - Zoom pulse on each beat (cyan overlay)        │
│ Color Grade - Color correction preset (green overlay)       │
└─────────────────────────────────────────────────────────────┘

Each region has:
• Start handle (drag to adjust)
• End handle (drag to adjust)
• Colored overlay showing where effect applies
• Label showing effect name
• Right-click to remove
```

---

## Path Handling (Important!)

Windows and UE handle paths differently:

```
UE gives us:     /Users/samue/Downloads/video.mp4
                 or
                 C:/Users/samue/Downloads/video.mp4

FFmpeg needs:    C:\Users\samue\Downloads\video.mp4

The normalizePath() function in beatsync_capi.cpp handles this:
1. Converts / to \
2. Adds C: prefix if path starts with \Users\
3. Resolves relative paths (..\..\..) to absolute
```

---

## Dependencies

| Dependency | Purpose | Location |
|------------|---------|----------|
| Unreal Engine 5.7 | GUI framework | C:\Program Files\Epic Games\UE_5.7\ |
| Visual Studio 2022 | C++ compiler | C:\Program Files\Microsoft Visual Studio\2022\ |
| FFmpeg | Video processing | C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared\ |
| ONNX Runtime | AI beat detection (optional) | Bundled with DLL |

---

## Current Status (2026-01-10)

### What Works
- UI with psychedelic neon theme
- Audio loading and waveform display
- Beat detection and BPM calculation
- Beat markers on waveform
- Zoom and pan controls
- Selection handles
- Effect regions (visual only)
- Single video processing
- Multi-video (folder) selection
- Standalone packaged build

### Known Issues
- Video processing can fail if beat times exceed video duration
- Effect regions are drawn but not yet applied to video output
- Preview button not implemented

### What's Next
1. Apply effect regions to actual video processing
2. Add video duration validation
3. Implement preview functionality
4. Add project save/load

---

## Quick Reference Commands

```powershell
# Build DLL
msbuild build\TripSitterBeatSync.sln -t:beatsync_backend_shared -p:Configuration=Release -p:Platform=x64

# Copy DLL to plugin
cp build\Release\beatsync_backend_shared.dll "...\Plugins\TripSitterUE\ThirdParty\beatsync\lib\x64\"

# Check FFmpeg logs
Get-Content "C:\Users\samue\AppData\Local\Temp\beatsync_ffmpeg_extract.log" -Tail 50
Get-Content "C:\Users\samue\AppData\Local\Temp\beatsync_ffmpeg_concat.log" -Tail 50

# Find temp segment files
Get-ChildItem "C:\Users\samue\AppData\Local\Temp\beatsync_segment_*.mp4"

# Run standalone build
& "C:\Users\samue\Desktop\TripSitterBuild\Windows\MyProject.exe"
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **DLL** | Dynamic Link Library - a compiled code file that other programs can use |
| **FFI** | Foreign Function Interface - way for one language to call code in another |
| **Slate** | Unreal Engine's UI framework (like HTML/CSS but for C++) |
| **Widget** | A UI element (button, text box, etc.) |
| **Beat Grid** | Data structure holding all detected beat times |
| **BPM** | Beats Per Minute - tempo of the music |
| **Segment** | A small piece of video extracted between two timestamps |
| **Concatenate** | Join multiple video files into one |
| **UAT** | Unreal Automation Tool - builds and packages UE projects |

---

## File Locations Quick Reference

```
Source Code:
  C:\Users\samue\Desktop\BeatSyncEditor\src\

Built DLL:
  C:\Users\samue\Desktop\BeatSyncEditor\build\Release\beatsync_backend_shared.dll

UE Plugin:
  C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\

Plugin DLL Location:
  ...\Plugins\TripSitterUE\ThirdParty\beatsync\lib\x64\

Standalone Build:
  C:\Users\samue\Desktop\TripSitterBuild\Windows\

FFmpeg:
  C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe

Temp Files:
  C:\Users\samue\AppData\Local\Temp\beatsync_*.mp4
  C:\Users\samue\AppData\Local\Temp\beatsync_*.log
```

---

## Testing Guide

### Quick Test (5 minutes)

**What you need:**
- A short audio file (30 seconds to 2 minutes is ideal for testing)
- A video file that's at least as long as your audio
- OR a folder with 3-5 short video clips

**Steps:**

1. **Run the standalone build:**
   ```
   C:\Users\samue\Desktop\TripSitterBuild\Windows\MyProject.exe
   ```

2. **Load an audio file:**
   - Click "Browse..." next to Audio File
   - Select your test audio (MP3, WAV, or FLAC)
   - You should see the waveform appear

3. **Load video source:**
   - For single video: Click "File..." and select a video
   - For multi-clip: Click "Folder..." and select a folder with videos

4. **Set output location:**
   - Click "Browse..." next to Output File
   - Choose where to save (e.g., Desktop\test_output.mp4)

5. **Click "START SYNC"**
   - Watch the progress bar
   - If it fails, check the error message

6. **Verify output:**
   - Open the output video in VLC or similar
   - Check that cuts happen on beats
   - If multi-clip, verify it cycles through different videos

### Test Cases to Try

| Test | What to Check | Expected Result |
|------|---------------|-----------------|
| Short audio (30s) + long video (5min) | Does it work at all? | Output matches audio length |
| Long audio (5min) + short video (30s) | Video shorter than audio | Should fail or loop video |
| Multi-clip (3 videos) | Cycles through videos | Beat 0→vid1, Beat 1→vid2, Beat 2→vid3, Beat 3→vid1... |
| "Every 2nd Beat" setting | Fewer cuts | Half as many cuts as "Every Beat" |
| "Every 4th Beat" setting | Even fewer cuts | Quarter as many cuts |

### What Success Looks Like

```
✓ Waveform displays when audio loads
✓ Yellow beat markers appear on waveform
✓ Progress bar moves during processing
✓ Output video plays without errors
✓ Video cuts align with music beats
✓ Multi-clip mode cycles through all videos
```

### What Failure Looks Like (and what to do)

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| "Backend not loaded" | DLL missing or dependencies missing | Copy all DLLs from ThirdParty folder |
| "Video processing failed" | FFmpeg issue or path problem | Check log files (see below) |
| App crashes on start | Missing DLL | Run from command line to see error |
| No waveform appears | Audio loading failed | Try different audio format |
| Output video is empty | Segment extraction failed | Check extract log |
| Output video has no audio | Audio not muxed | Known issue - audio track not added yet |

### Checking Log Files

When something fails, always check these logs:

```powershell
# See extraction errors (each segment)
Get-Content "C:\Users\samue\AppData\Local\Temp\beatsync_ffmpeg_extract.log" -Tail 100

# See concatenation errors (joining segments)
Get-Content "C:\Users\samue\AppData\Local\Temp\beatsync_ffmpeg_concat.log" -Tail 100

# Check if segment files were created
Get-ChildItem "C:\Users\samue\AppData\Local\Temp\beatsync_segment_*.mp4" | Measure-Object
```

**Common log errors and meanings:**

| Log Message | Meaning | Fix |
|-------------|---------|-----|
| "Input video not found" | Path is wrong | Check file exists, path slashes |
| "Output file is empty" | Seek time > video duration | Use shorter audio or longer video |
| "FFmpeg not found" | FFmpeg not installed | Install FFmpeg, set BEATSYNC_FFMPEG_PATH |
| "Permission denied" | Can't write to output | Choose different output location |

---

## Best Practices

### For Development

1. **Always rebuild DLL after C++ changes**
   ```powershell
   # Quick rebuild command
   msbuild build\TripSitterBeatSync.sln -t:beatsync_backend_shared -p:Configuration=Release -p:Platform=x64
   ```

2. **Copy DLL before testing**
   ```powershell
   Copy-Item "build\Release\beatsync_backend_shared.dll" `
     "C:\Users\samue\OneDrive\Documents\Unreal Projects\MyProject\Plugins\TripSitterUE\ThirdParty\beatsync\lib\x64\"
   ```

3. **Clean UE intermediate files if plugin changes don't show**
   ```powershell
   Remove-Item -Recurse "...\Plugins\TripSitterUE\Intermediate"
   Remove-Item -Recurse "...\Plugins\TripSitterUE\Binaries"
   ```

4. **Use version control (git)**
   - Commit after each working feature
   - Use branches for experimental changes
   - Don't commit Intermediate/ or Binaries/ folders

5. **Test with small files first**
   - Use 30-second audio clips for quick iteration
   - Full songs take longer to process

### For the Standalone Build

1. **After re-packaging, always re-copy DLLs**
   ```powershell
   Copy-Item "...\ThirdParty\beatsync\lib\x64\*.dll" `
     "C:\Users\samue\Desktop\TripSitterBuild\Windows\MyProject\Binaries\Win64\"
   ```

2. **Test standalone build separately from UE Editor**
   - UE Editor and standalone may behave differently
   - Always verify both work

3. **Keep FFmpeg accessible**
   - Either set `BEATSYNC_FFMPEG_PATH` environment variable
   - Or ensure FFmpeg is in system PATH
   - Or place ffmpeg.exe in the app folder

### For Users (Distribution)

When sharing the app with others:

1. **Include all required DLLs:**
   - beatsync_backend_shared.dll
   - avcodec-62.dll, avformat-62.dll, avutil-60.dll, etc.
   - onnxruntime.dll, onnxruntime_providers_shared.dll
   - swresample-6.dll, swscale-9.dll

2. **Include FFmpeg or document requirement:**
   - Either bundle FFmpeg with the app
   - Or provide installation instructions

**Important: Do not bundle Python for BeatNet**
- BeatNet Python integration is **opt-in only** and should not be included in packaged releases. For development, enable python invocation with CMake option `ENABLE_BEATNET_PYTHON=ON` and set `BEATSYNC_ENABLE_PYTHON=1` when running Python integration tests locally.

3. **Test on a clean machine:**
   - Install on a PC without dev tools
   - Make sure all dependencies are bundled

---

## Next Steps (Development Roadmap)

### Immediate (Should Fix Now)

1. **Video duration validation**
   - Before processing, check if audio length exceeds video length
   - Show warning: "Video is shorter than audio. Output will be truncated."
   - Or: Loop the video source to fill the audio duration

2. **Add audio to output**
   - Currently output video has no audio
   - Need to mux the original audio track into the final video
   - Use `VideoWriter::addAudioTrack()` after concatenation

3. **Better error messages**
   - Instead of "Video processing failed", show specific error
   - "Beat time 120.5s exceeds video duration 60.0s"

### Short Term (Next Features)

4. **Apply effect regions to video**
   - User draws effect regions on waveform
   - These should actually apply to the output video
   - Pass effect regions to VideoWriter

5. **Preview before export**
   - Show a 5-second preview of the output
   - Helps user verify settings before full export

6. **Progress detail**
   - Show "Extracting segment 5/120..."
   - Show "Concatenating..."
   - Show ETA

### Medium Term (Polish)

7. **Project save/load**
   - Save beat analysis to file
   - Save effect regions
   - Resume work later

8. **Undo/redo for effect regions**
   - Ctrl+Z to undo last region change

9. **Keyboard shortcuts**
   - Space = play/pause preview
   - +/- = zoom waveform
   - Delete = remove selected effect region

### Long Term (Future Versions)

10. **GPU-accelerated preview**
    - Real-time preview using UE rendering

11. **Transition effects**
    - Fade, wipe, dissolve between clips

12. **macOS support**
    - ARM64 (Apple Silicon) build
    - Universal binary

---

## Recommended Test Files

For consistent testing, use these types of files:

**Audio (good for testing):**
- EDM/electronic music (clear beats)
- 120-140 BPM typical
- 30 seconds to 2 minutes for quick tests
- MP3 or WAV format

**Video (good for testing):**
- 1080p resolution (matches default output)
- At least 2 minutes long
- Variety of scenes (helps see where cuts happen)
- MP4 with H.264 codec (most compatible)

**Multi-clip folder:**
- 3-5 different video clips
- Each at least 30 seconds
- Named alphabetically (clip_01.mp4, clip_02.mp4, etc.)
- Similar resolution/framerate for best results

---

## Debugging Checklist

When something doesn't work, go through this checklist:

```
□ Is the DLL in the right place?
  → Check: Plugins\TripSitterUE\ThirdParty\beatsync\lib\x64\beatsync_backend_shared.dll

□ Are all dependency DLLs present?
  → Check: avcodec, avformat, avutil, swresample, swscale, onnxruntime

□ Is FFmpeg installed and accessible?
  → Test: Open PowerShell, type "ffmpeg -version"
  → If not found, set BEATSYNC_FFMPEG_PATH environment variable

□ Are file paths valid?
  → No special characters in path
  → Path exists and is accessible
  → Try moving files to simple path like C:\test\

□ Is the video long enough?
  → Video duration should be >= audio duration
  → Or use multi-clip mode with enough total video

□ Check the log files
  → C:\Users\samue\AppData\Local\Temp\beatsync_ffmpeg_extract.log
  → C:\Users\samue\AppData\Local\Temp\beatsync_ffmpeg_concat.log

□ Try a different audio/video file
  → Rule out file-specific issues

□ Restart the app
  → Sometimes fixes DLL loading issues

□ Rebuild the DLL
  → If you changed C++ code, always rebuild
```
