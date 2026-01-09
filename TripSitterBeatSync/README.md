# TripSitter BeatSync - Unreal Engine 5.7

A psychedelic audio-synchronized video editing application built with Unreal Engine 5.7.

## Features

- **Beat Detection**: Automatic beat detection from audio files (MP3, WAV, FLAC, OGG)
- **BPM Estimation**: Accurate BPM calculation
- **Video Processing**: Cut and synchronize videos to detected beats
- **Beat Visualizer**: Interactive timeline showing beat positions
- **Video Preview**: Built-in video player with playback controls
- **Cross-Platform**: macOS and Windows support

## Prerequisites

- **Unreal Engine 5.7.1** (installed via Epic Games Launcher)
- **Xcode** (macOS) or **Visual Studio 2022** (Windows)
- **FFmpeg** (for audio/video processing)
  - macOS: `brew install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html

## Setup

### 1. Build the Backend Library

The BeatSync backend (audio analysis & video processing) needs to be built first:

```bash
cd TripSitterBeatSync
./build_backend.sh
```

This will:
- Build the `libbeatsync_backend.dylib` shared library
- Copy it to `ThirdParty/beatsync/lib/Mac/`

### 2. Open in Unreal Engine

1. Right-click `TripSitterBeatSync.uproject`
2. Select "Open with Unreal Engine 5.7"
3. Wait for the project to compile

### 3. Create the UI

In the UE Editor:

1. Create a new Widget Blueprint: `Content/UI/WBP_MainMenu`
2. Add the `BeatSyncWidget` as the root
3. Design your UI layout using UMG

## Project Structure

```
TripSitterBeatSync/
├── Config/                     # UE config files
├── Content/
│   └── UI/
│       └── Textures/          # UI assets (wallpaper, header, etc.)
├── Plugins/
│   └── TripSitterUE/          # BeatSync plugin
│       └── Source/
│           └── TripSitterUE/
│               ├── Public/    # Headers
│               │   ├── BeatsyncLoader.h
│               │   ├── BeatsyncSubsystem.h
│               │   ├── BeatSyncWidget.h
│               │   ├── BeatVisualizerWidget.h
│               │   └── VideoPreviewWidget.h
│               └── Private/   # Implementation
├── Source/
│   └── TripSitterBeatSync/    # Game module
├── ThirdParty/
│   └── beatsync/
│       └── lib/
│           └── Mac/           # Backend library
└── TripSitterBeatSync.uproject
```

## Blueprint Usage

### BeatSyncWidget

The main widget for controlling BeatSync. Bind these functions to UI buttons:

- `BrowseAudioFile()` - Open audio file picker
- `BrowseVideoFolder()` - Select folder with video clips
- `BrowseOutputFolder()` - Select output destination
- `StartAnalysis()` - Analyze audio for beats
- `StartProcessing()` - Process videos with beat sync
- `CancelOperation()` - Cancel current operation

Properties to bind to UI:
- `SelectedAudioFile` - Current audio file path
- `SelectedVideoFolder` - Current video folder
- `CurrentBeatData` - Beat analysis results
- `CurrentProgress` - Operation progress (0-1)
- `StatusMessage` - Current status text
- `bIsProcessing` - Whether an operation is running

### BeatVisualizerWidget

Timeline visualization of detected beats:

- `SetBeatData(FBeatData)` - Set beats to display
- `SetPlaybackPosition(float)` - Update playhead position
- `SetVisibleRange(float, float)` - Set visible time range

### VideoPreviewWidget

Video playback with controls:

- `LoadVideo(FString)` - Load a video file
- `Play()` / `Pause()` / `Stop()` - Playback controls
- `SeekToTime(float)` - Seek to timestamp
- `GetCurrentTime()` / `GetDuration()` - Timing info

## Color Scheme (Psychedelic Theme)

| Element | Color | Hex |
|---------|-------|-----|
| Primary | Neon Cyan | #00D9FF |
| Secondary | Neon Purple | #8B00FF |
| Background | Dark Blue | #0A0A1A |
| Accent | Hot Pink | #FF0080 |
| Text | Light Blue | #C8DCFF |

## Troubleshooting

### "Beatsync library not found"

Run `./build_backend.sh` to build the backend library.

### "FFmpeg not found"

Install FFmpeg:
- macOS: `brew install ffmpeg`
- Windows: Download and add to PATH

### Build errors in Xcode

1. Right-click .uproject → Generate Xcode Project
2. Open the generated .xcworkspace
3. Build the "TripSitterBeatSyncEditor" scheme

## License

MIT License - see LICENSE file
