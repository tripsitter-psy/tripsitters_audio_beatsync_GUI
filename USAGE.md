# BeatSync Editor - Usage Guide

## Quick Start

The executable is located at:
```
C:\Users\samue\Desktop\BeatSyncEditor\build\bin\Release\beatsync.exe
```

## Commands

### Analyze Audio File

Detect beats in an audio file and display the results:

```bash
cd C:\Users\samue\Desktop\BeatSyncEditor\build\bin\Release
beatsync.exe analyze path\to\your\song.mp3
```

### With Custom Sensitivity

Adjust beat detection sensitivity (0.0 = conservative, 1.0 = aggressive):

```bash
beatsync.exe analyze song.mp3 --sensitivity 0.7
```

## Example Output

```
========================================
BeatSync Audio Analyzer
========================================

Analyzing: song.mp3
Sensitivity: 0.5

Audio loaded: 180.5s, 44100 Hz, 7958400 samples
Detected 458 beats

========================================
Analysis Results
========================================

BeatGrid Information:
  Number of beats: 458
  BPM: 120.5
  Duration: 180.245 seconds
  Average interval: 0.498 seconds

Beat Timestamps (first 20):
  Beat   1:  0:  0.523 (0.523s)
  Beat   2:  0:  1.021 (1.021s)
  Beat   3:  0:  1.519 (1.519s)
  Beat   4:  0:  2.017 (2.017s)
  ...

========================================
Analysis complete!
========================================
```

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- FLAC (.flac)
- AAC (.aac, .m4a)
- OGG (.ogg)
- WMA (.wma)
- And many more (anything FFmpeg supports)

## Tips

### Finding the Right Sensitivity

- **0.3-0.4**: Best for music with clear, strong beats (EDM, hip-hop)
- **0.5** (default): Good general-purpose setting
- **0.6-0.7**: Better for music with subtle beats (jazz, classical)
- **0.8-0.9**: Detects very subtle beats (may have false positives)

### Performance

- Analysis typically runs 10-20x faster than real-time
- A 3-minute song analyzes in about 10-20 seconds
- Memory usage: ~40MB per minute of audio

### Troubleshooting

**"Could not open audio file"**
- Make sure the file path is correct
- Try using absolute paths instead of relative paths
- Ensure the file is not corrupted

**"No beats detected"**
- Try increasing sensitivity: `--sensitivity 0.7`
- Make sure the audio file actually contains music
- Check if the file is silence or very quiet

**Missing DLL errors**
- FFmpeg DLLs should be in the same directory as beatsync.exe
- If missing, copy them from: `C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared\bin\`

## Adding to PATH (Optional)

To run `beatsync` from anywhere:

1. Press Win + R, type `sysdm.cpl`, press Enter
2. Advanced tab → Environment Variables
3. Under System variables, find "Path", click Edit
4. Click New, add: `C:\Users\samue\Desktop\BeatSyncEditor\build\bin\Release`
5. Click OK on all dialogs

Now you can run from any directory:
```bash
beatsync analyze C:\Music\song.mp3
```

## What's Next?

Phase 1 is complete! The next phases will add:

**Phase 2 - Video Processing**
- Load and split video files
- Extract frames at beat timestamps
- Command: `beatsync split video.mp4 --at-beats`

**Phase 3 - Auto-Sync**
- Automatically cut video to match music beats
- Command: `beatsync create video.mp4 audio.mp3 --strategy downbeat`

**Phase 4 - GUI**
- Visual timeline editor
- Drag-and-drop interface
- Real-time preview
- Export with effects

## Need Help?

See the full documentation in `README.md` and `BUILD.md`.

## Examples

Analyze your favorite song:
```bash
beatsync analyze "C:\Music\favorite_song.mp3"
```

Check a high-energy track:
```bash
beatsync analyze "C:\Music\edm_track.mp3" --sensitivity 0.4
```

Analyze a subtle jazz piece:
```bash
beatsync analyze "C:\Music\jazz.mp3" --sensitivity 0.7
```
