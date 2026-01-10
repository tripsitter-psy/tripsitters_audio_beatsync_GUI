# TripSitter BeatSync

Beat-synchronized video editor built with Unreal Engine 5.7. Analyzes audio for beats and cuts video clips to match.

## Current State (Jan 2026)

| Feature | Status | Notes |
|---------|--------|-------|
| Beat Detection | Working | AI (BeatNet) + native fallback |
| Stem Separation | Ready | DemucsLite ONNX |
| Video Processing | Working | FFmpeg concat + mux |
| Mac Build | Working | Shipping config |
| Windows Build | Ready | Needs testing |

## Quick Start

### Mac
```bash
# 1. Export AI models (optional)
cd scripts && ./setup_ai_models.sh

# 2. Build
/Users/Shared/Epic\ Games/UE_5.7/Engine/Build/BatchFiles/RunUAT.sh BuildCookRun \
  -project=$(pwd)/TripSitterBeatSync.uproject -platform=Mac \
  -clientconfig=Shipping -cook -stage -archive
```

### Windows
See [Windows Setup](#windows-setup) below.

## Project Structure

```
TripSitterBeatSync/
├── Plugins/TripSitterUE/Source/TripSitterUE/
│   ├── Private/
│   │   ├── BeatsyncSubsystem.cpp  # Audio analysis + video processing
│   │   └── ONNXInference.cpp      # BeatNet/Demucs AI inference
│   └── Public/
│       ├── BeatsyncSubsystem.h
│       └── ONNXInference.h
├── Source/TripSitterBeatSync/
│   ├── BeatSyncHUD.cpp            # Full UI (file dialogs, panels, waveform)
│   └── BeatSyncHUD.h
├── ThirdParty/onnx_models/        # AI models
│   ├── beatnet.onnx               # 7MB - beat detection
│   └── demucs.onnx                # 119MB - stem separation
├── scripts/
│   ├── setup_ai_models.sh         # Mac model export
│   ├── export_beatnet_onnx.py
│   └── export_demucs_onnx.py
└── TripSitterBeatSync.uproject
```

## Beat Detection

The app tries AI first, falls back to native:

1. **BeatNet AI** - ONNX neural network (if models present)
2. **Native C++** - Onset/energy detection (always available)

Debug info shows which method: `[BeatNet AI]` or `[Native]`

---

## Windows Setup

### Requirements
- Unreal Engine 5.7
- Visual Studio 2022 (with C++ game dev workload)
- FFmpeg in PATH
- Python 3.8+ (for model export)

### Build Steps
```batch
:: 1. Generate project files
right-click TripSitterBeatSync.uproject -> Generate Visual Studio project files

:: 2. Open in VS2022 and build Win64 Shipping

:: 3. Or use command line:
RunUAT.bat BuildCookRun -project="path\to\TripSitterBeatSync.uproject" ^
  -platform=Win64 -clientconfig=Shipping -cook -stage -archive
```

### ONNX GPU Acceleration (NVIDIA CUDA)

For GPU-accelerated AI inference with NVIDIA:

1. **Install CUDA Toolkit 11.8+**
   - Download: https://developer.nvidia.com/cuda-toolkit

2. **Install cuDNN 8.6+**
   - Download: https://developer.nvidia.com/cudnn
   - Copy to CUDA install directory

3. **Verify installation**
   ```batch
   nvcc --version
   nvidia-smi
   ```

UE5's NNERuntimeORT auto-detects acceleration:

| Provider | Requirements | Performance |
|----------|--------------|-------------|
| CPU | None | Baseline |
| DirectML | Windows 10+ | 2-3x faster |
| CUDA | NVIDIA GPU + CUDA 11.8 | 5-10x faster |

The `TripSitterUE.Build.cs` already includes NNERuntimeORT - no code changes needed.

### Windows Model Export
```batch
cd scripts
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python export_beatnet_onnx.py
python export_demucs_onnx.py
```

---

## Dependencies

| Dependency | Purpose | Install |
|------------|---------|---------|
| UE 5.7 | Engine | Epic Games Launcher |
| FFmpeg | Video processing | `brew install ffmpeg` / ffmpeg.org |
| Python 3.8+ | Model export | python.org |
| torch | ONNX export | `pip install torch` |

## Troubleshooting

### "FFmpeg not found"
- Mac: `brew install ffmpeg`
- Windows: Download from ffmpeg.org, add bin folder to PATH

### "ONNX models not found"
Run the setup script or manually place models in `ThirdParty/onnx_models/`

### Beat detection shows wrong BPM
The AI models use random weights. For production accuracy, obtain trained weights from:
- BeatNet: https://github.com/mjhydri/BeatNet
- Demucs: https://github.com/adefossez/demucs

### Windows build errors
- Ensure VS2022 has "Game development with C++" workload
- Run "Generate Visual Studio project files" from .uproject context menu
