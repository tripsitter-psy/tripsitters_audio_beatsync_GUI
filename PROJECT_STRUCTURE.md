# TripSitter Project Structure Map

## Directory Locations

```
WORKING DIRECTORY (Source of Truth):
c:\Users\samue\Desktop\BeatSyncEditor\
├── src/                          # C++ backend source
│   ├── audio/                    # Beat detection, audio analysis
│   │   ├── OnnxBeatDetector.h    # BeatNet parameters
│   │   ├── AudioFluxBeatDetector.h/cpp  # Spectral flux parameters
│   │   └── SpectralFlux.h        # Basic flux fallback
│   └── backend/                  # C API (beatsync_capi.h/cpp)
├── build/                        # CMake build output
│   └── Release/
│       ├── beatsync_backend_shared.dll  # BUILT DLL
│       └── beatsync_backend_shared.lib  # BUILT LIB
├── unreal-prototype/
│   ├── Source/TripSitter/        # UE source files (EDIT THESE)
│   │   ├── Private/
│   │   │   ├── BeatsyncLoader.cpp
│   │   │   ├── BeatsyncProcessingTask.cpp  # Threshold config here!
│   │   │   ├── STripSitterMainWidget.cpp
│   │   │   └── SWaveformViewer.cpp
│   │   └── TripSitter.Build.cs
│   └── ThirdParty/beatsync/      # Copy of DLL for reference only
└── models/                       # ONNX models (beatnet.onnx, etc.)

UE5 SOURCE (Build Location):
D:\UnrealEngine\
├── Engine\Binaries\
│   ├── Win64\                    # RUNTIME location
│   │   ├── TripSitter.exe        # BUILT executable
│   │   ├── beatsync_backend_shared.dll  # MUST BE HERE for runtime
│   │   ├── models/               # ONNX models for AI detection
│   │   └── Resources/            # UI assets
│   └── ThirdParty\Beatsync\x64\  # LINK-TIME location
│       ├── beatsync_backend_shared.dll  # For UBT staging
│       └── beatsync_backend_shared.lib  # For linking
└── Engine\Source\Programs\TripSitter\  # UE source (SYNCED FROM Desktop)
    ├── Private\                  # Copied from unreal-prototype/Source/TripSitter/Private/
    └── TripSitter.Build.cs       # Copied from unreal-prototype/Source/TripSitter/
```

## Build Flow

### Step 1: Build Backend DLL (when src/ changes)
```powershell
cd c:\Users\samue\Desktop\BeatSyncEditor
cmake --build build --config Release --target beatsync_backend_shared
```
**Output**: `build/Release/beatsync_backend_shared.dll` and `.lib`

### Step 2: Deploy DLL (after backend build)
```powershell
# Copy to ThirdParty (for linking)
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'D:\UnrealEngine\Engine\Binaries\ThirdParty\Beatsync\x64\' -Force
Copy-Item 'build\Release\beatsync_backend_shared.lib' 'D:\UnrealEngine\Engine\Binaries\ThirdParty\Beatsync\x64\' -Force

# Copy to Win64 (for runtime)
Copy-Item 'build\Release\beatsync_backend_shared.dll' 'D:\UnrealEngine\Engine\Binaries\Win64\' -Force
```

### Step 3: Sync UE Source (when unreal-prototype/ changes)
```powershell
Copy-Item -Path 'c:\Users\samue\Desktop\BeatSyncEditor\unreal-prototype\Source\TripSitter\Private\*' -Destination 'D:\UnrealEngine\Engine\Source\Programs\TripSitter\Private\' -Recurse -Force
Copy-Item 'c:\Users\samue\Desktop\BeatSyncEditor\unreal-prototype\Source\TripSitter\TripSitter.Build.cs' 'D:\UnrealEngine\Engine\Source\Programs\TripSitter\' -Force
```

### Step 4: Build TripSitter.exe (after source sync)
```powershell
cd D:\UnrealEngine
Engine\Build\BatchFiles\Build.bat TripSitter Win64 Development
```
**Output**: `D:\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe`

### Step 5: Run Test
```powershell
D:\UnrealEngine\Engine\Binaries\Win64\TripSitter.exe
```

## Parameter Locations (Beat Detection)

### Backend Parameters (compiled into DLL):
- `src/audio/OnnxBeatDetector.h` - BeatNet thresholds
- `src/audio/AudioFluxBeatDetector.h` - Spectral flux thresholds
- `src/audio/SpectralFlux.h` - Basic flux parameters

### UI-to-Backend Config (in TripSitter.exe):
- `unreal-prototype/Source/TripSitter/Private/BeatsyncProcessingTask.cpp`
  - Line ~180: `AIConfig.BeatThreshold` and `AIConfig.DownbeatThreshold`
  - These values are passed to the DLL at runtime

## What Gets Used When

| Analysis Mode | Backend Code | UI Config |
|--------------|--------------|-----------|
| AI + Stem Separation | OnnxBeatDetector (BeatNet) | BeatsyncProcessingTask.cpp AIConfig |
| AI Only | OnnxBeatDetector (BeatNet) | BeatsyncProcessingTask.cpp AIConfig |
| Flux | AudioFluxBeatDetector | Uses defaults from header |
| Stems + Flux | AudioFluxBeatDetector + Demucs | Uses defaults from header |
| Energy | SpectralFlux | Uses defaults from header |

## Common Mistakes

1. **Forgetting to rebuild DLL** - Changes to src/ need DLL rebuild
2. **Forgetting to copy DLL to BOTH locations** - ThirdParty (link) AND Win64 (runtime)
3. **Forgetting to sync UE source** - Changes to unreal-prototype/ need sync + rebuild
4. **Old .lib file** - Causes heap corruption if DLL/lib mismatch
5. **Wrong FFmpeg DLLs** - Must be ~100MB from ThirdParty, not ~13MB from vcpkg
