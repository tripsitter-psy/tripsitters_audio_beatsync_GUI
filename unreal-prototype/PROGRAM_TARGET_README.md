# TripSitter Program Target Setup

This document explains how to build TripSitter as an **Unreal Engine Program**.

## Overview

TripSitter is designed to run as a standalone "Program" within the Unreal Engine ecosystem (similar to `UnrealFrontend` or `SlateViewer`). This means it does not use the standard "Game" loop but instead uses a lightweight application loop suitable for desktop tools.

## Build Instructions (Integration Method)

The standard way to build TripSitter is to integrate its source code into your Unreal Engine source tree.

### 1. Copy Source to Engine

Copy the `TripSitter` folder from `unreal-prototype/Source/` to your Engine's `Source/Programs/` directory.

```powershell
# Example PowerShell command
$UE_ROOT = "D:\UnrealEngine"
Copy-Item -Path "Source\TripSitter" -Destination "$UE_ROOT\Engine\Source\Programs\" -Recurse -Force
```

### 2. Build the Program

Use the Unreal Engine `Build.bat` script to compile the program.

```powershell
# Build for Windows 64-bit
& "$UE_ROOT\Engine\Build\BatchFiles\Build.bat" TripSitter Win64 Development
```

### 3. Run the Application

The built executable will be located in the Engine binaries folder:
```
$UE_ROOT/Engine/Binaries/Win64/TripSitter.exe
```

## Project Structure

(Note: The `.uproject` file in this directory is for standalone reference or legacy use. The primary build method involves source integration.)


```
Source/
├── TripSitter.Target.cs          # Program target configuration
└── TripSitter/
    ├── TripSitter.Build.cs       # Module build configuration
    ├── TripSitterMain.cpp        # Program entry point
    ├── TripSitterApp.h           # Main application class
    └── TripSitterApp.cpp         # Application implementation
```

## Key Differences from Game Target

| Game Target | Program Target |
|-------------|----------------|
| Full game world | No game world |
| Game viewport | Slate UI window |
| PlayerController/GameMode | Direct application control |
| Complex architecture | Simple desktop app |
| UMG (Unreal Motion Graphics)/Blueprint support | Slate UI only |

## Troubleshooting

### Build Errors
- Verify Unreal Engine 5.7 is installed
- Check that all dependencies are available

### Runtime Issues
- Make sure Slate UI is properly initialized
- Check that the application window is created successfully
- Verify that the main loop runs without errors

## Development Notes

- The application uses `FSlateApplication` for UI management
- Main window is created programmatically with Slate widgets
- The application runs its own message loop
- No game-specific systems are available (no UWorld, no AActor, etc.)