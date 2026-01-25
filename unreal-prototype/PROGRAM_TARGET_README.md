# TripSitter Program Target Setup

This document explains how to build and run TripSitter as a Program target instead of a Game target.

## What Changed

The project has been converted from a Game target to a Program target, which provides:

- **No game world**: Clean desktop application without viewport overhead
- **Native mouse cursor**: Works automatically through FSlateApplication
- **Simple architecture**: No PlayerController, GameMode, or complex game systems
- **Proper desktop app**: Similar to SlateViewer or UnrealFrontend

## Prerequisites

### Platform Compatibility
**Note: These instructions are for Windows.** The provided PowerShell script (`Setup-Engine-Symlink.ps1`), paths (e.g., `C:\Program Files\Epic Games\UE_5.7\Engine`), and the target binary path (`Binaries/Win64/TripSitter.exe`) are specific to Windows.
For macOS or Linux users, please check the repository root or relevant documentation for shell scripts (`.sh`) and appropriate Engine paths and build targets for your platform.

### 1. Engine Symlink Setup

For Program targets with installed engines, you need to create an Engine symlink:

```powershell
# Run PowerShell as Administrator
& ".\scripts\Setup-Engine-Symlink.ps1"
```

This creates a symlink from `Engine` -> `C:\Program Files\Epic Games\UE_5.7\Engine`

### 2. Build the Program Target

```powershell
# Generate project files
"C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\DotNET\UnrealBuildTool.exe" -projectfiles -project="TripSitter.uproject" -game -engine

# Build the program
"C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\DotNET\UnrealBuildTool.exe" TripSitter Win64 Development -Project="TripSitter.uproject"
```

### 3. Run the Application

The built executable will be in:
```
Binaries/Win64/TripSitter.exe
```

## Project Structure

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
- Ensure the Engine symlink is created correctly
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