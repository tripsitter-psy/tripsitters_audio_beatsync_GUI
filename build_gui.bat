@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Trip Sitter - Build GUI
echo ========================================
echo.

:: Set project directory
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

:: Check FFmpeg
if not exist "C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared" (
    echo [ERROR] FFmpeg not found at expected location
    echo Expected: C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared
    pause
    exit /b 1
)

:: Create build directory
if not exist "build_gui" mkdir build_gui

echo [1/3] Configuring CMake...
cd build_gui

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DFFMPEG_ROOT="C:/ffmpeg-dev/ffmpeg-master-latest-win64-gpl-shared"

if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    pause
    exit /b 1
)

echo.
echo [2/3] Building GUI (Release)...
cmake --build . --config Release

if errorlevel 1 (
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo.
echo [3/3] Copying assets...
if not exist "bin\Release\assets" mkdir "bin\Release\assets"
xcopy /Y /I "..\assets\*.*" "bin\Release\assets\"

echo.
echo ========================================
echo   Build Complete!
echo ========================================
echo.
echo Executable: %PROJECT_DIR%build_gui\bin\Release\TripSitter.exe
echo.
pause
