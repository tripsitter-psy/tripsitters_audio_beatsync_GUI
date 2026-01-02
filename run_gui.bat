@echo off
setlocal

echo ========================================
echo   Trip Sitter - Audio Beat Sync GUI
echo ========================================
echo.

:: Set project directory
set PROJECT_DIR=%~dp0

:: Check if executable exists
if not exist "%PROJECT_DIR%build_gui\bin\Release\TripSitter.exe" (
    echo [ERROR] TripSitter.exe not found!
    echo Please run build_gui.bat first
    echo.
    pause
    exit /b 1
)

:: Add FFmpeg to PATH temporarily
set PATH=C:\ffmpeg-dev\ffmpeg-master-latest-win64-gpl-shared\bin;%PATH%

:: Launch GUI
cd /d "%PROJECT_DIR%build_gui\bin\Release"
start "" TripSitter.exe

exit
