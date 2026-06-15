@echo off
REM rebuild_beatsync.bat - Double-click to rebuild BeatSync backend

setlocal enabledelayedexpansion

REM Change to script directory
cd /d "%~dp0"

echo.
echo === BeatSync Backend Rebuild ===
echo.
echo Running: powershell -NoProfile -ExecutionPolicy Bypass -File rebuild_beatsync.ps1 -Deploy
echo.

REM Run the PowerShell rebuild script with deployment
powershell -NoProfile -ExecutionPolicy Bypass -File "rebuild_beatsync.ps1" -Deploy

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo REBUILD FAILED - Press any key to close
    pause
    exit /b 1
) else (
    echo.
    echo REBUILD SUCCESSFUL - Press any key to close
    pause
    exit /b 0
)
