@echo off
echo Rebuilding TripSitter GUI...
echo.

cd /d "%~dp0build"

echo Opening Visual Studio solution...
echo You can rebuild by:
echo 1. Right-click "TripSitter" project
echo 2. Select "Rebuild"
echo.
echo Or press Ctrl+Shift+B to build all
echo.

start BeatSyncEditor.sln

echo.
echo Visual Studio opened. Please rebuild the solution.
pause
