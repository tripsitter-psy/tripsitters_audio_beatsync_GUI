@echo off
echo ========================================
echo   Quick Rebuild - TripSitter GUI
echo ========================================
echo.

REM Use Developer Command Prompt to rebuild
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -no_logo

cd /d "%~dp0build"

echo Building TripSitter.exe...
msbuild TripSitter.vcxproj /p:Configuration=Release /v:minimal /nologo

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   Build Successful!
    echo ========================================
    echo.
    echo Executable: build\bin\Release\TripSitter.exe
    echo.
    echo Assets copied automatically during build.
    echo.
) else (
    echo.
    echo ========================================
    echo   Build Failed!
    echo ========================================
    echo.
)

pause
