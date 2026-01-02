@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Trip Sitter - wxWidgets Setup
echo ========================================
echo.

:: Check if wxWidgets is already installed
if exist "C:\wxWidgets-3.2.4" (
    echo [OK] wxWidgets found at C:\wxWidgets-3.2.4
    goto :build_wxwidgets
)

echo [1/3] Downloading wxWidgets 3.2.4...
echo This may take a few minutes...
echo.

:: Download wxWidgets
curl -L "https://github.com/wxWidgets/wxWidgets/releases/download/v3.2.4/wxWidgets-3.2.4.zip" -o "%TEMP%\wxWidgets.zip"

if errorlevel 1 (
    echo [ERROR] Failed to download wxWidgets
    echo Please download manually from: https://www.wxwidgets.org/downloads/
    exit /b 1
)

echo [2/3] Extracting wxWidgets...
powershell -command "Expand-Archive -Path '%TEMP%\wxWidgets.zip' -DestinationPath 'C:\wxWidgets-3.2.4' -Force"

if errorlevel 1 (
    echo [ERROR] Failed to extract wxWidgets
    exit /b 1
)

del "%TEMP%\wxWidgets.zip"

:build_wxwidgets
echo [3/3] Building wxWidgets (this will take 10-15 minutes)...
echo Please be patient...
echo.

cd /d "C:\wxWidgets-3.2.4\build\msw"

:: Build Release configuration
msbuild wx_vc17.sln /p:Configuration=Release /p:Platform=x64 /m

if errorlevel 1 (
    echo [ERROR] Failed to build wxWidgets
    echo Make sure you have Visual Studio 2022 installed
    exit /b 1
)

echo.
echo ========================================
echo   wxWidgets Setup Complete!
echo ========================================
echo.
echo Location: C:\wxWidgets-3.2.4
echo.
