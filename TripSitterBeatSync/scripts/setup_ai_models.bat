@echo off
REM TripSitter BeatSync - AI Model Setup (Windows)
REM Exports BeatNet and Demucs models to ONNX format

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set OUTPUT_DIR=%PROJECT_DIR%\ThirdParty\onnx_models

echo ==========================================
echo TripSitter BeatSync - AI Model Setup
echo ==========================================
echo.

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Install Python 3.8+ from https://python.org
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Create virtual environment
set VENV_DIR=%SCRIPT_DIR%.venv
if not exist "%VENV_DIR%" (
    echo.
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Install requirements
echo.
echo Installing Python dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r "%SCRIPT_DIR%requirements.txt"

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Export BeatNet
echo.
echo ==========================================
echo Exporting BeatNet model...
echo ==========================================
python "%SCRIPT_DIR%export_beatnet_onnx.py"

REM Export Demucs
echo.
echo ==========================================
echo Exporting Demucs model...
echo ==========================================
python "%SCRIPT_DIR%export_demucs_onnx.py"

REM Summary
echo.
echo ==========================================
echo Setup Complete
echo ==========================================

if exist "%OUTPUT_DIR%\beatnet.onnx" (
    echo [OK] BeatNet model exported
) else (
    echo [FAIL] BeatNet model
)

if exist "%OUTPUT_DIR%\demucs.onnx" (
    echo [OK] Demucs model exported
) else (
    echo [FAIL] Demucs model
)

echo.
echo For NVIDIA GPU acceleration:
echo   1. Install CUDA Toolkit 11.8+
echo   2. Install cuDNN 8.6+
echo   3. UE5 NNERuntimeORT will auto-detect CUDA
echo.

call deactivate
endlocal
