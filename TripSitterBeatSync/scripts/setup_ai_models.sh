#!/bin/bash
# TripSitter BeatSync - AI Model Setup Script
# This script exports BeatNet and Demucs models to ONNX format

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/ThirdParty/onnx_models"

echo "=========================================="
echo "TripSitter BeatSync - AI Model Setup"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed"
    echo "Install Python 3.8+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Export BeatNet model
echo ""
echo "=========================================="
echo "Exporting BeatNet model..."
echo "=========================================="
python3 "$SCRIPT_DIR/export_beatnet_onnx.py"
BEATNET_STATUS=$?

# Export Demucs model
echo ""
echo "=========================================="
echo "Exporting Demucs model..."
echo "=========================================="
python3 "$SCRIPT_DIR/export_demucs_onnx.py"
DEMUCS_STATUS=$?

# Summary
echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="

if [ -f "$OUTPUT_DIR/beatnet.onnx" ]; then
    echo "✓ BeatNet model: $OUTPUT_DIR/beatnet.onnx"
else
    echo "✗ BeatNet model: FAILED"
fi

if [ -f "$OUTPUT_DIR/demucs.onnx" ]; then
    echo "✓ Demucs model: $OUTPUT_DIR/demucs.onnx"
else
    echo "✗ Demucs model: FAILED"
fi

echo ""
echo "IMPORTANT: The exported models use random weights."
echo "For production use, you need to obtain pre-trained weights:"
echo ""
echo "BeatNet: https://github.com/mjhydri/BeatNet"
echo "Demucs: https://github.com/adefossez/demucs"
echo ""
echo "Next steps:"
echo "1. Rebuild the UE5 project"
echo "2. The app will automatically detect and load the ONNX models"
echo "3. Beat detection UI will show 'BeatNet AI' when AI is active"
echo ""

# Print model sizes
echo ""
echo "Model sizes:"
if [ -f "$OUTPUT_DIR/beatnet.onnx" ]; then
    ls -lh "$OUTPUT_DIR/beatnet.onnx" | awk '{print "  BeatNet: " $5}'
fi
if [ -f "$OUTPUT_DIR/demucs.onnx" ]; then
    ls -lh "$OUTPUT_DIR/demucs.onnx" | awk '{print "  Demucs:  " $5}'
fi

# Deactivate virtual environment
deactivate

exit 0
