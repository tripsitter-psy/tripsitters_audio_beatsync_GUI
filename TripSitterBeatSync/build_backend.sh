#!/bin/bash

# Build script for TripSitter BeatSync backend library
# This script builds the beatsync_backend shared library for use with Unreal Engine

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build_backend"
OUTPUT_DIR="$SCRIPT_DIR/ThirdParty/beatsync/lib/Mac"

echo "=== TripSitter BeatSync Backend Build ==="
echo "Project root: $PROJECT_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Output dir: $OUTPUT_DIR"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing via Homebrew..."
    brew install ffmpeg
fi

# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo "CMake not found. Installing via Homebrew..."
    brew install cmake
fi

# Configure
cd "$BUILD_DIR"
cmake "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBEATSYNC_BUILD_SHARED=ON \
    -DBUILD_GUI=OFF \
    -DBUILD_CLI=OFF \
    -DBUILD_TESTS=OFF \
    -DUSE_ESSENTIA=OFF \
    -DUSE_BEATNET=OFF \
    -DUSE_DEMUCS=OFF \
    -DUSE_GLTRANSITIONS=OFF \
    -DUSE_AUDIOREACTIVE=OFF \
    -DUSE_ONNX=OFF

# Build
cmake --build . --config Release --target beatsync_backend

# Copy library to output
if [ -f "$BUILD_DIR/lib/libbeatsync_backend.dylib" ]; then
    cp "$BUILD_DIR/lib/libbeatsync_backend.dylib" "$OUTPUT_DIR/"
    echo "Library copied to: $OUTPUT_DIR/libbeatsync_backend.dylib"
elif [ -f "$BUILD_DIR/libbeatsync_backend.dylib" ]; then
    cp "$BUILD_DIR/libbeatsync_backend.dylib" "$OUTPUT_DIR/"
    echo "Library copied to: $OUTPUT_DIR/libbeatsync_backend.dylib"
else
    echo "Warning: Could not find built library. Searching..."
    find "$BUILD_DIR" -name "*.dylib" -o -name "*.so" 2>/dev/null
fi

echo "=== Build Complete ==="
echo "You can now open TripSitterBeatSync.uproject in Unreal Engine 5.7"
