#!/bin/bash
# macOS Build Script for TripSitter BeatSync
# This script configures and builds the project using Homebrew dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}TripSitter BeatSync - macOS Build Script${NC}"
echo "========================================"

# Detect architecture
ARCH=$(uname -m)
echo -e "${YELLOW}Detected architecture: ${ARCH}${NC}"

# Default build type
BUILD_TYPE=${1:-Release}
echo -e "${YELLOW}Build type: ${BUILD_TYPE}${NC}"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Error: Homebrew not found. Please install Homebrew first.${NC}"
    echo "Visit: https://brew.sh"
    exit 1
fi

# Check for required tools
echo "Checking for required tools..."
for tool in cmake ninja; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${YELLOW}Installing $tool via Homebrew...${NC}"
        brew install $tool
    else
        echo -e "${GREEN}✓ $tool found${NC}"
    fi
done

# Check for FFmpeg
if ! brew list ffmpeg &> /dev/null; then
    echo -e "${YELLOW}Installing FFmpeg via Homebrew...${NC}"
    brew install ffmpeg
else
    echo -e "${GREEN}✓ FFmpeg found${NC}"
fi

# Check for wxWidgets
if ! brew list wxwidgets &> /dev/null; then
    echo -e "${YELLOW}Installing wxWidgets via Homebrew...${NC}"
    brew install wxwidgets
else
    echo -e "${GREEN}✓ wxWidgets found${NC}"
fi

# Get Homebrew prefixes
FFMPEG_PREFIX=$(brew --prefix ffmpeg)
WXWIDGETS_PREFIX=$(brew --prefix wxwidgets)

echo ""
echo "Configuration:"
echo "  FFmpeg: ${FFMPEG_PREFIX}"
echo "  wxWidgets: ${WXWIDGETS_PREFIX}"
echo "  Architecture: ${ARCH}"
echo "  Build Type: ${BUILD_TYPE}"
echo ""

# Configure
echo -e "${GREEN}Configuring build...${NC}"
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_OSX_ARCHITECTURES=${ARCH} \
    -DFFMPEG_ROOT=${FFMPEG_PREFIX} \
    -DwxWidgets_ROOT_DIR=${WXWIDGETS_PREFIX}

# Build
echo -e "${GREEN}Building...${NC}"
cmake --build build --config ${BUILD_TYPE}

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Executables:"
echo "  CLI: ./build/bin/${BUILD_TYPE}/beatsync"
echo "  GUI: ./build/bin/${BUILD_TYPE}/TripSitter.app"
echo ""
echo "To run the GUI:"
echo "  ./build/bin/${BUILD_TYPE}/TripSitter.app/Contents/MacOS/TripSitter"
echo ""
echo "To package as DMG:"
echo "  pushd build && cpack -C ${BUILD_TYPE} && popd"
echo ""
