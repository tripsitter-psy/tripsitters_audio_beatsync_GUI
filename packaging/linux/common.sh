# Shared settings for the Linux packaging scripts. Sourced, not executed.
# Override any of these in the environment.

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"

APP_ID="${APP_ID:-io.github.tripsitter_psy.TripSitter}"
APP_NAME="MTVTripSitter"
# Version comes from CMakeLists.txt (BEATSYNC_VERSION) unless overridden.
APP_VERSION="${APP_VERSION:-$(sed -n 's/^set(BEATSYNC_VERSION "\([0-9.]*\)".*/\1/p' "$REPO_ROOT/CMakeLists.txt")}"

# Unreal Engine source tree that built Engine/Binaries/Linux/TripSitter.
UE_ROOT="${UE_ROOT:-$HOME/UE5_Source/UnrealEngine}"

# Third-party prebuilt bundles (see BUILD.md / packaging/linux/README.md).
FFMPEG_ROOT="${FFMPEG_ROOT:-$REPO_ROOT/thirdparty/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1}"
ORT_ROOT="${ORT_ROOT:-$REPO_ROOT/thirdparty/onnxruntime-linux-x64-gpu-1.23.2}"

# Optional CUDA runtime libraries to bundle (cuBLAS, cuFFT, cuRAND, cudart, cuDNN).
# Leave empty for a CPU-only bundle (the app falls back to CPU automatically).
# Example: CUDA_LIB_DIRS="$HOME/cuda-12.8/lib64:/usr/local/cudnn-9.10.2/lib"
CUDA_LIB_DIRS="${CUDA_LIB_DIRS:-}"

BUILD_ROOT="$REPO_ROOT/build"
PORTABLE_OUT="$BUILD_ROOT/linux-portable"     # container build output
DIST_DIR="$BUILD_ROOT/linux-dist"             # staged bundle + packages
APPDIR="$DIST_DIR/AppDir"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-localhost/tripsitter-linux-build:ubuntu22.04}"
CONTAINER_TOOL="${CONTAINER_TOOL:-$(command -v podman || command -v docker || true)}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
