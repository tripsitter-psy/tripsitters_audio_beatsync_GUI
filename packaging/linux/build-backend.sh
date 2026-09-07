#!/usr/bin/env bash
# Build the portable backend (libbeatsync_backend_shared.so + beatsync CLI) in an
# Ubuntu 22.04 container so it runs on any glibc >= 2.35 distribution.
#
#   packaging/linux/build-backend.sh            # build image if needed, then build
#   packaging/linux/build-backend.sh --rebuild-image
#
# Inputs (downloaded once, see README.md):
#   thirdparty/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1/   BtbN FFmpeg shared build
#   thirdparty/onnxruntime-linux-x64-gpu-1.23.2/            ONNX Runtime GPU tarball
#   thirdparty/audioFlux/                                   AudioFlux source (GCC patch applied)
# Output: build/linux-portable/stage/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -n "$CONTAINER_TOOL" ] || die "podman or docker is required"
[ -d "$FFMPEG_ROOT/lib" ] || die "FFmpeg not found at $FFMPEG_ROOT (see README.md)"
[ -d "$ORT_ROOT/lib" ]    || die "ONNX Runtime not found at $ORT_ROOT (see README.md)"
[ -d "$REPO_ROOT/thirdparty/audioFlux/src" ] || die "AudioFlux source missing in thirdparty/audioFlux"

if [ "${1:-}" = "--rebuild-image" ] || ! "$CONTAINER_TOOL" image exists "$CONTAINER_IMAGE" 2>/dev/null; then
    log "Building container image $CONTAINER_IMAGE"
    "$CONTAINER_TOOL" build -t "$CONTAINER_IMAGE" -f "$PKG_DIR/Containerfile" "$PKG_DIR"
fi

mkdir -p "$PORTABLE_OUT"
# Third-party trees may live outside the repo; mount them at stable paths.
rel() { python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$2"; }
MOUNTS=(-v "$REPO_ROOT:/src:Z")
FF_IN="/src/$(rel "$FFMPEG_ROOT" "$REPO_ROOT")"; case "$FF_IN" in /src/..*) MOUNTS+=(-v "$FFMPEG_ROOT:/ffmpeg:Z"); FF_IN=/ffmpeg;; esac
ORT_IN="/src/$(rel "$ORT_ROOT" "$REPO_ROOT")";  case "$ORT_IN" in /src/..*) MOUNTS+=(-v "$ORT_ROOT:/onnxruntime:Z"); ORT_IN=/onnxruntime;; esac

USERNS=()
[ "$(basename "$CONTAINER_TOOL")" = podman ] && USERNS=(--userns=keep-id)

log "Building backend in $CONTAINER_IMAGE"
"$CONTAINER_TOOL" run --rm "${USERNS[@]}" "${MOUNTS[@]}" \
    -e FFMPEG_ROOT="$FF_IN" -e ORT_ROOT="$ORT_IN" \
    -e BEATSYNC_VERSION="$APP_VERSION" -e JOBS="${JOBS:-$(nproc)}" \
    "$CONTAINER_IMAGE" bash /src/packaging/linux/container-build.sh
log "Portable backend ready in $PORTABLE_OUT/stage"
