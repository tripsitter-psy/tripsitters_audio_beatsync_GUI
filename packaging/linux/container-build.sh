#!/usr/bin/env bash
# Runs INSIDE the Ubuntu 22.04 container (see Containerfile). Builds AudioFlux,
# the backend shared library and the beatsync CLI against the bundled FFmpeg and
# ONNX Runtime, then stamps portable RPATHs so everything resolves relative to
# the bundle. Invoked by build-backend.sh; do not call directly on the host.
set -euo pipefail

SRC=${SRC:-/src}
OUT=${OUT:-/src/build/linux-portable}
FFMPEG_ROOT=${FFMPEG_ROOT:?}
ORT_ROOT=${ORT_ROOT:?}
JOBS=${JOBS:-$(nproc)}

echo "== AudioFlux"
AF_SRC="$SRC/thirdparty/audioFlux"
AF_BUILD="$OUT/audioflux-build"
AF_ROOT="$OUT/audioflux-root"
cmake -S "$AF_SRC/src" -B "$AF_BUILD" -G Ninja \
    -DCMAKE_SYSTEM_NAME=linux -DCMAKE_BUILD_TYPE=Release
cmake --build "$AF_BUILD" -j"$JOBS"
rm -rf "$AF_ROOT"; mkdir -p "$AF_ROOT/lib"
cp -r "$AF_SRC/include" "$AF_ROOT/include"
cp "$AF_BUILD"/libaudioflux.so "$AF_ROOT/lib/"

echo "== Backend + CLI"
BUILD="$OUT/build"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    -DFFMPEG_ROOT="$FFMPEG_ROOT" \
    -DCMAKE_PREFIX_PATH="$ORT_ROOT" \
    -DAUDIOFLUX_ROOT="$AF_ROOT" \
    ${BEATSYNC_VERSION:+-DBEATSYNC_VERSION="$BEATSYNC_VERSION"}
cmake --build "$BUILD" -j"$JOBS" --target beatsync_backend_shared beatsync

echo "== Collect + patch RPATHs"
STAGE="$OUT/stage"
rm -rf "$STAGE"; mkdir -p "$STAGE/lib" "$STAGE/bin"
cp "$BUILD/libbeatsync_backend_shared.so" "$STAGE/"
cp "$BUILD/bin/Release/beatsync" "$STAGE/bin/"
cp "$AF_ROOT/lib/libaudioflux.so" "$STAGE/lib/"
# libgomp comes from the container's GCC; older hosts may lack a new enough copy.
cp -L "$(gcc -print-file-name=libgomp.so.1)" "$STAGE/lib/"
# libsamplerate (LGPL/BSD-2) from Ubuntu; bundled so it is not a host requirement.
cp -L /usr/lib/x86_64-linux-gnu/libsamplerate.so.0 "$STAGE/lib/"

# The backend and CLI find FFmpeg/ONNX/AudioFlux/gomp/samplerate in lib/ next to
# them. DT_RPATH (not RUNPATH) is used on purpose: it is inherited when
# libonnxruntime dlopen()s its providers, which have no RPATH of their own.
patchelf --remove-rpath "$STAGE/libbeatsync_backend_shared.so"
patchelf --force-rpath --set-rpath '$ORIGIN/lib:$ORIGIN/lib/cuda' "$STAGE/libbeatsync_backend_shared.so"
patchelf --remove-rpath "$STAGE/bin/beatsync"
patchelf --force-rpath --set-rpath '$ORIGIN/../lib:$ORIGIN/../lib/cuda' "$STAGE/bin/beatsync"
patchelf --set-rpath '$ORIGIN' "$STAGE/lib/libaudioflux.so"

echo "== Verify glibc/libstdc++ floor"
for f in "$STAGE/libbeatsync_backend_shared.so" "$STAGE/bin/beatsync" "$STAGE/lib/libaudioflux.so"; do
    printf '%-60s glibc<=%s  libstdc++<=%s\n' "$(basename "$f")" \
        "$(objdump -T "$f" | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)" \
        "$(objdump -T "$f" | grep -o 'GLIBCXX_[0-9.]*' | sort -Vu | tail -1)"
done
echo "== Done: $STAGE"
