#!/usr/bin/env bash
# Single-file AppImage from the staged bundle. Downloads appimagetool on first use.
#
#   packaging/linux/make-appimage.sh  ->  build/linux-dist/MTVTripSitter-<ver>-x86_64.AppImage
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -x "$APPDIR/AppRun" ] || die "Run stage.sh first"

TOOLS="$REPO_ROOT/tools"; mkdir -p "$TOOLS"
APPIMAGETOOL="$TOOLS/appimagetool-x86_64.AppImage"
if [ ! -x "$APPIMAGETOOL" ]; then
    log "Downloading appimagetool"
    curl -fL -o "$APPIMAGETOOL" \
        https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$APPIMAGETOOL"
fi

OUT="$DIST_DIR/$APP_NAME-$APP_VERSION-x86_64.AppImage"
rm -f "$OUT"
log "Building $OUT"

# A type-2 AppImage is the static runtime followed by a squashfs of the AppDir.
# The runtime only mounts zstd/zlib images, and appimagetool's defaults produce
# a ~2.15 GB image with the CUDA runtime on board, just over GitHub's 2 GiB
# release-asset limit. squashfs-tools in the build container lets us use zstd at
# its maximum level with 1 MB blocks, which is enough to get under the limit;
# appimagetool is the fallback when no container tool is available.
RUNTIME="$TOOLS/appimage-runtime-x86_64"
if [ -n "$CONTAINER_TOOL" ] && "$CONTAINER_TOOL" image exists "$CONTAINER_IMAGE" 2>/dev/null; then
    if [ ! -s "$RUNTIME" ]; then
        log "Downloading AppImage type-2 runtime"
        curl -fL -o "$RUNTIME" https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64
    fi
    SQUASH="$DIST_DIR/appimage.squashfs"
    rm -f "$SQUASH"
    USERNS=(); [ "$(basename "$CONTAINER_TOOL")" = podman ] && USERNS=(--userns=keep-id)
    "$CONTAINER_TOOL" run --rm "${USERNS[@]}" -v "$DIST_DIR:/dist:Z" "$CONTAINER_IMAGE" \
        mksquashfs /dist/AppDir /dist/appimage.squashfs -comp zstd -Xcompression-level 22 -b 1M \
            -noappend -no-xattrs -all-root -processors "$(nproc)" -quiet
    cat "$RUNTIME" "$SQUASH" > "$OUT"
    rm -f "$SQUASH"
    chmod +x "$OUT"
else
    # --appimage-extract-and-run: works without FUSE on the build machine.
    ARCH=x86_64 "$APPIMAGETOOL" --appimage-extract-and-run --comp zstd -n "$APPDIR" "$OUT"
fi
log "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
