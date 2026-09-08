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
# --appimage-extract-and-run: works without FUSE on the build machine.
# xz squashfs: with the CUDA runtime bundled a zstd image is ~2.15 GB, just over
# GitHub's 2 GiB release-asset limit; xz brings it to ~1.7 GB (slower first start).
ARCH=x86_64 "$APPIMAGETOOL" --appimage-extract-and-run --comp xz -n "$APPDIR" "$OUT"
log "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
