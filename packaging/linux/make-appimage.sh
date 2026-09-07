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
# zstd squashfs keeps the 300 MB of ONNX models and 550 MB of libraries reasonable.
ARCH=x86_64 "$APPIMAGETOOL" --appimage-extract-and-run --comp zstd -n "$APPDIR" "$OUT"
log "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
