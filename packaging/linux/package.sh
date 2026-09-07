#!/usr/bin/env bash
# One-shot Linux release: portable backend -> staged bundle -> tarball + AppImage + Flatpak.
#
#   packaging/linux/package.sh [--skip-backend] [--no-tarball] [--no-appimage] [--no-flatpak]
#
# Environment (see common.sh): UE_ROOT, CUDA_LIB_DIRS, APP_VERSION, FFMPEG_ROOT, ORT_ROOT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DO_BACKEND=1 DO_TARBALL=1 DO_APPIMAGE=1 DO_FLATPAK=1
for a in "$@"; do case "$a" in
    --skip-backend) DO_BACKEND=0;; --no-tarball) DO_TARBALL=0;;
    --no-appimage) DO_APPIMAGE=0;; --no-flatpak) DO_FLATPAK=0;;
    *) die "unknown option $a";; esac; done

[ "$DO_BACKEND" = 1 ] && "$PKG_DIR/build-backend.sh"
"$PKG_DIR/stage.sh"
[ "$DO_TARBALL" = 1 ]  && "$PKG_DIR/make-tarball.sh"
[ "$DO_APPIMAGE" = 1 ] && "$PKG_DIR/make-appimage.sh"
[ "$DO_FLATPAK" = 1 ]  && "$PKG_DIR/make-flatpak.sh"
log "Packages in $DIST_DIR:"
ls -lh "$DIST_DIR" | grep -E '\.(tar\.xz|AppImage|flatpak)$' || true
