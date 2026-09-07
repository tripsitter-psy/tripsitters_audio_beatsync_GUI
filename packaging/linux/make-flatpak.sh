#!/usr/bin/env bash
# Build the Flatpak from the staged bundle and export a single-file .flatpak.
#
#   packaging/linux/make-flatpak.sh            # build + bundle
#   packaging/linux/make-flatpak.sh --install  # ...and install it for the current user
#
# Needs flatpak-builder (system package, or `flatpak install flathub org.flatpak.Builder`)
# and the org.freedesktop.Sdk//25.08 runtime.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -x "$APPDIR/AppRun" ] || die "Run stage.sh first"

MANIFEST="$PKG_DIR/flatpak/$APP_ID.yml"
FP_BUILD="$DIST_DIR/flatpak-build"
FP_REPO="$DIST_DIR/flatpak-repo"
BUNDLE="$DIST_DIR/$APP_NAME-$APP_VERSION-x86_64.flatpak"

if command -v flatpak-builder >/dev/null; then
    BUILDER=(flatpak-builder)
elif flatpak info org.flatpak.Builder >/dev/null 2>&1; then
    BUILDER=(flatpak run org.flatpak.Builder)
else
    die "flatpak-builder not found: dnf/apt install flatpak-builder, or: flatpak install flathub org.flatpak.Builder"
fi
flatpak info org.freedesktop.Sdk//25.08 >/dev/null 2>&1 || \
    die "Missing SDK: flatpak install flathub org.freedesktop.Sdk//25.08"

log "Building Flatpak ($APP_ID $APP_VERSION)"
"${BUILDER[@]}" --force-clean --disable-rofiles-fuse --repo="$FP_REPO" \
    --default-branch=stable "$FP_BUILD" "$MANIFEST"

log "Exporting single-file bundle"
flatpak build-bundle --runtime-repo=https://flathub.org/repo/flathub.flatpakrepo \
    "$FP_REPO" "$BUNDLE" "$APP_ID" stable
log "Wrote $BUNDLE ($(du -h "$BUNDLE" | cut -f1))"

if [ "${1:-}" = "--install" ]; then
    log "Installing for current user"
    flatpak install --user -y "$BUNDLE"
    echo "Run with: flatpak run $APP_ID"
fi
