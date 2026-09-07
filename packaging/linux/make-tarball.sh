#!/usr/bin/env bash
# Portable tarball: works on any x86_64 distro with glibc >= 2.35 (Ubuntu 22.04,
# Debian 12, Fedora 36, openSUSE Leap 15.5 and newer). Extract anywhere and run
# ./TripSitter.sh, or ./install-desktop-entry.sh to add a launcher menu entry.
#
#   packaging/linux/make-tarball.sh   ->  build/linux-dist/MTVTripSitter-<ver>-linux-x86_64.tar.xz
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -x "$APPDIR/AppRun" ] || die "Run stage.sh first"

NAME="$APP_NAME-$APP_VERSION-linux-x86_64"
WORK="$DIST_DIR/tarball"
rm -rf "$WORK"; mkdir -p "$WORK"
cp -a "$APPDIR" "$WORK/$NAME"
# AppImage-only bits are not needed in the tarball.
rm -f "$WORK/$NAME/.DirIcon" "$WORK/$NAME/$APP_ID.desktop" "$WORK/$NAME/$APP_ID.png"
mv "$WORK/$NAME/AppRun" "$WORK/$NAME/TripSitter.sh"
ln -sf ../../TripSitter.sh "$WORK/$NAME/usr/bin/tripsitter"
install -m 755 "$PKG_DIR/install-desktop-entry.sh" "$WORK/$NAME/install-desktop-entry.sh"
cp "$PKG_DIR/README-tarball.md" "$WORK/$NAME/README.md"

log "Compressing $NAME.tar.xz (this takes a few minutes)"
tar -C "$WORK" -cf - "$NAME" | xz -T0 -4 > "$DIST_DIR/$NAME.tar.xz"
rm -rf "$WORK"
log "Wrote $DIST_DIR/$NAME.tar.xz ($(du -h "$DIST_DIR/$NAME.tar.xz" | cut -f1))"
