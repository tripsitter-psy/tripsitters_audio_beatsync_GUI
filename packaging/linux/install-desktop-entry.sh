#!/bin/sh
# Adds (or removes, with --uninstall) a launcher menu entry for this extracted
# tarball. Only touches ~/.local/share; the application itself stays where it is.
set -e
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
APP_ID="$(basename "$HERE"/usr/share/applications/*.desktop .desktop)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/512x512/apps"
if [ "$1" = "--uninstall" ]; then
    rm -f "$APPS/$APP_ID.desktop" "$ICONS/$APP_ID.png"
    echo "Removed launcher entry."
    exit 0
fi
mkdir -p "$APPS" "$ICONS"
sed "s|^Exec=tripsitter|Exec=\"$HERE/TripSitter.sh\"|" "$HERE/usr/share/applications/$APP_ID.desktop" > "$APPS/$APP_ID.desktop"
cp "$HERE/usr/share/icons/hicolor/512x512/apps/$APP_ID.png" "$ICONS/$APP_ID.png"
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
echo "Installed launcher entry 'MTV TripSitter' -> $HERE/TripSitter.sh"
echo "Run '$0 --uninstall' to remove it."
