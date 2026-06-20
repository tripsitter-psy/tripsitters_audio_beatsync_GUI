#!/bin/sh
# Stage the macOS runtime for the ImGui app: copy the backend + AudioFlux dylibs
# next to the GUI binary and ensure the backend can resolve @rpath/libaudioflux.
#
# Run after (re)building either the backend (build-mac/) or the GUI (imgui-gui/build/).
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/imgui-gui/build"
BACKEND="$ROOT/build-mac/libbeatsync_backend_shared.dylib"
AUDIOFLUX="$ROOT/third_party/audioflux/lib/libaudioflux.dylib"

[ -f "$BACKEND" ]   || { echo "Backend dylib not found: $BACKEND (build it first)"; exit 1; }
[ -d "$DEST" ]      || { echo "GUI build dir not found: $DEST (build the GUI first)"; exit 1; }

cp "$BACKEND" "$DEST/"
[ -f "$AUDIOFLUX" ] && cp "$AUDIOFLUX" "$DEST/"

# Ensure @rpath/libaudioflux.dylib resolves next to the backend dylib.
if ! otool -l "$DEST/libbeatsync_backend_shared.dylib" | grep -q "@loader_path"; then
    install_name_tool -add_rpath @loader_path "$DEST/libbeatsync_backend_shared.dylib"
fi

echo "Staged backend + AudioFlux into $DEST"
