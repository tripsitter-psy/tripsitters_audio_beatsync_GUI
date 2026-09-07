#!/usr/bin/env bash
# Assemble the self-contained bundle (AppDir layout) that the tarball, AppImage and
# Flatpak are all made from.
#
#   packaging/linux/stage.sh
#
# Requires: build/linux-portable/stage (from build-backend.sh) and a UE tree with a
# built Engine/Binaries/Linux/TripSitter (UE_ROOT). Set CUDA_LIB_DIRS to bundle the
# CUDA runtime for GPU inference (see common.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGE="$PORTABLE_OUT/stage"
UE_BIN="$UE_ROOT/Engine/Binaries/Linux"
[ -f "$STAGE/libbeatsync_backend_shared.so" ] || die "Run build-backend.sh first ($STAGE missing)"
[ -x "$UE_BIN/TripSitter" ] || die "TripSitter binary not found at $UE_BIN/TripSitter (set UE_ROOT)"
[ -d "$UE_ROOT/Engine/Content/Slate" ] || die "Engine/Content/Slate missing under $UE_ROOT"

# patchelf: use the host copy if present, otherwise the build container.
run_patchelf() {
    if command -v patchelf >/dev/null; then patchelf "$@"; return; fi
    local args=() a
    for a in "$@"; do args+=("${a/#$APPDIR/\/appdir}"); done
    "$CONTAINER_TOOL" run --rm -v "$APPDIR:/appdir:Z" "$CONTAINER_IMAGE" patchelf "${args[@]}"
}

log "Staging bundle into $APPDIR"
rm -rf "$APPDIR"
BIN="$APPDIR/Engine/Binaries/Linux"
mkdir -p "$BIN/lib" "$BIN/bin" "$BIN/models" "$BIN/Resources" \
         "$APPDIR/Engine/Content" "$APPDIR/Engine/Shaders" "$APPDIR/licenses" \
         "$APPDIR/usr/share/applications" "$APPDIR/usr/share/metainfo" \
         "$APPDIR/usr/share/icons/hicolor/512x512/apps" "$APPDIR/usr/bin"

# --- Unreal program + runtime data -------------------------------------------
cp "$UE_BIN/TripSitter" "$BIN/"
cp -r "$UE_ROOT/Engine/Content/Slate" "$APPDIR/Engine/Content/"
mkdir -p "$APPDIR/Engine/Content/SlateDebug/Fonts"
cp "$UE_ROOT/Engine/Content/SlateDebug/Fonts/LastResort.ttf" "$APPDIR/Engine/Content/SlateDebug/Fonts/" 2>/dev/null || true
# ICU data: the EFIGS subset (English/French/Italian/German/Spanish) is what shipped
# games use; the engine looks for Internationalization/icudt64l.
ICU_SRC="$UE_ROOT/Engine/Content/Internationalization"
mkdir -p "$APPDIR/Engine/Content/Internationalization"
if [ -d "$ICU_SRC/EFIGS/icudt64l" ]; then
    cp -r "$ICU_SRC/EFIGS/icudt64l" "$APPDIR/Engine/Content/Internationalization/"
else
    cp -r "$ICU_SRC/icudt64l" "$APPDIR/Engine/Content/Internationalization/"
fi
cp -r "$UE_ROOT/Engine/Config" "$APPDIR/Engine/"
cp -r "$UE_ROOT/Engine/Shaders/StandaloneRenderer" "$APPDIR/Engine/Shaders/"
# Marks the tree as an installed (redistributed) engine build; harmless for a Program.
mkdir -p "$APPDIR/Engine/Build"; : > "$APPDIR/Engine/Build/InstalledBuild.txt"
# Window/taskbar icon: LinuxWindow.cpp loads <ProjectDir>/Content/Splash/Icon.bmp,
# which for a Program target is Engine/Programs/TripSitter/Content/Splash/Icon.bmp.
mkdir -p "$APPDIR/Engine/Programs/TripSitter/Content/Splash"
cp "$REPO_ROOT/unreal-prototype/Source/TripSitter/Resources/Icon.bmp" "$APPDIR/Engine/Programs/TripSitter/Content/Splash/Icon.bmp"

# --- Backend, CLI, third-party runtime libraries ------------------------------
cp "$STAGE/libbeatsync_backend_shared.so" "$BIN/"
cp "$STAGE/bin/beatsync" "$BIN/bin/"
cp "$STAGE"/lib/* "$BIN/lib/"

# FFmpeg (BtbN GPL shared build): CLI has RPATH $ORIGIN/../lib already.
cp "$FFMPEG_ROOT/bin/ffmpeg" "$FFMPEG_ROOT/bin/ffprobe" "$BIN/bin/"
for f in "$FFMPEG_ROOT"/lib/*.so.*.*.*; do
    real="$(basename "$f")"; cp "$f" "$BIN/lib/$real"
    soname="${real%.*.*}"; ln -sf "$real" "$BIN/lib/$soname"
done
cp "$FFMPEG_ROOT/LICENSE.txt" "$APPDIR/licenses/FFmpeg-LICENSE.txt" 2>/dev/null || true

# ONNX Runtime + GPU providers (TensorRT provider omitted: TensorRT is not bundled).
cp "$ORT_ROOT/lib/libonnxruntime.so.1.23.2" "$BIN/lib/"
ln -sf libonnxruntime.so.1.23.2 "$BIN/lib/libonnxruntime.so.1"
cp "$ORT_ROOT/lib/libonnxruntime_providers_shared.so" "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so" "$BIN/lib/"
cp "$ORT_ROOT/LICENSE" "$APPDIR/licenses/onnxruntime-LICENSE.txt" 2>/dev/null || true
cp "$REPO_ROOT/thirdparty/audioFlux/LICENSE.md" "$APPDIR/licenses/audioFlux-LICENSE.md" 2>/dev/null || true

# --- Optional CUDA runtime ---------------------------------------------------------
if [ -n "$CUDA_LIB_DIRS" ]; then
    log "Bundling CUDA runtime from $CUDA_LIB_DIRS"
    mkdir -p "$BIN/lib/cuda"
    IFS=: read -r -a cuda_dirs <<< "$CUDA_LIB_DIRS"
    find_cuda_lib() { local d; for d in "${cuda_dirs[@]}"; do [ -e "$d/$1" ] && { echo "$d/$1"; return; }; done; return 1; }
    # Start from the CUDA provider's direct dependencies and follow NEEDED entries
    # that resolve inside CUDA_LIB_DIRS. cuDNN sub-libraries are dlopen'd by name,
    # so every libcudnn_*.so.9 is included explicitly.
    queue=($(readelf -d "$BIN/lib/libonnxruntime_providers_cuda.so" | sed -n 's/.*NEEDED.*\[\(.*\)\]/\1/p'))
    for d in "${cuda_dirs[@]}"; do for f in "$d"/libcudnn*.so.9; do [ -e "$f" ] && queue+=("$(basename "$f")"); done; done
    declare -A seen
    while [ ${#queue[@]} -gt 0 ]; do
        name="${queue[0]}"; queue=("${queue[@]:1}")
        [ -n "${seen[$name]:-}" ] && continue; seen[$name]=1
        src="$(find_cuda_lib "$name")" || continue
        cp -L "$src" "$BIN/lib/cuda/$name"
        queue+=($(readelf -d "$BIN/lib/cuda/$name" | sed -n 's/.*NEEDED.*\[\(.*\)\]/\1/p'))
    done
    for f in "$BIN"/lib/cuda/*.so*; do run_patchelf --set-rpath '$ORIGIN' "$f"; done
    ls "$BIN/lib/cuda" | sed 's/^/    /'
fi
# The CUDA provider ships without an RPATH; make it find cuBLAS/cuDNN next to it.
run_patchelf --set-rpath '$ORIGIN:$ORIGIN/cuda' "$BIN/lib/libonnxruntime_providers_cuda.so"

# --- Models, resources, launcher, desktop integration --------------------------
for m in beatnet.onnx demucs.onnx htdemucs.onnx rife.onnx upscale.onnx upscale_2x.onnx; do
    [ -e "$REPO_ROOT/models/$m" ] && cp -L "$REPO_ROOT/models/$m" "$BIN/models/"
done
cp "$REPO_ROOT"/unreal-prototype/Source/TripSitter/Resources/{Corpta.otf,TitleHeader.png,wallpaper.png,icon.png,TitleIcon.png} "$BIN/Resources/"
cp "$REPO_ROOT/LICENSE" "$APPDIR/licenses/LICENSE"
cp "$REPO_ROOT/THIRD_PARTY_LICENSES.md" "$APPDIR/licenses/" 2>/dev/null || true

install -m 755 "$PKG_DIR/AppRun" "$APPDIR/AppRun"
ln -sf ../../AppRun "$APPDIR/usr/bin/tripsitter"
sed "s/@APP_ID@/$APP_ID/g" "$PKG_DIR/tripsitter.desktop" > "$APPDIR/usr/share/applications/$APP_ID.desktop"
ln -sf "usr/share/applications/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
sed -e "s/@APP_ID@/$APP_ID/g" -e "s/@APP_VERSION@/$APP_VERSION/g" -e "s/@DATE@/$(date +%F)/g" \
    "$PKG_DIR/tripsitter.metainfo.xml" > "$APPDIR/usr/share/metainfo/$APP_ID.metainfo.xml"
# Square 512x512 icon with transparent padding from the (non-square) source art.
magick "$REPO_ROOT/unreal-prototype/Source/TripSitter/Resources/icon.png" -background none -gravity center \
    -resize 512x512 -extent 512x512 "$APPDIR/usr/share/icons/hicolor/512x512/apps/$APP_ID.png"
ln -sf "usr/share/icons/hicolor/512x512/apps/$APP_ID.png" "$APPDIR/$APP_ID.png"
ln -sf "$APP_ID.png" "$APPDIR/.DirIcon"

chmod -R u+rwX,go+rX "$APPDIR"
log "Bundle staged: $(du -sh "$APPDIR" | cut -f1) in $APPDIR"
