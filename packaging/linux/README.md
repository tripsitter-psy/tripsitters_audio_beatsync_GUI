# Linux packaging

Three distributable formats are produced from **one staged bundle**:

| Output | File | Runs on |
|---|---|---|
| Portable tarball | `build/linux-dist/MTVTripSitter-<ver>-linux-x86_64.tar.xz` | any x86_64 distro with glibc ≥ 2.35 (Ubuntu 22.04+, Debian 12+, Fedora 36+, openSUSE Leap 15.5+, Arch) |
| AppImage | `build/linux-dist/MTVTripSitter-<ver>-x86_64.AppImage` | same as the tarball, single file, needs FUSE (or `--appimage-extract-and-run`) |
| Flatpak | `build/linux-dist/MTVTripSitter-<ver>-x86_64.flatpak` | any distro with Flatpak + Flathub (runtime `org.freedesktop.Platform//25.08`) |

```bash
packaging/linux/package.sh              # everything: backend -> stage -> tarball + AppImage + Flatpak
packaging/linux/package.sh --skip-backend --no-flatpak   # reuse the last backend build, skip Flatpak
```

Or step by step: `build-backend.sh` → `stage.sh` → `make-tarball.sh` / `make-appimage.sh` / `make-flatpak.sh`.

## How it fits together

```
build-backend.sh   podman/docker, ubuntu:22.04  ->  build/linux-portable/stage/
                   libbeatsync_backend_shared.so, beatsync CLI, libaudioflux, libgomp, libsamplerate
                   (RPATH $ORIGIN/lib so nothing is looked up on the host)
stage.sh           ->  build/linux-dist/AppDir/
                   Engine/Binaries/Linux/TripSitter          from $UE_ROOT (built with UBT, glibc 2.28 floor)
                   Engine/Binaries/Linux/lib/                 FFmpeg 8 (BtbN GPL shared), ONNX Runtime 1.23 + CUDA provider,
                                                              AudioFlux, libgomp, libsamplerate [, cuda/]
                   Engine/Binaries/Linux/bin/                 ffmpeg, ffprobe, beatsync
                   Engine/Binaries/Linux/{models,Resources}/  ONNX models, fonts, images
                   Engine/{Content/Slate,Content/Internationalization,Config,Shaders}   Unreal runtime data
                   Engine/Programs/TripSitter/Content/Splash/Icon.bmp                   window icon
                   AppRun, .desktop, metainfo, icon, licenses/
make-*.sh          wrap the AppDir
```

Why the split: the Unreal GUI must be built by UnrealBuildTool from a UE source tree (cannot
happen inside flatpak-builder, and the UE EULA forbids redistributing that source), so all three
packages ship the prebuilt binary. The backend *is* rebuilt from source, in an Ubuntu 22.04
container, purely to pin its glibc/libstdc++ requirement low enough for other distributions
(a Fedora-built `.so` needs glibc 2.43). The Flatpak is therefore not Flathub-eligible; it is for
a self-hosted repo or the single-file bundle.

## One-time inputs

```bash
cd thirdparty
# FFmpeg 8 shared build with x264/x265/NVENC (GPL, same source as the Windows package)
curl -LO https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz
tar xJf ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz
# ONNX Runtime GPU tarball (see BUILD.md for the lib64/include fixups)
# AudioFlux source with the GCC patch from BUILD.md applied
```

Tools on the host: `podman` (or `docker`), `xz`, ImageMagick (`magick`, for the icon), and for the
Flatpak either the `flatpak-builder` package or `flatpak install flathub org.flatpak.Builder`, plus
`flatpak install flathub org.freedesktop.Sdk//25.08`. `appimagetool` is downloaded into `tools/`
automatically.

Environment overrides live in `common.sh`: `UE_ROOT`, `APP_VERSION`, `FFMPEG_ROOT`, `ORT_ROOT`,
`CUDA_LIB_DIRS`.

## GPU acceleration

By default the packages are **CPU-only for inference** (NVENC encoding still works: FFmpeg dlopens
the driver's `libnvidia-encode`). ONNX Runtime's CUDA provider needs cuBLAS, cuFFT, cuRAND, cudart
and cuDNN 9, about 2.3 GB uncompressed, so they are opt-in:

```bash
CUDA_LIB_DIRS="$HOME/cuda-12.8/lib64:/usr/local/cudnn-9.10.2/lib" packaging/linux/stage.sh
```

`stage.sh` follows the provider's `NEEDED` chain into those directories, copies what it finds into
`Engine/Binaries/Linux/lib/cuda/`, and sets `$ORIGIN` RPATHs. The user then only needs the NVIDIA
driver (535+). In the Flatpak the driver comes from `org.freedesktop.Platform.GL.nvidia-*`, which
Flatpak installs automatically to match the host driver. TensorRT is not bundled; the
TensorRT → CUDA → CPU fallback in the backend handles that.

## Runtime behaviour worth knowing

* `AppRun` passes `-SaveToUserDir`, so Unreal logs/config go to `~/.config/Epic/TripSitter/Saved/`
  instead of the (read-only) bundle. The backend writes its FFmpeg diagnostics to the working
  directory, so `AppRun` starts in `~/.local/state/tripsitter/logs/`.
* `BEATSYNC_FFMPEG_PATH` is set to the bundled `ffmpeg`; export it yourself to override.
* The Flatpak uses `--filesystem=host` because the file pickers are Unreal's own (no portal
  support), and `--device=all` for `/dev/nvidia*`.
* Wayland: dropdowns render in-window (see CLAUDE.md). GNOME does not show per-window icons, it
  matches the app id `TripSitter` against `StartupWMClass` in the desktop file; KDE/X11 use
  `Icon.bmp`. The Slate-drawn title bar icon comes from `Resources/TitleIcon.png`.

## Testing a package

```bash
./build/linux-dist/MTVTripSitter-1.0.0-x86_64.AppImage
tar xf build/linux-dist/MTVTripSitter-1.0.0-linux-x86_64.tar.xz && MTVTripSitter-1.0.0-linux-x86_64/TripSitter.sh
flatpak install --user build/linux-dist/MTVTripSitter-1.0.0-x86_64.flatpak && flatpak run io.github.tripsitter_psy.TripSitter
```

To check portability of a library: `objdump -T lib.so | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1`.
