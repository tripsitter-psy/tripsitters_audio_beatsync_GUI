# MTV TripSitter — portable Linux build

Run `./TripSitter.sh`. Nothing needs to be installed; everything (Unreal runtime,
FFmpeg, ONNX Runtime, AI models) is inside this folder.

Optional: `./install-desktop-entry.sh` adds "MTV TripSitter" to your application
menu (`--uninstall` removes it again).

Requirements: x86_64 Linux with glibc 2.35 or newer (Ubuntu 22.04+, Debian 12+,
Fedora 36+, openSUSE Leap 15.5+, Arch), a working OpenGL driver, X11 or Wayland.

GPU acceleration: if this build ships `Engine/Binaries/Linux/lib/cuda/`, NVIDIA
GPUs are used automatically through the proprietary driver (535 or newer). Without
that folder, or without an NVIDIA driver, analysis runs on the CPU. NVENC video
encoding only needs the driver.

Logs live in `~/.config/Epic/TripSitter/Saved/Logs/` and `~/.local/state/tripsitter/logs/`.

The command-line tool is `Engine/Binaries/Linux/bin/beatsync` and the bundled
FFmpeg is `Engine/Binaries/Linux/bin/ffmpeg`. Set `BEATSYNC_FFMPEG_PATH` to use a
different FFmpeg. Licenses for bundled components are in `licenses/`.
