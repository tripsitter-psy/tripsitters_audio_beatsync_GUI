# Patreon post: TripSitter downloads

Copy the section below into a Patreon post (set it to the tier you want, or public).
Replace the `<link>` placeholders with your download links (GitHub Releases, Google
Drive, etc.). Keep the file names as they are so the instructions match.

---

## 🎬 MTV TripSitter — download & install

TripSitter cuts your video footage to the beat of a track. Drop in a tune and some
clips, pick how you want it cut, hit **Start Sync**, and it hands you a finished
beat-synced video. Beat detection is tuned for psytrance/EDM (it locks onto the kick),
with AI stem separation, dynamic cut density that follows the energy of the track,
slow-mo speed ramps with AI frame interpolation, AI upscaling of low-res clips, and
beat-driven effects.

### Downloads

**Linux** (any recent 64-bit distro — Ubuntu 22.04+, Debian 12+, Fedora, Arch, openSUSE, Mint, Pop!_OS…)

- **Flatpak** (recommended): `MTVTripSitter-1.0.0-x86_64.flatpak` — <link>
- **AppImage** (no install, one file): `MTVTripSitter-1.0.0-x86_64.AppImage` — <link>
- **Portable folder**: `MTVTripSitter-1.0.0-linux-x86_64.tar.xz` — <link>

**Windows 10/11**

- Installer: `MTVTripSitter-1.0.0-Windows-AMD64.exe` — <link>
- Portable: `MTVTripSitter-1.0.0-Windows-AMD64.zip` — <link>

### How to install

**Flatpak** — needs Flatpak with Flathub set up (Fedora, Pop!_OS, Mint, elementary
have it out of the box; on Ubuntu run `sudo apt install flatpak` once and add Flathub
from https://flathub.org/setup). Then:

```
flatpak install MTVTripSitter-1.0.0-x86_64.flatpak
flatpak run io.github.tripsitter_psy.TripSitter
```

It also shows up in your app menu as **MTV TripSitter**. NVIDIA users: Flatpak
fetches the matching driver package automatically.

**AppImage** — make it executable and run it:

```
chmod +x MTVTripSitter-1.0.0-x86_64.AppImage
./MTVTripSitter-1.0.0-x86_64.AppImage
```

(Right-click → Properties → "Allow executing as program" also works. If it complains
about FUSE, run it with `--appimage-extract-and-run`.)

**Portable folder** — extract the archive anywhere and run `TripSitter.sh`.
`install-desktop-entry.sh` inside the folder adds it to your app menu.

**Windows** — run the installer, or unzip the portable build and start
`Engine\Binaries\Win64\TripSitter.exe`.

### What you need

- 64-bit x86 machine, 8 GB RAM or more, a few GB of free disk for temp files while rendering.
- Any GPU with working OpenGL drivers for the app itself.
- **NVIDIA GPU (GTX 10-series or newer, driver 535+)** for the fast path: video
  encoding uses NVENC, and the AI features (stem separation, RIFE slow-mo, upscaling)
  run on the GPU. Without an NVIDIA card everything still works on the CPU, it's just
  slower — expect AI upscaling and stem separation to take a while.
- Linux build tested on Fedora 44 and Ubuntu 22.04 on both Wayland and X11.

### Quick start

1. **Select Audio** → your track.
2. **Select Video(s)** → one clip, or a folder of clips to cycle through.
3. Choose an analysis mode (**Stems + Flux** gives the best kick detection for psy),
   press **Analyze** and check the beat markers on the waveform. You can add/move/delete
   them by hand.
4. Pick output resolution / FPS, and optionally Dynamic Sync, speed ramps, upscaling
   and effects.
5. **Start Sync**. The status line shows the current stage and the time remaining.

Rendering time depends a lot on what you enable: plain beat cuts are fast (a few
minutes), AI upscaling of 4K footage or RIFE slow-mo can take an hour or more.

### Known limitations

- The AI features need an NVIDIA GPU to be fast; on AMD/Intel or without a GPU they
  run on the CPU and take many times longer.
- File dialogs are the app's own, so on Flatpak the app has full home-folder access.
- No Mac build yet.

### Problems?

Logs are in `~/.config/Epic/TripSitter/Saved/Logs/TripSitter.log` (Linux) or
`Engine\Programs\TripSitter\Saved\Logs\` inside the install folder (Windows). Post them in the comments or
open an issue at https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI/issues
together with what you clicked and what happened.

Thanks for supporting the project 💜
