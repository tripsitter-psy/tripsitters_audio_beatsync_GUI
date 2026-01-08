macOS build quickstart

Requirements
- macOS 12+ (or newer supported by UE5.3)
- Xcode + Command Line Tools
- CMake (3.20+), Ninja, pkg-config (install via Homebrew: `brew install cmake ninja pkg-config`)
- FFmpeg (recommended: shared build). Install via Homebrew: `brew install ffmpeg`
- Optional: vcpkg (if you prefer vcpkg-managed deps)

Build steps (core backend + GUI)

1. Clone repository and create build directory:

   ```bash
   git clone <repo>
   cd BeatSyncEditor
   mkdir build && cd build
   ```

2. Configure (example using system ffmpeg):

   ```bash
   cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBEATSYNC_BUILD_SHARED=ON
   ```

   If you use vcpkg:

   ```bash
   cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release -DBEATSYNC_BUILD_SHARED=ON
   ```

3. Build:

   ```bash
   cmake --build . --config Release -j$(sysctl -n hw.ncpu)
   ```

4. Run the backend smoke test (verifies FFmpeg path resolution and minimal run):

   ```bash
   ./bin/Release/backend_smoke
   ```

Notes
- The shared library will be copied to `unreal-prototype/ThirdParty/beatsync/lib/Mac/libbeatsync_backend.dylib` for the Unreal plugin to pick up.
- If you want to run the Unreal Editor automation test on macOS, you will need a self-hosted machine with UE5.3 installed; use the Editor automation command provided in `unreal-prototype/README.md` and point to the macOS UE5Editor-Cmd app (typically under `/Applications/Epic Games/UE_5.3/Engine/Binaries/Mac/UE5Editor-Cmd.app/Contents/MacOS/UE5Editor-Cmd`).
- Ensure that any FFmpeg dylibs that your app needs are either installed system-wide or copied into your app bundle (Contents/Frameworks) when packaging.