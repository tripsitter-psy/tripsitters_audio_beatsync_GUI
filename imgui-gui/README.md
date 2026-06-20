# TripSitter ImGui frontend

A cross-platform [Dear ImGui](https://github.com/ocornut/imgui) frontend for the
BeatSync backend, as an alternative to the Unreal Engine 5 / Slate GUI. It reuses
the **entire** C++ backend unchanged by talking to the same `beatsync_capi.h` C
API — the app `dlopen`s `libbeatsync_backend_shared` at runtime (see
`src/BackendLoader.*`), so the GUI has zero compile-time dependency on the backend.

## Layout

| File | Purpose |
|------|---------|
| `src/main.cpp` | GLFW + OpenGL3 bootstrap, theme/font load, frame loop |
| `src/TripSitterApp.*` | UI sections, waveform rendering, threaded analysis/processing |
| `src/BackendLoader.*` | Portable `dlopen` binding of the `bs_*` C API |
| `src/Theme.h` | Neon palette ported from the Slate widget |
| `src/FileDialog.*` | Native file pickers (macOS done; Win/Linux stubbed) |
| `stage_mac.sh` | Copies backend + AudioFlux dylibs next to the binary, fixes rpath |

## Build (macOS)

Prereqs: `brew install cmake ninja ffmpeg onnxruntime libomp` and Xcode CLT.

### 1. Backend dylib (with AI + AudioFlux)

AudioFlux must be built once from source:

```bash
git clone --depth 1 https://github.com/libAudioFlux/audioFlux.git /tmp/audioFlux
# its CMake only defines the target for lowercase "darwin", and Apple clang needs
# libomp instead of bare -fopenmp (patch the darwin branch of src/CMakeLists.txt:
#   -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include  +  link libomp.dylib)
cmake -S /tmp/audioFlux/src -B /tmp/audioFlux/macOSBuild -DCMAKE_SYSTEM_NAME=darwin -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/audioFlux/macOSBuild --config Release
mkdir -p third_party/audioflux/lib
cp -R /tmp/audioFlux/include third_party/audioflux/include
cp /tmp/audioFlux/macOSBuild/libaudioflux.dylib third_party/audioflux/lib/
```

Then build the backend (run from the repo root):

```bash
cmake -S . -B build-mac -G Ninja -DCMAKE_BUILD_TYPE=Release -DBEATSYNC_BUILD_SHARED=ON \
  -DFFMPEG_ROOT="$(brew --prefix ffmpeg)" \
  -DCMAKE_PREFIX_PATH="$(brew --prefix onnxruntime)" \
  -DUSE_ONNX=ON -DUSE_AUDIOFLUX=ON -DAUDIOFLUX_ROOT="$(pwd)/third_party/audioflux" \
  -DBUILD_TESTS=OFF
cmake --build build-mac --target beatsync_backend_shared
```

(ONNX and AudioFlux are optional — the app falls back to Energy detection without them.)

### 2. GUI

```bash
cmake -S imgui-gui -B imgui-gui/build -G Ninja   # fetches ImGui + GLFW
cmake --build imgui-gui/build
./imgui-gui/stage_mac.sh                          # stage runtime dylibs beside the binary
./imgui-gui/build/TripSitterImGui
```

## Models

AI/Stems modes load ONNX models from `models/` (gitignored). The app searches
`models/`, `../models/`, `../../models/` upward from the working directory.
