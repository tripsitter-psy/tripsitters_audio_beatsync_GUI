Title: ONNX Runtime (v1.23.2 via vcpkg) - Heap corruption during simple inference on Windows (Exit code 0xc0000374)

Summary
-------
On Windows, ONNX Runtime v1.23.2 (installed via vcpkg) can cause a crash with exit code 0xc0000374 (heap corruption) when running a minimal inference that returns a constant tensor. The crash reproduces quickly (within 0.1-0.4s) and is observable with a small stub ONNX model.

Reproduction (minimal)
----------------------
Repository with repro: https://github.com/tripsitter-psy/tripsitters_audio_beatsync_GUI (branch: fix/onnx-double-free-regression-test)

Files to run:
- tests/onnx_inference_helper.cpp (calls OnnxBeatDetector::analyze repeatedly in a separate process)
- tests/models/beat_stub.onnx (tiny stub model that outputs constant 3-element tensor)

Steps:
1. Build the project (Windows / MSVC / x64):
   cmake -S . -B build -DUSE_TRACING=ON
   cmake --build build --config Debug -- /m
2. Run the helper process (isolated process test):
   cd build
   .\tests\Debug\test_onnx_inference_helper.exe
3. Observe crash: exit code 0xc0000374 (heap corruption) reported by the OS or via `ctest -R onnx_inference_helper -V`.

Notes & environment
-------------------
- ONNX Runtime version: 1.23.2 (from vcpkg baseline; see vcpkg_installed/x64-windows/share/onnxruntime/onnxruntimeConfigVersion.cmake -> set(PACKAGE_VERSION "1.23.2"))
- OS: Windows (developer tested on Windows 11)
- Compiler: MSVC / Visual Studio 2022
- vcpkg manifest: this repo uses vcpkg manifest-mode; the affected package is `onnxruntime`.

Collected artifacts
-------------------
- Small model: tests/models/beat_stub.onnx (included in repo)
- Regression test: tests/test_onnx_detector_regression.cpp and tests/onnx_inference_helper.cpp
- Example crash dump: (attach .dmp produced by ProcDump if available)
- Trace logs: beatsync-trace.log (if tracing enabled)

Suggestions for triage
----------------------
- Run the helper process under a debugger (WinDbg) and capture `!analyze -v` and the thread stack when the crash occurs.
- Try official Microsoft ONNX Runtime prebuilt binaries (download from releases) to see if the crash is specific to vcpkg build.
- If reproducible, bisect vcpkg/onnxruntime commits or compare builds between 1.18.x -> 1.23.2 to find the regression point.

I can attach a minimal .dmp and the exact ctest logs if helpful; please advise if you'd prefer a reduced repro that isolates provider (CPU vs CUDA) or a sanitized core dump.
