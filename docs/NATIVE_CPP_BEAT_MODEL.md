# Native C++ Beat Model Replacement — Plan & Notes

Goal: remove Python at runtime from beat detection packaging by providing a native solution. This can be done in two main ways (ordered by recommended approach):

1) ONNX-based (recommended for quickest, smallest runtime)
- Train or locate a small PyTorch model (e.g., TCN / BeatNet) for beat detection.
- Export to ONNX using a reproducible script (PyTorch -> ONNX). Target an ONNX opset that our ONNX Runtime supports (opsset 12 is safe for older runtimes).
- Validate the ONNX model with onnxruntime (CPU and, optionally, CUDA) in CI.
- Optionally quantize (INT8 / FP16) to reduce binary size and CPU/GPU memory pressure.
- Advantages: no Python at runtime; inference via onnxruntime C++ is straightforward and well-supported.

2) Native C++ inference (port model into C++ runtime)
- Option A: Use libtorch C++ API (heavy dependency; easier port of PyTorch models). Requires bundling libtorch in installers.
- Option B: Re-implement the model architecture in lightweight C++ (e.g., hand-coded TCN or simple ConvNet) and load serialized weights (custom loader). This minimizes dependencies but requires more engineering effort and careful numeric validation.
- Advantages: single-language build, fine-grained control over optimizations, smaller runtime if reimplemented carefully.

Checklist / Tasks for final packaging
- License check: ensure model weights and training data license (e.g., CC-BY) permit redistribution in packaged app.
- Repro script: Create `tools/convert_model.py` to convert from PyTorch weights to ONNX and validate shape/outputs.
- CI: Add validation step to run a small inference on a test asset and assert expected behavior (beat times within tolerance).
- Packaging: Decide whether to ship ONNX model blob in repo (recommended) or generate it during release build via reproducible conversion.
- Fallbacks: Keep the high-quality spectral-flux C++ fallback for cases where model can't be used (no runtime, GPU missing, etc.).
- Metrics: Add unit tests and benchmarks to measure detection accuracy and latency. Define acceptable thresholds.
- Size/perf targets: Define max model size and inference time budgets for target platforms.

Testing and validation
- Unit tests that assert detection on small synthetic click-tracks (existing `tests/test_spectral_flux.cpp` demonstrates approach).
- Regression tests comparing C+++ONNX inference outputs against a known Python reference.

Maintenance notes
- Prefer shipping an ONNX model + conversion script in `tools/` for reproducibility.
- Pin ONNX opset and onnxruntime versions in CI and document the supported runtime versions in `DEVELOPMENT_CONTEXT.md`.

If you'd like, I can: convert/commit a small pre-trained model (if licensing permits), add the conversion script, or draft CI steps to run ONNX validation on PRs.
