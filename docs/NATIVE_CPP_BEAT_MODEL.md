# Native C++ Beat Model Replacement — Plan & Notes

Goal: remove Python at runtime from beat detection packaging by providing a native solution. This can be done in two main ways (ordered by recommended approach):

1) ONNX-based (recommended for quickest, smallest runtime)
- Train or locate a small PyTorch model (e.g., TCN / BeatNet) for beat detection.
- Export to ONNX using a reproducible script (PyTorch -> ONNX). Target an ONNX opset in the modern, compatible range.
	- The current script `tools/convert_pytorch_to_onnx.py` defaults to **opset 12** for broader test and runtime compatibility.
	- If newer ONNX features are required, opset 15–18 is recommended. Note that ONNX Runtime v1.20+ supports up to opset 21, and v1.14+ supports opset 18.
- Validate the ONNX model with onnxruntime (CPU and, optionally, CUDA) in CI.
- Optionally quantize (INT8 / FP16) to reduce binary size and CPU/GPU memory pressure.
- Advantages: no Python at runtime; inference via onnxruntime C++ is straightforward and well-supported.

2) Native C++ inference (port model into C++ runtime)
- Option A: Use libtorch C++ API (heavy dependency; easier port of PyTorch models). Requires bundling libtorch in installers.
- Option B: Re-implement the model architecture in lightweight C++ (e.g., hand-coded TCN or simple ConvNet) and load serialized weights (custom loader). This minimizes dependencies but requires more engineering effort and careful numeric validation.
- Advantages: single-language build, fine-grained control over optimizations, smaller runtime if reimplemented carefully.

Checklist / Tasks for final packaging

- License check: ensure model weights and training data license (e.g., CC-BY) permit redistribution in packaged app.
- Repro script: Create `tools/convert_pytorch_to_onnx.py` to convert from PyTorch weights to ONNX and validate shape/outputs.
- CI: Add validation step to run a small inference on a test asset and assert expected behavior (beat times within tolerance).
- Packaging: Decide whether to ship ONNX model blob in repo (recommended) or generate it during release build via reproducible conversion.
- Fallbacks: Keep the high-quality spectral-flux C++ fallback for cases where model can't be used (no runtime, GPU missing, etc.).
- Metrics: Add unit tests and benchmarks to measure detection accuracy and latency. Define acceptable thresholds.
- Size/perf targets: Define max model size and inference time budgets for target platforms.

Model versioning and security
- Model versioning: Adopt a semantic or date-based versioning scheme for the ONNX model (e.g., model_version: 1.0.0 or 2026-01-19). Record model_version and model_metadata in a sidecar JSON or as ONNX metadata, and update `tools/convert_pytorch_to_onnx.py` to embed version info in the output artifact.
- Security/integrity: For each ONNX/model blob, compute and store a checksum (SHA256 or stronger) or cryptographic signature. Publish checksums/signatures alongside the model in the repo or release artifacts. In CI and release flows, verify model integrity before packaging or deployment. Document the verification process and include rollback/compatibility notes for model updates (e.g., maintain compatibility with previous model versions or provide a fallback if verification fails).

Testing and validation
- Unit tests that assert detection on small synthetic click-tracks (existing `tests/test_spectral_flux.cpp` demonstrates approach).
- Regression tests comparing C++ ONNX inference outputs against a known Python reference.

Maintenance notes
- Prefer shipping an ONNX model + conversion script in `tools/` for reproducibility.
- Pin ONNX opset and onnxruntime versions in CI and document the supported runtime versions in `DEVELOPMENT_CONTEXT.md`.

If you'd like, I can: convert/commit a small pre-trained model (if licensing permits), add the conversion script, or draft CI steps to run ONNX validation on PRs.
