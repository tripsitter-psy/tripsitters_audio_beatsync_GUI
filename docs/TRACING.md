# Tracing & Observability (OpenTelemetry)

This repository includes optional OpenTelemetry tracing support for the backend and a minimal UE plugin integration.

## Quick start (local)

1. Install OpenTelemetry C++ SDK (recommended via vcpkg):
   - vcpkg install opentelemetry-cpp
     - Example for Windows: `vcpkg install opentelemetry-cpp:x64-windows`
     - Example for Linux: `vcpkg install opentelemetry-cpp:x64-linux`
      - Example for macOS (Intel): `vcpkg install opentelemetry-cpp:x64-osx`
      - Example for macOS (Apple Silicon): `vcpkg install opentelemetry-cpp:arm64-osx`
   - By default, vcpkg uses the triplet set by your environment or toolchain file. You can override the triplet by appending `:<triplet>` to the package name, or by setting the `VCPKG_DEFAULT_TRIPLET` environment variable.
   - If using vcpkg toolchain, configure CMake with `-DBEATSYNC_ENABLE_TRACING=ON` and ensure you use the matching vcpkg toolchain and triplet for your platform.

2. Start a local OTLP collector + Jaeger (for viewing traces):
   - From project root:
     - Windows (PowerShell): `tools\tracing\start-tracing-collector.ps1`
     - macOS/Linux: `sh tools/tracing/start-tracing-collector.sh`
   - Collector endpoints: OTLP gRPC: `localhost:4317`, OTLP HTTP: `localhost:4318`.
   - Jaeger UI: http://localhost:16686

3. Build with tracing enabled:
   - cmake -S . -B build -DBEATSYNC_ENABLE_TRACING=ON
   - cmake --build build --config Release --target beatsync_backend_shared

4. In Unreal plugin, the `TripSitterUE` module calls `bs_initialize_tracing("tripsitter")` on startup when the backend DLL is present. You can also call `bs_initialize_tracing` manually.

5. View traces in Jaeger or use the AI Toolkit trace viewer in VS Code:
   - Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P) and run the command `AI Toolkit: Open Trace Viewer` (`ai-mlstudio.tracing.open`).
   - Point the viewer to `http://localhost:4318`.
   - For more details, see the [AI Toolkit extension documentation](https://marketplace.visualstudio.com/items?itemName=ms-ai-tools.ai-toolkit).

## Notes & recommendations

- When tracing is disabled or the SDK is not discovered, the tracing APIs are no-ops and have minimal runtime overhead.
- The collector configuration included (`tools/tracing/otel-collector-config.yaml`) receives OTLP traces and forwards them to Jaeger and logs.
- For CI, consider running the collector in a container and harvesting trace output as part of smoke tests.

## Files added

- `src/backend/tracing.h/.cpp` - tracing init/shutdown helpers
- Uses the project tracing API in `src/tracing/Tracing.h` - prefer `TRACE_FUNC()`/`TRACE_SCOPE(name)` and `::BeatSync::tracing::Span`.
- `src/backend/beatsync_capi.h/.cpp` - added `bs_initialize_tracing`, `bs_shutdown_tracing`, and lightweight span helpers for C API callers
- `unreal-prototype/Plugins/TripSitterUE/Public/BeatsyncLoader.h` and `unreal-prototype/Plugins/TripSitterUE/Private/BeatsyncLoader.cpp` - added wrappers for span helpers and tracing related exports
- `unreal-prototype/Plugins/TripSitterUE/Private/TripSitterUEModule.*` - calls backend tracing init/shutdown on module startup/shutdown
- `tools/tracing/docker-compose.yml` and helper scripts - start OTLP collector + Jaeger

## Future Work

- Add more spans to backend hot paths (currently several key functions are instrumented)
- Add CI smoke test that verifies a trace is emitted when running a small analyze+cut workflow
- Integrate AI Toolkit trace viewer commands into the repository docs