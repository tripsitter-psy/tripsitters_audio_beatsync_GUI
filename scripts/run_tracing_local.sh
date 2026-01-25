#!/usr/bin/env bash
set -euo pipefail

# Build and run tracing smoke test locally (Linux/macOS)
BUILD_DIR=build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Use Ninja if available, otherwise let CMake pick default (usually Unix Makefiles)
if command -v ninja >/dev/null 2>&1; then
  GENERATOR="-G Ninja"
else
  GENERATOR=""
fi

cmake .. $GENERATOR -DCMAKE_BUILD_TYPE=Debug -DBEATSYNC_ENABLE_TRACING=ON -DBEATSYNC_ENABLE_TESTS=ON
cmake --build . --config Debug
# Set trace output inside build dir so it is easy to find
export BEATSYNC_TRACE_OUT="$PWD/beatsync-trace.log"
rc=0
ctest -R tracing -V --output-on-failure || rc=$?

if [ -f "beatsync-trace.log" ]; then
  echo "Tracing output written to: $PWD/beatsync-trace.log"
else
  echo "No trace file found. Tracing may not have been enabled for the test run."
fi

# Exit with the ctest return code (0 = success, non-zero = some tests failed)
exit $rc
