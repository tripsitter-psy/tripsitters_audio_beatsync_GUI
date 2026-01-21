#!/usr/bin/env bash
set -euo pipefail

# Build and run tracing smoke test locally (Linux/macOS)
BUILD_DIR=build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBEATSYNC_ENABLE_TRACING=ON -DBEATSYNC_ENABLE_TESTS=ON
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
