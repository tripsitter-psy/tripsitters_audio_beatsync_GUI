#!/usr/bin/env bash
set -euo pipefail

# Portable way to get number of CPU cores (works on Linux and macOS)
if command -v nproc &>/dev/null; then
    NUM_JOBS=$(nproc)
elif command -v sysctl &>/dev/null && sysctl -n hw.ncpu &>/dev/null; then
    NUM_JOBS=$(sysctl -n hw.ncpu)
else
    NUM_JOBS=1
fi

OUT_VIDEO="build/tmp/out_gpu_test.mp4"
AUDIO="build/tmp/gpu_stress.wav"
DURATION=${1:-600}

mkdir -p build/tmp

if [ ! -f "$AUDIO" ]; then
  echo "Generating $AUDIO ($DURATION s)"
  python3 tools/generate_long_test_wav.py "$AUDIO" "$DURATION"
fi

GPU_LOG="build/gpu_log.csv"
rm -f "$GPU_LOG"

# Start background GPU logging every 10s
( while true; do nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits >> "$GPU_LOG" || true; sleep 10; done ) &
NVPID=$!
# Ensure the background GPU logger is cleaned up on exit/interrupt
trap 'kill -TERM "${NVPID}" 2>/dev/null || true; wait "${NVPID}" 2>/dev/null || true' EXIT INT TERM

STDOUT_LOG="build/long_video_stdout.log"
STDERR_LOG="build/long_video_stderr.log"
rm -f "$STDOUT_LOG" "$STDERR_LOG"

# Possible locations for beatsync binary
CANDIDATES=("build/bin/Release/beatsync" "build/bin/beatsync" "build/beatsync" "./build/bin/beatsync")
BEATSYNC=""
for c in "${CANDIDATES[@]}"; do
  if [ -x "$c" ]; then
    BEATSYNC="$c"
    break
  fi
done
if [ -z "$BEATSYNC" ]; then
  echo "beatsync executable not found in expected locations; building..."
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_ONNX=ON
  cmake --build build --config Release -j "$NUM_JOBS" --target beatsync
  for c in "${CANDIDATES[@]}"; do
    if [ -x "$c" ]; then
      BEATSYNC="$c"
      break
    fi
  done
fi

if [ -z "$BEATSYNC" ]; then
  echo "Could not locate beatsync binary after build" >&2
  kill $NVPID || true
  exit 2
fi

set +e
# Try several CLI forms
$BEATSYNC create "$OUT_VIDEO" "$AUDIO" --strategy downbeat --gpu > "$STDOUT_LOG" 2>"$STDERR_LOG"
RC=$?
if [ $RC -ne 0 ] || [ ! -f "$OUT_VIDEO" ]; then
  echo "First form failed (exit $RC), trying without --gpu"
  $BEATSYNC create "$OUT_VIDEO" "$AUDIO" --strategy downbeat >> "$STDOUT_LOG" 2>>"$STDERR_LOG"
  RC=$?
fi
if [ $RC -ne 0 ] || [ ! -f "$OUT_VIDEO" ]; then
  echo "Second form failed (exit $RC), trying without strategy"
  $BEATSYNC create "$OUT_VIDEO" "$AUDIO" >> "$STDOUT_LOG" 2>>"$STDERR_LOG"
  RC=$?
fi
set -e

# Stop GPU logging
kill $NVPID || true
wait $NVPID 2>/dev/null || true

echo "Test finished with exit code $RC"
echo "GPU log: $GPU_LOG"
echo "Stdout: $STDOUT_LOG"
echo "Stderr: $STDERR_LOG"

if [ $RC -ne 0 ]; then
  exit $RC
fi

exit 0
