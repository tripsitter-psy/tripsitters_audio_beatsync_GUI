#!/usr/bin/env bash
set -euo pipefail
ID=${1:-corrupt_audio}
AUDIO_DUR=${2:-5}
VIDEO_DUR=${3:-5}
GEN_ARGS=${4:-"--corrupt-audio-header --corrupt-bytes 256"}
EXPECT_FAILURE=${EXPECT_FAILURE:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP="$ROOT/evaluation/tmp/$ID"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found; install and add to PATH" >&2; exit 2; }
command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe not found; install and add to PATH" >&2; exit 2; }

rm -rf "$TMP"
mkdir -p "$TMP"

echo "Running generator: python $ROOT/evaluation/generate_synthetic.py --outdir $TMP --id $ID --audio-duration $AUDIO_DUR --video-duration $VIDEO_DUR $GEN_ARGS"
python "$ROOT/evaluation/generate_synthetic.py" --outdir "$TMP" --id "$ID" --audio-duration $AUDIO_DUR --video-duration $VIDEO_DUR $GEN_ARGS || true

echo "Generated files:"; ls -l "$TMP" || true

AUDIO=$(ls "$TMP" | grep "^$ID" | grep -E '\.(wav|mp3|m4a)$' || true)
VIDEO=$(ls "$TMP" | grep "^$ID" | grep -E '\.(mp4|mov|m4a)$' || true)

if [ -n "$AUDIO" ]; then
  echo "\nffprobe (audio):"
  ffprobe -v error -show_streams "$TMP/$AUDIO" 2>&1 || true
else
  echo "No audio file found"
fi

if [ -n "$VIDEO" ]; then
  echo "\nffprobe (video):"
  ffprobe -v error -show_streams "$TMP/$VIDEO" 2>&1 || true
else
  echo "No video file found"
fi

OUT="$TMP/$ID.out.mp4"
BEATSYNC="$ROOT/build/bin/Release/beatsync"
PASS=0
if [ -x "$BEATSYNC" ] && [ -n "$VIDEO" ] && [ -n "$AUDIO" ]; then
  echo "\nRunning beatsync sync..."
  "$BEATSYNC" sync "$TMP/$VIDEO" "$TMP/$AUDIO" -o "$OUT" >"$TMP/beatsync.out" 2>"$TMP/beatsync.err" || true
  RC=$?
  echo "beatsync rc=$RC"
  echo "beatsync stderr:"; cat "$TMP/beatsync.err" || true
else
  echo "beatsync not found or missing media; skipping beatsync run"
  RC=127
fi

if [ -f "$OUT" ]; then
  echo "\nffprobe (output):"
  ffprobe -v error -show_streams "$OUT" 2>&1 || true
else
  echo "No output file produced"
fi

if [ -f "$ROOT/traces.jsonl" ]; then
  echo "\nLast traces.jsonl lines:"
  tail -n 40 "$ROOT/traces.jsonl" || true
else
  echo "No traces.jsonl found"
fi

# decide pass/fail
if [ "$EXPECT_FAILURE" -eq 1 ]; then
  if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
    PASS=1
  else
    # if out exists, check ffprobe streams
    if ffprobe -v error -show_streams "$OUT" >/dev/null 2>&1; then
      PASS=0
    else
      PASS=1
    fi
  fi
else
  if [ -f "$OUT" ] && ffprobe -v error -show_streams "$OUT" >/dev/null 2>&1; then
    PASS=1
  else
    PASS=0
  fi
fi

if [ $PASS -eq 1 ]; then
  echo "\nSanity check verdict: PASS"
  exit 0
else
  echo "\nSanity check verdict: FAIL"
  exit 2
fi
