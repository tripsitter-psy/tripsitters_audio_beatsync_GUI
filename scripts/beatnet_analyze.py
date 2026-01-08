#!/usr/bin/env python3
"""
Placeholder BeatNet bridge.

Expected behavior (future):
- Load an audio file path from argv
- Run BeatNet model to detect beats
- Print JSON to stdout: {"beats": [seconds...], "bpm": float}

Current behavior: prints a friendly stub message so callers know BeatNet is not yet wired.
"""
import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: beatnet_analyze.py <audio_file>", file=sys.stderr)
        return 1

    audio_path = sys.argv[1]
    payload = {
        "beats": [],
        "bpm": None,
        "warning": "BeatNet placeholder; no analysis run",
        "audio": audio_path,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
