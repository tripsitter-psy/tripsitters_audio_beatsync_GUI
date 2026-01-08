#!/usr/bin/env python3
"""
Placeholder Demucs wrapper for stem separation.

Expected behavior (future):
- Accept an input audio path and output directory
- Run Demucs v4 to separate stems (drums, bass, vocals, other)
- Emit JSON with stem file paths and progress updates

Current behavior: prints a stub message so callers know separation is not yet implemented.
"""
import json
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: demucs_separate.py <audio_file> <output_dir>", file=sys.stderr)
        return 1

    audio_path, out_dir = sys.argv[1], sys.argv[2]
    payload = {
        "stems": {},
        "warning": "Demucs placeholder; no stems produced",
        "audio": audio_path,
        "output": out_dir,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
