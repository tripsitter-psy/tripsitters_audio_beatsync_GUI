#!/usr/bin/env python3
"""
Mock BeatNet script for testing.

This script simulates BeatNet output for integration tests.
It is gated by the BEATSYNC_TEST_MOCK_PYTHON environment variable.

Usage:
    BEATSYNC_TEST_MOCK_PYTHON=1 python mock_beatnet.py <audio_file>

Output:
    JSON with mock beat data to stdout
"""

import json
import sys
import os


def main() -> int:
    # Only run if explicitly enabled for testing
    if os.environ.get("BEATSYNC_TEST_MOCK_PYTHON") != "1":
        print(json.dumps({
            "error": "Mock script not enabled (set BEATSYNC_TEST_MOCK_PYTHON=1)",
            "beats": [],
            "bpm": None
        }))
        return 1

    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Usage: mock_beatnet.py <audio_file>",
            "beats": [],
            "bpm": None
        }))
        return 1

    audio_path = sys.argv[1]

    # Return mock beat data
    mock_result = {
        "beats": [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
        "bpm": 120.0,
        "downbeats": [0.25, 2.25],
        "backend": "mock",
        "audio": audio_path
    }

    print(json.dumps(mock_result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
