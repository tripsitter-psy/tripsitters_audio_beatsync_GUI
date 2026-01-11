#!/usr/bin/env python3
import sys
import json

# Simple fake BeatNet script: prints a JSON object with beats extracted from filename length
# Usage: beatnet_fake.py <audioFilePath>

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"beats": []}))
        sys.exit(0)
    path = sys.argv[1]
    # Generate some deterministic beat times based on hash of path
    seed = sum(ord(c) for c in path) % 10
    beats = [0.5 + i * 0.5 + seed * 0.01 for i in range(3)]
    out = {"beats": beats, "bpm": 120.0}
    print(json.dumps(out))

if __name__ == '__main__':
    main()
