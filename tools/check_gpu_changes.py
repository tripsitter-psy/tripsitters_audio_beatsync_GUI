#!/usr/bin/env python3
"""
Check if the current branch contains GPU/CUDA/ONNX-related changes compared to a base ref.
Usage: python tools/check_gpu_changes.py --base main
Exit codes:
  0 = GPU-related changes detected (success = found what we're looking for)
  1 = No GPU-related changes found
Prints matched files and matching keywords.
"""
import sys
import argparse
import subprocess
import re

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='main', help='Base ref to diff against (default: main)')
args = parser.parse_args()
BASE = args.base

# Patterns to look for in file paths or in diffs
PATH_PATTERNS = [r"onnx", r"cuda", r"cudnn", r"ptx", r"onnxruntime", r"BEATSYNC_ONNX_USE_CUDA", r"onnxruntime_providers_cuda", r"\.cu$", r"\.cuh$"]
CONTENT_PATTERNS = [r"BEATSYNC_ONNX_USE_CUDA", r"onnxruntime", r"onnx", r"CUDA", r"cuDNN", r"onnxruntime_providers_cuda"]

# Gather changed files between base and HEAD
try:
    subprocess.run(['git', 'fetch', 'origin', BASE], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    pass

try:
    res = subprocess.run(['git', 'diff', '--name-only', f'origin/{BASE}...HEAD'], capture_output=True, text=True, check=False)
    files = [f for f in res.stdout.splitlines() if f.strip()]
except Exception as e:
    print('Error running git diff:', e)
    sys.exit(1)

matched = []
for f in files:
    for p in PATH_PATTERNS:
        if re.search(p, f, re.IGNORECASE):
            matched.append((f, f'path match: {p}'))
            break
    else:
        # If file is small text, inspect content diff
        try:
            diff = subprocess.run(['git', 'diff', f'origin/{BASE}...HEAD', '--', f], capture_output=True, text=True, check=False).stdout
            for p in CONTENT_PATTERNS:
                if re.search(p, diff, re.IGNORECASE):
                    matched.append((f, f'content match: {p}'))
                    break
        except Exception:
            pass

if not matched:
    print('No GPU/ONNX-related changes detected.')
    sys.exit(1)  # Exit 1 = no GPU changes found

print('Detected GPU/ONNX-related changes:')
for f, reason in matched:
    print(f'- {f} ({reason})')

# Suggest review label
print('\nIf these changes are intended to affect GPU behavior, please add label: test-cuda')
sys.exit(0)  # Exit 0 = GPU changes detected (success = found what we're looking for)
