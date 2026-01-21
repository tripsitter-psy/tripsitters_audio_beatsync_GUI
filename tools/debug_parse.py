"""
Parses and extracts beat timing data from a JSON file (e.g., ONNX model output) for debugging.

- Expects input: JSON file at 'tests/fixtures/test_audio.onnx.json' containing a "beats" array.
- Usage: Run directly to print the file's content and parsed beat values.
- Output: Prints the raw file content, the position of the "beats" array, and the extracted list of beat times.
- Exits with an error message if the file is missing or unreadable.
"""
import sys
import json
filename = 'tests/fixtures/test_audio.onnx.json'
try:
    with open(filename) as f:
        s = f.read()
except (FileNotFoundError, OSError) as e:
    print(f"Error: Could not open file '{filename}': {e}", file=sys.stderr)
    sys.exit(1)

try:
    data = json.loads(s)
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse JSON: {e}", file=sys.stderr)
    sys.exit(1)

if "beats" not in data:
    print("Error: 'beats' key not found in JSON.", file=sys.stderr)
    sys.exit(1)
if not isinstance(data["beats"], list):
    print("Error: 'beats' key is not a list.", file=sys.stderr)
    sys.exit(1)
try:
    beats2 = [float(x) for x in data["beats"]]
    print('beats2', beats2)
except Exception as e:
    print(f"Error: Failed to convert beats to float: {e}", file=sys.stderr)
    sys.exit(1)
