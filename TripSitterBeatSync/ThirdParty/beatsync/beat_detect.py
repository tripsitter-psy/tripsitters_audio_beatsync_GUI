#!/usr/bin/env python3
"""
TripSitter BeatSync - AI Audio Analysis
Uses librosa for professional beat detection and tempo estimation
"""

import sys
import json
import librosa
import numpy as np

def analyze_audio(filepath):
    """
    Analyze audio file for beat detection and tempo estimation.
    Returns JSON with beats, bpm, and duration.
    """
    try:
        # Load audio file
        y, sr = librosa.load(filepath, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Beat tracking using librosa's beat tracker
        # This uses onset strength and dynamic programming for accurate beat detection
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Convert to scalar if needed (librosa can return array)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        # Convert beat frames to timestamps
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # For more accurate tempo, use beat_track with prior
        # This helps lock to common tempos (like 140 BPM)
        if len(beat_times) > 2:
            # Calculate tempo from actual beat intervals
            intervals = np.diff(beat_times)
            if len(intervals) > 0:
                median_interval = np.median(intervals)
                measured_tempo = 60.0 / median_interval

                # If measured tempo is close to double/half of detected, adjust
                if 0.45 < tempo / measured_tempo < 0.55:
                    tempo = measured_tempo
                elif 1.8 < tempo / measured_tempo < 2.2:
                    tempo = measured_tempo

        result = {
            "success": True,
            "beats": beat_times.tolist(),
            "bpm": round(tempo, 2),
            "duration": round(duration, 3),
            "beat_count": len(beat_times)
        }

    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "beats": [],
            "bpm": 0,
            "duration": 0,
            "beat_count": 0
        }

    return result

def main():
    import os
    # Write debug log
    debug_path = os.path.expanduser("~/Desktop/beatsync_debug.txt")
    with open(debug_path, "w") as f:
        f.write(f"Args: {sys.argv}\n")
        if len(sys.argv) >= 2:
            f.write(f"File path: {sys.argv[1]}\n")
            f.write(f"File exists: {os.path.exists(sys.argv[1])}\n")

    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No file path provided"}))
        sys.exit(1)

    filepath = sys.argv[1]
    result = analyze_audio(filepath)

    # Append result to debug
    with open(debug_path, "a") as f:
        f.write(f"Result: {result}\n")

    print(json.dumps(result))

if __name__ == "__main__":
    main()
