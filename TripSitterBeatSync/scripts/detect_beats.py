#!/usr/bin/env python3
"""
Beat detection using librosa - called by TripSitter BeatSync
Outputs JSON with BPM and beat timestamps
"""

import sys
import json
import librosa
import numpy as np

def detect_beats(audio_path):
    """Detect beats and tempo from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(y) / sr

        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')

        # Convert tempo to float if it's an array
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        # Convert beat frames to timestamps
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Also detect onset strength for more precise beat locations
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Refine beats using dynamic programming beat tracker for better accuracy
        tempo_refined, beats_refined = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            units='time',
            tightness=100  # Higher = stricter tempo consistency
        )

        if hasattr(tempo_refined, '__len__'):
            tempo_refined = float(tempo_refined[0]) if len(tempo_refined) > 0 else tempo
        else:
            tempo_refined = float(tempo_refined)

        # Use the refined results
        result = {
            "success": True,
            "bpm": round(tempo_refined, 2),
            "duration": round(duration, 3),
            "beat_count": len(beats_refined),
            "beats": [round(float(t), 4) for t in beats_refined],
            "method": "librosa"
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "bpm": 0,
            "duration": 0,
            "beat_count": 0,
            "beats": []
        }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: detect_beats.py <audio_file>"}))
        return 1

    audio_path = sys.argv[1]
    result = detect_beats(audio_path)
    print(json.dumps(result))

    return 0 if result["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
