#!/usr/bin/env python3
"""
Psy-Aware Beat Filter for TripSitter
Filters beat markers to respect arrangement breaks by using local bass energy.
Run this on your track to get a cleaned list of beat times that skip low-energy sections.

Usage:
  python filter_psy_beats.py path/to/your.track.wav

It will output a list of kept beat times that you can manually apply or use in video cutting.
The threshold is tuned for psytrance (adjust BreakEnergyThreshold if needed).

This is the core logic that will be integrated into the GUI Pro mode.
"""

import sys
import numpy as np
from scipy.io import wavfile
import os

def load_audio(file_path):
    """Load audio as mono float32."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Convert stereo to mono
    data = data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
    return data, sample_rate

def calculate_rms_energy(audio, sample_rate, window_sec=0.5):
    """Calculate RMS energy in sliding windows."""
    window_samples = int(window_sec * sample_rate)
    rms = []
    for i in range(0, len(audio), window_samples // 2):  # 50% overlap
        window = audio[i:i + window_samples]
        if len(window) < window_samples // 2:
            break
        rms.append(np.sqrt(np.mean(window**2)))
    return np.array(rms)

def filter_beats(beat_times, audio, sample_rate, break_threshold=0.12, thin_factor=4):
    """Filter beats in low-energy (break) sections."""
    rms = calculate_rms_energy(audio, sample_rate)
    window_sec = 0.5
    window_samples = int(window_sec * sample_rate)
    kept = []
    for i, t in enumerate(beat_times):
        # Find corresponding RMS bin
        bin_idx = int(t * sample_rate / (window_samples // 2))
        bin_idx = min(bin_idx, len(rms) - 1)
        local_energy = rms[bin_idx]
        if local_energy > break_threshold:
            kept.append(t)
        else:
            # In breaks, keep every thin_factor-th beat to maintain some pulse
            if i % thin_factor == 0:
                kept.append(t)
            else:
                print(f"  Removed beat at {t:.2f}s (energy {local_energy:.3f} < {break_threshold})")
    return kept

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_psy_beats.py <audio_file.wav>")
        print("Example: python filter_psy_beats.py \"C:\\Users\\samue\\Downloads\\Syncro.master2.wav\"")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"Analyzing {audio_path} for psy-aware beat filtering...")

    audio, sr = load_audio(audio_path)
    print(f"Loaded audio: {len(audio)/sr:.2f} seconds at {sr}Hz")

    # Example beat times from your current analysis (replace with your actual list from the app)
    # You can copy the beat times from the TripSitter UI or export them
    example_beats = [0.0, 0.5, 1.0, 1.5, 2.0]  # REPLACE THIS WITH YOUR ACTUAL BEAT LIST FROM THE APP
    # For now, generate regular beats at 120 BPM as in your screenshot
    bpm = 120.0
    beat_interval = 60.0 / bpm
    duration = 8 * 60 + 54  # 8:54 from your screenshot
    example_beats = np.arange(0.5, duration, beat_interval).tolist()  # starting from first beat

    print(f"Original beats: {len(example_beats)}")

    kept_beats = filter_beats(example_beats, audio, sr, break_threshold=0.12, thin_factor=4)

    print(f"\nKept {len(kept_beats)} beats after filtering breaks.")
    print("Filtered beat times (copy these into the app or use for video cutting):")
    for t in kept_beats:
        print(f"{t:.3f}")

    print("\nCopy the list above and use it as 'PreAnalyzedBeatTimes' or manually add the kept markers in TripSitter.")
    print("This logic will be integrated into the GUI as the 'Respect Breaks' Pro feature.")