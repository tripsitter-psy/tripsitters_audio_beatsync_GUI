#!/usr/bin/env python3
"""
BeatNet bridge script for beat detection.

This script provides beat detection for audio files and outputs JSON to stdout.
It supports multiple backends with automatic fallback:
1. BeatNet (if installed) - state-of-the-art neural beat tracking
2. librosa (fallback) - traditional DSP-based beat tracking

Output format:
{"beats": [0.5, 1.0, 1.5, ...], "bpm": 120.0, "downbeats": [...]}

Usage:
    python beatnet_analyze.py <audio_file>
    python beatnet_analyze.py <audio_file> --backend librosa
    python beatnet_analyze.py <audio_file> --verbose

Exit codes:
    0 - Success
    1 - Invalid arguments
    2 - File not found
    3 - Analysis failed
    4 - No suitable backend available
"""

import json
import sys
import os
from pathlib import Path


def detect_beats_beatnet(audio_path: str, verbose: bool = False) -> dict:
    """Detect beats using BeatNet neural network."""
    try:
        from BeatNet.BeatNet import BeatNet

        if verbose:
            print(f"Using BeatNet backend...", file=sys.stderr)

        # Initialize BeatNet with offline mode for file analysis
        estimator = BeatNet(
            1,  # Model number (1 is recommended)
            mode='offline',
            inference_model='DBN',
            plot=[],
            thread=False
        )

        # Process audio file
        output = estimator.process(audio_path)

        # output is array of [beat_time, beat_type] where beat_type 1=downbeat, 2=beat
        beats = []
        downbeats = []

        for beat_time, beat_type in output:
            beats.append(float(beat_time))
            if beat_type == 1:
                downbeats.append(float(beat_time))

        # Compute BPM from beat intervals
        bpm = None
        if len(beats) >= 2:
            intervals = [beats[i+1] - beats[i] for i in range(len(beats)-1)]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval > 0:
                bpm = 60.0 / avg_interval

        return {
            "beats": beats,
            "downbeats": downbeats,
            "bpm": bpm,
            "backend": "beatnet"
        }

    except ImportError:
        if verbose:
            print("BeatNet not installed, trying fallback...", file=sys.stderr)
        raise
    except Exception as e:
        if verbose:
            print(f"BeatNet failed: {e}", file=sys.stderr)
        raise


def detect_beats_librosa(audio_path: str, verbose: bool = False) -> dict:
    """Detect beats using librosa (fallback method)."""
    try:
        import librosa
        import numpy as np

        if verbose:
            print(f"Using librosa backend...", file=sys.stderr)

        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Estimate downbeats (every 4th beat starting from first strong beat)
        # This is a simplification - real downbeat detection requires more analysis
        downbeats = []
        if len(beat_times) >= 4:
            # Use onset strength to find likely downbeat positions
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # Simple heuristic: every 4th beat
            downbeats = beat_times[::4].tolist()

        # Handle numpy scalar for tempo
        bpm_value = float(tempo) if np.isscalar(tempo) else float(tempo[0]) if len(tempo) > 0 else None

        return {
            "beats": beat_times.tolist(),
            "downbeats": downbeats,
            "bpm": bpm_value,
            "backend": "librosa"
        }

    except ImportError:
        if verbose:
            print("librosa not installed", file=sys.stderr)
        raise
    except Exception as e:
        if verbose:
            print(f"librosa failed: {e}", file=sys.stderr)
        raise


def detect_beats_madmom(audio_path: str, verbose: bool = False) -> dict:
    """Detect beats using madmom (alternative fallback)."""
    try:
        import madmom

        if verbose:
            print(f"Using madmom backend...", file=sys.stderr)

        # Use RNN beat processor
        proc = madmom.features.beats.RNNBeatProcessor()
        act = proc(audio_path)

        # Use DBN beat tracker
        beat_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        beats = beat_proc(act)

        # Compute BPM
        bpm = None
        if len(beats) >= 2:
            intervals = [beats[i+1] - beats[i] for i in range(len(beats)-1)]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval > 0:
                bpm = 60.0 / avg_interval

        return {
            "beats": beats.tolist(),
            "downbeats": [],  # madmom basic doesn't provide downbeats
            "bpm": bpm,
            "backend": "madmom"
        }

    except ImportError:
        if verbose:
            print("madmom not installed", file=sys.stderr)
        raise
    except Exception as e:
        if verbose:
            print(f"madmom failed: {e}", file=sys.stderr)
        raise


def analyze_audio(audio_path: str, backend: str = "auto", verbose: bool = False) -> dict:
    """
    Analyze audio file for beats using available backend.

    Args:
        audio_path: Path to audio file
        backend: "auto", "beatnet", "librosa", or "madmom"
        verbose: Print debug info to stderr

    Returns:
        Dictionary with beats, downbeats, bpm, and backend used
    """
    backends = []

    if backend == "auto":
        backends = [
            ("beatnet", detect_beats_beatnet),
            ("librosa", detect_beats_librosa),
            ("madmom", detect_beats_madmom),
        ]
    elif backend == "beatnet":
        backends = [("beatnet", detect_beats_beatnet)]
    elif backend == "librosa":
        backends = [("librosa", detect_beats_librosa)]
    elif backend == "madmom":
        backends = [("madmom", detect_beats_madmom)]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    last_error = None
    for name, func in backends:
        try:
            result = func(audio_path, verbose)
            if result and result.get("beats"):
                return result
        except Exception as e:
            last_error = e
            if verbose:
                print(f"Backend {name} failed: {e}", file=sys.stderr)
            continue

    raise RuntimeError(f"All backends failed. Last error: {last_error}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect beats in audio file and output JSON"
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--backend",
        choices=["auto", "beatnet", "librosa", "madmom"],
        default="auto",
        help="Beat detection backend (default: auto)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print debug info to stderr"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(json.dumps({
            "error": f"File not found: {args.audio_file}",
            "beats": [],
            "bpm": None
        }))
        return 2

    try:
        result = analyze_audio(
            str(audio_path),
            backend=args.backend,
            verbose=args.verbose
        )

        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))

        return 0

    except RuntimeError as e:
        print(json.dumps({
            "error": str(e),
            "beats": [],
            "bpm": None,
            "warning": "No beat detection backend available. Install librosa: pip install librosa"
        }))
        return 4

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "beats": [],
            "bpm": None
        }))
        return 3


if __name__ == "__main__":
    sys.exit(main())
