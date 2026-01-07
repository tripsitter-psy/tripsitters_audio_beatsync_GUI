#!/usr/bin/env python3
"""
BeatNet beat detection wrapper.

Runs BeatNet neural network for beat and downbeat detection.
Outputs JSON with beat timestamps and BPM.

Requirements:
    pip install BeatNet numpy

Usage:
    python beatnet_analyze.py <audio_file>
"""
import json
import sys
from pathlib import Path


def find_beatnet():
    """Check if BeatNet is available."""
    try:
        from BeatNet.BeatNet import BeatNet
        return True
    except ImportError:
        return False


def run_beatnet(audio_path: str) -> dict:
    """
    Run BeatNet beat detection on an audio file.
    
    Args:
        audio_path: Path to input audio file
    
    Returns:
        Dictionary with beat times, downbeats, and BPM
    """
    audio_path = Path(audio_path).resolve()
    
    if not audio_path.exists():
        return {"error": f"Audio file not found: {audio_path}", "beats": []}
    
    if not find_beatnet():
        return {
            "error": "BeatNet not installed. Install with: pip install BeatNet",
            "beats": []
        }
    
    try:
        from BeatNet.BeatNet import BeatNet
        import numpy as np
        
        # Initialize BeatNet
        # Mode 1: offline (whole file), Mode 2: online (streaming)
        # Model: 1 = trained on ballroom, 2 = trained on rock/pop
        estimator = BeatNet(
            1,  # model number
            mode='offline',
            inference_model='PF',  # Particle Filtering for best accuracy
            plot=[],
            thread=False
        )
        
        # Process audio
        output = estimator.process(str(audio_path))
        
        # output is a numpy array with shape (N, 2): [time, beat_number]
        # beat_number 1 = downbeat, 2/3/4 = other beats in bar
        beats = []
        downbeats = []
        
        if output is not None and len(output) > 0:
            for row in output:
                time = float(row[0])
                beat_num = int(row[1])
                beats.append(time)
                if beat_num == 1:
                    downbeats.append(time)
        
        # Estimate BPM from beat intervals
        bpm = None
        if len(beats) > 1:
            intervals = np.diff(beats)
            median_interval = np.median(intervals)
            if median_interval > 0:
                bpm = 60.0 / median_interval
        
        return {
            "status": "success",
            "audio": str(audio_path),
            "beats": beats,
            "downbeats": downbeats,
            "bpm": bpm,
            "num_beats": len(beats)
        }
        
    except Exception as e:
        return {
            "error": f"BeatNet analysis failed: {str(e)}",
            "beats": []
        }


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Usage: beatnet_analyze.py <audio_file>"
        }))
        return 1
    
    audio_path = sys.argv[1]
    result = run_beatnet(audio_path)
    print(json.dumps(result))
    
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
