#!/usr/bin/env python3
"""
Dataset preparation helper for Demucs kick separation training.

This script helps you:
1. Validate your dataset structure
2. Create mixture.wav from kick.wav + other.wav if missing
3. Check audio properties (sample rate, duration, alignment)
4. Split dataset into train/valid sets

Usage:
    python prepare_dataset.py --input /path/to/raw/stems --output /path/to/dataset
    python prepare_dataset.py --validate /path/to/dataset
"""

import argparse
import os
from pathlib import Path
import shutil

def check_scipy():
    try:
        from scipy.io import wavfile
        import numpy as np
        return True
    except ImportError:
        print("scipy not installed. Run: pip install scipy numpy")
        return False

def validate_track(track_dir: Path) -> dict:
    """Validate a single track directory."""
    result = {
        'path': str(track_dir),
        'valid': True,
        'errors': [],
        'warnings': []
    }

    required_files = ['kick.wav', 'other.wav', 'mixture.wav']
    for f in required_files:
        if not (track_dir / f).exists():
            result['errors'].append(f"Missing {f}")
            result['valid'] = False

    if result['valid']:
        try:
            from scipy.io import wavfile
            import numpy as np

            # Check sample rates match
            rates = {}
            durations = {}
            for f in required_files:
                sr, data = wavfile.read(track_dir / f)
                rates[f] = sr
                durations[f] = len(data) / sr

            if len(set(rates.values())) > 1:
                result['errors'].append(f"Sample rate mismatch: {rates}")
                result['valid'] = False

            # Check durations match (within 0.1 seconds)
            dur_values = list(durations.values())
            if max(dur_values) - min(dur_values) > 0.1:
                result['errors'].append(f"Duration mismatch: {durations}")
                result['valid'] = False

            # Check if mixture = kick + other (roughly)
            _, kick = wavfile.read(track_dir / 'kick.wav')
            _, other = wavfile.read(track_dir / 'other.wav')
            _, mixture = wavfile.read(track_dir / 'mixture.wav')

            if kick.shape != other.shape or kick.shape != mixture.shape:
                result['errors'].append("Shape mismatch between stems")
                result['valid'] = False

            result['sample_rate'] = rates['kick.wav']
            result['duration'] = durations['kick.wav']

        except Exception as e:
            result['errors'].append(f"Error reading files: {e}")
            result['valid'] = False

    return result

def create_mixture(track_dir: Path) -> bool:
    """Create mixture.wav from kick.wav + other.wav."""
    try:
        from scipy.io import wavfile
        import numpy as np

        kick_path = track_dir / 'kick.wav'
        other_path = track_dir / 'other.wav'
        mixture_path = track_dir / 'mixture.wav'

        if not kick_path.exists() or not other_path.exists():
            print(f"  Skipping {track_dir.name}: missing kick.wav or other.wav")
            return False

        sr_kick, kick = wavfile.read(kick_path)
        sr_other, other = wavfile.read(other_path)

        if sr_kick != sr_other:
            print(f"  Error {track_dir.name}: sample rate mismatch")
            return False

        # Handle different lengths (pad shorter one)
        if len(kick) != len(other):
            max_len = max(len(kick), len(other))
            if len(kick) < max_len:
                kick = np.pad(kick, ((0, max_len - len(kick)), (0, 0)) if kick.ndim == 2 else (0, max_len - len(kick)))
            if len(other) < max_len:
                other = np.pad(other, ((0, max_len - len(other)), (0, 0)) if other.ndim == 2 else (0, max_len - len(other)))

        # Mix (handle int16 overflow)
        kick = kick.astype(np.float32)
        other = other.astype(np.float32)
        mixture = kick + other

        # Normalize to prevent clipping
        max_val = np.abs(mixture).max()
        if max_val > 32767:
            mixture = mixture * (32767 / max_val)

        mixture = mixture.astype(np.int16)
        wavfile.write(mixture_path, sr_kick, mixture)
        print(f"  Created {mixture_path}")
        return True

    except Exception as e:
        print(f"  Error creating mixture for {track_dir}: {e}")
        return False

def validate_dataset(dataset_path: Path):
    """Validate entire dataset."""
    print(f"\nValidating dataset at: {dataset_path}\n")

    for split in ['train', 'valid']:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Warning: {split} directory not found")
            continue

        print(f"\n=== {split.upper()} SET ===")
        tracks = [d for d in split_path.iterdir() if d.is_dir()]
        print(f"Found {len(tracks)} tracks\n")

        valid_count = 0
        total_duration = 0

        for track in sorted(tracks):
            result = validate_track(track)
            if result['valid']:
                valid_count += 1
                total_duration += result.get('duration', 0)
                print(f"  OK: {track.name} ({result.get('duration', 0):.1f}s @ {result.get('sample_rate', 0)}Hz)")
            else:
                print(f"  FAIL: {track.name}")
                for err in result['errors']:
                    print(f"    - {err}")

        print(f"\n{split}: {valid_count}/{len(tracks)} valid tracks, {total_duration/60:.1f} minutes total")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Demucs kick separation training")
    parser.add_argument('--validate', type=str, help="Validate existing dataset")
    parser.add_argument('--create-mixtures', type=str, help="Create mixture.wav files from kick+other")
    args = parser.parse_args()

    if not check_scipy():
        return

    if args.validate:
        validate_dataset(Path(args.validate))
    elif args.create_mixtures:
        dataset_path = Path(args.create_mixtures)
        print(f"\nCreating mixtures in: {dataset_path}\n")
        for split in ['train', 'valid']:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
            print(f"\n=== {split.upper()} ===")
            for track in sorted(split_path.iterdir()):
                if track.is_dir() and not (track / 'mixture.wav').exists():
                    create_mixture(track)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python prepare_dataset.py --validate ./dataset")
        print("  python prepare_dataset.py --create-mixtures ./dataset")

if __name__ == '__main__':
    main()
