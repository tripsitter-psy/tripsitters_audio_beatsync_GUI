#!/usr/bin/env python3
"""
Generate a long WAV file (mono 16-bit PCM) for GPU stress testing.
Usage: python tools/generate_long_test_wav.py out.wav duration_seconds sample_rate
Defaults: out.wav=build/tmp/gpu_stress.wav, duration=600 (10 minutes), sample_rate=22050
"""
import sys
import wave
import struct
import math

def generate(path='build/tmp/gpu_stress.wav', duration=600, rate=22050):
    nframes = int(duration * rate)
    # Use a sweeping sine wave to exercise analysis
    freq0 = 220.0
    freq1 = 1760.0
    amplitude = 0.3

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        chunk_size = rate * 1  # 1 second chunks for efficiency
        for start_idx in range(0, nframes, chunk_size):
            end_idx = min(start_idx + chunk_size, nframes)
            chunk_data = []
            for i in range(start_idx, end_idx):
                t = i / float(rate)
                # linear sweep
                frac = t / float(duration)
                freq = freq0 + (freq1 - freq0) * frac
                sample = amplitude * math.sin(2.0 * math.pi * freq * t)
                # small added noise to avoid pure tone
                # no external RNG to keep deterministic
                noise = 0.001 * math.sin(2.0 * math.pi * 1234.0 * t)
                val = sample + noise
                # clamp
                val = max(-0.9999, min(0.9999, val))
                intv = int(val * 32767.0)
                chunk_data.append(intv)
            
            # Pack entire chunk at once
            wf.writeframes(struct.pack('<' + 'h' * len(chunk_data), *chunk_data))

if __name__ == '__main__':
    out = 'build/tmp/gpu_stress.wav'
    dur = 600
    rate = 22050
    if len(sys.argv) > 1:
        out = sys.argv[1]
    if len(sys.argv) > 2:
        dur = int(sys.argv[2])
    if len(sys.argv) > 3:
        rate = int(sys.argv[3])
    import os
    dirpath = os.path.dirname(out)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    print(f'Generating {out} ({dur}s @ {rate}Hz) ...')
    generate(out, dur, rate)
    print('Done')
