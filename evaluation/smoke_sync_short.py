#!/usr/bin/env python3
import subprocess
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "evaluation" / "tmp" / "smoke_sync_short"

if __name__ == '__main__':
    TMP.mkdir(parents=True, exist_ok=True)
    gen = ROOT / "evaluation" / "generate_synthetic.py"
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; skipping smoke test")
        sys.exit(0)
    # generate
    cmd = [sys.executable, str(gen), "--outdir", str(TMP), "--audio-duration", "6", "--video-duration", "6", "--id", "sync_short"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(r.stdout)
    print(r.stderr, file=sys.stderr)
    if r.returncode != 0:
        print("Generator failed")
        sys.exit(2)
    # find audio/video
    video = TMP / "sync_short.mp4"
    audio = TMP / "sync_short.wav"
    if not video.exists() or not audio.exists():
        print("Generated files missing")
        sys.exit(2)
    beatsync = ROOT / "build" / "bin" / "Release" / ("beatsync.exe" if sys.platform.startswith('win') else "beatsync")
    cmd = [str(beatsync), "sync", str(video), str(audio), "-o", str(TMP / "out.mp4")]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(r.stdout)
    print(r.stderr, file=sys.stderr)
    if r.returncode != 0:
        print("beatsync failed")
        sys.exit(2)
    out = TMP / "out.mp4"
    if not out.exists():
        print("Output missing")
        sys.exit(2)
    # check playable
    if shutil.which("ffprobe"):
        r = subprocess.run(["ffprobe","-v","error","-show_streams", str(out)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0 or len(r.stdout) == 0:
            print("Output not playable")
            sys.exit(2)
    print("Smoke test passed")
    sys.exit(0)
