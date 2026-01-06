#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path

# Create a synthetic audio (sine) and a color video of given durations.
# Requires ffmpeg in PATH; if missing, the script exits cleanly with a message.

def run_cmd(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)


def create_audio(path: Path, duration: float):
    # Use ffmpeg sine source
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-ar", "44100",
        "-ac", "2",
        str(path)
    ]
    run_cmd(cmd)


def create_video(path: Path, duration: float):
    # color + timestamp overlay
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=darkred:s=640x360:d={duration}",
        "-vf", "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='%{pts\\:hms}':x=10:y=10:fontsize=24:fontcolor=white",
        "-c:v", "libx264",
        "-t", str(duration),
        str(path)
    ]
    run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./evaluation/tmp", help="Output directory")
    parser.add_argument("--audio-duration", type=float, default=5.0)
    parser.add_argument("--video-duration", type=float, default=5.0)
    parser.add_argument("--id", default="sample")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    audio = outdir / f"{args.id}.wav"
    video = outdir / f"{args.id}.mp4"

    # check ffmpeg
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH; synthetic media generation requires ffmpeg. Skipping generation.")
        return 1

    create_audio(audio, args.audio_duration)
    create_video(video, args.video_duration)

    print("Generated:", audio, video)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
