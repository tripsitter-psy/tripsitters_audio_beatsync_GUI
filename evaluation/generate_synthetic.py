#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path

# Create a synthetic audio (sine or silence) and a color video of given durations.
# Supports multiclip generation. Requires ffmpeg in PATH; if missing, the script exits cleanly.

def run_cmd(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)


def create_audio(path: Path, duration: float, sample_rate: int = 44100, silent: bool = False, channels: int = 2, codec: str = 'wav'):
    # Use ffmpeg sine source or anullsrc for silence
    if silent:
        src = f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}"
        base_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", src,
            "-t", str(duration),
            "-ar", str(sample_rate),
            "-ac", str(channels),
        ]
    else:
        base_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}:sample_rate={sample_rate}",
            "-ar", str(sample_rate),
            "-ac", str(channels),
        ]

    if codec == 'mp3':
        out = str(path.with_suffix('.mp3'))
        cmd = base_cmd + ["-codec:a", "libmp3lame", out]
    elif codec in ('aac', 'm4a'):
        out = str(path.with_suffix('.m4a'))
        cmd = base_cmd + ["-codec:a", "aac", out]
    else:
        out = str(path.with_suffix('.wav'))
        cmd = base_cmd + [out]

    run_cmd(cmd)
    return Path(out)


def create_video(path: Path, duration: float, resolution: str = "640x360", label: str = ""):
    # color + timestamp overlay
    fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # On macOS and Windows font paths may differ; let ffmpeg choose default if font not found
    vf = f"drawtext=fontfile={fontfile}:text='%{{pts\\:hms}} {label}':x=10:y=10:fontsize=24:fontcolor=white"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=darkred:s={resolution}:d={duration}",
        "-vf", vf,
        "-c:v", "libx264",
        "-t", str(duration),
        str(path)
    ]
    run_cmd(cmd)


def create_clip_set(outdir: Path, num_clips: int, clip_duration: float):
    clips_dir = outdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_clips):
        resolution = "640x360" if (i % 2 == 0) else "854x480"
        clip = clips_dir / f"clip_{i:03d}.mp4"
        create_video(clip, clip_duration, resolution=resolution, label=str(i))
    return clips_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./evaluation/tmp", help="Output directory")
    parser.add_argument("--audio-duration", type=float, default=5.0)
    parser.add_argument("--video-duration", type=float, default=5.0)
    parser.add_argument("--id", default="sample")
    parser.add_argument("--num-clips", type=int, default=0, help="Number of clips to generate (multiclip mode)")
    parser.add_argument("--clip-duration", type=float, default=5.0, help="Duration for each clip when generating multiclip set")
    parser.add_argument("--silent", action='store_true', help="Generate silent audio")
    parser.add_argument("--audio-sr", type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--channels", type=int, default=2, help="Number of audio channels (1=mono,2=stereo)")
    parser.add_argument("--audio-codec", type=str, default="wav", help="Audio codec/format: wav, mp3, aac")
    parser.add_argument("--truncate-audio", type=int, default=0, help="Truncate audio by percent (0-99)")
    parser.add_argument("--truncate-video", type=int, default=0, help="Truncate video by percent (0-99)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    audio = outdir / f"{args.id}.wav"

    # check ffmpeg
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH; synthetic media generation requires ffmpeg. Skipping generation.")
        return 1

    create_audio(audio, args.audio_duration, sample_rate=args.audio_sr, silent=args.silent)

    audio_path = create_audio(audio, args.audio_duration, sample_rate=args.audio_sr, silent=args.silent, channels=args.channels, codec=args.audio_codec)

    if args.num_clips and args.num_clips > 0:
        clips_dir = create_clip_set(outdir, args.num_clips, args.clip_duration)
        print("Generated clip set:", clips_dir)
    else:
        video = outdir / f"{args.id}.mp4"
        create_video(video, args.video_duration)
        print("Generated:", audio_path, video)

    # Optionally truncate audio or video
    if args.truncate_audio and args.truncate_audio > 0:
        p = audio_path
        size = p.stat().st_size
        cut = int(size * (100 - args.truncate_audio) / 100.0)
        with open(p, 'rb+') as f:
            f.truncate(cut)
        print(f"Truncated audio to {cut} bytes ({args.truncate_audio}% removed)")

    if args.truncate_video and args.truncate_video > 0 and not args.num_clips:
        p = video
        size = p.stat().st_size
        cut = int(size * (100 - args.truncate_video) / 100.0)
        with open(p, 'rb+') as f:
            f.truncate(cut)
        print(f"Truncated video to {cut} bytes ({args.truncate_video}% removed)")

    return 0

if __name__ == '__main__':
    raise SystemExit(main())

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
