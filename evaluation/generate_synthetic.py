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
    # corruption options
    parser.add_argument("--corrupt-audio-header", action='store_true', help="Corrupt the first bytes of the audio file (header damage)")
    parser.add_argument("--corrupt-video-header", action='store_true', help="Corrupt the first bytes of the video file (header damage)")
    parser.add_argument("--zero-audio-prefix", action='store_true', help="Overwrite the first 1024 bytes of audio with zeros")
    parser.add_argument("--mismatch-video-ext", action='store_true', help="Rename video file to a wrong extension (e.g., .m4a) to simulate mismatch")
    parser.add_argument("--corrupt-middle", action='store_true', help="Corrupt bytes in the middle of the file to damage frames")
    parser.add_argument("--append-junk", action='store_true', help="Append random junk bytes to the end of the file to simulate stream garbage")
    parser.add_argument("--strip-mp4-moov", action='store_true', help="Zero out the MP4 'moov' atom region to break the container header")
    parser.add_argument("--corrupt-bytes", type=int, default=1024, help="Number of bytes to corrupt/zero at file start or middle")
    parser.add_argument("--junk-bytes", type=int, default=1024, help="Number of junk bytes to append when --append-junk is used")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    audio = outdir / f"{args.id}.wav"

    # check ffmpeg
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH; synthetic media generation requires ffmpeg. Skipping generation.")
        return 1

    # generate base audio and video
    audio_path = create_audio(audio, args.audio_duration, sample_rate=args.audio_sr, silent=args.silent, channels=args.channels, codec=args.audio_codec)

    if args.num_clips and args.num_clips > 0:
        clips_dir = create_clip_set(outdir, args.num_clips, args.clip_duration)
        print("Generated clip set:", clips_dir)
        video = None
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

    if args.truncate_video and args.truncate_video > 0 and video is not None:
        p = video
        size = p.stat().st_size
        cut = int(size * (100 - args.truncate_video) / 100.0)
        with open(p, 'rb+') as f:
            f.truncate(cut)
        print(f"Truncated video to {cut} bytes ({args.truncate_video}% removed)")

    # Apply corruption operations
    def corrupt_file_header(p: Path, n: int):
        print(f"Corrupting header of {p} ({n} bytes)")
        with open(p, 'r+b') as f:
            f.seek(0)
            data = bytearray(f.read(n))
            for i in range(len(data)):
                data[i] = (data[i] ^ 0xFF) & 0xFF
            f.seek(0)
            f.write(data)

    def zero_prefix(p: Path, n: int):
        print(f"Zeroing first {n} bytes of {p}")
        with open(p, 'r+b') as f:
            f.seek(0)
            f.write(b'\x00' * min(n, p.stat().st_size))

    def corrupt_middle(p: Path, n: int):
        print(f"Corrupting middle {n} bytes of {p}")
        size = p.stat().st_size
        if size == 0:
            return
        start = max(0, size // 2 - n // 2)
        with open(p, 'r+b') as f:
            f.seek(start)
            data = bytearray(f.read(n))
            for i in range(len(data)):
                data[i] = (data[i] ^ 0xAA) & 0xFF
            f.seek(start)
            f.write(data)

    def append_junk(p: Path, n: int):
        print(f"Appending {n} junk bytes to {p}")
        import os
        with open(p, 'ab') as f:
            f.write(os.urandom(n))

    def strip_mp4_moov(p: Path, n: int):
        print(f"Attempting to strip/zero 'moov' atom region in {p}")
        data = p.read_bytes()
        idx = data.find(b'moov')
        if idx == -1:
            print('moov atom not found; skipping')
            return
        start = max(0, idx - 512)
        end = min(len(data), idx + n + 512)
        # zero out region to damage header
        data = data[:start] + (b'\x00' * (end - start)) + data[end:]
        p.write_bytes(data)
        print(f"Zeroed bytes {start}-{end} in {p} to corrupt moov")

    if args.corrupt_audio_header:
        corrupt_file_header(audio_path, args.corrupt_bytes)

    if args.zero_audio_prefix:
        zero_prefix(audio_path, args.corrupt_bytes)

    if args.corrupt_middle:
        corrupt_middle(audio_path, args.corrupt_bytes)

    if args.append_junk:
        append_junk(audio_path, args.junk_bytes)

    if args.corrupt_video_header and video is not None:
        corrupt_file_header(video, args.corrupt_bytes)

    if args.corrupt_middle and video is not None:
        corrupt_middle(video, args.corrupt_bytes)

    if args.append_junk and video is not None:
        append_junk(video, args.junk_bytes)

    if args.strip_mp4_moov and video is not None:
        strip_mp4_moov(video, args.corrupt_bytes)

    if args.mismatch_video_ext and video is not None:
        new = video.with_suffix('.m4a')
        print(f"Renaming video {video} -> {new} to mismatch extension")
        video.rename(new)
        video = new

    return 0


