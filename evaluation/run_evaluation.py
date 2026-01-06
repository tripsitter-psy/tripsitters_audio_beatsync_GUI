#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluation"
TMP_DIR = EVAL_DIR / "tmp"
RESULTS = EVAL_DIR / "results.json"
QUERIES = EVAL_DIR / "queries.json"


def run_cmd(cmd, check=True):
    print("CMD:", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and r.returncode != 0:
        print("ERROR:", r.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return r


def ffprobe_duration(path: Path) -> float:
    if not shutil.which("ffprobe"):
        print("ffprobe not found; falling back to file size heuristic")
        return -1.0
    r = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(path)], check=True)
    return float(r.stdout.strip())


def measure_alignment(audio_path: Path, video_path: Path) -> float:
    a = ffprobe_duration(audio_path)
    v = ffprobe_duration(video_path)
    if a <= 0 or v <= 0:
        return abs((a if a > 0 else 0) - (v if v > 0 else 0))
    return abs(a - v)


def check_playable(path: Path) -> bool:
    # try ffprobe to see if streams present
    try:
        if shutil.which("ffprobe"):
            r = run_cmd(["ffprobe", "-v", "error", "-show_streams", str(path)], check=False)
            return r.returncode == 0 and len(r.stdout) > 0
        else:
            return path.exists() and path.stat().st_size > 1024
    except Exception:
        return False


def parse_traces() -> dict:
    traces = {}
    trace_file = ROOT / "traces.jsonl"
    if not trace_file.exists():
        return traces
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            name = obj.get('span')
            traces.setdefault(name, 0)
            traces[name] += 1
    return traces


def run_query(q):
    tmp = TMP_DIR / q['id']
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # generate synthetic media
    gen = Path("./evaluation/generate_synthetic.py")
    try:
        if q.get('mode', 'sync') == 'multiclip':
            cmd = ["python", str(gen), "--outdir", str(tmp), "--audio-duration", str(q['audio_duration']), "--num-clips", str(q['num_clips']), "--clip-duration", str(q['clip_duration']), "--id", q['id']]
        else:
            cmd = ["python", str(gen), "--outdir", str(tmp), "--audio-duration", str(q['audio_duration']), "--video-duration", str(q.get('video_duration', q['audio_duration'])), "--id", q['id']]
        if q.get('silent'):
            cmd.append('--silent')
        if q.get('audio_sr'):
            cmd.extend(['--audio-sr', str(q.get('audio_sr'))])
        if q.get('audio_channels'):
            cmd.extend(['--channels', str(q.get('audio_channels'))])
        if q.get('audio_codec'):
            cmd.extend(['--audio-codec', str(q.get('audio_codec'))])
        if q.get('truncate_audio'):
            cmd.extend(['--truncate-audio', str(q.get('truncate_audio'))])
        if q.get('truncate_video'):
            cmd.extend(['--truncate-video', str(q.get('truncate_video'))])

        run_cmd(cmd)
    except Exception as e:
        return {"id": q['id'], "error": str(e)}

    # Locate generated audio file (could be .wav, .mp3, .m4a)
    audio = None
    for p in tmp.iterdir():
        if p.name.startswith(q['id']) and p.suffix.lower() in ('.wav', '.mp3', '.m4a'):
            audio = p
            break
    if audio is None:
        return {"id": q['id'], "error": "Generated audio not found"}

    out = tmp / f"{q['id']}.out.mp4"

    # determine run command based on mode
    start = time.time()
    try:
        if q.get('mode', 'sync') == 'multiclip':
            clips_dir = tmp / 'clips'
            run_cmd([str(ROOT / "build" / "bin" / "Release" / "beatsync.exe"), "multiclip", str(clips_dir), str(audio), "-o", str(out)], check=True)
        else:
            video = tmp / f"{q['id']}.mp4"
            run_cmd([str(ROOT / "build" / "bin" / "Release" / "beatsync.exe"), "sync", str(video), str(audio), "-o", str(out)], check=True)
    except Exception as e:
        return {"id": q['id'], "error": str(e)}
    elapsed = time.time() - start

    # collect metrics
    result = {
        "id": q['id'],
        "runtime_s": elapsed,
        "output_exists": out.exists(),
        "playable": check_playable(out) if out.exists() else False,
        "alignment_s": measure_alignment(audio, out) if out.exists() else -1
    }
    return result


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUERIES, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    results = []
    for q in queries:
        print("Running query:", q['id'])
        try:
            r = run_query(q)
            results.append(r)
        except Exception as e:
            results.append({"id": q.get('id', 'unknown'), "error": str(e)})

    # aggregate
    traces = parse_traces()
    report = {"results": results, "traces": traces}

    with open(RESULTS, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2)

    print("Evaluation complete. Results written to:", RESULTS)
    print(json.dumps(report, indent=2))

    # simple pass/fail: all outputs exist and alignment < 1s
    passed = True
    for r in results:
        if r.get('error') or not r.get('output_exists') or r.get('alignment_s', 999) > 1.0:
            passed = False
            break

    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
