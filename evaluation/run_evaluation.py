#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import time
import sys
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
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed trace lines
                continue
            name = obj.get('span')
            traces.setdefault(name, 0)
            traces[name] += 1
    return traces


def run_query(q):
    tmp = TMP_DIR / q['id']
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # Dump environment and paths for debugging generator invocation
    try:
        with open(tmp / "generator.env.txt", 'w', encoding='utf-8') as ef:
            ef.write(f"python_executable={sys.executable}\n")
            ef.write(f"ffmpeg_which={shutil.which('ffmpeg')}\n")
            ef.write(f"PATH={os.environ.get('PATH')}\n")
    except Exception:
        pass

    # generate synthetic media
    gen = Path("./evaluation/generate_synthetic.py")

    # Build generator command
    if q.get('mode', 'sync') == 'multiclip':
        cmd = [sys.executable, str(gen), "--outdir", str(tmp), "--audio-duration", str(q['audio_duration']), "--num-clips", str(q['num_clips']), "--clip-duration", str(q['clip_duration']), "--id", q['id']]
    else:
        cmd = [sys.executable, str(gen), "--outdir", str(tmp), "--audio-duration", str(q['audio_duration']), "--video-duration", str(q.get('video_duration', q['audio_duration'])), "--id", q['id']]
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
    # corruption options
    if q.get('corrupt_audio_header'):
        cmd.append('--corrupt-audio-header')
    if q.get('corrupt_video_header'):
        cmd.append('--corrupt-video-header')
    if q.get('zero_audio_prefix'):
        cmd.append('--zero-audio-prefix')
    if q.get('mismatch_video_ext'):
        cmd.append('--mismatch-video-ext')
    if q.get('corrupt_middle'):
        cmd.append('--corrupt-middle')
    if q.get('append_junk'):
        cmd.append('--append-junk')
    if q.get('strip_mp4_moov'):
        cmd.append('--strip-mp4-moov')
    if q.get('corrupt_bytes'):
        cmd.extend(['--corrupt-bytes', str(q.get('corrupt_bytes'))])
    if q.get('junk_bytes'):
        cmd.extend(['--junk-bytes', str(q.get('junk_bytes'))])

    # run generator with captured output and persist logs to tmp
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        (tmp / "generator.stdout.txt").write_text(r.stdout, encoding='utf-8')
    except Exception:
        pass
    try:
        (tmp / "generator.stderr.txt").write_text(r.stderr, encoding='utf-8')
    except Exception:
        pass

    if r.returncode != 0:
        return {"id": q['id'], "error": "generator failed", "generator_stdout": r.stdout, "generator_stderr": r.stderr}

    # Locate generated audio file (could be .wav, .mp3, .m4a)
    audio = None
    for p in tmp.iterdir():
        if p.name.startswith(q['id']) and p.suffix.lower() in ('.wav', '.mp3', '.m4a'):
            audio = p
            break
    if audio is None:
        # include generator logs and directory listing to aid debugging
        files = [p.name for p in tmp.iterdir()]
        return {"id": q['id'], "error": "Generated audio not found", "generator_stdout": r.stdout, "generator_stderr": r.stderr, "tmp_files": files}

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
        expect_failure = q.get('expect_failure', False)
        return {"id": q['id'], "error": str(e), "passed": expect_failure}
    elapsed = time.time() - start

    # collect metrics
    result = {
        "id": q['id'],
        "runtime_s": elapsed,
        "output_exists": out.exists(),
        "playable": check_playable(out) if out.exists() else False,
        "alignment_s": measure_alignment(audio, out) if out.exists() else -1
    }

    # evaluate pass/fail for this query
    expect_failure = q.get('expect_failure', False)
    alignment_threshold = q.get('alignment_threshold', 1.0)
    if expect_failure:
        # Pass if we either errored earlier (handled above) or output exists but is not playable
        result['passed'] = (not result['output_exists']) or (not result['playable'])
    else:
        result['passed'] = result['output_exists'] and result['playable'] and (result['alignment_s'] >= 0 and result['alignment_s'] <= alignment_threshold)

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

    # aggregate pass/fail based on per-query 'passed'
    overall_passed = True
    for r in results:
        if not r.get('passed', False):
            overall_passed = False
            break

    return 0 if overall_passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
