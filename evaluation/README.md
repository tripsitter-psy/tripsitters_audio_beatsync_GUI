Evaluation framework

Run evaluation locally:

1. Ensure `beatsync` CLI is built: `cmake --build build --config Release`
2. Ensure `ffmpeg` and `ffprobe` are installed and on PATH
3. Run:

```bash
python evaluation/run_evaluation.py
```

The test set includes:
- `sync_short`, `sync_short_mismatch` (basic sync tests)
- `sync_long` (120s full-run performance test)
- `multiclip_basic`, `multiclip_mismatch` (multiclip folder tests)
- `silent_audio` (edge-case with silent audio)
- `sample_rate_8k`, `sample_rate_48k` (sample-rate/resampling tests)
- `mono_audio` (mono vs stereo behavior)
- `codec_mp3` (MP3 input handling)
- `truncate_audio`, `truncate_video` (corrupt/truncated file handling)

Corruption tests and flags:
- `corrupt_audio_header` — flips the first N bytes of the audio file (default N=1024 via `--corrupt-bytes`) to simulate header/frame corruption; queries with this flag will usually set `"expect_failure": true`.
- `zero_audio_prefix` — overwrites the first N bytes with zeros to simulate missing header; typically paired with `"expect_failure": true`.
- `corrupt_video_header` — flips the first N bytes of the video file to corrupt its header/stream.
- `mismatch_video_ext` — renames the video file to a wrong extension (e.g., `.m4a`) to simulate extension/content mismatch.

Generator flags (supported by `generate_synthetic.py`): `--corrupt-audio-header`, `--corrupt-video-header`, `--zero-audio-prefix`, `--mismatch-video-ext`, `--corrupt-middle`, `--append-junk`, `--strip-mp4-moov`, `--corrupt-bytes`, `--junk-bytes`.

New corruption variants:
- `--corrupt-middle` — corrupts a block of bytes centered in the file (default `--corrupt-bytes` length); useful to damage framed audio/video content in the middle of the stream.
- `--append-junk` — appends random bytes to the end of the file (default `--junk-bytes`), simulating trailing garbage or clipped network streams.
- `--strip-mp4-moov` — attempts to locate the MP4 `moov` atom and zero out a region around it, breaking the container header (useful to simulate broken MOV/MP4 containers).

Virtual environment setup (project-local .venv) 🔧

- Use the provided helper to create a project-local `.venv` and install Python deps:
  - Windows (PowerShell): `scripts/setup_venv.ps1`
  - macOS/Linux: `scripts/setup_venv.sh`

- The venv uses `scripts/requirements.txt` (see `scripts/requirements.txt` in the repo). Note: `ffmpeg`/`ffprobe` are still required system binaries and must be installed separately (Homebrew, apt, Chocolatey, Scoop, or manually).

Examples:

```powershell
# Create .venv and install deps on Windows (PowerShell)
.\\scripts\\setup_venv.ps1
# Activate
.\\.venv\\Scripts\\Activate.ps1
```

```bash
# Create .venv and install deps on macOS/Linux
./scripts/setup_venv.sh
# Activate
source .venv/bin/activate
```
Runner semantics:
- Use `"expect_failure": true` in `queries.json` to indicate that a query is expected to fail or produce an unplayable output for corrupted inputs. The runner marks such queries as passed when the pipeline fails as expected (process error, no output, or non-playable output).
- For normal tests, the runner uses `alignment_threshold` (default 1.0s) to decide pass/fail based on audio/video alignment.

Output report: `evaluation/results.json` and collected traces in `traces.jsonl`.

CI: There is a workflow that runs this evaluation on Ubuntu runners (installs ffmpeg via apt). To add macOS jobs, update `.github/workflows/evaluation.yml` with an `runs-on: macos-latest` job and ensure ffmpeg is available via Homebrew.