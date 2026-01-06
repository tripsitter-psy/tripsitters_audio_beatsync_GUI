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

Output report: `evaluation/results.json` and collected traces in `traces.jsonl`.

CI: There is a workflow that runs this evaluation on Ubuntu runners (installs ffmpeg via apt). To add macOS jobs, update `.github/workflows/evaluation.yml` with an `runs-on: macos-latest` job and ensure ffmpeg is available via Homebrew.