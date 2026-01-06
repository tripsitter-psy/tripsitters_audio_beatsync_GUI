Evaluation framework

Run evaluation locally:

1. Ensure `beatsync` CLI is built: `cmake --build build --config Release`
2. Ensure `ffmpeg` and `ffprobe` are installed and on PATH
3. Run:

```bash
python evaluation/run_evaluation.py
```

Output report: `evaluation/results.json` and collected traces in `traces.jsonl`.

CI: There is a workflow that runs this evaluation on Ubuntu runners (installs ffmpeg via apt).