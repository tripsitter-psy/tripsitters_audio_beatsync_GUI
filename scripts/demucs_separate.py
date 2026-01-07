#!/usr/bin/env python3
"""
Demucs stem separation wrapper.

Runs Demucs v4 to separate audio into stems (drums, bass, vocals, other).
Outputs JSON with stem file paths for downstream processing.

Requirements:
    pip install demucs torch torchaudio

Usage:
    python demucs_separate.py <audio_file> <output_dir> [--model htdemucs]
"""
import json
import sys
import subprocess
from pathlib import Path


def find_demucs():
    """Find demucs executable or module."""
    # Try as module first
    try:
        import demucs
        return "module"
    except ImportError:
        pass
    
    # Try as CLI command
    import shutil
    if shutil.which("demucs"):
        return "cli"
    
    return None


def run_demucs(audio_path: str, output_dir: str, model: str = "htdemucs") -> dict:
    """
    Run Demucs separation and return stem paths.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated stems
        model: Demucs model to use (htdemucs, htdemucs_ft, mdx_extra)
    
    Returns:
        Dictionary with stem paths and metadata
    """
    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not audio_path.exists():
        return {"error": f"Audio file not found: {audio_path}", "stems": {}}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    demucs_type = find_demucs()
    if demucs_type is None:
        return {
            "error": "Demucs not installed. Install with: pip install demucs torch torchaudio",
            "stems": {}
        }
    
    # Run demucs
    try:
        if demucs_type == "module":
            cmd = [
                sys.executable, "-m", "demucs",
                "-n", model,
                "-o", str(output_dir),
                str(audio_path)
            ]
        else:
            cmd = [
                "demucs",
                "-n", model,
                "-o", str(output_dir),
                str(audio_path)
            ]
        
        # Print progress to stderr for the GUI to capture
        print(json.dumps({"status": "running", "message": "Starting Demucs separation..."}), flush=True)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            return {
                "error": f"Demucs failed: {result.stderr}",
                "stems": {}
            }
        
    except subprocess.TimeoutExpired:
        return {"error": "Demucs timed out after 10 minutes", "stems": {}}
    except Exception as e:
        return {"error": f"Failed to run Demucs: {str(e)}", "stems": {}}
    
    # Find output stems
    # Demucs outputs to: output_dir/model_name/track_name/stem.wav
    track_name = audio_path.stem
    stems_dir = output_dir / model / track_name
    
    stems = {}
    stem_names = ["drums", "bass", "vocals", "other"]
    
    for stem_name in stem_names:
        stem_path = stems_dir / f"{stem_name}.wav"
        if stem_path.exists():
            stems[stem_name] = str(stem_path)
    
    if not stems:
        # Try alternative path structure
        for stem_name in stem_names:
            stem_path = output_dir / track_name / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = str(stem_path)
    
    return {
        "status": "success",
        "model": model,
        "audio": str(audio_path),
        "output_dir": str(output_dir),
        "stems": stems
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(json.dumps({
            "error": "Usage: demucs_separate.py <audio_file> <output_dir> [--model htdemucs]"
        }))
        return 1
    
    audio_path = sys.argv[1]
    output_dir = sys.argv[2]
    model = "htdemucs"
    
    # Parse optional model argument
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model = sys.argv[idx + 1]
    
    result = run_demucs(audio_path, output_dir, model)
    print(json.dumps(result))
    
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
