# Demucs Kick-Only Separation Training

This directory contains the configuration for training a custom Demucs model that separates only the kick drum from psytrance/EDM tracks.

## Dataset Structure

Create your dataset with this structure:

```
dataset/
├── train/
│   ├── track001/
│   │   ├── kick.wav       # Isolated kick drum (export from DAW)
│   │   ├── other.wav      # Full mix minus kick
│   │   └── mixture.wav    # Full mix (kick + other)
│   ├── track002/
│   │   ├── kick.wav
│   │   ├── other.wav
│   │   └── mixture.wav
│   └── ...
└── valid/
    ├── track101/
    │   ├── kick.wav
    │   ├── other.wav
    │   └── mixture.wav
    └── ...
```

## Audio Requirements

- **Sample rate**: 44100 Hz (Demucs default)
- **Bit depth**: 16-bit or 24-bit WAV
- **Channels**: Stereo (or mono, will be converted)
- **Duration**: 30 seconds minimum per track recommended
- **Alignment**: All stems must be perfectly time-aligned

## Creating Training Data from Your DAW

### From Ableton Live:
1. Solo the kick track(s)
2. Export as `kick.wav`
3. Mute kick, export everything else as `other.wav`
4. Unmute all, export full mix as `mixture.wav`

### Quick Export Script (if using multiple projects):
```python
# Example: Create mixture from stems if you only have kick + other
from scipy.io import wavfile
import numpy as np

kick_sr, kick = wavfile.read('kick.wav')
other_sr, other = wavfile.read('other.wav')
mixture = kick + other
wavfile.write('mixture.wav', kick_sr, mixture)
```

## Dataset Size Recommendations

| Size | Tracks | Notes |
|------|--------|-------|
| Minimum | 50 | Basic training, may overfit |
| Recommended | 100-200 | Good generalization |
| Optimal | 500+ | Best results, longer training |

## Training Commands

### Install Demucs
```bash
pip install demucs
```

### Train the Model
```bash
# From demucs repo root
dora run -d \
    dset=your_kick_dataset \
    model=htdemucs \
    segment=7.8 \
    sources="['kick', 'other']"
```

### Fine-tune from Pretrained Drums Model
```bash
# Start from the drums model for faster convergence
dora run -d \
    continue_from=htdemucs_ft \
    dset=your_kick_dataset \
    sources="['kick', 'other']"
```

## Configuration File

See `kick_config.yaml` for the full training configuration.

## Export to ONNX

After training, export for use in TripSitter:

```bash
python -m tools.export YOUR_SIGNATURE
# Then convert to ONNX:
python scripts/convert_demucs_to_onnx.py --model path/to/model.th --output models/kick_separator.onnx
```

## Tips for Psytrance

1. **Include variety**: Different kick styles (punchy, subby, layered)
2. **Include edge cases**: Tracks with heavy sidechaining, offbeat patterns
3. **Augmentation**: The trainer will pitch-shift and time-stretch automatically
4. **Validation split**: Keep 10-20% of tracks for validation
