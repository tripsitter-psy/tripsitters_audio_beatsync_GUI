"""
Convert a minimal PyTorch beat model to ONNX (opset 12) for use in tests.
Usage:
  python tools/convert_pytorch_to_onnx.py --out tests/models/beat_reference.onnx

If PyTorch is not installed, the script prints instructions and exits non-zero.
The model is intentionally tiny and deterministic for testing (no training required).
"""
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="tests/models/beat_reference.onnx")
args = parser.parse_args()

try:
    import torch
    import torch.nn as nn
except Exception as e:
    print("PyTorch not available: ", e)
    print("Install PyTorch (https://pytorch.org) or run the fallback generator tools/generate_beat_stub.py")
    sys.exit(2)

class SimpleBeatModel(nn.Module):
    """A tiny deterministic model: averages input and maps to 3 beat times as floats.
    Input shape: (1, L) where L is audio frame count (we'll give a dummy input during export).
    This model is only for testing model export and inference plumbing, not for accuracy."""
    def __init__(self):
        super().__init__()
        # initialize weights deterministically
        torch.manual_seed(0)
        # simple linear layers to produce 3 outputs
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, x):
        # downsample / pool to fixed size 64
        x = nn.functional.adaptive_avg_pool1d(x, 64)
        out = self.fc(x)
        # Map outputs to a plausible beat time range by scaling (0.2..2.0)
        out = torch.sigmoid(out) * 1.8 + 0.2
        return out

model = SimpleBeatModel()
model.eval()
# Dummy input: batch 1, 1 channel, 256 samples
x = torch.zeros(1,1,256)

try:
    torch.onnx.export(model, x, args.out, export_params=True, opset_version=12,
                      do_constant_folding=True, input_names=['input'], output_names=['beats'])
    print(f"Saved {args.out}")
except Exception as e:
    print("Failed to export model:", e)
    sys.exit(1)
