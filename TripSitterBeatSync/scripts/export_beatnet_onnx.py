#!/usr/bin/env python3
"""
Export BeatNet model to ONNX format for TripSitter BeatSync
BeatNet: CRNN-based online joint beat/downbeat/tempo tracking

Reference: https://github.com/mjhydri/BeatNet
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# BeatNet architecture - simplified CRNN for beat detection
class BeatNetCRNN(nn.Module):
    """
    Simplified BeatNet-style CRNN for beat detection
    Input: Mel spectrogram (batch, 1, time_frames, n_mels)
    Output: Beat probabilities (batch, time_frames, 2) - [beat_prob, downbeat_prob]
    """
    def __init__(self, n_mels=80, hidden_size=128, num_layers=2):
        super().__init__()

        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Pool only in frequency

            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # After 3 max pools in frequency: n_mels/8
        self.freq_bins = n_mels // 8
        self.cnn_out_features = 128 * self.freq_bins

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self.cnn_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )

        # Output layer: beat and downbeat probabilities
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),  # 2 outputs: beat_prob, downbeat_prob
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, 1, time, freq)
        batch_size, _, time_steps, _ = x.shape

        # CNN
        x = self.conv_layers(x)  # (batch, 128, time, freq/8)

        # Reshape for RNN: (batch, time, features)
        x = x.permute(0, 2, 1, 3)  # (batch, time, 128, freq/8)
        x = x.reshape(batch_size, time_steps, -1)  # (batch, time, 128*freq/8)

        # RNN
        x, _ = self.rnn(x)  # (batch, time, hidden*2)

        # FC
        x = self.fc(x)  # (batch, time, 2)

        return x


def create_beatnet_model(n_mels=80, hidden_size=128):
    """Create and initialize BeatNet model with random weights"""
    model = BeatNetCRNN(n_mels=n_mels, hidden_size=hidden_size)
    model.eval()
    return model


def export_to_onnx(model, output_path, n_mels=80, sequence_length=256):
    """Export model to ONNX format"""
    # Create dummy input: (batch, channels, time_frames, mel_bins)
    # batch=1, channels=1 (mono), time_frames=fixed, mel_bins=n_mels
    dummy_input = torch.randn(1, 1, sequence_length, n_mels)

    # Use legacy export path (dynamo=False) for better compatibility
    # Fixed shapes work better with UE5's NNE runtime
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['mel_spectrogram'],
        output_names=['beat_activations'],
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamo=False  # Use legacy export for compatibility
    )

    print(f"Exported BeatNet ONNX model to: {output_path}")


def load_pretrained_weights(model, weights_path):
    """Load pre-trained weights if available"""
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Loaded pre-trained weights from: {weights_path}")
            return True
        except Exception as e:
            print(f"Could not load weights: {e}")
    return False


def download_beatnet_weights():
    """
    Download pre-trained BeatNet weights from the official repo.
    Returns path to weights file or None if download failed.
    """
    import urllib.request
    import tempfile

    # BeatNet GitHub releases - check for available weights
    urls = [
        # These would be the actual weight URLs from BeatNet releases
        "https://github.com/mjhydri/BeatNet/releases/download/v1.0/beatnet_weights.pth",
    ]

    cache_dir = Path.home() / ".cache" / "tripsitter" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "beatnet_weights.pth"

    if weights_path.exists():
        return str(weights_path)

    for url in urls:
        try:
            print(f"Attempting to download weights from: {url}")
            urllib.request.urlretrieve(url, str(weights_path))
            return str(weights_path)
        except Exception as e:
            print(f"Download failed: {e}")
            continue

    return None


def main():
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "ThirdParty" / "onnx_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "beatnet.onnx"

    # Model parameters
    n_mels = 80  # Number of mel frequency bins
    hidden_size = 128  # LSTM hidden size
    sequence_length = 256  # Default sequence length for export

    print("Creating BeatNet model...")
    model = create_beatnet_model(n_mels=n_mels, hidden_size=hidden_size)

    # Try to load pre-trained weights
    weights_path = download_beatnet_weights()
    if weights_path:
        load_pretrained_weights(model, weights_path)
    else:
        print("Warning: Using randomly initialized weights. For production, train the model or obtain pre-trained weights.")

    print(f"Exporting to ONNX (n_mels={n_mels}, seq_len={sequence_length})...")
    export_to_onnx(model, str(output_path), n_mels=n_mels, sequence_length=sequence_length)

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")

        # Print model info
        print(f"Model inputs: {[i.name for i in onnx_model.graph.input]}")
        print(f"Model outputs: {[o.name for o in onnx_model.graph.output]}")
    except ImportError:
        print("Install onnx package to verify: pip install onnx")
    except Exception as e:
        print(f"Verification warning: {e}")

    print(f"\nBeatNet ONNX model saved to: {output_path}")
    print("Next steps:")
    print("1. For production, obtain trained BeatNet weights")
    print("2. Run export_demucs_onnx.py to export Demucs model")
    print("3. Both models will be loaded by the UE5 NNE runtime")

    return 0


if __name__ == "__main__":
    sys.exit(main())
