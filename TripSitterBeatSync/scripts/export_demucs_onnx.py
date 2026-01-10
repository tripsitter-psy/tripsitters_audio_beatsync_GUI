#!/usr/bin/env python3
"""
Export Demucs model to ONNX format for TripSitter BeatSync
Demucs: Music source separation model (drums, bass, vocals, other)

Reference: https://github.com/adefossez/demucs
Based on GSOC 2025 ONNX export work
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Simplified Demucs encoder-decoder architecture for ONNX export
class DemucsEncoder(nn.Module):
    """Encoder block for Demucs"""
    def __init__(self, in_channels, out_channels, kernel_size=8, stride=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//4)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DemucsDecoder(nn.Module):
    """Decoder block for Demucs"""
    def __init__(self, in_channels, out_channels, kernel_size=8, stride=4):
        super().__init__()
        self.convt = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//4)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.convt(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DemucsLite(nn.Module):
    """
    Lightweight Demucs-style model for stem separation
    Optimized for ONNX export and real-time inference

    Input: (batch, 2, samples) - stereo audio
    Output: (batch, sources, 2, samples) - separated stems
           sources: drums, bass, other, vocals
    """
    def __init__(self, sources=4, audio_channels=2, channels=48, depth=4):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.depth = depth

        # Initial convolution
        self.initial = nn.Conv1d(audio_channels, channels, kernel_size=7, padding=3)

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = channels
        for i in range(depth):
            out_ch = in_ch * 2
            self.encoders.append(DemucsEncoder(in_ch, out_ch))
            in_ch = out_ch

        # LSTM at bottleneck
        self.lstm = nn.LSTM(in_ch, in_ch, num_layers=2, bidirectional=True, batch_first=True)

        # Projection to match skip connection dimensions
        self.lstm_proj = nn.Conv1d(in_ch * 2, in_ch, kernel_size=1)

        # Decoder
        self.decoders = nn.ModuleList()
        # After LSTM projection, in_ch is back to pre-LSTM dimensions
        for i in range(depth):
            out_ch = in_ch // 2
            self.decoders.append(DemucsDecoder(in_ch, out_ch))
            in_ch = out_ch

        # Final conv for each source
        self.final = nn.Conv1d(in_ch, sources * audio_channels, kernel_size=7, padding=3)

    def forward(self, x):
        # x: (batch, 2, samples)
        batch_size, _, length = x.shape

        # Pad to power of 2 for easier processing
        pad_length = 2 ** (self.depth + 8)  # Ensure divisible after all downsampling
        if length % pad_length != 0:
            pad_amount = pad_length - (length % pad_length)
            x = torch.nn.functional.pad(x, (0, pad_amount))
            padded_length = x.shape[-1]
        else:
            pad_amount = 0
            padded_length = length

        # Initial conv
        x = self.initial(x)

        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # LSTM at bottleneck
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # (batch, channels*2, time) due to bidirectional

        # Project back to match skip connection dimensions
        x = self.lstm_proj(x)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i+1)]
            # Ensure same size
            if x.shape[-1] != skip.shape[-1]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = x + skip  # Skip connection
            x = decoder(x)

        # Final conv
        x = self.final(x)  # (batch, sources*2, samples)

        # Reshape to (batch, sources, 2, samples)
        x = x.view(batch_size, self.sources, self.audio_channels, -1)

        # Remove padding
        if pad_amount > 0:
            x = x[..., :length]

        return x


def create_demucs_model(sources=4, audio_channels=2, channels=48):
    """Create Demucs model"""
    model = DemucsLite(sources=sources, audio_channels=audio_channels, channels=channels)
    model.eval()
    return model


def export_to_onnx(model, output_path, sample_rate=44100, chunk_duration=10.0):
    """Export model to ONNX format"""
    # Input: stereo audio chunk
    # Use smaller chunk for faster inference (2^16 = 65536 samples = ~1.5 seconds)
    chunk_samples = 2 ** 16

    dummy_input = torch.randn(1, 2, chunk_samples)

    # Use legacy export path (dynamo=False) for better compatibility
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['audio_input'],
        output_names=['separated_stems'],
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamo=False  # Use legacy export for compatibility
    )

    print(f"Exported Demucs ONNX model to: {output_path}")


def load_pretrained_weights(model, weights_path):
    """Load pre-trained weights if available"""
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pre-trained weights from: {weights_path}")
            return True
        except Exception as e:
            print(f"Could not load weights: {e}")
    return False


def download_demucs_weights():
    """
    Download pre-trained Demucs weights.
    Note: Official Demucs weights require the full model architecture.
    This lightweight version needs its own trained weights.
    """
    import urllib.request

    cache_dir = Path.home() / ".cache" / "tripsitter" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "demucs_lite_weights.pth"

    if weights_path.exists():
        return str(weights_path)

    # For the lite version, we'd need custom trained weights
    # The full Demucs weights aren't compatible with this simplified architecture
    print("Note: DemucsLite requires separately trained weights.")
    print("For now, using random initialization. Train or obtain compatible weights for production.")

    return None


def main():
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "ThirdParty" / "onnx_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "demucs.onnx"

    # Model parameters
    sources = 4  # drums, bass, other, vocals
    audio_channels = 2  # stereo
    channels = 48  # base channel width

    print("Creating DemucsLite model...")
    model = create_demucs_model(sources=sources, audio_channels=audio_channels, channels=channels)

    # Try to load pre-trained weights
    weights_path = download_demucs_weights()
    if weights_path:
        load_pretrained_weights(model, weights_path)
    else:
        print("Warning: Using randomly initialized weights.")

    print(f"Exporting to ONNX (sources={sources}, channels={channels})...")
    export_to_onnx(model, str(output_path))

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

    print(f"\nDemucs ONNX model saved to: {output_path}")
    print("\nStem order in output:")
    print("  0: drums")
    print("  1: bass")
    print("  2: other (instruments)")
    print("  3: vocals")

    return 0


if __name__ == "__main__":
    sys.exit(main())
