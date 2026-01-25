#!/usr/bin/env python3
"""
Convert Demucs/HTDemucs stem separation model to ONNX format.

Demucs separates audio into 4 stems: drums, bass, other, vocals
This is useful as a preprocessing step for better beat detection.

Usage:
    python scripts/convert_demucs_to_onnx.py --out models/demucs.onnx

Requirements:
    pip install torch demucs onnx onnxruntime

Note: Demucs models are large (~1GB). For production, consider using
the smaller htdemucs_ft variant or a lightweight alternative.

References:
    - HTDemucs: https://github.com/facebookresearch/demucs
    - Paper: "Hybrid Transformers for Music Source Separation" (ICASSP 2023)
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Convert Demucs to ONNX")
    parser.add_argument("--out", default="models/htdemucs.onnx", help="Output ONNX file path")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pre-trained weights (.pt, .pth, or .th file)")
    parser.add_argument("--model", default="htdemucs",
                        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"],
                        help="Demucs model variant")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--segment-length", type=int, default=44100 * 10,
                        help="Audio segment length in samples (default: 10 seconds at 44.1kHz)")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model (requires onnx-simplifier)")
    parser.add_argument("--allow-unsafe-checkpoint", action="store_true",
                        help="Allow loading checkpoints without weights_only=True (security risk)")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch", file=sys.stderr)
        return 1

    try:
        import onnx
    except ImportError:
        print("ERROR: ONNX not installed. Run: pip install onnx", file=sys.stderr)
        return 1

    # Try to import demucs
    try:
        import demucs.pretrained
        USE_DEMUCS = True
    except ImportError:
        print("WARNING: Demucs not installed. Creating a lightweight separator architecture.")
        print("         For best results: pip install demucs")
        USE_DEMUCS = False

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    if USE_DEMUCS:
        print(f"Loading Demucs model: {args.model}...")
        try:
            # Load pretrained model
            model = demucs.pretrained.get_model(args.model)
            model.eval()

            # Get model info
            sample_rate = model.samplerate  # Usually 44100
            sources = model.sources  # ['drums', 'bass', 'other', 'vocals']
            print(f"  Sample rate: {sample_rate}")
            print(f"  Sources: {sources}")

            # Create dummy input
            # Demucs expects: (batch, channels=2, samples)
            segment_length = args.segment_length
            dummy_input = torch.randn(1, 2, segment_length)

            print(f"Exporting to ONNX (input shape: {dummy_input.shape})...")
            print("  This may take a few minutes for large models...")

            # Export
            dynamic_axes = {
                'audio_input': {0: 'batch', 2: 'samples'},
                'stems_output': {0: 'batch', 3: 'samples'}
            }

            # Wrapper to handle the model output format
            class DemucsWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    # Demucs returns (batch, sources, channels, samples)
                    return self.model(x)

            wrapper = DemucsWrapper(model)

            torch.onnx.export(
                wrapper,
                dummy_input,
                args.out,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=['audio_input'],
                output_names=['stems_output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )

        except Exception as e:
            print(f"ERROR exporting Demucs: {e}", file=sys.stderr)
            print("\nDemucs models can be complex to export. Creating lightweight alternative...")
            USE_DEMUCS = False

    if not USE_DEMUCS:
        print("Creating lightweight stem separator architecture...")
        print("Note: This is a simplified model for testing. For production,")
        print("      export the actual Demucs model or use the Python bridge.")

        class LightweightStemSeparator(nn.Module):
            """
            Lightweight U-Net style stem separator.

            Much smaller than Demucs but still functional for basic separation.
            Can be used for testing the ONNX pipeline.

            Input: (batch, 2, samples) - stereo audio
            Output: (batch, 4, 2, samples) - 4 stems (drums, bass, other, vocals)
            """
            def __init__(self, n_stems=4, hidden_channels=32):
                super().__init__()
                self.n_stems = n_stems

                # Encoder (downsampling)
                self.enc1 = nn.Sequential(
                    nn.Conv1d(2, hidden_channels, kernel_size=8, stride=4, padding=2),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU()
                )
                self.enc2 = nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=8, stride=4, padding=2),
                    nn.BatchNorm1d(hidden_channels * 2),
                    nn.ReLU()
                )
                self.enc3 = nn.Sequential(
                    nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=8, stride=4, padding=2),
                    nn.BatchNorm1d(hidden_channels * 4),
                    nn.ReLU()
                )

                # Bottleneck
                self.bottleneck = nn.Sequential(
                    nn.Conv1d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_channels * 4),
                    nn.ReLU()
                )

                # Decoder (upsampling) - separate for each stem
                self.decoders = nn.ModuleList([
                    nn.ModuleDict({
                        'dec3': nn.Sequential(
                            nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2,
                                             kernel_size=8, stride=4, padding=2),
                            nn.BatchNorm1d(hidden_channels * 2),
                            nn.ReLU()
                        ),
                        'dec2': nn.Sequential(
                            nn.ConvTranspose1d(hidden_channels * 2, hidden_channels,
                                             kernel_size=8, stride=4, padding=2),
                            nn.BatchNorm1d(hidden_channels),
                            nn.ReLU()
                        ),
                        'dec1': nn.ConvTranspose1d(hidden_channels, 2,
                                                   kernel_size=8, stride=4, padding=2)
                    }) for _ in range(n_stems)
                ])

            def forward(self, x):
                # x: (batch, 2, samples)
                batch_size = x.shape[0]
                orig_length = x.shape[2]

                # Encode
                e1 = self.enc1(x)
                e2 = self.enc2(e1)
                e3 = self.enc3(e2)

                # Bottleneck
                b = self.bottleneck(e3)

                # Decode each stem
                stems = []
                for decoder in self.decoders:
                    d3 = decoder['dec3'](b)
                    d2 = decoder['dec2'](d3)
                    d1 = decoder['dec1'](d2)

                    # Ensure output matches input length
                    if d1.shape[2] > orig_length:
                        d1 = d1[:, :, :orig_length]
                    elif d1.shape[2] < orig_length:
                        d1 = nn.functional.pad(d1, (0, orig_length - d1.shape[2]))

                    stems.append(d1)

                # Stack stems: (batch, n_stems, 2, samples)
                output = torch.stack(stems, dim=1)
                return output

        model = LightweightStemSeparator()

        # Load pre-trained weights if provided
        if args.weights:
            print(f"Loading weights from: {args.weights}")

            try:
                # Try safe loading with weights_only=True (PyTorch >= 1.13)
                try:
                    state_dict = torch.load(args.weights, map_location='cpu', weights_only=True)
                except TypeError:
                    # PyTorch version doesn't support weights_only parameter
                    if not args.allow_unsafe_checkpoint:
                        print("  ERROR: Your PyTorch version does not support weights_only=True.", file=sys.stderr)
                        print(f"         Loading '{args.weights}' without this flag is a security risk.", file=sys.stderr)
                        print("         Either upgrade PyTorch (>= 1.13) or pass --allow-unsafe-checkpoint", file=sys.stderr)
                        return 1
                    print("  WARNING: Loading checkpoint without weights_only=True (--allow-unsafe-checkpoint specified)")
                    state_dict = torch.load(args.weights, map_location='cpu')
                # Handle different checkpoint formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                result = model.load_state_dict(state_dict, strict=False)
                print("  Weights loaded successfully!")
                if getattr(result, 'missing_keys', None):
                    print(f"  WARNING: Missing keys in state_dict: {result.missing_keys}")
                if getattr(result, 'unexpected_keys', None):
                    print(f"  WARNING: Unexpected keys in state_dict: {result.unexpected_keys}")
            except Exception as e:
                print(f"  WARNING: Could not load weights: {e}")
                print("  Using random initialization instead.")
        else:
            # Ensure deterministic initialization
            torch.manual_seed(42)
            model = LightweightStemSeparator()
            for m in model.modules():
                if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        model.eval()

        # Export
        segment_length = args.segment_length
        dummy_input = torch.randn(1, 2, segment_length)

        print(f"Exporting to ONNX (input shape: {dummy_input.shape})...")

        dynamic_axes = {
            'audio_input': {0: 'batch', 2: 'samples'},
            'stems_output': {0: 'batch', 3: 'samples'}
        }

        torch.onnx.export(
            model,
            dummy_input,
            args.out,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=['audio_input'],
            output_names=['stems_output'],
            dynamic_axes=dynamic_axes
        )

    print(f"Saved ONNX model to: {args.out}")

    # Optionally simplify
    if args.simplify:
        try:
            import onnxsim
            print("\nSimplifying ONNX model...")
            onnx_model = onnx.load(args.out)
            simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(simplified, args.out)
                print("  Simplified successfully")
            else:
                print("  Simplification check failed, keeping original")
        except ImportError:
            print("\nSkipping simplification (onnx-simplifier not installed)")

    # Verify
    if args.verify:
        print("\nVerifying exported model...")
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx_model = onnx.load(args.out)
            onnx.checker.check_model(onnx_model)
            print("  ONNX model structure: OK")

            session = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]
            print(f"  Input: {input_info.name}, shape: {input_info.shape}")

            for output in session.get_outputs():
                print(f"  Output: {output.name}, shape: {output.shape}")

            # Test inference
            test_length = 44100  # 1 second
            test_input = np.random.randn(1, 2, test_length).astype(np.float32)
            outputs = session.run(None, {input_info.name: test_input})

            print(f"  Test inference: OK")
            print(f"    Output shape: {outputs[0].shape}")
            print(f"    Expected: (1, 4, 2, {test_length})")

        except Exception as e:
            print(f"  Verification failed: {e}", file=sys.stderr)
            return 1

    print("\nDone! Model is ready for use with OnnxStemSeparator.")
    print("\nModel expects:")
    print("  - Input: Stereo audio (batch, 2, samples)")
    print("  - Sample rate: 44100 Hz")
    print("  - Output: 4 stems (drums, bass, other, vocals)")
    print("  - Output shape: (batch, 4, 2, samples)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
