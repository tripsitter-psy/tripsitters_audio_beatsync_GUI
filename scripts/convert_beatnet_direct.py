#!/usr/bin/env python3
"""
Directly export BeatNet model to ONNX using the installed BeatNet library.

This extracts the actual pre-trained weights from BeatNet.

Usage:
    python scripts/convert_beatnet_direct.py --out models/beatnet_real.onnx --verify

Requirements:
    pip install beatnet torch onnx onnxruntime
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Export BeatNet directly to ONNX")
    parser.add_argument("--out", default="models/beatnet_real.onnx", help="Output ONNX file")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                        help="BeatNet mode: 1=offline, 2=online, 3=streaming")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: PyTorch not installed", file=sys.stderr)
        return 1

    try:
        import onnx
    except ImportError:
        print("ERROR: ONNX not installed", file=sys.stderr)
        return 1

    try:
        from BeatNet.BeatNet import BeatNet
    except ImportError:
        print("ERROR: BeatNet not installed. Run: pip install beatnet", file=sys.stderr)
        return 1

    print(f"Loading BeatNet model (mode={args.mode})...")
    beatnet = BeatNet(args.mode)
    original_model = beatnet.model
    original_model.eval()

    # Print model structure
    print(f"\nOriginal model structure:")
    print(original_model)

    # Get the state dict with trained weights
    state_dict = original_model.state_dict()
    print(f"\nModel weights:")
    for name, param in state_dict.items():
        print(f"  {name}: {param.shape}")

    # Create a clean wrapper that's ONNX-exportable
    # The original model expects: (batch, time, features)
    # But we want to feed mel spectrograms: (batch, 1, n_mels, time)



    # Actually, let's just trace the original model directly
    # First understand its exact input requirements
    print("\n=== Testing original model ===")

    # BeatNet processes audio frames, not spectrograms directly
    # The input to the neural network is extracted features
    # Let's check what shape it expects

    # From BeatNet source: input is (batch, seq_len, dim_in)
    # where dim_in=272 (spectral features per frame)
    # seq_len is the number of time frames
    dim_in = original_model.dim_in  # 272
    test_input = torch.randn(1, 10, dim_in)  # 10 time frames

    with torch.no_grad():
        try:
            output = original_model(test_input)
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output sample: {output[0, :5]}")
        except Exception as e:
            print(f"  Error with test input: {e}")

    # Create output directory
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    print("\n=== Exporting to ONNX ===")

    # Use a wrapper that handles the reshape properly
    class ExportableBDA(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # Just pass through - model handles its own logic
            return self.model(x)

    export_model = ExportableBDA(original_model)
    export_model.eval()

    # BeatNet expects (batch, seq_len, dim_in)
    # dim_in=272 (spectral features per frame)
    dim_in = original_model.dim_in
    seq_len = 10  # Number of time frames
    dummy_input = torch.randn(1, seq_len, dim_in)

    dynamic_axes = {
        'input': {0: 'batch', 1: 'seq_len'},
        'output': {0: 'batch', 1: 'seq_len'}
    }

    try:
        torch.onnx.export(
            export_model,
            dummy_input,
            args.out,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            dynamo=False
        )
        print(f"  Saved: {args.out}")
    except Exception as e:
        print(f"  Export failed: {e}")

        # Try without dynamic axes
        print("\n  Retrying with fixed input size...")
        try:
            torch.onnx.export(
                export_model,
                dummy_input,
                args.out,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamo=False
            )
            print(f"  Saved: {args.out}")
        except Exception as e2:
            print(f"  Retry also failed: {e2}", file=sys.stderr)
            return 1

    # Verify
    if args.verify:
        print("\n=== Verifying ===")
        try:
            import onnxruntime as ort
            import numpy as np

            onnx_model = onnx.load(args.out)
            onnx.checker.check_model(onnx_model)
            print("  ONNX structure: OK")

            session = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]
            print(f"  Input: {input_info.name}, shape: {input_info.shape}")

            for output in session.get_outputs():
                print(f"  Output: {output.name}, shape: {output.shape}")

            # Test inference - use same shape as dummy_input
            test_input = np.random.randn(1, seq_len, dim_in).astype(np.float32)
            outputs = session.run(None, {input_info.name: test_input})
            print(f"  Test inference: OK")
            print(f"    Output shape: {outputs[0].shape}")
            print(f"    Output sample (probabilities): {outputs[0][0, 0]}")

        except Exception as e:
            print(f"  Verification failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n=== Done ===")
    print("\nBeatNet model expects:")
    print(f"  - Input: (batch, seq_len, {dim_in}) - audio feature frames")
    print("  - Output: (batch, seq_len, 3) - [no_beat, beat, downbeat] probabilities")
    print("\nNote: BeatNet uses internal feature extraction (spectral flux).")
    print("For direct audio input, use the mel spectrogram models instead.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
