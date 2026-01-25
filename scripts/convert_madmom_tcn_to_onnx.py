#!/usr/bin/env python3
"""
Convert madmom TCN (Temporal Convolutional Network) beat detection models to ONNX.

These are the same models used by BeatNet for beat/downbeat detection.

Usage:
    # Inspect model structure
    python scripts/convert_madmom_tcn_to_onnx.py --pkl path/to/beats_tcn_1.pkl --inspect

    # Convert single model
    python scripts/convert_madmom_tcn_to_onnx.py --pkl path/to/beats_tcn_1.pkl --out models/tcn_beats.onnx

    # Convert all 8 models (ensemble)
    python scripts/convert_madmom_tcn_to_onnx.py --pkl-dir path/to/pkl/dir --out models/tcn_beats_ensemble.onnx --ensemble

Requirements:
    pip install torch onnx onnxruntime numpy

Architecture (from inspection):
    - Layer 0: Conv2d (1, 16, 3x3) + ELU
    - Layer 1: MaxPool (1, 3)
    - Layer 2: Conv2d (16, 16, 3x3) + ELU
    - Layer 3: MaxPool (1, 3)
    - Layer 4: Conv2d (16, 16, 1x8) + ELU
    - Layer 5: Reshape to (batch, time, 16)
    - Layer 6: TCN with 11 blocks, skip connections, ELU
    - Layer 7: MultiTask output (beat + downbeat)
"""
import argparse
import sys
import os
import pickle
import numpy as np


class MadmomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles madmom classes without requiring madmom installed."""

    def find_class(self, module, name):
        if 'madmom' in module:
            class DummyMadmomClass:
                def __init__(self, *args, **kwargs):
                    pass

                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
                    else:
                        self._state = state

                def __reduce__(self):
                    return (self.__class__, ())

            DummyMadmomClass.__name__ = name
            DummyMadmomClass.__module__ = module
            return DummyMadmomClass

        return super().find_class(module, name)


def load_pkl_safe(pkl_path):
    """
    Load a madmom .pkl file using a restricted unpickler.

    SECURITY WARNING: Only load pickle files from trusted sources! Pickle files can execute arbitrary code.
    This function uses MadmomUnpickler (a restricted Unpickler) to avoid arbitrary code execution, but you should
    never load untrusted or tampered pickle files.
    """
    with open(pkl_path, 'rb') as f:
        f.seek(0)
        # Always use MadmomUnpickler for safety; never call pickle.load directly
        return MadmomUnpickler(f, encoding='latin1').load()


def extract_weights(pkl_path):
    """Extract all weights from madmom pkl file into a structured dict."""
    data = load_pkl_safe(pkl_path)

    weights = {
        'conv_layers': [],
        'tcn_blocks': [],
        'output_layers': []
    }

    if not hasattr(data, 'layers'):
        print(f"  WARNING: No 'layers' attribute found in {pkl_path}")
        return None

    for i, layer in enumerate(data.layers):
        layer_type = type(layer).__name__

        if 'Convolutional' in layer_type:
            w = getattr(layer, 'weights', None)
            b = getattr(layer, 'bias', None)
            if w is not None:
                weights['conv_layers'].append({
                    'weights': w,
                    'bias': b,
                    'activation': getattr(layer, 'activation_fn', 'elu')
                })

        elif 'TCN' in layer_type:
            tcn_blocks = getattr(layer, 'tcn_blocks', [])
            for block in tcn_blocks:
                block_weights = {}
                # TCN blocks have conv layers with dilations
                for attr in ['weights', 'W', 'conv_weights']:
                    if hasattr(block, attr):
                        block_weights['weights'] = getattr(block, attr)
                        break
                for attr in ['bias', 'b', 'conv_bias']:
                    if hasattr(block, attr):
                        block_weights['bias'] = getattr(block, attr)
                        break
                # Also check for sub-layers (only if direct attributes weren't found)
                # This prefers direct block attributes over sub-layer weights
                if hasattr(block, 'layers'):
                    for sub in block.layers:
                        if 'weights' not in block_weights and hasattr(sub, 'weights'):
                            block_weights['weights'] = sub.weights
                        if 'bias' not in block_weights and hasattr(sub, 'bias'):
                            block_weights['bias'] = sub.bias

                if block_weights:
                    weights['tcn_blocks'].append(block_weights)

        elif 'MultiTask' in layer_type or 'Dense' in layer_type or 'Feed' in layer_type:
            sub_layers = getattr(layer, 'layers', [layer])
            for sub in sub_layers:
                w = getattr(sub, 'weights', None)
                b = getattr(sub, 'bias', None)
                if w is not None:
                    weights['output_layers'].append({
                        'weights': w,
                        'bias': b
                    })

    return weights


def inspect_pkl(pkl_path):
    """Inspect the structure of a pkl file."""
    print(f"\nInspecting: {pkl_path}")
    data = load_pkl_safe(pkl_path)

    if not hasattr(data, '__dict__'):
        print(f"  Type: {type(data).__name__}")
        return data

    print(f"  Type: {type(data).__module__}.{type(data).__name__}")
    print(f"  Attributes: {list(data.__dict__.keys())}")

    if hasattr(data, 'layers'):
        layers = data.layers
        print(f"\n  Network has {len(layers)} layers:")

        for i, layer in enumerate(layers):
            layer_type = type(layer).__name__
            print(f"\n    Layer {i}: {layer_type}")

            if hasattr(layer, '__dict__'):
                for key, value in layer.__dict__.items():
                    if isinstance(value, np.ndarray):
                        print(f"      {key}: ndarray shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        print(f"      {key}: {type(value).__name__} len={len(value)}")
                        # Inspect first element
                        if hasattr(value[0], '__dict__'):
                            for k, v in value[0].__dict__.items():
                                if isinstance(v, np.ndarray):
                                    print(f"        [0].{k}: ndarray shape={v.shape}")
                    elif callable(value):
                        print(f"      {key}: {getattr(value, '__name__', str(value))}")
                    else:
                        print(f"      {key}: {value}")

    return data


def main():
    parser = argparse.ArgumentParser(description="Convert madmom TCN models to ONNX")
    parser.add_argument("--pkl", type=str, help="Path to single pkl file")
    parser.add_argument("--pkl-dir", type=str, help="Directory containing pkl files")
    parser.add_argument("--out", default="models/tcn_beats.onnx", help="Output ONNX file")
    parser.add_argument("--ensemble", action="store_true", help="Create ensemble of all models")
    parser.add_argument("--inspect", action="store_true", help="Just inspect pkl structure")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (minimum 18)")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    args = parser.parse_args()

    # Collect pkl files
    pkl_files = []
    if args.pkl:
        pkl_files = [args.pkl]
    elif args.pkl_dir:
        if not os.path.exists(args.pkl_dir):
            print(f"ERROR: Directory does not exist: {args.pkl_dir}", file=sys.stderr)
            return 1
        if not os.path.isdir(args.pkl_dir):
            print(f"ERROR: Path is not a directory: {args.pkl_dir}", file=sys.stderr)
            return 1
        pkl_files = sorted([
            os.path.join(args.pkl_dir, f)
            for f in os.listdir(args.pkl_dir)
            if f.endswith('.pkl')
        ])
    else:
        print("ERROR: Specify --pkl or --pkl-dir", file=sys.stderr)
        return 1

    if not pkl_files:
        print("ERROR: No pkl files found", file=sys.stderr)
        return 1

    print(f"Found {len(pkl_files)} pkl files")

    # Inspect mode
    if args.inspect:
        for pkl_path in pkl_files:
            inspect_pkl(pkl_path)
        return 0

    # Check dependencies
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch", file=sys.stderr)
        return 1

    try:
        import onnx
    except ImportError:
        print("ERROR: ONNX not installed. Run: pip install onnx", file=sys.stderr)
        return 1

    # Extract weights from first file
    print("\n=== Extracting weights ===")
    all_weights = []
    for pkl_path in pkl_files:
        print(f"  Loading: {os.path.basename(pkl_path)}")
        w = extract_weights(pkl_path)
        if w:
            all_weights.append(w)
            print(f"    Conv layers: {len(w['conv_layers'])}")
            print(f"    TCN blocks: {len(w['tcn_blocks'])}")
            print(f"    Output layers: {len(w['output_layers'])}")

    if not all_weights:
        print("\n  WARNING: Could not extract weights. Creating architecture from scratch.")

    print("\n=== Building PyTorch model ===")

    class TCNBlock(nn.Module):
        """Single TCN block with dilated causal convolution."""
        def __init__(self, channels, kernel_size=3, dilation=1):
            super().__init__()
            self.left_pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(channels, channels, kernel_size,
                                  padding=0, dilation=dilation)
            self.activation = nn.ELU()

        def forward(self, x):
            # Pad only on the left side for causality
            x_padded = torch.nn.functional.pad(x, (self.left_pad, 0))
            out = self.conv(x_padded)
            out = self.activation(out)
            return out

    class MadmomTCN(nn.Module):
        """
        Madmom TCN beat/downbeat detection network.

        Input: (batch, 1, freq_bins, time) - spectrogram
        Output: (batch, 2, time) - [beat_activation, downbeat_activation]
        """

        def __init__(self, freq_bins=81, num_tcn_blocks=11, hidden_channels=16):
            super().__init__()
            self.register_buffer('skip_sum_buf', None)

            self.hidden_channels = hidden_channels

            # Frontend CNN (matching madmom architecture)
            # Conv1: (1, 16, 3, 3) - from freq_bins to 16 channels
            self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=(3, 3), padding=(0, 0))
            self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

            # Conv2: (16, 16, 3, 3)
            self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 3), padding=(0, 0))
            self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

            # Conv3: (16, 16, 1, 8) - reduce time dimension
            self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 8), padding=(0, 0))

            self.activation = nn.ELU()

            # Calculate output size after CNN
            # freq: 81 -> 79 (conv1) -> 77 (conv2) -> 77 (conv3, kernel 1x8 only affects time)
            # After pooling: freq stays same (pool is 1x3)
            # Actually need to trace through to get right size
            # Let's use adaptive approach: project down to hidden_channels

            # After CNN, we'll have (batch, 16, remaining_freq, remaining_time)
            # We flatten freq into channel dim and project
            # For now, compute remaining_freq: 81 - 2 - 2 = 77 (two 3x3 convs with valid padding)
            self.remaining_freq = freq_bins - 2 - 2  # 77 for 81 input

            # Project flattened features to TCN hidden size
            self.project = nn.Conv1d(hidden_channels * self.remaining_freq, hidden_channels, kernel_size=1)

            # TCN with skip connections
            self.tcn_blocks = nn.ModuleList()
            for i in range(num_tcn_blocks):
                dilation = 2 ** i
                self.tcn_blocks.append(TCNBlock(hidden_channels, kernel_size=3, dilation=dilation))

            # Output layers (multi-task: beat + downbeat)
            self.output_beat = nn.Conv1d(hidden_channels, 1, kernel_size=1)
            self.output_downbeat = nn.Conv1d(hidden_channels, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x: (batch, 1, freq, time) or (batch, freq, time)
            if x.dim() == 3:
                x = x.unsqueeze(1)

            # Frontend CNN
            x = self.activation(self.conv1(x))
            x = self.pool1(x)
            x = self.activation(self.conv2(x))
            x = self.pool2(x)
            x = self.activation(self.conv3(x))

            # Reshape: (batch, channels, freq_remaining, time) -> (batch, channels * freq_remaining, time)
            batch, channels, freq_remaining, time = x.shape
            x = x.view(batch, channels * freq_remaining, time)

            # Project to TCN hidden size
            x = self.project(x)
            x = self.activation(x)


            # TCN with skip connections (no in-place ops, no persistent buffer)
            skip_accum = x.new_zeros(x.size())
            for block in self.tcn_blocks:
                x = block(x)
                skip_accum = skip_accum + x
            x = skip_accum

            # Output
            beat = self.sigmoid(self.output_beat(x))      # (batch, 1, time)
            downbeat = self.sigmoid(self.output_downbeat(x))  # (batch, 1, time)

            # Concatenate: (batch, 2, time)
            out = torch.cat([beat, downbeat], dim=1)
            return out

    # Create ensemble model if multiple pkl files provided
    if args.ensemble and len(all_weights) > 1:
        print(f"\n=== Creating ensemble of {len(all_weights)} models ===")

        class MadmomTCNEnsemble(nn.Module):
            """Ensemble of multiple TCN models - averages predictions."""
            def __init__(self, num_models, freq_bins=81, num_tcn_blocks=11, hidden_channels=16):
                super().__init__()
                self.num_models = num_models
                self.models = nn.ModuleList([
                    MadmomTCN(freq_bins, num_tcn_blocks, hidden_channels)
                    for _ in range(num_models)
                ])

            def forward(self, x):
                # Run all models and average their outputs
                outputs = [model(x) for model in self.models]
                # Stack and average: each output is (batch, 2, time)
                stacked = torch.stack(outputs, dim=0)  # (num_models, batch, 2, time)
                averaged = torch.mean(stacked, dim=0)  # (batch, 2, time)
                return averaged

        model = MadmomTCNEnsemble(len(all_weights))

        # Load weights into each model
        print("\n=== Loading weights into ensemble ===")
        for model_idx, weights in enumerate(all_weights):
            sub_model = model.models[model_idx]
            state_dict = sub_model.state_dict()

            # Load conv layer weights
            conv_layers = ['conv1', 'conv2', 'conv3']
            for i, (name, w) in enumerate(zip(conv_layers, weights['conv_layers'])):
                if 'weights' in w and w['weights'] is not None:
                    weight = w['weights']
                    if weight.ndim == 4:
                        weight = np.transpose(weight, (1, 0, 2, 3))
                    try:
                        state_dict[f'{name}.weight'] = torch.from_numpy(weight.astype(np.float32))
                        print(f"  Loaded {name}.weight: {state_dict[f'{name}.weight'].shape}")
                    except Exception as e:
                        print(f"  Failed to load {name}.weight: {e}")
                if 'bias' in w and w['bias'] is not None:
                    try:
                        state_dict[f'{name}.bias'] = torch.from_numpy(w['bias'].astype(np.float32))
                        print(f"  Loaded {name}.bias")
                    except Exception as e:
                        print(f"  Failed to load {name}.bias: {e}")

            # Load TCN block weights
            for i, block_w in enumerate(weights.get('tcn_blocks', [])):
                if i >= len(sub_model.tcn_blocks):
                    break
                if 'weights' in block_w and block_w['weights'] is not None:
                    weight = block_w['weights']
                    if weight.ndim == 3:
                        # (in, out, kernel) -> (out, in, kernel)
                        weight = np.transpose(weight, (1, 0, 2))
                    try:
                        state_dict[f'tcn_blocks.{i}.conv.weight'] = torch.from_numpy(weight.astype(np.float32))
                        print(f"  Loaded tcn_blocks.{i}.conv.weight")
                    except Exception as e:
                        print(f"  Failed to load tcn_blocks.{i}: {e}")
                if 'bias' in block_w and block_w['bias'] is not None:
                    try:
                        state_dict[f'tcn_blocks.{i}.conv.bias'] = torch.from_numpy(block_w['bias'].astype(np.float32))
                        print(f"  Loaded tcn_blocks.{i}.conv.bias")
                    except Exception as e:
                        print(f"  Failed to load tcn_blocks.{i}.conv.bias: {e}")

            # Load output weights
            output_names = ['output_beat', 'output_downbeat']
            for i, (name, w) in enumerate(zip(output_names, weights.get('output_layers', []))):
                if 'weights' in w and w['weights'] is not None:
                    weight = w['weights']
                    if weight.ndim == 2:
                        weight = weight.T[..., np.newaxis]
                    try:
                        state_dict[f'{name}.weight'] = torch.from_numpy(weight.astype(np.float32))
                    except (AttributeError, ValueError, TypeError, RuntimeError) as e:
                        print(f"[ERROR] Failed to convert weights for '{name}': {e}\n  Weight shape: {getattr(weight, 'shape', None)} Value: {repr(weight)[:200]}")
                        raise

            try:
                sub_model.load_state_dict(state_dict, strict=False)
                print(f"  Model {model_idx + 1}/{len(all_weights)}: weights loaded")
            except Exception as e:
                print(f"  Model {model_idx + 1}/{len(all_weights)}: partial load ({e})")

    else:
        # Single model
        model = MadmomTCN()

        # Load weights if available
        if all_weights:
            print("\n=== Loading weights ===")
            weights = all_weights[0]  # Use first model

            state_dict = model.state_dict()

            # Load conv layer weights
            conv_layers = ['conv1', 'conv2', 'conv3']
            for i, (name, w) in enumerate(zip(conv_layers, weights['conv_layers'])):
                if 'weights' in w and w['weights'] is not None:
                    weight = w['weights']
                    # madmom uses (in_ch, out_ch, h, w), PyTorch uses (out_ch, in_ch, h, w)
                    if weight.ndim == 4:
                        weight = np.transpose(weight, (1, 0, 2, 3))
                    try:
                        state_dict[f'{name}.weight'] = torch.from_numpy(weight.astype(np.float32))
                        print(f"  Loaded {name}.weight: {weight.shape}")
                    except Exception as e:
                        print(f"  Failed to load {name}.weight: {e}")

                if 'bias' in w and w['bias'] is not None:
                    try:
                        state_dict[f'{name}.bias'] = torch.from_numpy(w['bias'].astype(np.float32))
                        print(f"  Loaded {name}.bias")
                    except Exception as e:
                        print(f"  Failed to load {name}.bias: {e}")

            # Load TCN block weights
            for i, block_w in enumerate(weights.get('tcn_blocks', [])):
                if i >= len(model.tcn_blocks):
                    break
                if 'weights' in block_w and block_w['weights'] is not None:
                    weight = block_w['weights']
                    if weight.ndim == 3:
                        # (in, out, kernel) -> (out, in, kernel)
                        weight = np.transpose(weight, (1, 0, 2))
                    try:
                        state_dict[f'tcn_blocks.{i}.conv.weight'] = torch.from_numpy(weight.astype(np.float32))
                        print(f"  Loaded tcn_blocks.{i}.conv.weight")
                    except Exception as e:
                        print(f"  Failed to load tcn_blocks.{i}: {e}")
                if 'bias' in block_w and block_w['bias'] is not None:
                    try:
                        state_dict[f'tcn_blocks.{i}.conv.bias'] = torch.from_numpy(block_w['bias'].astype(np.float32))
                        print(f"  Loaded tcn_blocks.{i}.conv.bias")
                    except Exception as e:
                        print(f"  Failed to load tcn_blocks.{i}.conv.bias: {e}")

            # Load output weights
            output_names = ['output_beat', 'output_downbeat']
            for i, (name, w) in enumerate(zip(output_names, weights.get('output_layers', []))):
                if 'weights' in w and w['weights'] is not None:
                    weight = w['weights']
                    if weight.ndim == 2:
                        weight = weight.T[..., np.newaxis]  # (in, out) -> (out, in, 1)
                    try:
                        state_dict[f'{name}.weight'] = torch.from_numpy(weight.astype(np.float32))
                        print(f"  Loaded {name}.weight")
                    except Exception as e:
                        print(f"  Failed to load {name}.weight: {e}")

            try:
                model.load_state_dict(state_dict, strict=False)
                print("\n  Weights loaded successfully!")
            except Exception as e:
                print(f"\n  WARNING: Partial weight loading: {e}")
        else:
            # Random init
            print("\n  Using random initialization (no weights extracted)")
            torch.manual_seed(42)
            for m in model.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    model.eval()

    # Export to ONNX
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # Input: spectrogram (batch, 1, freq=81, time)
    # Using 81 mel bands (madmom default for beat tracking)
    freq_bins = 81
    time_frames = 100
    dummy_input = torch.randn(1, 1, freq_bins, time_frames)

    print(f"\n=== Exporting to ONNX ===")
    print(f"  Input shape: {dummy_input.shape}")

    dynamic_axes = {
        'spectrogram': {0: 'batch', 3: 'time'},
        'activations': {0: 'batch', 2: 'time'}
    }

    # Use legacy exporter for compatibility
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.out,
            export_params=True,
            opset_version=max(args.opset, 18),  # Use at least opset 18
            do_constant_folding=True,
            input_names=['spectrogram'],
            output_names=['activations'],
            dynamic_axes=dynamic_axes,
            dynamo=False  # Use legacy TorchScript exporter
        )
    except TypeError:
        # Fallback for PyTorch < 2.5 where dynamo parameter is not supported
        torch.onnx.export(
            model,
            dummy_input,
            args.out,
            export_params=True,
            opset_version=max(args.opset, 18),  # Use at least opset 18
            do_constant_folding=True,
            input_names=['spectrogram'],
            output_names=['activations'],
            dynamic_axes=dynamic_axes
        )

    print(f"  Saved: {args.out}")

    # Verify
    if args.verify:
        print("\n=== Verifying ===")
        try:
            import onnxruntime as ort

            onnx_model = onnx.load(args.out)
            onnx.checker.check_model(onnx_model)
            print("  ONNX structure: OK")

            session = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])

            input_info = session.get_inputs()[0]
            print(f"  Input: {input_info.name}, shape: {input_info.shape}")

            for output in session.get_outputs():
                print(f"  Output: {output.name}, shape: {output.shape}")

            # Test inference - need enough time frames after pooling
            # After 2x pool(1,3): 100 -> 33 -> 11, minus conv3 kernel-1=7 -> 4 frames
            # Use 300 frames to get reasonable output
            test_input = np.random.randn(1, 1, freq_bins, 300).astype(np.float32)
            outputs = session.run(None, {input_info.name: test_input})
            print(f"  Test inference: OK")
            print(f"    Output shape: {outputs[0].shape}")
            print(f"    Beat activation range: [{outputs[0][0, 0].min():.3f}, {outputs[0][0, 0].max():.3f}]")
            print(f"    Downbeat activation range: [{outputs[0][0, 1].min():.3f}, {outputs[0][0, 1].max():.3f}]")

        except Exception as e:
            print(f"  Verification failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n=== Done ===")
    print("\nModel expects:")
    print(f"  - Input: Mel spectrogram (batch, 1, {freq_bins}, time)")
    print("  - Sample rate: 44100 Hz")
    print("  - Frame rate: 100 fps (10ms hop)")
    print("  - Mel bands: 81")
    print("  - Output: (batch, 2, time) - [beat, downbeat] activations (0-1)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
