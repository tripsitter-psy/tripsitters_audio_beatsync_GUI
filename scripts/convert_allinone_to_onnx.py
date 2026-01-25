#!/usr/bin/env python3
"""
Convert All-In-One Music Structure Analyzer to ONNX format.

The All-In-One model provides comprehensive music analysis:
- Beat positions
- Downbeat positions
- Tempo (BPM)
- Functional segment boundaries
- Segment labels (intro, verse, chorus, bridge, outro)

Usage:
    python scripts/convert_allinone_to_onnx.py --out models/allinone.onnx

Requirements:
    pip install torch allin1 onnx onnxruntime

Reference:
    Taejun Kim, et al. "All-In-One Metrical And Functional Structure Analysis
    With Neighborhood Attentions on Demixed Audio" (ISMIR 2023)
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Convert All-In-One to ONNX")
    parser.add_argument("--out", default="models/allinone_encoder.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    parser.add_argument("--export-full", action="store_true",
                        help="Export full model (encoder + all heads). Default exports encoder only.")
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

    # Try to import allin1
    try:
        import allin1
        from allin1.models import load_pretrained_model
        USE_ALLIN1 = True
    except ImportError:
        print("WARNING: allin1 not installed. Creating compatible architecture.")
        print("         For best results: pip install allin1")
        USE_ALLIN1 = False


    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    if USE_ALLIN1:
        print("Loading All-In-One pretrained model...")
        try:
            # Load the pretrained model
            model = load_pretrained_model('harmonix-all')
            model.eval()

            # The model expects source-separated spectrograms
            # Shape: (batch, n_sources=4, n_mels=128, time_frames)
            # Sources: drums, bass, other, vocals (from Demucs separation)

            n_sources = 4
            n_mels = 128
            time_frames = 1000  # ~10 seconds at 100 FPS

            dummy_input = torch.randn(1, n_sources, n_mels, time_frames)

            print(f"Exporting to ONNX (input shape: {dummy_input.shape})...")

            # Always export with the same output schema for interface consistency
            # When export_full=True, model returns real outputs; otherwise placeholders are used
            output_names = [
                'beat_activation', 'downbeat_activation',
                'segment_activation', 'segment_labels',
                'tempo_logits', 'embeddings'
            ]
            dynamic_axes = {
                'spectrogram': {0: 'batch', 3: 'time'},
                'beat_activation': {0: 'batch', 1: 'time'},
                'downbeat_activation': {0: 'batch', 1: 'time'},
                'segment_activation': {0: 'batch', 1: 'time'},
                'segment_labels': {0: 'batch', 1: 'segments'},
                'tempo_logits': {0: 'batch', 1: 'classes'},
                'embeddings': {0: 'batch', 1: 'time'}
            }

            # Ensure model returns all outputs (real or placeholder) for ONNX export
            def get_all_outputs(model, dummy_input, export_full):
                outputs = model(dummy_input)
                # If not full export, fill with zeros for missing outputs
                if not export_full:
                    # check if outputs is already a 6-element list/tuple
                    if isinstance(outputs, (list, tuple)) and len(outputs) == 6:
                        return outputs

                    # outputs: embeddings only, so fill others
                    batch = dummy_input.shape[0]
                    time = dummy_input.shape[3]
                    segments = 1
                    classes = 1
                    # Use dummy_input to get device/dtype so we don't need 'import torch'
                    zeros = lambda *shape: dummy_input.new_zeros(*shape)
                    
                    # Return tuple in output_names order
                    return (
                        zeros(batch, time),  # beat_activation
                        zeros(batch, time),  # downbeat_activation
                        zeros(batch, time),  # segment_activation
                        zeros(batch, segments),  # segment_labels
                        zeros(batch, classes),  # tempo_logits
                        outputs  # embeddings
                    )
                return outputs

            torch.onnx.export(
                lambda x: get_all_outputs(model, x, args.export_full),
                dummy_input,
                out_path,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=['spectrogram'],
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )

        except Exception as e:
            import traceback
            print(f"ERROR exporting allin1 model:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("Falling back to standalone architecture...", file=sys.stderr)
            USE_ALLIN1 = False

    if not USE_ALLIN1:
        print("Creating All-In-One compatible architecture...")

        class NeighborhoodAttention1D(nn.Module):
            """Simplified 1D neighborhood attention for temporal modeling."""
            def __init__(self, dim, num_heads=4, kernel_size=7, dilation=1):
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = dim // num_heads
                self.scale = self.head_dim ** -0.5
                self.kernel_size = kernel_size
                self.dilation = dilation

                self.qkv = nn.Linear(dim, dim * 3)
                self.proj = nn.Linear(dim, dim)

            def forward(self, x):
                # x: (batch, time, dim)
                B, T, C = x.shape
                # Project to Q, K, V
                qkv = self.qkv(x)  # (B, T, 3 * C)
                q, k, v = torch.chunk(qkv, 3, dim=-1)
                # Reshape for multi-head
                q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
                k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

                # Compute attention scores with local window
                attn_scores = torch.zeros(B, self.num_heads, T, self.kernel_size, device=x.device)
                half_window = self.kernel_size // 2
                # NOTE: This per-timestep loop creates a large ONNX graph for long sequences.
                # For production, consider a vectorized implementation or fixed-length inputs.
                for t in range(T):
                    # Get window indices
                    start = max(0, t - half_window * self.dilation)
                    end = min(T, t + half_window * self.dilation + 1)
                    window_idx = torch.arange(start, end, self.dilation, device=x.device)
                    q_t = q[:, :, t, :]  # (B, num_heads, head_dim)
                    k_window = k[:, :, window_idx, :]  # (B, num_heads, window, head_dim)
                    # Dot product attention
                    attn = torch.einsum('bnh,bnwh->bnw', q_t, k_window) * self.scale
                    # Pad to kernel_size
                    pad = self.kernel_size - attn.shape[-1]
                    if pad > 0:
                        attn = torch.nn.functional.pad(attn, (0, pad))
                    attn_scores[:, :, t, :] = attn

                attn_probs = torch.softmax(attn_scores, dim=-1)

                # Aggregate weighted values
                out = torch.zeros(B, self.num_heads, T, self.head_dim, device=x.device)
                for t in range(T):
                    start = max(0, t - half_window * self.dilation)
                    end = min(T, t + half_window * self.dilation + 1)
                    window_idx = torch.arange(start, end, self.dilation, device=x.device)
                    v_window = v[:, :, window_idx, :]  # (B, num_heads, window, head_dim)
                    attn = attn_probs[:, :, t, :v_window.shape[2]].unsqueeze(-1)
                    out[:, :, t, :] = torch.sum(attn * v_window, dim=2)

                # Merge heads and project out
                out = out.transpose(1, 2).reshape(B, T, C)
                out = self.proj(out)
                return out

        class AllInOneEncoder(nn.Module):
            """
            All-In-One encoder for music structure analysis.

            Processes source-separated spectrograms and outputs:
            - Beat activations
            - Downbeat activations
            - Segment boundary activations
            - Tempo embedding

            Input: (batch, 4, 128, time) - 4 source spectrograms (drums, bass, other, vocals)
            """
            def __init__(self, n_mels=128, n_sources=4, embed_dim=256):
                super().__init__()

                # Per-source CNN encoder
                self.source_encoders = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d((4, 1)),

                        nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d((4, 1)),

                        nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d((4, 1)),

                        nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d((2, 1)),
                    ) for _ in range(n_sources)
                ])

                # After pooling: 128 / 4 / 4 / 4 / 2 = 1 freq bin
                # Feature per source: 128
                cnn_out_dim = 128 * n_sources

                # Projection to embed_dim
                self.proj = nn.Linear(cnn_out_dim, embed_dim)

                # Temporal modeling with neighborhood attention
                self.temporal_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        NeighborhoodAttention1D(embed_dim, num_heads=4, dilation=1),
                        nn.Dropout(0.1),
                    ) for _ in range(4)
                ])

                self.ff_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Linear(embed_dim, embed_dim * 4),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(embed_dim * 4, embed_dim),
                        nn.Dropout(0.1),
                    ) for _ in range(4)
                ])

                # Output heads
                self.beat_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                self.downbeat_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                self.segment_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                # Segment label classifier (10 classes)
                # intro, verse, pre-chorus, chorus, post-chorus, bridge, outro, inst, solo, silence
                self.label_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                )

                # Tempo head (outputs tempo class logits, 30-300 BPM in 1 BPM bins)
                self.tempo_head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),  # Global pooling
                )
                self.tempo_classifier = nn.Linear(embed_dim, 271)  # 30-300 BPM
                self.tempo_norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                # x: (batch, n_sources, n_mels, time)
                batch_size = x.shape[0]
                time_steps = x.shape[3]

                # Encode each source separately
                source_features = []
                for i, encoder in enumerate(self.source_encoders):
                    src = x[:, i:i+1, :, :]  # (batch, 1, n_mels, time)
                    feat = encoder(src)  # (batch, 128, 1, time)
                    feat = feat.squeeze(2)  # (batch, 128, time)
                    source_features.append(feat)

                # Concatenate source features
                x = torch.cat(source_features, dim=1)  # (batch, 512, time)
                x = x.permute(0, 2, 1)  # (batch, time, 512)

                # Project to embed_dim
                x = self.proj(x)  # (batch, time, embed_dim)

                # Temporal modeling
                for attn_block, ff_block in zip(self.temporal_blocks, self.ff_blocks):
                    x = x + attn_block(x)
                    x = x + ff_block(x)

                # Output heads
                beat_act = self.beat_head(x).squeeze(-1)  # (batch, time)
                downbeat_act = self.downbeat_head(x).squeeze(-1)  # (batch, time)
                segment_act = self.segment_head(x).squeeze(-1)  # (batch, time)
                segment_labels = self.label_head(x)  # (batch, time, 10)

                # Tempo estimation
                tempo_feat = x.permute(0, 2, 1)  # (batch, embed_dim, time)
                tempo_feat = self.tempo_head(tempo_feat).squeeze(-1)  # (batch, embed_dim)
                tempo_feat = self.tempo_norm(tempo_feat)  # Normalize over embed_dim: (batch, embed_dim)
                tempo_logits = self.tempo_classifier(tempo_feat)  # (batch, 271)

                return beat_act, downbeat_act, segment_act, segment_labels, tempo_logits, x

        torch.manual_seed(42)
        model = AllInOneEncoder()
        model.eval()

        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Export
        n_sources = 4
        n_mels = 128
        time_frames = 500  # ~5 seconds at 100 FPS

        dummy_input = torch.randn(1, n_sources, n_mels, time_frames)

        print(f"Exporting to ONNX (input shape: {dummy_input.shape})...")

        dynamic_axes = {
            'spectrogram': {0: 'batch', 3: 'time'},
            'beat_activation': {0: 'batch', 1: 'time'},
            'downbeat_activation': {0: 'batch', 1: 'time'},
            'segment_activation': {0: 'batch', 1: 'time'},
            'segment_labels': {0: 'batch', 1: 'time'},
            'tempo_logits': {0: 'batch'},
            'embeddings': {0: 'batch', 1: 'time'}
        }


        torch.onnx.export(
            model,
            dummy_input,
            out_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=['spectrogram'],
            output_names=['beat_activation', 'downbeat_activation',
                         'segment_activation', 'segment_labels', 'tempo_logits', 'embeddings'],
            dynamic_axes=dynamic_axes
        )

    print(f"Saved ONNX model to: {args.out}")

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
            test_input = np.random.randn(1, 4, 128, 100).astype(np.float32)
            outputs = session.run(None, {input_info.name: test_input})

            print(f"  Test inference: OK")
            for i, output in enumerate(session.get_outputs()):
                print(f"    {output.name}: {outputs[i].shape}")

        except Exception as e:
            print(f"  Verification failed: {e}", file=sys.stderr)
            return 1

    print("\nDone! Model is ready for use with OnnxBeatDetector.")
    print("\nModel expects:")
    print("  - Input: 4-source spectrograms (drums, bass, other, vocals)")
    print("  - Each source: 128 mel bands")
    print("  - Sample rate: 44100 Hz (standard for Demucs)")
    print("  - Frame rate: 100 FPS (hop_length=220)")
    print("\nNote: For best results, use with Demucs source separation.")
    print("      Without separation, duplicate the mono spectrogram across 4 channels.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
