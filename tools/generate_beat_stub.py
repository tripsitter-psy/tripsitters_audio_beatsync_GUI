import onnx
from onnx import helper, TensorProto
import onnx.checker
from pathlib import Path


def build_and_save_beat_stub():
    """Build and save beat stub ONNX models for testing."""
    # Create a constant tensor node with beat times
    beats = [0.5, 1.0, 1.5]
    const_tensor = helper.make_tensor(name='const_beats', data_type=TensorProto.FLOAT, dims=[len(beats)], vals=beats)
    const_node = helper.make_node('Constant', inputs=[], outputs=['beats_out'], value=const_tensor)

    # Create output value info
    output = helper.make_tensor_value_info('beats_out', TensorProto.FLOAT, [len(beats)])

    # Create graph
    graph = helper.make_graph(nodes=[const_node], name='BeatStubGraph', inputs=[], outputs=[output], initializer=[])

    # Create model (force opset 12 for ONNX Runtime compatibility)
    opset12 = helper.make_operatorsetid('', 12)

    model = helper.make_model(graph, opset_imports=[opset12])

    # Ensure the model IR version is compatible with the ONNX Runtime in vcpkg (set to 7 for opset 12)
    model.ir_version = 7

    # Validate ONNX model before saving
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"ERROR: ONNX model validation failed, not writing files.\n{e}")
        raise

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / 'tests' / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    # Write two identical ONNX files:
    # - beat_reference.onnx is the immutable golden reference for validation
    # - beat_stub.onnx is the editable test fixture used in CI and local tests
    beat_stub_path = output_dir / 'beat_stub.onnx'
    beat_reference_path = output_dir / 'beat_reference.onnx'

    with open(beat_stub_path, 'wb') as f:
        f.write(model.SerializeToString())
    with open(beat_reference_path, 'wb') as f:
        f.write(model.SerializeToString())

    print(f'Saved {beat_stub_path} and {beat_reference_path}')


if __name__ == "__main__":
    build_and_save_beat_stub()
