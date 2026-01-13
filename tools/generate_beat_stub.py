import onnx
from onnx import helper, TensorProto

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
# Ensure the model IR version is compatible with the ONNX Runtime in vcpkg (set to 12)
model.ir_version = 12

# Save model
with open('tests/models/beat_stub.onnx', 'wb') as f:
    f.write(model.SerializeToString())
with open('tests/models/beat_reference.onnx', 'wb') as f:
    f.write(model.SerializeToString())

print('Saved tests/models/beat_stub.onnx and tests/models/beat_reference.onnx')
