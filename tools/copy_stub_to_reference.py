import shutil
shutil.copyfile('tests/models/beat_stub.onnx', 'tests/models/beat_reference.onnx')
print('Copied beat_stub.onnx -> beat_reference.onnx')
