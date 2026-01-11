# vcpkg onnxruntime 1.23.2 causes heap corruption on Windows

## Issue Summary

The vcpkg-built ONNX Runtime 1.23.2 (`onnxruntime:x64-windows`) crashes with heap corruption (exit code 0xc0000374) during basic inference operations on Windows. The same code works correctly with official Microsoft ONNX Runtime 1.23.2 binaries downloaded from GitHub releases.

## Environment

- **OS**: Windows 11 (build 26200)
- **Compiler**: MSVC 19.38.33145.0 (Visual Studio 2022)
- **vcpkg baseline**: `a48bbf436be0867d0eadc42077f280f3123e3698`
- **onnxruntime version**: 1.23.2 (from vcpkg)
- **Architecture**: x64

## Reproduction Steps

1. Create a minimal ONNX model (constant output):
```python
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

graph = helper.make_graph(
    nodes=[helper.make_node('Constant', [], ['out'],
        value=numpy_helper.from_array(np.array([0.5, 1.0, 1.5], dtype=np.float32)))],
    name='TestGraph',
    inputs=[],
    outputs=[helper.make_tensor_value_info('out', TensorProto.FLOAT, [3])]
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 9
onnx.save(model, 'test_model.onnx')
```

2. Create a minimal C++ test:
```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, L"test_model.onnx", session_options);

    // Get output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto namePtr = session.GetOutputNameAllocated(0, allocator);
    std::string outputName = namePtr.get();
    const char* outputNames[] = {outputName.c_str()};

    // Run inference
    auto outputs = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputNames, 1);
    std::cout << "Success!" << std::endl;
    return 0;
}
```

3. Build with vcpkg ONNX Runtime:
```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Debug
```

4. Run the test - it crashes with exit code 0xc0000374 (STATUS_HEAP_CORRUPTION)

## Expected Behavior

The test should run successfully and print "Success!"

## Actual Behavior

The application crashes with exit code 0xc0000374 (STATUS_HEAP_CORRUPTION) before any output is printed. The crash occurs during ONNX Runtime initialization or early session creation.

## Verification with Official Binaries

Using the **exact same compiled test executable** but replacing vcpkg DLLs with official Microsoft binaries:

1. Download from: https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip
2. Copy `onnxruntime.dll` and `onnxruntime_providers_shared.dll` to the test directory
3. Run the test - **it works correctly**

| Binary Source | Version | Result |
|---------------|---------|--------|
| vcpkg | 1.23.2 | **CRASH** (0xc0000374) |
| Official Microsoft | 1.23.2 | **SUCCESS** |

## Analysis

This indicates the issue is specific to the vcpkg build configuration, not the ONNX Runtime source code itself. Possible causes:
- Incorrect compiler flags in vcpkg port
- Missing or incorrect debug/release runtime library linkage
- Heap allocator mismatch between vcpkg build and consuming application

## vcpkg.json

```json
{
  "name": "test",
  "version": "1.0.0",
  "dependencies": [
    {
      "name": "onnxruntime",
      "features": ["cuda"],
      "platform": "windows"
    }
  ],
  "builtin-baseline": "a48bbf436be0867d0eadc42077f280f3123e3698"
}
```

## Additional Context

- Crash occurs with both CPU-only and CUDA-enabled builds
- Crash happens during DLL initialization or very early in main()
- No output is printed before the crash
- The issue persists across multiple test scenarios (200 iterations of inference)
