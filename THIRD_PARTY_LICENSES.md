# Third-Party Licenses

MTV TripSitter includes software from the following third-party projects.

---

## FFmpeg

**License:** LGPL 2.1 or later (with optional GPL components)
**Website:** https://ffmpeg.org/
**Used for:** Audio/video decoding, encoding, and processing

```
FFmpeg is free software licensed under the LGPL or GPL depending on your choice of
configuration options. If you use FFmpeg or its constituent libraries, you must adhere
to the terms of the licenses.

This software uses libraries from the FFmpeg project under the LGPLv2.1.
See https://www.ffmpeg.org/legal.html for more information.
```

---

## ONNX Runtime

**License:** MIT
**Website:** https://onnxruntime.ai/
**Used for:** Neural network inference for AI beat detection

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## NVIDIA TensorRT (Optional)

**License:** NVIDIA Software License Agreement
**Website:** https://developer.nvidia.com/tensorrt
**Used for:** Accelerated neural network inference on NVIDIA RTX GPUs

```
TensorRT is proprietary software distributed by NVIDIA Corporation under the
NVIDIA Software License Agreement. Use of TensorRT is subject to the terms
and conditions of that agreement.

See https://docs.nvidia.com/deeplearning/tensorrt/sla/index.html for the
full license agreement.

Note: TensorRT components are optional and only included in builds that
explicitly enable RTX GPU acceleration.
```

---

## NVIDIA CUDA Toolkit (Optional)

**License:** NVIDIA CUDA Toolkit EULA
**Website:** https://developer.nvidia.com/cuda-toolkit
**Used for:** GPU acceleration for neural network inference

```
The CUDA Toolkit is proprietary software distributed by NVIDIA Corporation.
Use is subject to the NVIDIA CUDA Toolkit End User License Agreement.

See https://docs.nvidia.com/cuda/eula/index.html for the full license.
```

---

## AudioFlux (Optional)

**License:** MIT
**Website:** https://github.com/libAudioFlux/audioFlux
**Used for:** Spectral flux beat detection and audio signal processing

```
MIT License

Copyright (c) 2023 audioFlux

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## FFTW3 (via AudioFlux)

**License:** GPL 2 or later (or commercial license)
**Website:** https://www.fftw.org/
**Used for:** Fast Fourier Transform computations in AudioFlux

```
FFTW is free software; you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

FFTW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

Commercial licenses are also available from MIT.
See https://www.fftw.org/doc/License-and-Copyright.html
```

---

## Abseil (via ONNX Runtime)

**License:** Apache 2.0
**Website:** https://abseil.io/
**Used for:** C++ common libraries (ONNX Runtime dependency)

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
[Full Apache 2.0 license text available at the URL above]
```

---

## Protocol Buffers (via ONNX Runtime)

**License:** BSD 3-Clause
**Website:** https://protobuf.dev/
**Used for:** Data serialization (ONNX Runtime dependency)

```
Copyright 2008 Google Inc.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## RE2 (via ONNX Runtime)

**License:** BSD 3-Clause
**Website:** https://github.com/google/re2
**Used for:** Regular expression library (ONNX Runtime dependency)

```
Copyright (c) 2009 The RE2 Authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
   * Neither the name of Google Inc. nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## Catch2 (Development Only)

**License:** Boost Software License 1.0
**Website:** https://github.com/catchorg/Catch2
**Used for:** Unit testing framework (not distributed in release builds)

```
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

---

## Unreal Engine

**License:** Unreal Engine EULA
**Website:** https://www.unrealengine.com/
**Used for:** GUI framework (TripSitter application)

```
The TripSitter GUI is built using Unreal Engine technology.
Unreal Engine is a trademark or registered trademark of Epic Games, Inc.
in the United States of America and elsewhere.

Use of Unreal Engine is subject to the Unreal Engine End User License Agreement.
See https://www.unrealengine.com/eula for the full license agreement.

Note: The distributed TripSitter.exe binary includes Unreal Engine runtime
components. Distribution is permitted under the Unreal Engine EULA for
applications built with the engine.
```

---

## ONNX Models

The following ONNX models may be included:

### Demucs (Meta AI)
**License:** MIT
**Source:** https://github.com/facebookresearch/demucs
**Used for:** Audio source separation (stem separation)

### BeatNet
**License:** MIT
**Source:** https://github.com/mjhydri/BeatNet
**Used for:** Beat and downbeat tracking

### TCN Beat Detector
**License:** MIT
**Source:** Based on Temporal Convolutional Network architectures
**Used for:** Beat detection via temporal convolution

---

## Summary

| Component | License | Required |
|-----------|---------|----------|
| FFmpeg | LGPL 2.1+ | Yes |
| ONNX Runtime | MIT | Yes |
| Abseil | Apache 2.0 | Yes |
| Protocol Buffers | BSD 3-Clause | Yes |
| RE2 | BSD 3-Clause | Yes |
| TensorRT | NVIDIA EULA | Optional |
| CUDA | NVIDIA EULA | Optional |
| AudioFlux | MIT | Optional |
| FFTW3 | GPL 2+ | Optional |
| Unreal Engine | Epic Games EULA | Yes |
