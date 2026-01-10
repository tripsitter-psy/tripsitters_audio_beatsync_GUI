#pragma once

#include "BeatGrid.h"
#include <string>

namespace BeatSync {

// Prototype ONNX-based beat detector. This is a placeholder to wire the
// ONNX runtime into the project. For now, it reads an optional sidecar
// "<audioFilePath>.onnx.json" (same format as BeatNet) to simulate ONNX output.
class OnnxBeatDetector {
public:
    OnnxBeatDetector() = default;

    // Analyze audio file path and return beat grid. Returns empty grid on failure.
    BeatGrid analyze(const std::string& audioFilePath);
};

} // namespace BeatSync
