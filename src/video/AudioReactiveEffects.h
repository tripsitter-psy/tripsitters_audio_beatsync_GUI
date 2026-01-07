#pragma once

#include "../audio/FFTAnalyzer.h"
#include <string>
#include <vector>

namespace BeatSync {

// Parameters derived from FFT bands to drive video filter adjustments.
struct ReactiveParams {
    float zoomIntensity;   // driven by bass
    float saturation;      // driven by mids
    float brightness;      // driven by highs
};

// Maps FFT frequency band energies to FFmpeg filter parameters for audio-
// reactive video effects. Generates per-frame filter strings or keyframes.
class AudioReactiveEffects {
public:
    AudioReactiveEffects() = default;

    // Convert frequency bands to reactive parameters (normalized 0-1).
    ReactiveParams bandsToParams(const FrequencyBands& bands);

    // Generate an FFmpeg filter snippet for a given frame's reactive params.
    std::string buildFilterSnippet(const ReactiveParams& params);

private:
    // TODO: add smoothing, sensitivity curves
};

} // namespace BeatSync
