#pragma once

#include <vector>

namespace BeatSync {

// Simple spectral-flux based beat detector.
// Usage: call detectBeatsFromWaveform() with mono float samples (range [-1,1])
// and the sample rate. Returns beat times in seconds.
std::vector<double> detectBeatsFromWaveform(const std::vector<float>& samples, int sampleRate,
                                            int windowSize = 2048, int hopSize = 256,
                                            double smoothSigma = 1.0, double thresholdFactor = 1.2,
                                            double minBeatDistanceSeconds = 0.2);

} // namespace BeatSync
