#pragma once

#include <vector>
#include <cstddef>

namespace BeatSync {

// Frequency band energy buckets extracted via FFT.
struct FrequencyBands {
    float bass;  // 20-200 Hz
    float mids;  // 200-2000 Hz
    float highs; // 2000-20000 Hz
};

// Simple FFT-based analyzer extracting per-frame frequency band energies.
// Used to drive audio-reactive video effects.
class FFTAnalyzer {
public:
    FFTAnalyzer() = default;

    // Analyze a window of mono float samples and return band energies.
    FrequencyBands analyze(const std::vector<float>& samples, int sampleRate);

private:
    // TODO: add internal buffers, FFT plan (e.g., FFTW or pffft)
};

} // namespace BeatSync
