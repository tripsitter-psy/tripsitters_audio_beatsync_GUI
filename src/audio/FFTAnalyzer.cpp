#include "FFTAnalyzer.h"

namespace BeatSync {

FrequencyBands FFTAnalyzer::analyze(const std::vector<float>& samples, int sampleRate) {
    (void)samples;
    (void)sampleRate;

    FrequencyBands bands{0.0f, 0.0f, 0.0f};
    // TODO: run FFT, bucket magnitudes into bass/mids/highs
    return bands;
}

} // namespace BeatSync
