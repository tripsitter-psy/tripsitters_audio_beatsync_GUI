#include "FFTAnalyzer.h"
#include <complex>
#include <cmath>

namespace BeatSync {

// Naive DFT-based implementation (sufficient for small analysis windows).
static std::vector<std::complex<float>> naive_dft(const std::vector<float>& x) {
    size_t N = x.size();
    std::vector<std::complex<float>> X(N);
    const float TWOPI = 2.0f * 3.14159265358979323846f;
    for (size_t k = 0; k < N; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (size_t n = 0; n < N; ++n) {
            float angle = TWOPI * k * n / N;
            sum += std::complex<float>(x[n] * std::cos(angle), -x[n] * std::sin(angle));
        }
        X[k] = sum;
    }
    return X;
}

FrequencyBands FFTAnalyzer::analyze(const std::vector<float>& samples, int sampleRate) {
    FrequencyBands bands{0.0f, 0.0f, 0.0f};

    if (samples.empty() || sampleRate <= 0) return bands;

    // Run naive DFT on the window (caller should provide windowed samples)
    auto X = naive_dft(samples);
    size_t N = X.size();

    // Frequency per bin
    auto freq_of_bin = [&](size_t bin) {
        return (static_cast<float>(bin) * sampleRate) / static_cast<float>(N);
    };

    // Sum magnitudes into bands (20-200, 200-2000, 2000-20000 Hz)
    float bass = 0.0f, mids = 0.0f, highs = 0.0f;
    for (size_t k = 0; k < N / 2; ++k) { // only positive frequencies
        float freq = freq_of_bin(k);
        float mag = std::abs(X[k]);
        if (freq >= 20.0f && freq < 200.0f) bass += mag;
        else if (freq >= 200.0f && freq < 2000.0f) mids += mag;
        else if (freq >= 2000.0f && freq <= 20000.0f) highs += mag;
    }

    // Normalize by number of bins to keep values scale-independent
    float denom = static_cast<float>(N / 2);
    if (denom > 0.0f) {
        bands.bass = bass / denom;
        bands.mids = mids / denom;
        bands.highs = highs / denom;
    }

    return bands;
}

} // namespace BeatSync
