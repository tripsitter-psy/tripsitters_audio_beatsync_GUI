#include <catch2/catch_test_macros.hpp>
#include "audio/SpectralFlux.h"
#include <vector>
#include <limits>

// Helper: generate a click track (impulse at beat times)
static std::vector<float> makeClickTrack(const std::vector<double>& beatTimes, int sampleRate, double duration) {
    int n = int(duration * sampleRate);
    std::vector<float> s(n, 0.0f);
    for (double bt : beatTimes) {
        int idx = int(bt * sampleRate);
        if (idx >= 0 && idx < n) s[idx] = 1.0f; // unit impulse
    }
    // Simple lowpass filter to give clicks some width
    for (int i = 1; i < n; ++i) s[i] += 0.5f * s[i-1];
    return s;
}

TEST_CASE("SpectralFlux detects click track beats", "[spectral][detector]") {
    int sr = 22050;
    std::vector<double> beats = {0.5, 1.0, 1.5};
    double duration = 2.0;
    auto samples = makeClickTrack(beats, sr, duration);

    auto out = BeatSync::detectBeatsFromWaveform(samples, sr, 1024, 256, 1.5, 1.2);
    REQUIRE(out.size() == beats.size());
    // One-to-one matching: every detected beat must map to an expected beat within tolerance
    std::vector<bool> matched(beats.size(), false);
    for (double detected : out) {
        bool found = false;
        for (size_t i = 0; i < beats.size(); ++i) {
            if (!matched[i] && std::abs(detected - beats[i]) <= 0.05) {
                matched[i] = true;
                found = true;
                break;
            }
        }
        REQUIRE(found); // Fail if any detected beat does not match an expected beat
    }
    // Ensure all expected beats were matched
    for (bool m : matched) REQUIRE(m);
}

TEST_CASE("SpectralFlux ignores silence", "[spectral][edge]") {
    int sr = 22050;
    std::vector<float> silence(22050, 0.0f);
    auto out = BeatSync::detectBeatsFromWaveform(silence, sr);
    REQUIRE(out.empty());
}
