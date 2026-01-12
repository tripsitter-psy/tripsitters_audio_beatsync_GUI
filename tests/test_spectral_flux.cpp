#include <catch2/catch_test_macros.hpp>
#include "audio/SpectralFlux.h"
#include <vector>
#include <catch2/catch_approx.hpp>
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
    REQUIRE(out.size() >= beats.size());
    // For each expected beat, find the closest detected beat and check it's within tolerance
    for (double expected : beats) {
        double minDistance = std::numeric_limits<double>::max();
        for (double detected : out) {
            double distance = std::abs(detected - expected);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        REQUIRE(minDistance <= 0.05);
    }
}

TEST_CASE("SpectralFlux ignores silence", "[spectral][edge]") {
    int sr = 22050;
    std::vector<float> silence(22050, 0.0f);
    auto out = BeatSync::detectBeatsFromWaveform(silence, sr);
    REQUIRE(out.empty());
}
