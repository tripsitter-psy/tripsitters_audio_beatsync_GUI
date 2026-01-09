#include <catch2/catch_test_macros.hpp>
#include "audio/FFTAnalyzer.h"
#include <cmath>

static std::vector<float> make_sine(float freq, int sampleRate, int N) {
    std::vector<float> out(N);
    const float TWOPI = 2.0f * 3.14159265358979323846f;
    for (int i = 0; i < N; ++i) {
        out[i] = std::sin(TWOPI * freq * (static_cast<float>(i) / sampleRate));
    }
    return out;
}

TEST_CASE("FFTAnalyzer buckets energy to expected band", "[fft]") {
    BeatSync::FFTAnalyzer a;
    int sr = 48000;
    int N = 256; // small window

    auto sine_bass = make_sine(100.0f, sr, N);
    auto bands_bass = a.analyze(sine_bass, sr);

    auto sine_mids = make_sine(1000.0f, sr, N);
    auto bands_mids = a.analyze(sine_mids, sr);

    auto sine_high = make_sine(5000.0f, sr, N);
    auto bands_high = a.analyze(sine_high, sr);

    // Ensure dominant band is as expected
    REQUIRE(bands_bass.bass > bands_bass.mids);
    REQUIRE(bands_bass.bass > bands_bass.highs);

    REQUIRE(bands_mids.mids > bands_mids.bass);
    REQUIRE(bands_mids.mids > bands_mids.highs);

    REQUIRE(bands_high.highs > bands_high.bass);
    REQUIRE(bands_high.highs > bands_high.mids);
}
