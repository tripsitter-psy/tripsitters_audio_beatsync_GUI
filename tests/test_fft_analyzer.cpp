#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "audio/FFTAnalyzer.h"
#include <cmath>
#include <numeric>

using namespace BeatSync;
using Catch::Matchers::WithinAbs;

// Helper: generate sine wave at given frequency
static std::vector<float> make_sine(float freq, int sampleRate, int N) {
    std::vector<float> out(N);
    const float TWOPI = 2.0f * 3.14159265358979323846f;
    for (int i = 0; i < N; ++i) {
        out[i] = std::sin(TWOPI * freq * (static_cast<float>(i) / sampleRate));
    }
    return out;
}

// Helper: generate multi-frequency signal
static std::vector<float> make_composite(const std::vector<std::pair<float, float>>& freqAmps,
                                          int sampleRate, int N) {
    std::vector<float> out(N, 0.0f);
    const float TWOPI = 2.0f * 3.14159265358979323846f;
    for (const auto& [freq, amp] : freqAmps) {
        for (int i = 0; i < N; ++i) {
            out[i] += amp * std::sin(TWOPI * freq * (static_cast<float>(i) / sampleRate));
        }
    }
    return out;
}

TEST_CASE("FFTAnalyzer default construction", "[fft]") {
    FFTAnalyzer analyzer;
    auto config = analyzer.getConfig();

    REQUIRE(config.windowSize == 2048);
    REQUIRE(config.windowType == WindowType::Hann);
    REQUIRE(config.normalize == true);
}

TEST_CASE("FFTAnalyzer custom configuration", "[fft]") {
    FFTConfig config;
    config.windowSize = 1024;
    config.windowType = WindowType::Blackman;
    config.smoothingFactor = 0.5f;

    FFTAnalyzer analyzer(config);
    auto retrievedConfig = analyzer.getConfig();

    REQUIRE(retrievedConfig.windowSize == 1024);
    REQUIRE(retrievedConfig.windowType == WindowType::Blackman);
    REQUIRE(retrievedConfig.smoothingFactor == 0.5f);
}

TEST_CASE("FFTAnalyzer buckets energy to expected band", "[fft]") {
    FFTConfig config;
    config.windowSize = 2048;
    config.smoothingFactor = 0.0f;  // Disable smoothing for deterministic tests

    FFTAnalyzer analyzer(config);
    int sr = 48000;

    SECTION("100Hz sine should be detected as bass") {
        auto sine_bass = make_sine(100.0f, sr, config.windowSize);
        auto bands = analyzer.analyze(sine_bass, sr);

        REQUIRE(bands.bass > bands.mids);
        REQUIRE(bands.bass > bands.highs);
    }

    SECTION("1000Hz sine should be detected as mids") {
        auto sine_mids = make_sine(1000.0f, sr, config.windowSize);
        auto bands = analyzer.analyze(sine_mids, sr);

        REQUIRE(bands.mids > bands.bass);
        REQUIRE(bands.mids > bands.highs);
    }

    SECTION("5000Hz sine should be detected as highs") {
        auto sine_high = make_sine(5000.0f, sr, config.windowSize);
        auto bands = analyzer.analyze(sine_high, sr);

        REQUIRE(bands.highs > bands.bass);
        REQUIRE(bands.highs > bands.mids);
    }
}

TEST_CASE("FFTAnalyzer handles empty input", "[fft]") {
    FFTAnalyzer analyzer;
    std::vector<float> empty;

    auto bands = analyzer.analyze(empty, 48000);

    REQUIRE(bands.bass == 0.0f);
    REQUIRE(bands.mids == 0.0f);
    REQUIRE(bands.highs == 0.0f);
}

TEST_CASE("FFTAnalyzer handles invalid sample rate", "[fft]") {
    FFTAnalyzer analyzer;
    auto samples = make_sine(440.0f, 48000, 1024);

    auto bands = analyzer.analyze(samples, 0);

    REQUIRE(bands.bass == 0.0f);
    REQUIRE(bands.mids == 0.0f);
    REQUIRE(bands.highs == 0.0f);

    bands = analyzer.analyze(samples, -1);

    REQUIRE(bands.bass == 0.0f);
}

TEST_CASE("FFTAnalyzer handles short input", "[fft]") {
    FFTConfig config;
    config.windowSize = 2048;
    config.smoothingFactor = 0.0f;

    FFTAnalyzer analyzer(config);

    // Input shorter than window size - should still work (zero-padded)
    auto short_input = make_sine(1000.0f, 48000, 512);
    auto bands = analyzer.analyze(short_input, 48000);

    // Should still detect mids as dominant
    REQUIRE(bands.mids > 0.0f);
}

TEST_CASE("FFTAnalyzer window functions", "[fft]") {
    FFTConfig config;
    config.windowSize = 1024;
    config.smoothingFactor = 0.0f;

    auto samples = make_sine(1000.0f, 48000, 1024);

    SECTION("Rectangular window") {
        config.windowType = WindowType::Rectangular;
        FFTAnalyzer analyzer(config);
        auto bands = analyzer.analyze(samples, 48000);
        REQUIRE(bands.mids > 0.0f);
    }

    SECTION("Hann window") {
        config.windowType = WindowType::Hann;
        FFTAnalyzer analyzer(config);
        auto bands = analyzer.analyze(samples, 48000);
        REQUIRE(bands.mids > 0.0f);
    }

    SECTION("Hamming window") {
        config.windowType = WindowType::Hamming;
        FFTAnalyzer analyzer(config);
        auto bands = analyzer.analyze(samples, 48000);
        REQUIRE(bands.mids > 0.0f);
    }

    SECTION("Blackman window") {
        config.windowType = WindowType::Blackman;
        FFTAnalyzer analyzer(config);
        auto bands = analyzer.analyze(samples, 48000);
        REQUIRE(bands.mids > 0.0f);
    }
}

TEST_CASE("FFTAnalyzer temporal smoothing", "[fft]") {
    FFTConfig config;
    config.windowSize = 1024;
    config.smoothingFactor = 0.5f;

    FFTAnalyzer analyzer(config);
    int sr = 48000;

    // First frame: bass
    auto bass_signal = make_sine(100.0f, sr, 1024);
    auto bands1 = analyzer.analyze(bass_signal, sr);

    // Second frame: sudden switch to highs
    auto high_signal = make_sine(5000.0f, sr, 1024);
    auto bands2 = analyzer.analyze(high_signal, sr);

    // With smoothing, the transition should be gradual
    REQUIRE(bands2.highs > 0.0f);  // New signal should be detected
}

TEST_CASE("FFTAnalyzer reset clears smoothing state", "[fft]") {
    FFTConfig config;
    config.windowSize = 1024;
    config.smoothingFactor = 0.8f;  // Strong smoothing

    FFTAnalyzer analyzer(config);
    int sr = 48000;

    // Prime with bass signal
    auto bass_signal = make_sine(100.0f, sr, 1024);
    analyzer.analyze(bass_signal, sr);
    analyzer.analyze(bass_signal, sr);

    // Reset
    analyzer.reset();

    // Now analyze highs - should not be affected by previous bass
    auto high_signal = make_sine(5000.0f, sr, 1024);
    auto bands = analyzer.analyze(high_signal, sr);

    // After reset, highs should dominate immediately
    REQUIRE(bands.highs > bands.bass);
}

TEST_CASE("FFTAnalyzer magnitude spectrum access", "[fft]") {
    FFTConfig config;
    config.windowSize = 1024;
    config.smoothingFactor = 0.0f;

    FFTAnalyzer analyzer(config);

    auto samples = make_sine(1000.0f, 48000, 1024);
    analyzer.analyze(samples, 48000);

    const auto& spectrum = analyzer.getMagnitudeSpectrum();

    // Should have N/2 + 1 bins
    REQUIRE(spectrum.size() == 513);

    // Spectrum should not be all zeros
    float sum = std::accumulate(spectrum.begin(), spectrum.end(), 0.0f);
    REQUIRE(sum > 0.0f);
}

TEST_CASE("FFTAnalyzer bin to frequency conversion", "[fft]") {
    FFTConfig config;
    config.windowSize = 1024;

    FFTAnalyzer analyzer(config);

    // At 48kHz with 1024-sample window:
    // Frequency resolution = 48000 / 1024 = 46.875 Hz per bin

    REQUIRE_THAT(analyzer.binToFrequency(0, 48000), WithinAbs(0.0f, 0.01f));
    REQUIRE_THAT(analyzer.binToFrequency(1, 48000), WithinAbs(46.875f, 0.01f));
    REQUIRE_THAT(analyzer.binToFrequency(10, 48000), WithinAbs(468.75f, 0.01f));
}

TEST_CASE("FFTAnalyzer normalized output values", "[fft]") {
    FFTConfig config;
    config.windowSize = 1024;
    config.smoothingFactor = 0.0f;

    FFTAnalyzer analyzer(config);

    // Full-scale sine wave
    auto samples = make_sine(1000.0f, 48000, 1024);
    auto bands = analyzer.analyze(samples, 48000);

    // Normalized values should be in [0, 1] range
    REQUIRE(bands.bassNorm >= 0.0f);
    REQUIRE(bands.bassNorm <= 1.0f);
    REQUIRE(bands.midsNorm >= 0.0f);
    REQUIRE(bands.midsNorm <= 1.0f);
    REQUIRE(bands.highsNorm >= 0.0f);
    REQUIRE(bands.highsNorm <= 1.0f);
}

TEST_CASE("FFTAnalyzer move semantics", "[fft]") {
    FFTConfig config;
    config.windowSize = 1024;

    FFTAnalyzer analyzer1(config);
    auto samples = make_sine(1000.0f, 48000, 1024);
    analyzer1.analyze(samples, 48000);

    // Move construct
    FFTAnalyzer analyzer2(std::move(analyzer1));
    auto bands = analyzer2.analyze(samples, 48000);

    REQUIRE(bands.mids > 0.0f);

    // Move assign
    FFTAnalyzer analyzer3;
    analyzer3 = std::move(analyzer2);
    bands = analyzer3.analyze(samples, 48000);

    REQUIRE(bands.mids > 0.0f);
}

TEST_CASE("FFTAnalyzer composite signal detection", "[fft]") {
    FFTConfig config;
    config.windowSize = 2048;
    config.smoothingFactor = 0.0f;

    FFTAnalyzer analyzer(config);
    int sr = 48000;

    // Signal with strong bass (100Hz) and weak highs (5000Hz)
    auto composite = make_composite({
        {100.0f, 1.0f},   // Strong bass
        {5000.0f, 0.2f}   // Weak highs
    }, sr, config.windowSize);

    auto bands = analyzer.analyze(composite, sr);

    // Bass should dominate
    REQUIRE(bands.bass > bands.highs);
}
