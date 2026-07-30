#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "audio/DynamicSync.h"

#include <cmath>
#include <vector>

using BeatSync::DynamicSync;
using BeatSync::DynamicSyncConfig;

namespace {

// Build a synthetic track: quiet first third, mid-level middle, loud final third.
// 30 seconds at 8kHz keeps the test fast.
std::vector<float> makeThreeSectionAudio(int sampleRate, double secondsPerSection) {
    std::vector<float> samples;
    const double amps[3] = {0.05f, 0.4f, 0.95f};
    for (int section = 0; section < 3; ++section) {
        const size_t n = static_cast<size_t>(secondsPerSection * sampleRate);
        for (size_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(i) / sampleRate;
            samples.push_back(static_cast<float>(amps[section] * std::sin(2.0 * 3.14159265 * 220.0 * t)));
        }
    }
    return samples;
}

std::vector<double> makeBeats(double interval, double totalSeconds) {
    std::vector<double> beats;
    for (double t = 0.0; t < totalSeconds; t += interval) {
        beats.push_back(t);
    }
    return beats;
}

} // namespace

TEST_CASE("Energy envelope is normalized and tracks amplitude", "[dynamicsync]") {
    const int sr = 8000;
    auto samples = makeThreeSectionAudio(sr, 10.0);
    auto env = DynamicSync::computeEnergyEnvelope(samples, sr, 0.05, 2.0);

    REQUIRE_FALSE(env.empty());
    for (double v : env) {
        REQUIRE(v >= 0.0);
        REQUIRE(v <= 1.0);
    }
    // Middle of the quiet section should read low; middle of the loud section high
    const size_t quietIdx = static_cast<size_t>(5.0 / 0.05);
    const size_t loudIdx = static_cast<size_t>(25.0 / 0.05);
    REQUIRE(env[quietIdx] < 0.3);
    REQUIRE(env[loudIdx] > 0.7);
}

TEST_CASE("Beats classify into calm/normal/frantic bands", "[dynamicsync]") {
    const int sr = 8000;
    auto samples = makeThreeSectionAudio(sr, 10.0);
    auto env = DynamicSync::computeEnergyEnvelope(samples, sr, 0.05, 2.0);
    auto beats = makeBeats(0.5, 30.0);

    DynamicSyncConfig cfg;
    auto bands = DynamicSync::classifyBeats(beats, env, 0.05, cfg);
    REQUIRE(bands.size() == beats.size());

    // Sample well inside each section (away from smoothing boundaries)
    auto bandAt = [&](double t) {
        const size_t i = static_cast<size_t>(t / 0.5);
        return bands[i];
    };
    REQUIRE(bandAt(5.0) == 0);   // quiet -> calm
    REQUIRE(bandAt(15.0) == 1);  // mid -> normal
    REQUIRE(bandAt(25.0) == 2);  // loud -> frantic
}

TEST_CASE("Filtered beats follow per-band divisors", "[dynamicsync]") {
    const int sr = 8000;
    auto samples = makeThreeSectionAudio(sr, 10.0);
    auto env = DynamicSync::computeEnergyEnvelope(samples, sr, 0.05, 2.0);
    auto beats = makeBeats(0.5, 30.0);

    DynamicSyncConfig cfg;
    auto bands = DynamicSync::classifyBeats(beats, env, 0.05, cfg);
    auto filtered = DynamicSync::filterBeats(beats, env, 0.05, cfg);

    REQUIRE_FALSE(filtered.empty());
    REQUIRE(filtered.size() < beats.size());

    // Frantic section (divisor 1) keeps every beat: count beats >= 26s in both lists
    size_t inputFrantic = 0, outputFrantic = 0;
    for (size_t i = 0; i < beats.size(); ++i) {
        if (beats[i] >= 26.0 && bands[i] == 2) inputFrantic++;
    }
    for (double t : filtered) {
        if (t >= 26.0) outputFrantic++;
    }
    REQUIRE(outputFrantic >= inputFrantic);

    // Output is still sorted and a subset of the input
    for (size_t i = 1; i < filtered.size(); ++i) {
        REQUIRE(filtered[i] > filtered[i - 1]);
    }
}

TEST_CASE("Empty envelope keeps every beat", "[dynamicsync]") {
    auto beats = makeBeats(0.5, 10.0);
    DynamicSyncConfig cfg;
    auto filtered = DynamicSync::filterBeats(beats, {}, 0.05, cfg);
    REQUIRE(filtered.size() == beats.size());
}

TEST_CASE("Flat energy reads as normal band", "[dynamicsync]") {
    std::vector<float> flat(8000 * 10, 0.5f);
    auto env = DynamicSync::computeEnergyEnvelope(flat, 8000, 0.05, 2.0);
    REQUIRE_FALSE(env.empty());
    for (double v : env) {
        REQUIRE(v == Catch::Approx(0.5));
    }
}
