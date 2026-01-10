#include <catch2/catch_test_macros.hpp>
#include "audio/OnnxBeatDetector.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

TEST_CASE("OnnxBeatDetector reads ONNX sidecar", "[onnx][prototype]") {
    BeatSync::OnnxBeatDetector d;
    auto grid = d.analyze("tests/fixtures/test_audio");    INFO("grid: " << grid.toString());    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 3);
    REQUIRE(grid.getBeatAt(0) == Approx(0.5).margin(1e-6));
}

TEST_CASE("OnnxBeatDetector runs stub ONNX model", "[onnx][integration]") {
    BeatSync::OnnxBeatDetector d;

    // Resolve model path relative to the test source location
    namespace fs = std::filesystem;
    fs::path test_src = __FILE__;
    fs::path repo_root = test_src.parent_path().parent_path();
    fs::path model_path = repo_root / "tests" / "models" / "beat_stub.onnx";

    REQUIRE(fs::exists(model_path));

    d.setOnnxModelPath(model_path.string());

    // Use a path that does not have a sidecar so the ONNX model is used.
    auto grid = d.analyze("tests/fixtures/nonexistent_audio_for_model");

    // Strict ONNX assertions: model must load and produce expected beats
    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 3);
    REQUIRE(grid.getBeatAt(0) == Approx(0.5).margin(1e-6));
}

TEST_CASE("OnnxBeatDetector falls back to spectral flux on audio file", "[spectral][integration]") {
    BeatSync::OnnxBeatDetector d;

    // Generate a short click-track WAV file in tests/fixtures
    int sr = 22050;
    std::vector<double> beats = {0.5, 1.0, 1.5};
    double duration = 2.0;
    int n = int(duration * sr);
    std::vector<float> s(n, 0.0f);
    for (double bt : beats) {
        int idx = int(bt * sr);
        if (idx >= 0 && idx < n) s[idx] = 1.0f;
    }
    for (int i = 1; i < n; ++i) s[i] += 0.5f * s[i-1];

    // Write WAV (16-bit PCM)
    auto writeWav = [&](const std::string &path, const std::vector<float>& samples, int sr) {
        std::ofstream out(path, std::ios::binary);
        // WAV header
        int16_t audioFormat = 1; // PCM
        int16_t numChannels = 1;
        int32_t sampleRate = sr;
        int16_t bitsPerSample = 16;
        int32_t byteRate = sampleRate * numChannels * bitsPerSample/8;
        int32_t blockAlign = numChannels * bitsPerSample/8;
        int32_t dataSize = int(samples.size()) * numChannels * bitsPerSample/8;

        out.write("RIFF",4);
        int32_t chunkSize = 36 + dataSize;
        out.write(reinterpret_cast<const char*>(&chunkSize), 4);
        out.write("WAVE",4);
        out.write("fmt ",4);
        int32_t subChunk1Size = 16;
        out.write(reinterpret_cast<const char*>(&subChunk1Size),4);
        out.write(reinterpret_cast<const char*>(&audioFormat),2);
        out.write(reinterpret_cast<const char*>(&numChannels),2);
        out.write(reinterpret_cast<const char*>(&sampleRate),4);
        out.write(reinterpret_cast<const char*>(&byteRate),4);
        out.write(reinterpret_cast<const char*>(&blockAlign),2);
        out.write(reinterpret_cast<const char*>(&bitsPerSample),2);
        out.write("data",4);
        out.write(reinterpret_cast<const char*>(&dataSize),4);

        for (float v : samples) {
            float clipped = std::max(-1.0f, std::min(1.0f, v));
            int16_t iv = int16_t(clipped * 32767.0f);
            out.write(reinterpret_cast<const char*>(&iv), sizeof(iv));
        }
        out.close();
    };

    std::string wavPath = "tests/fixtures/test_clicks.wav";
    writeWav(wavPath, s, sr);

    auto grid = d.analyze(wavPath);
    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() >= beats.size());
    REQUIRE(grid.getBeatAt(0) == Approx(0.5).margin(0.05));
}

