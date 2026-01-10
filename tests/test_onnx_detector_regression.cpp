#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "audio/OnnxBeatDetector.h"
#include <filesystem>
#include <fstream>
#include <iostream>

using Catch::Approx;

TEST_CASE("OnnxBeatDetector repeated inference regression", "[onnx][regression]") {
    BeatSync::OnnxBeatDetector d;

    namespace fs = std::filesystem;
    fs::path test_src = __FILE__;
    fs::path repo_root = test_src.parent_path().parent_path();
    fs::path model_path = repo_root / "tests" / "models" / "beat_stub.onnx";

    REQUIRE(fs::exists(model_path));
    d.setOnnxModelPath(model_path.string());

    // Run inference repeatedly to catch heap corruption / allocator issues
    const int iterations = 200;
    for (int i = 0; i < iterations; ++i) {
        INFO("iteration=" << i);
        auto grid = d.analyze("tests/fixtures/nonexistent_audio_for_model");
        REQUIRE(!grid.isEmpty());
        REQUIRE(grid.getNumBeats() == 3);
        REQUIRE(grid.getBeatAt(0) == Approx(0.5).margin(1e-6));
    }
}
