#include <catch2/catch_test_macros.hpp>
#include "audio/OnnxBeatDetector.h"

TEST_CASE("OnnxBeatDetector reads ONNX sidecar", "[onnx][prototype]") {
    BeatSync::OnnxBeatDetector d;
    auto grid = d.analyze("tests/fixtures/test_audio");
    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 3);
    REQUIRE(grid.getBeatAt(0) == Approx(0.5).margin(1e-6));
}
