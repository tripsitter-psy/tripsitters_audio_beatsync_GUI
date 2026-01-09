#include <catch2/catch_test_macros.hpp>
#include "audio/BeatNetBridge.h"
#include <fstream>

TEST_CASE("BeatNetBridge parses sidecar JSON", "[beatnet]") {
    using namespace BeatSync;
    // Prepare a simple sidecar file with beat timestamps
    const char* sidecar = "test_audio.beatnet.json";
    std::ofstream out(sidecar);
    out << "[0.5, 1.0, 1.5]";
    out.close();

    BeatNetBridge b;
    auto grid = b.analyze("test_audio");

    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 3);
    REQUIRE(grid.getBeatAt(0) == Approx(0.5).margin(1e-6));

    // cleanup
    std::remove(sidecar);
}

TEST_CASE("BeatNetBridge returns error when no sidecar", "[beatnet]") {
    using namespace BeatSync;
    BeatNetBridge b;
    auto grid = b.analyze("no_such_file");
    REQUIRE(grid.isEmpty());
    REQUIRE(b.getLastError().size() > 0);
}
