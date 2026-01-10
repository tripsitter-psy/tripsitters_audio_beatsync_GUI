#include <catch2/catch_test_macros.hpp>
#include "audio/BeatNetBridge.h"
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

TEST_CASE("BeatNetBridge: Python integration (gated)", "[beatnet][python][integration]") {
    // This test is gated by the BEATSYNC_ENABLE_PYTHON=1 env var to avoid running in default CI
    const char* envVal = std::getenv("BEATSYNC_ENABLE_PYTHON");
    if (!envVal || std::string(envVal) != "1") {
        WARN("BEATSYNC_ENABLE_PYTHON not set; skipping Python integration test");
        return;
    }

    // Locate the test fixture script
    fs::path scriptPath = fs::path("tests") / "fixtures" / "beatnet_fake.py";
    REQUIRE(fs::exists(scriptPath));

    BeatSync::BeatNetBridge bridge;
    BeatSync::BeatNetConfig cfg;
    cfg.mode = BeatSync::BeatNetMode::PythonOptIn;
    cfg.pythonEnabled = true;
    cfg.scriptPath = scriptPath.string();

    bridge.setConfig(cfg);

    auto grid = bridge.analyze("test_audio");

    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() >= 1);
}
