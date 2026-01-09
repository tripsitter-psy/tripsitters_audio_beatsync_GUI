#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "audio/BeatNetBridge.h"
#include <fstream>
#include <cstdio>

using namespace BeatSync;
using Catch::Matchers::WithinAbs;

// Helper to create a temporary sidecar file
class TempSidecarFile {
public:
    TempSidecarFile(const std::string& audioPath, const std::string& content)
        : m_path(audioPath + ".beatnet.json")
    {
        std::ofstream out(m_path);
        out << content;
        out.close();
    }

    ~TempSidecarFile() {
        std::remove(m_path.c_str());
    }

    const std::string& path() const { return m_path; }

private:
    std::string m_path;
};

TEST_CASE("BeatNetBridge default construction", "[beatnet]") {
    BeatNetBridge bridge;
    auto config = bridge.getConfig();

    REQUIRE(config.mode == BeatNetMode::SidecarOnly);
    REQUIRE(config.pythonPath.empty());
    REQUIRE(config.timeoutMs == 60000);
}

TEST_CASE("BeatNetBridge custom configuration", "[beatnet]") {
    BeatNetConfig config;
    config.mode = BeatNetMode::PythonOptIn;
    config.pythonPath = "/usr/bin/python3";
    config.scriptPath = "/path/to/script.py";
    config.verbose = true;

    BeatNetBridge bridge(config);
    auto retrievedConfig = bridge.getConfig();

    REQUIRE(retrievedConfig.mode == BeatNetMode::PythonOptIn);
    REQUIRE(retrievedConfig.pythonPath == "/usr/bin/python3");
    REQUIRE(retrievedConfig.scriptPath == "/path/to/script.py");
    REQUIRE(retrievedConfig.verbose == true);
}

TEST_CASE("BeatNetBridge parses simple sidecar JSON array", "[beatnet]") {
    // Create a simple sidecar file with beat timestamps
    TempSidecarFile sidecar("test_audio", "[0.5, 1.0, 1.5, 2.0]");

    BeatNetBridge bridge;
    auto grid = bridge.analyze("test_audio");

    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 4);
    REQUIRE_THAT(grid.getBeatAt(0), WithinAbs(0.5, 1e-6));
    REQUIRE_THAT(grid.getBeatAt(1), WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(grid.getBeatAt(2), WithinAbs(1.5, 1e-6));
    REQUIRE_THAT(grid.getBeatAt(3), WithinAbs(2.0, 1e-6));
}

TEST_CASE("BeatNetBridge parses full JSON format", "[beatnet]") {
    // Create a sidecar file with full JSON format
    TempSidecarFile sidecar("test_audio_full",
        R"({"beats": [0.5, 1.0, 1.5, 2.0], "bpm": 120.0, "downbeats": [0.5, 2.0]})");

    BeatNetBridge bridge;
    auto grid = bridge.analyze("test_audio_full");

    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 4);
    REQUIRE_THAT(grid.getBPM(), WithinAbs(120.0, 1e-6));
}

TEST_CASE("BeatNetBridge returns error when no sidecar", "[beatnet]") {
    BeatNetBridge bridge;
    auto grid = bridge.analyze("nonexistent_file");

    REQUIRE(grid.isEmpty());
    REQUIRE(bridge.hasError());
    REQUIRE(bridge.getLastError().find("sidecar not found") != std::string::npos);
}

TEST_CASE("BeatNetBridge handles empty sidecar", "[beatnet]") {
    TempSidecarFile sidecar("test_empty", "[]");

    BeatNetBridge bridge;
    auto grid = bridge.analyze("test_empty");

    REQUIRE(grid.isEmpty());
    REQUIRE(bridge.hasError());
}

TEST_CASE("BeatNetBridge handles malformed JSON", "[beatnet]") {
    TempSidecarFile sidecar("test_malformed", "{ this is not valid json }");

    BeatNetBridge bridge;
    auto grid = bridge.analyze("test_malformed");

    // Should still try to extract numbers
    REQUIRE(grid.isEmpty());
}

TEST_CASE("BeatNetBridge SidecarOnly mode", "[beatnet]") {
    BeatNetConfig config;
    config.mode = BeatNetMode::SidecarOnly;

    BeatNetBridge bridge(config);

    // Without a sidecar, should fail
    auto grid = bridge.analyze("no_sidecar_file");
    REQUIRE(grid.isEmpty());

    // With a sidecar, should succeed
    TempSidecarFile sidecar("sidecar_test", "[1.0, 2.0, 3.0]");
    grid = bridge.analyze("sidecar_test");
    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 3);
}

TEST_CASE("BeatNetBridge configuration via setters", "[beatnet]") {
    BeatNetBridge bridge;

    bridge.setPythonPath("/custom/python");
    bridge.setScriptPath("/custom/script.py");

    auto config = bridge.getConfig();
    REQUIRE(config.pythonPath == "/custom/python");
    REQUIRE(config.scriptPath == "/custom/script.py");
}

TEST_CASE("BeatNetBridge getDefaultScriptPath", "[beatnet]") {
    std::string defaultPath = BeatNetBridge::getDefaultScriptPath();

    // Should contain "beatnet_analyze.py"
    REQUIRE(defaultPath.find("beatnet_analyze.py") != std::string::npos);

#ifdef _WIN32
    // On Windows, should use backslashes
    REQUIRE(defaultPath.find("scripts\\") != std::string::npos);
#else
    // On Unix, should use forward slashes
    REQUIRE(defaultPath.find("scripts/") != std::string::npos);
#endif
}

TEST_CASE("BeatNetBridge parses scientific notation", "[beatnet]") {
    // Create a sidecar file with scientific notation
    TempSidecarFile sidecar("test_scientific", "[1.5e-1, 2.0e0, 3.5e1]");

    BeatNetBridge bridge;
    auto grid = bridge.analyze("test_scientific");

    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 3);
    REQUIRE_THAT(grid.getBeatAt(0), WithinAbs(0.15, 1e-6));
    REQUIRE_THAT(grid.getBeatAt(1), WithinAbs(2.0, 1e-6));
    REQUIRE_THAT(grid.getBeatAt(2), WithinAbs(35.0, 1e-6));
}

TEST_CASE("BeatNetBridge computes BPM from intervals", "[beatnet]") {
    // Create a sidecar with regular beat intervals (no explicit BPM)
    // Beats every 0.5 seconds = 120 BPM
    TempSidecarFile sidecar("test_bpm_compute", "[0.0, 0.5, 1.0, 1.5, 2.0]");

    BeatNetBridge bridge;
    auto grid = bridge.analyze("test_bpm_compute");

    REQUIRE(!grid.isEmpty());
    // BPM should be computed from average interval
    double expectedBPM = 60.0 / 0.5;  // 120 BPM
    REQUIRE_THAT(grid.getBPM(), WithinAbs(expectedBPM, 1.0));
}

TEST_CASE("BeatNetBridge progress callback not called for sidecar", "[beatnet]") {
    TempSidecarFile sidecar("test_progress", "[1.0, 2.0]");

    BeatNetConfig config;
    config.mode = BeatNetMode::SidecarOnly;

    BeatNetBridge bridge(config);

    // Progress callback should not be called for sidecar-only mode
    bool callbackCalled = false;
    auto callback = [&callbackCalled](float progress, const std::string& status) {
        callbackCalled = true;
    };

    auto grid = bridge.analyze("test_progress", callback);
    REQUIRE(!grid.isEmpty());
    // Sidecar mode doesn't invoke callback
    REQUIRE(!callbackCalled);
}

TEST_CASE("BeatNetBridge error cleared between calls", "[beatnet]") {
    BeatNetBridge bridge;

    // First call fails
    auto grid1 = bridge.analyze("nonexistent1");
    REQUIRE(bridge.hasError());

    // Create valid sidecar
    TempSidecarFile sidecar("valid_file", "[1.0]");

    // Second call succeeds
    auto grid2 = bridge.analyze("valid_file");
    REQUIRE(!bridge.hasError());
    REQUIRE(!grid2.isEmpty());
}

TEST_CASE("BeatNetBridge handles whitespace in JSON", "[beatnet]") {
    TempSidecarFile sidecar("test_whitespace",
        R"({
            "beats": [
                0.5,
                1.0,
                1.5
            ],
            "bpm": 120
        })");

    BeatNetBridge bridge;
    auto grid = bridge.analyze("test_whitespace");

    REQUIRE(!grid.isEmpty());
    REQUIRE(grid.getNumBeats() == 3);
}

TEST_CASE("BeatNetBridge PythonRequired mode without Python", "[beatnet]") {
    BeatNetConfig config;
    config.mode = BeatNetMode::PythonRequired;
    config.pythonPath = "/nonexistent/python";  // Invalid path

    BeatNetBridge bridge(config);

    auto grid = bridge.analyze("some_audio_file");
    REQUIRE(grid.isEmpty());
    REQUIRE(bridge.hasError());
    // Error should mention Python not available
    REQUIRE(bridge.getLastError().find("Python") != std::string::npos);
}
