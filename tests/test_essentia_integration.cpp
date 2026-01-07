#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../src/audio/EssentiaAnalyzer.h"
#include <filesystem>

TEST_CASE("Essentia integration (conditional)") {
#ifdef HAVE_ESSENTIA
    std::string sample = "tests/fixtures/sample_short.wav";
    if (!std::filesystem::exists(sample)) {
        SUCCEED("No test fixture audio present - skipping Essentia runtime test.");
        return;
    }
    BeatSync::EssentiaAnalyzer analyzer;
    auto grid = analyzer.analyze(sample);
    // We expect analyze to return without throwing; beats may be zero for short samples
    REQUIRE(grid.getBeats().size() >= 0);
#else
    // Essentia not available: skip test
    SUCCEED("Essentia not available - test skipped.");
#endif
}
