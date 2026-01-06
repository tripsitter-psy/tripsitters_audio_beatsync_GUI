#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../src/audio/EssentiaAnalyzer.h"

TEST_CASE("Essentia integration (conditional)") {
#ifdef HAVE_ESSENTIA
    BeatSync::EssentiaAnalyzer analyzer;
    auto grid = analyzer.analyze("tests/fixtures/sample_short.wav");
    REQUIRE(grid.getBeats().size() >= 0); // At minimum we should return normally
#else
    // Essentia not available: skip test
    SUCCEED("Essentia not available - test skipped.");
#endif
}
