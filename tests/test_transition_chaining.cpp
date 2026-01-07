#include <catch2/catch_all.hpp>
#include "../src/video/TransitionLibrary.h"
#include <filesystem>

using namespace BeatSync;

TEST_CASE("buildChainedGlTransitionFilter constructs chained transitions", "[transition][chain]") {
    // Locate repo root relative to this test source file so test works regardless of CTest working dir
    std::filesystem::path repoRoot = std::filesystem::path(__FILE__).parent_path().parent_path();
    std::filesystem::path transDir = repoRoot / "assets" / "transitions";

    TransitionLibrary lib;

    // Directory must exist for test to be meaningful
    REQUIRE(std::filesystem::exists(transDir));

    REQUIRE(lib.loadFromDirectory(transDir.string()));
    REQUIRE(!lib.getTransitions().empty());

    // Ensure the named transition is available
    const TransitionShader* fadeShader = lib.findByName("fade");
    REQUIRE(fadeShader != nullptr);

    std::string fc = lib.buildChainedGlTransitionFilter(3, "fade", 0.3);
    REQUIRE(!fc.empty());

    // Expect chained transitions between 0-1 and (t1)-2
    REQUIRE(fc.find("[0:v][1:v]") != std::string::npos);
    REQUIRE(fc.find("[t1][2:v]") != std::string::npos);
    REQUIRE(fc.find("[t2]") != std::string::npos);
}
