#include <catch2/catch_all.hpp>
#include <string>
#include <filesystem>
#include "../src/video/TransitionLibrary.h"

using namespace BeatSync;

TEST_CASE("TransitionLibrary loads shaders and builds filters", "[transition]") {
    std::filesystem::path repoRoot = std::filesystem::current_path();
    // Attempt to find assets/transitions relative to repo root
    std::filesystem::path transDir = repoRoot / "assets" / "transitions";

    TransitionLibrary lib;
    bool ok = lib.loadFromDirectory(transDir.string());

    REQUIRE(ok);
    auto const & transitions = lib.getTransitions();
    REQUIRE(!transitions.empty());

    // Expect at least the provided fade.glsl to be discovered
    const TransitionShader* fade = lib.findByName("fade");
    REQUIRE(fade != nullptr);
    CHECK(fade->category == "blend");

    std::string filter = lib.buildGlTransitionFilter("fade", 0.3);
    CHECK(filter.find("gltransition") != std::string::npos);
    CHECK(filter.find("duration=0.300") != std::string::npos);
}
