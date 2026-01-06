#include <catch2/catch_all.hpp>
#include "../src/video/VideoWriter.h"

using namespace BeatSync;

TEST_CASE("buildGlTransitionFilterComplex constructs chained transitions", "[transition][chain]") {
    VideoWriter writer;
    std::string fc = writer.buildGlTransitionFilterComplex(3, "fade", 0.3);

    // Expect chained transitions between 0-1 and (t1)-2
    REQUIRE(fc.find("[0:v][1:v]") != std::string::npos);
    REQUIRE(fc.find("[t1][2:v]") != std::string::npos);
    REQUIRE(fc.find("[t2]") != std::string::npos);
}
