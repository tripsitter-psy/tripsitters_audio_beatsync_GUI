#include <catch2/catch_all.hpp>
#include "../src/video/VideoWriter.h"

using namespace BeatSync;

// Test accessor for private methods
class VideoWriterTestAccess {
public:
    static std::string buildGlTransitionFilterComplex(VideoWriter& writer, size_t numInputs,
                                                       const std::string& transitionName, double duration) {
        return writer.buildGlTransitionFilterComplex(numInputs, transitionName, duration);
    }
};

TEST_CASE("buildGlTransitionFilterComplex constructs chained transitions", "[transition][chain]") {
    VideoWriter writer;
    std::string fc = VideoWriterTestAccess::buildGlTransitionFilterComplex(writer, 3, "fade", 0.3);

    // The function returns empty if transition assets aren't available
    // This is expected in CI/test environments without the assets directory
    if (fc.empty()) {
        // No transition assets available in CI; treat as an intended pass instead of skipping
        SKIP("Transition assets not available - skipping content validation");
        return;
    }

    // When assets are available, expect chained transitions between 0-1 and (t1)-2
    REQUIRE(fc.find("[0:v][1:v]") != std::string::npos);
    REQUIRE(fc.find("[t1][2:v]") != std::string::npos);
    REQUIRE(fc.find("[t2]") != std::string::npos);
}
