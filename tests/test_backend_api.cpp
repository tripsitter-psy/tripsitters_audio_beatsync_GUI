#include <catch2/catch_test_macros.hpp>

// Minimal backend API test - verifies that the C API header compiles and links
#include "backend/beatsync_capi.h"

TEST_CASE("Backend API version", "[backend]") {
    // Verify version retrieval works
    const char* version = bs_get_version();
    REQUIRE(version != nullptr);
    INFO("Backend version: " << version);
}

TEST_CASE("Backend initialization", "[backend]") {
    // Initialize and shutdown should work without crashes
    int result = bs_init();
    REQUIRE(result == 0);

    bs_shutdown();
}
