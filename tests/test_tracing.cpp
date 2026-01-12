#include <catch2/catch_test_macros.hpp>
#include "tracing/Tracing.h"
#include <fstream>
#include <filesystem>

using namespace BeatSync;

TEST_CASE("Tracing writes start and end records to file", "[tracing]") {
    auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    auto tmp = std::filesystem::temp_directory_path() / ("beatsync_test_trace_" + std::to_string(now_ns) + ".log");
    tracing::InitTracing(tmp.string());
    {
        TRACE_SCOPE("test-span");
    }
    tracing::ShutdownTracing();

    REQUIRE(std::filesystem::exists(tmp));
    {
        std::ifstream in(tmp.string());
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        REQUIRE(content.find("START test-span") != std::string::npos);
        REQUIRE(content.find("END test-span") != std::string::npos);
    }
    // Cleanup
    try { std::filesystem::remove(tmp); } catch(...) { /* best-effort cleanup */ }
}
