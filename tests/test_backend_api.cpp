/**
 * Backend API Unit Tests
 * Tests the C API functions for correctness
 */

#include <catch2/catch_test_macros.hpp>
#include "../src/backend/beatsync_capi.h"

TEST_CASE("AudioAnalyzer creation and destruction", "[backend][audio]") {
    void* analyzer = bs_create_audio_analyzer();
    REQUIRE(analyzer != nullptr);

    bs_destroy_audio_analyzer(analyzer);
    bs_destroy_audio_analyzer(nullptr);  // Should not crash
}

TEST_CASE("VideoWriter creation and destruction", "[backend][video]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    bs_destroy_video_writer(writer);
    bs_destroy_video_writer(nullptr);  // Should not crash
}

TEST_CASE("FFmpeg path resolution", "[backend][ffmpeg]") {
    const char* ffmpegPath = bs_resolve_ffmpeg_path();
    REQUIRE(ffmpegPath != nullptr);
    // Path may be empty if FFmpeg not installed, that's OK
}

TEST_CASE("Beatgrid null safety", "[backend][audio]") {
    bs_beatgrid_t grid = {nullptr, 0, 0.0, 0.0};
    bs_free_beatgrid(&grid);
    bs_free_beatgrid(nullptr);  // Should not crash
}

TEST_CASE("Error handling with null arguments", "[backend][error]") {
    SECTION("bs_analyze_audio rejects null args") {
        int result = bs_analyze_audio(nullptr, nullptr, nullptr);
        REQUIRE(result != 0);
    }

    SECTION("bs_video_cut_at_beats rejects null args") {
        int result = bs_video_cut_at_beats(nullptr, nullptr, nullptr, 0, nullptr, 0.0);
        REQUIRE(result != 0);
    }

    SECTION("bs_video_concatenate rejects null args") {
        int result = bs_video_concatenate(nullptr, 0, nullptr);
        REQUIRE(result != 0);
    }
}
