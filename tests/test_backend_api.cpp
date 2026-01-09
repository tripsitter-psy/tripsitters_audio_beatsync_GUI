/**
 * Backend API Unit Tests
 * Tests the C API functions for correctness
 */

#include "../src/backend/beatsync_capi.h"
#include <iostream>
#include <cstring>
#include <cmath>

#define TEST(name) std::cout << "Testing " << name << "... "
#define PASS() std::cout << "PASS\n"
#define FAIL(msg) do { std::cout << "FAIL: " << msg << "\n"; failures++; } while(0)

int main() {
    std::cout << "=== Backend API Unit Tests ===\n\n";
    int failures = 0;

    // Test AudioAnalyzer creation/destruction
    TEST("bs_create_audio_analyzer");
    void* analyzer = bs_create_audio_analyzer();
    if (analyzer) {
        PASS();
    } else {
        FAIL("returned null");
    }

    TEST("bs_destroy_audio_analyzer");
    bs_destroy_audio_analyzer(analyzer);
    bs_destroy_audio_analyzer(nullptr);  // Should not crash
    PASS();

    // Test VideoWriter creation/destruction
    TEST("bs_create_video_writer");
    void* writer = bs_create_video_writer();
    if (writer) {
        PASS();
    } else {
        FAIL("returned null");
    }

    TEST("bs_destroy_video_writer");
    bs_destroy_video_writer(writer);
    bs_destroy_video_writer(nullptr);  // Should not crash
    PASS();

    // Test FFmpeg path resolution
    TEST("bs_resolve_ffmpeg_path");
    const char* ffmpegPath = bs_resolve_ffmpeg_path();
    if (ffmpegPath) {
        std::cout << "PASS (path: " << (ffmpegPath[0] ? ffmpegPath : "<empty>") << ")\n";
    } else {
        FAIL("returned null");
    }

    // Test beatgrid handling
    TEST("bs_free_beatgrid (null safety)");
    bs_beatgrid_t grid = {nullptr, 0, 0.0, 0.0};
    bs_free_beatgrid(&grid);
    bs_free_beatgrid(nullptr);  // Should not crash
    PASS();

    // Test error handling with invalid args
    TEST("bs_analyze_audio (null args)");
    int result = bs_analyze_audio(nullptr, nullptr, nullptr);
    if (result != 0) {
        PASS();
    } else {
        FAIL("should return error for null args");
    }

    TEST("bs_video_cut_at_beats (null args)");
    result = bs_video_cut_at_beats(nullptr, nullptr, nullptr, 0, nullptr, 0.0);
    if (result != 0) {
        PASS();
    } else {
        FAIL("should return error for null args");
    }

    TEST("bs_video_concatenate (null args)");
    result = bs_video_concatenate(nullptr, 0, nullptr);
    if (result != 0) {
        PASS();
    } else {
        FAIL("should return error for null args");
    }

    // Summary
    std::cout << "\n=== Test Summary ===\n";
    if (failures == 0) {
        std::cout << "All tests passed!\n";
        return 0;
    } else {
        std::cout << failures << " test(s) failed.\n";
        return 1;
    }
}
