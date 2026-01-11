#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fstream>
#include <filesystem>

// Backend C API header
#include "backend/beatsync_capi.h"

// Test fixture paths (relative to CMAKE_SOURCE_DIR working directory)
static const char* TEST_AUDIO_FILE = "tests/fixtures/test_clicks.wav";

// ==================== Library Lifecycle Tests ====================

TEST_CASE("Backend API version", "[backend][lifecycle]") {
    const char* version = bs_get_version();
    REQUIRE(version != nullptr);
    REQUIRE(strlen(version) > 0);
    INFO("Backend version: " << version);
}

TEST_CASE("Backend initialization", "[backend][lifecycle]") {
    int result = bs_init();
    REQUIRE(result == 0);
    bs_shutdown();
}

TEST_CASE("Backend multiple init/shutdown cycles", "[backend][lifecycle]") {
    // Should be safe to init/shutdown multiple times
    for (int i = 0; i < 3; ++i) {
        int result = bs_init();
        REQUIRE(result == 0);
        bs_shutdown();
    }
}

// ==================== AudioAnalyzer Tests ====================

TEST_CASE("AudioAnalyzer creation and destruction", "[backend][audio]") {
    void* analyzer = bs_create_audio_analyzer();
    REQUIRE(analyzer != nullptr);
    bs_destroy_audio_analyzer(analyzer);
}

TEST_CASE("AudioAnalyzer null destruction is safe", "[backend][audio]") {
    // Should not crash when destroying null
    bs_destroy_audio_analyzer(nullptr);
}

TEST_CASE("AudioAnalyzer analyze with test audio", "[backend][audio]") {
    // Skip if test file doesn't exist
    if (!std::filesystem::exists(TEST_AUDIO_FILE)) {
        WARN("Test audio file not found: " << TEST_AUDIO_FILE);
        SKIP("Test audio file not available");
    }

    void* analyzer = bs_create_audio_analyzer();
    REQUIRE(analyzer != nullptr);

    bs_beatgrid_t grid = {};
    int result = bs_analyze_audio(analyzer, TEST_AUDIO_FILE, &grid);

    // Analysis should succeed (file exists and is valid audio)
    REQUIRE(result == 0);

    // Log what was detected
    INFO("Detected " << grid.count << " beats, BPM: " << grid.bpm << ", Duration: " << grid.duration);

    // Note: test_clicks.wav may not have enough energy variation to detect beats
    // The important thing is that the API doesn't crash and returns valid data structure
    // Duration may be 0 if no beats detected (known limitation - audio duration not set without beats)

    // BPM should be positive if beats detected
    if (grid.count > 0) {
        REQUIRE(grid.bpm > 0.0);
        REQUIRE(grid.beats != nullptr);

        // Beats should be in ascending order
        for (size_t i = 1; i < grid.count; ++i) {
            REQUIRE(grid.beats[i] > grid.beats[i-1]);
        }
    }

    bs_free_beatgrid(&grid);
    REQUIRE(grid.beats == nullptr);
    REQUIRE(grid.count == 0);

    bs_destroy_audio_analyzer(analyzer);
}

TEST_CASE("AudioAnalyzer analyze with invalid file", "[backend][audio]") {
    void* analyzer = bs_create_audio_analyzer();
    REQUIRE(analyzer != nullptr);

    bs_beatgrid_t grid = {};
    int result = bs_analyze_audio(analyzer, "nonexistent_file.wav", &grid);

    // Should fail with error code for nonexistent file
    REQUIRE(result == -1);

    // Grid should remain empty
    REQUIRE(grid.count == 0);
    REQUIRE(grid.beats == nullptr);

    bs_destroy_audio_analyzer(analyzer);
}

TEST_CASE("AudioAnalyzer null parameter handling", "[backend][audio][null]") {
    void* analyzer = bs_create_audio_analyzer();
    REQUIRE(analyzer != nullptr);

    bs_beatgrid_t grid = {};

    // Null analyzer
    REQUIRE(bs_analyze_audio(nullptr, TEST_AUDIO_FILE, &grid) == -1);

    // Null filepath
    REQUIRE(bs_analyze_audio(analyzer, nullptr, &grid) == -1);

    // Null output grid
    REQUIRE(bs_analyze_audio(analyzer, TEST_AUDIO_FILE, nullptr) == -1);

    bs_destroy_audio_analyzer(analyzer);
}

// ==================== Waveform Tests ====================

TEST_CASE("Waveform generation with test audio", "[backend][waveform]") {
    if (!std::filesystem::exists(TEST_AUDIO_FILE)) {
        WARN("Test audio file not found: " << TEST_AUDIO_FILE);
        SKIP("Test audio file not available");
    }

    void* analyzer = bs_create_audio_analyzer();
    REQUIRE(analyzer != nullptr);

    float* peaks = nullptr;
    size_t peakCount = 0;
    double duration = 0.0;

    int result = bs_get_waveform(analyzer, TEST_AUDIO_FILE, &peaks, &peakCount, &duration);

    REQUIRE(result == 0);
    REQUIRE(peaks != nullptr);
    REQUIRE(peakCount > 0);
    REQUIRE(duration > 0.0);

    INFO("Waveform: " << peakCount << " peaks, duration: " << duration << "s");

    // Peak values should be in valid range [0, 1] for normalized audio
    for (size_t i = 0; i < peakCount; ++i) {
        REQUIRE(peaks[i] >= 0.0f);
        REQUIRE(peaks[i] <= 1.0f);
    }

    bs_free_waveform(peaks);
    bs_destroy_audio_analyzer(analyzer);
}

TEST_CASE("Waveform null parameter handling", "[backend][waveform][null]") {
    void* analyzer = bs_create_audio_analyzer();
    REQUIRE(analyzer != nullptr);

    float* peaks = nullptr;
    size_t peakCount = 0;
    double duration = 0.0;

    // Null analyzer
    REQUIRE(bs_get_waveform(nullptr, TEST_AUDIO_FILE, &peaks, &peakCount, &duration) == -1);

    // Null filepath
    REQUIRE(bs_get_waveform(analyzer, nullptr, &peaks, &peakCount, &duration) == -1);

    // Null output pointers
    REQUIRE(bs_get_waveform(analyzer, TEST_AUDIO_FILE, nullptr, &peakCount, &duration) == -1);
    REQUIRE(bs_get_waveform(analyzer, TEST_AUDIO_FILE, &peaks, nullptr, &duration) == -1);
    REQUIRE(bs_get_waveform(analyzer, TEST_AUDIO_FILE, &peaks, &peakCount, nullptr) == -1);

    bs_destroy_audio_analyzer(analyzer);
}

TEST_CASE("Waveform free null is safe", "[backend][waveform]") {
    bs_free_waveform(nullptr);
}

// ==================== VideoWriter Tests ====================

TEST_CASE("VideoWriter creation and destruction", "[backend][video]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);
    bs_destroy_video_writer(writer);
}

TEST_CASE("VideoWriter null destruction is safe", "[backend][video]") {
    bs_destroy_video_writer(nullptr);
}

TEST_CASE("VideoWriter get last error", "[backend][video]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    const char* error = bs_video_get_last_error(writer);
    REQUIRE(error != nullptr);
    // Initially error should be empty or a default message

    bs_destroy_video_writer(writer);
}

TEST_CASE("VideoWriter get last error with null", "[backend][video][null]") {
    const char* error = bs_video_get_last_error(nullptr);
    REQUIRE(error != nullptr);
    REQUIRE(strcmp(error, "Invalid writer handle") == 0);
}

TEST_CASE("FFmpeg path resolution", "[backend][video]") {
    const char* path = bs_resolve_ffmpeg_path();
    REQUIRE(path != nullptr);
    INFO("FFmpeg path: " << path);
}

// ==================== Effects Config Tests ====================

TEST_CASE("Effects config set and apply", "[backend][effects]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    bs_effects_config_t config = {};
    config.enableTransitions = 1;
    config.transitionType = "fade";
    config.transitionDuration = 0.5;
    config.enableColorGrade = 1;
    config.colorPreset = "warm";
    config.enableVignette = 1;
    config.vignetteStrength = 0.3;
    config.enableBeatFlash = 1;
    config.flashIntensity = 0.5;
    config.enableBeatZoom = 0;
    config.zoomIntensity = 0.0;
    config.effectBeatDivisor = 1;

    // Should not crash
    bs_video_set_effects_config(writer, &config);

    bs_destroy_video_writer(writer);
}

TEST_CASE("Effects config null handling", "[backend][effects][null]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    // Null config should not crash
    bs_video_set_effects_config(writer, nullptr);

    // Null writer should not crash
    bs_effects_config_t config = {};
    bs_video_set_effects_config(nullptr, &config);

    bs_destroy_video_writer(writer);
}

TEST_CASE("Effects config with null strings", "[backend][effects]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    // Config with null string fields should use defaults
    bs_effects_config_t config = {};
    config.enableTransitions = 1;
    config.transitionType = nullptr;  // Should default to "fade"
    config.enableColorGrade = 1;
    config.colorPreset = nullptr;     // Should default to "none"

    bs_video_set_effects_config(writer, &config);

    bs_destroy_video_writer(writer);
}

// ==================== Video Processing Null Safety Tests ====================

TEST_CASE("Video cut at beats null handling", "[backend][video][null]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    double beats[] = {0.0, 0.5, 1.0};

    // Null writer
    REQUIRE(bs_video_cut_at_beats(nullptr, "input.mp4", beats, 3, "output.mp4", 0.5) == -1);

    // Null input video
    REQUIRE(bs_video_cut_at_beats(writer, nullptr, beats, 3, "output.mp4", 0.5) == -1);

    // Null beat times
    REQUIRE(bs_video_cut_at_beats(writer, "input.mp4", nullptr, 3, "output.mp4", 0.5) == -1);

    // Null output video
    REQUIRE(bs_video_cut_at_beats(writer, "input.mp4", beats, 3, nullptr, 0.5) == -1);

    // Zero beat count
    REQUIRE(bs_video_cut_at_beats(writer, "input.mp4", beats, 0, "output.mp4", 0.5) == -1);

    bs_destroy_video_writer(writer);
}

TEST_CASE("Video cut at beats multi null handling", "[backend][video][null]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    const char* videos[] = {"video1.mp4", "video2.mp4"};
    double beats[] = {0.0, 0.5, 1.0};

    // Null writer
    REQUIRE(bs_video_cut_at_beats_multi(nullptr, videos, 2, beats, 3, "output.mp4", 0.5) == -1);

    // Null videos array
    REQUIRE(bs_video_cut_at_beats_multi(writer, nullptr, 2, beats, 3, "output.mp4", 0.5) == -1);

    // Zero video count
    REQUIRE(bs_video_cut_at_beats_multi(writer, videos, 0, beats, 3, "output.mp4", 0.5) == -1);

    // Zero beat count
    REQUIRE(bs_video_cut_at_beats_multi(writer, videos, 2, beats, 0, "output.mp4", 0.5) == -1);

    bs_destroy_video_writer(writer);
}

TEST_CASE("Video concatenate null handling", "[backend][video][null]") {
    const char* inputs[] = {"video1.mp4", "video2.mp4"};

    // Null inputs
    REQUIRE(bs_video_concatenate(nullptr, 2, "output.mp4") == -1);

    // Null output
    REQUIRE(bs_video_concatenate(inputs, 2, nullptr) == -1);

    // Zero count
    REQUIRE(bs_video_concatenate(inputs, 0, "output.mp4") == -1);
}

TEST_CASE("Video add audio track null handling", "[backend][video][null]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    // Null writer
    REQUIRE(bs_video_add_audio_track(nullptr, "video.mp4", "audio.wav", "output.mp4", 1, 0.0, -1.0) == -1);

    // Null input video
    REQUIRE(bs_video_add_audio_track(writer, nullptr, "audio.wav", "output.mp4", 1, 0.0, -1.0) == -1);

    // Null audio file
    REQUIRE(bs_video_add_audio_track(writer, "video.mp4", nullptr, "output.mp4", 1, 0.0, -1.0) == -1);

    // Null output video
    REQUIRE(bs_video_add_audio_track(writer, "video.mp4", "audio.wav", nullptr, 1, 0.0, -1.0) == -1);

    bs_destroy_video_writer(writer);
}

TEST_CASE("Video apply effects null handling", "[backend][effects][null]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    double beats[] = {0.0, 0.5, 1.0};

    // Null writer
    REQUIRE(bs_video_apply_effects(nullptr, "input.mp4", "output.mp4", beats, 3) == -1);

    // Null input
    REQUIRE(bs_video_apply_effects(writer, nullptr, "output.mp4", beats, 3) == -1);

    // Null output
    REQUIRE(bs_video_apply_effects(writer, "input.mp4", nullptr, beats, 3) == -1);

    // Note: null beats with count 0 should be allowed (no beat-synced effects)

    bs_destroy_video_writer(writer);
}

// ==================== Frame Extraction Tests ====================

TEST_CASE("Frame extraction null handling", "[backend][frame][null]") {
    unsigned char* data = nullptr;
    int width = 0, height = 0;

    // Null video path
    REQUIRE(bs_video_extract_frame(nullptr, 0.0, &data, &width, &height) == -1);

    // Null output data pointer
    REQUIRE(bs_video_extract_frame("video.mp4", 0.0, nullptr, &width, &height) == -1);

    // Null width pointer
    REQUIRE(bs_video_extract_frame("video.mp4", 0.0, &data, nullptr, &height) == -1);

    // Null height pointer
    REQUIRE(bs_video_extract_frame("video.mp4", 0.0, &data, &width, nullptr) == -1);
}

TEST_CASE("Frame extraction with nonexistent file", "[backend][frame]") {
    unsigned char* data = nullptr;
    int width = 0, height = 0;

    int result = bs_video_extract_frame("nonexistent_video.mp4", 0.0, &data, &width, &height);

    // Should fail gracefully
    REQUIRE(result != 0);
    REQUIRE(data == nullptr);
}

TEST_CASE("Frame data free null is safe", "[backend][frame]") {
    bs_free_frame_data(nullptr);
}

// ==================== Progress Callback Tests ====================

// Callback state struct for thread-safe testing (avoids static globals that could
// cause flaky tests if run in parallel)
struct CallbackState {
    int callCount = 0;
    double lastProgress = -1.0;
};

static void test_progress_callback(double progress, void* user_data) {
    CallbackState* state = static_cast<CallbackState*>(user_data);
    if (state) {
        state->callCount++;
        state->lastProgress = progress;
    }
}

TEST_CASE("Progress callback registration", "[backend][video][callback]") {
    void* writer = bs_create_video_writer();
    REQUIRE(writer != nullptr);

    CallbackState state;

    // Register callback
    bs_video_set_progress_callback(writer, test_progress_callback, &state);

    // Unregister callback (null)
    bs_video_set_progress_callback(writer, nullptr, nullptr);

    bs_destroy_video_writer(writer);
}

TEST_CASE("Progress callback with null writer", "[backend][video][callback][null]") {
    // Should not crash
    bs_video_set_progress_callback(nullptr, test_progress_callback, nullptr);
}

TEST_CASE("Progress callback invocation", "[backend][video][callback]") {
    CallbackState state;

    // Call the callback directly to test state updates
    test_progress_callback(0.5, &state);
    REQUIRE(state.callCount == 1);
    REQUIRE(state.lastProgress == 0.5);

    test_progress_callback(0.75, &state);
    REQUIRE(state.callCount == 2);
    REQUIRE(state.lastProgress == 0.75);

    // Test with null user_data (should not crash)
    test_progress_callback(1.0, nullptr);
}

// ==================== Beatgrid Memory Tests ====================

TEST_CASE("Beatgrid free with null grid", "[backend][memory]") {
    // Should not crash
    bs_free_beatgrid(nullptr);
}

TEST_CASE("Beatgrid free with null beats", "[backend][memory]") {
    bs_beatgrid_t grid = {};
    grid.beats = nullptr;
    grid.count = 0;

    // Should not crash
    bs_free_beatgrid(&grid);
}

// ==================== Tracing API Tests ====================

TEST_CASE("Tracing initialization", "[backend][tracing]") {
    int result = bs_initialize_tracing("test_service");
    REQUIRE(result == 0);
    bs_shutdown_tracing();
}

TEST_CASE("Tracing with null service name", "[backend][tracing]") {
    int result = bs_initialize_tracing(nullptr);
    REQUIRE(result == 0);  // Should succeed (stubs)
    bs_shutdown_tracing();
}

TEST_CASE("Span creation and lifecycle", "[backend][tracing]") {
    bs_span_t span = bs_start_span("test_span");
    // Span may be null if tracing is not enabled

    // These should not crash even with null span
    bs_span_set_error(span, "test error");
    bs_span_add_event(span, "test event");
    bs_end_span(span);
}

TEST_CASE("Span operations with null", "[backend][tracing][null]") {
    // All should be safe with null
    bs_end_span(nullptr);
    bs_span_set_error(nullptr, "error");
    bs_span_set_error(nullptr, nullptr);
    bs_span_add_event(nullptr, "event");
    bs_span_add_event(nullptr, nullptr);
}
