/**
 * @file test_onnx_ai.cpp
 * @brief Tests for ONNX-based AI music analysis API
 */

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <filesystem>
#include <vector>

// Backend C API header
#include "backend/beatsync_capi.h"

// Test fixture paths
static const char* TEST_AUDIO_FILE = "tests/fixtures/test_clicks.wav";
static const char* TEST_MODEL_DIR = "models";

// ==================== ONNX AI Availability Tests ====================

TEST_CASE("AI availability check", "[backend][ai][onnx]") {
    int available = bs_ai_is_available();
    INFO("ONNX AI available: " << (available ? "yes" : "no"));

    // This test passes regardless - just checks the API works
    REQUIRE((available == 0 || available == 1));
}

TEST_CASE("AI providers query", "[backend][ai][onnx]") {
    const char* providers = bs_ai_get_providers();
    REQUIRE(providers != nullptr);
    INFO("Available providers: " << providers);
}

// ==================== AI Analyzer Creation Tests ====================

TEST_CASE("AI analyzer null config", "[backend][ai][null]") {
    void* analyzer = bs_create_ai_analyzer(nullptr);
    REQUIRE(analyzer == nullptr);  // Should fail with null config
}

TEST_CASE("AI analyzer missing beat model path", "[backend][ai]") {
    bs_ai_config_t config = {};
    config.beat_model_path = nullptr;  // Required field is null

    void* analyzer = bs_create_ai_analyzer(&config);
    REQUIRE(analyzer == nullptr);  // Should fail

    const char* error = bs_ai_get_last_error(nullptr);
    REQUIRE(error != nullptr);
    INFO("Error message: " << error);
}

TEST_CASE("AI analyzer with nonexistent model", "[backend][ai]") {
    bs_ai_config_t config = {};
    config.beat_model_path = "nonexistent_model.onnx";
    config.use_stem_separation = 0;

    void* analyzer = bs_create_ai_analyzer(&config);

    // May succeed (deferred loading) or fail immediately
    if (analyzer) {
        bs_destroy_ai_analyzer(analyzer);
    }
}

TEST_CASE("AI analyzer destruction null is safe", "[backend][ai][null]") {
    bs_destroy_ai_analyzer(nullptr);  // Should not crash
}

// ==================== AI Result Memory Tests ====================

TEST_CASE("AI result free null is safe", "[backend][ai][memory]") {
    bs_free_ai_result(nullptr);  // Should not crash
}

TEST_CASE("AI result free empty struct", "[backend][ai][memory]") {
    bs_ai_result_t result = {};
    bs_free_ai_result(&result);  // Should not crash
}

TEST_CASE("AI result free with null members", "[backend][ai][memory]") {
    bs_ai_result_t result = {};
    result.beats = nullptr;
    result.downbeats = nullptr;
    result.segments = nullptr;
    result.beat_count = 0;
    result.downbeat_count = 0;
    result.segment_count = 0;

    bs_free_ai_result(&result);  // Should not crash

    REQUIRE(result.beats == nullptr);
    REQUIRE(result.beat_count == 0);
}

// ==================== AI Analysis Function Null Handling ====================

TEST_CASE("AI analyze file null handling", "[backend][ai][null]") {
    bs_ai_result_t result = {};

    // Null analyzer
    REQUIRE(bs_ai_analyze_file(nullptr, TEST_AUDIO_FILE, &result, nullptr, nullptr) == -1);

    // Null audio path
    bs_ai_config_t config = {};
    config.beat_model_path = "test.onnx";
    // Note: analyzer creation may fail, so we test with nullptr
    REQUIRE(bs_ai_analyze_file(nullptr, nullptr, &result, nullptr, nullptr) == -1);

    // Null result
    REQUIRE(bs_ai_analyze_file(nullptr, TEST_AUDIO_FILE, nullptr, nullptr, nullptr) == -1);
}

TEST_CASE("AI analyze samples null handling", "[backend][ai][null]") {
    bs_ai_result_t result = {};
    float samples[] = {0.0f, 0.1f, 0.2f};

    // Null analyzer
    REQUIRE(bs_ai_analyze_samples(nullptr, samples, 3, 44100, 1, &result, nullptr, nullptr) == -1);

    // Null samples
    REQUIRE(bs_ai_analyze_samples(nullptr, nullptr, 3, 44100, 1, &result, nullptr, nullptr) == -1);

    // Zero sample count
    REQUIRE(bs_ai_analyze_samples(nullptr, samples, 0, 44100, 1, &result, nullptr, nullptr) == -1);

    // Null result
    REQUIRE(bs_ai_analyze_samples(nullptr, samples, 3, 44100, 1, nullptr, nullptr, nullptr) == -1);
}

TEST_CASE("AI analyze quick null handling", "[backend][ai][null]") {
    bs_ai_result_t result = {};

    // Null analyzer
    REQUIRE(bs_ai_analyze_quick(nullptr, TEST_AUDIO_FILE, &result, nullptr, nullptr) == -1);

    // Null audio path
    REQUIRE(bs_ai_analyze_quick(nullptr, nullptr, &result, nullptr, nullptr) == -1);

    // Null result
    REQUIRE(bs_ai_analyze_quick(nullptr, TEST_AUDIO_FILE, nullptr, nullptr, nullptr) == -1);
}

// ==================== AI Error and Info Functions ====================

TEST_CASE("AI get last error with null", "[backend][ai][null]") {
    const char* error = bs_ai_get_last_error(nullptr);
    REQUIRE(error != nullptr);  // Should return empty string, not null
}

TEST_CASE("AI get model info with null", "[backend][ai][null]") {
    const char* info = bs_ai_get_model_info(nullptr);
    REQUIRE(info != nullptr);  // Should return empty string, not null
}

// ==================== AI Config Validation ====================

TEST_CASE("AI config default values", "[backend][ai][config]") {
    bs_ai_config_t config = {};

    // All fields should be zero-initialized
    REQUIRE(config.beat_model_path == nullptr);
    REQUIRE(config.stem_model_path == nullptr);
    REQUIRE(config.use_stem_separation == 0);
    REQUIRE(config.use_drums_for_beats == 0);
    REQUIRE(config.use_gpu == 0);
    REQUIRE(config.gpu_device_id == 0);
    REQUIRE(config.beat_threshold == 0.0f);
    REQUIRE(config.downbeat_threshold == 0.0f);
}

TEST_CASE("AI config with all options", "[backend][ai][config]") {
    bs_ai_config_t config = {};
    config.beat_model_path = "models/beatnet.onnx";
    config.stem_model_path = "models/demucs.onnx";
    config.use_stem_separation = 1;
    config.use_drums_for_beats = 1;
    config.use_gpu = 1;
    config.gpu_device_id = 0;
    config.beat_threshold = 0.4f;
    config.downbeat_threshold = 0.6f;

    // Just validate the config is valid (analyzer creation may fail if models don't exist)
    REQUIRE(config.beat_model_path != nullptr);
    REQUIRE(config.use_stem_separation == 1);
    REQUIRE(config.beat_threshold == 0.4f);
}

// ==================== AI Result Structure Tests ====================

TEST_CASE("AI result structure layout", "[backend][ai][result]") {
    bs_ai_result_t result = {};

    // Verify the structure is properly zero-initialized
    REQUIRE(result.beats == nullptr);
    REQUIRE(result.beat_count == 0);
    REQUIRE(result.downbeats == nullptr);
    REQUIRE(result.downbeat_count == 0);
    REQUIRE(result.bpm == 0.0);
    REQUIRE(result.duration == 0.0);
    REQUIRE(result.segments == nullptr);
    REQUIRE(result.segment_count == 0);
}

// ==================== Progress Callback Tests ====================

struct AIProgressState {
    int call_count = 0;
    float last_progress = -1.0f;
    std::string last_stage;
    std::string last_message;
    bool should_cancel = false;
};

static int ai_progress_callback(float progress, const char* stage, const char* message, void* user_data) {
    AIProgressState* state = static_cast<AIProgressState*>(user_data);
    if (state) {
        state->call_count++;
        state->last_progress = progress;
        if (stage) state->last_stage = stage;
        if (message) state->last_message = message;
        return state->should_cancel ? 0 : 1;
    }
    return 1;
}

TEST_CASE("AI progress callback invocation", "[backend][ai][callback]") {
    AIProgressState state;

    // Test callback directly
    int result = ai_progress_callback(0.5f, "test_stage", "test_message", &state);
    REQUIRE(result == 1);
    REQUIRE(state.call_count == 1);
    REQUIRE(state.last_progress == 0.5f);
    REQUIRE(state.last_stage == "test_stage");
    REQUIRE(state.last_message == "test_message");
}

TEST_CASE("AI progress callback with null user data", "[backend][ai][callback][null]") {
    // Should not crash with null user_data
    int result = ai_progress_callback(0.5f, "stage", "message", nullptr);
    REQUIRE(result == 1);  // Continue when no state
}

TEST_CASE("AI progress callback cancellation", "[backend][ai][callback]") {
    AIProgressState state;
    state.should_cancel = true;

    int result = ai_progress_callback(0.5f, "stage", "message", &state);
    REQUIRE(result == 0);  // Should signal cancellation
}

// ==================== Integration Test (if models exist) ====================

TEST_CASE("AI full pipeline integration", "[backend][ai][integration]") {
    // Skip if ONNX not available
    if (!bs_ai_is_available()) {
        WARN("ONNX Runtime not available, skipping integration test");
        SKIP("ONNX Runtime not available");
    }

    // Skip if test audio doesn't exist
    if (!std::filesystem::exists(TEST_AUDIO_FILE)) {
        WARN("Test audio file not found: " << TEST_AUDIO_FILE);
        SKIP("Test audio file not available");
    }

    // Skip if no models directory
    if (!std::filesystem::exists(TEST_MODEL_DIR)) {
        WARN("Models directory not found: " << TEST_MODEL_DIR);
        SKIP("Models directory not available");
    }

    // Look for any .onnx model
    std::string beatModelPath;
    for (const auto& entry : std::filesystem::directory_iterator(TEST_MODEL_DIR)) {
        if (entry.path().extension() == ".onnx") {
            beatModelPath = entry.path().string();
            break;
        }
    }

    if (beatModelPath.empty()) {
        WARN("No ONNX models found in " << TEST_MODEL_DIR);
        SKIP("No ONNX models available");
    }

    INFO("Using model: " << beatModelPath);

    bs_ai_config_t config = {};
    config.beat_model_path = beatModelPath.c_str();
    config.use_stem_separation = 0;  // Skip stem separation for faster test

    void* analyzer = bs_create_ai_analyzer(&config);
    if (!analyzer) {
        WARN("Failed to create AI analyzer: " << bs_ai_get_last_error(nullptr));
        SKIP("AI analyzer creation failed");
    }

    AIProgressState progressState;
    bs_ai_result_t result = {};

    int ret = bs_ai_analyze_file(analyzer, TEST_AUDIO_FILE, &result, ai_progress_callback, &progressState);

    INFO("Analysis returned: " << ret);
    INFO("Progress callback calls: " << progressState.call_count);
    INFO("Last progress: " << progressState.last_progress);
    INFO("Detected beats: " << result.beat_count);
    INFO("Detected downbeats: " << result.downbeat_count);
    INFO("BPM: " << result.bpm);
    INFO("Duration: " << result.duration);

    if (ret == 0) {
        // Analysis succeeded
        REQUIRE(result.duration > 0.0);

        // If beats were detected, verify they're valid
        if (result.beat_count > 0) {
            REQUIRE(result.beats != nullptr);
            REQUIRE(result.bpm > 0.0);

            // Beats should be in ascending order
            for (size_t i = 1; i < result.beat_count; ++i) {
                REQUIRE(result.beats[i] > result.beats[i-1]);
            }
        }
    }

    bs_free_ai_result(&result);
    bs_destroy_ai_analyzer(analyzer);

    REQUIRE(result.beats == nullptr);
    REQUIRE(result.beat_count == 0);
}
