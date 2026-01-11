/**
 * BeatSync C API Implementation
 * Provides C-compatible FFI for Unreal Engine plugin consumption
 */

#define BEATSYNC_CAPI_EXPORT
#include "beatsync_capi.h"
#include "../audio/AudioAnalyzer.h"
#include "../video/VideoWriter.h"

#include <cstdlib>
#include <cstring>
#include <string>

// Thread-local storage for last error and resolved paths
static thread_local std::string g_lastError;
static thread_local std::string g_ffmpegPath;

// AudioAnalyzer wrapper
BEATSYNC_API void* bs_create_audio_analyzer() {
    try {
        return new BeatSync::AudioAnalyzer();
    } catch (const std::exception& e) {
        g_lastError = e.what();
        return nullptr;
    }
}

BEATSYNC_API void bs_destroy_audio_analyzer(void* analyzer) {
    if (analyzer) {
        delete static_cast<BeatSync::AudioAnalyzer*>(analyzer);
    }
}

BEATSYNC_API int bs_analyze_audio(void* analyzer, const char* filepath, bs_beatgrid_t* outGrid) {
    if (!analyzer || !filepath || !outGrid) {
        g_lastError = "Invalid arguments";
        return -1;
    }

    try {
        auto* aa = static_cast<BeatSync::AudioAnalyzer*>(analyzer);
        BeatSync::BeatGrid grid = aa->analyze(filepath);

        const auto& beats = grid.getBeats();
        outGrid->count = beats.size();
        outGrid->bpm = grid.getBPM();
        outGrid->duration = grid.getAudioDuration();

        if (outGrid->count > 0) {
            outGrid->beats = static_cast<double*>(malloc(sizeof(double) * outGrid->count));
            if (!outGrid->beats) {
                g_lastError = "Memory allocation failed";
                return -2;
            }
            memcpy(outGrid->beats, beats.data(), sizeof(double) * outGrid->count);
        } else {
            outGrid->beats = nullptr;
        }

        return 0;
    } catch (const std::exception& e) {
        g_lastError = e.what();
        return -3;
    }
}

BEATSYNC_API void bs_free_beatgrid(bs_beatgrid_t* grid) {
    if (grid && grid->beats) {
        free(grid->beats);
        grid->beats = nullptr;
        grid->count = 0;
    }
}

// VideoWriter wrapper
BEATSYNC_API void* bs_create_video_writer() {
    try {
        return new BeatSync::VideoWriter();
    } catch (const std::exception& e) {
        g_lastError = e.what();
        return nullptr;
    }
}

BEATSYNC_API void bs_destroy_video_writer(void* writer) {
    if (writer) {
        delete static_cast<BeatSync::VideoWriter*>(writer);
    }
}

BEATSYNC_API const char* bs_video_get_last_error(void* writer) {
    if (writer) {
        auto* vw = static_cast<BeatSync::VideoWriter*>(writer);
        g_lastError = vw->getLastError();
    }
    return g_lastError.c_str();
}

BEATSYNC_API const char* bs_resolve_ffmpeg_path() {
    try {
        BeatSync::VideoWriter vw;
        g_ffmpegPath = vw.resolveFfmpegPath();
        return g_ffmpegPath.c_str();
    } catch (const std::exception& e) {
        g_lastError = e.what();
        return "";
    }
}

// Progress callback bridge
struct ProgressBridge {
    bs_progress_cb callback;
    void* user_data;
};

static thread_local ProgressBridge g_progressBridge = {nullptr, nullptr};

static void progressCallbackBridge(double progress) {
    if (g_progressBridge.callback) {
        g_progressBridge.callback(progress, g_progressBridge.user_data);
    }
}

BEATSYNC_API void bs_video_set_progress_callback(void* writer, bs_progress_cb cb, void* user_data) {
    if (!writer) return;

    g_progressBridge.callback = cb;
    g_progressBridge.user_data = user_data;

    auto* vw = static_cast<BeatSync::VideoWriter*>(writer);
    if (cb) {
        vw->setProgressCallback(progressCallbackBridge);
    } else {
        vw->setProgressCallback(nullptr);
    }
}

BEATSYNC_API int bs_video_cut_at_beats(void* writer, const char* inputVideo,
                                        const double* beatTimes, size_t count,
                                        const char* outputVideo, double clipDuration) {
    if (!writer || !inputVideo || !beatTimes || !outputVideo) {
        g_lastError = "Invalid arguments";
        return -1;
    }

    try {
        auto* vw = static_cast<BeatSync::VideoWriter*>(writer);

        // Build a BeatGrid from the provided times
        BeatSync::BeatGrid grid;
        for (size_t i = 0; i < count; ++i) {
            grid.addBeat(beatTimes[i]);
        }

        bool success = vw->cutAtBeats(inputVideo, grid, outputVideo, clipDuration);
        return success ? 0 : -2;
    } catch (const std::exception& e) {
        g_lastError = e.what();
        return -3;
    }
}

BEATSYNC_API int bs_video_concatenate(const char** inputs, size_t count, const char* outputVideo) {
    if (!inputs || count == 0 || !outputVideo) {
        g_lastError = "Invalid arguments";
        return -1;
    }

    try {
        BeatSync::VideoWriter vw;
        std::vector<std::string> inputPaths;
        inputPaths.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (inputs[i]) {
                inputPaths.emplace_back(inputs[i]);
            }
        }

        bool success = vw.concatenateVideos(inputPaths, outputVideo);
        return success ? 0 : -2;
    } catch (const std::exception& e) {
        g_lastError = e.what();
        return -3;
    }
}
