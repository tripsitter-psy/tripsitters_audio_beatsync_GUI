/**
 * BeatSync C API Implementation
 * Wraps C++ classes for use by external consumers (Unreal Engine plugin, etc.)
 */

#include "beatsync_capi.h"
#include "../audio/AudioAnalyzer.h"
#include "../audio/BeatGrid.h"
#include "../video/VideoWriter.h"
#include "../video/VideoProcessor.h"
#include "../tracing/Tracing.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>
#include <unordered_map>

// FFmpeg includes for frame conversion
extern "C" {
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

// Thread-local storage for error messages and resolved paths
static thread_local std::string s_lastError;
static thread_local std::string s_ffmpegPath;

// Store effects config per writer (since VideoWriter doesn't expose getter)
static std::unordered_map<void*, BeatSync::EffectsConfig> s_effectsConfigs;
static std::mutex s_effectsConfigsMutex;

// ==================== Version and Init API ====================

extern "C" {

BEATSYNC_API const char* bs_get_version() {
    return "1.0.0";
}

BEATSYNC_API int bs_init() {
    // No global initialization needed currently
    return 0;
}

BEATSYNC_API void bs_shutdown() {
    std::lock_guard<std::mutex> lock(s_effectsConfigsMutex);
    s_effectsConfigs.clear();
}

// ==================== AudioAnalyzer API ====================

BEATSYNC_API void* bs_create_audio_analyzer() {
    try {
        return new BeatSync::AudioAnalyzer();
    } catch (...) {
        return nullptr;
    }
}

BEATSYNC_API void bs_destroy_audio_analyzer(void* analyzer) {
    if (analyzer) {
        delete static_cast<BeatSync::AudioAnalyzer*>(analyzer);
    }
}

BEATSYNC_API int bs_analyze_audio(void* analyzer, const char* filepath, bs_beatgrid_t* outGrid) {
    TRACE_FUNC();
    if (!analyzer || !filepath || !outGrid) {
        return -1;
    }

    try {
        auto* a = static_cast<BeatSync::AudioAnalyzer*>(analyzer);

        // Use the analyze() method which loads and detects in one call
        BeatSync::BeatGrid grid = a->analyze(filepath);

        // Check if analysis failed (duration will be 0 if file couldn't be loaded)
        if (grid.getAudioDuration() <= 0.0) {
            s_lastError = a->getLastError();
            if (s_lastError.empty()) {
                s_lastError = "Failed to analyze audio file";
            }
            return -1;
        }

        const auto& beats = grid.getBeats();

        // Allocate output buffer
        outGrid->count = beats.size();
        outGrid->bpm = grid.getBPM();
        outGrid->duration = grid.getAudioDuration();

        if (beats.empty()) {
            outGrid->beats = nullptr;
        } else {
            outGrid->beats = static_cast<double*>(malloc(beats.size() * sizeof(double)));
            if (!outGrid->beats) {
                s_lastError = "Memory allocation failure";
                return -1;
            }
            memcpy(outGrid->beats, beats.data(), beats.size() * sizeof(double));
        }

        return 0;
    } catch (const std::exception& e) {
        s_lastError = e.what();
        return -1;
    } catch (...) {
        s_lastError = "Unknown exception during analysis";
        return -1;
    }
}

BEATSYNC_API void bs_free_beatgrid(bs_beatgrid_t* grid) {
    if (grid && grid->beats) {
        free(grid->beats);
        grid->beats = nullptr;
        grid->count = 0;
    }
}

BEATSYNC_API int bs_get_waveform(void* analyzer, const char* filepath,
                                  float** outPeaks, size_t* outCount, double* outDuration) {
    TRACE_FUNC();
    if (!analyzer || !filepath || !outPeaks || !outCount || !outDuration) {
        return -1;
    }

    try {
        auto* a = static_cast<BeatSync::AudioAnalyzer*>(analyzer);

        // Load audio file to get samples using the public loadAudioFile method
        auto audioData = a->loadAudioFile(filepath);

        if (audioData.samples.empty()) {
            s_lastError = "Failed to load audio file for waveform";
            return -1;
        }

        *outDuration = audioData.duration;

        // Downsample to approximately 2000 peak values for visualization
        const size_t targetPeaks = 2000;
        size_t samplesPerPeak = audioData.samples.size() / targetPeaks;
        if (samplesPerPeak < 1) samplesPerPeak = 1;

        size_t actualPeaks = (audioData.samples.size() + samplesPerPeak - 1) / samplesPerPeak;

        float* peaks = static_cast<float*>(malloc(actualPeaks * sizeof(float)));
        if (!peaks) {
            s_lastError = "Memory allocation failure";
            return -1;
        }

        for (size_t i = 0; i < actualPeaks; ++i) {
            size_t start = i * samplesPerPeak;
            size_t end = std::min(start + samplesPerPeak, audioData.samples.size());

            float maxVal = 0.0f;
            for (size_t j = start; j < end; ++j) {
                float absVal = std::abs(audioData.samples[j]);
                if (absVal > maxVal) maxVal = absVal;
            }
            peaks[i] = maxVal;
        }

        *outPeaks = peaks;
        *outCount = actualPeaks;
        return 0;
    } catch (const std::exception& e) {
        s_lastError = e.what();
        return -1;
    } catch (...) {
        s_lastError = "Exception during waveform generation";
        return -1;
    }
}

BEATSYNC_API void bs_free_waveform(float* peaks) {
    if (peaks) {
        free(peaks);
    }
}

// ==================== VideoWriter API ====================

BEATSYNC_API void* bs_create_video_writer() {
    TRACE_FUNC();
    try {
        auto* writer = new BeatSync::VideoWriter();
        {
            std::lock_guard<std::mutex> lock(s_effectsConfigsMutex);
            s_effectsConfigs[writer] = BeatSync::EffectsConfig();
        }
        return writer;
    } catch (...) {
        return nullptr;
    }
}

BEATSYNC_API void bs_destroy_video_writer(void* writer) {
    TRACE_FUNC();
    if (writer) {
        {
            std::lock_guard<std::mutex> lock(s_effectsConfigsMutex);
            s_effectsConfigs.erase(writer);
        }
        delete static_cast<BeatSync::VideoWriter*>(writer);
    }
}

BEATSYNC_API const char* bs_video_get_last_error(void* writer) {
    if (!writer) {
        return "Invalid writer handle";
    }
    auto* w = static_cast<BeatSync::VideoWriter*>(writer);
    s_lastError = w->getLastError();
    return s_lastError.c_str();
}

BEATSYNC_API const char* bs_resolve_ffmpeg_path() {
    try {
        BeatSync::VideoWriter w;
        s_ffmpegPath = w.resolveFfmpegPath();
        return s_ffmpegPath.c_str();
    } catch (...) {
        return "";
    }
}

BEATSYNC_API void bs_video_set_progress_callback(void* writer, bs_progress_cb cb, void* user_data) {
    if (!writer) return;

    auto* w = static_cast<BeatSync::VideoWriter*>(writer);

    if (cb) {
        w->setProgressCallback([cb, user_data](double progress) {
            cb(progress, user_data);
        });
    } else {
        w->setProgressCallback(nullptr);
    }
}

BEATSYNC_API int bs_video_cut_at_beats(void* writer, const char* inputVideo,
                                        const double* beatTimes, size_t count,
                                        const char* outputVideo, double clipDuration) {
    TRACE_FUNC();
    if (!writer || !inputVideo || !beatTimes || !outputVideo || count == 0) {
        return -1;
    }

    try {
        auto* w = static_cast<BeatSync::VideoWriter*>(writer);

        // Build BeatGrid from beat times
        BeatSync::BeatGrid grid;
        for (size_t i = 0; i < count; ++i) {
            grid.addBeat(beatTimes[i]);
        }

        if (clipDuration <= 0 && count > 1) {
            // Use interval between beats if not specified
            clipDuration = beatTimes[1] - beatTimes[0];
        }

        bool success = w->cutAtBeats(inputVideo, grid, outputVideo, clipDuration);
        return success ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

BEATSYNC_API int bs_video_cut_at_beats_multi(void* writer, const char** inputVideos, size_t videoCount,
                                              const double* beatTimes, size_t beatCount,
                                              const char* outputVideo, double clipDuration) {
    TRACE_FUNC();
    if (!writer || !inputVideos || !beatTimes || !outputVideo || videoCount == 0 || beatCount == 0) {
        return -1;
    }

    try {
        auto* w = static_cast<BeatSync::VideoWriter*>(writer);

        // Collect input videos
        std::vector<std::string> videos;
        for (size_t i = 0; i < videoCount; ++i) {
            if (inputVideos[i]) {
                videos.push_back(inputVideos[i]);
            }
        }

        if (videos.empty()) {
            s_lastError = "No input videos provided";
            return -1;
        }

        // Build segments from beats, cycling through videos
        std::vector<std::string> tempFiles;
        size_t videoIdx = 0;

        for (size_t i = 0; i < beatCount; ++i) {
            double startTime = beatTimes[i];
            double endTime = (i + 1 < beatCount) ? beatTimes[i + 1] : (startTime + clipDuration);

            if (clipDuration > 0) {
                endTime = startTime + clipDuration;
            }

            std::string tempFile = std::string(outputVideo) + "_seg" + std::to_string(i) + ".mp4";

            double duration = endTime - startTime;
            // Use segment start time as position within the source video (modular approach)
            double sourceStart = 0.0;
            {
                BeatSync::VideoProcessor proc;
                if (proc.open(videos[videoIdx])) {
                    double srcDur = proc.getInfo().duration;
                    proc.close();
                    if (srcDur > 0) {
                        sourceStart = fmod(startTime, srcDur);
                    } else {
                        sourceStart = fmod(startTime, 30.0);  // Fallback
                    }
                } else {
                    sourceStart = fmod(startTime, 30.0);  // Fallback
                }
            }

            if (w->copySegmentFast(videos[videoIdx], sourceStart, duration, tempFile)) {
                tempFiles.push_back(tempFile);
            }

            videoIdx = (videoIdx + 1) % videos.size();
        }

        if (tempFiles.empty()) {
            s_lastError = "Failed to extract any segments";
            return -1;
        }

        // Concatenate all segments
        bool success = w->concatenateVideos(tempFiles, outputVideo);

        // Clean up temp files
        for (const auto& f : tempFiles) {
            remove(f.c_str());
        }

        return success ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

BEATSYNC_API int bs_video_concatenate(const char** inputs, size_t count, const char* outputVideo) {
    TRACE_FUNC();
    if (!inputs || !outputVideo || count == 0) {
        return -1;
    }

    try {
        BeatSync::VideoWriter w;

        std::vector<std::string> inputVideos;
        for (size_t i = 0; i < count; ++i) {
            if (inputs[i]) {
                inputVideos.push_back(inputs[i]);
            }
        }

        if (inputVideos.empty()) {
            return -1;
        }

        bool success = w.concatenateVideos(inputVideos, outputVideo);
        return success ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

BEATSYNC_API int bs_video_add_audio_track(void* writer, const char* inputVideo, const char* audioFile,
                                           const char* outputVideo, int trimToShortest,
                                           double audioStart, double audioEnd) {
    TRACE_FUNC();
    if (!writer || !inputVideo || !audioFile || !outputVideo) {
        return -1;
    }

    try {
        auto* w = static_cast<BeatSync::VideoWriter*>(writer);

        bool success = w->addAudioTrack(
            inputVideo,
            audioFile,
            outputVideo,
            trimToShortest != 0,
            audioStart,
            audioEnd
        );

        return success ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

// ==================== Effects API ====================

BEATSYNC_API void bs_video_set_effects_config(void* writer, const bs_effects_config_t* config) {
    if (!writer || !config) return;

    try {
        auto* w = static_cast<BeatSync::VideoWriter*>(writer);

        BeatSync::EffectsConfig cfg;
        cfg.enableTransitions = config->enableTransitions != 0;
        cfg.transitionType = config->transitionType ? config->transitionType : "fade";
        cfg.transitionDuration = config->transitionDuration;

        cfg.enableColorGrade = config->enableColorGrade != 0;
        cfg.colorPreset = config->colorPreset ? config->colorPreset : "none";

        cfg.enableVignette = config->enableVignette != 0;
        cfg.vignetteStrength = config->vignetteStrength;

        cfg.enableBeatFlash = config->enableBeatFlash != 0;
        cfg.flashIntensity = config->flashIntensity;

        cfg.enableBeatZoom = config->enableBeatZoom != 0;
        cfg.zoomIntensity = config->zoomIntensity;

        cfg.effectBeatDivisor = config->effectBeatDivisor;

        // Store in our map for later retrieval
        {
            std::lock_guard<std::mutex> lock(s_effectsConfigsMutex);
            s_effectsConfigs[writer] = cfg;
        }

        w->setEffectsConfig(cfg);
    } catch (...) {
        // Silently ignore errors
    }
}

BEATSYNC_API int bs_video_apply_effects(void* writer, const char* inputVideo,
                                         const char* outputVideo,
                                         const double* beatTimes, size_t beatCount) {
    TRACE_FUNC();
    if (!writer || !inputVideo || !outputVideo) {
        return -1;
    }

    try {
        auto* w = static_cast<BeatSync::VideoWriter*>(writer);

        // Get stored config and update beat times
        BeatSync::EffectsConfig cfg;
        {
            std::lock_guard<std::mutex> lock(s_effectsConfigsMutex);
            auto it = s_effectsConfigs.find(writer);
            if (it != s_effectsConfigs.end()) {
                cfg = it->second;
            }
        }

        cfg.beatTimesInOutput.clear();
        cfg.originalBeatIndices.clear();

        if (beatTimes && beatCount > 0) {
            for (size_t i = 0; i < beatCount; ++i) {
                cfg.beatTimesInOutput.push_back(beatTimes[i]);
                cfg.originalBeatIndices.push_back(i);
            }
        }
        w->setEffectsConfig(cfg);
        {
            std::lock_guard<std::mutex> lock(s_effectsConfigsMutex);
            s_effectsConfigs[writer] = cfg;
        }

        bool success = w->applyEffects(inputVideo, outputVideo);
        return success ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

// ==================== Frame Extraction API ====================

BEATSYNC_API int bs_video_extract_frame(const char* videoPath, double timestamp,
                                         unsigned char** outData, int* outWidth, int* outHeight) {
    TRACE_FUNC();
    if (!videoPath || !outData || !outWidth || !outHeight) {
        return -1;
    }

    try {
        BeatSync::VideoProcessor vp;
        if (!vp.open(videoPath)) {
            s_lastError = "Failed to open video file";
            return -1;
        }

        if (!vp.seekToTimestamp(timestamp)) {
            s_lastError = "Failed to seek to timestamp";
            return -1;
        }

        AVFrame* frame = nullptr;
        if (!vp.readFrame(&frame) || !frame) {
            s_lastError = "Failed to read frame";
            return -1;
        }

        int srcW = frame->width;
        int srcH = frame->height;
        *outWidth = srcW;
        *outHeight = srcH;

        // Create scaler to convert to RGB24
        SwsContext* swsCtx = sws_getContext(
            srcW, srcH, static_cast<AVPixelFormat>(frame->format),
            srcW, srcH, AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );

        if (!swsCtx) {
            s_lastError = "Failed to create scaler context";
            return -1;
        }

        // Allocate output buffer
        int rgbSize = srcW * srcH * 3;
        unsigned char* rgbData = static_cast<unsigned char*>(malloc(rgbSize));
        if (!rgbData) {
            sws_freeContext(swsCtx);
            s_lastError = "Failed to allocate RGB buffer";
            return -1;
        }

        // Setup destination pointers
        uint8_t* dstData[4] = { rgbData, nullptr, nullptr, nullptr };
        int dstLinesize[4] = { srcW * 3, 0, 0, 0 };

        // Convert frame to RGB24
        sws_scale(swsCtx, frame->data, frame->linesize, 0, srcH, dstData, dstLinesize);

        sws_freeContext(swsCtx);

        *outData = rgbData;
        return 0;
    } catch (const std::exception& e) {
        s_lastError = e.what();
        return -1;
    } catch (...) {
        s_lastError = "Exception during frame extraction";
        return -1;
    }
}

BEATSYNC_API void bs_free_frame_data(unsigned char* data) {
    if (data) {
        free(data);
    }
}

// ==================== Tracing API (stubs when not enabled) ====================

BEATSYNC_API int bs_initialize_tracing(const char* service_name) {
    // Tracing is optional - return success even if not implemented
    (void)service_name;
    return 0;
}

BEATSYNC_API void bs_shutdown_tracing() {
    // No-op when tracing is not enabled
}

BEATSYNC_API bs_span_t bs_start_span(const char* name) {
    (void)name;
    return nullptr;
}

BEATSYNC_API void bs_end_span(bs_span_t span) {
    (void)span;
}

BEATSYNC_API void bs_span_set_error(bs_span_t span, const char* msg) {
    (void)span;
    (void)msg;
}

BEATSYNC_API void bs_span_add_event(bs_span_t span, const char* event) {
    (void)span;
    (void)event;
}

} // extern "C"
