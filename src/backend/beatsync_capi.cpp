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
#include <iostream>

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

// Version constant defined at build time and exported for consumers
// Note: BEATSYNC_API is applied so this symbol is visible to DLL consumers
extern "C" {
BEATSYNC_API const char* const BS_VERSION = BEATSYNC_VERSION;

BEATSYNC_API const char* bs_get_version() {
    return BS_VERSION;
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

    const double DEFAULT_DURATION = 30.0;  // Configurable fallback duration

    // RAII helper to ensure temp files are cleaned up even if an exception is thrown
    struct TempFileCleanup {
        std::vector<std::string>& files;
        ~TempFileCleanup() {
            for (const auto& f : files) {
                remove(f.c_str());
            }
        }
    };

    std::vector<std::string> tempFiles;
    TempFileCleanup cleanup{tempFiles};

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

        // Precompute and cache video durations
        std::vector<double> durations(videos.size(), 0.0);
        for (size_t i = 0; i < videos.size(); ++i) {
            BeatSync::VideoProcessor proc;
            if (proc.open(videos[i])) {
                durations[i] = proc.getInfo().duration;
                proc.close();
            }
            // If duration <= 0, we'll use DEFAULT_DURATION later
        }

        // Build segments from beats, cycling through videos
        size_t videoIdx = 0;

        for (size_t i = 0; i < beatCount; ++i) {
            double startTime = beatTimes[i];
            double endTime;
            const double MIN_SEGMENT_DURATION = 0.1; // Minimum 100ms segment
            
            if (i + 1 < beatCount) {
                endTime = beatTimes[i + 1];
            } else {
                // Last segment: use clipDuration if positive, otherwise minimum duration
                endTime = startTime + std::max(clipDuration, MIN_SEGMENT_DURATION);
            }

            double duration = endTime - startTime;
            // Ensure duration is positive
            if (duration <= 0) {
                duration = MIN_SEGMENT_DURATION;
                endTime = startTime + duration;
            }

            std::string tempFile = std::string(outputVideo) + "_seg" + std::to_string(i) + ".mp4";

            // Use segment start time as position within the source video (modular approach)
            double sourceStart = 0.0;
            double cachedDuration = durations[videoIdx];
            if (cachedDuration > 0) {
                sourceStart = fmod(startTime, cachedDuration);
            } else {
                sourceStart = fmod(startTime, DEFAULT_DURATION);  // Fallback
            }

            if (w->copySegmentFast(videos[videoIdx], sourceStart, duration, tempFile)) {
                tempFiles.push_back(tempFile);
            } else {
                std::cerr << "Failed to copy segment: videoIdx=" << videoIdx 
                         << ", sourceStart=" << sourceStart 
                         << ", duration=" << duration 
                         << ", tempFile=" << tempFile << std::endl;
                s_lastError = "Failed to copy video segment " + std::to_string(i);
                return -1; // Propagate failure instead of silently continuing
            }

            videoIdx = (videoIdx + 1) % videos.size();
        }

        if (tempFiles.empty()) {
            s_lastError = "Failed to extract any segments";
            return -1;
        }

        // Concatenate all segments
        bool success = w->concatenateVideos(tempFiles, outputVideo);

        // Cleanup handled by TempFileCleanup destructor
        return success ? 0 : -1;
    } catch (...) {
        // Cleanup handled by TempFileCleanup destructor
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
    if (!writer) return;

    try {
        auto* w = static_cast<BeatSync::VideoWriter*>(writer);

        // If config is nullptr, reset to default/disabled effects
        if (!config) {
            {
                std::lock_guard<std::mutex> lock(s_effectsConfigsMutex);
                s_effectsConfigs.erase(writer);
            }
            BeatSync::EffectsConfig defaultCfg;  // All effects disabled by default
            w->setEffectsConfig(defaultCfg);
            return;
        }

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
    } catch (const std::exception& e) {
        s_lastError = e.what();
    } catch (...) {
        s_lastError = "unknown error in bs_video_set_effects_config";
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

    // Initialize output params
    *outData = nullptr;
    *outWidth = 0;
    *outHeight = 0;

    SwsContext* swsCtx = nullptr;
    unsigned char* rgbData = nullptr;

    try {
        BeatSync::VideoProcessor vp;
        if (!vp.open(videoPath)) {
            s_lastError = "Failed to open video file: " + vp.getLastError();
            return -1;
        }

        if (!vp.seekToTimestamp(timestamp)) {
            s_lastError = "Failed to seek to timestamp: " + vp.getLastError();
            return -1;
        }

        // readFrame returns pointer to internal frame storage owned by VideoProcessor
        // DO NOT call av_frame_free on this - VideoProcessor destructor handles it
        AVFrame* frame = nullptr;
        if (!vp.readFrame(&frame) || !frame) {
            s_lastError = "Failed to read frame: " + vp.getLastError();
            return -1;
        }

        int srcW = frame->width;
        int srcH = frame->height;

        if (srcW <= 0 || srcH <= 0) {
            s_lastError = "Invalid frame dimensions";
            return -1;
        }

        *outWidth = srcW;
        *outHeight = srcH;

        // Create scaler to convert to RGB24
        swsCtx = sws_getContext(
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
        rgbData = static_cast<unsigned char*>(malloc(rgbSize));
        if (!rgbData) {
            sws_freeContext(swsCtx);
            s_lastError = "Failed to allocate RGB buffer";
            return -1;
        }

        // Setup destination pointers
        uint8_t* dstData[4] = { rgbData, nullptr, nullptr, nullptr };
        int dstLinesize[4] = { srcW * 3, 0, 0, 0 };

        // Convert frame to RGB24
        int ret = sws_scale(swsCtx, frame->data, frame->linesize, 0, srcH, dstData, dstLinesize);
        sws_freeContext(swsCtx);
        swsCtx = nullptr;

        if (ret < 0) {
            free(rgbData);
            s_lastError = "Failed to scale frame";
            return -1;
        }

        // VideoProcessor destructor will clean up the frame when vp goes out of scope
        // We've already copied the pixel data to rgbData, so we're good

        *outData = rgbData;
        return 0;
    } catch (const std::exception& e) {
        if (swsCtx) {
            sws_freeContext(swsCtx);
        }
        if (rgbData) {
            free(rgbData);
        }
        s_lastError = e.what();
        return -1;
    } catch (...) {
        if (swsCtx) {
            sws_freeContext(swsCtx);
        }
        if (rgbData) {
            free(rgbData);
        }
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

// ==================== ONNX AI Analysis API ====================
} // Close extern "C" temporarily for C++ includes

#ifdef USE_ONNX
#include "../audio/OnnxMusicAnalyzer.h"
#include "../audio/OnnxBeatDetector.h"
#include "../audio/OnnxStemSeparator.h"
#endif

extern "C" { // Resume extern "C" for C API functions

// Thread-local storage for AI-related strings
static thread_local std::string s_aiLastError;
static thread_local std::string s_aiModelInfo;
static thread_local std::string s_aiProviders;

// Note: segment labels are heap-allocated per-result in bs_ai_analyze_* functions
// and freed by bs_free_ai_result(). No thread-local storage needed.

BEATSYNC_API void* bs_create_ai_analyzer(const bs_ai_config_t* config) {
#ifndef USE_ONNX
    s_aiLastError = "ONNX Runtime not available. Rebuild with USE_ONNX=ON";
    return nullptr;
#else
    if (!config || !config->beat_model_path) {
        s_aiLastError = "Invalid configuration: beat_model_path is required";
        return nullptr;
    }

    try {
        auto* analyzer = new BeatSync::OnnxMusicAnalyzer();

        BeatSync::MusicAnalyzerConfig cfg;
        cfg.beatModelPath = config->beat_model_path;

        if (config->stem_model_path) {
            cfg.stemModelPath = config->stem_model_path;
        }

        cfg.useStemSeparation = config->use_stem_separation != 0;
        cfg.useDrumsForBeats = config->use_drums_for_beats != 0;

        // Beat detection config
        cfg.beatConfig.useGPU = config->use_gpu != 0;
        cfg.beatConfig.gpuDeviceId = config->gpu_device_id;
        if (config->beat_threshold > 0.0f) {
            cfg.beatConfig.beatThreshold = config->beat_threshold;
        }
        if (config->downbeat_threshold > 0.0f) {
            cfg.beatConfig.downbeatThreshold = config->downbeat_threshold;
        }

        // Stem separation config
        cfg.stemConfig.useGPU = config->use_gpu != 0;
        cfg.stemConfig.gpuDeviceId = config->gpu_device_id;

        if (!analyzer->initialize(cfg)) {
            s_aiLastError = analyzer->getLastError();
            delete analyzer;
            return nullptr;
        }

        return analyzer;
    } catch (const std::exception& e) {
        s_aiLastError = e.what();
        return nullptr;
    } catch (...) {
        s_aiLastError = "Unknown error creating AI analyzer";
        return nullptr;
    }
#endif
}

BEATSYNC_API void bs_destroy_ai_analyzer(void* analyzer) {
#ifdef USE_ONNX
    if (analyzer) {
        delete static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);
    }
#else
    (void)analyzer;
#endif
}

BEATSYNC_API int bs_ai_analyze_file(void* analyzer, const char* audio_path,
                                     bs_ai_result_t* out_result,
                                     bs_ai_progress_cb progress_cb, void* user_data) {
#ifndef USE_ONNX
    s_aiLastError = "ONNX Runtime not available";
    return -1;
#else
    TRACE_FUNC();

    if (!analyzer || !audio_path || !out_result) {
        s_aiLastError = "Invalid parameters";
        return -1;
    }

    try {
        auto* a = static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);

        // Load audio file using AudioAnalyzer
        BeatSync::AudioAnalyzer audioLoader;
        auto audioData = audioLoader.loadAudioFile(audio_path);

        if (audioData.samples.empty()) {
            s_aiLastError = "Failed to load audio file: " + audioLoader.getLastError();
            return -1;
        }

        // Convert mono to stereo (interleaved)
        std::vector<float> stereoSamples(audioData.samples.size() * 2);
        for (size_t i = 0; i < audioData.samples.size(); ++i) {
            stereoSamples[i * 2] = audioData.samples[i];
            stereoSamples[i * 2 + 1] = audioData.samples[i];
        }

        // Progress wrapper
        BeatSync::MusicAnalysisProgress progressWrapper = nullptr;
        if (progress_cb) {
            progressWrapper = [progress_cb, user_data](float progress, const std::string& stage, const std::string& message) {
                return progress_cb(progress, stage.c_str(), message.c_str(), user_data) != 0;
            };
        }

        // Run analysis
        BeatSync::MusicAnalysisResult result = a->analyze(stereoSamples, audioData.sampleRate, progressWrapper);

        // Copy results to C struct
        memset(out_result, 0, sizeof(bs_ai_result_t));

        // Beats
        if (!result.beats.empty()) {
            out_result->beats = static_cast<double*>(malloc(result.beats.size() * sizeof(double)));
            if (out_result->beats) {
                memcpy(out_result->beats, result.beats.data(), result.beats.size() * sizeof(double));
                out_result->beat_count = result.beats.size();
            }
        }

        // Downbeats
        if (!result.downbeats.empty()) {
            out_result->downbeats = static_cast<double*>(malloc(result.downbeats.size() * sizeof(double)));
            if (out_result->downbeats) {
                memcpy(out_result->downbeats, result.downbeats.data(), result.downbeats.size() * sizeof(double));
                out_result->downbeat_count = result.downbeats.size();
            }
        }

        out_result->bpm = result.bpm;
        out_result->duration = result.duration;

        // Segments
        if (!result.segments.empty()) {
            size_t segSize = result.segments.size() * sizeof(out_result->segments[0]);
            out_result->segments = static_cast<decltype(out_result->segments)>(malloc(segSize));

            if (out_result->segments) {
                for (size_t i = 0; i < result.segments.size(); ++i) {
                    out_result->segments[i].start_time = result.segments[i].startTime;
                    out_result->segments[i].end_time = result.segments[i].endTime;
                    // Allocate and copy label string
                    if (!result.segments[i].label.empty()) {
                        size_t len = result.segments[i].label.size();
                        char* labelCopy = static_cast<char*>(malloc(len + 1));
                        memcpy(labelCopy, result.segments[i].label.c_str(), len);
                        labelCopy[len] = '\0';
                        out_result->segments[i].label = labelCopy;
                    } else {
                        out_result->segments[i].label = nullptr;
                    }
                    out_result->segments[i].confidence = result.segments[i].confidence;
                }
                out_result->segment_count = result.segments.size();
            }
        }

        return 0;

    } catch (const std::exception& e) {
        s_aiLastError = e.what();
        return -1;
    } catch (...) {
        s_aiLastError = "Unknown error during AI analysis";
        return -1;
    }
#endif
}

BEATSYNC_API int bs_ai_analyze_samples(void* analyzer,
                                        const float* samples, size_t sample_count,
                                        int sample_rate, int num_channels,
                                        bs_ai_result_t* out_result,
                                        bs_ai_progress_cb progress_cb, void* user_data) {
#ifndef USE_ONNX
    s_aiLastError = "ONNX Runtime not available";
    return -1;
#else
    TRACE_FUNC();

    if (!analyzer || !samples || sample_count == 0 || !out_result) {
        s_aiLastError = "Invalid parameters";
        return -1;
    }

    try {
        auto* a = static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);

        // Prepare samples (ensure stereo interleaved)
        std::vector<float> stereoSamples;
        if (num_channels == 1) {
            // Convert mono to stereo
            stereoSamples.resize(sample_count * 2);
            for (size_t i = 0; i < sample_count; ++i) {
                stereoSamples[i * 2] = samples[i];
                stereoSamples[i * 2 + 1] = samples[i];
            }
        } else if (num_channels == 2) {
            stereoSamples.assign(samples, samples + sample_count);
        } else {
            // Mix down to stereo
            size_t numFrames = sample_count / num_channels;
            stereoSamples.resize(numFrames * 2);
            for (size_t i = 0; i < numFrames; ++i) {
                float sum = 0.0f;
                for (int c = 0; c < num_channels; ++c) {
                    sum += samples[i * num_channels + c];
                }
                float avg = sum / num_channels;
                stereoSamples[i * 2] = avg;
                stereoSamples[i * 2 + 1] = avg;
            }
        }

        // Progress wrapper
        BeatSync::MusicAnalysisProgress progressWrapper = nullptr;
        if (progress_cb) {
            progressWrapper = [progress_cb, user_data](float progress, const std::string& stage, const std::string& message) {
                return progress_cb(progress, stage.c_str(), message.c_str(), user_data) != 0;
            };
        }

        // Run analysis
        BeatSync::MusicAnalysisResult result = a->analyze(stereoSamples, sample_rate, progressWrapper);

        // Copy results (same as bs_ai_analyze_file)
        memset(out_result, 0, sizeof(bs_ai_result_t));

        if (!result.beats.empty()) {
            out_result->beats = static_cast<double*>(malloc(result.beats.size() * sizeof(double)));
            if (out_result->beats) {
                memcpy(out_result->beats, result.beats.data(), result.beats.size() * sizeof(double));
                out_result->beat_count = result.beats.size();
            }
        }

        if (!result.downbeats.empty()) {
            out_result->downbeats = static_cast<double*>(malloc(result.downbeats.size() * sizeof(double)));
            if (out_result->downbeats) {
                memcpy(out_result->downbeats, result.downbeats.data(), result.downbeats.size() * sizeof(double));
                out_result->downbeat_count = result.downbeats.size();
            }
        }

        out_result->bpm = result.bpm;
        out_result->duration = result.duration;

        return 0;

    } catch (const std::exception& e) {
        s_aiLastError = e.what();
        return -1;
    }
#endif
}

BEATSYNC_API int bs_ai_analyze_quick(void* analyzer, const char* audio_path,
                                      bs_ai_result_t* out_result,
                                      bs_ai_progress_cb progress_cb, void* user_data) {
#ifndef USE_ONNX
    s_aiLastError = "ONNX Runtime not available";
    return -1;
#else
    TRACE_FUNC();

    if (!analyzer || !audio_path || !out_result) {
        s_aiLastError = "Invalid parameters";
        return -1;
    }

    try {
        auto* a = static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);

        // Load audio
        BeatSync::AudioAnalyzer audioLoader;
        auto audioData = audioLoader.loadAudioFile(audio_path);

        if (audioData.samples.empty()) {
            s_aiLastError = "Failed to load audio file: " + audioLoader.getLastError();
            return -1;
        }

        // Convert to stereo
        std::vector<float> stereoSamples(audioData.samples.size() * 2);
        for (size_t i = 0; i < audioData.samples.size(); ++i) {
            stereoSamples[i * 2] = audioData.samples[i];
            stereoSamples[i * 2 + 1] = audioData.samples[i];
        }

        // Progress wrapper
        BeatSync::MusicAnalysisProgress progressWrapper = nullptr;
        if (progress_cb) {
            progressWrapper = [progress_cb, user_data](float progress, const std::string& stage, const std::string& message) {
                return progress_cb(progress, stage.c_str(), message.c_str(), user_data) != 0;
            };
        }

        // Quick analysis (no stem separation)
        BeatSync::MusicAnalysisResult result = a->analyzeQuick(stereoSamples, audioData.sampleRate, progressWrapper);

        // Copy results
        memset(out_result, 0, sizeof(bs_ai_result_t));

        if (!result.beats.empty()) {
            out_result->beats = static_cast<double*>(malloc(result.beats.size() * sizeof(double)));
            if (out_result->beats) {
                memcpy(out_result->beats, result.beats.data(), result.beats.size() * sizeof(double));
                out_result->beat_count = result.beats.size();
            }
        }

        if (!result.downbeats.empty()) {
            out_result->downbeats = static_cast<double*>(malloc(result.downbeats.size() * sizeof(double)));
            if (out_result->downbeats) {
                memcpy(out_result->downbeats, result.downbeats.data(), result.downbeats.size() * sizeof(double));
                out_result->downbeat_count = result.downbeats.size();
            }
        }

        out_result->bpm = result.bpm;
        out_result->duration = result.duration;

        return 0;

    } catch (const std::exception& e) {
        s_aiLastError = e.what();
        return -1;
    }
#endif
}

BEATSYNC_API void bs_free_ai_result(bs_ai_result_t* result) {
    if (!result) return;

    if (result->beats) {
        free(result->beats);
        result->beats = nullptr;
    }
    if (result->downbeats) {
        free(result->downbeats);
        result->downbeats = nullptr;
    }
    if (result->segments) {
        for (size_t i = 0; i < result->segment_count; ++i) {
            if (result->segments[i].label) {
                free((void*)result->segments[i].label);
                result->segments[i].label = nullptr;
            }
        }
        free(result->segments);
        result->segments = nullptr;
    }

    result->beat_count = 0;
    result->downbeat_count = 0;
    result->segment_count = 0;
}

BEATSYNC_API const char* bs_ai_get_last_error(void* analyzer) {
#ifdef USE_ONNX
    if (analyzer) {
        auto* a = static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);
        s_aiLastError = a->getLastError();
    }
#else
    (void)analyzer;
#endif
    return s_aiLastError.c_str();
}

BEATSYNC_API const char* bs_ai_get_model_info(void* analyzer) {
#ifdef USE_ONNX
    if (analyzer) {
        auto* a = static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);
        s_aiModelInfo = a->getModelInfo();
        return s_aiModelInfo.c_str();
    }
#else
    (void)analyzer;
#endif
    return "";
}

BEATSYNC_API int bs_ai_is_available() {
#ifdef USE_ONNX
    return BeatSync::OnnxBeatDetector::isOnnxRuntimeAvailable() ? 1 : 0;
#else
    return 0;
#endif
}

BEATSYNC_API const char* bs_ai_get_providers() {
#ifdef USE_ONNX
    auto providers = BeatSync::OnnxBeatDetector::getAvailableProviders();
    s_aiProviders.clear();
    for (size_t i = 0; i < providers.size(); ++i) {
        if (i > 0) s_aiProviders += ", ";
        s_aiProviders += providers[i];
    }
    return s_aiProviders.c_str();
#else
    return "";
#endif
}

} // extern "C"
