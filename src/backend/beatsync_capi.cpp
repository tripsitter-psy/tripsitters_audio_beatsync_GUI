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

#ifdef USE_AUDIOFLUX
#include "../audio/AudioFluxBeatDetector.h"
#endif

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <excpt.h>
#endif

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
} // extern "C" closed after C API symbols

extern "C" {

// ==================== AudioAnalyzer API ====================

BEATSYNC_API void* bs_create_audio_analyzer() {
    try {
        return new BeatSync::AudioAnalyzer();
    } catch (const std::exception& e) {
        s_lastError = std::string("Error creating AudioAnalyzer: ") + e.what();
        return nullptr;
    } catch (...) {
        s_lastError = "Unknown error creating AudioAnalyzer";
        return nullptr;
    }
}

BEATSYNC_API void bs_destroy_audio_analyzer(void* analyzer) {
    if (analyzer) {
        delete static_cast<BeatSync::AudioAnalyzer*>(analyzer);
    }
}

BEATSYNC_API void bs_set_bpm_hint(void* analyzer, double bpm) {
    if (analyzer) {
        static_cast<BeatSync::AudioAnalyzer*>(analyzer)->setBPMHint(bpm);
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

        // Downsample to approximately 20000 peak values for visualization
        // Higher resolution allows seeing individual transients (kick drums, snares)
        // At 44100 Hz, 20000 peaks for a 5-min track = ~15ms per peak
        const size_t targetPeaks = 20000;
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

// Helper: compute frequency band energy using AudioFlux STFT when available
#ifdef USE_AUDIOFLUX
extern "C" {
#include "flux_base.h"
#include "stft_algorithm.h"
}
#endif

BEATSYNC_API int bs_get_waveform_bands(void* analyzer, const char* filepath,
                                        bs_waveform_bands_t* out_bands) {
    TRACE_FUNC();
    if (!analyzer || !filepath || !out_bands) {
        return -1;
    }

    // Initialize output to zeros
    out_bands->bass_peaks = nullptr;
    out_bands->mid_peaks = nullptr;
    out_bands->high_peaks = nullptr;
    out_bands->count = 0;
    out_bands->duration = 0.0;

    try {
        auto* a = static_cast<BeatSync::AudioAnalyzer*>(analyzer);
        auto audioData = a->loadAudioFile(filepath);

        if (audioData.samples.empty()) {
            s_lastError = "Failed to load audio file for waveform bands";
            return -1;
        }

        out_bands->duration = audioData.duration;
        const int sampleRate = audioData.sampleRate;

        // FFT parameters - 2048 samples at 44100 Hz gives ~21 Hz per bin
        const int radix2Exp = 11;  // 2^11 = 2048
        const size_t fftSize = 1 << radix2Exp;
        const size_t hopSize = 256;  // Smaller hop = more time resolution
        const size_t targetPeaks = 20000;  // High resolution for seeing kick transients

        // Calculate frequency bin ranges
        // Bass: 20-200 Hz, Mids: 200-2000 Hz, Highs: 2000+ Hz
        const double binWidth = static_cast<double>(sampleRate) / fftSize;
        const size_t bassStartBin = static_cast<size_t>(20.0 / binWidth);
        const size_t bassEndBin = static_cast<size_t>(200.0 / binWidth);
        const size_t midEndBin = static_cast<size_t>(2000.0 / binWidth);
        const size_t highEndBin = fftSize / 2;

#ifdef USE_AUDIOFLUX
        // Use AudioFlux for fast STFT computation
        STFTObj stftObj = nullptr;
        WindowType windowType = Window_Hann;
        int slideLength = static_cast<int>(hopSize);
        int isContinue = 0;

        if (stftObj_new(&stftObj, radix2Exp, &windowType, &slideLength, &isContinue) != 0 || !stftObj) {
            s_lastError = "Failed to create AudioFlux STFT object";
            return -1;
        }

        int numFrames = stftObj_calTimeLength(stftObj, static_cast<int>(audioData.samples.size()));
        if (numFrames <= 0) {
            stftObj_free(stftObj);
            s_lastError = "Audio too short for frequency analysis";
            return -1;
        }

        // Allocate STFT buffers (AudioFlux outputs fftLength values per frame)
        std::vector<float> stftReal(numFrames * fftSize, 0.0f);
        std::vector<float> stftImag(numFrames * fftSize, 0.0f);

        // Compute STFT
        stftObj_stft(stftObj, audioData.samples.data(), static_cast<int>(audioData.samples.size()),
                     stftReal.data(), stftImag.data());
        stftObj_free(stftObj);

        // Extract energy per band from STFT
        std::vector<float> bassEnergy(numFrames, 0.0f);
        std::vector<float> midEnergy(numFrames, 0.0f);
        std::vector<float> highEnergy(numFrames, 0.0f);

        for (int frame = 0; frame < numFrames; ++frame) {
            float bassSum = 0.0f, midSum = 0.0f, highSum = 0.0f;

            // Sum magnitude in each band
            for (size_t bin = bassStartBin; bin <= bassEndBin && bin < fftSize/2; ++bin) {
                size_t idx = frame * fftSize + bin;
                float mag = std::sqrt(stftReal[idx] * stftReal[idx] + stftImag[idx] * stftImag[idx]);
                bassSum += mag;
            }
            for (size_t bin = bassEndBin + 1; bin <= midEndBin && bin < fftSize/2; ++bin) {
                size_t idx = frame * fftSize + bin;
                float mag = std::sqrt(stftReal[idx] * stftReal[idx] + stftImag[idx] * stftImag[idx]);
                midSum += mag;
            }
            for (size_t bin = midEndBin + 1; bin < highEndBin; ++bin) {
                size_t idx = frame * fftSize + bin;
                float mag = std::sqrt(stftReal[idx] * stftReal[idx] + stftImag[idx] * stftImag[idx]);
                highSum += mag;
            }

            // Normalize by bin count
            size_t bassBins = bassEndBin - bassStartBin + 1;
            size_t midBins = midEndBin - bassEndBin;
            size_t highBins = highEndBin - midEndBin - 1;

            bassEnergy[frame] = bassBins > 0 ? bassSum / bassBins : 0.0f;
            midEnergy[frame] = midBins > 0 ? midSum / midBins : 0.0f;
            highEnergy[frame] = highBins > 0 ? highSum / highBins : 0.0f;
        }

#else
        // Fallback: compute energy using simple bandpass approximation
        // This is faster than DFT but less accurate
        if (audioData.samples.size() < fftSize) {
            s_lastError = "Audio too short for frequency analysis";
            return -1;
        }
        size_t numFrames = (audioData.samples.size() - fftSize) / hopSize + 1;

        std::vector<float> bassEnergy(numFrames, 0.0f);
        std::vector<float> midEnergy(numFrames, 0.0f);
        std::vector<float> highEnergy(numFrames, 0.0f);

        // Simple energy estimation using difference (high-pass) and smoothing (low-pass)
        for (size_t frame = 0; frame < numFrames; ++frame) {
            size_t start = frame * hopSize;
            size_t end = std::min(start + fftSize, audioData.samples.size());

            float sumLow = 0.0f, sumHigh = 0.0f, sumTotal = 0.0f;
            float prevSample = 0.0f;

            for (size_t i = start; i < end; ++i) {
                float sample = audioData.samples[i];
                float absSample = std::abs(sample);
                sumTotal += absSample;

                // High frequencies: difference between consecutive samples
                float diff = std::abs(sample - prevSample);
                sumHigh += diff;
                prevSample = sample;
            }

            // Approximate: bass = smoothed total, highs = differences, mids = remainder
            bassEnergy[frame] = sumTotal / (end - start);
            highEnergy[frame] = sumHigh / (end - start);
            midEnergy[frame] = std::max(0.0f, bassEnergy[frame] - highEnergy[frame] * 0.5f);
        }
#endif

        // Normalize each band to 0-1 range
        auto normalizeArray = [](std::vector<float>& arr) {
            float maxVal = 0.0f;
            for (float v : arr) {
                if (v > maxVal) maxVal = v;
            }
            if (maxVal > 0.0f) {
                for (float& v : arr) {
                    v /= maxVal;
                }
            }
        };

        normalizeArray(bassEnergy);
        normalizeArray(midEnergy);
        normalizeArray(highEnergy);

        // Downsample to target number of peaks
        size_t numFramesU = static_cast<size_t>(numFrames);
        size_t framesPerPeak = numFramesU / targetPeaks;
        if (framesPerPeak < 1) framesPerPeak = 1;
        size_t actualPeaks = (numFramesU + framesPerPeak - 1) / framesPerPeak;

        // Allocate output arrays
        out_bands->bass_peaks = static_cast<float*>(malloc(actualPeaks * sizeof(float)));
        out_bands->mid_peaks = static_cast<float*>(malloc(actualPeaks * sizeof(float)));
        out_bands->high_peaks = static_cast<float*>(malloc(actualPeaks * sizeof(float)));

        if (!out_bands->bass_peaks || !out_bands->mid_peaks || !out_bands->high_peaks) {
            free(out_bands->bass_peaks);
            free(out_bands->mid_peaks);
            free(out_bands->high_peaks);
            out_bands->bass_peaks = nullptr;
            out_bands->mid_peaks = nullptr;
            out_bands->high_peaks = nullptr;
            s_lastError = "Memory allocation failure for waveform bands";
            return -1;
        }

        // Downsample by taking max in each window
        for (size_t i = 0; i < actualPeaks; ++i) {
            size_t start = i * framesPerPeak;
            size_t end = std::min(start + framesPerPeak, numFramesU);

            float maxBass = 0.0f, maxMid = 0.0f, maxHigh = 0.0f;
            for (size_t j = start; j < end; ++j) {
                if (bassEnergy[j] > maxBass) maxBass = bassEnergy[j];
                if (midEnergy[j] > maxMid) maxMid = midEnergy[j];
                if (highEnergy[j] > maxHigh) maxHigh = highEnergy[j];
            }

            out_bands->bass_peaks[i] = maxBass;
            out_bands->mid_peaks[i] = maxMid;
            out_bands->high_peaks[i] = maxHigh;
        }

        out_bands->count = actualPeaks;
        return 0;

    } catch (const std::exception& e) {
        s_lastError = e.what();
        return -1;
    } catch (...) {
        s_lastError = "Exception during waveform bands generation";
        return -1;
    }
}

BEATSYNC_API void bs_free_waveform_bands(bs_waveform_bands_t* bands) {
    if (bands) {
        free(bands->bass_peaks);
        free(bands->mid_peaks);
        free(bands->high_peaks);
        bands->bass_peaks = nullptr;
        bands->mid_peaks = nullptr;
        bands->high_peaks = nullptr;
        bands->count = 0;
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
    } catch (const std::exception& e) {
        s_lastError = std::string("Error creating VideoWriter: ") + e.what();
        return nullptr;
    } catch (...) {
        s_lastError = "Unknown exception creating VideoWriter";
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

    // File-based logging for diagnostics - SINGLE VIDEO PATH
    // Declare outside #ifdef so references compile in all builds
    std::string logPath;
    FILE* logFile = nullptr;
#ifdef ENABLE_BS_DEBUG_LOG
#ifdef _WIN32
    wchar_t tempPath[MAX_PATH + 1];
    if (GetTempPathW(MAX_PATH + 1, tempPath) > 0) {
        char narrowPath[MAX_PATH + 1];
        WideCharToMultiByte(CP_UTF8, 0, tempPath, -1, narrowPath, MAX_PATH + 1, nullptr, nullptr);
        logPath = std::string(narrowPath) + "beatsync_single_cut.log";
    }
#endif
    if (!logPath.empty()) {
        logFile = fopen(logPath.c_str(), "w");
        if (logFile) {
            fprintf(logFile, "=== bs_video_cut_at_beats (SINGLE) ENTERED (v2025.01.20) ===\n");
            fprintf(logFile, "writer=%p, inputVideo=%s\n", writer, inputVideo ? inputVideo : "(null)");
            fprintf(logFile, "beatTimes=%p, count=%zu, clipDuration=%.6f\n", (void*)beatTimes, count, clipDuration);
            fprintf(logFile, "outputVideo=%s\n", outputVideo ? outputVideo : "(null)");
            fflush(logFile);
        }
    }
#endif

    if (!writer || !inputVideo || !beatTimes || !outputVideo || count == 0) {
        if (logFile) {
            fprintf(logFile, "ERROR: Invalid parameters\n");
            fclose(logFile);
        }
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
        // Fallback for single-beat case
        if (clipDuration <= 0 && count == 1) {
            clipDuration = 1.0; // Default to 1 second if only one beat and no duration
        }

        if (logFile) {
            fprintf(logFile, "Calling cutAtBeats with clipDuration=%.6f, beatCount=%zu\n", clipDuration, count);
            fflush(logFile);
        }

        bool success = w->cutAtBeats(inputVideo, grid, outputVideo, clipDuration);

        if (logFile) {
            fprintf(logFile, "cutAtBeats returned %s\n", success ? "SUCCESS" : "FAILURE");
            if (!success) {
                fprintf(logFile, "Last error: %s\n", w->getLastError().c_str());
            }
            fclose(logFile);
        }
        return success ? 0 : -1;
    } catch (const std::exception& e) {
        if (logFile) {
            fprintf(logFile, "EXCEPTION: %s\n", e.what());
            fclose(logFile);
        }
        return -1;
    } catch (...) {
        if (logFile) {
            fprintf(logFile, "UNKNOWN EXCEPTION\n");
            fclose(logFile);
        }
        return -1;
    }
}

BEATSYNC_API int bs_video_cut_at_beats_multi(void* writer, const char** inputVideos, size_t videoCount,
                                              const double* beatTimes, size_t beatCount,
                                              const char* outputVideo, double clipDuration) {
    TRACE_FUNC();

    // File-based logging for diagnostics - MUST be first thing to catch all issues
    std::string logPath;
    FILE* logFile = nullptr;
#ifdef _WIN32
    wchar_t tempPath[MAX_PATH + 1];
    if (GetTempPathW(MAX_PATH + 1, tempPath) > 0) {
        char narrowPath[MAX_PATH + 1];
        WideCharToMultiByte(CP_UTF8, 0, tempPath, -1, narrowPath, MAX_PATH + 1, nullptr, nullptr);
        logPath = std::string(narrowPath) + "beatsync_multi_cut.log";
    }
#endif
    if (!logPath.empty()) {
        logFile = fopen(logPath.c_str(), "w");
        if (logFile) {
            fprintf(logFile, "=== bs_video_cut_at_beats_multi ENTERED (v2025.01.20) ===\n");
            fprintf(logFile, "writer=%p, inputVideos=%p, videoCount=%zu\n", writer, (void*)inputVideos, videoCount);
            fprintf(logFile, "beatTimes=%p, beatCount=%zu, clipDuration=%.6f\n", (void*)beatTimes, beatCount, clipDuration);
            fprintf(logFile, "outputVideo=%s\n", outputVideo ? outputVideo : "(null)");
            fflush(logFile);
        }
    }

    if (!writer || !inputVideos || !beatTimes || !outputVideo || videoCount == 0 || beatCount == 0) {
        s_lastError = "Invalid parameters to bs_video_cut_at_beats_multi";
        if (logFile) {
            fprintf(logFile, "ERROR: Invalid parameters - writer=%d inputVideos=%d beatTimes=%d outputVideo=%d videoCount=%zu beatCount=%zu\n",
                    writer != nullptr, inputVideos != nullptr, beatTimes != nullptr, outputVideo != nullptr, videoCount, beatCount);
            fclose(logFile);
        }
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
            if (logFile) { fprintf(logFile, "ERROR: No input videos provided\n"); fclose(logFile); }
            return -1;
        }

        if (logFile) {
            fprintf(logFile, "Collected %zu videos:\n", videos.size());
            for (size_t i = 0; i < videos.size(); ++i) {
                fprintf(logFile, "  [%zu] %s\n", i, videos[i].c_str());
            }
            fflush(logFile);
        }

        // Precompute and cache video durations
        std::vector<double> durations(videos.size(), 0.0);
        for (size_t i = 0; i < videos.size(); ++i) {
            BeatSync::VideoProcessor proc;
            if (proc.open(videos[i])) {
                durations[i] = proc.getInfo().duration;
                proc.close();
            }
            if (logFile) {
                fprintf(logFile, "Video[%zu] duration=%.3f sec\n", i, durations[i]);
            }
            // If duration <= 0, we'll use DEFAULT_DURATION later
        }
        if (logFile) fflush(logFile);

        // ================================================================
        // GENERALIZED GAP HANDLING: Detect long gaps anywhere in the timeline
        // ================================================================
        // Long gaps (intro, breakdown, outro) should play full video clips
        // without cutting. Normal beat intervals get cut as usual.
        // "Long gap" = gap > 3x the average beat interval

        // Step 1: Calculate average beat interval to detect long gaps
        double avgBeatInterval = clipDuration;  // Default fallback
        if (beatCount >= 2) {
            double totalInterval = 0.0;
            size_t intervalCount = 0;
            for (size_t i = 1; i < beatCount; ++i) {
                double interval = beatTimes[i] - beatTimes[i - 1];
                // Only count "normal" intervals (skip outliers that are likely gaps)
                if (interval > 0.05 && interval < 5.0) {
                    totalInterval += interval;
                    intervalCount++;
                }
            }
            if (intervalCount > 0) {
                avgBeatInterval = totalInterval / intervalCount;
            }
        }

        // Gap threshold: 3x average interval indicates a "break" section
        const double GAP_THRESHOLD_MULTIPLIER = 3.0;
        double longGapThreshold = avgBeatInterval * GAP_THRESHOLD_MULTIPLIER;

        if (logFile) {
            fprintf(logFile, "\n=== GAP DETECTION ===\n");
            fprintf(logFile, "Average beat interval: %.3f sec\n", avgBeatInterval);
            fprintf(logFile, "Long gap threshold (3x avg): %.3f sec\n", longGapThreshold);
            fflush(logFile);
        }

        // Step 2: Create sorted list of videos by duration (longest first) for filling gaps
        std::vector<std::pair<double, size_t>> sortedByDuration;
        for (size_t i = 0; i < videos.size(); ++i) {
            double dur = durations[i] > 0 ? durations[i] : DEFAULT_DURATION;
            sortedByDuration.push_back({dur, i});
        }
        std::sort(sortedByDuration.begin(), sortedByDuration.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        if (logFile) {
            fprintf(logFile, "Videos sorted by duration (longest first):\n");
            for (size_t i = 0; i < std::min(sortedByDuration.size(), (size_t)5); ++i) {
                fprintf(logFile, "  [%zu] %.3f sec\n", sortedByDuration[i].second, sortedByDuration[i].first);
            }
            if (sortedByDuration.size() > 5) {
                fprintf(logFile, "  ... and %zu more\n", sortedByDuration.size() - 5);
            }
            fflush(logFile);
        }

        // Helper lambda to fill a time gap with full-length videos (longest first)
        size_t totalSegmentCount = 0;
        size_t longVideoIdx = 0;  // Index into sortedByDuration for gap filling

        auto fillGapWithFullVideos = [&](double gapStart, double gapEnd, const char* gapType) {
            double gapDuration = gapEnd - gapStart;
            if (gapDuration < 0.1) return;  // Skip tiny gaps

            if (logFile) {
                fprintf(logFile, "\n=== %s SECTION (%.3f to %.3f sec, duration=%.3f) ===\n",
                        gapType, gapStart, gapEnd, gapDuration);
                fflush(logFile);
            }

            double timeRemaining = gapDuration;
            size_t gapSegCount = 0;

            while (timeRemaining > 0.1) {
                // Get next longest video (cycling if needed)
                size_t vidIdx = sortedByDuration[longVideoIdx % sortedByDuration.size()].second;
                double vidDuration = sortedByDuration[longVideoIdx % sortedByDuration.size()].first;

                // Use full video duration or remaining time, whichever is smaller
                double segmentDuration = std::min(vidDuration, timeRemaining);

                // Generate temp file name
                std::string outStr(outputVideo);
                size_t dotPos = outStr.rfind('.');
                std::string tempFile;
                if (dotPos != std::string::npos) {
                    tempFile = outStr.substr(0, dotPos) + "_seg" + std::to_string(totalSegmentCount) + outStr.substr(dotPos);
                } else {
                    tempFile = outStr + "_seg" + std::to_string(totalSegmentCount) + ".mp4";
                }

                if (logFile) {
                    fprintf(logFile, "GapSeg[%zu] video[%zu]=%s full_duration=%.3f using=%.3f (remaining=%.3f) -> %s\n",
                            gapSegCount, vidIdx, videos[vidIdx].c_str(), vidDuration, segmentDuration, timeRemaining, tempFile.c_str());
                    fflush(logFile);
                }

                // Extract from start of video (full clip, no cutting)
                if (w->copySegmentFast(videos[vidIdx], 0.0, segmentDuration, tempFile)) {
                    tempFiles.push_back(tempFile);
                    timeRemaining -= segmentDuration;
                    totalSegmentCount++;
                    gapSegCount++;
                } else {
                    std::string errMsg = w->getLastError();
                    if (logFile) {
                        fprintf(logFile, "WARNING: Failed to copy gap segment: %s (continuing...)\n", errMsg.c_str());
                        fflush(logFile);
                    }
                }

                longVideoIdx++;
            }

            if (logFile) {
                fprintf(logFile, "%s complete: %zu segments\n", gapType, gapSegCount);
                fflush(logFile);
            }
        };

        // Step 3: Process timeline - handle gaps and beat-synced sections
        size_t beatVideoIdx = 0;  // Cycling index for beat-synced sections
        const double MIN_SEGMENT_DURATION = 0.1;

        // Check for intro gap (before first beat)
        if (beatTimes[0] > longGapThreshold) {
            fillGapWithFullVideos(0.0, beatTimes[0], "INTRO");
        }

        if (logFile) {
            fprintf(logFile, "\n=== PROCESSING %zu BEATS ===\n", beatCount);
            fflush(logFile);
        }

        for (size_t i = 0; i < beatCount; ++i) {
            double beatTime = beatTimes[i];
            double nextTime;

            if (i + 1 < beatCount) {
                nextTime = beatTimes[i + 1];
            } else {
                // Last beat - use clipDuration for final segment
                nextTime = beatTime + std::max(clipDuration, MIN_SEGMENT_DURATION);
            }

            double gapDuration = nextTime - beatTime;

            // Check if this is a long gap (breakdown section)
            if (gapDuration > longGapThreshold && i + 1 < beatCount) {
                // This is a breakdown/gap section - fill with full videos
                fillGapWithFullVideos(beatTime, nextTime, "BREAKDOWN");
            } else {
                // Normal beat interval - cut as usual
                double duration = gapDuration;
                if (duration <= 0) {
                    duration = MIN_SEGMENT_DURATION;
                }

                // Generate temp file name
                std::string outStr(outputVideo);
                size_t dotPos = outStr.rfind('.');
                std::string tempFile;
                if (dotPos != std::string::npos) {
                    tempFile = outStr.substr(0, dotPos) + "_seg" + std::to_string(totalSegmentCount) + outStr.substr(dotPos);
                } else {
                    tempFile = outStr + "_seg" + std::to_string(totalSegmentCount) + ".mp4";
                }

                // Use segment start time as position within source video (modular approach)
                double sourceStart = 0.0;
                double sourceDuration = duration;
                double cachedDuration = durations[beatVideoIdx];

                if (cachedDuration > 0) {
                    sourceStart = fmod(beatTime, cachedDuration);

                    // Clamp tiny values to 0
                    if (sourceStart < 0.001) {
                        sourceStart = 0.0;
                    }

                    // Ensure we don't exceed available content
                    double availableFromStart = cachedDuration - sourceStart;
                    if (availableFromStart < sourceDuration) {
                        if (cachedDuration >= sourceDuration) {
                            sourceStart = 0.0;
                        } else {
                            sourceStart = 0.0;
                            sourceDuration = cachedDuration;
                        }
                    }
                } else {
                    sourceStart = fmod(beatTime, DEFAULT_DURATION);
                }

                // Log periodically
                if (logFile && (totalSegmentCount < 5 || totalSegmentCount % 100 == 0)) {
                    fprintf(logFile, "BeatSeg[%zu] beat[%zu] video[%zu]=%s sourceStart=%.6f dur=%.6f -> %s\n",
                            totalSegmentCount, i, beatVideoIdx, videos[beatVideoIdx].c_str(),
                            sourceStart, sourceDuration, tempFile.c_str());
                    fflush(logFile);
                }

                if (w->copySegmentFast(videos[beatVideoIdx], sourceStart, sourceDuration, tempFile)) {
                    tempFiles.push_back(tempFile);
                    totalSegmentCount++;
                } else {
                    std::string errMsg = w->getLastError();
                    s_lastError = "Failed to copy video segment " + std::to_string(i) + ": " + errMsg;
                    if (logFile) {
                        fprintf(logFile, "FAILED BeatSeg[%zu]: beatVideoIdx=%zu sourceStart=%.6f dur=%.6f err=%s\n",
                                totalSegmentCount, beatVideoIdx, sourceStart, sourceDuration, errMsg.c_str());
                        fclose(logFile);
                    }
                    return -1;
                }

                beatVideoIdx = (beatVideoIdx + 1) % videos.size();
            }
        }

        if (tempFiles.empty()) {
            s_lastError = "Failed to extract any segments";
            if (logFile) { fprintf(logFile, "ERROR: No segments extracted\n"); fclose(logFile); }
            return -1;
        }

        if (logFile) {
            fprintf(logFile, "All %zu segments extracted, concatenating...\n", tempFiles.size());
            fflush(logFile);
        }

        // Concatenate all segments
        bool success = w->concatenateVideos(tempFiles, outputVideo);

        if (logFile) {
            fprintf(logFile, "Concatenation %s\n", success ? "SUCCEEDED" : "FAILED");
            if (!success) {
                fprintf(logFile, "Concat error: %s\n", w->getLastError().c_str());
            }
            fclose(logFile);
        }

        // Cleanup handled by TempFileCleanup destructor
        return success ? 0 : -1;
    } catch (const std::exception& e) {
        // Cleanup handled by TempFileCleanup destructor
        s_lastError = std::string("Error during video segment extraction: ") + e.what();
        if (logFile) { fprintf(logFile, "EXCEPTION: %s\n", e.what()); fclose(logFile); }
        return -1;
    } catch (...) {
        // Cleanup handled by TempFileCleanup destructor
        s_lastError = "Unknown exception in bs_video_extract_segments";
        if (logFile) { fprintf(logFile, "UNKNOWN EXCEPTION\n"); fclose(logFile); }
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

BEATSYNC_API int bs_video_normalize_sources(void* writer, const char** inputVideos, size_t videoCount,
                                             char** normalizedPaths, size_t pathBufferSize) {
    TRACE_FUNC();

    // File-based logging for diagnostics - NORMALIZE
    std::string logPath;
    FILE* logFile = nullptr;
#ifdef _WIN32
    wchar_t tempPath[MAX_PATH + 1];
    if (GetTempPathW(MAX_PATH + 1, tempPath) > 0) {
        char narrowPath[MAX_PATH + 1];
        WideCharToMultiByte(CP_UTF8, 0, tempPath, -1, narrowPath, MAX_PATH + 1, nullptr, nullptr);
        logPath = std::string(narrowPath) + "beatsync_normalize.log";
    }
#endif
    if (!logPath.empty()) {
        logFile = fopen(logPath.c_str(), "w");
        if (logFile) {
            fprintf(logFile, "=== bs_video_normalize_sources ENTERED (v2025.01.20) ===\n");
            fprintf(logFile, "writer=%p, inputVideos=%p, videoCount=%zu\n", writer, (void*)inputVideos, videoCount);
            fprintf(logFile, "normalizedPaths=%p, pathBufferSize=%zu\n", (void*)normalizedPaths, pathBufferSize);
            fflush(logFile);
        }
    }

    if (!writer || !inputVideos || !normalizedPaths || videoCount == 0 || pathBufferSize == 0) {
        if (logFile) {
            fprintf(logFile, "ERROR: Invalid parameters\n");
            fclose(logFile);
        }
        return -1;
    }

    try {
        auto* w = static_cast<BeatSync::VideoWriter*>(writer);

        std::vector<std::string> inputs;
        for (size_t i = 0; i < videoCount; ++i) {
            if (inputVideos[i]) {
                inputs.push_back(inputVideos[i]);
            }
        }

        if (inputs.empty()) {
            s_lastError = "No valid input videos provided";
            if (logFile) { fprintf(logFile, "ERROR: No valid input videos\n"); fclose(logFile); }
            return -1;
        }

        if (logFile) {
            fprintf(logFile, "Collected %zu input videos, calling normalizeVideos...\n", inputs.size());
            fflush(logFile);
        }

        std::vector<std::string> outputPaths;
        if (!w->normalizeVideos(inputs, outputPaths)) {
            s_lastError = w->getLastError();
            if (logFile) {
                fprintf(logFile, "ERROR: normalizeVideos failed: %s\n", s_lastError.c_str());
                fclose(logFile);
            }
            return -1;
        }

        if (logFile) {
            fprintf(logFile, "normalizeVideos returned %zu output paths\n", outputPaths.size());
        }

        // Copy paths to output buffers
        for (size_t i = 0; i < outputPaths.size() && i < videoCount; ++i) {
            if (!normalizedPaths[i]) continue;
            if (outputPaths[i].length() >= pathBufferSize) {
                s_lastError = "Normalized path exceeds buffer size for video " + std::to_string(i);
                if (logFile) { fprintf(logFile, "ERROR: %s\n", s_lastError.c_str()); fclose(logFile); }
                return -1;
            }
            strncpy(normalizedPaths[i], outputPaths[i].c_str(), pathBufferSize - 1);
            normalizedPaths[i][pathBufferSize - 1] = '\0';
        }

        if (logFile) {
            fprintf(logFile, "SUCCESS: Normalized %zu videos\n", outputPaths.size());
            fclose(logFile);
        }
        return 0;
    } catch (const std::exception& e) {
        s_lastError = e.what();
        if (logFile) { fprintf(logFile, "EXCEPTION: %s\n", e.what()); fclose(logFile); }
        return -1;
    } catch (...) {
        s_lastError = "Unknown exception in bs_video_normalize_sources";
        if (logFile) { fprintf(logFile, "UNKNOWN EXCEPTION\n"); fclose(logFile); }
        return -1;
    }
}

BEATSYNC_API void bs_video_cleanup_normalized(char** normalizedPaths, size_t count) {
    if (!normalizedPaths) return;

    for (size_t i = 0; i < count; ++i) {
        if (normalizedPaths[i] && normalizedPaths[i][0] != '\0') {
            std::remove(normalizedPaths[i]);
        }
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

BEATSYNC_API int bs_video_set_effects_config(void* writer, const bs_effects_config_t* config) {
    if (!writer) return 1;
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
            return 0;
        }

        // Basic validation: transitionType required only if enableTransitions is set
        if ((config->enableTransitions && config->transitionType == nullptr) || config->effectBeatDivisor <= 0) {
            s_lastError = "Invalid effects config: transitionType null when transitions enabled or effectBeatDivisor <= 0";
            return 2;
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

        cfg.effectStartTime = config->effectStartTime;
        cfg.effectEndTime = config->effectEndTime;

        // Store in our map for later retrieval
        {
            std::lock_guard<std::mutex> lock(s_effectsConfigsMutex);
            s_effectsConfigs[writer] = cfg;
        }

        w->setEffectsConfig(cfg);
        return 0;
    } catch (const std::exception& e) {
        s_lastError = e.what();
        return 3;
    } catch (...) {
        s_lastError = "unknown error in bs_video_set_effects_config";
        return 4;
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

        // Allocate output buffer with overflow check
        size_t rgbSize = 0;
        if (srcW > 0 && srcH > 0 && (size_t)srcW <= SIZE_MAX / (size_t)srcH && ((size_t)srcW * (size_t)srcH) <= SIZE_MAX / 3) {
            rgbSize = (size_t)srcW * (size_t)srcH * 3;
        } else {
            sws_freeContext(swsCtx);
            s_lastError = "Invalid or too large image dimensions for RGB buffer allocation";
            return -1;
        }
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
        // Use default threshold (0.5) if sentinel (-1.0 or any negative) is provided
        if (config->beat_threshold >= 0.0f) {
            cfg.beatConfig.beatThreshold = config->beat_threshold;
        } // else: use default
        if (config->downbeat_threshold >= 0.0f) {
            cfg.beatConfig.downbeatThreshold = config->downbeat_threshold;
        } // else: use default

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
        try {
            delete static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);
        } catch (const std::exception& e) {
            std::cerr << "[BeatSync] Warning: Exception destroying AI analyzer: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[BeatSync] Warning: Unknown exception destroying AI analyzer" << std::endl;
        }
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
            } else {
                // Allocation failed - no prior allocations to clean up at this point
                s_aiLastError = "Memory allocation failed while copying beats";
                out_result->beat_count = 0;
                return -1;
            }
        }

        // Downbeats
        if (!result.downbeats.empty()) {
            out_result->downbeats = static_cast<double*>(malloc(result.downbeats.size() * sizeof(double)));
            if (out_result->downbeats) {
                memcpy(out_result->downbeats, result.downbeats.data(), result.downbeats.size() * sizeof(double));
                out_result->downbeat_count = result.downbeats.size();
            } else {
                // Free beats if previously allocated
                if (out_result->beats) {
                    free(out_result->beats);
                    out_result->beats = nullptr;
                    out_result->beat_count = 0;
                }
                s_aiLastError = "Memory allocation failed while copying downbeats";
                out_result->downbeat_count = 0;
                return -1;
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
                        if (!labelCopy) {
                            // Allocation failed, cleanup all previous allocations
                            for (size_t j = 0; j < i; ++j) {
                                if (out_result->segments[j].label) free(out_result->segments[j].label);
                            }
                            free(out_result->segments);
                            out_result->segments = nullptr;
                            out_result->segment_count = 0;
                            if (out_result->beats) { free(out_result->beats); out_result->beats = nullptr; out_result->beat_count = 0; }
                            if (out_result->downbeats) { free(out_result->downbeats); out_result->downbeats = nullptr; out_result->downbeat_count = 0; }
                            return -1;
                        }
                        memcpy(labelCopy, result.segments[i].label.c_str(), len);
                        labelCopy[len] = '\0';
                        out_result->segments[i].label = labelCopy;
                    } else {
                        out_result->segments[i].label = nullptr;
                    }
                    out_result->segments[i].confidence = result.segments[i].confidence;
                }
                out_result->segment_count = result.segments.size();
            } else {
                // Allocation failed, cleanup previous allocations
                if (out_result->beats) { free(out_result->beats); out_result->beats = nullptr; out_result->beat_count = 0; }
                if (out_result->downbeats) { free(out_result->downbeats); out_result->downbeats = nullptr; out_result->downbeat_count = 0; }
                out_result->segments = nullptr;
                out_result->segment_count = 0;
                return -1;
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
            if (!out_result->beats) {
                // Allocation failed
                out_result->beat_count = 0;
                out_result->downbeats = nullptr;
                out_result->downbeat_count = 0;
                return -1;
            }
            memcpy(out_result->beats, result.beats.data(), result.beats.size() * sizeof(double));
            out_result->beat_count = result.beats.size();
        }

        if (!result.downbeats.empty()) {
            out_result->downbeats = static_cast<double*>(malloc(result.downbeats.size() * sizeof(double)));
            if (!out_result->downbeats) {
                // Allocation failed, cleanup beats if allocated
                if (out_result->beats) {
                    free(out_result->beats);
                    out_result->beats = nullptr;
                    out_result->beat_count = 0;
                }
                out_result->downbeat_count = 0;
                return -1;
            }
            memcpy(out_result->downbeats, result.downbeats.data(), result.downbeats.size() * sizeof(double));
            out_result->downbeat_count = result.downbeats.size();
        }


        out_result->bpm = result.bpm;
        out_result->duration = result.duration;
        // Segments are not returned by bs_ai_analyze_samples (see bs_ai_analyze_file for segment support)
        out_result->segments = nullptr;
        out_result->segment_count = 0;
        return 0;

    } catch (const std::exception& e) {
        s_aiLastError = e.what();
        return -1;
    } catch (...) {
        s_aiLastError = "Unknown exception";
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
            } else {
                // Allocation failed: return error (no other allocations to clean up at this point)
                out_result->beats = nullptr;
                out_result->beat_count = 0;
                s_aiLastError = "Memory allocation failed while copying beats";
                return -1;
            }
        }

        if (!result.downbeats.empty()) {
            out_result->downbeats = static_cast<double*>(malloc(result.downbeats.size() * sizeof(double)));
            if (out_result->downbeats) {
                memcpy(out_result->downbeats, result.downbeats.data(), result.downbeats.size() * sizeof(double));
                out_result->downbeat_count = result.downbeats.size();
            } else {
                // Allocation failed: free beats if previously allocated and return error
                if (out_result->beats) {
                    free(out_result->beats);
                    out_result->beats = nullptr;
                    out_result->beat_count = 0;
                }
                out_result->downbeats = nullptr;
                out_result->downbeat_count = 0;
                s_aiLastError = "Memory allocation failed while copying downbeats";
                return -1;
            }
        }

        out_result->bpm = result.bpm;
        out_result->duration = result.duration;

        return 0;

    } catch (const std::exception& e) {
        s_aiLastError = e.what();
        return -1;
    } catch (...) {
        s_aiLastError = "Unknown exception";
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

static thread_local std::string s_activeProvider;

BEATSYNC_API int bs_ai_is_gpu_enabled(void* analyzer) {
#ifdef USE_ONNX
    if (!analyzer) return 0;
    auto* musicAnalyzer = static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);
    // Access through the beat detector
    const auto* beatDetector = musicAnalyzer->getBeatDetector();
    if (!beatDetector) return 0;
    return beatDetector->isGPUEnabled() ? 1 : 0;
#else
    (void)analyzer;
    return 0;
#endif
}

BEATSYNC_API const char* bs_ai_get_active_provider(void* analyzer) {
#ifdef USE_ONNX
    if (!analyzer) {
        s_activeProvider = "None";
        return s_activeProvider.c_str();
    }
    auto* musicAnalyzer = static_cast<BeatSync::OnnxMusicAnalyzer*>(analyzer);
    const auto* beatDetector = musicAnalyzer->getBeatDetector();
    if (!beatDetector) {
        s_activeProvider = "None";
        return s_activeProvider.c_str();
    }
    s_activeProvider = beatDetector->getActiveProvider();
    return s_activeProvider.c_str();
#else
    (void)analyzer;
    s_activeProvider = "None";
    return s_activeProvider.c_str();
#endif
}

// =============================================================================
// AudioFlux Spectral Analysis Implementation
// =============================================================================

BEATSYNC_API int bs_audioflux_is_available() {
#ifdef USE_AUDIOFLUX
    try {
        return AudioFluxBeatDetector::isAvailable() ? 1 : 0;
    } catch (...) {
        return 0;
    }
#else
    return 0;
#endif
}

BEATSYNC_API int bs_audioflux_analyze(const char* audio_path,
                                       bs_ai_result_t* out_result,
                                       bs_ai_progress_cb progress_cb, void* user_data) {
#ifdef USE_AUDIOFLUX
    if (!audio_path || !out_result) {
        s_lastError = "Invalid parameters: null audio_path or out_result";
        return -1;
    }

    // Initialize result
    out_result->beats = nullptr;
    out_result->beat_count = 0;
    out_result->downbeats = nullptr;
    out_result->downbeat_count = 0;
    out_result->bpm = 0.0;
    out_result->duration = 0.0;
    out_result->segments = nullptr;
    out_result->segment_count = 0;

    try {
        // Load audio file
        BeatSync::AudioAnalyzer audioAnalyzer;
        auto audioData = audioAnalyzer.loadAudioFile(audio_path);
        if (audioData.samples.empty()) {
            s_lastError = "Failed to load audio file: " + std::string(audio_path);
            return -2;
        }

        // Create AudioFlux detector
        AudioFluxBeatDetector detector;

        // Create progress wrapper
        auto progressWrapper = [progress_cb, user_data](float progress, const char* stage) -> bool {
            if (progress_cb) {
                // Convention: nonzero return means continue, zero means cancel (matches ONNX wrapper)
                return progress_cb(progress, stage, "", user_data) != 0;
            }
            return true;
        };

        // Run detection
        auto result = detector.detect(audioData.samples, audioData.sampleRate, progressWrapper);

        if (!result.error.empty()) {
            s_lastError = result.error;
            return -3;
        }

        // Copy results
        out_result->beat_count = result.beats.size();
        if (out_result->beat_count > 0) {
            out_result->beats = static_cast<double*>(malloc(out_result->beat_count * sizeof(double)));
            if (!out_result->beats) {
                s_lastError = "Failed to allocate memory for beats array";
                out_result->beat_count = 0;
                return -6;
            }
            std::memcpy(out_result->beats, result.beats.data(), out_result->beat_count * sizeof(double));
        }

        out_result->bpm = result.bpm;
        out_result->duration = static_cast<double>(audioData.samples.size()) / audioData.sampleRate;

        return 0;
    } catch (const std::exception& e) {
        if (out_result && out_result->beats) {
            free(out_result->beats);
            out_result->beats = nullptr;
            out_result->beat_count = 0;
        }
        s_lastError = std::string("AudioFlux analysis exception: ") + e.what();
        return -4;
    } catch (...) {
        if (out_result && out_result->beats) {
            free(out_result->beats);
            out_result->beats = nullptr;
            out_result->beat_count = 0;
        }
        s_lastError = "AudioFlux analysis crashed with unknown exception";
        return -5;
    }
#else
    (void)audio_path;
    (void)out_result;
    (void)progress_cb;
    (void)user_data;
    s_lastError = "AudioFlux support not compiled in";
    return -1;
#endif
}

BEATSYNC_API int bs_audioflux_analyze_with_stems(const char* audio_path,
                                                  const char* stem_model_path,
                                                  bs_ai_result_t* out_result,
                                                  bs_ai_progress_cb progress_cb, void* user_data) {
#if defined(USE_AUDIOFLUX) && defined(USE_ONNX)
    if (!audio_path || !out_result) {
        s_lastError = "Invalid parameters: null audio_path or out_result";
        return -1;
    }

    // Initialize result
    out_result->beats = nullptr;
    out_result->beat_count = 0;
    out_result->downbeats = nullptr;
    out_result->downbeat_count = 0;
    out_result->bpm = 0.0;
    out_result->duration = 0.0;
    out_result->segments = nullptr;
    out_result->segment_count = 0;

    try {
        // Load audio file
        if (progress_cb) progress_cb(0.05f, "Loading audio...", "", user_data);

        BeatSync::AudioAnalyzer audioAnalyzer;
        auto audioData = audioAnalyzer.loadAudioFile(audio_path);
        if (audioData.samples.empty()) {
            s_lastError = "Failed to load audio file: " + std::string(audio_path);
            return -2;
        }

        std::vector<float> drumsAudio;
        int drumsSampleRate = audioData.sampleRate;

        // If stem model path provided, use stem separation
        if (stem_model_path && strlen(stem_model_path) > 0) {
            if (progress_cb) progress_cb(0.1f, "Separating stems (extracting drums)...", "", user_data);

            BeatSync::OnnxStemSeparator stemSeparator;
            BeatSync::StemSeparatorConfig stemConfig;
            stemConfig.useGPU = true;
            stemConfig.sampleRate = 44100;

            if (!stemSeparator.loadModel(stem_model_path, stemConfig)) {
                std::cerr << "[StemsFlux] Failed to load stem model, falling back to full mix" << std::endl;
                // Fall back to full mix
                drumsAudio = audioData.samples;
            } else {
                // Create progress wrapper for stem separation
                auto stemProgress = [progress_cb, user_data](float p, const std::string& msg) -> bool {
                    if (progress_cb) {
                        // Map stem progress (0-1) to overall progress (0.1-0.5)
                        float overallProgress = 0.1f + p * 0.4f;
                        return progress_cb(overallProgress, msg.c_str(), "", user_data) != 0;
                    }
                    return true;
                };

                // Separate and extract drums stem
                auto stemResult = stemSeparator.separateMono(audioData.samples, audioData.sampleRate, stemProgress);
                drumsAudio = stemResult.getMonoStem(BeatSync::StemType::Drums);
                drumsSampleRate = stemResult.sampleRate;

                std::cerr << "[StemsFlux] Extracted drums stem: " << drumsAudio.size() << " samples at " << drumsSampleRate << " Hz" << std::endl;
            }
        } else {
            // No stem model, use full mix
            drumsAudio = audioData.samples;
        }

        if (drumsAudio.empty()) {
            s_lastError = "Failed to extract drums stem";
            return -3;
        }

        // Now run AudioFlux on the drums stem
        if (progress_cb) progress_cb(0.55f, "Running AudioFlux beat detection on drums...", "", user_data);

        AudioFluxBeatDetector detector;

        // Create progress wrapper for AudioFlux (maps 0-1 to 0.55-1.0)
        auto fluxProgress = [progress_cb, user_data](float progress, const char* stage) -> bool {
            if (progress_cb) {
                float overallProgress = 0.55f + progress * 0.45f;
                return progress_cb(overallProgress, stage, "", user_data) != 0;
            }
            return true;
        };

        // Run detection on drums stem
        auto result = detector.detect(drumsAudio, drumsSampleRate, fluxProgress);

        if (!result.error.empty()) {
            s_lastError = result.error;
            return -4;
        }

        // Copy results
        out_result->beat_count = result.beats.size();
        if (out_result->beat_count > 0) {
            out_result->beats = static_cast<double*>(malloc(out_result->beat_count * sizeof(double)));
            if (!out_result->beats) {
                s_lastError = "Failed to allocate memory for beats array";
                out_result->beat_count = 0;
                return -6;
            }
            std::memcpy(out_result->beats, result.beats.data(), out_result->beat_count * sizeof(double));
        }

        out_result->bpm = result.bpm;
        out_result->duration = static_cast<double>(audioData.samples.size()) / audioData.sampleRate;

        std::cerr << "[StemsFlux] Analysis complete: " << result.beats.size() << " beats, " << result.bpm << " BPM" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        s_lastError = std::string("Stems+AudioFlux analysis exception: ") + e.what();
        return -5;
    } catch (...) {
        s_lastError = "Stems+AudioFlux analysis crashed with unknown exception";
        return -6;
    }
#else
    (void)audio_path;
    (void)stem_model_path;
    (void)out_result;
    (void)progress_cb;
    (void)user_data;
    s_lastError = "AudioFlux or ONNX support not compiled in";
    return -1;
#endif
}

} // extern "C"
