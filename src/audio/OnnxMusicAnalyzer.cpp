// #include "OnnxBeatDetector.h"
// NOTE: OnnxBeatDetector registration must be ensured by explicit call to BeatSync::ensureOnnxBeatDetectorIsLinked() from main() or your library init routine, or by using linker flags (e.g., --whole-archive).
/**
 * @file OnnxMusicAnalyzer.cpp
 * @brief Unified music analysis pipeline combining stem separation and beat detection
 */

#include "OnnxMusicAnalyzer.h"
#include "tracing/Tracing.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
#include <fstream>

#include "utils/DebugLogger.h"

namespace BeatSync {

// Debug logging helper - writes to file since Windows GUI apps don't show stderr
static void debugLog(const std::string& msg) {
    DebugLogger::getInstance().log(msg);
}

struct OnnxMusicAnalyzer::Impl {
    MusicAnalyzerConfig config;
    std::string lastError;

    std::unique_ptr<OnnxStemSeparator> stemSeparator;
    std::unique_ptr<OnnxBeatDetector> beatDetector;

    bool stemSeparatorLoaded = false;
    bool beatDetectorLoaded = false;

    bool initialize(const MusicAnalyzerConfig& cfg) {
        // Reset state
        stemSeparator = nullptr;
        stemSeparatorLoaded = false;
        beatDetector = nullptr;
        beatDetectorLoaded = false;
        lastError.clear();
        config = cfg;

        // Load stem separator if path provided and enabled
        std::string stemErrorMsg;
        if (cfg.useStemSeparation && !cfg.stemModelPath.empty()) {
            stemSeparator = std::make_unique<OnnxStemSeparator>();
            if (stemSeparator->loadModel(cfg.stemModelPath, cfg.stemConfig)) {
                stemSeparatorLoaded = true;
            } else {
                stemErrorMsg = "Failed to load stem separator: " + stemSeparator->getLastError();
                stemSeparator = nullptr;
                stemSeparatorLoaded = false;
                // Continue without stem separation
            }
        }

        // Load beat detector (required)
        if (!cfg.beatModelPath.empty()) {
            beatDetector = std::make_unique<OnnxBeatDetector>();
            if (beatDetector->loadModel(cfg.beatModelPath, cfg.beatConfig)) {
                beatDetectorLoaded = true;
            } else {
                lastError = "Failed to load beat detector: " + beatDetector->getLastError();
                beatDetector = nullptr;
                beatDetectorLoaded = false;
                return false;
            }
        } else {
            lastError = "Beat model path is required";
            beatDetector = nullptr;
            beatDetectorLoaded = false;
            return false;
        }

        // Only set lastError if both failed; otherwise clear it
        if (!beatDetectorLoaded) {
            // lastError already set above for beat detector
            return false;
        }
        lastError.clear();
        return true;
    }

    std::vector<float> monoToStereo(const std::vector<float>& mono) {
        std::vector<float> stereo(mono.size() * 2);
        for (size_t i = 0; i < mono.size(); ++i) {
            stereo[i * 2] = mono[i];
            stereo[i * 2 + 1] = mono[i];
        }
        return stereo;
    }

    std::vector<float> stereoToMono(const std::vector<float>& stereo) {
        size_t numSamples = stereo.size() / 2;
        std::vector<float> mono(numSamples);
        for (size_t i = 0; i < numSamples; ++i) {
            mono[i] = (stereo[i * 2] + stereo[i * 2 + 1]) * 0.5f;
        }
        return mono;
    }

    MusicAnalysisResult runPipeline(const std::vector<float>& stereoSamples, int sampleRate,
                                    bool useStemSep, MusicAnalysisProgress progress) {
        MusicAnalysisResult result;
        result.sampleRate = sampleRate;
        // Explicit cast to double to preserve precision for very large buffers
        result.duration = static_cast<double>(stereoSamples.size()) / 2.0 / static_cast<double>(sampleRate);

        // Progress stages
        float stemProgress = 0.0f;
        float beatProgress = 0.0f;

        {
            std::ostringstream oss;
            oss << "[BeatSync] MusicAnalyzer pipeline: stereo samples=" << stereoSamples.size()
                << " sampleRate=" << sampleRate << " duration=" << result.duration << "s"
                << " useStemSep=" << useStemSep << " stemSeparatorLoaded=" << stemSeparatorLoaded;
            debugLog(oss.str());
        }

        // Stage 1: Stem Separation (if enabled and loaded)
        std::vector<float> audioForBeatDetection;

        if (useStemSep && stemSeparatorLoaded && stemSeparator) {
            bool shouldContinue = true;
            if (progress) {
                shouldContinue = progress(0.0f, "Stem Separation", "Separating audio into stems...");
            }
            if (!shouldContinue) {
                lastError = "Stem separation cancelled by user.";
                return result;
            }

            bool cancelled = false;
            auto stemCallback = [&](float p, const std::string& msg) {
                stemProgress = p;
                if (progress) {
                    bool cont = progress(p * 0.6f, "Stem Separation", msg);
                    if (!cont) {
                        cancelled = true;
                        return false;
                    }
                }
                return true;
            };

            StemSeparationResult stemResult = stemSeparator->separate(stereoSamples, sampleRate, stemCallback);
            if (cancelled) {
                lastError = "Stem separation cancelled by user.";
                return result;
            }

            // Debug: log stem sizes
            {
                std::ostringstream oss;
                oss << "[BeatSync] Stem separation results:"
                    << " drums=" << stemResult.stems[0].size()
                    << " bass=" << stemResult.stems[1].size()
                    << " other=" << stemResult.stems[2].size()
                    << " vocals=" << stemResult.stems[3].size()
                    << " sampleRate=" << stemResult.sampleRate;
                debugLog(oss.str());
            }

            if (!stemResult.stems[0].empty()) {
                result.stemSeparationUsed = true;

                // Use drums stem for primary beat detection
                if (config.useDrumsForBeats) {
                    audioForBeatDetection = stemResult.getMonoStem(StemType::Drums);

                    // Debug: analyze drums audio statistics
                    float drumsMin = 1e9f, drumsMax = -1e9f, drumsSum = 0.0f;
                    size_t count = audioForBeatDetection.size();
                    for (float v : audioForBeatDetection) {
                        drumsMin = std::min(drumsMin, v);
                        drumsMax = std::max(drumsMax, v);
                        drumsSum += std::abs(v);
                    }
                    {
                        std::ostringstream oss;
                        oss << "[BeatSync] Drums mono audio: samples=" << count
                            << " min=" << drumsMin << " max=" << drumsMax;
                        if (count)
                            oss << " meanAbs=" << (drumsSum / count);
                        else
                            oss << " meanAbs=N/A";
                        debugLog(oss.str());
                    }
                } else {
                    // Use original audio
                    audioForBeatDetection = stereoToMono(stereoSamples);
                }

                // Optionally run beat detection on each stem
                debugLog("[BeatSync] About to check returnStems/analyzePerStemBeats flags");
                if (config.returnStems || config.analyzePerStemBeats) {
                    debugLog("[BeatSync] Moving stem result to output");
                    result.rawStemResult = std::move(stemResult);
                    debugLog("[BeatSync] Stem result moved successfully");
                }
            } else {
                // Stem separation failed, fall back to original audio
                lastError = "Stem separation returned empty result, using original audio";
                audioForBeatDetection = stereoToMono(stereoSamples);
            }
        } else {
            // No stem separation, use original audio
            audioForBeatDetection = stereoToMono(stereoSamples);
        }

        debugLog("[BeatSync] About to start beat detection stage");

        // Stage 2: Beat Detection
        if (progress) progress(0.6f, "Beat Detection", "Analyzing rhythm...");

        debugLog("[BeatSync] Beat detection progress callback returned");

        if (!beatDetectorLoaded || !beatDetector) {
            lastError = "Beat detector not loaded";
            debugLog("[BeatSync] ERROR: Beat detector not loaded!");
            return result;
        }

        {
            std::ostringstream oss;
            oss << "[BeatSync] Beat detector loaded, preparing to analyze "
                << audioForBeatDetection.size() << " samples at " << sampleRate << " Hz";
            debugLog(oss.str());
        }

        auto beatCallback = [&](float p, const std::string& msg) {
            beatProgress = p;
            if (progress) {
                float total = useStemSep ? 0.6f + p * 0.4f : p;
                return progress(total, "Beat Detection", msg);
            }
            return true;
        };

        // Run beat detection on prepared audio
        // Note: OnnxBeatDetector expects sample rate of its target (usually 22050)
        // but handles resampling internally
        debugLog("[BeatSync] Calling beatDetector->analyzeDetailed()...");
        OnnxAnalysisResult beatResult = beatDetector->analyzeDetailed(
            audioForBeatDetection, sampleRate, beatCallback);
        debugLog("[BeatSync] beatDetector->analyzeDetailed() returned");

        // Copy results
        result.beats = beatResult.beats;
        result.downbeats = beatResult.downbeats;
        result.bpm = beatResult.bpm;
        result.segments = beatResult.segments;
        result.modelUsed = config.beatModelPath;

        if (config.returnRawActivations) {
            result.rawBeatResult = std::move(beatResult);
        }

        // Stage 3: Optional per-stem beat analysis
        // Note: result.rawStemResult is always populated if analyzePerStemBeats is enabled (see above)
        if (config.analyzePerStemBeats && result.stemSeparationUsed && !result.rawStemResult.stems[0].empty()) {
            if (progress) progress(0.9f, "Per-Stem Analysis", "Analyzing individual stems...");

            // Run beat detection on each stem
            for (int s = 0; s < 4; ++s) {
                auto stemMono = result.rawStemResult.getMonoStem(static_cast<StemType>(s));
                if (!stemMono.empty()) {
                    auto stemBeats = beatDetector->analyzeDetailed(stemMono, result.rawStemResult.sampleRate, nullptr);

                    switch (static_cast<StemType>(s)) {
                        case StemType::Drums:
                            result.stemBeats.drums = std::move(stemBeats.beats);
                            break;
                        case StemType::Bass:
                            result.stemBeats.bass = std::move(stemBeats.beats);
                            break;
                        case StemType::Other:
                            result.stemBeats.other = std::move(stemBeats.beats);
                            break;
                        case StemType::Vocals:
                            result.stemBeats.vocals = std::move(stemBeats.beats);
                            break;
                        default:
                            break;
                    }

                    // Clear temporary result to release memory immediately (prevents GPU memory accumulation)
                    stemBeats.beatActivation.clear();
                    stemBeats.beatActivation.shrink_to_fit();
                    stemBeats.downbeatActivation.clear();
                    stemBeats.downbeatActivation.shrink_to_fit();
                }

                // stemMono goes out of scope here, no manual cleanup needed
            }
        }

        if (progress) progress(1.0f, "Complete", "Analysis complete");

        // Note: GPU memory is managed by the ONNX Runtime session and will be released
        // when the detector is destroyed. Calling releaseGPUMemory() here would set
        // loaded=false in the detector while beatDetectorLoaded remains true, breaking
        // subsequent analyze() calls. Let ONNX Runtime manage its memory lifecycle.

        return result;
    }
};

// Public interface

OnnxMusicAnalyzer::OnnxMusicAnalyzer()
    : m_impl(std::make_unique<Impl>()) {
}

OnnxMusicAnalyzer::~OnnxMusicAnalyzer() = default;

OnnxMusicAnalyzer::OnnxMusicAnalyzer(OnnxMusicAnalyzer&&) noexcept = default;
OnnxMusicAnalyzer& OnnxMusicAnalyzer::operator=(OnnxMusicAnalyzer&&) noexcept = default;

bool OnnxMusicAnalyzer::initialize(const MusicAnalyzerConfig& config) {
    TRACE_FUNC();
    return m_impl->initialize(config);
}

bool OnnxMusicAnalyzer::isReady() const {
    return m_impl->beatDetectorLoaded;
}

const MusicAnalyzerConfig& OnnxMusicAnalyzer::getConfig() const {
    return m_impl->config;
}

MusicAnalysisResult OnnxMusicAnalyzer::analyze(const std::vector<float>& samples, int sampleRate,
                                                MusicAnalysisProgress progress) {
    TRACE_FUNC();
    return m_impl->runPipeline(samples, sampleRate, m_impl->config.useStemSeparation, progress);
}

MusicAnalysisResult OnnxMusicAnalyzer::analyzeMono(const std::vector<float>& monoSamples, int sampleRate,
                                                    MusicAnalysisProgress progress) {
    TRACE_FUNC();
    std::vector<float> stereo = m_impl->monoToStereo(monoSamples);
    return analyze(stereo, sampleRate, progress);
}

BeatGrid OnnxMusicAnalyzer::getBeatGrid(const std::vector<float>& samples, int sampleRate) {
    TRACE_FUNC();
    MusicAnalysisResult result = analyze(samples, sampleRate);

    BeatGrid grid;
    grid.setBeats(result.beats);
    grid.setBPM(result.bpm);
    grid.setAudioDuration(result.duration);
    return grid;
}

MusicAnalysisResult OnnxMusicAnalyzer::analyzeQuick(const std::vector<float>& samples, int sampleRate,
                                                     MusicAnalysisProgress progress) {
    TRACE_FUNC();
    // Skip stem separation for faster analysis
    return m_impl->runPipeline(samples, sampleRate, false, progress);
}

std::string OnnxMusicAnalyzer::getLastError() const {
    return m_impl->lastError;
}

std::string OnnxMusicAnalyzer::getModelInfo() const {
    std::ostringstream oss;
    oss << "=== Music Analyzer Configuration ===\n";

    if (m_impl->beatDetector && m_impl->beatDetectorLoaded) {
        oss << "\n[Beat Detector]\n";
        oss << m_impl->beatDetector->getModelInfo();
    } else {
        oss << "\n[Beat Detector] Not loaded\n";
    }

    if (m_impl->stemSeparator && m_impl->stemSeparatorLoaded) {
        oss << "\n[Stem Separator]\n";
        oss << m_impl->stemSeparator->getModelInfo();
    } else {
        oss << "\n[Stem Separator] " << (m_impl->config.useStemSeparation ? "Not loaded" : "Disabled") << "\n";
    }

    oss << "\n[Pipeline]\n";
    oss << "  Stem separation: " << (m_impl->config.useStemSeparation ? "Enabled" : "Disabled") << "\n";
    oss << "  Use drums for beats: " << (m_impl->config.useDrumsForBeats ? "Yes" : "No") << "\n";
    oss << "  Per-stem beat analysis: " << (m_impl->config.analyzePerStemBeats ? "Yes" : "No") << "\n";

    return oss.str();
}

const OnnxBeatDetector* OnnxMusicAnalyzer::getBeatDetector() const {
    return m_impl->beatDetector.get();
}

bool OnnxMusicAnalyzer::isGPUEnabled() const {
    if (m_impl->beatDetector && m_impl->beatDetectorLoaded) {
        return m_impl->beatDetector->isGPUEnabled();
    }
    return false;
}

std::string OnnxMusicAnalyzer::getActiveProvider() const {
    if (m_impl->beatDetector && m_impl->beatDetectorLoaded) {
        return m_impl->beatDetector->getActiveProvider();
    }
    return "None";
}

} // namespace BeatSync
