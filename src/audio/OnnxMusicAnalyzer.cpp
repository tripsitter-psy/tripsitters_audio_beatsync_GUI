#include "OnnxBeatDetector.h"
// Ensure OnnxBeatDetector translation unit is linked
namespace {
    struct LinkOnnxBeatDetector {
        LinkOnnxBeatDetector() { BeatSync::ensureOnnxBeatDetectorIsLinked(); }
    } g_linkOnnxBeatDetector;
}
/**
 * @file OnnxMusicAnalyzer.cpp
 * @brief Unified music analysis pipeline combining stem separation and beat detection
 */

#include "OnnxMusicAnalyzer.h"
#include "tracing/Tracing.h"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace BeatSync {

struct OnnxMusicAnalyzer::Impl {
    MusicAnalyzerConfig config;
    std::string lastError;

    std::unique_ptr<OnnxStemSeparator> stemSeparator;
    std::unique_ptr<OnnxBeatDetector> beatDetector;

    bool stemSeparatorLoaded = false;
    bool beatDetectorLoaded = false;

    bool initialize(const MusicAnalyzerConfig& cfg) {
        config = cfg;

        // Load stem separator if path provided and enabled
        if (cfg.useStemSeparation && !cfg.stemModelPath.empty()) {
            stemSeparator = std::make_unique<OnnxStemSeparator>();
            if (stemSeparator->loadModel(cfg.stemModelPath, cfg.stemConfig)) {
                stemSeparatorLoaded = true;
            } else {
                lastError = "Failed to load stem separator: " + stemSeparator->getLastError();
                // Continue without stem separation
                stemSeparatorLoaded = false;
            }
        }

        // Load beat detector (required)
        if (!cfg.beatModelPath.empty()) {
            beatDetector = std::make_unique<OnnxBeatDetector>();
            if (beatDetector->loadModel(cfg.beatModelPath, cfg.beatConfig)) {
                beatDetectorLoaded = true;
            } else {
                lastError = "Failed to load beat detector: " + beatDetector->getLastError();
                return false;
            }
        } else {
            lastError = "Beat model path is required";
            return false;
        }

        return beatDetectorLoaded;
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
        result.duration = stereoSamples.size() / 2.0 / sampleRate;

        // Progress stages
        float stemProgress = 0.0f;
        float beatProgress = 0.0f;

        auto reportProgress = [&](const std::string& stage) {
            if (progress) {
                float total = useStemSep ? (stemProgress * 0.6f + beatProgress * 0.4f)
                                         : beatProgress;
                return progress(total, stage, "");
            }
            return true;
        };

        // Stage 1: Stem Separation (if enabled and loaded)
        std::vector<float> audioForBeatDetection;

        if (useStemSep && stemSeparatorLoaded && stemSeparator) {
            if (progress) progress(0.0f, "Stem Separation", "Separating audio into stems...");

            auto stemCallback = [&](float p, const std::string& msg) {
                stemProgress = p;
                if (progress) {
                    return progress(p * 0.6f, "Stem Separation", msg);
                }
                return true;
            };

            StemSeparationResult stemResult = stemSeparator->separate(stereoSamples, sampleRate, stemCallback);

            if (!stemResult.stems[0].empty()) {
                result.stemSeparationUsed = true;

                // Use drums stem for primary beat detection
                if (config.useDrumsForBeats) {
                    audioForBeatDetection = stemResult.getMonoStem(StemType::Drums);
                } else {
                    // Use original audio
                    audioForBeatDetection = stereoToMono(stereoSamples);
                }

                // Optionally run beat detection on each stem
                if (config.returnStems || config.combineAllStems) {
                    result.rawStemResult = std::move(stemResult);
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

        // Stage 2: Beat Detection
        if (progress) progress(0.6f, "Beat Detection", "Analyzing rhythm...");

        if (!beatDetectorLoaded || !beatDetector) {
            lastError = "Beat detector not loaded";
            return result;
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
        OnnxAnalysisResult beatResult = beatDetector->analyzeDetailed(
            audioForBeatDetection, sampleRate, beatCallback);

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
        if (config.combineAllStems && result.stemSeparationUsed && !result.rawStemResult.stems[0].empty()) {
            if (progress) progress(0.9f, "Per-Stem Analysis", "Analyzing individual stems...");

            // Run beat detection on each stem
            for (int s = 0; s < 4; ++s) {
                auto stemMono = result.rawStemResult.getMonoStem(static_cast<StemType>(s));
                if (!stemMono.empty()) {
                    auto stemBeats = beatDetector->analyzeDetailed(stemMono, result.rawStemResult.sampleRate, nullptr);

                    switch (static_cast<StemType>(s)) {
                        case StemType::Drums:
                            result.stemBeats.drums = stemBeats.beats;
                            break;
                        case StemType::Bass:
                            result.stemBeats.bass = stemBeats.beats;
                            break;
                        case StemType::Other:
                            result.stemBeats.other = stemBeats.beats;
                            break;
                        case StemType::Vocals:
                            result.stemBeats.vocals = stemBeats.beats;
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        if (progress) progress(1.0f, "Complete", "Analysis complete");

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
    oss << "  Combine all stems: " << (m_impl->config.combineAllStems ? "Yes" : "No") << "\n";

    return oss.str();
}

} // namespace BeatSync
