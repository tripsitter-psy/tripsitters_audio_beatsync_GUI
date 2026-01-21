#pragma once

#include "BeatGrid.h"
#include "OnnxBeatDetector.h"
#include "OnnxStemSeparator.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace BeatSync {

/**
 * @brief Configuration for the unified music analyzer
 */
struct MusicAnalyzerConfig {
    // Model paths
    std::string beatModelPath;         ///< Path to beat detection model (BeatNet/AllInOne)
    std::string stemModelPath;         ///< Path to stem separation model (Demucs)

    // Pipeline options
    bool useStemSeparation = true;     ///< Enable stem separation before beat detection
    bool useDrumsForBeats = true;      ///< Use drums stem for beat detection (most accurate)
    bool analyzePerStemBeats = false;  ///< Perform beat detection separately for each stem

    // Beat detection config
    OnnxConfig beatConfig;

    // Stem separation config
    StemSeparatorConfig stemConfig;

    // Output options
    bool returnRawActivations = false; ///< Include raw neural network activations in result
    bool returnStems = false;          ///< Include separated stems in result
};

/**
 * @brief Complete music analysis result
 */
struct MusicAnalysisResult {
    // Beat/rhythm analysis
    std::vector<double> beats;         ///< Beat timestamps in seconds
    std::vector<double> downbeats;     ///< Downbeat timestamps in seconds
    double bpm = 0.0;                  ///< Estimated tempo

    // Structure analysis (if using AllInOne model)
    std::vector<MusicSegment> segments;

    // Optional: per-stem beat analysis
    struct StemBeats {
        std::vector<double> drums;
        std::vector<double> bass;
        std::vector<double> other;
        std::vector<double> vocals;
    } stemBeats;

    // Optional: raw data
    OnnxAnalysisResult rawBeatResult;
    StemSeparationResult rawStemResult;

    // Metadata
    double duration = 0.0;
    int sampleRate = 0;
    bool stemSeparationUsed = false;
    std::string modelUsed;
};

/**
 * @brief Progress callback for music analysis
 */
using MusicAnalysisProgress = std::function<bool(float progress, const std::string& stage, const std::string& message)>;

/**
 * @brief Unified ONNX-based music analyzer
 *
 * Combines stem separation and beat detection into a single pipeline:
 * 1. Load audio
 * 2. Separate into stems (drums, bass, other, vocals) using Demucs
 * 3. Run beat detection on drums stem (or combined)
 * 4. Return comprehensive analysis results
 *
 * This approach significantly improves beat detection accuracy, especially for
 * complex music with dense instrumentation.
 *
 * Usage:
 * @code
 *   OnnxMusicAnalyzer analyzer;
 *
 *   MusicAnalyzerConfig config;
 *   config.beatModelPath = "models/beatnet.onnx";
 *   config.stemModelPath = "models/htdemucs.onnx";
 *
 *   if (analyzer.initialize(config)) {
 *       MusicAnalysisResult result = analyzer.analyze(samples, sampleRate);
 *       // Use result.beats, result.bpm, result.segments...
 *   }
 * @endcode
 */
class OnnxMusicAnalyzer {
public:
    OnnxMusicAnalyzer();
    ~OnnxMusicAnalyzer();

    // Non-copyable
    OnnxMusicAnalyzer(const OnnxMusicAnalyzer&) = delete;
    OnnxMusicAnalyzer& operator=(const OnnxMusicAnalyzer&) = delete;

    // Movable
    OnnxMusicAnalyzer(OnnxMusicAnalyzer&&) noexcept;
    OnnxMusicAnalyzer& operator=(OnnxMusicAnalyzer&&) noexcept;

    /**
     * @brief Initialize the analyzer with models
     * @param config Configuration including model paths
     * @return true if initialization successful
     */
    bool initialize(const MusicAnalyzerConfig& config);

    /**
     * @brief Check if analyzer is ready
     */
    bool isReady() const;

    /**
     * @brief Get current configuration
     */
    const MusicAnalyzerConfig& getConfig() const;

    /**
     * @brief Analyze audio (stereo interleaved)
     * @param samples Stereo audio samples (interleaved L/R)
     * @param sampleRate Sample rate
     * @param progress Optional progress callback
     * @return Complete analysis result
     */
    MusicAnalysisResult analyze(const std::vector<float>& samples, int sampleRate,
                                MusicAnalysisProgress progress = nullptr);

    /**
     * @brief Analyze mono audio
     * @param monoSamples Mono audio samples
     * @param sampleRate Sample rate
     * @param progress Optional progress callback
     * @return Complete analysis result
     */
    MusicAnalysisResult analyzeMono(const std::vector<float>& monoSamples, int sampleRate,
                                    MusicAnalysisProgress progress = nullptr);

    /**
     * @brief Get a BeatGrid from analysis (convenience method)
     * @param samples Audio samples
     * @param sampleRate Sample rate
     * @return BeatGrid with detected beats and BPM
     */
    BeatGrid getBeatGrid(const std::vector<float>& samples, int sampleRate);

    /**
     * @brief Get last error message
     */
    std::string getLastError() const;

    /**
     * @brief Get info about loaded models
     */
    std::string getModelInfo() const;

    /**
     * @brief Quick analysis without stem separation
     *
     * Faster but potentially less accurate for complex music.
     */
    MusicAnalysisResult analyzeQuick(const std::vector<float>& samples, int sampleRate,
                                     MusicAnalysisProgress progress = nullptr);

    /**
     * @brief Get the underlying beat detector (for GPU status queries)
     */
    const OnnxBeatDetector* getBeatDetector() const;

    /**
     * @brief Check if GPU acceleration is enabled
     */
    bool isGPUEnabled() const;

    /**
     * @brief Get the active execution provider name
     */
    std::string getActiveProvider() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace BeatSync
