#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <array>

namespace BeatSync {

/**
 * @brief Stem types from source separation
 */
enum class StemType {
    Drums = 0,
    Bass = 1,
    Other = 2,
    Vocals = 3,
    NumStems = 4
};

/**
 * @brief Get stem name as string
 */
inline const char* getStemName(StemType stem) {
    switch (stem) {
        case StemType::Drums: return "drums";
        case StemType::Bass: return "bass";
        case StemType::Other: return "other";
        case StemType::Vocals: return "vocals";
        default: return "unknown";
    }
}

/**
 * @brief Configuration for stem separation
 */
struct StemSeparatorConfig {
    int sampleRate = 44100;           ///< Target sample rate (Demucs uses 44100)
    int segmentLength = 44100 * 10;   ///< Process audio in segments (10 seconds)
    int overlap = 44100;              ///< Overlap between segments for smooth transitions
    bool useGPU = false;              ///< Use GPU acceleration if available
    int gpuDeviceId = 0;              ///< GPU device ID
    int numThreads = 0;               ///< Number of threads (0 = auto)

    // Output options
    bool normalize = true;            ///< Normalize output stems
    float clipThreshold = 0.99f;      ///< Clip threshold for output
};

/**
 * @brief Result of stem separation
 */
struct StemSeparationResult {
    /// Separated stems: [drums, bass, other, vocals]
    /// Each stem is stereo: (2 channels, samples)
    std::array<std::vector<float>, 4> stems;

    /// Original audio sample rate
    int sampleRate = 44100;

    /// Duration in seconds
    double duration = 0.0;

    /// Get mono mix of a specific stem
    std::vector<float> getMonoStem(StemType stem) const {
        const auto& stereo = stems[static_cast<int>(stem)];
        size_t numSamples = stereo.size() / 2;
        std::vector<float> mono(numSamples);
        for (size_t i = 0; i < numSamples; ++i) {
            mono[i] = (stereo[i * 2] + stereo[i * 2 + 1]) * 0.5f;
        }
        return mono;
    }

    /// Get all stems mixed (should approximate original)
    std::vector<float> getMix() const {
        if (stems[0].empty()) return {};
        size_t numSamples = stems[0].size() / 2;
        std::vector<float> mix(numSamples * 2, 0.0f);
        for (int s = 0; s < 4; ++s) {
            for (size_t i = 0; i < stems[s].size(); ++i) {
                mix[i] += stems[s][i];
            }
        }
        return mix;
    }
};

/**
 * @brief Progress callback for stem separation
 * @param progress Progress value from 0.0 to 1.0
 * @param message Status message
 * @return true to continue, false to cancel
 */
using StemProgressCallback = std::function<bool(float progress, const std::string& message)>;

/**
 * @brief ONNX-based audio stem separator
 *
 * Separates audio into stems (drums, bass, other, vocals) using neural networks.
 * This preprocessing step can improve beat detection accuracy.
 *
 * Usage:
 * @code
 *   OnnxStemSeparator separator;
 *   if (separator.loadModel("models/htdemucs.onnx")) {
 *       StemSeparationResult result = separator.separate(samples, sampleRate);
 *       // Use result.stems[StemType::Drums] for beat detection
 *   }
 * @endcode
 */
class OnnxStemSeparator {
public:
    OnnxStemSeparator();
    ~OnnxStemSeparator();

    // Prevent copying
    OnnxStemSeparator(const OnnxStemSeparator&) = delete;
    OnnxStemSeparator& operator=(const OnnxStemSeparator&) = delete;

    // Allow moving
    OnnxStemSeparator(OnnxStemSeparator&&) noexcept;
    OnnxStemSeparator& operator=(OnnxStemSeparator&&) noexcept;

    /**
     * @brief Load an ONNX stem separation model
     * @param modelPath Path to .onnx model file (e.g., htdemucs.onnx)
     * @param config Optional configuration
     * @return true if model loaded successfully
     */
    bool loadModel(const std::string& modelPath, const StemSeparatorConfig& config = StemSeparatorConfig());

    /**
     * @brief Check if a model is loaded and ready
     */
    bool isLoaded() const;

    /**
     * @brief Get current configuration
     */
    const StemSeparatorConfig& getConfig() const;

    /**
     * @brief Update configuration
     */
    void setConfig(const StemSeparatorConfig& config);

    /**
     * @brief Separate audio into stems
     * @param samples Stereo audio samples (interleaved: L, R, L, R, ...)
     * @param sampleRate Sample rate of the input audio
     * @param progress Optional progress callback
     * @return Separation result with 4 stems
     *
     * Note: If input is mono, it will be converted to stereo.
     */
    StemSeparationResult separate(const std::vector<float>& samples, int sampleRate,
                                  StemProgressCallback progress = nullptr);

    /**
     * @brief Separate mono audio into stems (convenience method)
     * @param monoSamples Mono audio samples
     * @param sampleRate Sample rate of the input audio
     * @param progress Optional progress callback
     * @return Separation result with 4 stems
     */
    StemSeparationResult separateMono(const std::vector<float>& monoSamples, int sampleRate,
                                      StemProgressCallback progress = nullptr);

    /**
     * @brief Get a specific stem from stereo audio
     * @param samples Stereo audio samples
     * @param sampleRate Sample rate
     * @param stem Which stem to extract
     * @param progress Optional progress callback
     * @return Mono samples for the requested stem
     */
    std::vector<float> extractStem(const std::vector<float>& samples, int sampleRate,
                                   StemType stem, StemProgressCallback progress = nullptr);

    /**
     * @brief Get last error message
     */
    std::string getLastError() const;

    /**
     * @brief Get model information string
     */
    std::string getModelInfo() const;

    /**
     * @brief Check if ONNX Runtime is available
     */
    static bool isOnnxRuntimeAvailable();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// Ensure linker includes this translation unit
void ensureOnnxStemSeparatorIsLinked();

} // namespace BeatSync
