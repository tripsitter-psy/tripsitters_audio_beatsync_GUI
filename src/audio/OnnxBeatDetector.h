#pragma once

#include "BeatGrid.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>


namespace BeatSync {

/**
 * @brief Model types supported by OnnxBeatDetector
 */
enum class OnnxModelType {
    BeatNet,      ///< BeatNet CRNN - real-time beat/downbeat tracking
    AllInOne,     ///< All-In-One - full structure analysis (beat, downbeat, segments, tempo)
    TCN,          ///< Temporal Convolutional Network - lightweight beat tracking
    Custom        ///< Custom model with user-defined input/output names
};

/**
 * @brief Configuration for ONNX model inference
 */
struct OnnxConfig {
    OnnxModelType modelType = OnnxModelType::BeatNet;

    // Audio preprocessing parameters
    int sampleRate = 22050;           ///< Target sample rate for model input
    int nMels = 81;                   ///< Number of mel bands (81 for BeatNet, 128 for AllInOne)
    int hopLength = 441;              ///< Hop length in samples (~20ms at 22050 Hz)
    int windowLength = 2048;          ///< FFT window length
    float fmin = 30.0f;               ///< Minimum frequency for mel filterbank
    float fmax = 11000.0f;            ///< Maximum frequency for mel filterbank

    // Inference parameters
    int batchSize = 1;                ///< Batch size for inference
    bool useGPU = true;               ///< Use GPU acceleration if available (CUDA/DirectML)
    int gpuDeviceId = 0;              ///< GPU device ID
    int numThreads = 0;               ///< Number of threads (0 = auto)

    // Post-processing parameters
    float beatThreshold = 0.5f;       ///< Threshold for beat activation
    float downbeatThreshold = 0.5f;   ///< Threshold for downbeat activation
    float minBeatInterval = 0.2f;     ///< Minimum time between beats (seconds)

    // For AllInOne model
    bool enableSegments = true;       ///< Enable segment boundary detection
    float segmentThreshold = 0.5f;    ///< Threshold for segment boundaries
};

/**
 * @brief Segment information from All-In-One model
 */
struct MusicSegment {
    double startTime = 0.0;                 ///< Segment start time in seconds
    double endTime = 0.0;                   ///< Segment end time in seconds
    std::string label = "";                ///< Segment label (intro, verse, chorus, etc.)
    float confidence = 0.0f;                ///< Confidence score [0, 1]
};

/**
 * @brief Full analysis result from ONNX model
 */
struct OnnxAnalysisResult {
    std::vector<double> beats;        ///< Beat timestamps in seconds
    std::vector<double> downbeats;    ///< Downbeat timestamps in seconds
    double bpm = 0.0;                 ///< Estimated tempo in BPM
    std::vector<MusicSegment> segments; ///< Music structure segments (AllInOne only)

    // Raw activations (useful for visualization or custom post-processing)
    std::vector<float> beatActivation;
    std::vector<float> downbeatActivation;
    std::vector<float> segmentActivation;
};

/**
 * @brief Progress callback for long-running analysis
 * @param progress Progress value from 0.0 to 1.0
 * @param message Status message
 * @return true to continue, false to cancel
 */
using ProgressCallback = std::function<bool(float progress, const std::string& message)>;

/**
 * @brief ONNX-based neural network beat detector
 *
 * Provides native C++ inference for beat detection without Python dependencies.
 * Supports multiple model architectures (BeatNet, All-In-One, TCN).
 *
 * Usage:
 * @code
 *   OnnxBeatDetector detector;
 *   if (detector.loadModel("models/beatnet.onnx")) {
 *       BeatGrid grid = detector.analyze(samples, sampleRate);
 *       // or for full analysis:
 *       OnnxAnalysisResult result = detector.analyzeDetailed(samples, sampleRate);
 *   }
 * @endcode
 */
class OnnxBeatDetector {
public:
    OnnxBeatDetector();
    ~OnnxBeatDetector();

    // Prevent copying (ONNX session is not copyable)
    OnnxBeatDetector(const OnnxBeatDetector&) = delete;
    OnnxBeatDetector& operator=(const OnnxBeatDetector&) = delete;

    // Allow moving
    OnnxBeatDetector(OnnxBeatDetector&&) noexcept;
    OnnxBeatDetector& operator=(OnnxBeatDetector&&) noexcept;

    /**
     * @brief Load an ONNX model from file
     * @param modelPath Path to .onnx model file
     * @param config Optional configuration (auto-detected from model if not specified)
     * @return true if model loaded successfully
     */
    bool loadModel(const std::string& modelPath, const OnnxConfig& config = OnnxConfig());

    /**
     * @brief Check if a model is loaded and ready
     */
    bool isLoaded() const { return m_impl != nullptr && isLoadedImpl(); }

    OnnxModelType getModelType() const {
        if (!m_impl) return OnnxModelType();
        return getModelTypeImpl();
    }

    const OnnxConfig& getConfig() const {
        static OnnxConfig defaultConfig;
        if (!m_impl) return defaultConfig;
        return getConfigImpl();
    }

    void setConfig(const OnnxConfig& config) {
        if (m_impl) setConfigImpl(config);
    }

    BeatGrid analyze(const std::vector<float>& samples, int sampleRate, ProgressCallback progress = nullptr) {
        if (!m_impl) return BeatGrid();
        return analyzeImpl(samples, sampleRate, progress);
    }

    OnnxAnalysisResult analyzeDetailed(const std::vector<float>& samples, int sampleRate, ProgressCallback progress = nullptr) {
        if (!m_impl) return OnnxAnalysisResult();
        return analyzeDetailedImpl(samples, sampleRate, progress);
    }

    std::vector<double> processChunk(const std::vector<float>& chunk) {
        if (!m_impl) return {};
        return processChunkImpl(chunk);
    }

    void reset() {
        if (m_impl) resetImpl();
    }

    std::string getLastError() const {
        if (!m_impl) return "OnnxBeatDetector not loaded";
        return getLastErrorImpl();
    }

    std::string getModelInfo() const {
        if (!m_impl) return "";
        return getModelInfoImpl();
    }

    /**
     * @brief Check if ONNX Runtime is available
     */
    static bool isOnnxRuntimeAvailable();

    /**
     * @brief Get available execution providers (CPU, CUDA, DirectML, etc.)
     */
    static std::vector<std::string> getAvailableProviders();

    // Private implementation methods
private:
    bool isLoadedImpl() const;
    OnnxModelType getModelTypeImpl() const;
    const OnnxConfig& getConfigImpl() const;
    void setConfigImpl(const OnnxConfig& config);
    BeatGrid analyzeImpl(const std::vector<float>& samples, int sampleRate, ProgressCallback progress);
    OnnxAnalysisResult analyzeDetailedImpl(const std::vector<float>& samples, int sampleRate, ProgressCallback progress);
    std::vector<double> processChunkImpl(const std::vector<float>& chunk);
    void resetImpl();
    std::string getLastErrorImpl() const;
    std::string getModelInfoImpl() const;

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Utility class for mel spectrogram computation
 *
 * Provides efficient mel spectrogram extraction optimized for beat detection models.
 * Uses the same parameters as librosa for compatibility with Python-trained models.
 */
class MelSpectrogramExtractor {
public:
    MelSpectrogramExtractor(int sampleRate = 22050, int nMels = 81,
                            int nFft = 2048, int hopLength = 441,
                            float fmin = 30.0f, float fmax = 11000.0f);
    ~MelSpectrogramExtractor();

    /**
     * @brief Extract mel spectrogram from audio
     * @param samples Mono audio samples
     * @return Mel spectrogram as flat vector (row-major: n_mels x n_frames)
     */
    std::vector<float> extract(const std::vector<float>& samples);

    /**
     * @brief Get the number of frames for given sample count
     */
    int getNumFrames(int numSamples) const;

    /**
     * @brief Get number of mel bands
     */
    int getNumMels() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// Ensure linker includes this translation unit
void ensureOnnxBeatDetectorIsLinked();

} // namespace BeatSync
