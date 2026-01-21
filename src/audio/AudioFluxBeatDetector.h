#pragma once

#include <vector>
#include <string>
#include <functional>

// AudioFlux-based beat detection using spectral flux onset detection
// This provides more reliable beat detection than neural network approaches
// for music with clear percussive elements

class AudioFluxBeatDetector {
public:
    using ProgressCallback = std::function<bool(float progress, const char* stage)>;

    struct Config {
        int sampleRate = 22050;      // Target sample rate for analysis
        int fftSize = 2048;          // FFT window size (radix2_exp = 11)
        int hopLength = 512;         // Hop length between frames
        float onsetThreshold = 0.3f; // Onset detection threshold
        float minBeatInterval = 0.25f; // Minimum time between beats (240 BPM max)
    };

    struct Result {
        std::vector<double> beats;           // Beat times in seconds
        std::vector<float> onsetEnvelope;    // Onset strength over time
        double bpm;                          // Estimated BPM
        double confidence;                   // Detection confidence (0-1)
        std::string error;                   // Error message if failed
    };

    AudioFluxBeatDetector();
    ~AudioFluxBeatDetector();

    // Main detection function
    Result detect(const std::vector<float>& samples, int sampleRate,
                  ProgressCallback progress = nullptr);

    // Configure detection parameters
    void setConfig(const Config& config) { m_config = config; }
    const Config& getConfig() const { return m_config; }

    // Check if AudioFlux is available
    static bool isAvailable();

private:
    Config m_config;

    // Internal methods
    std::vector<float> computeSTFT(const std::vector<float>& samples, int& numFrames);
    std::vector<float> computeOnsetEnvelope(const float* stftReal, const float* stftImag,
                                             int numFrames, int numBins, int fftStride);
    std::vector<double> pickPeaks(const std::vector<float>& envelope, float threshold);
    std::vector<double> fillBeatGaps(const std::vector<double>& beats, double duration);
    double estimateBPM(const std::vector<double>& beats, double duration);
};
