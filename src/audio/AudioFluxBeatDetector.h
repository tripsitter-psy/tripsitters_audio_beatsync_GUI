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
        int hopLength = 256;         // Hop length between frames (~12ms) - tuned for EDM precision
        float onsetThreshold = 0.2f; // Onset detection threshold - lower for prominent EDM kicks
        float minBeatInterval = 0.2f; // Minimum time between beats (300 BPM max)

        // Low-frequency focus for kick drum detection (EDM/psytrance)
        bool lowFreqFocus = true;    // Only analyze kick drum frequencies
        float lowFreqMin = 30.0f;    // Minimum frequency Hz (sub-bass)
        float lowFreqMax = 200.0f;   // Maximum frequency Hz (kick fundamental + harmonics)

        // Energy gating to filter out quiet sections
        float energyGate = 0.01f;    // RMS energy gate - frames below this are ignored
        bool fillGaps = false;       // DISABLED - trust the detection, don't interpolate missing beats
        bool extendToEdges = false;  // DISABLED - don't create beats at track edges
    };

    struct Result {
        std::vector<double> beats;           // Beat times in seconds
        std::vector<float> onsetEnvelope;    // Onset strength over time
        double bpm = 0.0;                    // Estimated BPM
        double confidence = 0.0;             // Detection confidence (0-1)
        std::string error;                   // Error message if failed
    };

    struct STFTResult {
        std::vector<float> real;
        std::vector<float> imag;
        int numFrames = 0;
        int numBins = 0;
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
    STFTResult computeSTFT(const std::vector<float>& samples, int& numFrames);
    std::vector<float> computeOnsetEnvelope(const float* stftReal, const float* stftImag, int numFrames, int numBins, int fftStride);
    std::vector<double> pickPeaks(const std::vector<float>& envelope, float threshold);
    std::vector<double> fillBeatGaps(const std::vector<double>& beats, double duration);
    double estimateBPM(const std::vector<double>& beats, double duration);
};
