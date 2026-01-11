#pragma once

#include <vector>
#include <cstddef>
#include <memory>

namespace BeatSync {

// Frequency band energy buckets extracted via FFT.
struct FrequencyBands {
    float bass;  // 20-200 Hz
    float mids;  // 200-2000 Hz
    float highs; // 2000-20000 Hz

    // Normalized values (0.0 - 1.0 range, useful for visualization)
    float bassNorm;
    float midsNorm;
    float highsNorm;
};

// Window function types for FFT preprocessing
enum class WindowType {
    Rectangular,  // No windowing (introduces spectral leakage)
    Hann,         // Good general-purpose window (default)
    Hamming,      // Similar to Hann, slightly different sidelobe characteristics
    Blackman      // Best sidelobe suppression, wider main lobe
};

// Configuration for FFT analysis
struct FFTConfig {
    int windowSize = 2048;           // FFT window size (must be power of 2 for efficiency)
    WindowType windowType = WindowType::Hann;
    bool normalize = true;           // Normalize output by window size
    float smoothingFactor = 0.3f;    // Temporal smoothing (0 = none, 1 = full smoothing)
};

// Robust FFT-based analyzer using KissFFT for frequency band energy extraction.
// Used to drive audio-reactive video effects.
class FFTAnalyzer {
public:
    FFTAnalyzer();
    explicit FFTAnalyzer(const FFTConfig& config);
    ~FFTAnalyzer();

    // Non-copyable (owns FFT state)
    FFTAnalyzer(const FFTAnalyzer&) = delete;
    FFTAnalyzer& operator=(const FFTAnalyzer&) = delete;

    // Movable
    FFTAnalyzer(FFTAnalyzer&& other) noexcept;
    FFTAnalyzer& operator=(FFTAnalyzer&& other) noexcept;

    // Analyze a window of mono float samples and return band energies.
    // Samples should be normalized to [-1.0, 1.0] range.
    FrequencyBands analyze(const std::vector<float>& samples, int sampleRate);

    // Analyze with explicit configuration override
    FrequencyBands analyze(const std::vector<float>& samples, int sampleRate, const FFTConfig& config);

    // Get/set configuration
    const FFTConfig& getConfig() const { return m_config; }
    void setConfig(const FFTConfig& config);

    // Reset internal state (clears smoothing history)
    void reset();

    // Get the magnitude spectrum from the last analysis (for visualization)
    const std::vector<float>& getMagnitudeSpectrum() const { return m_magnitudeSpectrum; }

    // Get frequency for a given bin index
    float binToFrequency(size_t bin, int sampleRate) const;

private:
    FFTConfig m_config;

    // Internal FFT state
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    // Precomputed window coefficients
    std::vector<float> m_windowCoeffs;

    // Magnitude spectrum from last analysis
    std::vector<float> m_magnitudeSpectrum;

    // Previous frame values for smoothing
    FrequencyBands m_prevBands;
    bool m_hasPrevBands;

    // Helper methods
    void initWindow();
    void applyWindow(std::vector<float>& samples);
    FrequencyBands computeBands(int sampleRate);
};

} // namespace BeatSync
