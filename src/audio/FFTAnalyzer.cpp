#include "FFTAnalyzer.h"
#include "kiss_fft.h"
#include <cmath>
#include <algorithm>

namespace BeatSync {

// Internal implementation holding FFT buffers
struct FFTAnalyzer::Impl {
    std::vector<kiss_fft_cpx> fftOutput;
    int currentWindowSize = 0;
};

FFTAnalyzer::FFTAnalyzer()
    : m_impl(std::make_unique<Impl>())
    , m_hasPrevBands(false)
{
    m_prevBands = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    initWindow();
}

FFTAnalyzer::FFTAnalyzer(const FFTConfig& config)
    : m_config(config)
    , m_impl(std::make_unique<Impl>())
    , m_hasPrevBands(false)
{
    m_prevBands = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    initWindow();
}

FFTAnalyzer::~FFTAnalyzer() = default;

FFTAnalyzer::FFTAnalyzer(FFTAnalyzer&& other) noexcept
    : m_config(other.m_config)
    , m_impl(std::move(other.m_impl))
    , m_windowCoeffs(std::move(other.m_windowCoeffs))
    , m_magnitudeSpectrum(std::move(other.m_magnitudeSpectrum))
    , m_prevBands(other.m_prevBands)
    , m_hasPrevBands(other.m_hasPrevBands)
{
}

FFTAnalyzer& FFTAnalyzer::operator=(FFTAnalyzer&& other) noexcept {
    if (this != &other) {
        m_config = other.m_config;
        m_impl = std::move(other.m_impl);
        m_windowCoeffs = std::move(other.m_windowCoeffs);
        m_magnitudeSpectrum = std::move(other.m_magnitudeSpectrum);
        m_prevBands = other.m_prevBands;
        m_hasPrevBands = other.m_hasPrevBands;
    }
    return *this;
}

void FFTAnalyzer::setConfig(const FFTConfig& config) {
    bool needReinit = (config.windowSize != m_config.windowSize ||
                       config.windowType != m_config.windowType);
    m_config = config;
    if (needReinit) {
        initWindow();
    }
}

void FFTAnalyzer::reset() {
    m_hasPrevBands = false;
    m_prevBands = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    m_magnitudeSpectrum.clear();
}

void FFTAnalyzer::initWindow() {
    int N = m_config.windowSize;
    m_windowCoeffs.resize(N);

    const float PI = 3.14159265358979323846f;

    switch (m_config.windowType) {
        case WindowType::Rectangular:
            std::fill(m_windowCoeffs.begin(), m_windowCoeffs.end(), 1.0f);
            break;

        case WindowType::Hann:
            for (int i = 0; i < N; ++i) {
                m_windowCoeffs[i] = 0.5f * (1.0f - std::cos(2.0f * PI * i / (N - 1)));
            }
            break;

        case WindowType::Hamming:
            for (int i = 0; i < N; ++i) {
                m_windowCoeffs[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (N - 1));
            }
            break;

        case WindowType::Blackman:
            for (int i = 0; i < N; ++i) {
                m_windowCoeffs[i] = 0.42f
                    - 0.5f * std::cos(2.0f * PI * i / (N - 1))
                    + 0.08f * std::cos(4.0f * PI * i / (N - 1));
            }
            break;
    }
}

void FFTAnalyzer::applyWindow(std::vector<float>& samples) {
    size_t N = std::min(samples.size(), m_windowCoeffs.size());
    for (size_t i = 0; i < N; ++i) {
        samples[i] *= m_windowCoeffs[i];
    }
}

float FFTAnalyzer::binToFrequency(size_t bin, int sampleRate) const {
    return (static_cast<float>(bin) * sampleRate) / static_cast<float>(m_config.windowSize);
}

FrequencyBands FFTAnalyzer::computeBands(int sampleRate) {
    FrequencyBands bands{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (m_magnitudeSpectrum.empty()) return bands;

    size_t numBins = m_magnitudeSpectrum.size();
    float bass = 0.0f, mids = 0.0f, highs = 0.0f;
    int bassCount = 0, midsCount = 0, highsCount = 0;

    // Sum magnitudes into frequency bands
    for (size_t k = 0; k < numBins; ++k) {
        float freq = binToFrequency(k, sampleRate);
        float mag = m_magnitudeSpectrum[k];

        if (freq >= 20.0f && freq < 200.0f) {
            bass += mag;
            bassCount++;
        } else if (freq >= 200.0f && freq < 2000.0f) {
            mids += mag;
            midsCount++;
        } else if (freq >= 2000.0f && freq <= 20000.0f) {
            highs += mag;
            highsCount++;
        }
    }

    // Average by number of bins in each band (not total bins)
    bands.bass = bassCount > 0 ? bass / bassCount : 0.0f;
    bands.mids = midsCount > 0 ? mids / midsCount : 0.0f;
    bands.highs = highsCount > 0 ? highs / highsCount : 0.0f;

    // Normalize to 0-1 range based on typical audio levels
    // These scaling factors are empirically determined for typical music
    float maxMag = std::max({bands.bass, bands.mids, bands.highs, 0.001f});
    bands.bassNorm = std::min(1.0f, bands.bass / maxMag);
    bands.midsNorm = std::min(1.0f, bands.mids / maxMag);
    bands.highsNorm = std::min(1.0f, bands.highs / maxMag);

    return bands;
}

FrequencyBands FFTAnalyzer::analyze(const std::vector<float>& samples, int sampleRate) {
    return analyze(samples, sampleRate, m_config);
}

FrequencyBands FFTAnalyzer::analyze(const std::vector<float>& samples, int sampleRate, const FFTConfig& config) {
    FrequencyBands bands{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (samples.empty() || sampleRate <= 0) return bands;

    int N = config.windowSize;

    // Prepare input buffer with windowing
    std::vector<float> windowed(N, 0.0f);
    size_t copyLen = std::min(samples.size(), static_cast<size_t>(N));
    std::copy(samples.begin(), samples.begin() + copyLen, windowed.begin());

    // Apply window function
    if (m_windowCoeffs.size() != static_cast<size_t>(N)) {
        // Regenerate window if size changed
        FFTConfig tempConfig = config;
        m_config = tempConfig;
        initWindow();
    }
    applyWindow(windowed);

    // Allocate FFT output buffer
    m_impl->fftOutput.resize(N);

    // Perform FFT using KissFFT
    kiss_fftr(windowed.data(), m_impl->fftOutput.data(), N);

    // Compute magnitude spectrum (only positive frequencies: N/2 + 1 bins)
    size_t numBins = N / 2 + 1;
    m_magnitudeSpectrum.resize(numBins);

    for (size_t k = 0; k < numBins; ++k) {
        float re = m_impl->fftOutput[k].r;
        float im = m_impl->fftOutput[k].i;
        float mag = std::sqrt(re * re + im * im);

        // Normalize if requested
        if (config.normalize) {
            mag /= static_cast<float>(N);
        }

        m_magnitudeSpectrum[k] = mag;
    }

    // Compute band energies
    bands = computeBands(sampleRate);

    // Apply temporal smoothing
    if (config.smoothingFactor > 0.0f && m_hasPrevBands) {
        float alpha = config.smoothingFactor;
        bands.bass = alpha * m_prevBands.bass + (1.0f - alpha) * bands.bass;
        bands.mids = alpha * m_prevBands.mids + (1.0f - alpha) * bands.mids;
        bands.highs = alpha * m_prevBands.highs + (1.0f - alpha) * bands.highs;

        // Recalculate normalized values after smoothing
        float maxMag = std::max({bands.bass, bands.mids, bands.highs, 0.001f});
        bands.bassNorm = std::min(1.0f, bands.bass / maxMag);
        bands.midsNorm = std::min(1.0f, bands.mids / maxMag);
        bands.highsNorm = std::min(1.0f, bands.highs / maxMag);
    }

    // Store for next frame smoothing
    m_prevBands = bands;
    m_hasPrevBands = true;

    return bands;
}

} // namespace BeatSync
