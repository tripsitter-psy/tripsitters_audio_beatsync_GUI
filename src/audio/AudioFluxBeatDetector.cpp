#include "AudioFluxBeatDetector.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <sstream>
#include "../utils/DebugLogger.h"

// AudioFlux C API headers
extern "C" {
#include "flux_base.h"
#include "stft_algorithm.h"
#include "mir/onset_algorithm.h"
}

// Debug logging helper - writes to file since Windows GUI apps don't show stderr
static void debugLog(const std::string& msg) {
    BeatSync::DebugLogger::getInstance().log(msg);
}

// Helper: resample audio to target sample rate (band-limited, anti-aliased)
// NOTE: For production use, consider linking libsamplerate for higher quality resampling.
#include <stdexcept>
#ifdef HAVE_LIBSAMPLERATE
#include <samplerate.h>
#endif

static std::vector<float> resampleAudio(const std::vector<float>& input, int inputRate, int outputRate) {
    if (inputRate == outputRate) return input;
    if (input.empty()) return input;

#ifdef HAVE_LIBSAMPLERATE
    // High-quality resampling using libsamplerate (if linked):
    std::vector<float> output;
    SRC_DATA srcData;
    srcData.data_in = input.data();
    srcData.input_frames = static_cast<long>(input.size());

    // Safe ratio calculation with validation
    if (inputRate == 0) {
        throw std::runtime_error("resampleAudio: inputRate cannot be zero");
    }
    double ratio = static_cast<double>(outputRate) / inputRate;
    if (!std::isfinite(ratio) || ratio <= 0.0) {
        throw std::runtime_error("resampleAudio: invalid ratio");
    }

    // Safe output size calculation to prevent overflow
    double outputSizeD = static_cast<double>(input.size()) * ratio;
    constexpr size_t maxOutputSize = 500 * 1024 * 1024;  // 500MB limit
    if (outputSizeD > static_cast<double>(maxOutputSize)) {
        throw std::runtime_error("resampleAudio: output size would exceed memory limit");
    }
    size_t outputSize = static_cast<size_t>(std::ceil(outputSizeD));
    output.resize(outputSize);
    srcData.data_out = output.data();
    srcData.output_frames = static_cast<long>(outputSize);
    srcData.src_ratio = ratio;
    srcData.end_of_input = 1;
    int err = src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1);
    if (err != 0) throw std::runtime_error("libsamplerate error: " + std::string(src_strerror(err)));
    output.resize(srcData.output_frames_gen);
    return output;
#else
    // Fallback resampler with anti-aliasing filter for downsampling
    // Safe ratio calculation with validation
    if (inputRate == 0) {
        throw std::runtime_error("resampleAudio: inputRate cannot be zero");
    }
    double ratio = static_cast<double>(outputRate) / inputRate;
    if (!std::isfinite(ratio) || ratio <= 0.0) {
        throw std::runtime_error("resampleAudio: invalid ratio");
    }

    // Safe output size calculation to prevent overflow
    double outputSizeD = static_cast<double>(input.size()) * ratio;
    constexpr size_t maxOutputSize = 500 * 1024 * 1024;  // 500MB limit
    if (outputSizeD > static_cast<double>(maxOutputSize)) {
        throw std::runtime_error("resampleAudio: output size would exceed memory limit");
    }
    size_t outputSize = static_cast<size_t>(std::ceil(outputSizeD));

    // If we're downsampling, apply a FIR low-pass filter (Blackman-windowed sinc) to prevent aliasing
    if (outputRate < inputRate) {
        // Design parameters for anti-aliasing filter
        const int filterLen = 31; // odd-length FIR (tradeoff: quality vs speed)
        const int M = filterLen - 1;
        const double cutoff = 0.5 * static_cast<double>(outputRate); // Hz (Nyquist of output)
        const double nyquist = 0.5 * static_cast<double>(inputRate);
        const double normCutoff = std::min(0.999, cutoff / nyquist); // normalized (0..1)

        std::vector<double> taps(filterLen);
        const double PI = 3.14159265358979323846;
        for (int n = 0; n < filterLen; ++n) {
            int k = n - M / 2;
            if (k == 0) {
                taps[n] = normCutoff;
            } else {
                double x = PI * k;
                taps[n] = std::sin(PI * normCutoff * k) / x;
            }
            // Blackman window for better stopband attenuation
            double w = 0.42 - 0.5 * std::cos(2.0 * PI * n / M) + 0.08 * std::cos(4.0 * PI * n / M);
            taps[n] *= w;
        }
        // Normalize filter taps
        double sum = 0.0;
        for (double v : taps) sum += v;
        if (sum != 0.0) {
            for (double& v : taps) v /= sum;
        }

        // Convolve input with FIR filter (zero-pad edges)
        std::vector<float> filtered(input.size(), 0.0f);
        size_t N = input.size();
        for (size_t i = 0; i < N; ++i) {
            double acc = 0.0;
            for (int t = 0; t < filterLen; ++t) {
                int idx = static_cast<int>(i) + (t - M / 2);
                if (idx >= 0 && static_cast<size_t>(idx) < N) {
                    acc += taps[t] * static_cast<double>(input[idx]);
                }
            }
            filtered[i] = static_cast<float>(acc);
        }

        // Resample filtered signal with linear interpolation
        std::vector<float> output(outputSize);
        for (size_t i = 0; i < outputSize; ++i) {
            double srcPos = i / ratio;
            size_t srcIdx = static_cast<size_t>(srcPos);
            double frac = srcPos - srcIdx;
            if (srcIdx + 1 < filtered.size()) {
                output[i] = static_cast<float>(filtered[srcIdx] * (1.0 - frac) + filtered[srcIdx + 1] * frac);
            } else if (srcIdx < filtered.size()) {
                output[i] = filtered[srcIdx];
            } else {
                output[i] = 0.0f;
            }
        }
        return output;
    }

    // Upsampling case: simple linear interpolation (no anti-aliasing needed)
    std::vector<float> output(outputSize);
    for (size_t i = 0; i < outputSize; ++i) {
        double srcPos = i / ratio;
        size_t srcIdx = static_cast<size_t>(srcPos);
        double frac = srcPos - srcIdx;
        if (srcIdx + 1 < input.size()) {
            output[i] = static_cast<float>(input[srcIdx] * (1.0 - frac) + input[srcIdx + 1] * frac);
        } else if (srcIdx < input.size()) {
            output[i] = input[srcIdx];
        } else {
            output[i] = 0.0f;
        }
    }
    return output;
#endif
}

// Check if AudioFlux STFT is available and working
static bool isAudioFluxAvailable() {
    STFTObj stftObj = nullptr;
    WindowType windowType = Window_Hann;
    int radix2Exp = 11; // 2048 FFT
    int slideLength = 512;
    int isContinue = 0;

    int result = stftObj_new(&stftObj, radix2Exp, &windowType, &slideLength, &isContinue);
    if (result == 0 && stftObj != nullptr) {
        stftObj_free(stftObj);
        return true;
    }
    return false;
}

// Constructor
AudioFluxBeatDetector::AudioFluxBeatDetector() = default;

// Destructor
AudioFluxBeatDetector::~AudioFluxBeatDetector() = default;

// Static method to check if AudioFlux is available
bool AudioFluxBeatDetector::isAvailable() {
    return isAudioFluxAvailable();
}

AudioFluxBeatDetector::Result AudioFluxBeatDetector::detect(
    const std::vector<float>& samples, int sampleRate, ProgressCallback progress) {

    Result result;
    result.bpm = 0.0;
    result.confidence = 0.0;

    {
        std::ostringstream oss;
        oss << "[AudioFlux] detect() called with " << samples.size() << " samples at " << sampleRate << " Hz";
        debugLog(oss.str());
    }

    if (samples.empty()) {
        result.error = "Empty audio data";
        return result;
    }

    // Resample to target sample rate if needed
    if (progress && !progress(0.05f, "Resampling audio...")) {
        result.error = "Analysis cancelled by user";
        return result;
    }

    // Check original samples before resampling
    {
        float origMin = 1e30f, origMax = -1e30f, origSum = 0.0f;
        int nonZero = 0;
        size_t checkCount = std::min(samples.size(), static_cast<size_t>(10000));
        for (size_t i = 0; i < checkCount; ++i) {
            float v = samples[i];
            if (v != 0.0f) nonZero++;
            origMin = std::min(origMin, v);
            origMax = std::max(origMax, v);
            origSum += std::abs(v);
        }
        std::ostringstream oss;
        oss << "[AudioFlux] ORIGINAL audio check (first " << checkCount << " samples): min=" << origMin
            << " max=" << origMax << " meanAbs=" << (origSum/checkCount) << " nonZero=" << nonZero;
        debugLog(oss.str());
    }

    {
        std::ostringstream oss;
        oss << "[AudioFlux] Resampling from " << sampleRate << " to " << m_config.sampleRate;
        debugLog(oss.str());
    }
    std::vector<float> resampled = resampleAudio(samples, sampleRate, m_config.sampleRate);
    double duration = static_cast<double>(resampled.size()) / m_config.sampleRate;
    {
        std::ostringstream oss;
        oss << "[AudioFlux] Resampled to " << resampled.size() << " samples, duration=" << duration << "s";
        debugLog(oss.str());
    }

    // Compute STFT
    if (progress && !progress(0.1f, "Computing spectrogram...")) {
        result.error = "Analysis cancelled by user";
        return result;
    }

    // Validate fftSize is a positive power of two before deriving radix/length
    if (m_config.fftSize <= 0 || (m_config.fftSize & (m_config.fftSize - 1)) != 0) {
        result.error = "Invalid fftSize in configuration: " + std::to_string(m_config.fftSize) + ". fftSize must be a power of two.";
        return result;
    }

    // Determine radix2_exp from fftSize (e.g., 2048 = 2^11)
    int radix2Exp = static_cast<int>(std::log2(m_config.fftSize));
    int fftLength = 1 << radix2Exp;
    // IMPORTANT: AudioFlux STFT outputs fftLength values per frame (full FFT), not fftLength/2+1
    // We only use the first half (positive frequencies) for onset detection
    int numBins = fftLength / 2 + 1;

    {
        std::ostringstream oss;
        oss << "[AudioFlux] STFT params: radix2Exp=" << radix2Exp << ", fftLength=" << fftLength << ", numBins=" << numBins;
        debugLog(oss.str());
    }

    // Create STFT object
    STFTObj stftObj = nullptr;
    WindowType windowType = Window_Hann;
    int slideLength = m_config.hopLength;
    int isContinue = 0;

    debugLog("[AudioFlux] Creating STFT object...");
    int stftResult = stftObj_new(&stftObj, radix2Exp, &windowType, &slideLength, &isContinue);
    {
        std::ostringstream oss;
        oss << "[AudioFlux] stftObj_new returned " << stftResult << ", stftObj=" << stftObj;
        debugLog(oss.str());
    }

    if (stftResult != 0 || !stftObj) {
        result.error = "Failed to create STFT object";
        return result;
    }

    // Calculate number of frames
    debugLog("[AudioFlux] Calculating time length...");
    int numFrames = stftObj_calTimeLength(stftObj, static_cast<int>(resampled.size()));
    {
        std::ostringstream oss;
        oss << "[AudioFlux] numFrames=" << numFrames;
        debugLog(oss.str());
    }

    if (numFrames <= 0) {
        stftObj_free(stftObj);
        result.error = "Invalid frame count";
        return result;
    }

    // Safety check: limit buffer size to prevent crashes on very long files
    // NOTE: AudioFlux outputs fftLength values per frame, not numBins!
    // For a 6-minute track at 22050Hz with hop=512, we need ~128MB
    // Allow up to 1GB for tracks up to ~50 minutes
    const size_t maxBufferSize = 1024 * 1024 * 1024; // 1GB max
    
    // Each of stftReal and stftImag needs numFrames * fftLength floats
    size_t bufferSize = 2 * static_cast<size_t>(numFrames) * static_cast<size_t>(fftLength) * sizeof(float);
    {
        std::ostringstream oss;
        oss << "[AudioFlux] Buffer size needed: " << bufferSize << " bytes (" << numFrames << " frames x " << fftLength << " bins x 2 buffers)";
        debugLog(oss.str());
    }

    if (bufferSize > maxBufferSize) {
        stftObj_free(stftObj);
        result.error = "Audio file too long for analysis";
        return result;
    }

    // Allocate STFT output buffers
    // CRITICAL: AudioFlux STFT writes fftLength values per frame (full complex FFT output)
    debugLog("[AudioFlux] Allocating STFT buffers...");
    std::vector<float> stftReal;
    std::vector<float> stftImag;
    try {
        stftReal.resize(numFrames * fftLength);
        stftImag.resize(numFrames * fftLength);
    } catch (const std::bad_alloc&) {
        stftObj_free(stftObj);
        result.error = "Failed to allocate memory for STFT";
        return result;
    }
    {
        std::ostringstream oss;
        oss << "[AudioFlux] Buffers allocated: " << stftReal.size() << " floats each (for " << numFrames << " frames)";
        debugLog(oss.str());
    }

    // Compute STFT
    // NOTE: AudioFlux stft outputs mRealArr and mImageArr as flat arrays of size numFrames * numBins
    // where each frame has numBins complex values
    {
        std::ostringstream oss;
        oss << "[AudioFlux] Computing STFT (input size=" << resampled.size() << ")...";
        debugLog(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "[AudioFlux] Output buffer sizes: real=" << stftReal.size() << ", imag=" << stftImag.size();
        debugLog(oss.str());
    }

    // Check input audio data
    {
        float inputMin = 1e30f, inputMax = -1e30f, inputSum = 0.0f;
        int nonZero = 0;
        for (size_t i = 0; i < std::min(resampled.size(), static_cast<size_t>(10000)); ++i) {
            float v = resampled[i];
            if (v != 0.0f) nonZero++;
            inputMin = std::min(inputMin, v);
            inputMax = std::max(inputMax, v);
            inputSum += std::abs(v);
        }
        std::ostringstream oss;
        oss << "[AudioFlux] Input audio check (first 10000 samples): min=" << inputMin << " max=" << inputMax
            << " meanAbs=" << (inputSum/10000) << " nonZero=" << nonZero;
        debugLog(oss.str());
    }

    // Zero-initialize buffers to be safe
    std::fill(stftReal.begin(), stftReal.end(), 0.0f);
    std::fill(stftImag.begin(), stftImag.end(), 0.0f);

    stftObj_stft(stftObj, resampled.data(), static_cast<int>(resampled.size()),
                 stftReal.data(), stftImag.data());
    debugLog("[AudioFlux] STFT complete");

    // Debug: Check STFT buffer contents
    {
        float realMin = 1e30f, realMax = -1e30f, realSum = 0.0f;
        float imagMin = 1e30f, imagMax = -1e30f, imagSum = 0.0f;
        int nonZeroReal = 0, nonZeroImag = 0;
        size_t checkCount = std::min(stftReal.size(), static_cast<size_t>(100000));
        for (size_t i = 0; i < checkCount; ++i) {
            float r = stftReal[i];
            float im = stftImag[i];
            if (r != 0.0f) nonZeroReal++;
            if (im != 0.0f) nonZeroImag++;
            realMin = std::min(realMin, r);
            realMax = std::max(realMax, r);
            realSum += std::abs(r);
            imagMin = std::min(imagMin, im);
            imagMax = std::max(imagMax, im);
            imagSum += std::abs(im);
        }
        std::ostringstream oss;
        oss << "[AudioFlux] STFT buffer check (first " << checkCount << " values):"
            << " real: min=" << realMin << " max=" << realMax << " meanAbs=" << (realSum/checkCount) << " nonZero=" << nonZeroReal
            << " | imag: min=" << imagMin << " max=" << imagMax << " meanAbs=" << (imagSum/checkCount) << " nonZero=" << nonZeroImag;
        debugLog(oss.str());
    }

    stftObj_free(stftObj);
    debugLog("[AudioFlux] STFT object freed");

    if (progress && !progress(0.3f, "Computing onset envelope...")) {
        result.error = "Analysis cancelled by user";
        return result;
    }

    // Compute onset envelope using spectral flux
    // Note: stft buffer has fftLength stride, but we only use first numBins (positive frequencies)
    result.onsetEnvelope = computeOnsetEnvelope(stftReal.data(), stftImag.data(), numFrames, numBins, fftLength);

    // Safety check
    if (result.onsetEnvelope.empty()) {
        result.error = "Failed to compute onset envelope";
        return result;
    }

    // Log envelope statistics for debugging
    {
        float envMin = *std::min_element(result.onsetEnvelope.begin(), result.onsetEnvelope.end());
        float envMax = *std::max_element(result.onsetEnvelope.begin(), result.onsetEnvelope.end());
        float envSum = std::accumulate(result.onsetEnvelope.begin(), result.onsetEnvelope.end(), 0.0f);
        float envMean = envSum / result.onsetEnvelope.size();
        std::ostringstream oss;
        oss << "[AudioFlux] Onset envelope: size=" << result.onsetEnvelope.size()
            << ", min=" << envMin << ", max=" << envMax << ", mean=" << envMean;
        debugLog(oss.str());
    }

    if (progress && !progress(0.5f, "Detecting beats...")) {
        result.error = "Analysis cancelled by user";
        return result;
    }

    // Adaptive threshold based on envelope statistics - use lower percentile for better sensitivity

    std::vector<float> sortedEnv = result.onsetEnvelope;
    std::sort(sortedEnv.begin(), sortedEnv.end());

    float adaptiveThreshold = 0.0f;
    float maxEnv = sortedEnv.empty() ? 0.0f : sortedEnv.back();
    if (sortedEnv.size() < 4) {
        // Fallback: use mean or a fraction of max if not enough data for percentiles
        float meanEnv = sortedEnv.empty() ? 0.0f : std::accumulate(sortedEnv.begin(), sortedEnv.end(), 0.0f) / sortedEnv.size();
        adaptiveThreshold = std::max(meanEnv, maxEnv * 0.12f); // fallback: mean or 12% of max
    } else {
        size_t p75Idx = std::min(sortedEnv.size() - 1, static_cast<size_t>(sortedEnv.size() * 0.75));
        size_t p25Idx = std::min(sortedEnv.size() - 1, static_cast<size_t>(sortedEnv.size() * 0.25));
        float p75 = sortedEnv[p75Idx];
        float p25 = sortedEnv[p25Idx];
        float iqr = p75 - p25;  // Interquartile range
        adaptiveThreshold = p25 + 0.5f * iqr;
    }
    // Ensure threshold isn't too low (avoid noise) or too high (miss beats)
    adaptiveThreshold = std::max(adaptiveThreshold, maxEnv * 0.12f);  // At least 12% of max (raised from 8%)
    adaptiveThreshold = std::min(adaptiveThreshold, maxEnv * 0.30f);  // At most 30% of max (raised from 25%)

    // Use configured onsetThreshold as a minimum floor
    adaptiveThreshold = std::max(adaptiveThreshold, m_config.onsetThreshold);

    {
        std::ostringstream oss;
        oss << "[AudioFlux] Adaptive threshold: " << adaptiveThreshold << " (maxEnv=" << maxEnv << ", configThreshold=" << m_config.onsetThreshold << ")";
        debugLog(oss.str());
    }

    // Pick peaks from onset envelope
    result.beats = pickPeaks(result.onsetEnvelope, adaptiveThreshold);

    {
        std::ostringstream oss;
        oss << "[AudioFlux] Peaks found before gap fill: " << result.beats.size();
        debugLog(oss.str());
    }

    // Convert frame indices to time (must be done BEFORE fillBeatGaps which expects seconds)
    float frameRate = static_cast<float>(m_config.sampleRate) / m_config.hopLength;
    for (auto& beat : result.beats) {
        beat = beat / frameRate;
    }

    // Post-process: fill gaps using beat grid interpolation (expects beats in seconds)
    size_t beatsBeforeFill = result.beats.size();
    result.beats = fillBeatGaps(result.beats, duration);

    {
        std::ostringstream oss;
        oss << "[AudioFlux] Beats after gap fill: " << result.beats.size() << " (added " << (result.beats.size() - beatsBeforeFill) << ")";
        debugLog(oss.str());
    }

    if (progress && !progress(0.8f, "Estimating tempo...")) {
        result.error = "Analysis cancelled by user";
        return result;
    }

    // Estimate BPM
    result.bpm = estimateBPM(result.beats, duration);

    {
        std::ostringstream oss;
        oss << "[AudioFlux] FINAL RESULT: " << result.beats.size() << " beats, BPM=" << result.bpm << ", duration=" << duration << "s";
        debugLog(oss.str());
    }

    // Calculate confidence based on beat regularity
    if (result.beats.size() > 10) {
        std::vector<double> intervals;
        for (size_t i = 1; i < result.beats.size(); ++i) {
            intervals.push_back(result.beats[i] - result.beats[i - 1]);
        }
        std::sort(intervals.begin(), intervals.end());
        double medianInterval = intervals[intervals.size() / 2];

        if (medianInterval <= std::numeric_limits<double>::epsilon()) {
            result.confidence = 0.0;
        } else {
            // Calculate variance from median
            double variance = 0.0;
            for (double interval : intervals) {
                double diff = interval - medianInterval;
                variance += diff * diff;
            }
            variance /= intervals.size();

            // Confidence: inverse of normalized variance
            double stdDev = std::sqrt(variance);
            double relativeStdDev = stdDev / medianInterval;
            result.confidence = std::max(0.0, 1.0 - relativeStdDev * 2.0);
        }
    }

    if (progress && !progress(1.0f, "Done")) {
        result.error = "Analysis cancelled by user";
        return result;
    }

    return result;
}

std::vector<float> AudioFluxBeatDetector::computeOnsetEnvelope(
    const float* stftReal, const float* stftImag, int numFrames, int numBins, int fftStride) {

    // Validate input parameters to prevent buffer overruns
    if (!stftReal || !stftImag || numFrames <= 0 || numBins <= 0 || fftStride <= 0) {
        debugLog("[AudioFlux] ERROR: Invalid STFT parameters in computeOnsetEnvelope");
        return {};
    }

    // Validate stride is sufficient for numBins (stride must be >= numBins to access all frequency bins)
    if (fftStride < numBins) {
        std::ostringstream oss;
        oss << "[AudioFlux] ERROR: fftStride (" << fftStride << ") < numBins (" << numBins << ") - would cause buffer underread";
        debugLog(oss.str());
        return {};
    }

    std::vector<float> envelope(numFrames, 0.0f);

    // Calculate frequency bin range for kick drum detection
    // binWidth = sampleRate / fftSize
    float binWidth = static_cast<float>(m_config.sampleRate) / m_config.fftSize;

    int minBin = 0;
    int maxBin = numBins;

    if (m_config.lowFreqFocus) {
        // Focus only on kick drum frequencies (typically 30-200Hz for EDM/psytrance)
        minBin = static_cast<int>(m_config.lowFreqMin / binWidth);
        maxBin = static_cast<int>(m_config.lowFreqMax / binWidth) + 1;
        minBin = std::max(0, minBin);
        maxBin = std::min(numBins, maxBin);

        {
            std::ostringstream oss;
            oss << "[AudioFlux] Low-freq focus enabled: bins " << minBin << "-" << maxBin
                << " (" << (minBin * binWidth) << "-" << (maxBin * binWidth) << " Hz)";
            debugLog(oss.str());
        }
    }

    // Compute magnitude spectrogram
    // Note: AudioFlux STFT buffer has fftStride (fftLength) values per frame,
    // but we only use the specified frequency range
    std::vector<float> prevMag(numBins, 0.0f);

    for (int t = 0; t < numFrames; ++t) {
        float flux = 0.0f;
        float energy = 0.0f;

        // Only sum spectral flux in the kick drum frequency range
        for (int b = minBin; b < maxBin; ++b) {
            // Use fftStride for frame indexing, not numBins
            int idx = t * fftStride + b;
            float real = stftReal[idx];
            float imag = stftImag[idx];
            float mag = std::sqrt(real * real + imag * imag);

            // Accumulate energy for RMS calculation
            energy += mag * mag;

            // Spectral flux: half-wave rectified difference
            float diff = mag - prevMag[b];
            if (diff > 0) {
                flux += diff;
            }

            prevMag[b] = mag;
        }

        // Calculate RMS energy for this frame and gate quiet frames
        int numBinsInRange = maxBin - minBin;
        if (numBinsInRange > 0) {
            float rms = std::sqrt(energy / numBinsInRange);
            if (rms < m_config.energyGate) {
                flux = 0.0f;  // Gate out quiet frames
            }
        }

        envelope[t] = flux;
    }

    // Normalize envelope
    float maxVal = *std::max_element(envelope.begin(), envelope.end());
    if (maxVal > 0) {
        for (float& v : envelope) {
            v /= maxVal;
        }
    }

    return envelope;
}

std::vector<double> AudioFluxBeatDetector::pickPeaks(const std::vector<float>& envelope, float threshold) {
    std::vector<double> peaks;

    float frameRate = static_cast<float>(m_config.sampleRate) / m_config.hopLength;
    int minIntervalFrames = static_cast<int>(m_config.minBeatInterval * frameRate);

    int lastPeakIdx = -minIntervalFrames * 2;

    // First pass: find all local maxima above threshold (less strict - only 1 neighbor each side)
    for (size_t i = 1; i + 1 < envelope.size(); ++i) {
        // Simple local maximum: higher than immediate neighbors
        if (envelope[i] > threshold &&
            envelope[i] > envelope[i - 1] &&
            envelope[i] >= envelope[i + 1]) {

            // Minimum interval check
            if (static_cast<int>(i) - lastPeakIdx >= minIntervalFrames) {
                peaks.push_back(static_cast<double>(i));
                lastPeakIdx = static_cast<int>(i);
            }
        }
    }

    // Second pass: if we still have gaps, lower threshold and look for prominent peaks
    // This helps catch beats that are slightly below the global threshold but still prominent locally
    if (peaks.size() > 4) {
        float avgInterval = 0;
        for (size_t i = 1; i < peaks.size(); ++i) {
            avgInterval += static_cast<float>(peaks[i] - peaks[i - 1]);
        }
        avgInterval /= (peaks.size() - 1);

        // Look for gaps larger than 1.8x average interval
        std::vector<double> additionalPeaks;
        for (size_t i = 1; i < peaks.size(); ++i) {
            double gap = peaks[i] - peaks[i - 1];
            if (gap > avgInterval * 1.8) {
                // Search for the strongest peak in this gap with lower threshold
                int startFrame = static_cast<int>(peaks[i - 1]) + minIntervalFrames;
                int endFrame = static_cast<int>(peaks[i]) - minIntervalFrames;

                float localThreshold = threshold * 0.75f;  // 75% of normal threshold (stricter)
                float maxVal = 0;
                int maxIdx = -1;

                for (int j = startFrame; j < endFrame && j < static_cast<int>(envelope.size()); ++j) {
                    if (envelope[j] > localThreshold && envelope[j] > maxVal) {
                        // Check it's a local max
                        if (j > 0 && j < static_cast<int>(envelope.size()) - 1 &&
                            envelope[j] > envelope[j - 1] && envelope[j] >= envelope[j + 1]) {
                            maxVal = envelope[j];
                            maxIdx = j;
                        }
                    }
                }

                if (maxIdx > 0) {
                    additionalPeaks.push_back(static_cast<double>(maxIdx));
                }
            }
        }

        // Merge additional peaks
        for (double p : additionalPeaks) {
            peaks.push_back(p);
        }
        std::sort(peaks.begin(), peaks.end());
    }

    return peaks;
}

std::vector<double> AudioFluxBeatDetector::fillBeatGaps(const std::vector<double>& beats, double duration) {
    if (beats.size() < 4) return beats;

    // First, estimate the dominant beat interval from existing beats
    std::vector<double> intervals;
    for (size_t i = 1; i < beats.size(); ++i) {
        double interval = beats[i] - beats[i - 1];
        if (interval > 0.15 && interval < 1.5) {  // Valid interval range
            intervals.push_back(interval);
        }
    }

    if (intervals.empty()) return beats;

    // Sort and find the mode (most common interval)
    std::sort(intervals.begin(), intervals.end());

    // Use histogram to find dominant interval (10ms bins)
    std::map<int, int> histogram;
    for (double interval : intervals) {
        int bin = static_cast<int>(interval * 100);  // 10ms bins
        histogram[bin]++;
    }

    int maxCount = 0;
    int modeBin = 0;
    for (const auto& pair : histogram) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            modeBin = pair.first;
        }
    }

    double beatInterval = modeBin / 100.0;

    // If the mode isn't well-supported, use median
    // Use safe threshold to handle small interval counts (avoid integer division yielding 0)
    int minSupport = std::max(1, static_cast<int>(intervals.size()) / 5);
    if (maxCount < minSupport) {
        beatInterval = intervals[intervals.size() / 2];
    }

    // Now fill gaps where interval is significantly larger than expected
    std::vector<double> filledBeats;
    filledBeats.push_back(beats[0]);

    for (size_t i = 1; i < beats.size(); ++i) {
        double gap = beats[i] - beats[i - 1];

        // If gap is more than 1.5x expected interval, we're missing beats
        if (gap > beatInterval * 1.5) {
            // Calculate how many beats should fit in this gap
            int missingBeats = static_cast<int>(std::round(gap / beatInterval)) - 1;

            if (missingBeats > 0 && missingBeats <= 2) {  // Only fill small gaps (1-2 beats), larger gaps are intentional
                double stepSize = gap / (missingBeats + 1);
                for (int j = 1; j <= missingBeats; ++j) {
                    double interpolatedBeat = beats[i - 1] + j * stepSize;
                    filledBeats.push_back(interpolatedBeat);
                }
            }
        }

        filledBeats.push_back(beats[i]);
    }

    // DISABLED BY DEFAULT: Extending to track edges creates false positives in intros/outros
    // Only enable if explicitly configured (extendToEdges = true)
    if (m_config.extendToEdges) {
        // Extend to beginning if first beat is late
        if (filledBeats[0] > beatInterval * 1.5) {
            std::vector<double> prependBeats;
            double t = filledBeats[0] - beatInterval;
            while (t > 0.1) {  // Don't go too close to start
                prependBeats.push_back(t);
                t -= beatInterval;
            }
            // Reverse and prepend
            std::reverse(prependBeats.begin(), prependBeats.end());
            prependBeats.insert(prependBeats.end(), filledBeats.begin(), filledBeats.end());
            filledBeats = std::move(prependBeats);
        }

        // Extend to end if last beat is early
        // Guard against zero or very small beatInterval to prevent infinite loop
        if (beatInterval >= 0.1) {
            double lastBeat = filledBeats.back();
            while (lastBeat + beatInterval < duration - 0.1) {
                lastBeat += beatInterval;
                filledBeats.push_back(lastBeat);
            }
        }
    }

    return filledBeats;
}

double AudioFluxBeatDetector::estimateBPM(const std::vector<double>& beats, double duration) {
    if (beats.size() < 4) return 0.0;

    // Compute inter-beat intervals
    std::vector<double> intervals;
    for (size_t i = 1; i < beats.size(); ++i) {
        double interval = beats[i] - beats[i - 1];
        if (interval > 0.2 && interval < 2.0) {  // 30-300 BPM range
            intervals.push_back(interval);
        }
    }

    if (intervals.empty()) return 0.0;

    // Sort for percentile calculations
    std::sort(intervals.begin(), intervals.end());

    // Use median for robustness (less sensitive to outliers than mean)
    double medianInterval = intervals[intervals.size() / 2];

    // Also compute mode by clustering intervals
    // Build histogram of intervals with 5ms bins
    std::map<int, int> histogram;
    for (double interval : intervals) {
        int bin = static_cast<int>(interval * 200);  // 5ms bins
        histogram[bin]++;
    }

    // Find the most common interval bin
    int maxCount = 0;
    int modeBin = 0;
    for (const auto& pair : histogram) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            modeBin = pair.first;
        }
    }
    double modeInterval = modeBin / 200.0;

    // Use mode if it has significant support, otherwise median
    // Use safe threshold to handle small interval counts (avoid integer division yielding 0)
    int minSupportBPM = std::max(1, static_cast<int>(intervals.size()) / 5);
    double bestInterval = (maxCount > minSupportBPM) ? modeInterval : medianInterval;

    // Also try subdivisions and multiples to find best fit
    std::vector<double> candidates = {
        bestInterval,
        bestInterval * 2.0,
        bestInterval / 2.0
    };

    // Score each candidate by how many intervals match (within 5%)
    double bestScore = 0;
    double bestCandidate = bestInterval;

    for (double candidate : candidates) {
        if (candidate < 0.2 || candidate > 2.0) continue;

        int matches = 0;
        for (double interval : intervals) {
            // Check if interval is close to candidate or its multiples
            double ratio = interval / candidate;
            double roundedRatio = std::round(ratio);
            if (roundedRatio >= 1 && roundedRatio <= 4) {
                double error = std::abs(ratio - roundedRatio) / roundedRatio;
                if (error < 0.05) {  // Within 5%
                    matches++;
                }
            }
        }

        double score = static_cast<double>(matches) / intervals.size();
        if (score > bestScore) {
            bestScore = score;
            bestCandidate = candidate;
        }
    }

    if (bestCandidate <= 1e-6) return 0.0;
    double bpm = 60.0 / bestCandidate;

    // Normalize to common range (70-180 BPM)
    // This handles cases where we detected half or double time
    while (bpm > 180.0) bpm /= 2.0;
    while (bpm > 0.0 && bpm < 70.0) bpm *= 2.0;

    return bpm;
}
