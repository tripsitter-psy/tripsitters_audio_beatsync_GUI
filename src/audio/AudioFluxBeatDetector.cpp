#include "AudioFluxBeatDetector.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <iostream>

// AudioFlux C API headers
extern "C" {
#include "flux_base.h"
#include "stft_algorithm.h"
#include "mir/onset_algorithm.h"
}

// Helper: resample audio to target sample rate (band-limited, anti-aliased)
// NOTE: For production use, consider linking libsamplerate for higher quality resampling.
#include <stdexcept>

static std::vector<float> resampleAudio(const std::vector<float>& input, int inputRate, int outputRate) {
    if (inputRate == outputRate) return input;
    if (input.empty()) return input;

#ifdef HAVE_LIBSAMPLERATE
    // High-quality resampling using libsamplerate (if linked):
    std::vector<float> output;
    SRC_DATA srcData;
    srcData.data_in = input.data();
    srcData.input_frames = static_cast<long>(input.size());
    double ratio = static_cast<double>(outputRate) / inputRate;
    size_t outputSize = static_cast<size_t>(input.size() * ratio);
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
    double ratio = static_cast<double>(outputRate) / inputRate;
    size_t outputSize = static_cast<size_t>(input.size() * ratio);

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

    std::cerr << "[AudioFlux] detect() called with " << samples.size() << " samples at " << sampleRate << " Hz" << std::endl;

    if (samples.empty()) {
        result.error = "Empty audio data";
        return result;
    }

    // Resample to target sample rate if needed
    if (progress && !progress(0.05f, "Resampling audio...")) return result;

    std::cerr << "[AudioFlux] Resampling from " << sampleRate << " to " << m_config.sampleRate << std::endl;
    std::vector<float> resampled = resampleAudio(samples, sampleRate, m_config.sampleRate);
    double duration = static_cast<double>(resampled.size()) / m_config.sampleRate;
    std::cerr << "[AudioFlux] Resampled to " << resampled.size() << " samples, duration=" << duration << "s" << std::endl;

    // Compute STFT
    if (progress && !progress(0.1f, "Computing spectrogram...")) return result;

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

    std::cerr << "[AudioFlux] STFT params: radix2Exp=" << radix2Exp << ", fftLength=" << fftLength << ", numBins=" << numBins << std::endl;

    // Create STFT object
    STFTObj stftObj = nullptr;
    WindowType windowType = Window_Hann;
    int slideLength = m_config.hopLength;
    int isContinue = 0;

    std::cerr << "[AudioFlux] Creating STFT object..." << std::endl;
    int stftResult = stftObj_new(&stftObj, radix2Exp, &windowType, &slideLength, &isContinue);
    std::cerr << "[AudioFlux] stftObj_new returned " << stftResult << ", stftObj=" << stftObj << std::endl;

    if (stftResult != 0 || !stftObj) {
        result.error = "Failed to create STFT object";
        return result;
    }

    // Calculate number of frames
    std::cerr << "[AudioFlux] Calculating time length..." << std::endl;
    int numFrames = stftObj_calTimeLength(stftObj, static_cast<int>(resampled.size()));
    std::cerr << "[AudioFlux] numFrames=" << numFrames << std::endl;

    if (numFrames <= 0) {
        stftObj_free(stftObj);
        result.error = "Invalid frame count";
        return result;
    }

    // Safety check: limit buffer size to prevent crashes on very long files
    // NOTE: AudioFlux outputs fftLength values per frame, not numBins!
    // For a 6-minute track at 22050Hz with hop=512, we need ~128MB
    // Allow up to 500MB for tracks up to ~25 minutes
    const size_t maxBufferSize = 500 * 1024 * 1024; // 500MB max
    size_t bufferSize = static_cast<size_t>(numFrames) * static_cast<size_t>(fftLength) * sizeof(float);
    std::cerr << "[AudioFlux] Buffer size needed: " << bufferSize << " bytes (" << numFrames << " frames x " << fftLength << " bins)" << std::endl;

    if (bufferSize > maxBufferSize) {
        stftObj_free(stftObj);
        result.error = "Audio file too long for analysis";
        return result;
    }

    // Allocate STFT output buffers
    // CRITICAL: AudioFlux STFT writes fftLength values per frame (full complex FFT output)
    std::cerr << "[AudioFlux] Allocating STFT buffers..." << std::endl;
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
    std::cerr << "[AudioFlux] Buffers allocated: " << stftReal.size() << " floats each (for " << numFrames << " frames)" << std::endl;

    // Compute STFT
    // NOTE: AudioFlux stft outputs mRealArr and mImageArr as flat arrays of size numFrames * numBins
    // where each frame has numBins complex values
    std::cerr << "[AudioFlux] Computing STFT (input size=" << resampled.size() << ")..." << std::endl;
    std::cerr << "[AudioFlux] Output buffer sizes: real=" << stftReal.size() << ", imag=" << stftImag.size() << std::endl;
    std::cerr.flush();

    // Zero-initialize buffers to be safe
    std::fill(stftReal.begin(), stftReal.end(), 0.0f);
    std::fill(stftImag.begin(), stftImag.end(), 0.0f);

    stftObj_stft(stftObj, resampled.data(), static_cast<int>(resampled.size()),
                 stftReal.data(), stftImag.data());
    std::cerr << "[AudioFlux] STFT complete" << std::endl;

    stftObj_free(stftObj);
    std::cerr << "[AudioFlux] STFT object freed" << std::endl;

    if (progress && !progress(0.3f, "Computing onset envelope...")) return result;

    // Compute onset envelope using spectral flux
    // Note: stft buffer has fftLength stride, but we only use first numBins (positive frequencies)
    result.onsetEnvelope = computeOnsetEnvelope(stftReal.data(), stftImag.data(), numFrames, numBins, fftLength);

    // Safety check
    if (result.onsetEnvelope.empty()) {
        result.error = "Failed to compute onset envelope";
        return result;
    }

    if (progress && !progress(0.5f, "Detecting beats...")) return result;

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
    adaptiveThreshold = std::max(adaptiveThreshold, maxEnv * 0.08f);  // At least 8% of max
    adaptiveThreshold = std::min(adaptiveThreshold, maxEnv * 0.25f);  // At most 25% of max

    // Pick peaks from onset envelope
    result.beats = pickPeaks(result.onsetEnvelope, adaptiveThreshold);

    // Post-process: fill gaps using beat grid interpolation
    result.beats = fillBeatGaps(result.beats, duration);

    // Convert frame indices to time
    float frameRate = static_cast<float>(m_config.sampleRate) / m_config.hopLength;
    for (auto& beat : result.beats) {
        beat = beat / frameRate;
    }

    if (progress && !progress(0.8f, "Estimating tempo...")) return result;

    // Estimate BPM
    result.bpm = estimateBPM(result.beats, duration);

    // Calculate confidence based on beat regularity
    if (result.beats.size() > 10) {
        std::vector<double> intervals;
        for (size_t i = 1; i < result.beats.size(); ++i) {
            intervals.push_back(result.beats[i] - result.beats[i - 1]);
        }
        std::sort(intervals.begin(), intervals.end());
        double medianInterval = intervals[intervals.size() / 2];

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

    if (progress && !progress(1.0f, "Done")) return result;

    return result;
}

std::vector<float> AudioFluxBeatDetector::computeOnsetEnvelope(
    const float* stftReal, const float* stftImag, int numFrames, int numBins, int fftStride) {

    std::vector<float> envelope(numFrames, 0.0f);

    // Compute magnitude spectrogram
    // Note: AudioFlux STFT buffer has fftStride (fftLength) values per frame,
    // but we only use the first numBins (positive frequencies)
    std::vector<float> prevMag(numBins, 0.0f);

    for (int t = 0; t < numFrames; ++t) {
        float flux = 0.0f;

        for (int b = 0; b < numBins; ++b) {
            // Use fftStride for frame indexing, not numBins
            int idx = t * fftStride + b;
            float real = stftReal[idx];
            float imag = stftImag[idx];
            float mag = std::sqrt(real * real + imag * imag);

            // Spectral flux: half-wave rectified difference
            float diff = mag - prevMag[b];
            if (diff > 0) {
                flux += diff;
            }

            prevMag[b] = mag;
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

                float localThreshold = threshold * 0.5f;  // 50% of normal threshold
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
    if (maxCount < static_cast<int>(intervals.size()) / 5) {
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

            if (missingBeats > 0 && missingBeats < 8) {  // Don't fill huge gaps (likely silence)
                double stepSize = gap / (missingBeats + 1);
                for (int j = 1; j <= missingBeats; ++j) {
                    double interpolatedBeat = beats[i - 1] + j * stepSize;
                    filledBeats.push_back(interpolatedBeat);
                }
            }
        }

        filledBeats.push_back(beats[i]);
    }

    // Also extend to beginning if first beat is late
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
    double lastBeat = filledBeats.back();
    while (lastBeat + beatInterval < duration - 0.1) {
        lastBeat += beatInterval;
        filledBeats.push_back(lastBeat);
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
    double bestInterval = (maxCount > intervals.size() / 5) ? modeInterval : medianInterval;

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

    double bpm = 60.0 / bestCandidate;

    // Normalize to common range (70-180 BPM)
    // This handles cases where we detected half or double time
    while (bpm > 180.0) bpm /= 2.0;
    while (bpm > 0.0 && bpm < 70.0) bpm *= 2.0;

    return bpm;
}
