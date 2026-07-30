#include "DynamicSync.h"

#include <algorithm>
#include <cmath>

namespace BeatSync {

std::vector<double> DynamicSync::computeEnergyEnvelope(const std::vector<float>& samples,
                                                       int sampleRate,
                                                       double hopSec,
                                                       double smoothingSec) {
    std::vector<double> envelope;
    if (samples.empty() || sampleRate <= 0 || hopSec <= 0.0) {
        return envelope;
    }

    const size_t hopSize = static_cast<size_t>(hopSec * sampleRate);
    if (hopSize == 0) {
        return envelope;
    }

    // RMS energy per hop
    envelope.reserve(samples.size() / hopSize + 1);
    for (size_t start = 0; start < samples.size(); start += hopSize) {
        const size_t end = std::min(start + hopSize, samples.size());
        double sum = 0.0;
        for (size_t i = start; i < end; ++i) {
            sum += static_cast<double>(samples[i]) * static_cast<double>(samples[i]);
        }
        envelope.push_back(std::sqrt(sum / static_cast<double>(end - start)));
    }

    // Moving-average smoothing
    const size_t window = std::max<size_t>(1, static_cast<size_t>(smoothingSec / hopSec));
    if (window > 1 && envelope.size() > window) {
        std::vector<double> smoothed(envelope.size());
        double runningSum = 0.0;
        size_t count = 0;
        const size_t half = window / 2;
        for (size_t i = 0; i < envelope.size(); ++i) {
            // Grow/shrink a centered window incrementally
            const size_t addIdx = i + half;
            if (addIdx < envelope.size()) {
                runningSum += envelope[addIdx];
                ++count;
            }
            if (i > half) {
                runningSum -= envelope[i - half - 1];
                --count;
            } else if (i == 0) {
                // Initialize with the leading half-window
                for (size_t j = 0; j < half && j < envelope.size(); ++j) {
                    runningSum += envelope[j];
                    ++count;
                }
            }
            smoothed[i] = (count > 0) ? runningSum / static_cast<double>(count) : envelope[i];
        }
        envelope.swap(smoothed);
    }

    // Percentile-based normalization to [0,1] (robust against outlier transients)
    std::vector<double> sorted(envelope);
    std::sort(sorted.begin(), sorted.end());
    const double lo = sorted[static_cast<size_t>(0.05 * (sorted.size() - 1))];
    const double hi = sorted[static_cast<size_t>(0.95 * (sorted.size() - 1))];
    const double range = hi - lo;
    if (range <= 1e-12) {
        // Flat energy: everything is "normal"
        std::fill(envelope.begin(), envelope.end(), 0.5);
        return envelope;
    }
    for (double& v : envelope) {
        v = std::clamp((v - lo) / range, 0.0, 1.0);
    }
    return envelope;
}

std::vector<int> DynamicSync::classifyBeats(const std::vector<double>& beats,
                                            const std::vector<double>& envelope,
                                            double hopSec,
                                            const DynamicSyncConfig& config) {
    std::vector<int> bands;
    bands.reserve(beats.size());
    if (beats.empty() || envelope.empty() || hopSec <= 0.0) {
        return bands;
    }

    for (double t : beats) {
        size_t idx = static_cast<size_t>(t / hopSec);
        idx = std::min(idx, envelope.size() - 1);
        const double e = envelope[idx];
        int band = 1;
        if (e < config.lowThreshold) {
            band = 0;
        } else if (e > config.highThreshold) {
            band = 2;
        }
        bands.push_back(band);
    }

    // Merge sections shorter than minSectionSec into the previous section so the
    // cut rate doesn't flip-flop on momentary energy dips/spikes.
    if (config.minSectionSec > 0.0) {
        size_t sectionStart = 0;
        for (size_t i = 1; i <= bands.size(); ++i) {
            if (i == bands.size() || bands[i] != bands[sectionStart]) {
                const double sectionLen = beats[i - 1] - beats[sectionStart];
                if (sectionStart > 0 && sectionLen < config.minSectionSec) {
                    const int prevBand = bands[sectionStart - 1];
                    for (size_t j = sectionStart; j < i; ++j) {
                        bands[j] = prevBand;
                    }
                    // Keep sectionStart: the merged run may now continue into i
                } else {
                    sectionStart = i;
                }
            }
        }
    }

    return bands;
}

std::vector<double> DynamicSync::filterBeats(const std::vector<double>& beats,
                                             const std::vector<double>& envelope,
                                             double hopSec,
                                             const DynamicSyncConfig& config) {
    std::vector<double> filtered;
    if (beats.empty()) {
        return filtered;
    }
    if (envelope.empty() || hopSec <= 0.0) {
        return beats;  // No envelope: keep every beat
    }

    const std::vector<int> bands = classifyBeats(beats, envelope, hopSec, config);
    const int divisors[3] = {
        std::max(1, config.calmDivisor),
        std::max(1, config.normalDivisor),
        std::max(1, config.franticDivisor)
    };

    filtered.reserve(beats.size());
    size_t beatInSection = 0;
    for (size_t i = 0; i < beats.size(); ++i) {
        if (i > 0 && bands[i] != bands[i - 1]) {
            beatInSection = 0;  // Section boundary always lands on a cut
        }
        const int divisor = divisors[bands[i]];
        if (beatInSection % static_cast<size_t>(divisor) == 0) {
            filtered.push_back(beats[i]);
        }
        ++beatInSection;
    }
    return filtered;
}

} // namespace BeatSync
