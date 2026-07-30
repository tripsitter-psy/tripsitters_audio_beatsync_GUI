#pragma once

#include <vector>
#include <cstddef>

namespace BeatSync {

/**
 * @brief Configuration for dynamic sync (energy-driven cut density)
 *
 * Dynamic sync makes the edit "breathe" with the track: calm sections cut
 * slower (every Nth beat), high-energy sections cut on every beat, and
 * sections in between cut at a middle rate. Sections are derived from the
 * track's smoothed energy envelope, normalized per-track so thresholds are
 * genre-independent.
 */
struct DynamicSyncConfig {
    double lowThreshold = 0.35;   // Normalized energy below this = calm section
    double highThreshold = 0.70;  // Normalized energy above this = frantic section
    int calmDivisor = 4;          // Cut every Nth beat in calm sections (breakdowns)
    int normalDivisor = 2;        // Cut every Nth beat in mid-energy sections
    int franticDivisor = 1;       // Cut every Nth beat in high-energy sections (drops)
    double smoothingSec = 2.0;    // Energy envelope smoothing window (seconds)
    double minSectionSec = 4.0;   // Minimum section length; shorter runs merge into neighbors
};

/**
 * @brief Energy-driven beat filtering for dynamic cut density
 */
class DynamicSync {
public:
    /**
     * @brief Compute a smoothed, per-track-normalized energy envelope
     * @param samples Mono audio samples
     * @param sampleRate Sample rate of the audio
     * @param hopSec Envelope sampling interval in seconds (e.g. 0.05)
     * @param smoothingSec Moving-average smoothing window in seconds
     * @return Envelope values in [0,1], one per hop; empty on bad input
     *
     * Normalization uses the 5th..95th percentile range so a few extreme
     * transients don't compress the useful dynamic range.
     */
    static std::vector<double> computeEnergyEnvelope(const std::vector<float>& samples,
                                                     int sampleRate,
                                                     double hopSec,
                                                     double smoothingSec);

    /**
     * @brief Classify each beat into an energy band (0=calm, 1=normal, 2=frantic)
     *
     * Applies the config thresholds to the envelope value at each beat time,
     * then merges sections shorter than minSectionSec into the previous section
     * so the cut rate doesn't flip-flop on momentary energy changes.
     */
    static std::vector<int> classifyBeats(const std::vector<double>& beats,
                                          const std::vector<double>& envelope,
                                          double hopSec,
                                          const DynamicSyncConfig& config);

    /**
     * @brief Filter a beat list to the dynamic cut density
     * @param beats Beat timestamps in seconds (sorted ascending)
     * @param envelope Energy envelope from computeEnergyEnvelope()
     * @param hopSec The hop interval the envelope was sampled at
     * @param config Thresholds and divisors
     * @return Subset of beats to cut on; equals input when all divisors are 1
     *
     * The first beat of every section is always kept so section boundaries
     * (e.g. the first kick of a drop) land on a cut.
     */
    static std::vector<double> filterBeats(const std::vector<double>& beats,
                                           const std::vector<double>& envelope,
                                           double hopSec,
                                           const DynamicSyncConfig& config);
};

} // namespace BeatSync
