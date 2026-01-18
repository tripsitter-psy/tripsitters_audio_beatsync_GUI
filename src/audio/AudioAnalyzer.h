#pragma once

#include "BeatGrid.h"
#include <string>
#include <vector>
#include <memory>

namespace BeatSync {

/**
 * @brief Analyzes audio files to detect beats and tempo
 *
 * Uses FFmpeg to decode audio and applies energy-based beat detection
 * algorithm to find beat positions in the audio track.
 */
class AudioAnalyzer {
public:
    AudioAnalyzer();
    ~AudioAnalyzer();

    /**
     * @brief Analyze an audio file and detect beats
     * @param audioFilePath Path to the audio file (MP3, WAV, FLAC, etc.)
     * @return BeatGrid containing detected beat timestamps and BPM
     */
    BeatGrid analyze(const std::string& audioFilePath);

    /**
     * @brief Set beat detection sensitivity (0.0 to 1.0)
     * @param sensitivity Higher values detect more beats, lower values are more conservative
     *                    Default: 0.5
     */
    void setSensitivity(double sensitivity);

    /**
     * @brief Set BPM hint for beat detection
     * @param bpm The expected BPM (0 to disable hint and auto-detect)
     *
     * When set, the analyzer will use this BPM to generate evenly-spaced beats
     * starting from the first detected beat onset. This is useful when the user
     * knows the track's BPM and wants consistent beat spacing.
     */
    void setBPMHint(double bpm);

    /**
     * @brief Get the current BPM hint (0 if not set)
     */
    double getBPMHint() const;

    /**
     * @brief Get last error message
     */
    std::string getLastError() const;

    // Audio processing
    struct AudioData {
        std::vector<float> samples;  // Mono audio samples
        int sampleRate;
        double duration;
    };

    /**
     * @brief Load and decode audio file using FFmpeg (exposed for waveform visualization)
     */
    AudioData loadAudioFile(const std::string& filePath);

private:
    double m_sensitivity;
    double m_bpmHint;  // 0 = auto-detect, >0 = use this BPM
    std::string m_lastError;

    /**
     * @brief Detect beats in audio samples using energy-based algorithm
     */
    std::vector<double> detectBeats(const AudioData& audio);

    /**
     * @brief Calculate energy in a frame of audio
     */
    double calculateEnergy(const std::vector<float>& samples, size_t start, size_t length);

    /**
     * @brief Estimate BPM from detected beats
     */
    double estimateBPM(const std::vector<double>& beats);

    /**
     * @brief Apply moving average filter for smoothing
     */
    std::vector<double> movingAverage(const std::vector<double>& data, size_t windowSize);
};

} // namespace BeatSync
