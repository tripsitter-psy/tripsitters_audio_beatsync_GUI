#pragma once

#include <vector>
#include <string>

namespace BeatSync {

/**
 * @brief Represents a grid of beat positions in an audio track
 *
 * Stores detected beat timestamps and provides utilities for
 * working with the beat grid (finding nearest beats, getting intervals, etc.)
 */
class BeatGrid {
public:
    BeatGrid();
    ~BeatGrid() = default;

    // Add a beat at a specific timestamp (in seconds)
    void addBeat(double timestamp);

    // Set all beats at once
    void setBeats(const std::vector<double>& timestamps);

    // Get all beat timestamps
    const std::vector<double>& getBeats() const;

    // Get number of beats
    size_t getNumBeats() const;

    // Get the beat closest to a given timestamp
    double getNearestBeat(double timestamp) const;

    // Get the beat interval (average time between beats)
    double getAverageBeatInterval() const;

    // Set/get BPM (beats per minute)
    void setBPM(double bpm);
    double getBPM() const;

    // Get duration of the audio (timestamp of last beat)
    double getDuration() const;

    // Set/get actual audio file duration (may be longer than last beat)
    void setAudioDuration(double duration);
    double getAudioDuration() const;

    // Clear all beats
    void clear();

    // Check if grid is empty
    bool isEmpty() const;

    // Get beat at specific index
    double getBeatAt(size_t index) const;

    // Find the index of the beat nearest to a timestamp
    size_t getNearestBeatIndex(double timestamp) const;

    // Print beat grid info (for debugging)
    std::string toString() const;

private:
    std::vector<double> m_beats;  // Beat timestamps in seconds
    double m_bpm;                 // Beats per minute
    double m_audioDuration;       // Actual audio file duration

    // Helper: ensure beats are sorted
    void sortBeats();
};

} // namespace BeatSync
