#include "BeatGrid.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>

namespace BeatSync {

BeatGrid::BeatGrid()
    : m_bpm(0.0)
    , m_audioDuration(0.0)
{
}

void BeatGrid::addBeat(double timestamp) {
    m_beats.push_back(timestamp);
    sortBeats();
}

void BeatGrid::setBeats(const std::vector<double>& timestamps) {
    m_beats = timestamps;
    sortBeats();
}

const std::vector<double>& BeatGrid::getBeats() const {
    return m_beats;
}

size_t BeatGrid::getNumBeats() const {
    return m_beats.size();
}

double BeatGrid::getNearestBeat(double timestamp) const {
    if (m_beats.empty()) {
        return 0.0;
    }

    size_t index = getNearestBeatIndex(timestamp);
    return m_beats[index];
}

double BeatGrid::getAverageBeatInterval() const {
    if (m_beats.size() < 2) {
        return 0.0;
    }

    double totalInterval = m_beats.back() - m_beats.front();
    return totalInterval / (m_beats.size() - 1);
}

void BeatGrid::setBPM(double bpm) {
    m_bpm = bpm;
}

double BeatGrid::getBPM() const {
    // If BPM was set manually, return it
    if (m_bpm > 0.0) {
        return m_bpm;
    }

    // Otherwise calculate from average beat interval
    double interval = getAverageBeatInterval();
    if (interval > 0.0) {
        return 60.0 / interval;
    }

    return 0.0;
}

double BeatGrid::getDuration() const {
    if (m_beats.empty()) {
        return 0.0;
    }
    return m_beats.back();
}

void BeatGrid::setAudioDuration(double duration) {
    m_audioDuration = duration;
}

double BeatGrid::getAudioDuration() const {
    // Return actual audio duration if set, otherwise fall back to last beat
    if (m_audioDuration > 0.0) {
        return m_audioDuration;
    }
    return getDuration();
}

void BeatGrid::clear() {
    m_beats.clear();
    m_bpm = 0.0;
    m_audioDuration = 0.0;
}

bool BeatGrid::isEmpty() const {
    return m_beats.empty();
}

double BeatGrid::getBeatAt(size_t index) const {
    if (index >= m_beats.size()) {
        return 0.0;
    }
    return m_beats[index];
}

size_t BeatGrid::getNearestBeatIndex(double timestamp) const {
    if (m_beats.empty()) {
        return 0;
    }

    if (timestamp <= m_beats.front()) {
        return 0;
    }
    if (timestamp >= m_beats.back()) {
        return m_beats.size() - 1;
    }

    // Binary search for the closest beat
    auto it = std::lower_bound(m_beats.begin(), m_beats.end(), timestamp);

    if (it == m_beats.begin()) {
        return 0;
    }

    // Check if the previous beat is closer
    auto prevIt = it - 1;
    double distToCurrent = std::abs(*it - timestamp);
    double distToPrev = std::abs(*prevIt - timestamp);

    if (distToPrev < distToCurrent) {
        return std::distance(m_beats.begin(), prevIt);
    }

    return std::distance(m_beats.begin(), it);
}

std::string BeatGrid::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    oss << "BeatGrid Information:\n";
    oss << "  Number of beats: " << m_beats.size() << "\n";
    oss << "  BPM: " << getBPM() << "\n";
    oss << "  Duration: " << getDuration() << " seconds\n";
    oss << "  Average interval: " << getAverageBeatInterval() << " seconds\n";

    if (m_beats.size() > 0 && m_beats.size() <= 10) {
        oss << "  Beat timestamps: ";
        for (size_t i = 0; i < m_beats.size(); ++i) {
            oss << m_beats[i];
            if (i < m_beats.size() - 1) {
                oss << ", ";
            }
        }
        oss << "\n";
    } else if (m_beats.size() > 10) {
        oss << "  First 5 beats: ";
        for (size_t i = 0; i < 5; ++i) {
            oss << m_beats[i] << ", ";
        }
        oss << "...\n";
        oss << "  Last 5 beats: ";
        for (size_t i = m_beats.size() - 5; i < m_beats.size(); ++i) {
            oss << m_beats[i];
            if (i < m_beats.size() - 1) {
                oss << ", ";
            }
        }
        oss << "\n";
    }

    return oss.str();
}

void BeatGrid::sortBeats() {
    std::sort(m_beats.begin(), m_beats.end());
}

} // namespace BeatSync
