#include "EssentiaAnalyzer.h"

namespace BeatSync {

BeatGrid EssentiaAnalyzer::analyze(const std::string& audioFilePath) {
    (void)audioFilePath; // unused until Essentia is wired

    BeatGrid grid;
    m_lastError = "Essentia integration not enabled (stub)";
    return grid;
}

} // namespace BeatSync
