#pragma once

#include "BeatGrid.h"
#include <string>

namespace BeatSync {

// Placeholder Essentia-backed analyzer scaffold. This is intentionally minimal
// until Essentia is added to the build. When wired, replace the stub body with
// calls into Essentia's rhythm/onset APIs.
class EssentiaAnalyzer {
public:
    EssentiaAnalyzer() = default;

    BeatGrid analyze(const std::string& audioFilePath);

    const std::string& getLastError() const { return m_lastError; }

private:
    std::string m_lastError;
};

} // namespace BeatSync
