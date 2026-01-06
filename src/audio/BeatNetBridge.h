#pragma once

#include "BeatGrid.h"
#include <string>

namespace BeatSync {

// Subprocess wrapper around scripts/beatnet_analyze.py. Invokes the Python
// script, parses JSON output, and populates a BeatGrid. Falls back to the
// built-in AudioAnalyzer if Python is unavailable or errors occur.
class BeatNetBridge {
public:
    BeatNetBridge() = default;

    // Analyze audio via BeatNet. Returns an empty grid if BeatNet unavailable.
    BeatGrid analyze(const std::string& audioFilePath);

    // Set path to Python executable (default: "python3" or "python" on Windows).
    void setPythonPath(const std::string& pythonPath);

    const std::string& getLastError() const { return m_lastError; }

private:
    std::string m_pythonPath;
    std::string m_lastError;
};

} // namespace BeatSync
