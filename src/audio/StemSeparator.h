#pragma once

#include <string>
#include <vector>

namespace BeatSync {

// Describes paths to separated audio stems (drums, bass, vocals, other).
struct StemPaths {
    std::string drums;
    std::string bass;
    std::string vocals;
    std::string other;
};

// Subprocess wrapper around scripts/demucs_separate.py. Invokes Demucs to
// separate an audio file into stems, returning paths to the produced files.
class StemSeparator {
public:
    StemSeparator() = default;

    // Run Demucs to separate stems. Returns empty paths on failure.
    StemPaths separate(const std::string& audioFilePath, const std::string& outputDir);

    // Set path to Python executable.
    void setPythonPath(const std::string& pythonPath);

    const std::string& getLastError() const { return m_lastError; }

private:
    std::string m_pythonPath;
    std::string m_lastError;
};

} // namespace BeatSync
