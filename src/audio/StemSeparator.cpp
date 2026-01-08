#include "StemSeparator.h"

namespace BeatSync {

void StemSeparator::setPythonPath(const std::string& pythonPath) {
    m_pythonPath = pythonPath;
}

StemPaths StemSeparator::separate(const std::string& audioFilePath, const std::string& outputDir) {
    (void)audioFilePath;
    (void)outputDir;

    StemPaths paths;
    m_lastError = "Stem separation not wired (stub)";
    // TODO: invoke python scripts/demucs_separate.py audioFilePath outputDir
    // TODO: parse JSON output and populate paths
    return paths;
}

} // namespace BeatSync
