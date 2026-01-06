#include "BeatNetBridge.h"
#include <cstdio>
#include <sstream>
#include <iostream>

namespace BeatSync {

void BeatNetBridge::setPythonPath(const std::string& pythonPath) {
    m_pythonPath = pythonPath;
}

BeatGrid BeatNetBridge::analyze(const std::string& audioFilePath) {
    (void)audioFilePath; // unused until subprocess call is implemented

    BeatGrid grid;
    m_lastError = "BeatNet bridge not wired (stub)";
    // TODO: build command: pythonPath scripts/beatnet_analyze.py audioFilePath
    // TODO: parse JSON output and populate grid
    return grid;
}

} // namespace BeatSync
