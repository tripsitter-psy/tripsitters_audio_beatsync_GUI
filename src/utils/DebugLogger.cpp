#include "DebugLogger.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <cstdlib>

#include <filesystem>

namespace BeatSync {

DebugLogger& DebugLogger::getInstance() {
    // Use a pointer that's never deleted to avoid static destruction order issues.
    // This prevents crashes if other static destructors try to log during shutdown.
    // The intentional "leak" is acceptable for a singleton that lives until program exit.
    static DebugLogger* instance = new DebugLogger();
    return *instance;
}

DebugLogger::DebugLogger() {
    // Use portable temp directory and path joining
    try {
        namespace fs = std::filesystem;
        fs::path tempDir = fs::temp_directory_path();
        fs::path logPath = tempDir / "beatsync_debug.log";
        logFile_.open(logPath.string(), std::ios::out | std::ios::app);
        if (logFile_.is_open()) {
            logFile_ << "[BeatSync] Debug log started at " << logPath << std::endl;
        }
    } catch (...) {
        if (logFile_.is_open()) logFile_.close();
    }
}

DebugLogger::~DebugLogger() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (logFile_.is_open()) {
        logFile_.close();
    }
}

void DebugLogger::log(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Always output to stderr, protected by mutex to prevent interleaving
    std::cerr << msg << std::endl;

    // Also write to file if initialized
    if (logFile_.is_open()) {
        logFile_ << msg << std::endl;
        logFile_.flush();
    }
}

} // namespace BeatSync