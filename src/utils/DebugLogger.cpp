#include "DebugLogger.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <cstdlib>

#include <filesystem>

namespace BeatSync {

DebugLogger& DebugLogger::getInstance() {
    static DebugLogger instance;
    return instance;
}

DebugLogger::DebugLogger() : logFile_(nullptr), initialized_(false) {
    // Use portable temp directory and path joining
    try {
        namespace fs = std::filesystem;
        fs::path tempDir = fs::temp_directory_path();
        fs::path logPath = tempDir / "beatsync_debug.log";
        logFile_ = new std::ofstream(logPath.string(), std::ios::out | std::ios::app);
        if (logFile_->is_open()) {
            *logFile_ << "[BeatSync] Debug log started at " << logPath << std::endl;
            initialized_ = true;
        } else {
            delete logFile_;
            logFile_ = nullptr;
        }
    } catch (...) {
        logFile_ = nullptr;
        initialized_ = false;
    }
}

DebugLogger::~DebugLogger() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (logFile_) {
        logFile_->close();
        delete logFile_;
        logFile_ = nullptr;
    }
}

void DebugLogger::log(const std::string& msg) {
    // Always output to stderr
    std::cerr << msg << std::endl;

    // Also write to file if initialized
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_ && logFile_ && logFile_->is_open()) {
        *logFile_ << msg << std::endl;
        logFile_->flush();
    }
}

} // namespace BeatSync