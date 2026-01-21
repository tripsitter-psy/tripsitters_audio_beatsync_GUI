#include "DebugLogger.h"

#include <iostream>
#include <fstream>
#include <mutex>
#include <cstdlib>

namespace BeatSync {

DebugLogger& DebugLogger::getInstance() {
    static DebugLogger instance;
    return instance;
}

DebugLogger::DebugLogger() : logFile_(nullptr), initialized_(false) {
    // Initialize on first access (lazy initialization)
    static std::once_flag initFlag;
    std::call_once(initFlag, [this]() {
        // Get temp directory
        const char* tempDir = std::getenv("TEMP");
        if (!tempDir) tempDir = std::getenv("TMP");
        if (!tempDir) tempDir = "C:\\Temp";

        std::string logPath = std::string(tempDir) + "\\beatsync_debug.log";
        logFile_ = new std::ofstream(logPath, std::ios::out | std::ios::app);

        if (logFile_->is_open()) {
            *logFile_ << "[BeatSync] Debug log started at " << logPath << std::endl;
            initialized_ = true;
        } else {
            delete logFile_;
            logFile_ = nullptr;
        }
    });
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