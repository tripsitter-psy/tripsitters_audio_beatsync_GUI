
#pragma once
#include <string>
#include <mutex>
#include <fstream>

namespace BeatSync {

// Thread-safe debug logging utility that writes to a file
// Used by audio processing components for debugging
class DebugLogger {
public:
    // Get the singleton instance
    static DebugLogger& getInstance();

    // Log a message (thread-safe)
    void log(const std::string& msg);

private:
    DebugLogger();
    ~DebugLogger();

    // Prevent copying and moving
    DebugLogger(const DebugLogger&) = delete;
    DebugLogger& operator=(const DebugLogger&) = delete;
    DebugLogger(DebugLogger&&) = delete;
    DebugLogger& operator=(DebugLogger&&) = delete;

    std::mutex mutex_;
    std::ofstream logFile_;
};

} // namespace BeatSync