#pragma once

#include "BeatGrid.h"
#include <string>
#include <functional>

namespace BeatSync {

// Analysis mode for BeatNetBridge
enum class BeatNetMode {
    SidecarOnly,      // Only use pre-generated sidecar JSON files (CI/deterministic)
    PythonOptIn,      // Try Python script first, fall back to sidecar
    PythonRequired    // Require Python script, fail if not available
};

// Configuration for BeatNetBridge
struct BeatNetConfig {
    BeatNetMode mode = BeatNetMode::SidecarOnly;  // Default: deterministic for CI
    std::string pythonPath;                        // Path to Python executable
    std::string scriptPath;                        // Path to beatnet_analyze.py
    int timeoutMs = 60000;                         // Timeout for Python subprocess (ms)
    bool verbose = false;                          // Enable verbose logging
};

// Progress callback for long-running analysis
using BeatNetProgressCallback = std::function<void(float progress, const std::string& status)>;

// Subprocess wrapper around scripts/beatnet_analyze.py. Invokes the Python
// script, parses JSON output, and populates a BeatGrid. Falls back to
// sidecar JSON files when configured or when Python is unavailable.
class BeatNetBridge {
public:
    BeatNetBridge();
    explicit BeatNetBridge(const BeatNetConfig& config);

    // Analyze audio via BeatNet. Returns an empty grid on failure.
    BeatGrid analyze(const std::string& audioFilePath);

    // Analyze with progress callback
    BeatGrid analyze(const std::string& audioFilePath, BeatNetProgressCallback progressCallback);

    // Configuration
    void setConfig(const BeatNetConfig& config) { m_config = config; }
    const BeatNetConfig& getConfig() const { return m_config; }

    // Legacy API compatibility
    void setPythonPath(const std::string& pythonPath) { m_config.pythonPath = pythonPath; }
    void setScriptPath(const std::string& scriptPath) { m_config.scriptPath = scriptPath; }

    // Error handling
    const std::string& getLastError() const { return m_lastError; }
    bool hasError() const { return !m_lastError.empty(); }

    // Check if Python is available and working
    bool isPythonAvailable() const;

    // Get the path to the default script (relative to executable)
    static std::string getDefaultScriptPath();

private:
    BeatNetConfig m_config;
    std::string m_lastError;

    // Internal methods
    BeatGrid analyzeWithPython(const std::string& audioFilePath, BeatNetProgressCallback progressCallback);
    BeatGrid analyzeWithSidecar(const std::string& audioFilePath);
    bool parseJsonOutput(const std::string& jsonStr, BeatGrid& outGrid);
    std::string runPythonProcess(const std::string& audioFilePath, int& exitCode);
    std::string findPythonExecutable() const;
};

} // namespace BeatSync
