#include "BeatNetBridge.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <cstdlib>
#include <array>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace BeatSync {

BeatNetBridge::BeatNetBridge() {
    // Check environment variable for runtime opt-in
    const char* envVal = std::getenv("BEATSYNC_ENABLE_PYTHON");
    if (envVal && (std::string(envVal) == "1" || std::string(envVal) == "true")) {
        m_config.pythonEnabled = true;
    }
}

BeatNetBridge::BeatNetBridge(const BeatNetConfig& config)
    : m_config(config)
{
    // Check environment variable for runtime opt-in (can override config)
    const char* envVal = std::getenv("BEATSYNC_ENABLE_PYTHON");
    if (envVal && (std::string(envVal) == "1" || std::string(envVal) == "true")) {
        m_config.pythonEnabled = true;
    }
}

std::string BeatNetBridge::getDefaultScriptPath() {
    // Look for script relative to executable or in standard locations
#ifdef _WIN32
    return "scripts\\beatnet_analyze.py";
#else
    return "scripts/beatnet_analyze.py";
#endif
}

bool BeatNetBridge::isPythonEnabled() const {
    // Python is only enabled if:
    // 1. Compile-time flag ENABLE_BEATNET_PYTHON is set, AND
    // 2. Runtime flag pythonEnabled is true (via setPythonEnabled() or env var)
#ifdef ENABLE_BEATNET_PYTHON
    return m_config.pythonEnabled;
#else
    // Compile-time disabled - Python subprocess is never available
    return false;
#endif
}

std::string BeatNetBridge::findPythonExecutable() const {
    // Only search for Python if enabled
    if (!isPythonEnabled()) {
        return "";
    }

    // Use configured path if provided
    if (!m_config.pythonPath.empty()) {
        return m_config.pythonPath;
    }

    // Try common Python executable names
#ifdef _WIN32
    // On Windows, try python first (Python 3 installer uses this)
    const char* candidates[] = {"python", "python3", "py -3"};
#else
    const char* candidates[] = {"python3", "python"};
#endif

    for (const char* candidate : candidates) {
        std::string testCmd = std::string(candidate) + " --version";
#ifdef _WIN32
        // Use system() for simple availability check
        int result = system((testCmd + " >nul 2>&1").c_str());
#else
        int result = system((testCmd + " >/dev/null 2>&1").c_str());
#endif
        if (result == 0) {
            return candidate;
        }
    }

    return ""; // No Python found
}

bool BeatNetBridge::isPythonAvailable() const {
    // First check if Python is enabled (compile-time + runtime)
    if (!isPythonEnabled()) {
        return false;
    }
    // Then check if Python executable exists
    std::string python = findPythonExecutable();
    return !python.empty();
}

std::string BeatNetBridge::runPythonProcess(const std::string& audioFilePath, int& exitCode) {
#ifndef ENABLE_BEATNET_PYTHON
    // Compile-time disabled - return error
    exitCode = -1;
    return "Python subprocess disabled at compile time";
#else
    if (!isPythonEnabled()) {
        exitCode = -1;
        return "Python subprocess disabled at runtime";
    }

    std::string python = findPythonExecutable();
    if (python.empty()) {
        exitCode = -1;
        return "";
    }

    std::string scriptPath = m_config.scriptPath.empty() ? getDefaultScriptPath() : m_config.scriptPath;

    // Build command line
    std::ostringstream cmd;
#ifdef _WIN32
    cmd << "\"" << python << "\" \"" << scriptPath << "\" \"" << audioFilePath << "\"";
#else
    cmd << python << " \"" << scriptPath << "\" \"" << audioFilePath << "\"";
#endif

    std::string output;

#ifdef _WIN32
    // Windows implementation using _popen
    FILE* pipe = _popen(cmd.str().c_str(), "r");
    if (!pipe) {
        exitCode = -1;
        return "";
    }

    std::array<char, 4096> buffer;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    exitCode = _pclose(pipe);
#else
    // POSIX implementation using popen
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        exitCode = -1;
        return "";
    }

    std::array<char, 4096> buffer;
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        output += buffer.data();
    }

    int status = pclose(pipe);
    exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif

    return output;
#endif // ENABLE_BEATNET_PYTHON
}

bool BeatNetBridge::parseJsonOutput(const std::string& jsonStr, BeatGrid& outGrid) {
    // Simple JSON parser for BeatNet output format:
    // {"beats": [0.5, 1.0, 1.5, ...], "bpm": 120.0, "downbeats": [...]}

    // Extract beats array
    std::regex beatsRe(R"("beats"\s*:\s*\[([^\]]*)\])");
    std::smatch beatsMatch;
    std::vector<double> beats;

    if (std::regex_search(jsonStr, beatsMatch, beatsRe)) {
        std::string beatsStr = beatsMatch[1].str();
        std::regex numRe(R"([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?)");
        auto begin = std::sregex_iterator(beatsStr.begin(), beatsStr.end(), numRe);
        auto end = std::sregex_iterator();

        for (auto it = begin; it != end; ++it) {
            try {
                double val = std::stod(it->str());
                beats.push_back(val);
            } catch (...) {
                // Ignore parse errors for individual values
            }
        }
    }

    if (beats.empty()) {
        return false;
    }

    outGrid.setBeats(beats);

    // Extract BPM if present
    std::regex bpmRe(R"("bpm"\s*:\s*([0-9]+\.?[0-9]*))");
    std::smatch bpmMatch;
    if (std::regex_search(jsonStr, bpmMatch, bpmRe)) {
        try {
            double bpm = std::stod(bpmMatch[1].str());
            outGrid.setBPM(bpm);
        } catch (...) {
            // BPM parsing failed, compute from beats
        }
    }

    // If BPM not set, compute from beat intervals
    if (outGrid.getBPM() <= 0.0 && beats.size() >= 2) {
        double avgInterval = outGrid.getAverageBeatInterval();
        if (avgInterval > 0.0) {
            outGrid.setBPM(60.0 / avgInterval);
        }
    }

    return true;
}

BeatGrid BeatNetBridge::analyzeWithSidecar(const std::string& audioFilePath) {
    BeatGrid grid;

    std::string sidecar = audioFilePath + ".beatnet.json";
    std::ifstream in(sidecar);
    if (!in) {
        m_lastError = "BeatNet sidecar not found: " + sidecar;
        return grid;
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    std::string content = ss.str();

    // Try parsing as full JSON first
    if (parseJsonOutput(content, grid)) {
        return grid;
    }

    // Fall back to simple array format [0.5, 1.0, 1.5]
    std::regex numRe(R"([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?)");
    auto begin = std::sregex_iterator(content.begin(), content.end(), numRe);
    auto end = std::sregex_iterator();
    std::vector<double> beats;

    for (auto it = begin; it != end; ++it) {
        try {
            double val = std::stod(it->str());
            beats.push_back(val);
        } catch (...) {
            // Ignore parse errors
        }
    }

    if (!beats.empty()) {
        grid.setBeats(beats);
    } else {
        m_lastError = "No beats parsed from sidecar: " + sidecar;
    }

    return grid;
}

BeatGrid BeatNetBridge::analyzeWithPython(const std::string& audioFilePath, BeatNetProgressCallback progressCallback) {
    BeatGrid grid;

#ifndef ENABLE_BEATNET_PYTHON
    m_lastError = "Python subprocess disabled at compile time";
    return grid;
#else
    if (!isPythonEnabled()) {
        m_lastError = "Python subprocess disabled at runtime";
        return grid;
    }

    if (progressCallback) {
        progressCallback(0.0f, "Starting BeatNet analysis...");
    }

    int exitCode = 0;
    std::string output = runPythonProcess(audioFilePath, exitCode);

    if (exitCode != 0) {
        m_lastError = "BeatNet Python script failed with exit code " + std::to_string(exitCode);
        if (!output.empty()) {
            m_lastError += ": " + output;
        }
        return grid;
    }

    if (progressCallback) {
        progressCallback(0.5f, "Parsing BeatNet output...");
    }

    if (!parseJsonOutput(output, grid)) {
        m_lastError = "Failed to parse BeatNet JSON output";
        if (m_config.verbose) {
            m_lastError += ": " + output;
        }
        return grid;
    }

    if (progressCallback) {
        progressCallback(1.0f, "Analysis complete");
    }

    return grid;
#endif // ENABLE_BEATNET_PYTHON
}

BeatGrid BeatNetBridge::analyze(const std::string& audioFilePath) {
    return analyze(audioFilePath, nullptr);
}

BeatGrid BeatNetBridge::analyze(const std::string& audioFilePath, BeatNetProgressCallback progressCallback) {
    m_lastError.clear();
    BeatGrid grid;

    switch (m_config.mode) {
        case BeatNetMode::SidecarOnly:
            // Only use sidecar files (CI/deterministic mode)
            return analyzeWithSidecar(audioFilePath);

        case BeatNetMode::PythonOptIn:
            // Try Python first (if enabled), fall back to sidecar
            if (isPythonAvailable()) {
                grid = analyzeWithPython(audioFilePath, progressCallback);
                if (!grid.isEmpty()) {
                    return grid;
                }
                // Python failed, try sidecar
                if (m_config.verbose) {
                    // Preserve Python error but try sidecar
                    std::string pythonError = m_lastError;
                    grid = analyzeWithSidecar(audioFilePath);
                    if (grid.isEmpty()) {
                        m_lastError = "Python failed: " + pythonError + "; Sidecar failed: " + m_lastError;
                    }
                    return grid;
                }
            }
            // Python not available or not enabled, try sidecar
            return analyzeWithSidecar(audioFilePath);

        case BeatNetMode::PythonRequired:
            // Require Python, fail if not available or not enabled
            if (!isPythonEnabled()) {
                m_lastError = "Python is required but not enabled (set BEATSYNC_ENABLE_PYTHON=1 or call setPythonEnabled(true))";
                return grid;
            }
            if (!isPythonAvailable()) {
                m_lastError = "Python is required but not available";
                return grid;
            }
            return analyzeWithPython(audioFilePath, progressCallback);
    }

    return grid;
}

} // namespace BeatSync
