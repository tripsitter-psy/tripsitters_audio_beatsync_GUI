#include "BeatNetBridge.h"
#include <cstdio>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#endif

#include "../utils/ProcessUtils.h"

namespace BeatSync {

void BeatNetBridge::setPythonPath(const std::string& pythonPath) {
    m_pythonPath = pythonPath;
}

BeatGrid BeatNetBridge::analyze(const std::string& audioFilePath) {
    BeatGrid grid;
    m_lastError.clear();

    // First, look for bundled executable (no Python needed)
    std::string exeTool;
    std::vector<std::string> exeSearchPaths;
    
#ifdef _WIN32
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        exeSearchPaths.push_back((exeDir / "beatnet_analyze.exe").string());
        exeSearchPaths.push_back((exeDir / "ai_tools" / "beatnet_analyze.exe").string());
        exeSearchPaths.push_back((exeDir.parent_path() / "ai_tools" / "beatnet_analyze.exe").string());
    }
    exeSearchPaths.push_back("beatnet_analyze.exe");
    exeSearchPaths.push_back("scripts/dist/beatnet_analyze.exe");
#endif
    
    for (const auto& ep : exeSearchPaths) {
        if (std::filesystem::exists(ep)) {
            exeTool = std::filesystem::absolute(ep).string();
            break;
        }
    }
    
    // Build command - either bundled exe or Python script
    std::ostringstream cmd;
    
    if (!exeTool.empty()) {
        // Use bundled executable (no Python dependency)
        std::cout << "Using bundled BeatNet executable: " << exeTool << std::endl;
        cmd << "\"" << exeTool << "\" \"" << audioFilePath << "\"";
    } else {
        // Fall back to Python script
        std::string python = m_pythonPath.empty() ? "python" : m_pythonPath;
        
        // Find the script path
        std::string scriptPath;
        
        std::vector<std::string> searchPaths = {
            "scripts/beatnet_analyze.py",
            "../scripts/beatnet_analyze.py",
            "../../scripts/beatnet_analyze.py",
        };
        
#ifdef _WIN32
        if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
            std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
            searchPaths.push_back((exeDir / "scripts" / "beatnet_analyze.py").string());
            searchPaths.push_back((exeDir.parent_path() / "scripts" / "beatnet_analyze.py").string());
        }
#endif
        
        for (const auto& sp : searchPaths) {
            if (std::filesystem::exists(sp)) {
                scriptPath = std::filesystem::absolute(sp).string();
                break;
            }
        }
        
        if (scriptPath.empty()) {
            m_lastError = "Could not find beatnet_analyze.exe or beatnet_analyze.py - AI beat detection not available. "
                          "Install Python with: pip install torch torchaudio BeatNet numpy";
            return grid;
        }
        
        std::cout << "Using Python script: " << scriptPath << std::endl;
        cmd << "\"" << python << "\" \"" << scriptPath << "\" \"" << audioFilePath << "\"";
    }
    
    std::cout << "Running BeatNet: " << cmd.str() << std::endl;
    
    // Execute command
    std::string output;
    int exitCode;
    
#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), output);
#else
    FILE* pipe = popen((cmd.str() + " 2>&1").c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute BeatNet script";
        return grid;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    exitCode = pclose(pipe);
#endif
    
    if (exitCode != 0 && output.find("\"beats\"") == std::string::npos) {
        m_lastError = "BeatNet script failed with exit code " + std::to_string(exitCode);
        if (!output.empty()) {
            m_lastError += ": " + output;
        }
        return grid;
    }
    
    // Parse JSON output - extract beats array and bpm
    // Format: {"beats": [0.5, 1.0, 1.5, ...], "bpm": 120.0, ...}
    
    // Extract beats array
    std::vector<double> beats;
    size_t beatsStart = output.find("\"beats\": [");
    if (beatsStart != std::string::npos) {
        beatsStart += 10; // skip past "beats": [
        size_t beatsEnd = output.find("]", beatsStart);
        if (beatsEnd != std::string::npos) {
            std::string beatsStr = output.substr(beatsStart, beatsEnd - beatsStart);
            // Parse comma-separated doubles
            std::istringstream iss(beatsStr);
            std::string token;
            while (std::getline(iss, token, ',')) {
                try {
                    // Remove whitespace
                    size_t start = token.find_first_not_of(" \t\n\r");
                    size_t end = token.find_last_not_of(" \t\n\r");
                    if (start != std::string::npos && end != std::string::npos) {
                        double t = std::stod(token.substr(start, end - start + 1));
                        beats.push_back(t);
                    }
                } catch (...) {
                    // Skip invalid tokens
                }
            }
        }
    }
    
    // Extract BPM
    double bpm = 0.0;
    size_t bpmStart = output.find("\"bpm\": ");
    if (bpmStart != std::string::npos) {
        bpmStart += 7;
        try {
            bpm = std::stod(output.substr(bpmStart));
        } catch (...) {
            // BPM parsing failed, will estimate from beats
        }
    }
    
    // Check for error in output
    if (output.find("\"error\"") != std::string::npos) {
        size_t errStart = output.find("\"error\": \"");
        if (errStart != std::string::npos) {
            errStart += 10;
            size_t errEnd = output.find("\"", errStart);
            if (errEnd != std::string::npos) {
                m_lastError = output.substr(errStart, errEnd - errStart);
            }
        }
    }
    
    if (beats.empty()) {
        if (m_lastError.empty()) {
            m_lastError = "No beats detected by BeatNet";
        }
        return grid;
    }
    
    // Populate grid
    grid.setBeats(beats);
    if (bpm > 0) {
        grid.setBPM(bpm);
    }
    
    std::cout << "BeatNet detected " << beats.size() << " beats, BPM: " << bpm << std::endl;
    
    return grid;
}

} // namespace BeatSync
