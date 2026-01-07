#include "StemSeparator.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "../utils/ProcessUtils.h"

namespace BeatSync {

void StemSeparator::setPythonPath(const std::string& pythonPath) {
    m_pythonPath = pythonPath;
}

StemPaths StemSeparator::separate(const std::string& audioFilePath, const std::string& outputDir) {
    StemPaths paths;
    m_lastError.clear();

    // First, look for bundled executable (no Python needed)
    std::string exeTool;
    std::vector<std::string> exeSearchPaths;
    
#ifdef _WIN32
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        exeSearchPaths.push_back((exeDir / "demucs_separate.exe").string());
        exeSearchPaths.push_back((exeDir / "ai_tools" / "demucs_separate.exe").string());
        exeSearchPaths.push_back((exeDir.parent_path() / "ai_tools" / "demucs_separate.exe").string());
    }
    exeSearchPaths.push_back("demucs_separate.exe");
    exeSearchPaths.push_back("scripts/dist/demucs_separate.exe");
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
        std::cout << "Using bundled Demucs executable: " << exeTool << std::endl;
        cmd << "\"" << exeTool << "\" \"" << audioFilePath << "\" \"" << outputDir << "\"";
    } else {
        // Fall back to Python script
        std::string python = m_pythonPath.empty() ? "python" : m_pythonPath;
        
        // Find the script path
        std::string scriptPath;
        
        std::vector<std::string> searchPaths = {
            "scripts/demucs_separate.py",
            "../scripts/demucs_separate.py",
            "../../scripts/demucs_separate.py",
        };
        
#ifdef _WIN32
        if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
            std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
            searchPaths.push_back((exeDir / "scripts" / "demucs_separate.py").string());
            searchPaths.push_back((exeDir.parent_path() / "scripts" / "demucs_separate.py").string());
        }
#endif
        
        for (const auto& sp : searchPaths) {
            if (std::filesystem::exists(sp)) {
                scriptPath = std::filesystem::absolute(sp).string();
                break;
            }
        }
        
        if (scriptPath.empty()) {
            m_lastError = "Could not find demucs_separate.exe or demucs_separate.py - Stem separation not available. "
                          "Install Python with: pip install torch torchaudio demucs numpy";
            return paths;
        }
        
        std::cout << "Using Python script: " << scriptPath << std::endl;
        cmd << "\"" << python << "\" \"" << scriptPath << "\" \"" << audioFilePath << "\" \"" << outputDir << "\"";
    }
    
    std::cout << "Running Demucs: " << cmd.str() << std::endl;
    
    // Execute command
    std::string output;
    int exitCode;
    
#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), output);
#else
    // Unix: use popen
    FILE* pipe = popen((cmd.str() + " 2>&1").c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute Demucs script";
        return paths;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    exitCode = pclose(pipe);
#endif
    
    if (exitCode != 0) {
        m_lastError = "Demucs script failed with exit code " + std::to_string(exitCode);
        if (!output.empty()) {
            m_lastError += ": " + output;
        }
        return paths;
    }
    
    // Parse JSON output - simple parsing for expected fields
    // Look for "stems": { "drums": "path", "bass": "path", ... }
    auto extractPath = [&output](const std::string& key) -> std::string {
        std::string searchKey = "\"" + key + "\": \"";
        size_t pos = output.find(searchKey);
        if (pos == std::string::npos) return "";
        pos += searchKey.length();
        size_t endPos = output.find("\"", pos);
        if (endPos == std::string::npos) return "";
        return output.substr(pos, endPos - pos);
    };
    
    paths.drums = extractPath("drums");
    paths.bass = extractPath("bass");
    paths.vocals = extractPath("vocals");
    paths.other = extractPath("other");
    
    // Check for error in output
    if (output.find("\"error\"") != std::string::npos) {
        std::string errorMsg = extractPath("error");
        if (!errorMsg.empty()) {
            m_lastError = errorMsg;
        }
    }
    
    if (paths.drums.empty() && paths.bass.empty() && paths.vocals.empty() && paths.other.empty()) {
        if (m_lastError.empty()) {
            m_lastError = "No stems produced by Demucs";
        }
    }
    
    return paths;
}

} // namespace BeatSync
