#include "TransitionLibrary.h"
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <iomanip>

namespace BeatSync {

bool TransitionLibrary::loadFromDirectory(const std::string& dirPath) {
    m_transitions.clear();
    m_lastError.clear();

    try {
        std::filesystem::path p(dirPath);
        if (!std::filesystem::exists(p) || !std::filesystem::is_directory(p)) {
            m_lastError = "Transition directory not found: " + dirPath;
            return false;
        }

        std::regex nameRe(R"(^\s*//\s*Name:\s*(.+)\s*$)", std::regex::icase);
        std::regex catRe(R"(^\s*//\s*Category:\s*(.+)\s*$)", std::regex::icase);

        for (auto& entry : std::filesystem::directory_iterator(p)) {
            if (!entry.is_regular_file()) continue;
            auto path = entry.path();
            if (path.extension() != ".glsl") continue;

            TransitionShader ts;
            ts.glslPath = path.string();
            ts.name = path.stem().string(); // default to filename without extension
            ts.category = "unknown";

            // Read first 64 lines to find metadata comments
            std::ifstream in(path);
            if (in) {
                std::string line;
                int lines = 0;
                while (std::getline(in, line) && lines++ < 64) {
                    std::smatch m;
                    if (std::regex_match(line, m, nameRe) && m.size() > 1) {
                        ts.name = m[1].str();
                        // trim whitespace
                        while (!ts.name.empty() && std::isspace((unsigned char)ts.name.front())) ts.name.erase(ts.name.begin());
                        while (!ts.name.empty() && std::isspace((unsigned char)ts.name.back())) ts.name.pop_back();
                    }
                    if (std::regex_match(line, m, catRe) && m.size() > 1) {
                        ts.category = m[1].str();
                        while (!ts.category.empty() && std::isspace((unsigned char)ts.category.front())) ts.category.erase(ts.category.begin());
                        while (!ts.category.empty() && std::isspace((unsigned char)ts.category.back())) ts.category.pop_back();
                    }
                }
            }

            m_transitions.push_back(std::move(ts));
        }

        if (m_transitions.empty()) {
            m_lastError = "No .glsl transition shaders found in: " + dirPath;
            return false;
        }

    } catch (const std::exception& ex) {
        m_lastError = std::string("Exception while scanning transitions: ") + ex.what();
        return false;
    }

    return true;
}

const TransitionShader* TransitionLibrary::findByName(const std::string& name) const {
    for (const auto& t : m_transitions) {
        if (t.name == name || t.glslPath == name) return &t;
    }
    return nullptr;
}

std::string TransitionLibrary::buildGlTransitionFilter(const std::string& name, double duration) const {
    const TransitionShader* t = findByName(name);
    if (!t) return "";

    // Use explicit source file path so custom transitions are found
    // Ensure single quotes around path and escape single quotes inside (unlikely)
    std::string src = t->glslPath;
    std::string escaped;
    escaped.reserve(src.size());
    for (char c : src) {
        if (c == '\'') escaped += "\\'"; else escaped += c;
    }

    std::ostringstream oss;
    oss << "gltransition=source='" << escaped << "':duration=" << std::fixed << std::setprecision(3) << duration;
    return oss.str();
}

} // namespace BeatSync
