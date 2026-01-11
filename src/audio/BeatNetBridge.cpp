#include "BeatNetBridge.h"
#include <fstream>
#include <regex>
#include <sstream>

namespace BeatSync {

void BeatNetBridge::setPythonPath(const std::string& pythonPath) {
    m_pythonPath = pythonPath;
}

// Minimal implementation: look for a sidecar JSON file named "<audioFilePath>.beatnet.json"
// with an array of beat timestamps (seconds). This avoids invoking Python in CI/tests
// while enabling integration via real BeatNet later.
BeatGrid BeatNetBridge::analyze(const std::string& audioFilePath) {
    BeatGrid grid;
    m_lastError.clear();

    std::string sidecar = audioFilePath + ".beatnet.json";
    std::ifstream in(sidecar);
    if (!in) {
        m_lastError = "BeatNet sidecar not found: " + sidecar;
        return grid; // empty grid
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    std::string content = ss.str();

    // Very small JSON parser: extract floats from array like [0.5, 1.0, 1.5]
    std::regex num_re(R"(([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?))");
    auto begin = std::sregex_iterator(content.begin(), content.end(), num_re);
    auto end = std::sregex_iterator();
    std::vector<double> beats;
    for (auto it = begin; it != end; ++it) {
        std::smatch m = *it;
        try {
            double v = std::stod(m.str());
            beats.push_back(v);
        } catch (...) {
            // ignore parse errors
        }
    }

    if (!beats.empty()) {
        grid.setBeats(beats);
        // Optionally compute BPM from average interval in BeatGrid implementation
    } else {
        m_lastError = "No beats parsed from sidecar: " + sidecar;
    }

    return grid;
}

} // namespace BeatSync
