#include "OnnxBeatDetector.h"
#include <fstream>
#include <regex>

namespace BeatSync {

BeatGrid OnnxBeatDetector::analyze(const std::string& audioFilePath) {
    BeatGrid grid;

    // Try ONNX sidecar first: <audioFilePath>.onnx.json
    std::string onnxSidecar = audioFilePath + ".onnx.json";
    std::ifstream in(onnxSidecar);
    if (!in) {
        // Fall back to BeatNet sidecar
        std::string beatnetSidecar = audioFilePath + ".beatnet.json";
        in.open(beatnetSidecar);
        if (!in) {
            return grid; // empty
        }
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    std::string content = ss.str();

    // Extract beats as numbers, same as BeatNet parsing
    std::regex numRe(R"([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?)");
    auto begin = std::sregex_iterator(content.begin(), content.end(), numRe);
    auto end = std::sregex_iterator();
    std::vector<double> beats;

    for (auto it = begin; it != end; ++it) {
        try {
            double v = std::stod(it->str());
            beats.push_back(v);
        } catch (...) {
            // ignore
        }
    }

    if (!beats.empty()) grid.setBeats(beats);

    return grid;
}

} // namespace BeatSync