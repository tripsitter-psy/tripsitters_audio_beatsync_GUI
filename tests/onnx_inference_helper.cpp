#include <iostream>
#include <filesystem>
#include "audio/OnnxBeatDetector.h"

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    fs::path test_src = __FILE__;
    fs::path repo_root = test_src.parent_path().parent_path();
    fs::path model_path = repo_root / "tests" / "models" / "beat_stub.onnx";

    if (!fs::exists(model_path)) {
        std::cerr << "Model not found: " << model_path << std::endl;
        return 2;
    }

    BeatSync::OnnxBeatDetector d;
    d.setOnnxModelPath(model_path.string());

    const int iterations = 200;
    for (int i = 0; i < iterations; ++i) {
        auto grid = d.analyze("tests/fixtures/nonexistent_audio_for_model");
        if (grid.isEmpty()) {
            std::cerr << "Iteration " << i << ": empty result" << std::endl;
            return 3;
        }
        if (grid.getNumBeats() != 3) {
            std::cerr << "Iteration " << i << ": unexpected beat count = " << grid.getNumBeats() << std::endl;
            return 4;
        }
        if (std::abs(grid.getBeatAt(0) - 0.5) > 1e-6) {
            std::cerr << "Iteration " << i << ": unexpected first beat = " << grid.getBeatAt(0) << std::endl;
            return 5;
        }
        if ((i+1) % 50 == 0) std::cerr << "Completed " << (i+1) << " iterations" << std::endl;
    }

    std::cout << "SUCCESS: all " << iterations << " iterations returned expected results" << std::endl;
    return 0;
}
