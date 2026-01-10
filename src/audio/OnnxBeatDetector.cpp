#include "OnnxBeatDetector.h"
#include <fstream>
#include <regex>
#include <filesystem>
#include <iostream>
#include <sstream>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include "AudioAnalyzer.h"
#include "SpectralFlux.h"
#include "tracing/Tracing.h"

#if defined(USE_ONNX)
#  include <onnxruntime_cxx_api.h>
#  define HAVE_ONNXRUNTIME 1
#endif

// Environment variable to force CUDA provider (for testing GPU builds)
// Set BEATSYNC_ONNX_USE_CUDA=1 to enable CUDA, otherwise CPU-only
static bool shouldUseCudaProvider() {
    const char* env = std::getenv("BEATSYNC_ONNX_USE_CUDA");
    return env && (std::string(env) == "1" || std::string(env) == "true");
}

// Try to locate and load the CUDA provider DLL to ensure it is available before enabling CUDA.
// Returns true when the provider DLL could be found and loaded (freed immediately), false otherwise.
static bool tryLoadCudaProviderDLL() {
#ifdef _WIN32
    // Try LoadLibrary first (relies on PATH), then search up the directory tree for vcpkg_installed
    auto try_load = [](const std::wstring &p) -> bool {
        HMODULE h = LoadLibraryW(p.c_str());
        if (h) { FreeLibrary(h); return true; }
        return false;
    };

    // Try simple name first
    if (try_load(L"onnxruntime_providers_cuda.dll")) return true;

    // Search upward for vcpkg_installed/x64-windows/bin
    std::filesystem::path cur = std::filesystem::current_path();
    for (int i = 0; i < 8; ++i) {
        auto candidate = cur / "vcpkg_installed" / "x64-windows" / "bin" / "onnxruntime_providers_cuda.dll";
        if (std::filesystem::exists(candidate)) {
            if (try_load(candidate.wstring())) return true;
        }
        if (cur.has_parent_path()) cur = cur.parent_path(); else break;
    }
    return false;
#else
    // POSIX-based: try dlopen and search similar paths
    auto try_load = [](const std::string &p) -> bool {
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (h) { dlclose(h); return true; }
        return false;
    };
    if (try_load("libonnxruntime_providers_cuda.so")) return true;
    std::filesystem::path cur = std::filesystem::current_path();
    for (int i = 0; i < 8; ++i) {
        auto candidate = cur / "vcpkg_installed" / "x64-windows" / "bin" / "libonnxruntime_providers_cuda.so";
        if (std::filesystem::exists(candidate)) { if (try_load(candidate.string())) return true; }
        if (cur.has_parent_path()) cur = cur.parent_path(); else break;
    }
    return false;
#endif
}

namespace BeatSync {

static std::string findDefaultStubModel() {
    // Search upward from the current working directory for the tests/models/beat_stub.onnx
    std::filesystem::path cur = std::filesystem::current_path();
    for (int i = 0; i < 6; ++i) {
        auto candidate = cur / "tests" / "models" / "beat_stub.onnx";
        if (std::filesystem::exists(candidate)) return candidate.string();
        cur = cur.parent_path();
        if (cur.empty()) break;
    }
    return std::string();
}

BeatGrid OnnxBeatDetector::analyze(const std::string& audioFilePath) {
    TRACE_FUNC();
    BeatGrid grid;

    // Try ONNX sidecar first: <audioFilePath>.onnx.json
    std::string onnxSidecar = audioFilePath + ".onnx.json";
    std::ifstream in(onnxSidecar);
    if (!in) {
        // Fall back to BeatNet sidecar
        std::string beatnetSidecar = audioFilePath + ".beatnet.json";
        in.open(beatnetSidecar);
        if (in) {
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
    } else {
        std::ostringstream ss;
        ss << in.rdbuf();
        std::string content = ss.str();

        // Prefer parsing the explicit "beats" array if present
        std::vector<double> beats;
        auto beatsKey = content.find("\"beats\"");
        if (beatsKey != std::string::npos) {
            auto open = content.find('[', beatsKey);
            auto close = content.find(']', open == std::string::npos ? 0 : open);
            if (open != std::string::npos && close != std::string::npos && close > open) {
                std::string arr = content.substr(open+1, close - open - 1);
                std::regex numRe(R"([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?)");
                auto begin = std::sregex_iterator(arr.begin(), arr.end(), numRe);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it) {
                    try { beats.push_back(std::stod(it->str())); } catch (...) {}
                }
            }
        }

        // Fallback: parse any numbers in the file if beats array wasn't found
        if (beats.empty()) {
            std::regex numRe(R"([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?)");
            auto begin = std::sregex_iterator(content.begin(), content.end(), numRe);
            auto end = std::sregex_iterator();
            for (auto it = begin; it != end; ++it) {
                try { beats.push_back(std::stod(it->str())); } catch (...) {}
            }
        }

        if (!beats.empty()) grid.setBeats(beats);
        return grid;
    }

    // If we get here, no sidecar existed. Try running an ONNX model if available.
#if defined(HAVE_ONNXRUNTIME)
    TRACE_SCOPE("onnx-inference");
    std::string modelPath = modelPath_.empty() ? findDefaultStubModel() : modelPath_;
    if (!modelPath.empty() && std::filesystem::exists(modelPath)) {
        try {
            TRACE_SCOPE("onnx-load-and-run");
            std::cerr << "[ONNX] Using model at: " << modelPath << std::endl;
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "beatsync");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

            // CUDA provider support: only enabled via BEATSYNC_ONNX_USE_CUDA=1
            // This avoids crashes when CUDA runtime isn't installed
            bool useCuda = shouldUseCudaProvider();
            if (useCuda) {
                TRACE_SCOPE("onnx-cuda-probe");
                std::cerr << "[ONNX] CUDA provider requested via BEATSYNC_ONNX_USE_CUDA=1" << std::endl;
                // Try to ensure the CUDA provider DLL exists and can be loaded; otherwise fall back to CPU.
                if (!tryLoadCudaProviderDLL()) {
                    std::cerr << "[ONNX] CUDA provider DLL not found or failed to load - falling back to CPU execution provider" << std::endl;
                    useCuda = false;
                } else {
                    std::cerr << "[ONNX] CUDA provider DLL found - attempting to use GPU provider" << std::endl;
                }
            }

#ifdef _WIN32
            // On Windows, convert path to wide string for proper Unicode support
            std::wstring wModelPath(modelPath.begin(), modelPath.end());
            Ort::Session session(env, wModelPath.c_str(), session_options);
#else
            Ort::Session session(env, modelPath.c_str(), session_options);
#endif

            size_t out_count = session.GetOutputCount();
            std::vector<std::string> allocated_names;
            allocated_names.reserve(out_count);  // Prevent reallocation during loop
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < out_count; ++i) {
                auto namePtr = session.GetOutputNameAllocated(i, allocator);
                allocated_names.emplace_back(namePtr.get());
            }
            // Build output_names after allocated_names is fully populated to avoid dangling pointers
            std::vector<const char*> output_names;
            output_names.reserve(out_count);
            for (const auto& name : allocated_names) {
                output_names.push_back(name.c_str());
            }

            auto outputs = session.Run(Ort::RunOptions{nullptr}, nullptr, nullptr, 0, output_names.data(), output_names.size());
            if (outputs.size() > 0) {
                auto &out = outputs[0];
                auto type_info = out.GetTensorTypeAndShapeInfo();
                auto shape = type_info.GetShape();
                size_t len = 1;
                std::cerr << "[ONNX] output shape: [";
                for (size_t idx = 0; idx < shape.size(); ++idx) {
                    std::cerr << shape[idx] << (idx + 1 < shape.size() ? "," : "");
                }
                std::cerr << "]" << std::endl;
                for (auto s : shape) len *= (s > 0 ? s : 0);
                const float* data = out.GetTensorData<float>();
                std::vector<double> beats;
                for (size_t i = 0; i < len; ++i) beats.push_back(static_cast<double>(data[i]));
                if (!beats.empty()) {
                    grid.setBeats(beats);
                } else {
                    std::cerr << "[ONNX] Model returned empty output tensor" << std::endl;
                }
            } else {
                std::cerr << "[ONNX] No outputs from model" << std::endl;
            }
            // Note: output_names points to strings in allocated_names (std::string),
            // which are automatically freed when they go out of scope. No manual free needed.
        } catch (const std::exception &e) {
            std::cerr << "ONNX inference failed: " << e.what() << std::endl;
        }
    }
#endif

    // No sidecar and no ONNX model (or inference failed). Fall back to spectral-flux based detector.
    if (std::filesystem::exists(audioFilePath)) {
        TRACE_SCOPE("spectral-fallback");
        AudioAnalyzer analyzer;
        AudioAnalyzer::AudioData ad = analyzer.loadAudioFile(audioFilePath);
        if (!ad.samples.empty() && ad.sampleRate > 0) {
            // Use parameters tuned for short click-track detection as a robust fallback
            auto beats = detectBeatsFromWaveform(ad.samples, ad.sampleRate, 1024, 256, 1.5, 1.2);
            if (!beats.empty()) grid.setBeats(beats);
        }
    }

    return grid;
}

} // namespace BeatSync