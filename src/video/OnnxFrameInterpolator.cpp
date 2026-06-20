#include "OnnxFrameInterpolator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <array>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace BeatSync {

struct OnnxFrameInterpolator::Impl {
    std::string lastError;

#ifdef USE_ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;

    // Cached I/O node names (model is img0, img1, timestep -> output).
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

    Impl() {
        try {
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BeatSyncRife");
        } catch (const std::exception& e) {
            lastError = std::string("Failed to initialize ONNX Runtime: ") + e.what();
        }
    }

    ~Impl() {
        try {
            if (session) session.reset();
            if (sessionOptions) sessionOptions.reset();
            if (env) env.reset();
        } catch (...) {
            // Best-effort cleanup.
        }
    }
#else
    Impl() = default;
    ~Impl() = default;
#endif
};

OnnxFrameInterpolator::OnnxFrameInterpolator() : m_impl(std::make_unique<Impl>()) {}
OnnxFrameInterpolator::~OnnxFrameInterpolator() = default;

bool OnnxFrameInterpolator::isAvailable() {
#ifdef USE_ONNX
    return true;
#else
    return false;
#endif
}

std::string OnnxFrameInterpolator::getLastError() const {
    return m_impl ? m_impl->lastError : "interpolator not constructed";
}

bool OnnxFrameInterpolator::isLoaded() const {
#ifdef USE_ONNX
    return m_impl && m_impl->session != nullptr;
#else
    return false;
#endif
}

bool OnnxFrameInterpolator::loadModel(const std::string& modelPath, bool useGPU, int gpuDeviceId) {
#ifndef USE_ONNX
    (void)modelPath; (void)useGPU; (void)gpuDeviceId;
    m_impl->lastError = "ONNX Runtime not available. Rebuild with USE_ONNX=ON";
    return false;
#else
    if (!m_impl->env) {
        m_impl->lastError = "ONNX Runtime environment not initialized";
        return false;
    }
    try {
        m_impl->sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_impl->sessionOptions->SetIntraOpNumThreads(0);
        m_impl->sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::string activeProvider = "CPU";
        if (useGPU) {
            const OrtApi& ortApi = Ort::GetApi();

            // CUDA execution provider (primary GPU path for RIFE; per-resolution
            // TensorRT engine builds aren't worth it for short variable clips).
            try {
                OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
                OrtStatus* status = ortApi.CreateCUDAProviderOptions(&cudaOptions);
                if (status == nullptr && cudaOptions != nullptr) {
                    const char* keys[] = {"device_id", "arena_extend_strategy"};
                    char deviceIdStr[16];
                    snprintf(deviceIdStr, sizeof(deviceIdStr), "%d", gpuDeviceId);
                    const char* values[] = {deviceIdStr, "kSameAsRequested"};
                    status = ortApi.UpdateCUDAProviderOptions(cudaOptions, keys, values, 2);
                    if (status == nullptr) {
                        status = ortApi.SessionOptionsAppendExecutionProvider_CUDA_V2(
                            static_cast<OrtSessionOptions*>(*m_impl->sessionOptions), cudaOptions);
                        if (status == nullptr) {
                            activeProvider = "CUDA";
                            std::cerr << "[BeatSync] RIFE: CUDA execution provider enabled" << std::endl;
                        } else {
                            const char* msg = ortApi.GetErrorMessage(status);
                            std::cerr << "[BeatSync] RIFE: CUDA append failed: " << (msg ? msg : "?") << std::endl;
                            ortApi.ReleaseStatus(status);
                        }
                    } else {
                        ortApi.ReleaseStatus(status);
                    }
                    ortApi.ReleaseCUDAProviderOptions(cudaOptions);
                } else if (status != nullptr) {
                    ortApi.ReleaseStatus(status);
                }
            } catch (...) {
                std::cerr << "[BeatSync] RIFE: CUDA provider exception" << std::endl;
            }

#ifdef _WIN32
            if (activeProvider == "CPU") {
                try {
                    m_impl->sessionOptions->AppendExecutionProvider("DML", {});
                    activeProvider = "DirectML";
                    std::cerr << "[BeatSync] RIFE: DirectML execution provider enabled" << std::endl;
                } catch (...) {
                    std::cerr << "[BeatSync] RIFE: DirectML fallback failed" << std::endl;
                }
            }
#endif
        }
        std::cerr << "[BeatSync] RIFE final execution provider: " << activeProvider << std::endl;

#ifdef _WIN32
        int wlen = MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, NULL, 0);
        if (wlen <= 0) {
            m_impl->lastError = "Failed to convert model path to wide string";
            return false;
        }
        std::wstring widePath(static_cast<size_t>(wlen), 0);
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, &widePath[0], wlen);
        widePath.resize(static_cast<size_t>(wlen) - 1);
        m_impl->session = std::make_unique<Ort::Session>(*m_impl->env, widePath.c_str(), *m_impl->sessionOptions);
#else
        m_impl->session = std::make_unique<Ort::Session>(*m_impl->env, modelPath.c_str(), *m_impl->sessionOptions);
#endif

        // Cache I/O names in model order.
        Ort::AllocatorWithDefaultOptions allocator;
        m_impl->inputNames.clear();
        size_t numInputs = m_impl->session->GetInputCount();
        for (size_t i = 0; i < numInputs; ++i) {
            auto name = m_impl->session->GetInputNameAllocated(i, allocator);
            m_impl->inputNames.emplace_back(name.get());
        }
        m_impl->outputNames.clear();
        size_t numOutputs = m_impl->session->GetOutputCount();
        for (size_t i = 0; i < numOutputs; ++i) {
            auto name = m_impl->session->GetOutputNameAllocated(i, allocator);
            m_impl->outputNames.emplace_back(name.get());
        }

        if (m_impl->inputNames.size() < 3 || m_impl->outputNames.empty()) {
            m_impl->lastError = "Unexpected RIFE model signature (need img0, img1, timestep inputs)";
            m_impl->session.reset();
            return false;
        }
        return true;
    } catch (const Ort::Exception& e) {
        m_impl->lastError = std::string("ONNX load failed: ") + e.what();
        m_impl->session.reset();
        return false;
    } catch (const std::exception& e) {
        m_impl->lastError = std::string("load failed: ") + e.what();
        m_impl->session.reset();
        return false;
    }
#endif
}

bool OnnxFrameInterpolator::interpolate(const uint8_t* rgb0, const uint8_t* rgb1,
                                        int width, int height, float t,
                                        std::vector<uint8_t>& outRgb) {
#ifndef USE_ONNX
    (void)rgb0; (void)rgb1; (void)width; (void)height; (void)t; (void)outRgb;
    m_impl->lastError = "ONNX Runtime not available";
    return false;
#else
    if (!isLoaded()) {
        m_impl->lastError = "No RIFE model loaded";
        return false;
    }
    if (!rgb0 || !rgb1 || width <= 0 || height <= 0) {
        m_impl->lastError = "Invalid interpolate arguments";
        return false;
    }
    try {
        // This RIFE export handles arbitrary spatial dims internally and returns
        // output at the input size (verified for non-/32 sizes), so feed native
        // width/height directly. (W2/H2 retained for generality; pad == 1 here.)
        const int pad = 1;
        const int W2 = ((width + pad - 1) / pad) * pad;
        const int H2 = ((height + pad - 1) / pad) * pad;
        const size_t plane = static_cast<size_t>(W2) * static_cast<size_t>(H2);

        std::vector<float> in0(3 * plane), in1(3 * plane);
        auto fill = [&](const uint8_t* src, std::vector<float>& dst) {
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < H2; ++y) {
                    const int sy = (y < height) ? y : (height - 1);
                    for (int x = 0; x < W2; ++x) {
                        const int sx = (x < width) ? x : (width - 1);
                        dst[static_cast<size_t>(c) * plane + static_cast<size_t>(y) * W2 + x] =
                            src[(static_cast<size_t>(sy) * width + sx) * 3 + c] / 255.0f;
                    }
                }
            }
        };
        fill(rgb0, in0);
        fill(rgb1, in1);

        float tval = std::min(1.0f, std::max(0.0f, t));

        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> imgShape{1, 3, H2, W2};
        std::array<int64_t, 1> tShape{1};

        std::array<Ort::Value, 3> inputs{
            Ort::Value::CreateTensor<float>(memInfo, in0.data(), in0.size(), imgShape.data(), imgShape.size()),
            Ort::Value::CreateTensor<float>(memInfo, in1.data(), in1.size(), imgShape.data(), imgShape.size()),
            Ort::Value::CreateTensor<float>(memInfo, &tval, 1, tShape.data(), tShape.size())
        };

        std::array<const char*, 3> inNames{
            m_impl->inputNames[0].c_str(),
            m_impl->inputNames[1].c_str(),
            m_impl->inputNames[2].c_str()
        };
        std::array<const char*, 1> outNames{ m_impl->outputNames[0].c_str() };

        auto outputs = m_impl->session->Run(Ort::RunOptions{nullptr},
                                            inNames.data(), inputs.data(), inputs.size(),
                                            outNames.data(), outNames.size());
        if (outputs.empty()) {
            m_impl->lastError = "RIFE produced no output";
            return false;
        }

        const float* od = outputs[0].GetTensorData<float>();
        auto oshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        // Expect [1,3,Ho,Wo] (Ho/Wo >= height/width). Crop top-left.
        if (oshape.size() != 4) {
            m_impl->lastError = "Unexpected RIFE output rank";
            return false;
        }
        const int Ho = static_cast<int>(oshape[2]);
        const int Wo = static_cast<int>(oshape[3]);
        if (Ho < height || Wo < width) {
            m_impl->lastError = "RIFE output smaller than input";
            return false;
        }
        const size_t oplane = static_cast<size_t>(Ho) * static_cast<size_t>(Wo);

        outRgb.resize(static_cast<size_t>(width) * height * 3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    float v = od[static_cast<size_t>(c) * oplane + static_cast<size_t>(y) * Wo + x];
                    v = std::min(1.0f, std::max(0.0f, v));
                    outRgb[(static_cast<size_t>(y) * width + x) * 3 + c] =
                        static_cast<uint8_t>(std::lround(v * 255.0f));
                }
            }
        }
        return true;
    } catch (const Ort::Exception& e) {
        m_impl->lastError = std::string("RIFE inference failed: ") + e.what();
        return false;
    } catch (const std::exception& e) {
        m_impl->lastError = std::string("RIFE inference error: ") + e.what();
        return false;
    }
#endif
}

} // namespace BeatSync
