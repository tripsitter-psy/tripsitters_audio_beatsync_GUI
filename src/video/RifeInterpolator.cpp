#include "RifeInterpolator.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace BeatSync {

namespace {
constexpr int kPadMultiple = 32;  // RIFE v4 requires dimensions divisible by 32

inline int padTo(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}
} // namespace

struct RifeInterpolator::Impl {
#ifdef USE_ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    std::string inputName;
    std::string outputName;
#endif
    bool loaded = false;
    std::string lastError;
    std::string activeProvider = "None";
};

RifeInterpolator::RifeInterpolator() : m_impl(std::make_unique<Impl>()) {}
RifeInterpolator::~RifeInterpolator() = default;

bool RifeInterpolator::loadModel(const std::string& modelPath, bool useGPU, int gpuDeviceId) {
#ifndef USE_ONNX
    (void)modelPath; (void)useGPU; (void)gpuDeviceId;
    m_impl->lastError = "ONNX Runtime not available in this build";
    return false;
#else
    try {
        m_impl->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RifeInterpolator");
        m_impl->sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_impl->sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        m_impl->activeProvider = "CPU";
        if (useGPU) {
            try {
                OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
                const OrtApi& ortApi = Ort::GetApi();
                OrtStatus* status = ortApi.CreateCUDAProviderOptions(&cudaOptions);
                if (status == nullptr && cudaOptions != nullptr) {
                    const char* keys[] = {"device_id"};
                    char deviceIdStr[16];
                    snprintf(deviceIdStr, sizeof(deviceIdStr), "%d", gpuDeviceId);
                    const char* values[] = {deviceIdStr};
                    status = ortApi.UpdateCUDAProviderOptions(cudaOptions, keys, values, 1);
                    if (status == nullptr) {
                        status = ortApi.SessionOptionsAppendExecutionProvider_CUDA_V2(
                            static_cast<OrtSessionOptions*>(*m_impl->sessionOptions), cudaOptions);
                        if (status == nullptr) {
                            m_impl->activeProvider = "CUDA";
                        } else {
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
                // CUDA unavailable; stay on CPU
            }
        }

        m_impl->session = std::make_unique<Ort::Session>(*m_impl->env, modelPath.c_str(),
                                                         *m_impl->sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        if (m_impl->session->GetInputCount() != 1 || m_impl->session->GetOutputCount() != 1) {
            m_impl->lastError = "Unexpected RIFE model signature (expected 1 input, 1 output)";
            m_impl->session.reset();
            return false;
        }
        m_impl->inputName = m_impl->session->GetInputNameAllocated(0, allocator).get();
        m_impl->outputName = m_impl->session->GetOutputNameAllocated(0, allocator).get();

        auto inputShape = m_impl->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape.size() != 4 || inputShape[1] != 11) {
            m_impl->lastError = "Unsupported RIFE model: expected input [1,11,H,W] (vs-mlrt export)";
            m_impl->session.reset();
            return false;
        }

        m_impl->loaded = true;
        std::cerr << "[BeatSync] RIFE model loaded (" << m_impl->activeProvider << "): "
                  << modelPath << std::endl;
        return true;
    } catch (const std::exception& e) {
        m_impl->lastError = std::string("Failed to load RIFE model: ") + e.what();
        m_impl->loaded = false;
        return false;
    }
#endif
}

bool RifeInterpolator::isLoaded() const {
    return m_impl->loaded;
}

std::string RifeInterpolator::getLastError() const {
    return m_impl->lastError;
}

std::string RifeInterpolator::getActiveProvider() const {
    return m_impl->activeProvider;
}

bool RifeInterpolator::interpolate(const uint8_t* img0, const uint8_t* img1,
                                   int width, int height, float timestep,
                                   std::vector<uint8_t>& outFrame) {
#ifndef USE_ONNX
    (void)img0; (void)img1; (void)width; (void)height; (void)timestep; (void)outFrame;
    m_impl->lastError = "ONNX Runtime not available in this build";
    return false;
#else
    if (!m_impl->loaded || !img0 || !img1 || width <= 0 || height <= 0) {
        m_impl->lastError = "Interpolator not loaded or bad input";
        return false;
    }

    try {
        const int padW = padTo(width, kPadMultiple);
        const int padH = padTo(height, kPadMultiple);
        const size_t plane = static_cast<size_t>(padW) * padH;

        // Channel layout (vs-mlrt "rife/" models):
        //   0-2 img0 RGB, 3-5 img1 RGB, 6 timestep,
        //   7 x-meshgrid 2x/(W-1)-1, 8 y-meshgrid 2y/(H-1)-1,
        //   9 constant 2/(W-1), 10 constant 2/(H-1)
        std::vector<float> input(11 * plane);

        for (int y = 0; y < padH; ++y) {
            const int srcY = std::min(y, height - 1);  // edge-replicate padding
            for (int x = 0; x < padW; ++x) {
                const int srcX = std::min(x, width - 1);
                const size_t srcIdx = (static_cast<size_t>(srcY) * width + srcX) * 3;
                const size_t dst = static_cast<size_t>(y) * padW + x;
                for (int c = 0; c < 3; ++c) {
                    input[c * plane + dst] = img0[srcIdx + c] / 255.0f;
                    input[(3 + c) * plane + dst] = img1[srcIdx + c] / 255.0f;
                }
                input[6 * plane + dst] = timestep;
                input[7 * plane + dst] = 2.0f * x / (padW - 1) - 1.0f;
                input[8 * plane + dst] = 2.0f * y / (padH - 1) - 1.0f;
                input[9 * plane + dst] = 2.0f / (padW - 1);
                input[10 * plane + dst] = 2.0f / (padH - 1);
            }
        }

        const int64_t shape[4] = {1, 11, padH, padW};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, input.data(), input.size(), shape, 4);

        const char* inNames[] = {m_impl->inputName.c_str()};
        const char* outNames[] = {m_impl->outputName.c_str()};
        auto outputs = m_impl->session->Run(Ort::RunOptions{nullptr},
                                            inNames, &inputTensor, 1, outNames, 1);

        const float* out = outputs[0].GetTensorData<float>();
        outFrame.resize(static_cast<size_t>(width) * height * 3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t src = static_cast<size_t>(y) * padW + x;
                const size_t dst = (static_cast<size_t>(y) * width + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    const float v = out[c * plane + src];
                    outFrame[dst + c] = static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        m_impl->lastError = std::string("RIFE inference failed: ") + e.what();
        return false;
    }
#endif
}

} // namespace BeatSync
