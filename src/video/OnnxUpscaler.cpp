#include "OnnxUpscaler.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace BeatSync {

namespace {
constexpr int kTileOverlap = 16;   // source pixels blended between adjacent tiles
constexpr int kDefaultTile = 512;  // source pixels per tile edge
} // namespace

struct OnnxUpscaler::Impl {
#ifdef USE_ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    std::string inputName;
    std::string outputName;
#endif
    bool loaded = false;
    int scale = 0;
    int tileSize = kDefaultTile;
    std::string lastError;
};

OnnxUpscaler::OnnxUpscaler() : m_impl(std::make_unique<Impl>()) {
#ifdef USE_ONNX
    try {
        m_impl->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BeatSyncUpscaler");
    } catch (const std::exception& e) {
        m_impl->lastError = std::string("ONNX env init failed: ") + e.what();
    }
#endif
}

OnnxUpscaler::~OnnxUpscaler() = default;

bool OnnxUpscaler::isAvailable() {
#ifdef USE_ONNX
    return true;
#else
    return false;
#endif
}

bool OnnxUpscaler::isLoaded() const { return m_impl->loaded; }
int OnnxUpscaler::getScale() const { return m_impl->scale; }
std::string OnnxUpscaler::getLastError() const { return m_impl->lastError; }

void OnnxUpscaler::setTileSize(int tilePixels) {
    m_impl->tileSize = std::max(64, tilePixels);
}

bool OnnxUpscaler::loadModel(const std::string& modelPath, bool useGPU, int gpuDeviceId) {
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
                            std::cerr << "[BeatSync] Upscaler: CUDA execution provider enabled" << std::endl;
                        } else {
                            const char* msg = ortApi.GetErrorMessage(status);
                            std::cerr << "[BeatSync] Upscaler: CUDA append failed: " << (msg ? msg : "?") << std::endl;
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
                std::cerr << "[BeatSync] Upscaler: CUDA provider exception" << std::endl;
            }

#ifdef _WIN32
            if (activeProvider == "CPU") {
                try {
                    m_impl->sessionOptions->AppendExecutionProvider("DML", {});
                    activeProvider = "DirectML";
                    std::cerr << "[BeatSync] Upscaler: DirectML execution provider enabled" << std::endl;
                } catch (...) {
                    std::cerr << "[BeatSync] Upscaler: DirectML fallback failed" << std::endl;
                }
            }
#endif
        }

        m_impl->session = std::make_unique<Ort::Session>(*m_impl->env, modelPath.c_str(),
                                                        *m_impl->sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        if (m_impl->session->GetInputCount() != 1 || m_impl->session->GetOutputCount() != 1) {
            m_impl->lastError = "Unsupported upscaler model (expected 1 input and 1 output)";
            m_impl->session.reset();
            return false;
        }
        m_impl->inputName = m_impl->session->GetInputNameAllocated(0, allocator).get();
        m_impl->outputName = m_impl->session->GetOutputNameAllocated(0, allocator).get();

        m_impl->loaded = true;

        // Detect the scale factor by running a small probe frame; exports rarely
        // declare it statically and 2x/3x/4x variants share the same signature.
        const int probe = 64;
        std::vector<uint8_t> probeIn(static_cast<size_t>(probe) * probe * 3, 128);
        std::vector<uint8_t> probeOut;
        m_impl->scale = 1;  // provisional so upscale() runs untiled
        m_impl->tileSize = probe;
        if (!upscale(probeIn.data(), probe, probe, probeOut)) {
            m_impl->lastError = "Upscaler probe inference failed: " + m_impl->lastError;
            m_impl->loaded = false;
            m_impl->session.reset();
            return false;
        }
        const size_t outPixels = probeOut.size() / 3;
        const int outEdge = static_cast<int>(std::lround(std::sqrt(static_cast<double>(outPixels))));
        m_impl->scale = std::max(1, outEdge / probe);
        m_impl->tileSize = kDefaultTile;

        if (m_impl->scale < 2) {
            m_impl->lastError = "Model does not appear to upscale (detected scale " +
                                std::to_string(m_impl->scale) + ")";
            m_impl->loaded = false;
            m_impl->session.reset();
            return false;
        }

        std::cerr << "[BeatSync] Upscaler loaded (" << activeProvider << ", "
                  << m_impl->scale << "x): " << modelPath << std::endl;
        return true;
    } catch (const std::exception& e) {
        m_impl->lastError = std::string("ONNX load failed: ") + e.what();
        m_impl->loaded = false;
        m_impl->session.reset();
        return false;
    }
#endif
}

bool OnnxUpscaler::upscale(const uint8_t* rgb, int width, int height,
                           std::vector<uint8_t>& outRgb) {
#ifndef USE_ONNX
    (void)rgb; (void)width; (void)height; (void)outRgb;
    m_impl->lastError = "ONNX Runtime not available";
    return false;
#else
    if (!m_impl->loaded || !rgb || width <= 0 || height <= 0) {
        m_impl->lastError = "Upscaler not loaded or invalid frame";
        return false;
    }

    try {
        const int scale = std::max(1, m_impl->scale);
        const int outW = width * scale;
        const int outH = height * scale;
        outRgb.assign(static_cast<size_t>(outW) * outH * 3, 0);

        // Tile the source with overlap; only the non-overlapping core of each
        // tile is written out, so seams never land on a tile boundary.
        const int tile = std::max(64, m_impl->tileSize);
        const int step = std::max(1, tile - 2 * kTileOverlap);

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char* inNames[] = {m_impl->inputName.c_str()};
        const char* outNames[] = {m_impl->outputName.c_str()};

        for (int ty = 0; ty < height; ty += step) {
            for (int tx = 0; tx < width; tx += step) {
                // Tile bounds in source pixels, expanded by the overlap margin
                const int x0 = std::max(0, tx - kTileOverlap);
                const int y0 = std::max(0, ty - kTileOverlap);
                const int x1 = std::min(width, tx + step + kTileOverlap);
                const int y1 = std::min(height, ty + step + kTileOverlap);
                const int tw = x1 - x0;
                const int th = y1 - y0;
                if (tw <= 0 || th <= 0) continue;

                // Pack tile as CHW float in [0,1]
                std::vector<float> input(static_cast<size_t>(3) * tw * th);
                const size_t plane = static_cast<size_t>(tw) * th;
                for (int y = 0; y < th; ++y) {
                    const uint8_t* srcRow = rgb + (static_cast<size_t>(y0 + y) * width + x0) * 3;
                    for (int x = 0; x < tw; ++x) {
                        const size_t dst = static_cast<size_t>(y) * tw + x;
                        input[dst] = srcRow[x * 3 + 0] / 255.0f;
                        input[plane + dst] = srcRow[x * 3 + 1] / 255.0f;
                        input[2 * plane + dst] = srcRow[x * 3 + 2] / 255.0f;
                    }
                }

                const int64_t shape[4] = {1, 3, th, tw};
                Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                    memInfo, input.data(), input.size(), shape, 4);
                auto outputs = m_impl->session->Run(Ort::RunOptions{nullptr},
                                                    inNames, &inputTensor, 1, outNames, 1);

                auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                if (outShape.size() != 4) {
                    m_impl->lastError = "Unexpected upscaler output rank";
                    return false;
                }
                const int oh = static_cast<int>(outShape[2]);
                const int ow = static_cast<int>(outShape[3]);
                const float* out = outputs[0].GetTensorData<float>();
                const size_t outPlane = static_cast<size_t>(ow) * oh;

                // During the probe pass we only need the output dimensions.
                if (scale == 1) {
                    outRgb.assign(outPlane * 3, 0);
                    return true;
                }

                // Copy back only this tile's core region (drop the overlap margin)
                const int coreX0 = tx;
                const int coreY0 = ty;
                const int coreX1 = std::min(width, tx + step);
                const int coreY1 = std::min(height, ty + step);
                for (int y = coreY0; y < coreY1; ++y) {
                    for (int x = coreX0; x < coreX1; ++x) {
                        // Position of this source pixel inside the tile output
                        const int sx = (x - x0) * scale;
                        const int sy = (y - y0) * scale;
                        for (int dy = 0; dy < scale; ++dy) {
                            for (int dx = 0; dx < scale; ++dx) {
                                const int oy = sy + dy;
                                const int ox = sx + dx;
                                if (oy < 0 || oy >= oh || ox < 0 || ox >= ow) continue;
                                const size_t src = static_cast<size_t>(oy) * ow + ox;
                                const size_t dstPix =
                                    (static_cast<size_t>(y * scale + dy) * outW + (x * scale + dx)) * 3;
                                if (dstPix + 2 >= outRgb.size()) continue;
                                for (int c = 0; c < 3; ++c) {
                                    const float v = out[c * outPlane + src];
                                    outRgb[dstPix + c] =
                                        static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        m_impl->lastError = std::string("Upscaler inference failed: ") + e.what();
        return false;
    }
#endif
}

} // namespace BeatSync
