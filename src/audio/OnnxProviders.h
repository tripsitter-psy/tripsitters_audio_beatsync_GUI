#pragma once
/**
 * @file OnnxProviders.h
 * @brief One place that decides which ONNX Runtime execution provider a session uses.
 *
 * Every AI component (beat detector, stem separator, RIFE, upscaler) used to carry
 * its own copy of the TensorRT -> CUDA -> CPU chain. This helper owns the chain
 * for all of them and extends it to non-NVIDIA GPUs:
 *
 *   TensorRT   NVIDIA RTX, fastest, optional (fixed shapes only, engine cache)
 *   CUDA       any NVIDIA GPU (GTX 10xx+)
 *   MIGraphX   AMD Radeon via ROCm 7.x (AMD's graph compiler EP)
 *   ROCm       AMD Radeon via ROCm (plain HIP kernels EP)
 *   OpenVINO   Intel Arc / Iris / UHD GPUs ("GPU" device), also Intel NPU
 *   DirectML   Windows only, any DX12 GPU (AMD/Intel/NVIDIA)
 *   OpenVINO   Intel CPUs ("CPU" device; still much faster than ORT's CPU EP)
 *   CPU        last resort
 *
 * A provider that is not installed simply fails to append and the chain moves on;
 * the result says what was actually chosen so callers (and the GUI) can tell the
 * user instead of silently running on the CPU.
 */

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Ort { struct SessionOptions; struct Session; struct Env; }

namespace BeatSync {

struct OnnxProviderRequest {
    std::string component;         ///< Log prefix, e.g. "Upscaler"
    int deviceId = 0;              ///< GPU index (CUDA/ROCm/MIGraphX)
    bool allowTensorRT = true;     ///< Some models (dynamic shapes) are worse on TensorRT
    bool trtFp16 = true;           ///< FP16 engines on tensor cores
    std::string trtCacheDir;       ///< Engine cache directory; empty = no cache
    bool allowOpenVinoCpu = true;  ///< Use OpenVINO's CPU device before ORT's plain CPU
    std::vector<std::string> skipProviders;  ///< Providers already known to fail (see createSessionWithFallback)
};

struct OnnxProviderResult {
    std::string name;    ///< "TensorRT", "CUDA", "MIGraphX", "ROCm", "OpenVINO", "DirectML", "CPU"
    std::string device;  ///< e.g. "GPU", "CPU", "NPU" for OpenVINO; empty otherwise
    bool isGpu = false;  ///< false means the AI stage will run on the CPU
    std::string detail;  ///< Human readable, e.g. "OpenVINO (Intel GPU)"

    /// True when a GPU-class provider is active (anything but CPU / OpenVINO-CPU).
    bool accelerated() const { return isGpu; }
};

/**
 * Append the best available execution providers to @p options, in fallback order.
 * Never throws; a CPU result is returned when nothing else could be appended.
 */
OnnxProviderResult appendBestExecutionProviders(Ort::SessionOptions& options,
                                                const OnnxProviderRequest& request);

/**
 * Create a session, walking down the provider chain when a provider appends but
 * then fails to initialise the device (missing driver, out of memory, unsupported
 * GPU). @p configure receives fresh SessionOptions on every attempt (threads,
 * graph optimisation level...). Returns nullptr with @p outError set only when
 * even the CPU provider fails.
 */
using SessionOptionsConfigurator = std::function<void(Ort::SessionOptions&)>;
std::unique_ptr<Ort::Session> createSessionWithFallback(Ort::Env& env,
                                                        const std::string& modelPath,
                                                        const OnnxProviderRequest& request,
                                                        const SessionOptionsConfigurator& configure,
                                                        OnnxProviderResult& outResult,
                                                        std::string& outError);

/**
 * Names of the providers compiled into / discoverable by this ONNX Runtime build
 * (OrtGetAvailableProviders). Purely informational.
 */
std::vector<std::string> availableProviderNames();

/**
 * Summary of the last provider chosen by any component, for status displays:
 *   "CUDA (NVIDIA GPU)", "OpenVINO (Intel GPU)", "CPU only - no GPU acceleration".
 * Empty until the first session has been created.
 */
std::string lastProviderSummary();

/**
 * Cheap startup probe: creates a throwaway session on a tiny in-memory model to
 * find out which provider this machine will actually use, so the GUI can warn
 * about CPU-only operation before the user starts a long render.
 */
OnnxProviderResult probeAcceleration();

} // namespace BeatSync
