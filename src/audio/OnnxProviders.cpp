#include "OnnxProviders.h"

#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>

namespace BeatSync {

namespace {

std::mutex g_summaryMutex;
std::string g_lastSummary;

void remember(const OnnxProviderResult& r) {
    std::lock_guard<std::mutex> lock(g_summaryMutex);
    g_lastSummary = r.detail;
}

std::string statusMessage(const OrtApi& api, OrtStatus* status) {
    std::string msg;
    if (status) {
        const char* m = api.GetErrorMessage(status);
        msg = m ? m : "unknown error";
        api.ReleaseStatus(status);
    }
    return msg;
}

bool skipped(const OnnxProviderRequest& req, const char* name) {
    for (const auto& s : req.skipProviders) if (s == name) return true;
    return false;
}

bool envDisabled(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

// ---- TensorRT ---------------------------------------------------------------
bool tryTensorRT(const OrtApi& api, OrtSessionOptions* so, const OnnxProviderRequest& req, std::string& err) {
    OrtTensorRTProviderOptionsV2* opts = nullptr;
    OrtStatus* st = api.CreateTensorRTProviderOptions(&opts);
    if (st || !opts) { err = statusMessage(api, st); return false; }

    char deviceId[16];
    std::snprintf(deviceId, sizeof(deviceId), "%d", req.deviceId);
    std::vector<const char*> keys{"device_id", "trt_fp16_enable"};
    std::vector<const char*> values{deviceId, req.trtFp16 ? "1" : "0"};
    if (!req.trtCacheDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(req.trtCacheDir, ec);
        if (!ec) {
            keys.push_back("trt_engine_cache_enable");  values.push_back("1");
            keys.push_back("trt_engine_cache_path");    values.push_back(req.trtCacheDir.c_str());
            keys.push_back("trt_timing_cache_enable");  values.push_back("1");
        }
    }
    st = api.UpdateTensorRTProviderOptions(opts, keys.data(), values.data(), keys.size());
    if (!st) st = api.SessionOptionsAppendExecutionProvider_TensorRT_V2(so, opts);
    api.ReleaseTensorRTProviderOptions(opts);
    if (st) { err = statusMessage(api, st); return false; }
    return true;
}

// ---- CUDA -------------------------------------------------------------------
bool tryCUDA(const OrtApi& api, OrtSessionOptions* so, const OnnxProviderRequest& req, std::string& err) {
    OrtCUDAProviderOptionsV2* opts = nullptr;
    OrtStatus* st = api.CreateCUDAProviderOptions(&opts);
    if (st || !opts) { err = statusMessage(api, st); return false; }
    char deviceId[16];
    std::snprintf(deviceId, sizeof(deviceId), "%d", req.deviceId);
    const char* keys[] = {"device_id", "arena_extend_strategy"};
    const char* values[] = {deviceId, "kSameAsRequested"};
    st = api.UpdateCUDAProviderOptions(opts, keys, values, 2);
    if (!st) st = api.SessionOptionsAppendExecutionProvider_CUDA_V2(so, opts);
    api.ReleaseCUDAProviderOptions(opts);
    if (st) { err = statusMessage(api, st); return false; }
    return true;
}

// ---- AMD: MIGraphX then ROCm --------------------------------------------------
bool tryMIGraphX(const OrtApi& api, OrtSessionOptions* so, const OnnxProviderRequest& req, std::string& err) {
    OrtMIGraphXProviderOptions opts{};
    opts.device_id = req.deviceId;
    opts.migraphx_fp16_enable = 1;
    OrtStatus* st = api.SessionOptionsAppendExecutionProvider_MIGraphX(so, &opts);
    if (st) { err = statusMessage(api, st); return false; }
    return true;
}

bool tryROCm(const OrtApi& api, OrtSessionOptions* so, const OnnxProviderRequest& req, std::string& err) {
    OrtROCMProviderOptions opts{};
    opts.device_id = req.deviceId;
    OrtStatus* st = api.SessionOptionsAppendExecutionProvider_ROCM(so, &opts);
    if (st) { err = statusMessage(api, st); return false; }
    return true;
}

// ---- Intel: OpenVINO --------------------------------------------------------
#ifndef _WIN32
#include <dlfcn.h>
#endif

// Ask OpenVINO itself which devices exist ("CPU", "GPU", "GPU.1", "NPU"...). The
// execution provider happily accepts device_type=GPU on a machine without an
// Intel GPU and quietly runs elsewhere, so this check is what keeps the chain
// honest. Uses libopenvino_c (shipped next to the provider) through dlopen so the
// backend has no link-time dependency on OpenVINO.
std::vector<std::string> openVinoDevices() {
    std::vector<std::string> devices;
#ifndef _WIN32
    void* lib = dlopen("libopenvino_c.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libopenvino_c.so.2530", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        // Try next to libonnxruntime (the bundle keeps all provider libs together).
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&OrtGetApiBase), &info) && info.dli_fname) {
            std::string dir = std::filesystem::path(info.dli_fname).parent_path().string();
            for (const char* name : {"libopenvino_c.so", "libopenvino_c.so.2530"}) {
                lib = dlopen((dir + "/" + name).c_str(), RTLD_NOW | RTLD_LOCAL);
                if (lib) break;
            }
        }
    }
    if (!lib) return devices;

    using ov_core_create_t = int (*)(void**);
    using ov_core_free_t = void (*)(void*);
    struct ov_available_devices { char** devices; size_t size; };
    using ov_get_devices_t = int (*)(void*, ov_available_devices*);
    using ov_free_devices_t = void (*)(ov_available_devices*);
    using ov_get_property_t = int (*)(void*, const char*, const char*, char**);
    using ov_free_t = void (*)(void*);
    auto create = reinterpret_cast<ov_core_create_t>(dlsym(lib, "ov_core_create"));
    auto free_core = reinterpret_cast<ov_core_free_t>(dlsym(lib, "ov_core_free"));
    auto get_devices = reinterpret_cast<ov_get_devices_t>(dlsym(lib, "ov_core_get_available_devices"));
    auto free_devices = reinterpret_cast<ov_free_devices_t>(dlsym(lib, "ov_available_devices_free"));
    auto get_property = reinterpret_cast<ov_get_property_t>(dlsym(lib, "ov_core_get_property"));
    auto ov_free = reinterpret_cast<ov_free_t>(dlsym(lib, "ov_free"));
    if (create && free_core && get_devices && free_devices) {
        void* core = nullptr;
        if (create(&core) == 0 && core) {
            ov_available_devices list{nullptr, 0};
            if (get_devices(core, &list) == 0) {
                for (size_t i = 0; i < list.size; ++i) {
                    if (!list.devices[i]) continue;
                    std::string name = list.devices[i];
                    // The GPU plugin enumerates every OpenCL GPU, including NVIDIA
                    // and AMD cards, which it does not actually support. Keep a GPU
                    // entry only when the device really is Intel; NVIDIA/AMD users
                    // are served by CUDA / MIGraphX above.
                    if (name.rfind("GPU", 0) == 0 && get_property && ov_free) {
                        char* full = nullptr;
                        if (get_property(core, name.c_str(), "FULL_DEVICE_NAME", &full) == 0 && full) {
                            std::string fullName = full;
                            ov_free(full);
                            std::cerr << "[BeatSync] OpenVINO " << name << " = " << fullName << std::endl;
                            if (fullName.find("Intel") == std::string::npos) continue;
                        }
                    }
                    devices.push_back(name);
                }
                free_devices(&list);
            }
            free_core(core);
        }
    }
    // Keep the library loaded: OpenVINO registers plugins globally and unloading
    // it while the provider is about to use it is not worth the risk.
#endif
    return devices;
}

bool hasOpenVinoDevice(const std::vector<std::string>& devices, const char* prefix) {
    for (const auto& d : devices) if (d.rfind(prefix, 0) == 0) return true;
    return false;
}

bool tryOpenVINO(const OrtApi& api, OrtSessionOptions* so, const char* deviceType, std::string& err) {
    const char* keys[] = {"device_type", "precision"};
    const char* values[] = {deviceType, "FP16"};
    size_t n = (std::strcmp(deviceType, "CPU") == 0) ? 1 : 2;  // CPU device prefers FP32
    OrtStatus* st = api.SessionOptionsAppendExecutionProvider_OpenVINO_V2(so, keys, values, n);
    if (st) { err = statusMessage(api, st); return false; }
    return true;
}

// ---- Windows: DirectML -------------------------------------------------------
bool tryDirectML(Ort::SessionOptions& options, std::string& err) {
#ifdef _WIN32
    try {
        options.AppendExecutionProvider("DML", {});
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
#else
    (void)options;
    err = "not available on this platform";
    return false;
#endif
}

// Minimal ONNX model (Identity: float[1] X -> Y) encoded by hand so the probe does
// not depend on a model file being present. Protobuf wire format, opset 13.
std::vector<uint8_t> tinyIdentityModel() {
    std::vector<uint8_t> out;
    auto varint = [](std::vector<uint8_t>& b, uint64_t v) {
        while (v >= 0x80) { b.push_back(static_cast<uint8_t>(v | 0x80)); v >>= 7; }
        b.push_back(static_cast<uint8_t>(v));
    };
    auto field = [&](std::vector<uint8_t>& b, int num, const std::vector<uint8_t>& payload) {
        varint(b, static_cast<uint64_t>(num << 3 | 2));
        varint(b, payload.size());
        b.insert(b.end(), payload.begin(), payload.end());
    };
    auto str = [&](std::vector<uint8_t>& b, int num, const char* s) {
        std::vector<uint8_t> p(s, s + std::strlen(s));
        field(b, num, p);
    };
    auto vint = [&](std::vector<uint8_t>& b, int num, uint64_t v) {
        varint(b, static_cast<uint64_t>(num << 3 | 0));
        varint(b, v);
    };

    // ValueInfoProto {name, type{tensor_type{elem_type=1 (float), shape{dim{dim_value=1}}}}}
    auto valueInfo = [&](const char* name) {
        std::vector<uint8_t> dim;   vint(dim, 1, 1);
        std::vector<uint8_t> shape; field(shape, 1, dim);
        std::vector<uint8_t> tensor; vint(tensor, 1, 1); field(tensor, 2, shape);
        std::vector<uint8_t> type;  field(type, 1, tensor);
        std::vector<uint8_t> vi;    str(vi, 1, name); field(vi, 2, type);
        return vi;
    };
    std::vector<uint8_t> node; str(node, 1, "X"); str(node, 2, "Y"); str(node, 4, "Identity");
    std::vector<uint8_t> graph;
    field(graph, 1, node);
    str(graph, 2, "probe");
    field(graph, 11, valueInfo("X"));
    field(graph, 12, valueInfo("Y"));
    std::vector<uint8_t> opset; str(opset, 1, ""); vint(opset, 2, 13);

    vint(out, 1, 8);          // ir_version
    field(out, 7, graph);     // graph
    field(out, 8, opset);     // opset_import
    return out;
}

} // namespace

std::vector<std::string> availableProviderNames() {
    try {
        return Ort::GetAvailableProviders();
    } catch (...) {
        return {};
    }
}

OnnxProviderResult appendBestExecutionProviders(Ort::SessionOptions& options,
                                                const OnnxProviderRequest& request) {
    const OrtApi& api = Ort::GetApi();
    OrtSessionOptions* so = static_cast<OrtSessionOptions*>(options);
    const std::string tag = "[BeatSync] " + (request.component.empty() ? std::string("ONNX") : request.component) + ": ";
    OnnxProviderResult result;
    result.name = "CPU";
    result.isGpu = false;
    result.detail = "CPU only - no GPU acceleration";

    auto log = [&](const std::string& s) { std::cerr << tag << s << std::endl; };
    std::string err;

    const bool forceCpu = envDisabled("BEATSYNC_FORCE_CPU");
    if (forceCpu) {
        log("BEATSYNC_FORCE_CPU set; using the CPU provider");
        remember(result);
        return result;
    }

    // NVIDIA -------------------------------------------------------------
    bool nvidia = false;
    if (request.allowTensorRT && !envDisabled("BEATSYNC_NO_TENSORRT") && !skipped(request, "TensorRT")) {
        if (tryTensorRT(api, so, request, err)) {
            result = {"TensorRT", "", true, "TensorRT (NVIDIA RTX GPU)"};
            nvidia = true;
            log("TensorRT execution provider enabled" +
                (request.trtCacheDir.empty() ? std::string() : " (engine cache: " + request.trtCacheDir + ")"));
        } else {
            log("TensorRT unavailable (" + err + ")");
        }
    }
    // CUDA is appended even behind TensorRT: it runs whatever TensorRT declines.
    if (!skipped(request, "CUDA") && tryCUDA(api, so, request, err)) {
        if (!nvidia) result = {"CUDA", "", true, "CUDA (NVIDIA GPU)"};
        nvidia = true;
        log("CUDA execution provider enabled");
    } else {
        log("CUDA unavailable (" + err + ")");
    }
    if (nvidia) { remember(result); return result; }

    // AMD ----------------------------------------------------------------
    if (!skipped(request, "MIGraphX") && tryMIGraphX(api, so, request, err)) {
        result = {"MIGraphX", "", true, "MIGraphX (AMD Radeon GPU, ROCm)"};
        log("MIGraphX execution provider enabled");
        // ROCm EP behind it for unsupported subgraphs, best effort.
        if (tryROCm(api, so, request, err)) log("ROCm execution provider enabled as secondary");
        remember(result); return result;
    }
    log("MIGraphX unavailable (" + err + ")");
    if (!skipped(request, "ROCm") && tryROCm(api, so, request, err)) {
        result = {"ROCm", "", true, "ROCm (AMD Radeon GPU)"};
        log("ROCm execution provider enabled");
        remember(result); return result;
    }
    log("ROCm unavailable (" + err + ")");

    // Intel GPU / NPU -----------------------------------------------------
    std::vector<std::string> ovDevices;
    if (!skipped(request, "OpenVINO") || request.allowOpenVinoCpu) {
        ovDevices = openVinoDevices();
        if (!ovDevices.empty()) {
            std::string list;
            for (const auto& d : ovDevices) list += (list.empty() ? "" : ", ") + d;
            log("OpenVINO devices: " + list);
        }
    }
    if (!skipped(request, "OpenVINO") && hasOpenVinoDevice(ovDevices, "GPU")) {
        if (tryOpenVINO(api, so, "GPU", err)) {
            result = {"OpenVINO", "GPU", true, "OpenVINO (Intel GPU)"};
            log("OpenVINO execution provider enabled (Intel GPU)");
            remember(result); return result;
        }
        log("OpenVINO GPU unavailable (" + err + ")");
    } else if (!skipped(request, "OpenVINO")) {
        log("OpenVINO: no Intel GPU device present");
    }
    if (!skipped(request, "OpenVINO") && hasOpenVinoDevice(ovDevices, "NPU")) {
        if (tryOpenVINO(api, so, "NPU", err)) {
            result = {"OpenVINO", "NPU", true, "OpenVINO (Intel NPU)"};
            log("OpenVINO execution provider enabled (Intel NPU)");
            remember(result); return result;
        }
        log("OpenVINO NPU unavailable (" + err + ")");
    }

    // Windows: any DX12 GPU ------------------------------------------------
    if (!skipped(request, "DirectML") && tryDirectML(options, err)) {
        result = {"DirectML", "", true, "DirectML (DirectX 12 GPU)"};
        log("DirectML execution provider enabled");
        remember(result); return result;
    }

    // Intel CPU through OpenVINO, still well ahead of the plain CPU provider.
    if (request.allowOpenVinoCpu && !skipped(request, "OpenVINO-CPU") && hasOpenVinoDevice(ovDevices, "CPU")
        && tryOpenVINO(api, so, "CPU", err)) {
        result = {"OpenVINO", "CPU", false, "OpenVINO (CPU) - no GPU acceleration"};
        log("OpenVINO execution provider enabled (CPU device); NO GPU acceleration");
        remember(result); return result;
    }

    log("WARNING: no GPU execution provider could be enabled; this stage will run on the CPU and be MUCH slower");
    remember(result);
    return result;
}

std::unique_ptr<Ort::Session> createSessionWithFallback(Ort::Env& env,
                                                        const std::string& modelPath,
                                                        const OnnxProviderRequest& request,
                                                        const SessionOptionsConfigurator& configure,
                                                        OnnxProviderResult& outResult,
                                                        std::string& outError) {
    OnnxProviderRequest req = request;
    const std::string tag = "[BeatSync] " + (req.component.empty() ? std::string("ONNX") : req.component) + ": ";
    for (int attempt = 0; attempt < 8; ++attempt) {
        Ort::SessionOptions options;
        if (configure) configure(options);
        outResult = appendBestExecutionProviders(options, req);
        try {
#ifdef _WIN32
            std::wstring wpath;
            int wlen = MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0);
            if (wlen > 0) {
                wpath.resize(static_cast<size_t>(wlen));
                MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, &wpath[0], wlen);
                wpath.resize(static_cast<size_t>(wlen - 1));
            }
            auto session = std::make_unique<Ort::Session>(env, wpath.c_str(), options);
#else
            auto session = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
#endif
            remember(outResult);
            return session;
        } catch (const std::exception& e) {
            outError = e.what();
            if (outResult.name == "CPU") {
                std::cerr << tag << "session creation failed on CPU: " << outError << std::endl;
                return nullptr;
            }
            std::cerr << tag << "session creation failed on " << outResult.detail << " (" << outError
                      << "); trying the next provider" << std::endl;
            // OpenVINO on CPU is the last non-CPU rung; mark it separately.
            req.skipProviders.push_back(outResult.name == "OpenVINO" && outResult.device == "CPU"
                                            ? "OpenVINO-CPU" : outResult.name);
        }
    }
    outError = "no execution provider could create a session";
    return nullptr;
}

std::string lastProviderSummary() {
    std::lock_guard<std::mutex> lock(g_summaryMutex);
    return g_lastSummary;
}

OnnxProviderResult probeAcceleration() {
    OnnxProviderResult result;
    result.name = "CPU";
    result.detail = "CPU only - no GPU acceleration";
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "beatsync_probe");
        Ort::SessionOptions options;
        OnnxProviderRequest req;
        req.component = "Probe";
        req.allowTensorRT = false;   // an engine build is wasted work for a probe
        req.allowOpenVinoCpu = true;
        result = appendBestExecutionProviders(options, req);
        // Creating the session is what actually initialises the device (driver,
        // context, allocators); a provider can append fine and still fail here.
        std::vector<uint8_t> model = tinyIdentityModel();
        Ort::Session session(env, model.data(), model.size(), options);
        float x = 1.0f;
        int64_t shape[1] = {1};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, &x, 1, shape, 1);
        const char* inNames[] = {"X"};
        const char* outNames[] = {"Y"};
        auto out = session.Run(Ort::RunOptions{nullptr}, inNames, &in, 1, outNames, 1);
        (void)out;
    } catch (const std::exception& e) {
        std::cerr << "[BeatSync] Probe: session on " << result.name << " failed (" << e.what()
                  << "); reporting CPU" << std::endl;
        result = {"CPU", "", false, "CPU only - GPU provider failed to initialise"};
    }
    remember(result);
    return result;
}

} // namespace BeatSync
