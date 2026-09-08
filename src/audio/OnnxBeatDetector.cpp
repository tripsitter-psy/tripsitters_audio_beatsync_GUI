/**
 * @file OnnxBeatDetector.cpp
 * @brief ONNX Runtime-based neural network beat detector implementation
 *
 * Provides native C++ inference for beat detection models (BeatNet, All-In-One, TCN)
 * without requiring Python dependencies.
 */

#include "OnnxBeatDetector.h"
#ifdef USE_ONNX
#include "OnnxProviders.h"
#endif
#include "tracing/Tracing.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <complex>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <filesystem>


#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
// DirectML support removed - use CUDA EP instead for GPU acceleration
// DirectML requires linking against directml.lib which isn't part of ONNX Runtime vcpkg
#endif

#ifdef USE_LIBSAMPLERATE
extern "C" {
#include <samplerate.h>
}
#endif

namespace BeatSync {

// Pi constant (C++17 compatible)
constexpr double PI = 3.14159265358979323846;

// ============================================================================
// MelSpectrogramExtractor Implementation
// ============================================================================

struct MelSpectrogramExtractor::Impl {
    int sampleRate;
    int nMels;
    int nFft;
    int hopLength;
    float fmin;
    float fmax;

    // Precomputed mel filterbank
    std::vector<std::vector<float>> melFilterbank;

    // FFT helpers
    std::vector<std::complex<double>> fftBuffer;
    std::vector<double> window;

    Impl(int sr, int mels, int fft, int hop, float fminHz, float fmaxHz)
        : sampleRate(sr), nMels(mels), nFft(fft), hopLength(hop), fmin(fminHz), fmax(fmaxHz) {
        initMelFilterbank();
        initWindow();
        fftBuffer.resize(nextPow2(nFft));
    }

    static int nextPow2(int v) {
        if (v <= 0) return 1; // Clamp non-positive input to 1
        int p = 1;
        // Guard against overflow: if p > INT_MAX/2, next shift would overflow
        while (p < v) {
            if (p > (std::numeric_limits<int>::max() >> 1)) {
                // Would overflow on next shift; clamp to INT_MAX
                return std::numeric_limits<int>::max();
            }
            p <<= 1;
        }
        return p;
    }

    // Convert frequency to mel scale
    static float hzToMel(float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    }

    // Convert mel to frequency
    static float melToHz(float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }

    void initMelFilterbank() {
        // Create mel filterbank following librosa conventions
        float melMin = hzToMel(fmin);
        float melMax = hzToMel(std::min(fmax, sampleRate / 2.0f));

        // Mel points
        std::vector<float> melPoints(nMels + 2);
        for (int i = 0; i < nMels + 2; ++i) {
            melPoints[i] = melMin + i * (melMax - melMin) / (nMels + 1);
        }

        // Convert to Hz and then to FFT bin indices
        std::vector<int> binPoints(nMels + 2);
        for (int i = 0; i < nMels + 2; ++i) {
            float hz = melToHz(melPoints[i]);
            binPoints[i] = static_cast<int>(std::floor((nFft + 1) * hz / sampleRate));
        }

        // Create triangular filters
        int nBins = nFft / 2 + 1;
        melFilterbank.resize(nMels);
        for (int m = 0; m < nMels; ++m) {
            melFilterbank[m].resize(nBins, 0.0f);

            int startBin = binPoints[m];
            int centerBin = binPoints[m + 1];
            int endBin = binPoints[m + 2];

            // Rising slope
            for (int k = startBin; k < centerBin; ++k) {
                if (k >= 0 && k < nBins && centerBin != startBin) {
                    melFilterbank[m][k] = static_cast<float>(k - startBin) / (centerBin - startBin);
                }
            }

            // Falling slope
            for (int k = centerBin; k < endBin; ++k) {
                if (k >= 0 && k < nBins && endBin != centerBin) {
                    melFilterbank[m][k] = static_cast<float>(endBin - k) / (endBin - centerBin);
                }
            }

            // Normalize (slaney normalization)
            float filterSum = 0.0f;
            for (float v : melFilterbank[m]) filterSum += v;
            if (filterSum > 0) {
                for (float& v : melFilterbank[m]) v /= filterSum;
            }
        }
    }

    void initWindow() {
        // Defensive: handle nFft <= 1
        window.resize(nFft);
        if (nFft <= 1) {
            if (nFft == 1) window[0] = 1.0;
            return;
        }
        // Hann window
        for (int n = 0; n < nFft; ++n) {
            window[n] = 0.5 * (1.0 - std::cos(2.0 * PI * n / (nFft - 1)));
        }
    }

    // In-place radix-2 FFT
    void fft(std::vector<std::complex<double>>& a) {
        const int n = static_cast<int>(a.size());
        int j = 0;
        for (int i = 1; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }

        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * PI / len;
            std::complex<double> wlen(std::cos(ang), std::sin(ang));
            for (int i = 0; i < n; i += len) {
                std::complex<double> w(1);
                for (int jj = 0; jj < len / 2; ++jj) {
                    std::complex<double> u = a[i + jj];
                    std::complex<double> v = a[i + jj + len / 2] * w;
                    a[i + jj] = u + v;
                    a[i + jj + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

    std::vector<float> computeMelSpectrogram(const std::vector<float>& samples) {
        // Use size_t for intermediate calculation to avoid overflow with large inputs
        size_t sampleCount = samples.size();
        int numFrames = 0;
        if (sampleCount >= static_cast<size_t>(nFft)) {
            size_t frameCount = 1 + (sampleCount - static_cast<size_t>(nFft)) / static_cast<size_t>(hopLength);
            // Clamp to INT_MAX to avoid overflow when casting to int
            numFrames = (frameCount > static_cast<size_t>(std::numeric_limits<int>::max()))
                        ? std::numeric_limits<int>::max()
                        : static_cast<int>(frameCount);
        }
        // numFrames is 0 if input is too short (sampleCount < nFft)

        if (numFrames == 0) return {};

        std::vector<float> melSpec(nMels * numFrames, 0.0f);
        int fftSize = nextPow2(nFft);
        int nBins = nFft / 2 + 1;

        for (int frame = 0; frame < numFrames; ++frame) {
            // Use size_t to prevent overflow when frame * hopLength exceeds INT_MAX
            size_t offset = static_cast<size_t>(frame) * static_cast<size_t>(hopLength);

            // Fill FFT buffer with windowed samples
            for (int i = 0; i < fftSize; ++i) {
                if (i < nFft && offset + static_cast<size_t>(i) < samples.size()) {
                    fftBuffer[i] = std::complex<double>(samples[offset + static_cast<size_t>(i)] * window[i], 0.0);
                } else {
                    fftBuffer[i] = std::complex<double>(0.0, 0.0);
                }
            }

            // Compute FFT
            fft(fftBuffer);

            // Compute power spectrum
            std::vector<float> powerSpec(nBins);
            for (int k = 0; k < nBins; ++k) {
                double re = fftBuffer[k].real();
                double im = fftBuffer[k].imag();
                powerSpec[k] = static_cast<float>(re * re + im * im);
            }

            // Apply mel filterbank
            for (int m = 0; m < nMels; ++m) {
                float melEnergy = 0.0f;
                for (int k = 0; k < nBins; ++k) {
                    melEnergy += powerSpec[k] * melFilterbank[m][k];
                }
                // Log compression (with floor to avoid log(0))
                melSpec[m * numFrames + frame] = std::log(std::max(melEnergy, 1e-10f));
            }
        }

        return melSpec;
    }
};

MelSpectrogramExtractor::MelSpectrogramExtractor(int sampleRate, int nMels,
                                                   int nFft, int hopLength,
                                                   float fmin, float fmax)
    : m_impl(std::make_unique<Impl>(sampleRate, nMels, nFft, hopLength, fmin, fmax)) {
}

MelSpectrogramExtractor::~MelSpectrogramExtractor() = default;

std::vector<float> MelSpectrogramExtractor::extract(const std::vector<float>& samples) {
    return m_impl->computeMelSpectrogram(samples);
}

int MelSpectrogramExtractor::getNumFrames(int numSamples) const {
    // Return 0 for inputs smaller than FFT window (matches computeMelSpectrogram behavior)
    if (numSamples < m_impl->nFft) {
        return 0;
    }
    return 1 + (numSamples - m_impl->nFft) / m_impl->hopLength;
}

int MelSpectrogramExtractor::getNumMels() const {
    return m_impl->nMels;
}

// ============================================================================
// OnnxBeatDetector Implementation
// ============================================================================

struct OnnxBeatDetector::Impl {
#ifdef USE_ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
#endif

    OnnxConfig config;
    std::string lastError;
    std::string modelPath;
    std::string activeProvider = "CPU";  // Tracks which EP the session is actually using
    bool activeProviderIsGpu = false;
    bool loaded = false;

    // Input/output tensor info
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;

    // Mel spectrogram extractor
    std::unique_ptr<MelSpectrogramExtractor> melExtractor;

    // Streaming state
    std::vector<float> streamBuffer;
    double streamTime = 0.0;

    Impl() {
#ifdef USE_ONNX
        try {
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BeatSyncOnnx");
        } catch (const std::exception& e) {
            lastError = std::string("Failed to initialize ONNX Runtime: ") + e.what();
        }
#endif
    }

    bool loadModel(const std::string& path, const OnnxConfig& cfg) {
#ifndef USE_ONNX
        lastError = "ONNX Runtime not available. Rebuild with USE_ONNX=ON";
        return false;
#else
        if (!env) {
            lastError = "ONNX Runtime environment not initialized";
            return false;
        }

        try {
            // Shared provider chain (OnnxProviders): TensorRT -> CUDA -> MIGraphX/ROCm
            // -> OpenVINO -> DirectML -> CPU. BeatNet runs fixed-size windows, so
            // TensorRT engines are cheap to build once and are cached on disk.
            OnnxProviderRequest req;
            req.component = "BeatDetector";
            req.deviceId = cfg.gpuDeviceId;
            req.allowTensorRT = cfg.useGPU;
            if (cfg.useGPU) {
                const char* xdg = std::getenv("XDG_CACHE_HOME");
                const char* home = std::getenv("HOME");
                const char* temp = std::getenv("TEMP");
                req.trtCacheDir = xdg ? std::string(xdg) + "/beatsync/trt_beat"
                                : home ? std::string(home) + "/.cache/beatsync/trt_beat"
                                : temp ? std::string(temp) + "/beatsync_trt_cache"
                                       : std::string();
            } else {
                req.skipProviders = {"TensorRT", "CUDA", "MIGraphX", "ROCm", "OpenVINO", "DirectML"};
            }
            OnnxProviderResult chosen;
            std::string sessionError;
            session = createSessionWithFallback(*env, path, req,
                [&](Ort::SessionOptions& o) {
                    o.SetIntraOpNumThreads(cfg.numThreads > 0 ? cfg.numThreads : 0);
                    o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                }, chosen, sessionError);
            if (!session) {
                lastError = "Failed to load model: " + sessionError;
                return false;
            }
            activeProvider = chosen.name;
            activeProviderIsGpu = chosen.isGpu;
            std::cerr << "[BeatSync] BeatDetector final execution provider: " << chosen.detail << std::endl;

            // Get input info
            size_t numInputs = session->GetInputCount();
            inputNames.clear();
            inputShapes.clear();
            for (size_t i = 0; i < numInputs; ++i) {
                auto namePtr = session->GetInputNameAllocated(i, allocator);
                inputNames.push_back(namePtr.get());

                auto typeInfo = session->GetInputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                inputShapes.push_back(tensorInfo.GetShape());
            }

            // Get output info
            size_t numOutputs = session->GetOutputCount();
            outputNames.clear();
            outputShapes.clear();
            for (size_t i = 0; i < numOutputs; ++i) {
                auto namePtr = session->GetOutputNameAllocated(i, allocator);
                outputNames.push_back(namePtr.get());

                auto typeInfo = session->GetOutputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                outputShapes.push_back(tensorInfo.GetShape());
            }

            // Auto-detect model type based on input/output names
            config = cfg;
            if (config.modelType == OnnxModelType::Custom || config.modelType == OnnxModelType::Unknown) {
                // Try to detect from output names
                for (const auto& name : outputNames) {
                    if (name.find("segment") != std::string::npos) {
                        config.modelType = OnnxModelType::AllInOne;
                        break;
                    }
                }
                if (config.modelType == OnnxModelType::Custom || config.modelType == OnnxModelType::Unknown) {
                    config.modelType = OnnxModelType::BeatNet;  // Default
                }
            }

            // Configure mel extractor based on model type
            switch (config.modelType) {
                case OnnxModelType::BeatNet:
                    config.nMels = 81;
                    config.hopLength = 441;
                    break;
                case OnnxModelType::AllInOne:
                    config.nMels = 128;
                    config.hopLength = 220;
                    break;
                case OnnxModelType::TCN:
                    config.nMels = 81;
                    config.hopLength = 441;
                    break;
                case OnnxModelType::Unknown:
                case OnnxModelType::Custom:
                    // Use default config values
                    break;
            }

            melExtractor = std::make_unique<MelSpectrogramExtractor>(
                config.sampleRate, config.nMels, config.windowLength,
                config.hopLength, config.fmin, config.fmax
            );

            modelPath = path;
            loaded = true;
            return true;

        } catch (const Ort::Exception& e) {
            lastError = std::string("ONNX Runtime error: ") + e.what();
            return false;
        } catch (const std::exception& e) {
            lastError = std::string("Error loading model: ") + e.what();
            return false;
        }
#endif
    }

    std::vector<float> resampleAudio(const std::vector<float>& samples, int srcRate, int dstRate) {
        if (samples.empty()) return {};
        if (srcRate == dstRate) return samples;

#ifdef USE_LIBSAMPLERATE
        // Use libsamplerate for high-quality band-limited resampling
        // Requires linking libsamplerate and including samplerate.h
        double ratio = static_cast<double>(dstRate) / srcRate;
        size_t newSize = static_cast<size_t>(samples.size() * ratio);
        std::vector<float> resampled(newSize);

        SRC_DATA srcData;
        srcData.data_in = const_cast<float*>(samples.data());
        srcData.input_frames = static_cast<long>(samples.size());
        srcData.data_out = resampled.data();
        srcData.output_frames = static_cast<long>(newSize);
        srcData.src_ratio = ratio;
        srcData.end_of_input = 1;

        int error = src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1);
        if (error != 0) {
            // Fallback to linear interpolation if libsamplerate fails
            for (size_t i = 0; i < newSize; ++i) {
                double srcIdx = i / ratio;
                size_t idx0 = static_cast<size_t>(srcIdx);
                size_t idx1 = std::min(idx0 + 1, samples.size() - 1);
                double frac = srcIdx - idx0;
                resampled[i] = static_cast<float>(samples[idx0] * (1.0 - frac) + samples[idx1] * frac);
            }
        } else {
            // If libsamplerate produced fewer frames than expected, resize
            if (static_cast<size_t>(srcData.output_frames_gen) < newSize) {
                resampled.resize(srcData.output_frames_gen);
            }
        }
        return resampled;
#else
        // Fallback: Simple linear interpolation resampling
        double ratio = static_cast<double>(dstRate) / srcRate;
        size_t newSize = static_cast<size_t>(samples.size() * ratio);
        std::vector<float> resampled(newSize);
        for (size_t i = 0; i < newSize; ++i) {
            double srcIdx = i / ratio;
            size_t idx0 = static_cast<size_t>(srcIdx);
            size_t idx1 = std::min(idx0 + 1, samples.size() - 1);
            double frac = srcIdx - idx0;
            resampled[i] = static_cast<float>(samples[idx0] * (1.0 - frac) + samples[idx1] * frac);
        }
        return resampled;
#endif
    }

    std::vector<double> peakPicking(const std::vector<float>& activation, float threshold,
                                    float minInterval, float frameRate) {
        std::vector<double> peaks;
        double lastPeakTime = -1e9;

        for (size_t i = 1; i + 1 < activation.size(); ++i) {
            // Check if local maximum above threshold
            if (activation[i] > threshold &&
                activation[i] >= activation[i - 1] &&
                activation[i] > activation[i + 1]) {

                double time = i / frameRate;
                if (time - lastPeakTime >= minInterval) {
                    peaks.push_back(time);
                    lastPeakTime = time;
                }
            }
        }

        return peaks;
    }

    double estimateBPM(const std::vector<double>& beats) {
        if (beats.size() < 2) return 0.0;

        std::vector<double> intervals;
        for (size_t i = 1; i < beats.size(); ++i) {
            intervals.push_back(beats[i] - beats[i - 1]);
        }

        // Use median interval for robustness
        std::sort(intervals.begin(), intervals.end());
        double medianInterval = intervals[intervals.size() / 2];

        if (medianInterval > 0) {
            return 60.0 / medianInterval;
        }
        return 0.0;
    }

    OnnxAnalysisResult runInference(const std::vector<float>& samples, int sampleRate,
                                    ProgressCallback progress) {
        OnnxAnalysisResult result;

#ifndef USE_ONNX
        lastError = "ONNX Runtime not available";
        return result;
#else
        if (!loaded || !session) {
            lastError = "Model not loaded";
            return result;
        }

        try {
            if (progress) progress(0.1f, "Resampling audio...");

            // Resample to target sample rate
            std::vector<float> resampled = resampleAudio(samples, sampleRate, config.sampleRate);

            if (progress) progress(0.2f, "Computing mel spectrogram...");

            // Compute mel spectrogram
            std::vector<float> melSpec = melExtractor->extract(resampled);
            int numFrames = melExtractor->getNumFrames(static_cast<int>(resampled.size()));
            int nMels = melExtractor->getNumMels();

            if (progress) progress(0.4f, "Running neural network...");

            // Prepare input tensor
            // Shape: (batch=1, channels=1, n_mels, time)
            // melSpec is already laid out as (n_mels * n_frames) which matches the flattened tensor
            std::vector<int64_t> inputShape = {1, 1, nMels, numFrames};
            size_t inputSize = static_cast<size_t>(nMels) * numFrames;

            // Create ONNX tensor directly from melSpec data (no copy needed)
            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto inputTensorOrt = Ort::Value::CreateTensor<float>(
                memoryInfo, melSpec.data(), inputSize,
                inputShape.data(), inputShape.size()
            );

            // Prepare input/output name arrays
            std::vector<const char*> inputNamePtrs;
            for (const auto& name : inputNames) {
                inputNamePtrs.push_back(name.c_str());
            }
            std::vector<const char*> outputNamePtrs;
            for (const auto& name : outputNames) {
                outputNamePtrs.push_back(name.c_str());
            }

            // Validate input count matches what we're providing
            if (inputNamePtrs.size() != 1) {
                lastError = "Model expects " + std::to_string(inputNamePtrs.size()) +
                           " inputs but only single-input models are currently supported";
                return result;
            }

            // Run inference
            auto outputs = session->Run(
                Ort::RunOptions{nullptr},
                inputNamePtrs.data(), &inputTensorOrt, 1,
                outputNamePtrs.data(), outputNamePtrs.size()
            );

            if (progress) progress(0.7f, "Post-processing results...");

            // Process outputs
            float frameRate = static_cast<float>(config.sampleRate) / config.hopLength;

            // Beat activation (first output)
            if (outputs.size() > 0) {
                auto& beatOutput = outputs[0];
                auto typeInfo = beatOutput.GetTensorTypeAndShapeInfo();
                if (typeInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                     lastError = "Model output 0 (beats) is not float type";
                     return result;
                }
                auto* beatData = beatOutput.GetTensorData<float>();
                auto beatShape = typeInfo.GetShape();

                // Copy activation
                size_t beatSize = 1;
                for (auto dim : beatShape) beatSize *= dim;
                result.beatActivation.assign(beatData, beatData + beatSize);

                // Peak picking for beats
                result.beats = peakPicking(result.beatActivation, config.beatThreshold,
                                          config.minBeatInterval, frameRate);
            }

            // Downbeat activation (second output)
            if (outputs.size() > 1) {
                auto& downbeatOutput = outputs[1];
                auto typeAndShape = downbeatOutput.GetTensorTypeAndShapeInfo();

                // Verify tensor element type is float before casting
                if (typeAndShape.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    std::cerr << "[OnnxBeatDetector] Warning: Downbeat output tensor is not float type, skipping\n";
                } else {
                    auto* downbeatData = downbeatOutput.GetTensorData<float>();
                    auto downbeatShape = typeAndShape.GetShape();

                    size_t downbeatSize = 1;
                    for (auto dim : downbeatShape) downbeatSize *= dim;
                    result.downbeatActivation.assign(downbeatData, downbeatData + downbeatSize);

                    result.downbeats = peakPicking(result.downbeatActivation, config.downbeatThreshold,
                                                   config.minBeatInterval * 2, frameRate);
                }
            }

            // Segment activation (third output for AllInOne)
            if (config.modelType == OnnxModelType::AllInOne && outputs.size() > 2) {
                auto& segmentOutput = outputs[2];
                auto segmentTypeInfo = segmentOutput.GetTensorTypeAndShapeInfo();

                // Verify tensor element type is float before casting
                if (segmentTypeInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    std::cerr << "[OnnxBeatDetector] Warning: Segment output tensor is not float type, skipping segment analysis\n";
                } else {
                auto* segmentData = segmentOutput.GetTensorData<float>();
                auto segmentShape = segmentTypeInfo.GetShape();

                size_t segmentSize = 1;
                for (auto dim : segmentShape) segmentSize *= dim;
                result.segmentActivation.assign(segmentData, segmentData + segmentSize);

                // Process segment boundaries and labels
                if (config.enableSegments && outputs.size() > 3) {
                    auto segmentBoundaries = peakPicking(result.segmentActivation,
                                                         config.segmentThreshold, 1.0f, frameRate);

                    // Get segment labels (fourth output)
                    auto& labelOutput = outputs[3];
                    auto labelTypeInfo = labelOutput.GetTensorTypeAndShapeInfo();

                    // Verify tensor element type is float before casting
                    if (labelTypeInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                        std::cerr << "[OnnxBeatDetector] Warning: Label output tensor is not float type, skipping segment labeling\n";
                    } else {
                    auto* labelData = labelOutput.GetTensorData<float>();
                    auto labelShape = labelTypeInfo.GetShape();

                    static const std::vector<std::string> SEGMENT_LABELS = {
                        "intro", "verse", "pre-chorus", "chorus", "post-chorus",
                        "bridge", "outro", "instrumental", "solo", "silence"
                    };

                    // Guard against empty tensor shape or null data
                    if (labelShape.empty() || !labelData) {
                        std::cerr << "[OnnxBeatDetector] Warning: Label tensor has empty shape or null data, skipping segment labeling\n";
                    } else {
                        // Check rank and dimensions
                        // Expected: [numFrames, numClasses] or [1, numFrames, numClasses]
                        int numClasses = 0;
                        int numFramesDim = 0;
                        size_t strideFrames = 0;
                        size_t strideClasses = 0;

                        if (labelShape.size() == 2) {
                            numFramesDim = static_cast<int>(labelShape[0]);
                            numClasses = static_cast<int>(labelShape[1]);
                            strideFrames = numClasses;
                            strideClasses = 1;
                        } else if (labelShape.size() == 3 && labelShape[0] == 1) {
                            numFramesDim = static_cast<int>(labelShape[1]);
                            numClasses = static_cast<int>(labelShape[2]);
                            strideFrames = numClasses;
                            strideClasses = 1;
                        } else {
                            std::cerr << "[OnnxBeatDetector] Warning: Unexpected label shape rank " << labelShape.size() << ", skipping\n";
                            numClasses = 0; // Skip loop
                        }

                        // Compute total label tensor size for bounds checking
                        size_t labelTensorSize = 1;
                        for (auto dim : labelShape) labelTensorSize *= dim;

                        // Skip segment labeling if no frames available
                        if (numFramesDim <= 0 || numClasses <= 0) {
                            std::cerr << "[OnnxBeatDetector] Warning: Label tensor has no frames or classes, skipping segment labeling\n";
                        } else {
                        for (size_t i = 0; i < segmentBoundaries.size(); ++i) {
                            MusicSegment seg;
                            seg.startTime = segmentBoundaries[i];
                            seg.endTime = (i + 1 < segmentBoundaries.size()) ?
                                          segmentBoundaries[i + 1] :
                                          resampled.size() / static_cast<double>(config.sampleRate);

                            // Find label with highest probability at segment start
                            int frameIdx = static_cast<int>(seg.startTime * frameRate);
                            // Clamp to model output frames (numFramesDim > 0 guaranteed by outer check)
                            frameIdx = std::clamp(frameIdx, 0, numFramesDim - 1);

                            float maxProb = -1e9f;
                            int maxClass = 0;
                            for (int c = 0; c < numClasses && c < static_cast<int>(SEGMENT_LABELS.size()); ++c) {
                                size_t labelIdx = static_cast<size_t>(frameIdx) * strideFrames + c * strideClasses;
                                if (labelIdx >= labelTensorSize) continue;  // Bounds check
                                float prob = labelData[labelIdx];
                                if (prob > maxProb) {
                                    maxProb = prob;
                                    maxClass = c;
                                }
                            }

                            seg.label = SEGMENT_LABELS[maxClass];
                            seg.confidence = 1.0f / (1.0f + std::exp(-maxProb));  // Sigmoid
                            result.segments.push_back(seg);
                        }
                        } // end else (numFramesDim > 0 && numClasses > 0)
                    }
                    } // end else (label tensor is float type)
                }
                } // end else (segment tensor is float type)
            }

            // Estimate BPM from beats
            result.bpm = estimateBPM(result.beats);

            if (progress) progress(1.0f, "Analysis complete");

        } catch (const Ort::Exception& e) {
            lastError = std::string("ONNX inference error: ") + e.what();
        } catch (const std::exception& e) {
            lastError = std::string("Inference error: ") + e.what();
        }

        return result;
#endif
    }
};

// OnnxBeatDetector public methods

OnnxBeatDetector::OnnxBeatDetector()
    : m_impl(std::make_unique<Impl>()) {
}

OnnxBeatDetector::~OnnxBeatDetector() = default;

OnnxBeatDetector::OnnxBeatDetector(OnnxBeatDetector&&) noexcept = default;
OnnxBeatDetector& OnnxBeatDetector::operator=(OnnxBeatDetector&&) noexcept = default;

bool OnnxBeatDetector::loadModel(const std::string& modelPath, const OnnxConfig& config) {
    TRACE_FUNC();
    return m_impl->loadModel(modelPath, config);
}

// Private implementation methods called from inline header methods
bool OnnxBeatDetector::isLoadedImpl() const {
    return m_impl->loaded;
}

OnnxModelType OnnxBeatDetector::getModelTypeImpl() const {
    return m_impl->config.modelType;
}

const OnnxConfig& OnnxBeatDetector::getConfigImpl() const {
    return m_impl->config;
}

void OnnxBeatDetector::setConfigImpl(const OnnxConfig& config) {
    // Check if mel extractor parameters changed - if so, recreate it
    bool needsNewExtractor = !m_impl->melExtractor ||
        m_impl->config.sampleRate != config.sampleRate ||
        m_impl->config.nMels != config.nMels ||
        m_impl->config.windowLength != config.windowLength ||
        m_impl->config.hopLength != config.hopLength ||
        m_impl->config.fmin != config.fmin ||
        m_impl->config.fmax != config.fmax;

    m_impl->config = config;

    if (needsNewExtractor) {
        m_impl->melExtractor = std::make_unique<MelSpectrogramExtractor>(
            config.sampleRate, config.nMels, config.windowLength,
            config.hopLength, config.fmin, config.fmax
        );
    }
}

BeatGrid OnnxBeatDetector::analyzeImpl(const std::vector<float>& samples, int sampleRate,
                                        ProgressCallback progress) {
    TRACE_FUNC();
    BeatGrid grid;

    OnnxAnalysisResult result = analyzeDetailedImpl(samples, sampleRate, progress);

    grid.setBeats(result.beats);
    grid.setBPM(result.bpm);
    grid.setAudioDuration(samples.size() / static_cast<double>(sampleRate));

    return grid;
}

OnnxAnalysisResult OnnxBeatDetector::analyzeDetailedImpl(const std::vector<float>& samples,
                                                          int sampleRate,
                                                          ProgressCallback progress) {
    TRACE_FUNC();
    return m_impl->runInference(samples, sampleRate, progress);
}

std::vector<double> OnnxBeatDetector::processChunkImpl(const std::vector<float>& chunk) {
    // For streaming, accumulate chunks and process when we have enough
    m_impl->streamBuffer.insert(m_impl->streamBuffer.end(), chunk.begin(), chunk.end());

    // Process when we have at least 0.5 seconds of audio
    int minSamples = m_impl->config.sampleRate / 2;
    if (static_cast<int>(m_impl->streamBuffer.size()) < minSamples) {
        return {};
    }

    // Run inference on buffer
    OnnxAnalysisResult result = m_impl->runInference(m_impl->streamBuffer,
                                                      m_impl->config.sampleRate, nullptr);

    // Calculate overlap parameters before processing beats
    double bufferDuration = m_impl->streamBuffer.size() / static_cast<double>(m_impl->config.sampleRate);
    // Keep last ~100ms for overlap with next chunk
    size_t overlapSamples = m_impl->config.sampleRate / 10;
    double overlapDuration = overlapSamples / static_cast<double>(m_impl->config.sampleRate);
    double cutoffTime = bufferDuration - overlapDuration;

    // Adjust beat times relative to stream position
    // Only include beats before the overlap region to avoid duplicates
    std::vector<double> adjustedBeats;
    for (double beat : result.beats) {
        if (beat < cutoffTime) {
            adjustedBeats.push_back(beat + m_impl->streamTime);
        }
    }

    // Update stream state
    m_impl->streamTime += bufferDuration;
    if (m_impl->streamBuffer.size() > overlapSamples) {
        // Move overlap samples to front - O(overlapSamples) instead of O(n) erase
        std::move(m_impl->streamBuffer.end() - overlapSamples,
                  m_impl->streamBuffer.end(),
                  m_impl->streamBuffer.begin());
        m_impl->streamBuffer.resize(overlapSamples);
        m_impl->streamTime -= overlapSamples / static_cast<double>(m_impl->config.sampleRate);
    } else {
        m_impl->streamBuffer.clear();
    }

    return adjustedBeats;
}

void OnnxBeatDetector::resetImpl() {
    m_impl->streamBuffer.clear();
    m_impl->streamTime = 0.0;
}

std::string OnnxBeatDetector::getLastErrorImpl() const {
    return m_impl->lastError;
}

std::string OnnxBeatDetector::getModelInfoImpl() const {
    std::ostringstream oss;
    oss << "Model: " << m_impl->modelPath << "\n";
    oss << "Type: ";
    switch (m_impl->config.modelType) {
        case OnnxModelType::Unknown: oss << "Unknown"; break;
        case OnnxModelType::BeatNet: oss << "BeatNet"; break;
        case OnnxModelType::AllInOne: oss << "All-In-One"; break;
        case OnnxModelType::TCN: oss << "TCN"; break;
        case OnnxModelType::Custom: oss << "Custom"; break;
    }
    oss << "\n";
    oss << "Inputs: ";
    for (size_t i = 0; i < m_impl->inputNames.size(); ++i) {
        oss << m_impl->inputNames[i] << " [";
        for (size_t j = 0; j < m_impl->inputShapes[i].size(); ++j) {
            if (j > 0) oss << ", ";
            oss << m_impl->inputShapes[i][j];
        }
        oss << "]";
        if (i + 1 < m_impl->inputNames.size()) oss << ", ";
    }
    oss << "\n";
    oss << "Outputs: ";
    for (size_t i = 0; i < m_impl->outputNames.size(); ++i) {
        oss << m_impl->outputNames[i];
        if (i + 1 < m_impl->outputNames.size()) oss << ", ";
    }
    oss << "\n";
    return oss.str();
}

bool OnnxBeatDetector::isOnnxRuntimeAvailable() {
#ifdef USE_ONNX
    return true;
#else
    return false;
#endif
}

std::vector<std::string> OnnxBeatDetector::getAvailableProviders() {
    std::vector<std::string> providers;
#ifdef USE_ONNX
    auto available = Ort::GetAvailableProviders();
    for (const auto& p : available) {
        providers.push_back(p);
    }
#endif
    return providers;
}

bool OnnxBeatDetector::isGPUEnabledImpl() const {
#ifdef USE_ONNX
    // Check the actual provider configured for this session (not build-time available providers)
    if (m_impl && m_impl->loaded) {
        return m_impl->activeProviderIsGpu;
    }
#endif
    return false;
}

std::string OnnxBeatDetector::getActiveProviderImpl() const {
#ifdef USE_ONNX
    if (m_impl && m_impl->loaded) {
        return m_impl->activeProvider;
    }
#endif
    return "None";
}

void OnnxBeatDetector::releaseGPUMemory() {
#ifdef USE_ONNX
    // Early return if impl is null
    if (!m_impl) {
        return;
    }
    // Reset the session to release GPU memory
    if (m_impl->session) {
        m_impl->session.reset();
        m_impl->loaded = false;
    }
#endif
}

void ensureOnnxBeatDetectorIsLinked() {
    // No-op; exists only to ensure this translation unit is present in builds
}

} // namespace BeatSync
