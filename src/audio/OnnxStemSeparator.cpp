/**
 * @file OnnxStemSeparator.cpp
 * @brief ONNX Runtime-based audio stem separator implementation
 *
 * Separates audio into stems (drums, bass, other, vocals) using neural networks.
 * Compatible with Demucs/HTDemucs models exported to ONNX format.
 */

#include "OnnxStemSeparator.h"
#include "tracing/Tracing.h"

#ifdef _WIN32
#define NOMINMAX
#endif

#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>
#include <mutex>

#include "utils/DebugLogger.h"

// Debug logging helper - writes to file since Windows GUI apps don't show stderr
static void debugLog(const std::string& msg) {
    BeatSync::DebugLogger::getInstance().log(msg);
}

#ifdef _WIN32
#include <Windows.h>
#endif

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#ifdef USE_LIBSAMPLERATE
#include <samplerate.h>
#endif

namespace BeatSync {

struct OnnxStemSeparator::Impl {
#ifdef USE_ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
#endif

    StemSeparatorConfig config;
    std::string lastError;
    std::string modelPath;
    bool loaded = false;
    bool gpuEnabled_ = false;

    // Model info
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;

    Impl() {
#ifdef USE_ONNX
        try {
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BeatSyncStemSep");
        } catch (const std::exception& e) {
            lastError = std::string("Failed to initialize ONNX Runtime: ") + e.what();
        }
#endif
    }

    ~Impl() {
#ifdef USE_ONNX
        auto start = std::chrono::steady_clock::now();
        try {
            // Explicitly reset session first to trigger ONNX Runtime cleanup and flush any pending GPU work
            if (session) session.reset();
            if (sessionOptions) sessionOptions.reset();
            if (env) env.reset();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            // Log if cleanup takes unusually long (indicates GPU memory pressure)
            if (elapsed > 100) {
                std::cerr << "[BeatSync] OnnxStemSeparator cleanup took " << elapsed << "ms" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[BeatSync] Warning: Exception during OnnxStemSeparator cleanup: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[BeatSync] Warning: Unknown exception during OnnxStemSeparator cleanup" << std::endl;
        }
#endif
    }

    bool loadModel(const std::string& path, const StemSeparatorConfig& cfg) {
#ifndef USE_ONNX
        lastError = "ONNX Runtime not available. Rebuild with USE_ONNX=ON";
        return false;
#else
        if (!env) {
            lastError = "ONNX Runtime environment not initialized";
            return false;
        }

        try {
            sessionOptions = std::make_unique<Ort::SessionOptions>();
            sessionOptions->SetIntraOpNumThreads(cfg.numThreads > 0 ? cfg.numThreads : 0);
            sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Memory configuration to prevent GPU memory exhaustion during long processing
            // Disable memory pattern optimization to reduce memory pooling
            sessionOptions->DisableMemPattern();
            // Don't use environment allocators (prefer per-session allocation)
            sessionOptions->AddConfigEntry("session.use_env_allocators", "0");

            // Try to use GPU if requested
            if (cfg.useGPU) {
                gpuEnabled_ = false;
                std::cerr << "[BeatSync] GPU acceleration requested (device_id=" << cfg.gpuDeviceId << ")" << std::endl;
                try {
                    OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
                    const OrtApi& ortApi = Ort::GetApi();
                    OrtStatus* status = ortApi.CreateCUDAProviderOptions(&cudaOptions);
                    if (status != nullptr) {
                        const char* msg = ortApi.GetErrorMessage(status);
                        std::cerr << "[BeatSync] CreateCUDAProviderOptions failed (StemSep): " << (msg ? msg : "unknown error") << std::endl;
                        ortApi.ReleaseStatus(status);
                    } else if (cudaOptions != nullptr) {
                        const char* keys[] = {"device_id"};
                        char deviceIdStr[16];
                        snprintf(deviceIdStr, sizeof(deviceIdStr), "%d", cfg.gpuDeviceId);
                        const char* values[] = {deviceIdStr};
                        status = ortApi.UpdateCUDAProviderOptions(cudaOptions, keys, values, 1);
                        if (status == nullptr) {
                            status = ortApi.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(*sessionOptions), cudaOptions);
                            if (status == nullptr) {
                                gpuEnabled_ = true;
                                std::cerr << "[BeatSync] CUDA execution provider enabled successfully (StemSep)" << std::endl;
                            } else {
                                const char* msg = ortApi.GetErrorMessage(status);
                                std::cerr << "[BeatSync] CUDA provider append failed (StemSep): " << (msg ? msg : "unknown error") << std::endl;
                                ortApi.ReleaseStatus(status);
                            }
                        } else {
                            const char* msg = ortApi.GetErrorMessage(status);
                            std::cerr << "[BeatSync] CUDA options update failed (StemSep): " << (msg ? msg : "unknown error") << std::endl;
                            ortApi.ReleaseStatus(status);
                        }
                        ortApi.ReleaseCUDAProviderOptions(cudaOptions);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[BeatSync] CUDA provider failed with exception (StemSep): " << e.what() << std::endl;
#ifdef _WIN32
                    try {
                        std::cerr << "[BeatSync] CUDA failed, attempting DirectML fallback (StemSep)" << std::endl;
                        sessionOptions->AppendExecutionProvider("DML", {});
                        gpuEnabled_ = true;
                        std::cerr << "[BeatSync] DirectML enabled successfully (StemSep)" << std::endl;
                    } catch (const std::exception& e2) {
                        std::cerr << "[BeatSync] DirectML fallback failed with exception (StemSep): " << e2.what() << std::endl;
                        // Fall back to CPU
                    } catch (...) {
                        std::cerr << "[BeatSync] DirectML fallback failed with unknown exception (StemSep)" << std::endl;
                        // Fall back to CPU
                    }
#endif
                } catch (...) {
                    std::cerr << "[BeatSync] CUDA provider failed with unknown exception (StemSep)" << std::endl;
#ifdef _WIN32
                    try {
                        std::cerr << "[BeatSync] CUDA failed, attempting DirectML fallback (StemSep)" << std::endl;
                        sessionOptions->AppendExecutionProvider("DML", {});
                        gpuEnabled_ = true;
                        std::cerr << "[BeatSync] DirectML enabled successfully (StemSep)" << std::endl;
                    } catch (const std::exception& e2) {
                        std::cerr << "[BeatSync] DirectML fallback failed with exception (StemSep): " << e2.what() << std::endl;
                        // Fall back to CPU
                    } catch (...) {
                        std::cerr << "[BeatSync] DirectML fallback failed with unknown exception (StemSep)" << std::endl;
                        // Fall back to CPU
                    }
#endif
                }
            }

            // Load model
#ifdef _WIN32
            int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, NULL, 0);
            std::wstring widePath(wlen, 0);
            MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &widePath[0], wlen);
            if (wlen > 0) widePath.resize(wlen - 1); // Remove trailing null
            session = std::make_unique<Ort::Session>(*env, widePath.c_str(), *sessionOptions);
#else
            session = std::make_unique<Ort::Session>(*env, path.c_str(), *sessionOptions);
#endif

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

            config = cfg;
            modelPath = path;
            loaded = true;

            // Debug: log model info
            debugLog("[BeatSync] StemSeparator model loaded: " + path);
            {
                std::ostringstream oss;
                oss << "[BeatSync] StemSeparator inputs: ";
                for (size_t i = 0; i < inputNames.size(); ++i) {
                    oss << inputNames[i] << " [";
                    for (size_t j = 0; j < inputShapes[i].size(); ++j) {
                        if (j > 0) oss << ", ";
                        oss << inputShapes[i][j];
                    }
                    oss << "] ";
                }
                debugLog(oss.str());
            }
            {
                std::ostringstream oss;
                oss << "[BeatSync] StemSeparator outputs: ";
                for (size_t i = 0; i < outputNames.size(); ++i) {
                    oss << outputNames[i] << " [";
                    for (size_t j = 0; j < outputShapes[i].size(); ++j) {
                        if (j > 0) oss << ", ";
                        oss << outputShapes[i][j];
                    }
                    oss << "] ";
                }
                debugLog(oss.str());
            }

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

    std::vector<float> resampleAudio(const std::vector<float>& samples, int srcRate, int dstRate, int channels) {
        if (srcRate == dstRate) return samples;

        double ratio = static_cast<double>(dstRate) / srcRate;
        size_t srcFrames = samples.size() / channels;
        size_t dstFrames = static_cast<size_t>(srcFrames * ratio);

        std::vector<float> resampled(dstFrames * channels);

#ifdef USE_LIBSAMPLERATE
        // High-quality bandlimited resampling with libsamplerate
        SRC_DATA srcData;
        srcData.data_in = samples.data();
        srcData.input_frames = static_cast<long>(srcFrames);
        srcData.data_out = resampled.data();
        srcData.output_frames = static_cast<long>(dstFrames);
        srcData.src_ratio = ratio;

        int error = src_simple(&srcData, SRC_SINC_BEST_QUALITY, channels);
        if (error == 0) {
            resampled.resize(srcData.output_frames_gen * channels);
            return resampled;
        }
        debugLog(std::string("[BeatSync] libsamplerate error: ") + src_strerror(error) + " - falling back to linear");
#endif
        // Fallback: linear interpolation with anti-aliasing pre-filter
        std::vector<float> filtered = samples;
        if (ratio < 1.0) {  // Downsampling
            float alpha = static_cast<float>(ratio * 0.5);
            for (size_t i = channels; i < filtered.size() - channels; ++i) {
                filtered[i] = filtered[i] * alpha + filtered[i - channels] * (1 - alpha);
            }
        }

        for (size_t i = 0; i < dstFrames; ++i) {
            double srcIdx = i / ratio;
            size_t idx0 = static_cast<size_t>(srcIdx);
            size_t idx1 = std::min(idx0 + 1, srcFrames - 1);
            double frac = srcIdx - idx0;

            for (int c = 0; c < channels; ++c) {
                float v0 = filtered[idx0 * channels + c];
                float v1 = filtered[idx1 * channels + c];
                resampled[i * channels + c] = static_cast<float>(v0 * (1.0 - frac) + v1 * frac);
            }
        }

        return resampled;
    }

    std::vector<float> monoToStereo(const std::vector<float>& mono) {
        std::vector<float> stereo(mono.size() * 2);
        for (size_t i = 0; i < mono.size(); ++i) {
            stereo[i * 2] = mono[i];
            stereo[i * 2 + 1] = mono[i];
        }
        return stereo;
    }

    // Apply overlap-add for smooth segment transitions
    void overlapAdd(std::vector<float>& output, const std::vector<float>& segment,
                    size_t offset, size_t fadeLength, int channels) {
        size_t segmentFrames = segment.size() / channels;

        float denom = (fadeLength > 0) ? static_cast<float>(fadeLength) : 1.0f;
        for (size_t i = 0; i < segmentFrames; ++i) {
            size_t outIdx = offset + i;
            if (outIdx >= output.size() / channels) break;

            // Compute fade weights for overlap region
            float fadeIn = 1.0f;
            float fadeOut = 1.0f;

            if (fadeLength > 0) {
                if (offset > 0 && i < fadeLength) {
                    fadeIn = static_cast<float>(i) / denom;
                }
                if (i >= segmentFrames - fadeLength) {
                    fadeOut = static_cast<float>(segmentFrames - 1 - i) / denom;
                }
            }

            float weight = fadeIn * fadeOut;

            for (int c = 0; c < channels; ++c) {
                size_t inPos = i * channels + c;
                size_t outPos = outIdx * channels + c;

                if (offset == 0 || i >= fadeLength || fadeLength == 0) {
                    // No overlap, direct copy
                    output[outPos] = segment[inPos];
                } else {
                    // Overlap region, blend
                    output[outPos] = output[outPos] * (1.0f - weight) + segment[inPos] * weight;
                }
            }
        }
    }

    StemSeparationResult runInference(const std::vector<float>& stereoSamples, int sampleRate,
                                      StemProgressCallback progress) {
        StemSeparationResult result;

#ifndef USE_ONNX
        lastError = "ONNX Runtime not available";
        return result;
#else
        if (!loaded || !session) {
            lastError = "Model not loaded";
            return result;
        }

        try {
            if (progress && !progress(0.05f, "Preparing audio...")) {
                lastError = "Stem separation cancelled by user.";
                return result;
            }

            // Resample to model's expected sample rate
            std::vector<float> resampled = resampleAudio(stereoSamples, sampleRate, config.sampleRate, 2);
            size_t totalFrames = resampled.size() / 2;

            result.sampleRate = config.sampleRate;
            result.duration = static_cast<double>(totalFrames) / config.sampleRate;

            // Initialize output stems
            for (int s = 0; s < 4; ++s) {
                result.stems[s].resize(totalFrames * 2, 0.0f);
            }

            // Process in segments with overlap
            // Use the model's expected input size from the input shape
            size_t modelExpectedSamples = 343980;  // From model input shape [1, 2, 343980]
            if (!inputShapes.empty() && inputShapes[0].size() >= 3 && inputShapes[0][2] > 0) {
                modelExpectedSamples = static_cast<size_t>(inputShapes[0][2]);
            }

            // Defensive validation: ensure modelExpectedSamples is at least 1
            if (modelExpectedSamples == 0) {
                debugLog("[BeatSync] Warning: modelExpectedSamples is 0, defaulting to 1");
                modelExpectedSamples = 1;
            }

            // Guard against invalid values: ensure segmentFrames >= 1 and hopFrames at least 1
            size_t segmentFrames = modelExpectedSamples;  // Use model's expected size
            size_t overlapFrames = segmentFrames / 4;     // 25% overlap
            // Ensure hopFrames is at least 1 to prevent division by zero
            size_t hopFrames = (segmentFrames > overlapFrames) ? (segmentFrames - overlapFrames) : 1;

            // Compute numSegments with guard against division by zero
            size_t numSegments = (hopFrames > 0) ? ((totalFrames + hopFrames - 1) / hopFrames) : 1;
            if (numSegments == 0) numSegments = 1;

            {
                std::ostringstream oss;
                oss << "[BeatSync] StemSeparator processing: totalFrames=" << totalFrames
                    << " segmentFrames=" << segmentFrames << " hopFrames=" << hopFrames
                    << " numSegments=" << numSegments;
                debugLog(oss.str());
            }

            for (size_t seg = 0; seg < numSegments; ++seg) {
                if (progress) {
                    float p = 0.1f + 0.8f * static_cast<float>(seg) / numSegments;
                    if (!progress(p, "Processing segment " + std::to_string(seg + 1) + "/" + std::to_string(numSegments))) {
                        lastError = "Stem separation cancelled by user.";
                        return result;
                    }
                }

                size_t startFrame = seg * hopFrames;
                size_t endFrame = std::min(startFrame + segmentFrames, totalFrames);
                size_t actualFrames = endFrame - startFrame;

                // Extract segment (pad if needed)
                std::vector<float> segment(segmentFrames * 2, 0.0f);
                for (size_t i = 0; i < actualFrames; ++i) {
                    segment[i * 2] = resampled[(startFrame + i) * 2];
                    segment[i * 2 + 1] = resampled[(startFrame + i) * 2 + 1];
                }

                // Prepare input tensor: (batch=1, channels=2, samples)
                std::vector<int64_t> inputShape = {1, 2, static_cast<int64_t>(segmentFrames)};
                size_t inputSize = 2 * segmentFrames;

                // Reshape from interleaved to planar: (2, samples)
                std::vector<float> inputTensor(inputSize);
                for (size_t i = 0; i < segmentFrames; ++i) {
                    inputTensor[i] = segment[i * 2];                    // Left channel
                    inputTensor[segmentFrames + i] = segment[i * 2 + 1]; // Right channel
                }

                // Debug: check input data on first segment
                if (seg == 0) {
                    float inMin = 1e9f, inMax = -1e9f, inSum = 0.0f;
                    for (size_t i = 0; i < inputSize; ++i) {
                        inMin = std::min(inMin, inputTensor[i]);
                        inMax = std::max(inMax, inputTensor[i]);
                        inSum += std::abs(inputTensor[i]);
                    }
                    std::ostringstream oss;
                    oss << "[BeatSync] StemSeparator input (seg 0): size=" << inputSize
                        << " min=" << inMin << " max=" << inMax
                        << " meanAbs=" << (inSum / inputSize);
                    debugLog(oss.str());
                }

                // Create ONNX tensor
                // Use OrtDeviceAllocator instead of OrtArenaAllocator to prevent memory pooling
                // which can cause GPU memory exhaustion during long processing sessions
                auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
                auto inputTensorOrt = Ort::Value::CreateTensor<float>(
                    memoryInfo, inputTensor.data(), inputSize,
                    inputShape.data(), inputShape.size()
                );

                // Prepare input/output names
                std::vector<const char*> inputNamePtrs;
                for (const auto& name : inputNames) {
                    inputNamePtrs.push_back(name.c_str());
                }
                std::vector<const char*> outputNamePtrs;
                for (const auto& name : outputNames) {
                    outputNamePtrs.push_back(name.c_str());
                }

                // Run inference
                auto outputs = session->Run(
                    Ort::RunOptions{nullptr},
                    inputNamePtrs.data(), &inputTensorOrt, 1,
                    outputNamePtrs.data(), outputNamePtrs.size()
                );

                // Process output: (batch=1, stems=4, channels=2, samples)
                if (!outputs.empty()) {
                    auto& stemOutput = outputs[0];
                    auto* stemData = stemOutput.GetTensorData<float>();
                    auto stemShape = stemOutput.GetTensorTypeAndShapeInfo().GetShape();

                    // Debug: log output shape on first segment
                    if (seg == 0) {
                        std::ostringstream oss;
                        oss << "[BeatSync] StemSeparator output shape: [";
                        for (size_t i = 0; i < stemShape.size(); ++i) {
                            if (i > 0) oss << ", ";
                            oss << stemShape[i];
                        }
                        oss << "]";
                        debugLog(oss.str());
                    }

                    // Extract each stem and overlap-add to result
                    int numStems = (stemShape.size() > 1) ? static_cast<int>(stemShape[1]) : kNumStems;
                    int numChannels = (stemShape.size() > 2) ? static_cast<int>(stemShape[2]) : 2;
                    size_t outputFrames = (stemShape.size() > 3) ? static_cast<size_t>(stemShape[3]) : segmentFrames;

                    // Debug: check raw output on first segment
                    if (seg == 0) {
                        size_t totalOutputSize = 1;
                        for (auto d : stemShape) totalOutputSize *= d;
                        float outMin = 1e9f, outMax = -1e9f, outSum = 0.0f;
                        for (size_t i = 0; i < totalOutputSize; ++i) {
                            outMin = std::min(outMin, stemData[i]);
                            outMax = std::max(outMax, stemData[i]);
                            outSum += std::abs(stemData[i]);
                        }
                        std::ostringstream oss;
                        float meanAbs = totalOutputSize > 0 ? (outSum / totalOutputSize) : 0.0f;
                        oss << "[BeatSync] StemSeparator raw output (seg 0): totalSize=" << totalOutputSize
                            << " min=" << outMin << " max=" << outMax
                            << " meanAbs=" << meanAbs
                            << " numStems=" << numStems << " numChannels=" << numChannels
                            << " outputFrames=" << outputFrames;
                        debugLog(oss.str());
                    }

                    // Validate tensor dimensions match expectations
                    size_t expectedSize = static_cast<size_t>(numStems) * numChannels * outputFrames;
                    size_t totalOutputSize = 1;
                    for (auto d : stemShape) totalOutputSize *= d;
                    if (totalOutputSize < expectedSize) {
                        debugLog("[BeatSync] Warning: Output tensor size mismatch, skipping segment");
                        continue;
                    }

                    for (int s = 0; s < std::min(numStems, kNumStems); ++s) {
                        // Convert from planar to interleaved
                        std::vector<float> stemSegment(actualFrames * 2);
                        for (size_t i = 0; i < actualFrames && i < outputFrames; ++i) {
                            size_t baseIdx = s * numChannels * outputFrames;
                            size_t leftIdx = baseIdx + i;
                            size_t rightIdx = baseIdx + outputFrames + i;

                            // Bounds checking for stemData access
                            if (leftIdx >= totalOutputSize || rightIdx >= totalOutputSize) {
                                debugLog("[BeatSync] Warning: Stem data index out of bounds, zero-filling remaining samples");
                                // Zero-fill the rest of stemSegment from i to end
                                std::fill(stemSegment.begin() + i * 2, stemSegment.end(), 0.0f);
                                break;
                            }

                            stemSegment[i * 2] = stemData[leftIdx];                    // Left
                            stemSegment[i * 2 + 1] = stemData[rightIdx]; // Right
                        }

                        // Overlap-add
                        overlapAdd(result.stems[s], stemSegment, startFrame, overlapFrames, 2);
                    }
                }
            }

            if (progress && !progress(0.95f, "Normalizing output...")) {
                lastError = "Stem separation cancelled by user.";
                return result;
            }

            // Debug: analyze stem content before normalization
            const char* stemNames[] = {"drums", "bass", "other", "vocals"};
            for (int s = 0; s < 4; ++s) {
                if (result.stems[s].empty()) continue;
                float minV = 1e9f, maxV = -1e9f, sumAbs = 0.0f;
                for (float v : result.stems[s]) {
                    minV = std::min(minV, v);
                    maxV = std::max(maxV, v);
                    sumAbs += std::abs(v);
                }
                std::ostringstream oss;
                oss << "[BeatSync] Stem[" << s << "] (" << stemNames[s] << "): samples="
                    << result.stems[s].size() << " min=" << minV << " max=" << maxV
                    << " meanAbs=" << (sumAbs / result.stems[s].size());
                debugLog(oss.str());
            }

            // Normalize output if requested
            if (config.normalize) {
                for (int s = 0; s < 4; ++s) {
                    float maxAbs = 0.0f;
                    for (float v : result.stems[s]) {
                        maxAbs = std::max(maxAbs, std::abs(v));
                    }
                    if (maxAbs > config.clipThreshold) {
                        float scale = config.clipThreshold / maxAbs;
                        for (float& v : result.stems[s]) {
                            v *= scale;
                        }
                    }
                }
            }

            if (progress) progress(1.0f, "Separation complete");

        } catch (const Ort::Exception& e) {
            lastError = std::string("ONNX inference error: ") + e.what();
        } catch (const std::exception& e) {
            lastError = std::string("Inference error: ") + e.what();
        }

        return result;
#endif
    }
};

// Public interface implementation

OnnxStemSeparator::OnnxStemSeparator()
    : m_impl(std::make_unique<Impl>()) {
}

OnnxStemSeparator::~OnnxStemSeparator() {
    m_impl.reset();
}

OnnxStemSeparator::OnnxStemSeparator(OnnxStemSeparator&&) noexcept = default;
OnnxStemSeparator& OnnxStemSeparator::operator=(OnnxStemSeparator&&) noexcept = default;

bool OnnxStemSeparator::loadModel(const std::string& modelPath, const StemSeparatorConfig& config) {
    TRACE_FUNC();
    return m_impl->loadModel(modelPath, config);
}

bool OnnxStemSeparator::isLoaded() const {
    return m_impl->loaded;
}

const StemSeparatorConfig& OnnxStemSeparator::getConfig() const {
    return m_impl->config;
}

void OnnxStemSeparator::setConfig(const StemSeparatorConfig& config) {
    m_impl->config = config;
}

StemSeparationResult OnnxStemSeparator::separate(const std::vector<float>& samples, int sampleRate,
                                                  StemProgressCallback progress) {
    TRACE_FUNC();
    return m_impl->runInference(samples, sampleRate, progress);
}

StemSeparationResult OnnxStemSeparator::separateMono(const std::vector<float>& monoSamples, int sampleRate,
                                                      StemProgressCallback progress) {
    TRACE_FUNC();
    std::vector<float> stereo = m_impl->monoToStereo(monoSamples);
    return separate(stereo, sampleRate, progress);
}

std::vector<float> OnnxStemSeparator::extractStem(const std::vector<float>& samples, int sampleRate,
                                                   StemType stem, StemProgressCallback progress) {
    TRACE_FUNC();
    StemSeparationResult result = separate(samples, sampleRate, progress);
    return result.getMonoStem(stem);
}

std::string OnnxStemSeparator::getLastError() const {
    return m_impl->lastError;
}

std::string OnnxStemSeparator::getModelInfo() const {
    std::ostringstream oss;
    oss << "Model: " << m_impl->modelPath << "\n";
    oss << "Sample Rate: " << m_impl->config.sampleRate << " Hz\n";
    oss << "Segment Length: " << m_impl->config.segmentLength << " samples\n";
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
    oss << "Stems: drums, bass, other, vocals\n";
    return oss.str();
}

bool OnnxStemSeparator::isOnnxRuntimeAvailable() {
#ifdef USE_ONNX
    return true;
#else
    return false;
#endif
}

void ensureOnnxStemSeparatorIsLinked() {
    // No-op; exists only to ensure this translation unit is present in builds
}

} // namespace BeatSync
