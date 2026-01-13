/**
 * @file OnnxStemSeparator.cpp
 * @brief ONNX Runtime-based audio stem separator implementation
 *
 * Separates audio into stems (drums, bass, other, vocals) using neural networks.
 * Compatible with Demucs/HTDemucs models exported to ONNX format.
 */

#include "OnnxStemSeparator.h"
#include "tracing/Tracing.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <iostream>

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
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

            // Try to use GPU if requested
            if (cfg.useGPU) {
                try {
                    OrtCUDAProviderOptions cudaOptions;
                    cudaOptions.device_id = cfg.gpuDeviceId;
                    sessionOptions->AppendExecutionProvider_CUDA(cudaOptions);
                } catch (...) {
#ifdef _WIN32
                    try {
                        sessionOptions->AppendExecutionProvider("DML", {});
                    } catch (...) {
                        // Fall back to CPU
                    }
#endif
                }
            }

            // Load model
#ifdef _WIN32
            std::wstring widePath(path.begin(), path.end());
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

        // Linear interpolation resampling (per channel)
        for (size_t i = 0; i < dstFrames; ++i) {
            double srcIdx = i / ratio;
            size_t idx0 = static_cast<size_t>(srcIdx);
            size_t idx1 = std::min(idx0 + 1, srcFrames - 1);
            double frac = srcIdx - idx0;

            for (int c = 0; c < channels; ++c) {
                float v0 = samples[idx0 * channels + c];
                float v1 = samples[idx1 * channels + c];
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

        for (size_t i = 0; i < segmentFrames; ++i) {
            size_t outIdx = offset + i;
            if (outIdx >= output.size() / channels) break;

            // Compute fade weights for overlap region
            float fadeIn = 1.0f;
            float fadeOut = 1.0f;

            if (offset > 0 && i < fadeLength) {
                fadeIn = static_cast<float>(i) / fadeLength;
            }
            if (i >= segmentFrames - fadeLength) {
                fadeOut = static_cast<float>(segmentFrames - 1 - i) / fadeLength;
            }

            float weight = fadeIn * fadeOut;

            for (int c = 0; c < channels; ++c) {
                size_t inPos = i * channels + c;
                size_t outPos = outIdx * channels + c;

                if (offset == 0 || i >= fadeLength) {
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
            if (progress) progress(0.05f, "Preparing audio...");

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
            size_t segmentFrames = config.segmentLength;
            size_t overlapFrames = config.overlap;
            size_t hopFrames = segmentFrames - overlapFrames;

            size_t numSegments = (totalFrames + hopFrames - 1) / hopFrames;
            if (numSegments == 0) numSegments = 1;

            for (size_t seg = 0; seg < numSegments; ++seg) {
                if (progress) {
                    float p = 0.1f + 0.8f * static_cast<float>(seg) / numSegments;
                    progress(p, "Processing segment " + std::to_string(seg + 1) + "/" + std::to_string(numSegments));
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

                // Create ONNX tensor
                auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
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

                    // Extract each stem and overlap-add to result
                    int numStems = (stemShape.size() > 1) ? static_cast<int>(stemShape[1]) : 4;
                    int numChannels = (stemShape.size() > 2) ? static_cast<int>(stemShape[2]) : 2;
                    size_t outputFrames = (stemShape.size() > 3) ? static_cast<size_t>(stemShape[3]) : segmentFrames;

                    for (int s = 0; s < std::min(numStems, 4); ++s) {
                        // Convert from planar to interleaved
                        std::vector<float> stemSegment(actualFrames * 2);
                        for (size_t i = 0; i < actualFrames && i < outputFrames; ++i) {
                            size_t baseIdx = s * numChannels * outputFrames;
                            stemSegment[i * 2] = stemData[baseIdx + i];                    // Left
                            stemSegment[i * 2 + 1] = stemData[baseIdx + outputFrames + i]; // Right
                        }

                        // Overlap-add
                        overlapAdd(result.stems[s], stemSegment, startFrame, overlapFrames, 2);
                    }
                }
            }

            if (progress) progress(0.95f, "Normalizing output...");

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

OnnxStemSeparator::~OnnxStemSeparator() = default;

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
