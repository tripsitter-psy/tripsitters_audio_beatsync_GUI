#include "AudioAnalyzer.h"
#include "BeatNetBridge.h"
#include "StemSeparator.h"

#ifdef USE_ONNX
#include "OnnxBeatDetector.h"
#include "OnnxStemSeparator.h"
#endif

#include <cmath>
#include <algorithm>
#include <iostream>
#include <filesystem>

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#endif

// FFmpeg includes
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

namespace BeatSync {

AudioAnalyzer::AudioAnalyzer()
    : m_sensitivity(0.5)
    , m_analysisMode(AnalysisMode::Energy)
{
}

AudioAnalyzer::~AudioAnalyzer() {
}

void AudioAnalyzer::setAnalysisMode(AnalysisMode mode) {
    m_analysisMode = mode;
}

void AudioAnalyzer::setPythonPath(const std::string& pythonPath) {
    m_pythonPath = pythonPath;
}

BeatGrid AudioAnalyzer::analyze(const std::string& audioFilePath) {
    m_lastError.clear();

    // Dispatch based on analysis mode
    switch (m_analysisMode) {
        case AnalysisMode::BeatNet:
            std::cout << "Using BeatNet analysis mode...\n";
            return analyzeWithBeatNet(audioFilePath);
            
        case AnalysisMode::DemucsPlus:
            std::cout << "Using Demucs + BeatNet analysis mode...\n";
            return analyzeWithDemucsPlus(audioFilePath);
            
        case AnalysisMode::Energy:
        default:
            // Fall through to energy-based analysis
            break;
    }

    // Delegate energy-based analysis to helper
    return analyzeEnergy(audioFilePath);
}

BeatGrid AudioAnalyzer::analyzeWithBeatNet(const std::string& audioFilePath) {
#ifdef USE_ONNX
    // Try ONNX-based detection first (no Python needed)
    OnnxBeatDetector onnxDetector;
    
    // Look for ONNX model in standard locations
    std::vector<std::string> modelSearchPaths = {
        "models/beatnet_tcn.onnx",
        "../models/beatnet_tcn.onnx",
        "../../models/beatnet_tcn.onnx",
    };
    
    #ifdef _WIN32
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        modelSearchPaths.push_back((exeDir / "models" / "beatnet_tcn.onnx").string());
        modelSearchPaths.push_back((exeDir.parent_path() / "models" / "beatnet_tcn.onnx").string());
    }
    #endif
    
    for (const auto& modelPath : modelSearchPaths) {
        if (std::filesystem::exists(modelPath)) {
            std::cout << "Found ONNX model: " << modelPath << std::endl;
            if (onnxDetector.initialize(modelPath)) {
                BeatGrid grid = onnxDetector.analyze(audioFilePath);
                if (!grid.isEmpty()) {
                    std::cout << "ONNX beat detection successful!" << std::endl;
                    return grid;
                }
            }
            break;
        }
    }
    std::cout << "ONNX model not found or failed, falling back to Python BeatNet..." << std::endl;
#endif
    
    // Fall back to Python-based BeatNet
    BeatNetBridge bridge;
    if (!m_pythonPath.empty()) {
        bridge.setPythonPath(m_pythonPath);
    }
    
    BeatGrid grid = bridge.analyze(audioFilePath);
    
    if (grid.isEmpty()) {
        m_lastError = bridge.getLastError();
        std::cout << "BeatNet failed, falling back to energy-based detection: " << m_lastError << std::endl;
        
        // Fallback to energy-based without mutating analysis mode
        return analyzeEnergy(audioFilePath);
    }
    
    // Set audio duration
    AudioData audio = loadAudioFile(audioFilePath);
    if (!audio.samples.empty()) {
        grid.setAudioDuration(audio.duration);
    }
    
    return grid;
}

BeatGrid AudioAnalyzer::analyzeWithDemucsPlus(const std::string& audioFilePath) {
    std::string analysisTarget = audioFilePath;  // Default to original audio
    
    // Create temp directory for stems
    std::string tempDir = std::filesystem::temp_directory_path().string();
    #ifdef _WIN32
    tempDir += "\\beatsync_stems\\";
    #else
    tempDir += "/beatsync_stems/";
    #endif
    
#ifdef USE_ONNX
    // Try ONNX-based stem separation first
    OnnxStemSeparator onnxSeparator;
    
    std::vector<std::string> modelSearchPaths = {
        "models/demucs_htdemucs.onnx",
        "models/stem_separator_simple.onnx",
        "../models/demucs_htdemucs.onnx",
    };
    
    #ifdef _WIN32
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        modelSearchPaths.insert(modelSearchPaths.begin(), 
            (exeDir / "models" / "demucs_htdemucs.onnx").string());
        modelSearchPaths.insert(modelSearchPaths.begin() + 1,
            (exeDir / "models" / "stem_separator_simple.onnx").string());
    }
    #endif
    
    bool onnxSeparated = false;
    for (const auto& modelPath : modelSearchPaths) {
        if (std::filesystem::exists(modelPath)) {
            std::cout << "Found ONNX stem separator: " << modelPath << std::endl;
            if (onnxSeparator.initialize(modelPath)) {
                std::cout << "Separating stems with ONNX..." << std::endl;
                OnnxStemPaths onnxStems = onnxSeparator.separate(audioFilePath, tempDir);
                if (onnxStems.success && !onnxStems.drums.empty() && 
                    std::filesystem::exists(onnxStems.drums)) {
                    std::cout << "Using ONNX drums stem: " << onnxStems.drums << std::endl;
                    analysisTarget = onnxStems.drums;
                    onnxSeparated = true;
                }
            }
            break;
        }
    }
    
    if (!onnxSeparated) {
        std::cout << "ONNX stem separation not available, trying Python Demucs..." << std::endl;
    }
#endif

    // Fall back to Python-based Demucs if ONNX didn't work
    if (analysisTarget == audioFilePath) {
        StemSeparator separator;
        if (!m_pythonPath.empty()) {
            separator.setPythonPath(m_pythonPath);
        }
        
        std::cout << "Separating stems with Demucs..." << std::endl;
        StemPaths stems = separator.separate(audioFilePath, tempDir);
        
        if (!stems.drums.empty() && std::filesystem::exists(stems.drums)) {
            std::cout << "Using drums stem for beat detection: " << stems.drums << std::endl;
            analysisTarget = stems.drums;
        } else {
            std::cout << "Demucs separation failed or drums stem not available: " 
                      << separator.getLastError() << std::endl;
            std::cout << "Falling back to full audio for BeatNet..." << std::endl;
        }
    }
    
    // Step 2: Run BeatNet on drums stem (or original audio if separation failed)
    BeatNetBridge bridge;
    if (!m_pythonPath.empty()) {
        bridge.setPythonPath(m_pythonPath);
    }
    
    BeatGrid grid = bridge.analyze(analysisTarget);
    
    if (grid.isEmpty()) {
        m_lastError = bridge.getLastError();
        std::cout << "BeatNet failed, falling back to energy-based detection: " << m_lastError << std::endl;
        
        // Final fallback to energy-based without mutating analysis mode
        return analyzeEnergy(audioFilePath);
    }
    
    // Set audio duration from original file
    AudioData audio = loadAudioFile(audioFilePath);
    if (!audio.samples.empty()) {
        grid.setAudioDuration(audio.duration);
    }
    
    return grid;
}

void AudioAnalyzer::setSensitivity(double sensitivity) {
    m_sensitivity = std::max(0.0, std::min(1.0, sensitivity));
}

std::string AudioAnalyzer::getLastError() const {
    return m_lastError;
}

AudioAnalyzer::AudioData AudioAnalyzer::loadAudioFile(const std::string& filePath) {
    AudioData result;
    result.sampleRate = 0;
    result.duration = 0.0;

    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    SwrContext* swrCtx = nullptr;

    try {
        // Open input file
        if (avformat_open_input(&formatCtx, filePath.c_str(), nullptr, nullptr) < 0) {
            throw std::runtime_error("Could not open audio file");
        }

        // Retrieve stream information
        if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
            throw std::runtime_error("Could not find stream information");
        }

        // Find the first audio stream
        int audioStreamIndex = -1;
        for (unsigned int i = 0; i < formatCtx->nb_streams; i++) {
            if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamIndex = i;
                break;
            }
        }

        if (audioStreamIndex == -1) {
            throw std::runtime_error("Could not find audio stream");
        }

        // Get codec parameters
        AVCodecParameters* codecParams = formatCtx->streams[audioStreamIndex]->codecpar;

        // Find decoder
        const AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
        if (!codec) {
            throw std::runtime_error("Could not find decoder");
        }

        // Allocate codec context
        codecCtx = avcodec_alloc_context3(codec);
        if (!codecCtx) {
            throw std::runtime_error("Could not allocate codec context");
        }

        // Copy codec parameters to codec context
        if (avcodec_parameters_to_context(codecCtx, codecParams) < 0) {
            throw std::runtime_error("Could not copy codec parameters");
        }

        // Open codec
        if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
            throw std::runtime_error("Could not open codec");
        }

        // Set up resampler to convert to mono float32
        swrCtx = swr_alloc();
        if (!swrCtx) {
            throw std::runtime_error("Could not allocate resampler");
        }

        // Configure resampler - using new FFmpeg 7+ channel layout API
        AVChannelLayout in_ch_layout, out_ch_layout;
        av_channel_layout_default(&in_ch_layout, codecCtx->ch_layout.nb_channels > 0 ? codecCtx->ch_layout.nb_channels : 2);
        av_channel_layout_default(&out_ch_layout, 1); // Mono

        av_opt_set_chlayout(swrCtx, "in_chlayout", &in_ch_layout, 0);
        av_opt_set_int(swrCtx, "in_sample_rate", codecCtx->sample_rate, 0);
        av_opt_set_sample_fmt(swrCtx, "in_sample_fmt", codecCtx->sample_fmt, 0);

        av_opt_set_chlayout(swrCtx, "out_chlayout", &out_ch_layout, 0);
        av_opt_set_int(swrCtx, "out_sample_rate", codecCtx->sample_rate, 0);
        av_opt_set_sample_fmt(swrCtx, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0);

        av_channel_layout_uninit(&in_ch_layout);
        av_channel_layout_uninit(&out_ch_layout);

        if (swr_init(swrCtx) < 0) {
            throw std::runtime_error("Could not initialize resampler");
        }

        result.sampleRate = codecCtx->sample_rate;

        // Calculate duration
        if (formatCtx->streams[audioStreamIndex]->duration != AV_NOPTS_VALUE) {
            result.duration = formatCtx->streams[audioStreamIndex]->duration *
                            av_q2d(formatCtx->streams[audioStreamIndex]->time_base);
        }

        // Allocate packet and frame
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();

        // Read frames and decode
        while (av_read_frame(formatCtx, packet) >= 0) {
            if (packet->stream_index == audioStreamIndex) {
                // Send packet to decoder
                if (avcodec_send_packet(codecCtx, packet) == 0) {
                    // Receive decoded frames
                    while (avcodec_receive_frame(codecCtx, frame) == 0) {
                        // Allocate buffer for resampled data
                        int outSamples = frame->nb_samples;
                        float* buffer = new float[outSamples];

                        uint8_t* outData = reinterpret_cast<uint8_t*>(buffer);

                        // Resample to mono float
                        int samplesOut = swr_convert(swrCtx, &outData, outSamples,
                                                    (const uint8_t**)frame->data, frame->nb_samples);

                        if (samplesOut > 0) {
                            // Append to result
                            result.samples.insert(result.samples.end(), buffer, buffer + samplesOut);
                        }

                        delete[] buffer;
                    }
                }
            }
            av_packet_unref(packet);
        }

        // Flush decoder
        avcodec_send_packet(codecCtx, nullptr);
        while (avcodec_receive_frame(codecCtx, frame) == 0) {
            int outSamples = frame->nb_samples;
            float* buffer = new float[outSamples];
            uint8_t* outData = reinterpret_cast<uint8_t*>(buffer);

            int samplesOut = swr_convert(swrCtx, &outData, outSamples,
                                        (const uint8_t**)frame->data, frame->nb_samples);

            if (samplesOut > 0) {
                result.samples.insert(result.samples.end(), buffer, buffer + samplesOut);
            }

            delete[] buffer;
        }

        // Update duration if not set
        if (result.duration == 0.0 && result.sampleRate > 0) {
            result.duration = static_cast<double>(result.samples.size()) / result.sampleRate;
        }

        // Cleanup
        av_frame_free(&frame);
        av_packet_free(&packet);

    } catch (...) {
        // Cleanup on error
        if (swrCtx) swr_free(&swrCtx);
        if (codecCtx) avcodec_free_context(&codecCtx);
        if (formatCtx) avformat_close_input(&formatCtx);
        throw;
    }

    // Cleanup
    swr_free(&swrCtx);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&formatCtx);

    return result;
}

std::vector<double> AudioAnalyzer::detectBeats(const AudioData& audio) {
    std::vector<double> beats;

    if (audio.samples.empty() || audio.sampleRate == 0) {
        return beats;
    }

    // Frame size for energy calculation (typically 20-50ms)
    const size_t frameSizeMs = 30;
    const size_t frameSize = (audio.sampleRate * frameSizeMs) / 1000;
    const size_t hopSize = frameSize / 2;  // 50% overlap

    // Calculate energy for each frame
    std::vector<double> energyEnvelope;
    for (size_t i = 0; i + frameSize < audio.samples.size(); i += hopSize) {
        double energy = calculateEnergy(audio.samples, i, frameSize);
        energyEnvelope.push_back(energy);
    }

    if (energyEnvelope.empty()) {
        return beats;
    }

    // Smooth energy envelope with moving average
    const size_t smoothWindowSize = 5;
    std::vector<double> smoothedEnergy = movingAverage(energyEnvelope, smoothWindowSize);

    // Calculate threshold for peak detection
    // Use mean + sensitivity factor * std deviation
    double mean = 0.0;
    for (double e : smoothedEnergy) {
        mean += e;
    }
    mean /= smoothedEnergy.size();

    double variance = 0.0;
    for (double e : smoothedEnergy) {
        variance += (e - mean) * (e - mean);
    }
    double stdDev = std::sqrt(variance / smoothedEnergy.size());

    // Threshold based on sensitivity (0.0 = mean, 1.0 = mean - 2*stdDev)
    double threshold = mean + (1.0 - m_sensitivity * 2.0) * stdDev;

    // Find peaks in energy envelope
    const size_t minBeatGapFrames = 10;  // Minimum frames between beats (~300ms)
    size_t lastBeatFrame = 0;

    for (size_t i = 1; i < smoothedEnergy.size() - 1; ++i) {
        // Check if this is a local maximum
        bool isLocalMax = (smoothedEnergy[i] > smoothedEnergy[i - 1]) &&
                         (smoothedEnergy[i] > smoothedEnergy[i + 1]);

        // Check if above threshold and far enough from last beat
        if (isLocalMax && smoothedEnergy[i] > threshold &&
            (i - lastBeatFrame) >= minBeatGapFrames) {

            // Convert frame index to time
            double timestamp = (i * hopSize) / static_cast<double>(audio.sampleRate);
            beats.push_back(timestamp);
            lastBeatFrame = i;
        }
    }

    return beats;
}

// Extracted energy-based analysis into helper to avoid mutating analysis mode during fallbacks
BeatGrid AudioAnalyzer::analyzeEnergy(const std::string& audioFilePath) {
    BeatGrid beatGrid;

    try {
        // Load and decode audio
        AudioData audio = loadAudioFile(audioFilePath);

        if (audio.samples.empty()) {
            m_lastError = "Failed to load audio file";
            return beatGrid;
        }

        std::cout << "Audio loaded: " << audio.duration << "s, "
                  << audio.sampleRate << " Hz, "
                  << audio.samples.size() << " samples\n";

        // Detect beats
        std::vector<double> beats = detectBeats(audio);

        if (beats.empty()) {
            m_lastError = "No beats detected";
            return beatGrid;
        }

        std::cout << "Detected " << beats.size() << " beats\n";

        // Set beats in grid
        beatGrid.setBeats(beats);

        // Set the actual audio file duration (for padding to full length)
        beatGrid.setAudioDuration(audio.duration);

        // Estimate and set BPM
        double bpm = estimateBPM(beats);
        beatGrid.setBPM(bpm);

        std::cout << "Estimated BPM: " << bpm << "\n";
        std::cout << "Audio duration: " << audio.duration << "s, Last beat: " << beatGrid.getDuration() << "s\n";

    } catch (const std::exception& e) {
        m_lastError = std::string("Exception during analysis: ") + e.what();
    }

    return beatGrid;
}


double AudioAnalyzer::calculateEnergy(const std::vector<float>& samples, size_t start, size_t length) {
    double energy = 0.0;
    size_t end = std::min(start + length, samples.size());

    for (size_t i = start; i < end; ++i) {
        energy += samples[i] * samples[i];
    }

    return energy / length;
}

double AudioAnalyzer::estimateBPM(const std::vector<double>& beats) {
    if (beats.size() < 2) {
        return 0.0;
    }

    // Calculate intervals between consecutive beats
    std::vector<double> intervals;
    for (size_t i = 1; i < beats.size(); ++i) {
        intervals.push_back(beats[i] - beats[i - 1]);
    }

    // Calculate median interval (more robust than mean)
    std::sort(intervals.begin(), intervals.end());
    double medianInterval = intervals[intervals.size() / 2];

    // Convert to BPM
    if (medianInterval > 0.0) {
        return 60.0 / medianInterval;
    }

    return 0.0;
}

std::vector<double> AudioAnalyzer::movingAverage(const std::vector<double>& data, size_t windowSize) {
    std::vector<double> result;

    if (data.size() < windowSize) {
        return data;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        double sum = 0.0;
        size_t count = 0;

        // Calculate window bounds
        size_t start = (i >= windowSize / 2) ? i - windowSize / 2 : 0;
        size_t end = std::min(i + windowSize / 2 + 1, data.size());

        for (size_t j = start; j < end; ++j) {
            sum += data[j];
            count++;
        }

        result.push_back(sum / count);
    }

    return result;
}

} // namespace BeatSync
