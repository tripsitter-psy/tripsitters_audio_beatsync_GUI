#include "AudioAnalyzer.h"
#include <cmath>
#include <algorithm>
#include <iostream>

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
{
}

AudioAnalyzer::~AudioAnalyzer() {
}

BeatGrid AudioAnalyzer::analyze(const std::string& audioFilePath) {
    BeatGrid beatGrid;
    m_lastError.clear();

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

        // Always set the audio duration (even if no beats detected)
        beatGrid.setAudioDuration(audio.duration);

        // Detect beats
        std::vector<double> beats = detectBeats(audio);

        if (beats.empty()) {
            m_lastError = "No beats detected";
            // Return grid with duration set but no beats
            return beatGrid;
        }

        std::cout << "Detected " << beats.size() << " beats\n";

        // Set beats in grid
        beatGrid.setBeats(beats);

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

        // Reserve capacity to avoid reallocations
        if (result.duration > 0.0 && result.sampleRate > 0) {
            size_t estimatedSamples = static_cast<size_t>(result.duration * result.sampleRate);
            result.samples.reserve(estimatedSamples);
        }

        // Allocate packet and frame
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();

        // Buffer for resampled data (reused to avoid allocations)
        std::vector<float> buffer;

        // Read frames and decode
        while (av_read_frame(formatCtx, packet) >= 0) {
            if (packet->stream_index == audioStreamIndex) {
                // Send packet to decoder
                if (avcodec_send_packet(codecCtx, packet) == 0) {
                    // Receive decoded frames
                    while (avcodec_receive_frame(codecCtx, frame) == 0) {
                        // Allocate buffer for resampled data
                        int outSamples = frame->nb_samples;
                        if (buffer.size() < static_cast<size_t>(outSamples)) {
                            buffer.resize(outSamples);
                        }

                        uint8_t* outData = reinterpret_cast<uint8_t*>(buffer.data());

                        // Resample to mono float
                        int samplesOut = swr_convert(swrCtx, &outData, outSamples,
                                                    (const uint8_t**)frame->data, frame->nb_samples);

                        if (samplesOut > 0) {
                            // Append to result
                            result.samples.insert(result.samples.end(), buffer.begin(), buffer.begin() + samplesOut);
                        }
                    }
                }
            }
            av_packet_unref(packet);
        }

        // Flush decoder
        avcodec_send_packet(codecCtx, nullptr);
        while (avcodec_receive_frame(codecCtx, frame) == 0) {
            int outSamples = frame->nb_samples;
            if (buffer.size() < static_cast<size_t>(outSamples)) {
                buffer.resize(outSamples);
            }
            uint8_t* outData = reinterpret_cast<uint8_t*>(buffer.data());

            int samplesOut = swr_convert(swrCtx, &outData, outSamples,
                                        (const uint8_t**)frame->data, frame->nb_samples);

            if (samplesOut > 0) {
                result.samples.insert(result.samples.end(), buffer.begin(), buffer.begin() + samplesOut);
            }
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
