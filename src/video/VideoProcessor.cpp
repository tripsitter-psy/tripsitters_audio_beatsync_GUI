#include "VideoProcessor.h"
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
}

namespace BeatSync {

VideoProcessor::VideoProcessor()
    : m_formatCtx(nullptr)
    , m_videoCodecCtx(nullptr)
    , m_audioCodecCtx(nullptr)
    , m_frame(nullptr)
    , m_packet(nullptr)
    , m_videoStreamIndex(-1)
    , m_audioStreamIndex(-1)
    , m_isOpen(false)
{
    m_info = {0, 0, 0.0, 0.0, 0, "", 0};
}

VideoProcessor::~VideoProcessor() {
    close();
}

bool VideoProcessor::open(const std::string& filePath) {
    close(); // Close any previously open video

    m_filePath = filePath;
    m_lastError.clear();

    // Open input file
    if (avformat_open_input(&m_formatCtx, filePath.c_str(), nullptr, nullptr) < 0) {
        m_lastError = "Could not open video file: " + filePath;
        return false;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(m_formatCtx, nullptr) < 0) {
        m_lastError = "Could not find stream information";
        avformat_close_input(&m_formatCtx);
        return false;
    }

    // Find video and audio streams
    for (unsigned int i = 0; i < m_formatCtx->nb_streams; i++) {
        if (m_formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && m_videoStreamIndex < 0) {
            m_videoStreamIndex = i;
        }
        if (m_formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && m_audioStreamIndex < 0) {
            m_audioStreamIndex = i;
        }
    }

    if (m_videoStreamIndex == -1) {
        m_lastError = "Could not find video stream";
        avformat_close_input(&m_formatCtx);
        return false;
    }

    // Initialize video codec
    if (!initializeCodecs()) {
        avformat_close_input(&m_formatCtx);
        return false;
    }

    // Extract video info
    AVStream* videoStream = m_formatCtx->streams[m_videoStreamIndex];
    m_info.width = m_videoCodecCtx->width;
    m_info.height = m_videoCodecCtx->height;
    m_info.fps = av_q2d(videoStream->r_frame_rate);
    m_info.codec = avcodec_get_name(m_videoCodecCtx->codec_id);
    m_info.bitrate = m_videoCodecCtx->bit_rate;

    // Calculate duration
    if (m_formatCtx->duration != AV_NOPTS_VALUE) {
        m_info.duration = m_formatCtx->duration / (double)AV_TIME_BASE;
    } else if (videoStream->duration != AV_NOPTS_VALUE) {
        m_info.duration = videoStream->duration * av_q2d(videoStream->time_base);
    }

    m_info.totalFrames = videoStream->nb_frames;
    if (m_info.totalFrames == 0 && m_info.duration > 0 && m_info.fps > 0) {
        m_info.totalFrames = static_cast<int64_t>(m_info.duration * m_info.fps);
    }

    // Allocate frame and packet
    m_frame = av_frame_alloc();
    m_packet = av_packet_alloc();

    if (!m_frame || !m_packet) {
        m_lastError = "Could not allocate frame/packet";
        cleanup();
        return false;
    }

    m_isOpen = true;
    return true;
}

void VideoProcessor::close() {
    cleanup();
    m_isOpen = false;
    m_filePath.clear();
    m_info = {0, 0, 0.0, 0.0, 0, "", 0};
}

bool VideoProcessor::isOpen() const {
    return m_isOpen;
}

VideoInfo VideoProcessor::getInfo() const {
    return m_info;
}

bool VideoProcessor::hasAudio() const {
    return m_audioStreamIndex >= 0;
}

std::string VideoProcessor::getFilePath() const {
    return m_filePath;
}

std::string VideoProcessor::getLastError() const {
    return m_lastError;
}

bool VideoProcessor::seekToTimestamp(double timestamp) {
    if (!m_isOpen) {
        m_lastError = "No video file is open";
        return false;
    }

    int64_t seekTarget = static_cast<int64_t>(timestamp * AV_TIME_BASE);

    if (av_seek_frame(m_formatCtx, -1, seekTarget, AVSEEK_FLAG_BACKWARD) < 0) {
        m_lastError = "Seek failed";
        return false;
    }

    // Flush codec buffers
    avcodec_flush_buffers(m_videoCodecCtx);
    if (m_audioCodecCtx) {
        avcodec_flush_buffers(m_audioCodecCtx);
    }

    return true;
}

bool VideoProcessor::readFrame(AVFrame** outFrame) {
    if (!m_isOpen) {
        m_lastError = "No video file is open";
        return false;
    }

    while (av_read_frame(m_formatCtx, m_packet) >= 0) {
        if (m_packet->stream_index == m_videoStreamIndex) {
            // Send packet to decoder
            if (avcodec_send_packet(m_videoCodecCtx, m_packet) == 0) {
                // Receive decoded frame
                if (avcodec_receive_frame(m_videoCodecCtx, m_frame) == 0) {
                    *outFrame = m_frame;
                    av_packet_unref(m_packet);
                    return true;
                }
            }
        }
        av_packet_unref(m_packet);
    }

    // Try to flush decoder
    avcodec_send_packet(m_videoCodecCtx, nullptr);
    if (avcodec_receive_frame(m_videoCodecCtx, m_frame) == 0) {
        *outFrame = m_frame;
        return true;
    }

    return false; // EOF or error
}

double VideoProcessor::getCurrentTimestamp() const {
    if (!m_isOpen || !m_frame) {
        return 0.0;
    }

    AVStream* stream = m_formatCtx->streams[m_videoStreamIndex];
    return m_frame->pts * av_q2d(stream->time_base);
}

bool VideoProcessor::initializeCodecs() {
    // Get video codec parameters
    AVCodecParameters* codecParams = m_formatCtx->streams[m_videoStreamIndex]->codecpar;

    // Find decoder
    const AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
    if (!codec) {
        m_lastError = "Could not find video decoder";
        return false;
    }

    // Allocate codec context
    m_videoCodecCtx = avcodec_alloc_context3(codec);
    if (!m_videoCodecCtx) {
        m_lastError = "Could not allocate video codec context";
        return false;
    }

    // Copy codec parameters
    if (avcodec_parameters_to_context(m_videoCodecCtx, codecParams) < 0) {
        m_lastError = "Could not copy video codec parameters";
        return false;
    }

    // Open codec
    if (avcodec_open2(m_videoCodecCtx, codec, nullptr) < 0) {
        m_lastError = "Could not open video codec";
        return false;
    }

    // Initialize audio codec if audio stream exists
    if (m_audioStreamIndex >= 0) {
        AVCodecParameters* audioParams = m_formatCtx->streams[m_audioStreamIndex]->codecpar;
        const AVCodec* audioCodec = avcodec_find_decoder(audioParams->codec_id);

        if (audioCodec) {
            m_audioCodecCtx = avcodec_alloc_context3(audioCodec);
            if (m_audioCodecCtx) {
                if (avcodec_parameters_to_context(m_audioCodecCtx, audioParams) >= 0) {
                    avcodec_open2(m_audioCodecCtx, audioCodec, nullptr);
                }
            }
        }
    }

    return true;
}

void VideoProcessor::cleanup() {
    if (m_frame) {
        av_frame_free(&m_frame);
        m_frame = nullptr;
    }

    if (m_packet) {
        av_packet_free(&m_packet);
        m_packet = nullptr;
    }

    if (m_videoCodecCtx) {
        avcodec_free_context(&m_videoCodecCtx);
        m_videoCodecCtx = nullptr;
    }

    if (m_audioCodecCtx) {
        avcodec_free_context(&m_audioCodecCtx);
        m_audioCodecCtx = nullptr;
    }

    if (m_formatCtx) {
        avformat_close_input(&m_formatCtx);
        m_formatCtx = nullptr;
    }

    m_videoStreamIndex = -1;
    m_audioStreamIndex = -1;
}

} // namespace BeatSync
