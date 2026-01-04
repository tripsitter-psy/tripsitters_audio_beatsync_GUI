#include <string>
#include <vector>
#pragma once

#include <string>
#include <vector>
#include <memory>

// Forward declarations
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;

namespace BeatSync {

/**
 * @brief Video metadata information
 */
struct VideoInfo {
    int width;              // Video width in pixels
    int height;             // Video height in pixels
    double fps;             // Frames per second
    double duration;        // Duration in seconds
    int64_t totalFrames;    // Total number of frames
    std::string codec;      // Video codec name
    int64_t bitrate;        // Bitrate in bits/second
};

/**
 * @brief Handles video file loading, decoding, and metadata extraction
 */
class VideoProcessor {
public:
    VideoProcessor();
    ~VideoProcessor();

    /**
     * @brief Open a video file
     * @param filePath Path to the video file
     * @return true if successful, false otherwise
     */
    bool open(const std::string& filePath);

    /**
     * @brief Close the currently open video
     */
    void close();

    /**
     * @brief Check if a video is currently open
     */
    bool isOpen() const;

    /**
     * @brief Get video metadata
     */
    VideoInfo getInfo() const;

    /**
     * @brief Get the path of the currently open video
     */
    std::string getFilePath() const;

    /**
     * @brief Get last error message
     */
    std::string getLastError() const;

    /**
     * @brief Seek to a specific timestamp (in seconds)
     * @param timestamp Time in seconds
     * @return true if successful
     */
    bool seekToTimestamp(double timestamp);

    /**
     * @brief Read the next frame
     * @param outFrame Output frame (caller should not free this)
     * @return true if frame read successfully, false if EOF or error
     */
    bool readFrame(AVFrame** outFrame);

    /**
     * @brief Get timestamp of current frame in seconds
     */
    double getCurrentTimestamp() const;

private:
    AVFormatContext* m_formatCtx;
    AVCodecContext* m_videoCodecCtx;
    AVCodecContext* m_audioCodecCtx;
    AVFrame* m_frame;
    AVPacket* m_packet;

    int m_videoStreamIndex;
    int m_audioStreamIndex;

    std::string m_filePath;
    std::string m_lastError;
    VideoInfo m_info;

    bool m_isOpen;

    // Helper methods
    bool initializeCodecs();
    void cleanup();
};

} // namespace BeatSync
