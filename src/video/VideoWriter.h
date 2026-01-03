#pragma once

#include "VideoProcessor.h"
#include "../audio/BeatGrid.h"
#include <string>
#include <vector>
#include <functional>

// Forward declarations
struct AVFormatContext;
struct AVCodecContext;
struct SwsContext;

namespace BeatSync {

/**
 * @brief Configuration for video effects
 */
struct EffectsConfig {
    // Transitions
    bool enableTransitions = false;
    std::string transitionType = "fade";  // fade, wipeleft, wiperight, dissolve, circlecrop
    double transitionDuration = 0.3;      // seconds

    // Visual filters
    bool enableColorGrade = false;
    std::string colorPreset = "none";     // warm, cool, vintage, vibrant

    bool enableVignette = false;
    double vignetteStrength = 0.5;        // 0.0 to 1.0

    bool enableBlur = false;
    double blurStrength = 2.0;            // sigma value

    // Beat effects
    bool enableBeatFlash = false;
    bool enableBeatZoom = false;
    double bpm = 120.0;                   // For beat-synced effects
};

/**
 * @brief Video segment definition
 */
struct VideoSegment {
    double startTime;  // Start time in seconds
    double endTime;    // End time in seconds
    std::string label; // Optional label for this segment
};

/**
 * @brief Handles video cutting, concatenation, and export
 */
class VideoWriter {
public:
    VideoWriter();
    ~VideoWriter();

    /**
     * @brief Cut video at beat timestamps
     * @param inputVideo Path to input video file
     * @param beatGrid Beat grid with timestamps
     * @param outputVideo Path to output video file
     * @param clipDuration Duration of each clip in seconds (0 = until next beat)
     * @return true if successful
     */
    bool cutAtBeats(const std::string& inputVideo,
                    const BeatGrid& beatGrid,
                    const std::string& outputVideo,
                    double clipDuration = 0.0);

    /**
     * @brief Extract video segments
     * @param inputVideo Path to input video
     * @param segments List of segments to extract
     * @param outputVideo Path to output video
     * @return true if successful
     */
    bool extractSegments(const std::string& inputVideo,
                        const std::vector<VideoSegment>& segments,
                        const std::string& outputVideo);

    /**
     * @brief Split video at specific timestamps and save each segment
     * @param inputVideo Path to input video
     * @param timestamps List of split points (in seconds)
     * @param outputPattern Output file pattern (e.g., "clip_%03d.mp4")
     * @return true if successful
     */
    bool splitVideo(const std::string& inputVideo,
                   const std::vector<double>& timestamps,
                   const std::string& outputPattern);

    /**
     * @brief Get last error message
     */
    std::string getLastError() const;

    /**
     * @brief Set progress callback
     * @param callback Function called with progress (0.0 to 1.0)
     */
    void setProgressCallback(std::function<void(double)> callback);

    /**
     * @brief Copy video segment using stream copy (fast, no re-encoding)
     */
    bool copySegmentFast(const std::string& inputVideo,
                        double startTime,
                        double duration,
                        const std::string& outputVideo);

    /**
     * @brief Concatenate multiple video files
     */
    bool concatenateVideos(const std::vector<std::string>& inputVideos,
                          const std::string& outputVideo);

    /**
     * @brief Add audio track to video file
     * @param inputVideo Path to input video (may have no audio)
     * @param audioFile Path to audio file to add
     * @param outputVideo Path to output video with audio
     * @param trimToShortest If true, output duration matches shorter of video/audio
     * @return true if successful
     */
    bool addAudioTrack(const std::string& inputVideo,
                       const std::string& audioFile,
                       const std::string& outputVideo,
                       bool trimToShortest = true);

    /**
     * @brief Set output video settings
     * @param width Output width in pixels
     * @param height Output height in pixels
     * @param fps Output frame rate
     */
    void setOutputSettings(int width, int height, int fps);

    /**
     * @brief Set video effects configuration
     * @param config Effects configuration
     */
    void setEffectsConfig(const EffectsConfig& config);

    /**
     * @brief Apply effects to concatenated video
     * @param inputVideo Path to concatenated video
     * @param outputVideo Path to output video with effects
     * @return true if successful
     */
    bool applyEffects(const std::string& inputVideo, const std::string& outputVideo);

private:
    std::string m_lastError;
    std::function<void(double)> m_progressCallback;

    // Output settings (defaults)
    int m_outputWidth = 1920;
    int m_outputHeight = 1080;
    int m_outputFps = 24;

    // Effects configuration
    EffectsConfig m_effects;

    /**
     * @brief Build FFmpeg filter chain from effects config
     * @return Filter chain string for -vf parameter
     */
    std::string buildEffectsFilterChain() const;

    /**
     * @brief Get color grade filter for preset
     * @param preset Color preset name
     * @return FFmpeg filter string
     */
    std::string getColorGradeFilter(const std::string& preset) const;

    /**
     * @brief Copy video segment with re-encoding (slower, more precise)
     */
    bool copySegmentPrecise(const std::string& inputVideo,
                           double startTime,
                           double duration,
                           const std::string& outputVideo);

    /**
     * @brief Get FFmpeg executable path
     * Checks environment variable BEATSYNC_FFMPEG_PATH, then PATH, then falls back to default
     */
    std::string getFFmpegPath() const;

    void reportProgress(double progress);
};

} // namespace BeatSync
