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
    double flashIntensity = 0.3;          // Flash brightness (0.1 to 1.0, default 0.3)
    
    bool enableBeatZoom = false;
    double zoomIntensity = 0.04;          // Zoom amount (0.01 to 0.15, default 0.04 = 4%)
    
    int effectBeatDivisor = 1;            // Effect on every Nth beat (1=every, 2=every other, 4=every 4th)
    
    // Effect region (for applying effects to a subset of the video)
    double effectStartTime = 0.0;         // Start time for effects (0 = from beginning)
    double effectEndTime = -1.0;          // End time for effects (-1 = to end)
    
    double bpm = 120.0;                   // For beat-synced effects (fallback)
    double firstBeatOffset = 0.0;         // Time of first beat (for proper sync)
    std::vector<double> beatTimesInOutput; // Precise beat times in output video timeline
    std::vector<size_t> originalBeatIndices; // Original beat indices (for divisor filtering)
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

    // Expose resolved FFmpeg path for diagnostics
    std::string resolveFfmpegPath() const;

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
                       bool trimToShortest = true,
                       double audioStart = 0.0,
                       double audioEnd = -1.0);

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

    // Build a chained gltransition filter_complex string for N inputs.
    // Example for N=3 returns something that contains: "[0:v][1:v]gltransition=... [t1];[t1][2:v]gltransition=... [t2]"
    std::string buildGlTransitionFilterComplex(size_t numInputs, const std::string& transitionName, double duration) const;

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
