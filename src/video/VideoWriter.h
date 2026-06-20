#pragma once

#include "VideoProcessor.h"
#include "../audio/BeatGrid.h"
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <memory>

// Forward declarations
struct AVFormatContext;
struct AVCodecContext;
struct SwsContext;
class VideoWriterTestAccess;  // Test access helper

namespace BeatSync {

class OnnxFrameInterpolator;  // Neural frame interpolation (RIFE), optional

/**
 * @brief Information about a detected video encoder
 */
struct GPUEncoderInfo {
    std::string encoderName = "";   // e.g., "h264_nvenc", "libx264"
    std::string preset = "";        // GPU-specific or libx264 preset
    bool isHardware = false;          // true if using GPU acceleration

    // Equality comparison operators
    bool operator==(const GPUEncoderInfo& other) const noexcept {
        return encoderName == other.encoderName &&
               preset == other.preset &&
               isHardware == other.isHardware;
    }

    bool operator!=(const GPUEncoderInfo& other) const noexcept {
        return !(*this == other);
    }
};


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
    
    // Global effect region (fallback when per-effect ranges are at defaults)
    double effectStartTime = 0.0;         // Start time for effects (0 = from beginning)
    double effectEndTime = -1.0;          // End time for effects (-1 = to end)

    // Per-effect time ranges.
    // When start==0.0 AND end<= 0.0, the global effectStartTime/effectEndTime is used instead.
    double colorGradeStartTime = 0.0;
    double colorGradeEndTime   = -1.0;

    double vignetteStartTime   = 0.0;
    double vignetteEndTime     = -1.0;

    double beatFlashStartTime  = 0.0;
    double beatFlashEndTime    = -1.0;

    double beatZoomStartTime   = 0.0;
    double beatZoomEndTime     = -1.0;
    
    double bpm = 120.0;                   // For beat-synced effects (fallback)
    double firstBeatOffset = 0.0;         // Time of first beat (for proper sync)
    std::vector<double> beatTimesInOutput; // Precise beat times in output video timeline
    std::vector<size_t> originalBeatIndices; // Original beat indices (for divisor filtering)
};

/**
 * @brief Configuration for per-clip speed ramps (slow-mo / speed-up)
 *
 * Speed ramps preserve beat-sync: each affected clip keeps its fixed OUTPUT
 * beat-slot duration. The multiplier only changes how much SOURCE footage is
 * sampled into that slot (source consumed = slotDuration * speed), via setpts.
 * A multiplier < 1.0 is slow-motion, > 1.0 is speed-up. Audio (the master
 * timeline) is never retimed by this feature.
 */
struct SpeedRampConfig {
    bool enabled = false;

    // Selection: which beat clips get a speed ramp.
    float affectedFraction = 0.25f;  // 0..1 portion of eligible clips affected
    uint32_t seed = 0;               // reproducible randomization
    int selectionMode = 0;           // 0=random, 1=every Nth clip, 2=every Nth via beat divisor
    int everyN = 4;                  // used by selectionMode 1/2

    // Direction + amount. Multipliers are clamped to [0.5, 2.0] so a single
    // atempo can keep per-clip audio length consistent for concatenation.
    float speedUpFraction = 0.5f;    // of affected clips, portion that speed up (>1x) vs slow down
    float slowMin = 0.5f;            // slow-mo range (<1.0); set min==max for a fixed amount
    float slowMax = 0.5f;
    float fastMin = 2.0f;            // speed-up range (>1.0)
    float fastMax = 2.0f;

    // Smoothness of the retimed video.
    int smoothing = 0;               // 0=duplicate frames (fast), 1=minterpolate optical flow (smoother)

    // Source-footage guard: clamp the multiplier toward 1.0 (or skip the ramp)
    // when a speed-up would need more source than is available from the clip's
    // start. When false, the ramp is skipped rather than clamped.
    bool guardClampToAvailable = true;

    static constexpr float kMinSpeed = 0.5f;
    static constexpr float kMaxSpeed = 2.0f;
};

/**
 * @brief Video segment definition
 */
struct VideoSegment {
    double startTime;  // Start time in seconds
    double endTime;    // End time in seconds
    std::string label; // Optional label for this segment
    double speed = 1.0; // Per-clip speed multiplier (1.0 = unchanged). See SpeedRampConfig.
};

/**
 * @brief Handles video cutting, concatenation, and export
 *
 * Note: Cache members are protected by an internal mutex; VideoWriter is not fully
 * thread-safe for all operations — callers should synchronize when sharing instances
 * across threads.
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
     * @brief Set cancel flag pointer for cooperative cancellation
     * @param flag Pointer to flag (owned by caller). If *flag becomes non-zero, processing aborts.
     */
    void setCancelFlag(const int* flag);

    /**
     * @brief Check if cancellation was requested
     * @return true if cancel flag is set and non-zero
     */
    bool isCancelled() const;

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
     * @brief Set per-clip speed ramp configuration
     * @param config Speed ramp configuration
     */
    void setSpeedConfig(const SpeedRampConfig& config);

    /**
     * @brief Get the active speed ramp configuration
     */
    const SpeedRampConfig& getSpeedConfig() const { return m_speed; }

    /**
     * @brief Compute a deterministic per-clip speed multiplier for each clip.
     *
     * Returns a vector of size clipCount. Clips not selected for a ramp get 1.0.
     * Selection and amounts are reproducible for a given config.seed. The
     * source-footage guard is NOT applied here (it needs per-clip source
     * availability) — apply it at extraction time.
     *
     * @param clipCount Number of beat clips
     * @param config Speed ramp configuration
     * @return Per-clip multipliers (1.0 = unchanged)
     */
    static std::vector<double> computeClipSpeeds(size_t clipCount, const SpeedRampConfig& config);

    /**
     * @brief Extract a single clip applying a per-clip speed multiplier (re-encodes).
     *
     * The output clip is always outputDuration seconds long (the beat slot);
     * source consumed = outputDuration * speed. speed < 1.0 is slow-motion,
     * > 1.0 is speed-up. speed == 1.0 behaves like a precise (re-encoded) copy.
     * Honors the active SpeedRampConfig.smoothing mode.
     *
     * @param inputVideo Source video path
     * @param sourceStart Start position in source (seconds)
     * @param outputDuration Output (beat-slot) duration (seconds)
     * @param speed Per-clip speed multiplier (clamped to [0.5, 2.0])
     * @param outputVideo Output clip path
     * @return true if successful
     */
    bool extractSpeedClip(const std::string& inputVideo,
                          double sourceStart,
                          double outputDuration,
                          double speed,
                          const std::string& outputVideo);

    /**
     * @brief Set the path to the RIFE ONNX model used for neural slow-mo
     * interpolation (SpeedRampConfig.smoothing == 2). If unset or the model
     * fails to load, interpolation falls back to minterpolate.
     */
    void setInterpolationModelPath(const std::string& path);

    /**
     * @brief Number of clips whose speed ramp was clamped/skipped by the
     * source-footage guard during the most recent extraction batch.
     */
    size_t getLastSpeedClampCount() const { return m_speedClampCount.load(); }

    /**
     * @brief Apply effects to concatenated video
     * @param inputVideo Path to concatenated video
     * @param outputVideo Path to output video with effects
     * @return true if successful
     */
    bool applyEffects(const std::string& inputVideo, const std::string& outputVideo);

    /**
     * @brief Pre-normalize a video to standard format for efficient segment extraction
     *
     * Converts video to target resolution, framerate, and codec once, so subsequent
     * segment extractions can use fast stream copy instead of per-segment re-encoding.
     * This reduces GPU memory pressure when processing many segments.
     *
     * @param inputVideo Path to source video
     * @param outputVideo Path to normalized output (typically in temp directory)
     * @return true if successful
     */
    bool normalizeVideo(const std::string& inputVideo, const std::string& outputVideo);

    /**
     * @brief Pre-normalize multiple videos and return paths to normalized versions
     *
     * @param inputVideos Vector of source video paths
     * @param normalizedPaths Output vector of normalized video paths (in temp directory)
     * @return true if all videos were normalized successfully
     */
    bool normalizeVideos(const std::vector<std::string>& inputVideos,
                         std::vector<std::string>& normalizedPaths);

private:
    // Allow test access to private methods
    friend class ::VideoWriterTestAccess;

    std::string m_lastError;
    std::function<void(double)> m_progressCallback;
    const int* m_cancelFlag = nullptr;  // External cancel flag (owned by caller)

    // Output settings (defaults)
    int m_outputWidth = 1920;
    int m_outputHeight = 1080;
    int m_outputFps = 24;

    // Effects configuration
    EffectsConfig m_effects;

    // Per-clip speed ramp configuration
    SpeedRampConfig m_speed;

    // Count of clips clamped/skipped by the source-footage guard in the current batch.
    mutable std::atomic<size_t> m_speedClampCount{0};

    // Neural frame interpolation (RIFE) for smooth slow-mo. Lazily loaded.
    std::string m_interpModelPath;
    std::unique_ptr<OnnxFrameInterpolator> m_interpolator;
    bool m_interpLoadAttempted = false;
    std::mutex m_interpMutex;

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
     *
     * @param speed Per-clip speed multiplier (1.0 = none). When != 1.0, applies
     *        setpts video retiming + atempo audio so the output is `duration`
     *        seconds long while consuming `duration * speed` of source.
     * @param smoothing 0 = duplicate frames, 1 = minterpolate optical flow.
     */
    bool copySegmentPrecise(const std::string& inputVideo,
                           double startTime,
                           double duration,
                           const std::string& outputVideo,
                           double speed = 1.0,
                           int smoothing = 0);

    /**
     * @brief Slow-mo speed clip rendered with neural frame interpolation (RIFE).
     *
     * Decodes the source window to RGB frames, synthesizes the output-rate frame
     * sequence by interpolating between source frames at arbitrary timesteps, and
     * encodes the result directly at the output fps (no setpts needed). Used for
     * SpeedRampConfig.smoothing == 2. Returns false (so callers can fall back) if
     * the model isn't available or any stage fails.
     */
    bool extractSpeedClipInterpolated(const std::string& inputVideo,
                                      double sourceStart,
                                      double outputDuration,
                                      double speed,
                                      const std::string& outputVideo);

    /** @brief Lazily load the interpolation model. Returns true if usable. */
    bool ensureInterpolator();

    /**
     * @brief Get FFmpeg executable path
     * Checks environment variable BEATSYNC_FFMPEG_PATH, then PATH, then falls back to default
     */
    std::string getFFmpegPath() const;

    void reportProgress(double progress);

    // GPU encoder detection and selection
    /**
     * @brief Probe if a specific encoder is available in FFmpeg
     * @param encoder Encoder name (e.g., "h264_nvenc")
     * @return true if encoder is available
     */
    bool probeEncoder(const std::string& encoder) const;

    /**
     * @brief Detect the best available encoder (GPU first, then software fallback)
     * @param speedPreset Speed preference ("ultrafast", "fast", "medium")
     * @return GPUEncoderInfo with encoder name, preset, and hardware flag
     */
    GPUEncoderInfo detectBestEncoder(const std::string& speedPreset = "fast") const;

    /**
     * @brief Get FFmpeg encoder arguments string
     * @param speedPreset Speed preference ("ultrafast", "fast", "medium")
     * @return FFmpeg arguments string like "-c:v h264_nvenc -preset p1 ..."
     */
    std::string getEncoderArgs(const std::string& speedPreset) const;

    /**
     * @brief Check if CUDA hardware acceleration is available for decoding
     * @return true if CUDA hwaccel is supported by FFmpeg
     */
    bool hasCudaHwaccel() const;

    /**
     * @brief Check if scale_cuda filter is available for GPU scaling
     * @return true if scale_cuda filter is supported by FFmpeg
     */
    bool hasScaleCudaFilter() const;

    /**
     * @brief Log GPU capabilities for diagnostics
     */
    void logGpuCapabilities() const;

    // Cached encoder info to avoid repeated probing
    // Access to these mutable caches is protected by m_cacheMutex
    mutable GPUEncoderInfo m_cachedEncoder = GPUEncoderInfo{};
    mutable bool m_encoderCacheValid = false;
    mutable int m_cudaHwaccelCache = -1;  // -1 = not checked, 0 = no, 1 = yes
    mutable int m_scaleCudaCache = -1;    // -1 = not checked, 0 = no, 1 = yes
    mutable std::recursive_mutex m_cacheMutex;


    // GPU memory management: Track segment extraction count.
    // Every GPU_RESET_INTERVAL segments, we force CPU mode and flush GPU memory to prevent crashes.
    // This periodic safety mechanism takes effect regardless of m_allowGpuThisSegment.
    // See copySegmentFast/copySegmentPrecise for logic.
    mutable size_t m_segmentsSinceGpuReset = 0;
    static constexpr size_t GPU_RESET_INTERVAL = 20;  // Force CPU+flush every 20 segments

    // Explicit per-segment GPU gate: set by copySegmentFast, consulted by copySegmentPrecise.
    // m_allowGpuThisSegment can disable GPU for a single segment even if the counter allows GPU.
    // The periodic reset (m_segmentsSinceGpuReset/GPU_RESET_INTERVAL) always takes precedence and forces CPU+flush when hit.
    // See copySegmentFast/copySegmentPrecise for details.
    mutable bool m_allowGpuThisSegment = true;

    /**
     * @brief Check if we should use GPU for current segment
     * Returns false every GPU_RESET_INTERVAL segments to allow GPU memory cleanup
     */
    bool shouldUseGpuForSegment();

    /**
     * @brief Reset segment counter (call at start of new batch operation)
     */
    void resetSegmentCounter();
};

} // namespace BeatSync
