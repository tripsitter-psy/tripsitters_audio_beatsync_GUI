// NOTE: This is the canonical C API header. All consumers should use this file as the source of truth.
// The build system (CMake) copies this file to ThirdParty/beatsync/include/beatsync_capi.h for Unreal integration and install/include for consumers.
#pragma once

#if defined(_WIN32)
  #ifdef BEATSYNC_CAPI_EXPORT
    #define BEATSYNC_API __declspec(dllexport)
  #elif defined(BEATSYNC_CAPI_IMPORT)
    #define BEATSYNC_API __declspec(dllimport)
  #else
    #define BEATSYNC_API
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  /* Ensure symbols for the C API are visible when building shared libs on Unix/macOS */
  #define BEATSYNC_API __attribute__((visibility("default")))
#else
  #define BEATSYNC_API
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*bs_progress_cb)(double progress, void* user_data);

// Library version constant (set at build time via CMake -DBEATSYNC_VERSION=...)
BEATSYNC_API extern const char* const BS_VERSION;

// Library version and lifecycle
BEATSYNC_API const char* bs_get_version();
BEATSYNC_API int bs_init();
BEATSYNC_API void bs_shutdown();

// Simple C representation of a beat grid
typedef struct bs_beatgrid_t {
    double* beats;   // owned buffer (malloc)
    size_t count;
    double bpm;
    double duration;
} bs_beatgrid_t;

// AudioAnalyzer
BEATSYNC_API void* bs_create_audio_analyzer();
BEATSYNC_API void bs_destroy_audio_analyzer(void* analyzer);
// Set BPM hint for beat detection (0 = auto-detect, >0 = use this BPM for evenly-spaced beats)
BEATSYNC_API void bs_set_bpm_hint(void* analyzer, double bpm);
// Returns 0 on success, non-zero on error. On success, outGrid is filled and must be freed with bs_free_beatgrid
BEATSYNC_API int bs_analyze_audio(void* analyzer, const char* filepath, bs_beatgrid_t* outGrid);
BEATSYNC_API void bs_free_beatgrid(bs_beatgrid_t* grid);

// VideoWriter
BEATSYNC_API void* bs_create_video_writer();
BEATSYNC_API void bs_destroy_video_writer(void* writer);
BEATSYNC_API const char* bs_video_get_last_error(void* writer); // returned string is owned by library (valid until next call)
BEATSYNC_API const char* bs_resolve_ffmpeg_path(); // returned string is owned by library
BEATSYNC_API void bs_video_set_progress_callback(void* writer, bs_progress_cb cb, void* user_data);
BEATSYNC_API int bs_video_cut_at_beats(void* writer, const char* inputVideo, const double* beatTimes, size_t count, const char* outputVideo, double clipDuration);
// Multi-video version: cycles through inputVideos for each beat
BEATSYNC_API int bs_video_cut_at_beats_multi(void* writer, const char** inputVideos, size_t videoCount,
                                              const double* beatTimes, size_t beatCount,
                                              const char* outputVideo, double clipDuration);
BEATSYNC_API int bs_video_concatenate(const char** inputs, size_t count, const char* outputVideo);
// Pre-normalize videos to common resolution/framerate before processing
// This reduces GPU memory pressure when processing many segments from different sources
// normalizedPaths: output array of paths to normalized videos (caller allocates array, library fills paths)
// pathBufferSize: size of each path buffer in normalizedPaths (recommend 512+ chars)
// Returns 0 on success, -1 on error
BEATSYNC_API int bs_video_normalize_sources(void* writer, const char** inputVideos, size_t videoCount,
                                             char** normalizedPaths, size_t pathBufferSize);
// Free normalized video files created by bs_video_normalize_sources
BEATSYNC_API void bs_video_cleanup_normalized(char** normalizedPaths, size_t count);
// Add audio track to video (combines video from first input with audio from second)
// audioStart/audioEnd: trim audio to selection (-1 for audioEnd means no trim)
// trimToShortest: if true, output ends when shorter stream ends
BEATSYNC_API int bs_video_add_audio_track(void* writer, const char* inputVideo, const char* audioFile,
                                           const char* outputVideo, int trimToShortest,
                                           double audioStart, double audioEnd);
// Waveform visualization - returns downsampled peak values for display
BEATSYNC_API int bs_get_waveform(void* analyzer, const char* filepath,
                                  float** outPeaks, size_t* outCount, double* outDuration);
BEATSYNC_API void bs_free_waveform(float* peaks);

// Frequency-colored waveform (Rekordbox/Traktor style)
// Returns 3 separate peak arrays for bass, mids, and highs frequency bands
// Bass: 20-200 Hz (red), Mids: 200-2000 Hz (cyan/blue), Highs: 2000+ Hz (white/yellow)
// Memory ownership and nullability:
// - All peak arrays (bass_peaks, mid_peaks, high_peaks) are allocated by the library using malloc() and must be freed only by calling bs_free_waveform_bands().
// - Callers must not free these buffers directly.
// - On failure, all pointers will be NULL and count will be zero.
// - Callers must always check for NULL before accessing arrays.
// - bs_free_waveform_bands(bs_waveform_bands_t*) will free all memory for the three peak arrays.
// - Do not attempt to free or reuse these pointers after calling bs_free_waveform_bands().
//
// See also: bs_waveform_bands_t, bs_free_waveform_bands
typedef struct {
    float* bass_peaks;    // Low frequency peaks (20-200 Hz) - malloc, freed by bs_free_waveform_bands
    float* mid_peaks;     // Mid frequency peaks (200-2000 Hz) - malloc, freed by bs_free_waveform_bands
    float* high_peaks;    // High frequency peaks (2000+ Hz) - malloc, freed by bs_free_waveform_bands
    size_t count;         // Number of peaks in each array
    double duration;      // Audio duration in seconds
} bs_waveform_bands_t;

// Get frequency-band waveform for professional DJ-style display
// Returns 0 on success, non-zero on error
// Result must be freed with bs_free_waveform_bands()
BEATSYNC_API int bs_get_waveform_bands(void* analyzer, const char* filepath,
                                        bs_waveform_bands_t* out_bands);
BEATSYNC_API void bs_free_waveform_bands(bs_waveform_bands_t* bands);

// Effects configuration for video processing
// NOTE: String fields (transitionType, colorPreset) are copied by the implementation
// into internal std::string storage. Callers only need to keep the pointers valid
// during the bs_video_set_effects_config() call itself; after the call returns,
// the caller may free or reuse the string memory.
// To reset/clear effects, call bs_video_set_effects_config() with a nullptr config.
typedef struct {
    int enableTransitions;
    const char* transitionType;  // "fade", "wipe", "dissolve", "zoom" - copied by implementation
    double transitionDuration;

    int enableColorGrade;
    const char* colorPreset;     // "warm", "cool", "vintage", "vibrant" - copied by implementation

    int enableVignette;
    double vignetteStrength;

    int enableBeatFlash;
    double flashIntensity;

    int enableBeatZoom;
    double zoomIntensity;

    int effectBeatDivisor;       // 1=every beat, 2=every 2nd, 4=every 4th, etc.

    double effectStartTime;      // Start time for effects in seconds (0 = from beginning)
    double effectEndTime;        // End time for effects in seconds (-1 = to end of video)
} bs_effects_config_t;

// Set effects configuration on video writer
// Returns 0 on success, non-zero on error
BEATSYNC_API int bs_video_set_effects_config(void* writer, const bs_effects_config_t* config);
// Apply effects to video using beat times for beat-synced effects
// Returns 0 on success, non-zero on error
BEATSYNC_API int bs_video_apply_effects(void* writer, const char* inputVideo,
                                         const char* outputVideo,
                                         const double* beatTimes, size_t beatCount);

// Extract single frame at timestamp as RGB24 data
// outData will be allocated (malloc) with width*height*3 bytes
// Returns 0 on success, caller must call bs_free_frame_data when done
BEATSYNC_API int bs_video_extract_frame(const char* videoPath, double timestamp,
                                         unsigned char** outData, int* outWidth, int* outHeight);
BEATSYNC_API void bs_free_frame_data(unsigned char* data);

// Tracing control
// Initialize tracing (returns 0 on success, non-zero on error)
BEATSYNC_API int bs_initialize_tracing(const char* service_name);
// Shutdown tracing and flush spans
BEATSYNC_API void bs_shutdown_tracing();

// Lightweight C API for creating spans from the consumer (returns opaque handle)
typedef void* bs_span_t;
BEATSYNC_API bs_span_t bs_start_span(const char* name);
BEATSYNC_API void bs_end_span(bs_span_t span);
BEATSYNC_API void bs_span_set_error(bs_span_t span, const char* msg);
BEATSYNC_API void bs_span_add_event(bs_span_t span, const char* event);

// =============================================================================
// ONNX AI Analysis (native neural network inference)
// =============================================================================

// AI progress callback with stage information
typedef int (*bs_ai_progress_cb)(float progress, const char* stage, const char* message, void* user_data);

// AI analyzer configuration
typedef struct {
    const char* beat_model_path;    // Path to beat detection ONNX model (required)
    const char* stem_model_path;    // Path to stem separation ONNX model (optional, NULL to disable)
    int use_stem_separation;        // Enable stem separation before beat detection
    int use_drums_for_beats;        // Use drums stem for beat detection (recommended)
    int use_gpu;                    // Enable GPU acceleration if available
    int gpu_device_id;              // GPU device ID (default 0)
    float beat_threshold;           // Beat activation threshold (0.0-1.0, default 0.5)
    float downbeat_threshold;       // Downbeat activation threshold (0.0-1.0, default 0.5)
} bs_ai_config_t;


// Segment structure for AI results
// Memory ownership: The label field is heap-allocated (malloc) and is freed by bs_free_ai_result().
typedef struct bs_segment_t {
  double start_time;
  double end_time;
  char* label;  // "intro", "verse", "chorus", etc. (heap-allocated; freed by bs_free_ai_result)
  float confidence;
} bs_segment_t;

// Extended beat grid with downbeats and segments
// Memory ownership and nullability:
// - All pointers (beats, downbeats, segments, and each segment->label) are allocated by the library and must be freed only by calling bs_free_ai_result().
// - Callers must not free these buffers directly.
// - On failure, all pointers will be NULL and counts zero; on partial results, some arrays may be NULL while others are populated.
// - Callers must always check for NULL before accessing arrays.
// - bs_free_ai_result(bs_ai_result_t*) will free all memory for beats, downbeats, segments, and segment labels.
// - Do not attempt to free or reuse these pointers after calling bs_free_ai_result().
//
// See also: bs_ai_result_t, bs_segment_t, bs_free_ai_result
typedef struct {
  double* beats;          // Beat timestamps (malloc, freed by bs_free_ai_result)
  size_t beat_count;
  double* downbeats;      // Downbeat timestamps (malloc, freed by bs_free_ai_result)
  size_t downbeat_count;
  double bpm;
  double duration;
  // Segment information (for All-In-One model)
  bs_segment_t* segments; // Array of segments (malloc, freed by bs_free_ai_result)
  size_t segment_count;
} bs_ai_result_t;

// Create AI analyzer with configuration
// Returns opaque handle or NULL on error
BEATSYNC_API void* bs_create_ai_analyzer(const bs_ai_config_t* config);

// Destroy AI analyzer
BEATSYNC_API void bs_destroy_ai_analyzer(void* analyzer);

// Run AI analysis on audio file
// Returns 0 on success, non-zero on error
// outResult must be freed with bs_free_ai_result()
BEATSYNC_API int bs_ai_analyze_file(void* analyzer, const char* audio_path,
                                     bs_ai_result_t* out_result,
                                     bs_ai_progress_cb progress_cb, void* user_data);

// Run AI analysis on audio samples (stereo interleaved float32)
BEATSYNC_API int bs_ai_analyze_samples(void* analyzer,
                                        const float* samples, size_t sample_count,
                                        int sample_rate, int num_channels,
                                        bs_ai_result_t* out_result,
                                        bs_ai_progress_cb progress_cb, void* user_data);

// Quick analysis without stem separation (faster but less accurate)
BEATSYNC_API int bs_ai_analyze_quick(void* analyzer, const char* audio_path,
                                      bs_ai_result_t* out_result,
                                      bs_ai_progress_cb progress_cb, void* user_data);

// Free AI result data
BEATSYNC_API void bs_free_ai_result(bs_ai_result_t* result);

// Get AI analyzer error message
BEATSYNC_API const char* bs_ai_get_last_error(void* analyzer);

// Get AI analyzer model info
BEATSYNC_API const char* bs_ai_get_model_info(void* analyzer);

// Check if ONNX Runtime is available
BEATSYNC_API int bs_ai_is_available();

// Get available ONNX execution providers (returns comma-separated string)
BEATSYNC_API const char* bs_ai_get_providers();

// Check if GPU is enabled for a specific analyzer instance (returns 1 if GPU active, 0 if CPU)
BEATSYNC_API int bs_ai_is_gpu_enabled(void* analyzer);

// Get the active execution provider name for an analyzer (e.g., "CUDAExecutionProvider" or "CPUExecutionProvider")
BEATSYNC_API const char* bs_ai_get_active_provider(void* analyzer);

// =============================================================================
// AudioFlux Spectral Analysis (signal processing-based, no GPU/Python needed)
// =============================================================================

// Check if AudioFlux is available
BEATSYNC_API int bs_audioflux_is_available();

// Analyze audio using AudioFlux spectral flux onset detection
// Returns 0 on success, non-zero on error
// outResult must be freed with bs_free_ai_result()
BEATSYNC_API int bs_audioflux_analyze(const char* audio_path,
                                       bs_ai_result_t* out_result,
                                       bs_ai_progress_cb progress_cb, void* user_data);

// Analyze audio using stem separation + AudioFlux (best quality)
// First separates drums using ONNX model, then runs AudioFlux on drums stem
// stem_model_path: path to demucs/htdemucs ONNX model (pass NULL to skip stem separation)
// Returns 0 on success, non-zero on error
// outResult must be freed with bs_free_ai_result()
BEATSYNC_API int bs_audioflux_analyze_with_stems(const char* audio_path,
                                                  const char* stem_model_path,
                                                  bs_ai_result_t* out_result,
                                                  bs_ai_progress_cb progress_cb, void* user_data);

#ifdef __cplusplus
}
#endif
