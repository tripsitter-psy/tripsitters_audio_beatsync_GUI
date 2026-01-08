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

// Simple C representation of a beat grid
typedef struct {
    double* beats;   // owned buffer (malloc)
    size_t count;
    double bpm;
    double duration;
} bs_beatgrid_t;

// AudioAnalyzer
BEATSYNC_API void* bs_create_audio_analyzer();
BEATSYNC_API void bs_destroy_audio_analyzer(void* analyzer);
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
BEATSYNC_API int bs_video_concatenate(const char** inputs, size_t count, const char* outputVideo);

#ifdef __cplusplus
}
#endif
