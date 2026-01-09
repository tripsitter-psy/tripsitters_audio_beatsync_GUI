// Stub implementation of beatsync backend for initial UE compilation
// Replace with full implementation by running build_backend.sh

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
  #define BEATSYNC_API __attribute__((visibility("default")))
#else
  #define BEATSYNC_API
#endif

typedef void (*bs_progress_cb)(double progress, void* user_data);

typedef struct {
    double* beats;
    size_t count;
    double bpm;
    double duration;
} bs_beatgrid_t;

// Stub implementations
BEATSYNC_API void* bs_create_audio_analyzer() {
    return (void*)1; // Non-null stub
}

BEATSYNC_API void bs_destroy_audio_analyzer(void* analyzer) {
    (void)analyzer;
}

BEATSYNC_API int bs_analyze_audio(void* analyzer, const char* filepath, bs_beatgrid_t* outGrid) {
    (void)analyzer;
    (void)filepath;
    if (outGrid) {
        outGrid->beats = NULL;
        outGrid->count = 0;
        outGrid->bpm = 120.0;
        outGrid->duration = 0.0;
    }
    return -1; // Return error - stub not implemented
}

BEATSYNC_API void bs_free_beatgrid(bs_beatgrid_t* grid) {
    if (grid && grid->beats) {
        free(grid->beats);
        grid->beats = NULL;
    }
}

BEATSYNC_API void* bs_create_video_writer() {
    return (void*)1; // Non-null stub
}

BEATSYNC_API void bs_destroy_video_writer(void* writer) {
    (void)writer;
}

BEATSYNC_API const char* bs_video_get_last_error(void* writer) {
    (void)writer;
    return "Backend not implemented - run build_backend.sh";
}

BEATSYNC_API const char* bs_resolve_ffmpeg_path() {
    return "/usr/local/bin/ffmpeg";
}

BEATSYNC_API void bs_video_set_progress_callback(void* writer, bs_progress_cb cb, void* user_data) {
    (void)writer;
    (void)cb;
    (void)user_data;
}

BEATSYNC_API int bs_video_cut_at_beats(void* writer, const char* inputVideo, const double* beatTimes, size_t count, const char* outputVideo, double clipDuration) {
    (void)writer;
    (void)inputVideo;
    (void)beatTimes;
    (void)count;
    (void)outputVideo;
    (void)clipDuration;
    return -1; // Return error - stub not implemented
}

BEATSYNC_API int bs_video_concatenate(const char** inputs, size_t count, const char* outputVideo) {
    (void)inputs;
    (void)count;
    (void)outputVideo;
    return -1; // Return error - stub not implemented
}
