// BackendLoader - portable (UE-free) loader for the beatsync C API.
//
// Dynamically loads libbeatsync_backend_shared (.dylib/.so/.dll) at runtime and
// binds the bs_* entry points as function pointers. Mirrors the role of the
// Unreal BeatsyncLoader but with zero engine dependencies, so the ImGui GUI can
// drive the exact same backend used by the Slate app.
#pragma once

#include <string>

extern "C" {
#include "beatsync_capi.h"
}

class BackendLoader
{
public:
    // Attempts to load the backend library. If `explicitPath` is empty, searches
    // a set of conventional locations relative to the executable plus the system
    // loader path. Returns true on success.
    bool Load(const std::string& explicitPath = "");
    void Unload();
    bool IsLoaded() const { return Handle != nullptr; }

    // Path that was actually loaded (for diagnostics / status bar).
    const std::string& LoadedPath() const { return ResolvedPath; }
    const std::string& LastError() const { return ErrorMessage; }

    // ---- Bound entry points (named exactly like the C API) -----------------
    // Lifecycle
    decltype(&bs_get_version)            bs_get_version            = nullptr;
    decltype(&bs_init)                   bs_init                   = nullptr;
    decltype(&bs_shutdown)               bs_shutdown               = nullptr;

    // Audio analyzer (basic / energy)
    decltype(&bs_create_audio_analyzer)  bs_create_audio_analyzer  = nullptr;
    decltype(&bs_destroy_audio_analyzer) bs_destroy_audio_analyzer = nullptr;
    decltype(&bs_get_analyzer_last_error) bs_get_analyzer_last_error = nullptr;
    decltype(&bs_set_bpm_hint)           bs_set_bpm_hint           = nullptr;
    decltype(&bs_analyze_audio)          bs_analyze_audio          = nullptr;
    decltype(&bs_free_beatgrid)          bs_free_beatgrid          = nullptr;
    decltype(&bs_get_waveform)           bs_get_waveform           = nullptr;
    decltype(&bs_free_waveform)          bs_free_waveform          = nullptr;
    decltype(&bs_get_waveform_bands)     bs_get_waveform_bands     = nullptr;
    decltype(&bs_free_waveform_bands)    bs_free_waveform_bands    = nullptr;

    // AI (ONNX) analysis
    decltype(&bs_create_ai_analyzer)     bs_create_ai_analyzer     = nullptr;
    decltype(&bs_destroy_ai_analyzer)    bs_destroy_ai_analyzer    = nullptr;
    decltype(&bs_ai_analyze_file)        bs_ai_analyze_file        = nullptr;
    decltype(&bs_ai_analyze_quick)       bs_ai_analyze_quick       = nullptr;
    decltype(&bs_free_ai_result)         bs_free_ai_result         = nullptr;
    decltype(&bs_ai_is_available)        bs_ai_is_available        = nullptr;
    decltype(&bs_ai_get_providers)       bs_ai_get_providers       = nullptr;
    decltype(&bs_ai_get_active_provider) bs_ai_get_active_provider = nullptr;
    decltype(&bs_ai_get_last_error)      bs_ai_get_last_error      = nullptr;

    // AudioFlux spectral analysis
    decltype(&bs_audioflux_is_available)      bs_audioflux_is_available      = nullptr;
    decltype(&bs_audioflux_analyze)           bs_audioflux_analyze           = nullptr;
    decltype(&bs_audioflux_analyze_with_stems) bs_audioflux_analyze_with_stems = nullptr;

    // Video writer
    decltype(&bs_create_video_writer)     bs_create_video_writer     = nullptr;
    decltype(&bs_destroy_video_writer)    bs_destroy_video_writer    = nullptr;
    decltype(&bs_video_get_last_error)    bs_video_get_last_error    = nullptr;
    decltype(&bs_resolve_ffmpeg_path)     bs_resolve_ffmpeg_path     = nullptr;
    decltype(&bs_video_set_progress_callback) bs_video_set_progress_callback = nullptr;
    decltype(&bs_video_set_cancel_flag)   bs_video_set_cancel_flag   = nullptr;
    decltype(&bs_video_is_cancelled)      bs_video_is_cancelled      = nullptr;
    decltype(&bs_video_cut_at_beats)      bs_video_cut_at_beats      = nullptr;
    decltype(&bs_video_cut_at_beats_multi) bs_video_cut_at_beats_multi = nullptr;
    decltype(&bs_video_concatenate)       bs_video_concatenate       = nullptr;
    decltype(&bs_video_add_audio_track)   bs_video_add_audio_track   = nullptr;
    decltype(&bs_video_set_effects_config) bs_video_set_effects_config = nullptr;
    decltype(&bs_video_apply_effects)     bs_video_apply_effects     = nullptr;
    decltype(&bs_video_extract_frame)     bs_video_extract_frame     = nullptr;
    decltype(&bs_free_frame_data)         bs_free_frame_data         = nullptr;

private:
    void* Handle = nullptr;
    std::string ResolvedPath;
    std::string ErrorMessage;

    bool BindSymbols();
};
