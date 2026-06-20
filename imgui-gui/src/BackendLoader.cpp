#include "BackendLoader.h"

#include <vector>

#if defined(_WIN32)
  #include <windows.h>
  static void* dl_open(const char* p)  { return (void*)LoadLibraryA(p); }
  static void* dl_sym(void* h, const char* s) { return (void*)GetProcAddress((HMODULE)h, s); }
  static void  dl_close(void* h)       { FreeLibrary((HMODULE)h); }
  static const char* kLibNames[] = { "beatsync_backend_shared.dll" };
#else
  #include <dlfcn.h>
  static void* dl_open(const char* p)  { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
  static void* dl_sym(void* h, const char* s) { return dlsym(h, s); }
  static void  dl_close(void* h)       { dlclose(h); }
  #if defined(__APPLE__)
    static const char* kLibNames[] = {
        "libbeatsync_backend_shared.dylib",
        "libbeatsync_backend.dylib",
    };
  #else
    static const char* kLibNames[] = {
        "libbeatsync_backend_shared.so",
        "libbeatsync_backend.so",
    };
  #endif
#endif

bool BackendLoader::Load(const std::string& explicitPath)
{
    if (Handle) return true;

    std::vector<std::string> candidates;
    if (!explicitPath.empty())
        candidates.push_back(explicitPath);

    // Conventional locations relative to a typical build/run layout.
    const char* relDirs[] = {
        "",                                  // system loader path / cwd
        "./",
        "../",
        "../../build/Release/",
        "../../build/",
        "../build/Release/",
        "../build/",
        "./Resources/",
    };
    for (const char* dir : relDirs)
        for (const char* name : kLibNames)
            candidates.push_back(std::string(dir) + name);

    for (const auto& path : candidates)
    {
        void* h = dl_open(path.c_str());
        if (h)
        {
            Handle = h;
            ResolvedPath = path;
            if (BindSymbols())
                return true;

            // Loaded but missing required symbols - treat as failure.
            ErrorMessage = "Loaded '" + path + "' but required bs_* symbols were missing";
            Unload();
            return false;
        }
    }

    ErrorMessage = "Could not locate the beatsync backend library "
                   "(build it with -DBEATSYNC_BUILD_SHARED=ON, or pass an explicit path)";
    return false;
}

void BackendLoader::Unload()
{
    if (Handle)
    {
        dl_close(Handle);
        Handle = nullptr;
    }
    ResolvedPath.clear();
}

// Bind a single symbol; record and bail on the first missing required one.
#define BIND(fn)                                                              \
    do {                                                                      \
        fn = reinterpret_cast<decltype(fn)>(dl_sym(Handle, #fn));             \
        if (!fn) { ErrorMessage = "missing symbol: " #fn; return false; }     \
    } while (0)

// Optional symbols may be absent depending on backend build flags.
#define BIND_OPT(fn) fn = reinterpret_cast<decltype(fn)>(dl_sym(Handle, #fn))

bool BackendLoader::BindSymbols()
{
    BIND(bs_get_version);
    BIND(bs_init);
    BIND(bs_shutdown);

    BIND(bs_create_audio_analyzer);
    BIND(bs_destroy_audio_analyzer);
    BIND(bs_set_bpm_hint);
    BIND(bs_analyze_audio);
    BIND(bs_free_beatgrid);
    BIND(bs_get_waveform);
    BIND(bs_free_waveform);
    BIND_OPT(bs_get_analyzer_last_error);
    BIND_OPT(bs_get_waveform_bands);
    BIND_OPT(bs_free_waveform_bands);

    BIND(bs_create_video_writer);
    BIND(bs_destroy_video_writer);
    BIND(bs_video_get_last_error);
    BIND(bs_video_cut_at_beats);
    BIND(bs_video_add_audio_track);
    BIND_OPT(bs_resolve_ffmpeg_path);
    BIND_OPT(bs_video_set_progress_callback);
    BIND_OPT(bs_video_set_cancel_flag);
    BIND_OPT(bs_video_is_cancelled);
    BIND_OPT(bs_video_cut_at_beats_multi);
    BIND_OPT(bs_video_concatenate);
    BIND_OPT(bs_video_set_effects_config);
    BIND_OPT(bs_video_apply_effects);
    BIND_OPT(bs_video_extract_frame);
    BIND_OPT(bs_free_frame_data);

    // AI / ONNX - optional (backend may be built without USE_ONNX).
    BIND_OPT(bs_create_ai_analyzer);
    BIND_OPT(bs_destroy_ai_analyzer);
    BIND_OPT(bs_ai_analyze_file);
    BIND_OPT(bs_ai_analyze_quick);
    BIND_OPT(bs_free_ai_result);
    BIND_OPT(bs_ai_is_available);
    BIND_OPT(bs_ai_get_providers);
    BIND_OPT(bs_ai_get_active_provider);
    BIND_OPT(bs_ai_get_last_error);

    // AudioFlux - optional (backend may be built without USE_AUDIOFLUX).
    BIND_OPT(bs_audioflux_is_available);
    BIND_OPT(bs_audioflux_analyze);
    BIND_OPT(bs_audioflux_analyze_with_stems);

    return true;
}

#undef BIND
#undef BIND_OPT
