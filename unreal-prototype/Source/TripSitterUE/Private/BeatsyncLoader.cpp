#include "BeatsyncLoader.h"
#include "HAL/ThreadSafeCounter.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Misc/OutputDeviceNull.h"
#include <cassert>
#include "beatsync_capi.h"

// Active operations tracking
static FThreadSafeCounter GActiveOperations;

void FBeatsyncLoader::IncrementActiveOperations() { GActiveOperations.Increment(); }
void FBeatsyncLoader::DecrementActiveOperations() { GActiveOperations.Decrement(); }
int32 FBeatsyncLoader::GetActiveOperations() { return GActiveOperations.GetValue(); }

// Internal callback data structure
struct FBeatsyncLoader::CallbackData
{
    FProgressCb Func;
    void* Key;
};

// Static storage for callback data to prevent leaks
static TMap<void*, TSharedPtr<FBeatsyncLoader::CallbackData>> GCallbackStorage;
static FCriticalSection GCallbackStorageMutex;

// Static trampoline function for progress callbacks - must have stable address
// Using a proper static function instead of a lambda for guaranteed ABI stability
// NOTE: user_data is the writer key (void*), NOT the CallbackData pointer, to avoid
// use-after-free if another thread removes the callback before we can validate it.
static void ProgressCallbackTrampoline(double progress, void* user_data)
{
    // user_data is the writer key, not the CallbackData pointer
    void* writerKey = user_data;
    if (!writerKey) return;

    // Take a shared_ptr copy to extend lifetime during callback invocation.
    // This prevents use-after-free if another thread removes the callback
    // from GCallbackStorage while we're invoking it.
    TSharedPtr<FBeatsyncLoader::CallbackData> localData;
    TFunction<void(double)> localFunc;
    {
        FScopeLock Lock(&GCallbackStorageMutex);
        TSharedPtr<FBeatsyncLoader::CallbackData>* found = GCallbackStorage.Find(writerKey);
        if (found && found->IsValid()) {
            localData = *found;  // Extend lifetime via ref count
            localFunc = localData->Func;
        }
    }

    if (localFunc) {
        localFunc(progress);
    }
    // localData releases here, after callback completes
}

// Function pointer types
using bs_resolve_ffmpeg_path_t = const char* (*)();
using bs_create_audio_analyzer_t = void* (*)();
using bs_destroy_audio_analyzer_t = void (*)(void*);
using bs_analyze_audio_t = int (*)(void*, const char*, bs_beatgrid_t*);
using bs_free_beatgrid_t = void (*)(bs_beatgrid_t*);
using bs_get_waveform_t = int (*)(void*, const char*, float**, size_t*, double*);
using bs_free_waveform_t = void (*)(float*);

using bs_create_video_writer_t = void* (*)();
using bs_destroy_video_writer_t = void (*)(void*);
using bs_video_get_last_error_t = const char* (*)(void*);
using bs_video_set_progress_callback_t = void (*)(void*, void(*)(double, void*), void*);
using bs_video_cut_at_beats_t = int (*)(void*, const char*, const double*, size_t, const char*, double);
using bs_video_cut_at_beats_multi_t = int (*)(void*, const char**, size_t, const double*, size_t, const char*, double);
using bs_video_concatenate_t = int (*)(const char**, size_t, const char*);
using bs_video_add_audio_track_t = int (*)(void*, const char*, const char*, const char*, int, double, double);
using bs_video_set_effects_config_t = void (*)(void*, const bs_effects_config_t*);
using bs_video_apply_effects_t = int (*)(void*, const char*, const char*, const double*, size_t);
using bs_video_extract_frame_t = int (*)(const char*, double, unsigned char**, int*, int*);
using bs_free_frame_data_t = void (*)(unsigned char*);

using bs_initialize_tracing_t = int (*)(const char*);
using bs_shutdown_tracing_t = void (*)();
using bs_start_span_t = void* (*)(const char*);
using bs_end_span_t = void (*)(void*);
using bs_span_set_error_t = void (*)(void*, const char*);
using bs_span_add_event_t = void (*)(void*, const char*);

struct FBeatsyncApi
{
    void* DllHandle = nullptr;
    bs_resolve_ffmpeg_path_t resolve_ffmpeg = nullptr;
    bs_create_audio_analyzer_t create_analyzer = nullptr;
    bs_destroy_audio_analyzer_t destroy_analyzer = nullptr;

    bs_analyze_audio_t analyze_audio = nullptr;
    bs_free_beatgrid_t free_beatgrid = nullptr;
    bs_get_waveform_t get_waveform = nullptr;
    bs_free_waveform_t free_waveform = nullptr;

    bs_create_video_writer_t create_writer = nullptr;
    bs_destroy_video_writer_t destroy_writer = nullptr;
    bs_video_get_last_error_t video_get_last_error = nullptr;
    bs_video_set_progress_callback_t video_set_progress = nullptr;
    bs_video_cut_at_beats_t video_cut_at_beats = nullptr;
    bs_video_cut_at_beats_multi_t video_cut_at_beats_multi = nullptr;
    bs_video_concatenate_t video_concatenate = nullptr;
    bs_video_add_audio_track_t video_add_audio = nullptr;

    bs_video_set_effects_config_t video_set_effects = nullptr;
    bs_video_apply_effects_t video_apply_effects = nullptr;

    bs_video_extract_frame_t video_extract_frame = nullptr;
    bs_free_frame_data_t free_frame_data = nullptr;

    bs_initialize_tracing_t initialize_tracing = nullptr;
    bs_shutdown_tracing_t shutdown_tracing = nullptr;
    bs_start_span_t start_span = nullptr;
    bs_end_span_t end_span = nullptr;
    bs_span_set_error_t span_set_error = nullptr;
    bs_span_add_event_t span_add_event = nullptr;
};

static FBeatsyncApi GApi;

bool FBeatsyncLoader::Initialize()
{
    if (GApi.DllHandle) return true;

    // Expected locations (platform-aware):
    // Windows: <repo>/unreal-prototype/ThirdParty/beatsync/lib/x64/beatsync_backend.dll
    // macOS:   <repo>/unreal-prototype/ThirdParty/beatsync/lib/Mac/libbeatsync_backend.dylib
    // Linux:   <repo>/unreal-prototype/ThirdParty/beatsync/lib/Linux/libbeatsync_backend.so

    FString ModuleDir = FPaths::ConvertRelativePathToFull(FPaths::ProjectDir()); // ProjectDir points to UE project when running in editor; for prototype use ModuleDirectory fallback

    FString Filename;
    FString Subdir;
#if PLATFORM_WINDOWS
    Filename = TEXT("beatsync_backend_shared.dll");
    Subdir = TEXT("x64");
#elif PLATFORM_MAC
    Filename = TEXT("libbeatsync_backend.dylib");
    Subdir = TEXT("Mac");
#else
    Filename = TEXT("libbeatsync_backend.so");
    Subdir = TEXT("Linux");
#endif

    FString DllPath = FPaths::Combine(FPaths::ProjectDir(), TEXT(".."), TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);

    // If ProjectDir is empty in some contexts, try module relative path
    if (!FPaths::FileExists(DllPath)) {
        FString Relative = FPaths::Combine(FPaths::ConvertRelativePathToFull(FPaths::EngineDir()), TEXT(".."));
        DllPath = FPaths::Combine(Relative, TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);
    }

    if (!FPaths::FileExists(DllPath)) {
        UE_LOG(LogTemp, Error, TEXT("Beatsync library not found at: %s"), *DllPath);
        return false;
    }

    GApi.DllHandle = FPlatformProcess::GetDllHandle(*DllPath);
    if (!GApi.DllHandle) {
        UE_LOG(LogTemp, Error, TEXT("Failed to load Beatsync library: %s"), *DllPath);
        return false;
    }

    GApi.resolve_ffmpeg = (bs_resolve_ffmpeg_path_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_resolve_ffmpeg_path"));
    GApi.create_analyzer = (bs_create_audio_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_create_audio_analyzer"));
    GApi.destroy_analyzer = (bs_destroy_audio_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_destroy_audio_analyzer"));

    // Optional video and effects functions
    GApi.analyze_audio = (bs_analyze_audio_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_analyze_audio"));
    GApi.free_beatgrid = (bs_free_beatgrid_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_free_beatgrid"));
    GApi.get_waveform = (bs_get_waveform_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_get_waveform"));
    GApi.free_waveform = (bs_free_waveform_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_free_waveform"));

    GApi.create_writer = (bs_create_video_writer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_create_video_writer"));
    GApi.destroy_writer = (bs_destroy_video_writer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_destroy_video_writer"));
    GApi.video_get_last_error = (bs_video_get_last_error_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_get_last_error"));
    GApi.video_set_progress = (bs_video_set_progress_callback_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_set_progress_callback"));
    GApi.video_cut_at_beats = (bs_video_cut_at_beats_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_cut_at_beats"));
    GApi.video_cut_at_beats_multi = (bs_video_cut_at_beats_multi_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_cut_at_beats_multi"));
    GApi.video_concatenate = (bs_video_concatenate_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_concatenate"));
    GApi.video_add_audio = (bs_video_add_audio_track_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_add_audio_track"));

    GApi.video_set_effects = (bs_video_set_effects_config_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_set_effects_config"));
    GApi.video_apply_effects = (bs_video_apply_effects_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_apply_effects"));

    GApi.video_extract_frame = (bs_video_extract_frame_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_extract_frame"));
    GApi.free_frame_data = (bs_free_frame_data_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_free_frame_data"));

    // Tracing functions (optional)
    GApi.initialize_tracing = (bs_initialize_tracing_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_initialize_tracing"));
    GApi.shutdown_tracing = (bs_shutdown_tracing_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_shutdown_tracing"));
    GApi.start_span = (bs_start_span_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_start_span"));
    GApi.end_span = (bs_end_span_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_end_span"));
    GApi.span_set_error = (bs_span_set_error_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_span_set_error"));
    GApi.span_add_event = (bs_span_add_event_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_span_add_event"));

    if (!GApi.resolve_ffmpeg || !GApi.create_analyzer || !GApi.destroy_analyzer) {
        UE_LOG(LogTemp, Error, TEXT("Required symbols not found in Beatsync DLL"));
        FPlatformProcess::FreeDllHandle(GApi.DllHandle);
        GApi.DllHandle = nullptr;
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("Beatsync DLL loaded: %s"), *DllPath);
    return true;
}

void FBeatsyncLoader::Shutdown()
{
    // Wait for active operations to finish (with timeout)
    const double TimeoutSeconds = 10.0;
    double StartTime = FPlatformTime::Seconds();
    while (GetActiveOperations() > 0 && (FPlatformTime::Seconds() - StartTime) < TimeoutSeconds)
    {
        FPlatformProcess::Sleep(0.05f);
    }
    if (GetActiveOperations() > 0)
    {
        // Continue with cleanup despite active operations. While this risks crashes if
        // operations access the DLL after unload, leaving resources loaded is worse for
        // application shutdown. Callers should ensure operations complete before shutdown.
        UE_LOG(LogTemp, Error, TEXT("BeatsyncLoader::Shutdown: Timed out waiting for %d active operations to finish! Proceeding with cleanup."), GetActiveOperations());
    }

    // Clear all callback data to prevent leaks
    {
        FScopeLock Lock(&GCallbackStorageMutex);
        GCallbackStorage.Empty();
    }

    if (GApi.DllHandle) {
        FPlatformProcess::FreeDllHandle(GApi.DllHandle);
        GApi.DllHandle = nullptr;
        GApi = {};
    }
}

bool FBeatsyncLoader::IsInitialized()
{
    return !!GApi.DllHandle;
}

FString FBeatsyncLoader::ResolveFFmpegPath()
{
    if (!GApi.resolve_ffmpeg) return FString();
    const char* p = GApi.resolve_ffmpeg();
    return p ? FString(UTF8_TO_TCHAR(p)) : FString();
}

void* FBeatsyncLoader::CreateAnalyzer()
{
    if (!GApi.create_analyzer) return nullptr;
    return GApi.create_analyzer();
}

void FBeatsyncLoader::DestroyAnalyzer(void* handle)
{
    if (!GApi.destroy_analyzer) return;
    GApi.destroy_analyzer(handle);
}

void* FBeatsyncLoader::CreateVideoWriter()
{
    if (!GApi.create_writer) return nullptr;
    return GApi.create_writer();
}

void FBeatsyncLoader::DestroyVideoWriter(void* writer)
{
    if (!writer) return;

    // Clear callback data under lock before destroying the writer
    {
        FScopeLock Lock(&GCallbackStorageMutex);
        GCallbackStorage.Remove(writer);
    }

    if (GApi.destroy_writer) {
        GApi.destroy_writer(writer);
    }
}

bool FBeatsyncLoader::AnalyzeAudio(void* handle, const FString& path, FBeatGrid& outGrid)
{
    if (!GApi.analyze_audio) return false;
    if (!handle)
    {
        UE_LOG(LogTemp, Error, TEXT("FBeatsyncLoader::AnalyzeAudio: null handle passed"));
        return false;
    }

    // Prepare C beatgrid
    bs_beatgrid_t grid = {};

    FTCHARToUTF8 PathUtf8(*path);
    int res = GApi.analyze_audio(handle, PathUtf8.Get(), &grid);
    if (res != 0) return false;

    outGrid.BPM = grid.bpm;
    outGrid.Duration = grid.duration;
    outGrid.Beats.Empty();
    if (grid.count > 0 && grid.beats != nullptr) {
        outGrid.Beats.Append(grid.beats, grid.count);
    }

    if (GApi.free_beatgrid) GApi.free_beatgrid(&grid);
    return true;
}

FString FBeatsyncLoader::GetVideoLastError(void* writer)
{
    if (!GApi.video_get_last_error) return FString();
    const char* p = GApi.video_get_last_error(writer);
    return p ? FString(UTF8_TO_TCHAR(p)) : FString();
}

void FBeatsyncLoader::SetProgressCallback(void* writer, FProgressCb cb)
{
    if (!GApi.video_set_progress) return;

    bool doRegister = false;
    {
        FScopeLock Lock(&GCallbackStorageMutex);
        GCallbackStorage.Remove(writer);
        if (cb)
        {
            TSharedPtr<CallbackData> data = MakeShared<CallbackData>();
            data->Func = cb;
            data->Key = writer;
            GCallbackStorage.Add(writer, data);
            doRegister = true;
        }
        // else: doRegister remains false
    }
    // Call the native setter outside the lock to avoid deadlock if callback is invoked immediately
    // Pass writer as user_data (the key) - trampoline will look up CallbackData by this key
    if (doRegister)
    {
        GApi.video_set_progress(writer, ProgressCallbackTrampoline, writer);
    }
    else
    {
        // Unregister native callback if cb is null or on removal
        GApi.video_set_progress(writer, nullptr, nullptr);
    }
}

bool FBeatsyncLoader::CutVideoAtBeats(void* writer, const FString& inputVideo, const TArray<double>& beatTimes, const FString& outputVideo, double clipDuration)
{
    if (!GApi.video_cut_at_beats) return false;

    FTCHARToUTF8 inputConverter(*inputVideo);
    FTCHARToUTF8 outputConverter(*outputVideo);
    int res = GApi.video_cut_at_beats(writer, inputConverter.Get(), beatTimes.GetData(), (size_t)beatTimes.Num(), outputConverter.Get(), clipDuration);
    return res == 0;
}

bool FBeatsyncLoader::CutVideoAtBeatsMulti(void* writer, const TArray<FString>& inputVideos, const TArray<double>& beatTimes, const FString& outputVideo, double clipDuration)
{
    if (!GApi.video_cut_at_beats_multi) return false;

    // Build char** array with persistent converters
    TArray<FTCHARToUTF8> converters;
    converters.Reserve(inputVideos.Num());
    TArray<const char*> arr;
    arr.Reserve(inputVideos.Num());
    for (const auto& s : inputVideos) {
        converters.Emplace(*s);
        arr.Add(converters.Last().Get());
    }

    FTCHARToUTF8 outputConverter(*outputVideo);
    int res = GApi.video_cut_at_beats_multi(writer, arr.GetData(), (size_t)arr.Num(), beatTimes.GetData(), (size_t)beatTimes.Num(), outputConverter.Get(), clipDuration);
    return res == 0;
}

void FBeatsyncLoader::SetEffectsConfig(void* writer, const FEffectsConfig& config)
{
    if (!GApi.video_set_effects)
    {
        UE_LOG(LogTemp, Warning, TEXT("FBeatsyncLoader::SetEffectsConfig: video_set_effects function not available"));
        return;
    }

    // Convert enum values to lowercase strings for C API
    FString TransitionTypeStr = TransitionTypeToString(config.TransitionType);
    FString ColorPresetStr = ColorPresetToString(config.ColorPreset);

    // Create persistent UTF-8 converters for string fields - these RAII objects keep the
    // UTF8 buffers alive until after the API call completes
    FTCHARToUTF8 TransitionTypeUtf8(*TransitionTypeStr);
    FTCHARToUTF8 ColorPresetUtf8(*ColorPresetStr);

    bs_effects_config_t cfg;

    cfg.enableTransitions = config.bEnableTransitions ? 1 : 0;
    cfg.transitionType = TransitionTypeUtf8.Get();
    cfg.transitionDuration = config.TransitionDuration;
    cfg.enableColorGrade = config.bEnableColorGrade ? 1 : 0;
    cfg.colorPreset = ColorPresetUtf8.Get();
    cfg.enableVignette = config.bEnableVignette ? 1 : 0;
    cfg.vignetteStrength = config.VignetteStrength;
    cfg.enableBeatFlash = config.bEnableBeatFlash ? 1 : 0;
    cfg.flashIntensity = config.FlashIntensity;
    cfg.enableBeatZoom = config.bEnableBeatZoom ? 1 : 0;
    cfg.zoomIntensity = config.ZoomIntensity;
    cfg.effectBeatDivisor = config.EffectBeatDivisor;
    cfg.effectStartTime = config.EffectStartTime;
    cfg.effectEndTime = config.EffectEndTime;

    GApi.video_set_effects(writer, &cfg);

    // Log the configuration for debugging
    UE_LOG(LogTemp, Verbose, TEXT("FBeatsyncLoader::SetEffectsConfig: writer=%p, transitions=%d, vignette=%d, flash=%d, zoom=%d"),
        writer, cfg.enableTransitions, cfg.enableVignette, cfg.enableBeatFlash, cfg.enableBeatZoom);
}

bool FBeatsyncLoader::ApplyEffects(void* writer, const FString& inputVideo, const FString& outputVideo, const TArray<double>& beatTimes)
{
    if (!GApi.video_apply_effects) return false;

    FTCHARToUTF8 inConv(*inputVideo);
    FTCHARToUTF8 outConv(*outputVideo);
    int res = GApi.video_apply_effects(writer, inConv.Get(), outConv.Get(), beatTimes.GetData(), (size_t)beatTimes.Num());
    return res == 0;
}

bool FBeatsyncLoader::AddAudioTrack(void* writer, const FString& inputVideo, const FString& audioFile, const FString& outputVideo, bool trimToShortest, double audioStart, double audioEnd)
{
    if (!GApi.video_add_audio) return false;

    // Create persistent converters for string parameters - these RAII objects keep the
    // UTF8 buffers alive until after the API call completes
    FTCHARToUTF8 InputVideoUtf8(*inputVideo);
    FTCHARToUTF8 AudioFileUtf8(*audioFile);
    FTCHARToUTF8 OutputVideoUtf8(*outputVideo);

    int res = GApi.video_add_audio(writer, InputVideoUtf8.Get(), AudioFileUtf8.Get(), OutputVideoUtf8.Get(), trimToShortest ? 1 : 0, audioStart, audioEnd);
    return res == 0;
}

bool FBeatsyncLoader::ExtractFrame(const FString& videoPath, double timestamp, TArray<uint8>& outRgb24, int32& outWidth, int32& outHeight)
{
    if (!GApi.video_extract_frame) return false;

    FTCHARToUTF8 VideoPathUtf8(*videoPath);
    unsigned char* data = nullptr;
    int w = 0, h = 0;
    int res = GApi.video_extract_frame(VideoPathUtf8.Get(), timestamp, &data, &w, &h);
    if (res != 0 || !data) return false;

    // Guard against overflow and unreasonable dimensions
    if (w <= 0 || h <= 0 || w > 16384 || h > 16384) {
        if (GApi.free_frame_data) GApi.free_frame_data(data);
        return false;
    }
    int64 size64 = static_cast<int64>(w) * h * 3;
    if (size64 > INT32_MAX) {
        if (GApi.free_frame_data) GApi.free_frame_data(data);
        return false;
    }
    int size = static_cast<int>(size64);
    outRgb24.SetNumUninitialized(size);
    memcpy(outRgb24.GetData(), data, size);
    outWidth = w;
    outHeight = h;

    if (GApi.free_frame_data) GApi.free_frame_data(data);
    return true;
}

FBeatsyncLoader::SpanHandle FBeatsyncLoader::StartSpan(const FString& name)
{
    if (!GApi.start_span) return nullptr;
    FTCHARToUTF8 TempName(*name);
    return GApi.start_span(TempName.Get());
}

void FBeatsyncLoader::EndSpan(SpanHandle h)
{
    if (!GApi.end_span) return;
    GApi.end_span(h);
}

void FBeatsyncLoader::SpanSetError(SpanHandle h, const FString& msg)
{
    if (!GApi.span_set_error) return;
    FTCHARToUTF8 TempMsg(*msg);
    GApi.span_set_error(h, TempMsg.Get());
}

void FBeatsyncLoader::SpanAddEvent(SpanHandle h, const FString& ev)
{
    if (!GApi.span_add_event) return;
    FTCHARToUTF8 TempEv(*ev);
    GApi.span_add_event(h, TempEv.Get());
}
