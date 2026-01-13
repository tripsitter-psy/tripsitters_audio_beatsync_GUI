#include "BeatsyncLoader.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

// C API structure matching beatsync_capi.h
struct bs_beatgrid_t {
    double* beats;
    size_t count;
    double bpm;
    double duration;
};

// Function pointer types matching C API
using bs_resolve_ffmpeg_path_t = const char* (*)();
using bs_create_audio_analyzer_t = void* (*)();
using bs_destroy_audio_analyzer_t = void (*)(void*);
using bs_analyze_audio_t = int (*)(void*, const char*, bs_beatgrid_t*);
using bs_free_beatgrid_t = void (*)(bs_beatgrid_t*);
using bs_create_video_writer_t = void* (*)();
using bs_destroy_video_writer_t = void (*)(void*);
using bs_video_get_last_error_t = const char* (*)(void*);
using bs_progress_cb = void (*)(double, void*);
using bs_video_set_progress_callback_t = void (*)(void*, bs_progress_cb, void*);
using bs_video_cut_at_beats_t = int (*)(void*, const char*, const double*, size_t, const char*, double);
using bs_video_cut_at_beats_multi_t = int (*)(void*, const char**, size_t, const double*, size_t, const char*, double);
using bs_video_concatenate_t = int (*)(const char**, size_t, const char*);
using bs_get_waveform_t = int (*)(void*, const char*, float**, size_t*, double*);
using bs_free_waveform_t = void (*)(float*);
using bs_video_add_audio_track_t = int (*)(void*, const char*, const char*, const char*, int, double, double);

// Effects config structure matching C API
struct bs_effects_config_t {
    int enableTransitions;
    const char* transitionType;
    double transitionDuration;
    int enableColorGrade;
    const char* colorPreset;
    int enableVignette;
    double vignetteStrength;
    int enableBeatFlash;
    double flashIntensity;
    int enableBeatZoom;
    double zoomIntensity;
    int effectBeatDivisor;
};

using bs_video_set_effects_config_t = void (*)(void*, const bs_effects_config_t*);
using bs_video_apply_effects_t = int (*)(void*, const char*, const char*, const double*, size_t);
using bs_video_extract_frame_t = int (*)(const char*, double, unsigned char**, int*, int*);
using bs_free_frame_data_t = void (*)(unsigned char*);

// AI analyzer C API types
struct bs_ai_config_t {
    const char* beat_model_path;
    const char* stem_model_path;
    int use_stem_separation;
    int use_drums_for_beats;
    int use_gpu;
    int gpu_device_id;
    float beat_threshold;
    float downbeat_threshold;
};

struct bs_ai_result_t {
    double* beats;
    size_t beat_count;
    double* downbeats;
    size_t downbeat_count;
    double bpm;
    double duration;
    void* segments; // We don't use segments in UE yet
    size_t segment_count;
};

using bs_ai_progress_cb = int (*)(float, const char*, const char*, void*);
using bs_ai_is_available_t = int (*)();
using bs_ai_get_providers_t = const char* (*)();
using bs_create_ai_analyzer_t = void* (*)(const bs_ai_config_t*);
using bs_destroy_ai_analyzer_t = void (*)(void*);
using bs_ai_analyze_file_t = int (*)(void*, const char*, bs_ai_result_t*, bs_ai_progress_cb, void*);
using bs_ai_analyze_quick_t = int (*)(void*, const char*, bs_ai_result_t*, bs_ai_progress_cb, void*);
using bs_free_ai_result_t = void (*)(bs_ai_result_t*);
using bs_ai_get_last_error_t = const char* (*)(void*);

struct FBeatsyncApi
{
    void* DllHandle = nullptr;
    bs_resolve_ffmpeg_path_t resolve_ffmpeg = nullptr;
    bs_create_audio_analyzer_t create_analyzer = nullptr;
    bs_destroy_audio_analyzer_t destroy_analyzer = nullptr;
    bs_analyze_audio_t analyze_audio = nullptr;
    bs_free_beatgrid_t free_beatgrid = nullptr;
    bs_create_video_writer_t create_video_writer = nullptr;
    bs_destroy_video_writer_t destroy_video_writer = nullptr;
    bs_video_get_last_error_t video_get_last_error = nullptr;
    bs_video_set_progress_callback_t video_set_progress_callback = nullptr;
    bs_video_cut_at_beats_t video_cut_at_beats = nullptr;
    bs_video_cut_at_beats_multi_t video_cut_at_beats_multi = nullptr;
    bs_video_concatenate_t video_concatenate = nullptr;
    bs_video_add_audio_track_t video_add_audio_track = nullptr;
    bs_get_waveform_t get_waveform = nullptr;
    bs_free_waveform_t free_waveform = nullptr;
    bs_video_set_effects_config_t video_set_effects_config = nullptr;
    bs_video_apply_effects_t video_apply_effects = nullptr;
    bs_video_extract_frame_t video_extract_frame = nullptr;
    bs_free_frame_data_t free_frame_data = nullptr;
    // AI analyzer functions
    bs_ai_is_available_t ai_is_available = nullptr;
    bs_ai_get_providers_t ai_get_providers = nullptr;
    bs_create_ai_analyzer_t create_ai_analyzer = nullptr;
    bs_destroy_ai_analyzer_t destroy_ai_analyzer = nullptr;
    bs_ai_analyze_file_t ai_analyze_file = nullptr;
    bs_ai_analyze_quick_t ai_analyze_quick = nullptr;
    bs_free_ai_result_t free_ai_result = nullptr;
    bs_ai_get_last_error_t ai_get_last_error = nullptr;
};

static FBeatsyncApi GApi;

// Per-writer progress callbacks to avoid race conditions and leaks
static TMap<void*, TUniquePtr<TFunction<void(double)>>> GProgressCallbacks;
static FCriticalSection GProgressCallbacksLock;
static std::atomic<int> GActiveProgressCallbacks{0};
static FEvent* GProgressCallbacksEvent = FPlatformProcess::GetSynchEventFromPool(true);

static void StaticProgressCallback(double Progress, void* UserData)
{
    TFunction<void(double)> LocalCallback;
    {
        FScopeLock Lock(&GProgressCallbacksLock);
        if (UserData) {
            TUniquePtr<TFunction<void(double)>>* CallbackPtr = GProgressCallbacks.Find(UserData);
            if (CallbackPtr && *CallbackPtr && **CallbackPtr) {
                LocalCallback = **CallbackPtr;  // Copy callback to invoke outside lock
            }
        }
    }  // Lock released here before invoking callback to prevent deadlock

    if (LocalCallback) {
        GActiveProgressCallbacks.fetch_add(1, std::memory_order_seq_cst);
        LocalCallback(Progress);
        int prev = GActiveProgressCallbacks.fetch_sub(1, std::memory_order_seq_cst);
        if (prev == 1 && GProgressCallbacksEvent) {
            GProgressCallbacksEvent->Trigger();
        }
    }
}

bool FBeatsyncLoader::Initialize()
{
    if (GApi.DllHandle) return true;

    FString Filename;
    FString Subdir;
#if PLATFORM_WINDOWS
    Filename = TEXT("beatsync_backend_shared.dll");
    Subdir = TEXT("x64");
#elif PLATFORM_MAC
    Filename = TEXT("libbeatsync_backend_shared.dylib");
    Subdir = TEXT("Mac");
#else
    Filename = TEXT("libbeatsync_backend_shared.so");
    Subdir = TEXT("Linux");
#endif

    // For Program target, DLL is either next to executable or in Engine source ThirdParty folder
    FString ExeDir = FPaths::GetPath(FPlatformProcess::ExecutablePath());
    FString DllPath = FPaths::Combine(ExeDir, Filename);

    // Try next to executable first
    if (!FPaths::FileExists(DllPath)) {
        UE_LOG(LogTemp, Log, TEXT("Beatsync DLL not next to exe, trying ThirdParty..."));
        // Try Engine/Source/Programs/TripSitter/ThirdParty (dev builds)
        FString LibDir = FPaths::Combine(ExeDir, TEXT(".."), TEXT(".."), TEXT("Source"), TEXT("Programs"), TEXT("TripSitter"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir);
        LibDir = FPaths::ConvertRelativePathToFull(LibDir);
        DllPath = FPaths::Combine(LibDir, Filename);
    }

    // Fallback: try old plugin paths for compatibility
    if (!FPaths::FileExists(DllPath)) {
        FString LibDir = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("TripSitterUE"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir);
        DllPath = FPaths::Combine(LibDir, Filename);
    }

    if (!FPaths::FileExists(DllPath)) {
        UE_LOG(LogTemp, Error, TEXT("Beatsync library not found. Tried: next to exe, Engine/Source/Programs/TripSitter/ThirdParty"));
        return false;
    }

    // Set LibDir for DLL search path
    FString LibDir = FPaths::GetPath(DllPath);

#if PLATFORM_WINDOWS
    // Add the library directory to DLL search path so dependencies (FFmpeg, onnxruntime) can be found
    FString AbsLibDir = FPaths::ConvertRelativePathToFull(LibDir);
    FPlatformProcess::AddDllDirectory(*AbsLibDir);
    UE_LOG(LogTemp, Log, TEXT("Added DLL search path: %s"), *AbsLibDir);
#endif

    GApi.DllHandle = FPlatformProcess::GetDllHandle(*DllPath);
    if (!GApi.DllHandle) {
        UE_LOG(LogTemp, Error, TEXT("Failed to load Beatsync library: %s"), *DllPath);
        return false;
    }

    // Load all function pointers
    GApi.resolve_ffmpeg = (bs_resolve_ffmpeg_path_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_resolve_ffmpeg_path"));
    GApi.create_analyzer = (bs_create_audio_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_create_audio_analyzer"));
    GApi.destroy_analyzer = (bs_destroy_audio_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_destroy_audio_analyzer"));
    GApi.analyze_audio = (bs_analyze_audio_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_analyze_audio"));
    GApi.free_beatgrid = (bs_free_beatgrid_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_free_beatgrid"));
    GApi.create_video_writer = (bs_create_video_writer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_create_video_writer"));
    GApi.destroy_video_writer = (bs_destroy_video_writer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_destroy_video_writer"));
    GApi.video_get_last_error = (bs_video_get_last_error_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_get_last_error"));
    GApi.video_set_progress_callback = (bs_video_set_progress_callback_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_set_progress_callback"));
    GApi.video_cut_at_beats = (bs_video_cut_at_beats_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_cut_at_beats"));
    GApi.video_cut_at_beats_multi = (bs_video_cut_at_beats_multi_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_cut_at_beats_multi"));
    GApi.video_concatenate = (bs_video_concatenate_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_concatenate"));
    GApi.video_add_audio_track = (bs_video_add_audio_track_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_add_audio_track"));
    GApi.get_waveform = (bs_get_waveform_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_get_waveform"));
    GApi.free_waveform = (bs_free_waveform_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_free_waveform"));
    GApi.video_set_effects_config = (bs_video_set_effects_config_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_set_effects_config"));
    GApi.video_apply_effects = (bs_video_apply_effects_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_apply_effects"));
    GApi.video_extract_frame = (bs_video_extract_frame_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_extract_frame"));
    GApi.free_frame_data = (bs_free_frame_data_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_free_frame_data"));

    // AI analyzer functions (ONNX neural network)
    GApi.ai_is_available = (bs_ai_is_available_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_ai_is_available"));
    GApi.ai_get_providers = (bs_ai_get_providers_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_ai_get_providers"));
    GApi.create_ai_analyzer = (bs_create_ai_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_create_ai_analyzer"));
    GApi.destroy_ai_analyzer = (bs_destroy_ai_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_destroy_ai_analyzer"));
    GApi.ai_analyze_file = (bs_ai_analyze_file_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_ai_analyze_file"));
    GApi.ai_analyze_quick = (bs_ai_analyze_quick_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_ai_analyze_quick"));
    GApi.free_ai_result = (bs_free_ai_result_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_free_ai_result"));
    GApi.ai_get_last_error = (bs_ai_get_last_error_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_ai_get_last_error"));

    // Check required symbols - audio analyzer
    if (!GApi.create_analyzer || !GApi.destroy_analyzer || !GApi.analyze_audio) {
        UE_LOG(LogTemp, Error, TEXT("Required audio analyzer symbols not found in Beatsync DLL"));
        FPlatformProcess::FreeDllHandle(GApi.DllHandle);
        GApi.DllHandle = nullptr;
        return false;
    }

    // Check required symbols - video writer (critical for video processing)
    if (!GApi.create_video_writer || !GApi.destroy_video_writer) {
        UE_LOG(LogTemp, Error, TEXT("Required video writer symbols not found in Beatsync DLL"));
        FPlatformProcess::FreeDllHandle(GApi.DllHandle);
        GApi.DllHandle = nullptr;
        return false;
    }

    // Log optional symbol availability for debugging
    if (!GApi.video_cut_at_beats) {
        UE_LOG(LogTemp, Warning, TEXT("Optional symbol bs_video_cut_at_beats not found"));
    }
    if (!GApi.video_add_audio_track) {
        UE_LOG(LogTemp, Warning, TEXT("Optional symbol bs_video_add_audio_track not found"));
    }
    if (!GApi.video_extract_frame) {
        UE_LOG(LogTemp, Warning, TEXT("Optional symbol bs_video_extract_frame not found"));
    }

    UE_LOG(LogTemp, Log, TEXT("Beatsync DLL loaded successfully: %s"), *DllPath);

    // Log AI availability
    if (GApi.ai_is_available && GApi.ai_is_available()) {
        UE_LOG(LogTemp, Log, TEXT("ONNX AI analyzer is available"));
        if (GApi.ai_get_providers) {
            const char* providers = GApi.ai_get_providers();
            if (providers) {
                UE_LOG(LogTemp, Log, TEXT("Available ONNX providers: %s"), UTF8_TO_TCHAR(providers));
            }
        }
    } else {
        UE_LOG(LogTemp, Warning, TEXT("ONNX AI analyzer is NOT available - will fall back to spectral flux"));
    }

    return true;
}

void FBeatsyncLoader::Shutdown()
{
    {
        FScopeLock Lock(&GProgressCallbacksLock);
        GProgressCallbacks.Empty();
    }
    // Wait for any in-flight callbacks to finish (with timeout to prevent indefinite hangs)
    if (GActiveProgressCallbacks.load() > 0 && GProgressCallbacksEvent) {
        constexpr uint32 TimeoutMs = 5000; // 5 second timeout
        if (!GProgressCallbacksEvent->Wait(TimeoutMs)) {
            UE_LOG(LogTemp, Warning, TEXT("Timeout waiting for progress callbacks to complete during shutdown"));
        }
    }
    if (GApi.DllHandle) {
        FPlatformProcess::FreeDllHandle(GApi.DllHandle);
        GApi = FBeatsyncApi();
    }
}

bool FBeatsyncLoader::IsInitialized()
{
    return GApi.DllHandle != nullptr;
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

void FBeatsyncLoader::DestroyAnalyzer(void* Handle)
{
    if (GApi.destroy_analyzer && Handle) {
        GApi.destroy_analyzer(Handle);
    }
}

bool FBeatsyncLoader::AnalyzeAudio(void* Analyzer, const FString& FilePath, FBeatGrid& OutGrid)
{
    if (!GApi.analyze_audio || !Analyzer) return false;

    bs_beatgrid_t CGrid = {};
    int Result = GApi.analyze_audio(Analyzer, TCHAR_TO_UTF8(*FilePath), &CGrid);

    bool bSuccess = false;
    if (Result == 0 && CGrid.beats && CGrid.count > 0) {
        OutGrid.Beats.SetNum(CGrid.count);
        FMemory::Memcpy(OutGrid.Beats.GetData(), CGrid.beats, CGrid.count * sizeof(double));
        OutGrid.BPM = CGrid.bpm;
        OutGrid.Duration = CGrid.duration;
        bSuccess = true;
    }

    // Always free the beatgrid if beats were allocated, regardless of Result
    if (CGrid.beats && GApi.free_beatgrid) {
        GApi.free_beatgrid(&CGrid);
    }

    return bSuccess;
}

void* FBeatsyncLoader::CreateVideoWriter()
{
    if (!GApi.create_video_writer) return nullptr;
    return GApi.create_video_writer();
}

void FBeatsyncLoader::DestroyVideoWriter(void* Handle)
{
    if (Handle) {
        // Remove progress callback entry to prevent UAF and leaks
        {
            FScopeLock Lock(&GProgressCallbacksLock);
            GProgressCallbacks.Remove(Handle);
        }

        if (GApi.destroy_video_writer) {
            GApi.destroy_video_writer(Handle);
        }
    }
}

FString FBeatsyncLoader::GetVideoLastError(void* Handle)
{
    if (!GApi.video_get_last_error || !Handle) return FString();
    const char* Err = GApi.video_get_last_error(Handle);
    return Err ? FString(UTF8_TO_TCHAR(Err)) : FString();
}

void FBeatsyncLoader::SetProgressCallback(void* Handle, TFunction<void(double)> Callback)
{
    if (!GApi.video_set_progress_callback || !Handle) return;

    {
        FScopeLock Lock(&GProgressCallbacksLock);
        // Replace or add the callback for this handle
        GProgressCallbacks.Add(Handle, MakeUnique<TFunction<void(double)>>(MoveTemp(Callback)));
    }
    
    GApi.video_set_progress_callback(Handle, StaticProgressCallback, Handle);
}

bool FBeatsyncLoader::CutVideoAtBeats(void* Handle, const FString& InputVideo, const TArray<double>& BeatTimes, const FString& OutputVideo, double ClipDuration)
{
    if (!GApi.video_cut_at_beats || !Handle || BeatTimes.Num() == 0) return false;

    int Result = GApi.video_cut_at_beats(
        Handle,
        TCHAR_TO_UTF8(*InputVideo),
        BeatTimes.GetData(),
        BeatTimes.Num(),
        TCHAR_TO_UTF8(*OutputVideo),
        ClipDuration
    );

    return Result == 0;
}

bool FBeatsyncLoader::CutVideoAtBeatsMulti(void* Handle, const TArray<FString>& InputVideos, const TArray<double>& BeatTimes, const FString& OutputVideo, double ClipDuration)
{
    if (!GApi.video_cut_at_beats_multi || !Handle || InputVideos.Num() == 0 || BeatTimes.Num() == 0) return false;

    // Convert FStrings to UTF8 and store them
    TArray<TArray<char>> ConvertedStrings;
    TArray<const char*> InputPtrs;

    for (const FString& Input : InputVideos) {
        FTCHARToUTF8 Converter(*Input);
        TArray<char> Utf8String;
        int32 Len = FCStringAnsi::Strlen(Converter.Get()) + 1;
        Utf8String.SetNum(Len);
        FMemory::Memcpy(Utf8String.GetData(), Converter.Get(), Len);
        ConvertedStrings.Add(MoveTemp(Utf8String));
        InputPtrs.Add(ConvertedStrings.Last().GetData());
    }

    int Result = GApi.video_cut_at_beats_multi(
        Handle,
        InputPtrs.GetData(),
        InputPtrs.Num(),
        BeatTimes.GetData(),
        BeatTimes.Num(),
        TCHAR_TO_UTF8(*OutputVideo),
        ClipDuration
    );

    return Result == 0;
}

bool FBeatsyncLoader::ConcatenateVideos(const TArray<FString>& Inputs, const FString& OutputVideo)
{
    if (!GApi.video_concatenate || Inputs.Num() == 0) return false;

    // Convert FStrings to UTF8 and store them
    TArray<TArray<char>> ConvertedStrings;
    TArray<const char*> InputPtrs;

    for (const FString& Input : Inputs) {
        FTCHARToUTF8 Converter(*Input);
        TArray<char> Utf8String;
        int32 Len = FCStringAnsi::Strlen(Converter.Get()) + 1;
        Utf8String.SetNum(Len);
        FMemory::Memcpy(Utf8String.GetData(), Converter.Get(), Len);
        ConvertedStrings.Add(MoveTemp(Utf8String));
        InputPtrs.Add(ConvertedStrings.Last().GetData());
    }

    int Result = GApi.video_concatenate(InputPtrs.GetData(), InputPtrs.Num(), TCHAR_TO_UTF8(*OutputVideo));
    return Result == 0;
}

bool FBeatsyncLoader::GetWaveform(void* Analyzer, const FString& FilePath, TArray<float>& OutPeaks, double& OutDuration)
{
    if (!GApi.get_waveform || !Analyzer) return false;

    float* Peaks = nullptr;
    size_t Count = 0;
    double Duration = 0.0;

    int Result = GApi.get_waveform(Analyzer, TCHAR_TO_UTF8(*FilePath), &Peaks, &Count, &Duration);

    if (Result == 0 && Peaks && Count > 0) {
        OutPeaks.SetNum(Count);
        FMemory::Memcpy(OutPeaks.GetData(), Peaks, Count * sizeof(float));
        OutDuration = Duration;

        if (GApi.free_waveform) {
            GApi.free_waveform(Peaks);
        }
        return true;
    }

    // Free Peaks if allocated on failure
    if (Peaks && GApi.free_waveform) {
        GApi.free_waveform(Peaks);
    }
    return false;
}

void FBeatsyncLoader::FreeWaveform(float* Peaks)
{
    if (GApi.free_waveform && Peaks) {
        GApi.free_waveform(Peaks);
    }
}

bool FBeatsyncLoader::AddAudioTrack(void* Handle, const FString& InputVideo, const FString& AudioFile,
                                     const FString& OutputVideo, bool bTrimToShortest,
                                     double AudioStart, double AudioEnd)
{
    if (!GApi.video_add_audio_track || !Handle) return false;

    int Result = GApi.video_add_audio_track(
        Handle,
        TCHAR_TO_UTF8(*InputVideo),
        TCHAR_TO_UTF8(*AudioFile),
        TCHAR_TO_UTF8(*OutputVideo),
        bTrimToShortest ? 1 : 0,
        AudioStart,
        AudioEnd
    );

    return Result == 0;
}

void FBeatsyncLoader::SetEffectsConfig(void* Handle, const FEffectsConfig& Config)
{
    if (!GApi.video_set_effects_config || !Handle) return;

    // Convert FStrings to UTF8 - need to keep them alive during the call
    FTCHARToUTF8 TransitionTypeUtf8(*Config.TransitionType);
    FTCHARToUTF8 ColorPresetUtf8(*Config.ColorPreset);

    bs_effects_config_t CConfig = {};
    CConfig.enableTransitions = Config.bEnableTransitions ? 1 : 0;
    CConfig.transitionType = TransitionTypeUtf8.Get();
    CConfig.transitionDuration = Config.TransitionDuration;
    CConfig.enableColorGrade = Config.bEnableColorGrade ? 1 : 0;
    CConfig.colorPreset = ColorPresetUtf8.Get();
    CConfig.enableVignette = Config.bEnableVignette ? 1 : 0;
    CConfig.vignetteStrength = Config.VignetteStrength;
    CConfig.enableBeatFlash = Config.bEnableBeatFlash ? 1 : 0;
    CConfig.flashIntensity = Config.FlashIntensity;
    CConfig.enableBeatZoom = Config.bEnableBeatZoom ? 1 : 0;
    CConfig.zoomIntensity = Config.ZoomIntensity;
    CConfig.effectBeatDivisor = Config.EffectBeatDivisor;

    GApi.video_set_effects_config(Handle, &CConfig);
}

bool FBeatsyncLoader::ApplyEffects(void* Handle, const FString& InputVideo, const FString& OutputVideo,
                                    const TArray<double>& BeatTimes)
{
    if (!GApi.video_apply_effects || !Handle) return false;

    int Result = GApi.video_apply_effects(
        Handle,
        TCHAR_TO_UTF8(*InputVideo),
        TCHAR_TO_UTF8(*OutputVideo),
        BeatTimes.Num() > 0 ? BeatTimes.GetData() : nullptr,
        BeatTimes.Num()
    );

    return Result == 0;
}

bool FBeatsyncLoader::ExtractFrame(const FString& VideoPath, double Timestamp,
                                    TArray<uint8>& OutData, int32& OutWidth, int32& OutHeight)
{
    if (!GApi.video_extract_frame) return false;

    unsigned char* FrameData = nullptr;
    int Width = 0;
    int Height = 0;

    int Result = GApi.video_extract_frame(
        TCHAR_TO_UTF8(*VideoPath),
        Timestamp,
        &FrameData,
        &Width,
        &Height
    );

    if (Result == 0 && FrameData && Width > 0 && Height > 0)
    {
        int32 DataSize = Width * Height * 3; // RGB24
        OutData.SetNum(DataSize);
        FMemory::Memcpy(OutData.GetData(), FrameData, DataSize);
        OutWidth = Width;
        OutHeight = Height;

        if (GApi.free_frame_data)
        {
            GApi.free_frame_data(FrameData);
        }
        return true;
    }

    return false;
}


// =============================================================================
// AI Analyzer (ONNX neural network - GPU accelerated)
// =============================================================================

bool FBeatsyncLoader::IsAIAvailable()
{
    if (!GApi.ai_is_available) return false;
    return GApi.ai_is_available() != 0;
}

FString FBeatsyncLoader::GetAIProviders()
{
    if (!GApi.ai_get_providers) return FString();
    const char* providers = GApi.ai_get_providers();
    return providers ? FString(UTF8_TO_TCHAR(providers)) : FString();
}

void* FBeatsyncLoader::CreateAIAnalyzer(const FAIConfig& Config)
{
    if (!GApi.create_ai_analyzer) return nullptr;

    // Convert FStrings to UTF8 - need to keep them alive during the call
    FTCHARToUTF8 BeatModelUtf8(*Config.BeatModelPath);
    FTCHARToUTF8 StemModelUtf8(*Config.StemModelPath);

    bs_ai_config_t CConfig = {};
    CConfig.beat_model_path = BeatModelUtf8.Get();
    CConfig.stem_model_path = Config.StemModelPath.IsEmpty() ? nullptr : StemModelUtf8.Get();
    CConfig.use_stem_separation = Config.bUseStemSeparation ? 1 : 0;
    CConfig.use_drums_for_beats = Config.bUseDrumsForBeats ? 1 : 0;
    CConfig.use_gpu = Config.bUseGPU ? 1 : 0;
    CConfig.gpu_device_id = Config.GPUDeviceId;
    CConfig.beat_threshold = Config.BeatThreshold;
    CConfig.downbeat_threshold = Config.DownbeatThreshold;

    void* Handle = GApi.create_ai_analyzer(&CConfig);
    if (Handle) {
        UE_LOG(LogTemp, Log, TEXT("Created AI analyzer with GPU=%d, model=%s"),
               Config.bUseGPU ? 1 : 0, *Config.BeatModelPath);
    }
    return Handle;
}

void FBeatsyncLoader::DestroyAIAnalyzer(void* Handle)
{
    if (GApi.destroy_ai_analyzer && Handle) {
        GApi.destroy_ai_analyzer(Handle);
    }
}

bool FBeatsyncLoader::AIAnalyzeFile(void* Analyzer, const FString& FilePath, FAIResult& OutResult)
{
    if (!GApi.ai_analyze_file || !Analyzer) return false;

    bs_ai_result_t CResult = {};
    int Result = GApi.ai_analyze_file(Analyzer, TCHAR_TO_UTF8(*FilePath), &CResult, nullptr, nullptr);

    if (Result == 0) {
        // Copy beats
        if (CResult.beats && CResult.beat_count > 0) {
            OutResult.Beats.SetNum(CResult.beat_count);
            FMemory::Memcpy(OutResult.Beats.GetData(), CResult.beats, CResult.beat_count * sizeof(double));
        }
        // Copy downbeats
        if (CResult.downbeats && CResult.downbeat_count > 0) {
            OutResult.Downbeats.SetNum(CResult.downbeat_count);
            FMemory::Memcpy(OutResult.Downbeats.GetData(), CResult.downbeats, CResult.downbeat_count * sizeof(double));
        }
        OutResult.BPM = CResult.bpm;
        OutResult.Duration = CResult.duration;

        if (GApi.free_ai_result) {
            GApi.free_ai_result(&CResult);
        }
        return true;
    }

    return false;
}

bool FBeatsyncLoader::AIAnalyzeQuick(void* Analyzer, const FString& FilePath, FAIResult& OutResult)
{
    if (!GApi.ai_analyze_quick || !Analyzer) return false;

    bs_ai_result_t CResult = {};
    int Result = GApi.ai_analyze_quick(Analyzer, TCHAR_TO_UTF8(*FilePath), &CResult, nullptr, nullptr);

    if (Result == 0) {
        // Copy beats
        if (CResult.beats && CResult.beat_count > 0) {
            OutResult.Beats.SetNum(CResult.beat_count);
            FMemory::Memcpy(OutResult.Beats.GetData(), CResult.beats, CResult.beat_count * sizeof(double));
        }
        // Copy downbeats
        if (CResult.downbeats && CResult.downbeat_count > 0) {
            OutResult.Downbeats.SetNum(CResult.downbeat_count);
            FMemory::Memcpy(OutResult.Downbeats.GetData(), CResult.downbeats, CResult.downbeat_count * sizeof(double));
        }
        OutResult.BPM = CResult.bpm;
        OutResult.Duration = CResult.duration;

        if (GApi.free_ai_result) {
            GApi.free_ai_result(&CResult);
        } else {
            UE_LOG(LogTemp, Warning, TEXT("FBeatsyncLoader::AIAnalyzeQuick: GApi.free_ai_result is null, CResult may leak!"));
        }
        return true;
    }

    return false;
}

FString FBeatsyncLoader::GetAILastError(void* Analyzer)
{
    if (!GApi.ai_get_last_error || !Analyzer) return FString();
    const char* Err = GApi.ai_get_last_error(Analyzer);
    return Err ? FString(UTF8_TO_TCHAR(Err)) : FString();
}
