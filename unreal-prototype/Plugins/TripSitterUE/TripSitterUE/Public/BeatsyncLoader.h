#pragma once

#include "CoreMinimal.h"
#include "Misc/Paths.h"
#include <functional>

struct FBeatGrid
{
    TArray<double> Beats;
    double BPM = 0.0;
    double Duration = 0.0;
};

struct FEffectsConfig
{
    bool bEnableTransitions = false;
    FString TransitionType;
    double TransitionDuration = 0.0;

    bool bEnableColorGrade = false;
    FString ColorPreset;

    bool bEnableVignette = false;
    double VignetteStrength = 0.0;

    bool bEnableBeatFlash = false;
    double FlashIntensity = 0.0;

    bool bEnableBeatZoom = false;
    double ZoomIntensity = 0.0;

    int32 EffectBeatDivisor = 1;
};

class FBeatsyncLoader
{
public:
    static bool Initialize();
    static void Shutdown();

    static FString ResolveFFmpegPath();

    // Analyzer
    static void* CreateAnalyzer();
    static void DestroyAnalyzer(void* handle);
    static bool AnalyzeAudio(void* handle, const FString& path, FBeatGrid& outGrid);

    // Video writer
    static void* CreateVideoWriter();
    static void DestroyVideoWriter(void* writer);
    static FString GetVideoLastError(void* writer);

    using FProgressCb = TFunction<void(double)>;
    static void SetProgressCallback(void* writer, FProgressCb cb);

    static bool CutVideoAtBeats(void* writer, const FString& inputVideo, const TArray<double>& beatTimes, const FString& outputVideo, double clipDuration);
    static bool CutVideoAtBeatsMulti(void* writer, const TArray<FString>& inputVideos, const TArray<double>& beatTimes, const FString& outputVideo, double clipDuration);
    static bool ApplyEffects(void* writer, const FString& inputVideo, const FString& outputVideo, const TArray<double>& beatTimes);
    static void SetEffectsConfig(void* writer, const FEffectsConfig& config);
    static bool AddAudioTrack(void* writer, const FString& inputVideo, const FString& audioFile, const FString& outputVideo, bool trimToShortest, double audioStart, double audioEnd);

    // Frame extraction
    // Note: ExtractFrame copies data to TArray and frees the C buffer internally
    static bool ExtractFrame(const FString& videoPath, double timestamp, TArray<uint8>& outRgb24, int32& outWidth, int32& outHeight);

    // Tracing helpers (opaque handles managed by backend)
    using SpanHandle = void*;
    static SpanHandle StartSpan(const FString& name);
    static void EndSpan(SpanHandle h);
    static void SpanSetError(SpanHandle h, const FString& msg);
    static void SpanAddEvent(SpanHandle h, const FString& ev);

    static bool IsInitialized();

private:
    struct CallbackData
    {
        FProgressCb Func;
    };
    // Internal state handled in .cpp
};
