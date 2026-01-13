#pragma once

#include "CoreMinimal.h"
#include "Misc/Paths.h"
#include <functional>


#include "BeatsyncTypes.h"

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

    // Callback data structure - public so static trampoline function can access it
    struct CallbackData
    {
        FProgressCb Func;
        void* Key = nullptr; // O(1) lookup key for callback storage (e.g., writer pointer)
    };

private:
    // Internal state handled in .cpp
};
