#pragma once

#include "CoreMinimal.h"
#include "Misc/Paths.h"
#include <functional>

#include "BeatsyncTypes.h" // Shared types (FBeatGrid, FEffectsConfig, etc.)

// AUTHORITATIVE HEADER: This is the canonical FBeatsyncLoader class declaration.
// All other modules should include this header for the complete API.
// For extended functionality, see FBeatsyncLoaderExtended in application-specific headers.
//
// CONSOLIDATION: Duplicate class definitions have been removed. The authoritative
// declaration includes: Initialize, ResolveFFmpegPath, CreateAnalyzer/DestroyAnalyzer,
// CreateVideoWriter/DestroyVideoWriter, SetProgressCallback, CutVideoAtBeats/CutVideoAtBeatsMulti,
// ApplyEffects, SetEffectsConfig, AddAudioTrack, ExtractFrame, StartSpan/EndSpan/SpanSetError/SpanAddEvent, IsInitialized.

class FBeatsyncLoader
{
public:
    static bool Initialize();

    /**
     * @brief Shuts down the BeatsyncLoader and clears all registered progress callback storage.
     *
     * This function only clears the internal callback registration storage (GCallbackStorage)
     * under lock (GCallbackStorageMutex). It does NOT cancel or block on any in-flight callbacks.
     *
     * Callbacks that are currently executing (in-flight) hold their own TSharedPtr copies of the
     * callback data via the trampoline mechanism, so their lifetime is independent of the storage.
     *
     * Callers MUST ensure that all asynchronous operations and work that could invoke callbacks
     * have fully completed before calling Shutdown. Failure to do so may result in use-after-free
     * or undefined behavior if a callback is invoked after storage is cleared.
     *
     * References: FBeatsyncLoader::Shutdown, GCallbackStorage, GCallbackStorageMutex, and the
     * ProgressCallbackTrampoline behavior in the implementation for details on lifetime guarantees.
     */
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
    struct CallbackData;  // Forward declaration - defined in .cpp
    // Internal state handled in .cpp
};
