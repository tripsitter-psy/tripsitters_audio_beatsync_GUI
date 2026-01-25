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

// Handle types for type safety
struct FAnalyzerHandle {
    void* Ptr = nullptr;
    bool IsValid() const { return Ptr != nullptr; }
};

struct FVideoWriterHandle {
    void* Ptr = nullptr;
    bool IsValid() const { return Ptr != nullptr; }
};

struct FSpanHandle {
    void* Ptr = nullptr;
    bool IsValid() const { return Ptr != nullptr; }
};

struct FAIAnalyzerHandle {
    void* Ptr = nullptr;
    bool IsValid() const { return Ptr != nullptr; }
};

class TRIPSITTERUE_API FBeatsyncLoader
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
    static FAnalyzerHandle CreateAnalyzer();
    static void DestroyAnalyzer(FAnalyzerHandle handle);
    static bool AnalyzeAudio(FAnalyzerHandle handle, const FString& path, FBeatGrid& outGrid);

    // Waveform Analysis
    static bool GetWaveform(FAnalyzerHandle handle, const FString& path, TArray<float>& outPeaks, double& outDuration);
    static bool GetWaveformBands(FAnalyzerHandle handle, const FString& path, TArray<float>& outBass, TArray<float>& outMid, TArray<float>& outHigh, double& outDuration);

    // AI Analysis
    static bool IsAIAvailable();
    static FString GetAIProviders();
    static FAIAnalyzerHandle CreateAIAnalyzer(const FAIConfig& Config);
    static void DestroyAIAnalyzer(FAIAnalyzerHandle Handle);
    static bool AIAnalyzeFile(FAIAnalyzerHandle Analyzer, const FString& FilePath, FAIResult& OutResult);
    static bool AIAnalyzeQuick(FAIAnalyzerHandle Analyzer, const FString& FilePath, FAIResult& OutResult);
    static FString GetAILastError(FAIAnalyzerHandle Analyzer);

    // AudioFlux Analysis
    static bool IsAudioFluxAvailable();
    static bool AudioFluxAnalyze(const FString& FilePath, FAIResult& OutResult);
    static bool AudioFluxAnalyzeWithStems(const FString& FilePath, const FString& StemModelPath, FAIResult& OutResult);

    // Video writer
    static FVideoWriterHandle CreateVideoWriter();
    static void DestroyVideoWriter(FVideoWriterHandle writer);
    static FString GetVideoLastError(FVideoWriterHandle writer);

    using FProgressCb = TFunction<void(double)>;
    static void SetProgressCallback(FVideoWriterHandle writer, FProgressCb cb);

    static bool CutVideoAtBeats(FVideoWriterHandle writer, const FString& inputVideo, const TArray<double>& beatTimes, const FString& outputVideo, double clipDuration);
    static bool CutVideoAtBeatsMulti(FVideoWriterHandle writer, const TArray<FString>& inputVideos, const TArray<double>& beatTimes, const FString& outputVideo, double clipDuration);
    static bool ApplyEffects(FVideoWriterHandle writer, const FString& inputVideo, const FString& outputVideo, const TArray<double>& beatTimes);
    static void SetEffectsConfig(FVideoWriterHandle writer, const FEffectsConfig& config);
    static bool AddAudioTrack(FVideoWriterHandle writer, const FString& inputVideo, const FString& audioFile, const FString& outputVideo, bool trimToShortest, double audioStart, double audioEnd);

    // Frame extraction
    // Note: ExtractFrame copies data to TArray and frees the C buffer internally
    static bool ExtractFrame(const FString& videoPath, double timestamp, TArray<uint8>& outRgb24, int32& outWidth, int32& outHeight);

    // Tracing helpers (opaque handles managed by backend)
    //
    // FSpanHandle wraps an opaque pointer to a backend span object.
    // Callers should use FSpanHandle::IsValid() to check validity before use,
    // though the span functions are null-safe and will silently no-op on invalid handles.

    /**
     * @brief Starts a new tracing span with the given name.
     * @param name The name/label for this span (e.g., "VideoProcessing", "AudioAnalysis").
     * @return A valid FSpanHandle on success, or an invalid handle (Ptr == nullptr) on failure
     *         (e.g., if tracing is not initialized or backend unavailable).
     *         Callers may check IsValid() but are not required to - passing an invalid handle
     *         to other span functions is safe (they will no-op).
     */
    static FSpanHandle StartSpan(const FString& name);

    /**
     * @brief Ends a tracing span and records its duration.
     * @param h The span handle returned by StartSpan. Null-safe: if h is invalid (nullptr),
     *          this function is a no-op and returns immediately without error.
     */
    static void EndSpan(FSpanHandle h);

    /**
     * @brief Marks a span as having encountered an error.
     * @param h The span handle. Null-safe: no-op if invalid.
     * @param msg The error message to record in the span.
     */
    static void SpanSetError(FSpanHandle h, const FString& msg);

    /**
     * @brief Adds a timestamped event to a span.
     * @param h The span handle. Null-safe: no-op if invalid.
     * @param ev The event name/description to record.
     */
    static void SpanAddEvent(FSpanHandle h, const FString& ev);

    static bool IsInitialized();

private:
    struct CallbackData;  // Forward declaration - defined in .cpp
    // Internal state handled in .cpp
};
