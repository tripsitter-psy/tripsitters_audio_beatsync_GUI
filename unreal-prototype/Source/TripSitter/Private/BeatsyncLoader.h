#pragma once

#include "CoreMinimal.h"

// Beat grid result from audio analysis
struct FBeatGrid
{
    TArray<double> Beats;
    double BPM = 0.0;
    double Duration = 0.0;
};

/**
 * Video transition types for beat-synced effects
 */
enum class ETransitionType : uint8
{
    Fade,
    Wipe,
    Dissolve,
    Zoom
};

/**
 * Color grading presets for video processing
 */
enum class EColorPreset : uint8
{
    Warm,
    Cool,
    Vintage,
    Vibrant
};

/**
 * Convert ETransitionType to lowercase string for C API
 */
inline FString TransitionTypeToString(ETransitionType Type)
{
    switch (Type)
    {
        case ETransitionType::Fade: return TEXT("fade");
        case ETransitionType::Wipe: return TEXT("wipe");
        case ETransitionType::Dissolve: return TEXT("dissolve");
        case ETransitionType::Zoom: return TEXT("zoom");
        default: return TEXT("fade");
    }
}

/**
 * Convert ETransitionType to display string for UI
 */
inline FString TransitionTypeToDisplayString(ETransitionType Type)
{
    switch (Type)
    {
        case ETransitionType::Fade: return TEXT("Fade");
        case ETransitionType::Wipe: return TEXT("Wipe");
        case ETransitionType::Dissolve: return TEXT("Dissolve");
        case ETransitionType::Zoom: return TEXT("Zoom");
        default: return TEXT("Fade");
    }
}

/**
 * Convert EColorPreset to lowercase string for C API
 */
inline FString ColorPresetToString(EColorPreset Preset)
{
    switch (Preset)
    {
        case EColorPreset::Warm: return TEXT("warm");
        case EColorPreset::Cool: return TEXT("cool");
        case EColorPreset::Vintage: return TEXT("vintage");
        case EColorPreset::Vibrant: return TEXT("vibrant");
        default: return TEXT("warm");
    }
}

/**
 * Convert EColorPreset to display string for UI
 */
inline FString ColorPresetToDisplayString(EColorPreset Preset)
{
    switch (Preset)
    {
        case EColorPreset::Warm: return TEXT("Warm");
        case EColorPreset::Cool: return TEXT("Cool");
        case EColorPreset::Vintage: return TEXT("Vintage");
        case EColorPreset::Vibrant: return TEXT("Vibrant");
        default: return TEXT("Warm");
    }
}

// Effects configuration for video processing
struct FEffectsConfig
{
    bool bEnableTransitions = false;
    ETransitionType TransitionType = ETransitionType::Fade;
    double TransitionDuration = 0.5;
    bool bEnableColorGrade = false;
    EColorPreset ColorPreset = EColorPreset::Warm;
    bool bEnableVignette = false;
    double VignetteStrength = 0.3;
    bool bEnableBeatFlash = false;
    double FlashIntensity = 0.5;
    bool bEnableBeatZoom = false;
    double ZoomIntensity = 0.1;
    int32 EffectBeatDivisor = 1;
    double EffectStartTime = 0.0;
    double EffectEndTime = -1.0;
};

// AI configuration for ONNX neural network analysis
struct FAIConfig
{
    FString BeatModelPath;
    FString StemModelPath;
    bool bEnableStemSeparation = false;
    bool bEnableDrumsForBeats = true;
    bool bEnableGPU = true;
    int32 GPUDeviceId = 0;
    float BeatThreshold = 0.66f;
    float DownbeatThreshold = 0.66f;
};

// AI analysis result
struct FAIResult
{
    TArray<double> Beats;
    TArray<double> Downbeats;
    double BPM = 0.0;
    double Duration = 0.0;
};

/**
 * Static loader class for the Beatsync backend DLL
 * Provides audio analysis, video processing, and AI beat detection
 */
class FBeatsyncLoader
{
public:
    // Initialization
    static bool Initialize();
    static void Shutdown();
    static bool IsInitialized();
    static FString ResolveFFmpegPath();

    // Audio Analysis
    static void* CreateAnalyzer();
    static void DestroyAnalyzer(void* Handle);
    static FString GetAnalyzerLastError(void* Analyzer);
    static void SetBPMHint(void* Analyzer, double BPM);  // 0 = auto-detect, >0 = use this BPM
    static bool AnalyzeAudio(void* Analyzer, const FString& FilePath, FBeatGrid& OutGrid);

    // Waveform visualization
    // Note: OutPeaks is always managed by the caller (TArray). No memory needs to be freed.
    // The internal buffer from the backend is copied and freed automatically.
    static bool GetWaveform(void* Analyzer, const FString& FilePath, TArray<float>& OutPeaks, double& OutDuration);
    static bool GetWaveformBands(void* Analyzer, const FString& FilePath,
                                  TArray<float>& OutBassPeaks, TArray<float>& OutMidPeaks,
                                  TArray<float>& OutHighPeaks, double& OutDuration);

    // Video Writer
    static void* CreateVideoWriter();
    static void DestroyVideoWriter(void* Handle);
    static FString GetVideoLastError(void* Handle);
    static void SetProgressCallback(void* Handle, TFunction<void(double)> Callback);

    // Video Processing
    static bool CutVideoAtBeats(void* Handle, const FString& InputVideo, const TArray<double>& BeatTimes,
                                 const FString& OutputVideo, double ClipDuration);
    static bool CutVideoAtBeatsMulti(void* Handle, const TArray<FString>& InputVideos, const TArray<double>& BeatTimes,
                                      const FString& OutputVideo, double ClipDuration);
    static bool ConcatenateVideos(const TArray<FString>& Inputs, const FString& OutputVideo);
    static bool AddAudioTrack(void* Handle, const FString& InputVideo, const FString& AudioFile,
                               const FString& OutputVideo, bool bTrimToShortest, double AudioStart, double AudioEnd);

    // Video Normalization
    static bool NormalizeVideos(void* Handle, const TArray<FString>& InputVideos, TArray<FString>& OutNormalizedPaths);
    static void CleanupNormalizedVideos(const TArray<FString>& NormalizedPaths);

    // Effects
    static void SetEffectsConfig(void* Handle, const FEffectsConfig& Config);
    static bool ApplyEffects(void* Handle, const FString& InputVideo, const FString& OutputVideo,
                              const TArray<double>& BeatTimes);

    // Frame Extraction
    static bool ExtractFrame(const FString& VideoPath, double Timestamp,
                              TArray<uint8>& OutData, int32& OutWidth, int32& OutHeight);

    // AI Analyzer (ONNX neural network - GPU accelerated)
    static bool IsAIAvailable();
    static FString GetAIProviders();
    static void* CreateAIAnalyzer(const FAIConfig& Config);
    static void DestroyAIAnalyzer(void* Handle);
    static bool AIAnalyzeFile(void* Analyzer, const FString& FilePath, FAIResult& OutResult);
    static bool AIAnalyzeQuick(void* Analyzer, const FString& FilePath, FAIResult& OutResult);
    static FString GetAILastError(void* Analyzer);

    // AudioFlux Analyzer (signal processing - CPU only)
    static bool IsAudioFluxAvailable();
    static bool AudioFluxAnalyze(const FString& FilePath, FAIResult& OutResult);
    static bool AudioFluxAnalyzeWithStems(const FString& FilePath, const FString& StemModelPath, FAIResult& OutResult);
};
