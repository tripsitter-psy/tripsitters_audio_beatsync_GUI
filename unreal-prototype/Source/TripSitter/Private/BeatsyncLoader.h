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
    /** Beat divisor for effects (must be >= 1 to prevent division by zero).
     *  Value of 1 = every beat, 2 = every other beat, etc. */
    int32 EffectBeatDivisor = 1;
    double EffectStartTime = 0.0;
    double EffectEndTime = -1.0;
};

// Per-clip speed ramp configuration (slow-mo / speed-up).
// Affected clips keep their beat-slot duration; the multiplier only changes how
// much source footage is sampled into the slot. Multipliers clamp to [0.5, 2.0].
struct FSpeedRampConfig
{
    bool bEnabled = false;
    float AffectedFraction = 0.25f;   // 0..1 portion of beat clips affected (random mode)
    uint32 Seed = 0;                  // reproducible randomization
    int32 SelectionMode = 0;          // 0=random, 1=every Nth, 2=every Nth (beat divisor)
    int32 EveryN = 4;
    float SpeedUpFraction = 0.5f;     // of affected clips, portion that speed up (>1x)
    float SlowMin = 0.5f;             // slow-mo range (<1.0); min==max for fixed amount
    float SlowMax = 0.5f;
    float FastMin = 2.0f;             // speed-up range (>1.0)
    float FastMax = 2.0f;
    int32 Smoothing = 0;              // 0=duplicate frames, 1=minterpolate optical flow
    bool bGuardClampToAvailable = true;
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
    // Kick-only mode for psytrance/EDM (applies 200Hz low-pass filter before beat detection)
    bool bKickOnlyMode = false;
    float KickFreqCutoff = 200.0f;
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
    /** Set progress callback for video processing operations.
     *  WARNING: The callback may be invoked from a worker thread. If updating UI,
     *  the callback implementation MUST marshal calls to the GameThread using
     *  AsyncTask(ENamedThreads::GameThread, ...) to avoid Slate threading assertions.
     *  The callback must remain valid for the lifetime of the video processing operation. */
    static void SetProgressCallback(void* Handle, TFunction<void(double)> Callback);
    // Stage-aware progress ("upscale", "normalize", "cut", "effects", "mux" with the
    // stage's own 0..1 completion). Called on a backend worker thread. No-op when the
    // loaded backend predates bs_video_set_stage_progress_callback.
    static void SetStageProgressCallback(void* Handle, TFunction<void(const FString&, double)> Callback);
    // Duration/size/fps of a media file via the backend (libavformat). False if unsupported.
    static bool ProbeVideo(const FString& Path, double& OutDuration, int32& OutWidth, int32& OutHeight, double& OutFps);

    /** Set cancel flag for video processing operations.
     *  Pass a pointer to an int that will be checked periodically during processing.
     *  Set *CancelFlag to non-zero to request cancellation.
     *  The flag must remain valid for the lifetime of the video processing operation. */
    static void SetCancelFlag(void* Handle, const int* CancelFlag);

    /** Set output resolution + frame rate. 1920x1080 = landscape (default), 1080x1920 = vertical/portrait.
     *  Must be called before cut/normalize operations. */
    static void SetOutputSettings(void* Handle, int Width, int Height, int Fps);

    // Neural source upscaling (applied during normalization). The model is
    // resolved next to the executable; pass bEnabled=false to disable.
    // ModelFile is a filename inside the executable's models/ directory
    // (e.g. "upscale.onnx" for 4x, "upscale_2x.onnx" for 2x). Empty disables.
    static void SetUpscaleConfig(void* Handle, const FString& ModelFile, int32 TileSize = 512,
                                 int32 MaxSourceEdge = 1440);

    /** Check if video processing was cancelled */
    static bool IsCancelled(void* Handle);

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

    // Dynamic sync (energy-driven cut density) + speed ramps
    static bool DynamicSyncFilterBeats(const FString& AudioPath, const TArray<double>& Beats, TArray<double>& OutFiltered);
    static bool DynamicSyncClassifyBeats(const FString& AudioPath, const TArray<double>& Beats, TArray<int32>& OutBands);
    static void SetSpeedRampConfig(void* Handle, bool bEnabled, double CalmSpeed, double NormalSpeed,
                                   double FranticSpeed, const FString& InterpMode, const TArray<int32>& BeatBands);
    static bool ApplyEffects(void* Handle, const FString& InputVideo, const FString& OutputVideo,
                              const TArray<double>& BeatTimes);

    // Per-clip speed ramps. Call before a cut operation. Pass bEnabled=false to disable.
    static void SetSpeedConfig(void* Handle, const FSpeedRampConfig& Config);
    // Clips clamped/skipped by the source-footage guard during the most recent cut.
    static int32 GetSpeedClampCount(void* Handle);
    // RIFE ONNX model path for neural slow-mo interpolation (smoothing == 2). Empty to clear.
    static void SetInterpolationModel(void* Handle, const FString& OnnxPath);

    // Frame Extraction
    static bool ExtractFrame(const FString& VideoPath, double Timestamp,
                              TArray<uint8>& OutData, int32& OutWidth, int32& OutHeight);

    // AI Analyzer (ONNX neural network - GPU accelerated)
    static bool IsAIAvailable();
    static FString GetAIProviders();
    // Which execution provider AI stages will really use on this machine (creates a
    // throwaway session, ~1 s on a GPU). Returns 1 = GPU, 0 = CPU only, -1 = unknown.
    // OutSummary e.g. "CUDA (NVIDIA GPU)" or "CPU only - no GPU acceleration".
    static int32 ProbeAcceleration(FString& OutSummary);
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
