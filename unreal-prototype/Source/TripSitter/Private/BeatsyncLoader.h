#pragma once

#include "CoreMinimal.h"

// Beat grid structure matching C API
struct FBeatGrid
{
    TArray<double> Beats;
    double BPM = 0.0;
    double Duration = 0.0;
};

// Effects configuration for video processing
struct FEffectsConfig
{
    bool bEnableTransitions = false;
    FString TransitionType = TEXT("fade");
    float TransitionDuration = 0.5f;

    bool bEnableColorGrade = false;
    FString ColorPreset = TEXT("warm");

    bool bEnableVignette = false;
    float VignetteStrength = 0.5f;

    bool bEnableBeatFlash = false;
    float FlashIntensity = 0.5f;

    bool bEnableBeatZoom = false;
    float ZoomIntensity = 0.5f;

    int32 EffectBeatDivisor = 1;
};

// AI analyzer configuration (ONNX-based neural network inference)
struct FAIConfig
{
    FString BeatModelPath;          // Path to beat detection ONNX model
    FString StemModelPath;          // Path to stem separation ONNX model (optional)
    bool bUseStemSeparation = false; // Enable stem separation before beat detection
    bool bUseDrumsForBeats = true;   // Use drums stem for beat detection (recommended)
    bool bUseGPU = true;             // Enable GPU acceleration if available
    int32 GPUDeviceId = 0;           // GPU device ID
    float BeatThreshold = 0.5f;      // Beat activation threshold (0.0-1.0)
    float DownbeatThreshold = 0.5f;  // Downbeat activation threshold (0.0-1.0)
};

// AI analysis result (extended beat grid)
struct FAIResult
{
    TArray<double> Beats;
    TArray<double> Downbeats;
    double BPM = 0.0;
    double Duration = 0.0;
};

class FBeatsyncLoader
{
public:
    static bool Initialize();
    static void Shutdown();
    static bool IsInitialized();

    // FFmpeg
    static FString ResolveFFmpegPath();

    // Audio Analyzer
    static void* CreateAnalyzer();
    static void DestroyAnalyzer(void* Handle);
    static bool AnalyzeAudio(void* Analyzer, const FString& FilePath, FBeatGrid& OutGrid);

    // Waveform visualization
    static bool GetWaveform(void* Analyzer, const FString& FilePath, TArray<float>& OutPeaks, double& OutDuration);
    static void FreeWaveform(float* Peaks);

    // Video Writer
    static void* CreateVideoWriter();
    static void DestroyVideoWriter(void* Handle);
    static FString GetVideoLastError(void* Handle);
    static void SetProgressCallback(void* Handle, TFunction<void(double)> Callback);
    static bool CutVideoAtBeats(void* Handle, const FString& InputVideo, const TArray<double>& BeatTimes, const FString& OutputVideo, double ClipDuration);
    // Multi-video version: cycles through input videos for each beat
    static bool CutVideoAtBeatsMulti(void* Handle, const TArray<FString>& InputVideos, const TArray<double>& BeatTimes, const FString& OutputVideo, double ClipDuration);
    static bool ConcatenateVideos(const TArray<FString>& Inputs, const FString& OutputVideo);
    // Add audio track to video file
    // audioStart/audioEnd: trim audio to selection (-1 for audioEnd means no trim)
    // bTrimToShortest: if true, output ends when shorter stream ends
    static bool AddAudioTrack(void* Handle, const FString& InputVideo, const FString& AudioFile,
                              const FString& OutputVideo, bool bTrimToShortest = true,
                              double AudioStart = 0.0, double AudioEnd = -1.0);

    // Effects configuration
    static void SetEffectsConfig(void* Handle, const FEffectsConfig& Config);
    static bool ApplyEffects(void* Handle, const FString& InputVideo, const FString& OutputVideo,
                             const TArray<double>& BeatTimes);

    // Frame extraction for preview
    static bool ExtractFrame(const FString& VideoPath, double Timestamp,
                             TArray<uint8>& OutData, int32& OutWidth, int32& OutHeight);
    static void FreeFrameData(uint8* Data);

    // AI Analyzer (ONNX neural network) - GPU accelerated
    static bool IsAIAvailable();
    static FString GetAIProviders(); // Returns available execution providers (e.g., "CUDAExecutionProvider, CPUExecutionProvider")
    static void* CreateAIAnalyzer(const FAIConfig& Config);
    static void DestroyAIAnalyzer(void* Handle);
    static bool AIAnalyzeFile(void* Analyzer, const FString& FilePath, FAIResult& OutResult);
    static bool AIAnalyzeQuick(void* Analyzer, const FString& FilePath, FAIResult& OutResult); // Faster, no stem separation
    static FString GetAILastError(void* Analyzer);
};
