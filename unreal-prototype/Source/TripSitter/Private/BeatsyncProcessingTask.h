#pragma once

#include "CoreMinimal.h"
#include "Async/AsyncWork.h"
#include "HAL/ThreadSafeBool.h"
#include "BeatsyncLoader.h"
#include <atomic>

// Analysis mode enum (matches STripSitterMainWidget::EAnalysisMode)
enum class EAnalysisModeParam
{
    Energy = 0,    // Fast CPU-based spectral flux
    AIBeat = 1,    // AI beat detection (ONNX)
    AIStems = 2    // AI + stem separation (best accuracy)
    ,
    AudioFlux = 3, // AudioFlux-based detector
    StemsFlux = 4  // Stem-aware AudioFlux detector
};

// Stem effect type enum (matches STripSitterMainWidget::EStemEffect)
enum class EStemEffectParam : uint8
{
    None = 0,
    Flash = 1,
    Zoom = 2,
    Vignette = 3,
    ColorGrade = 4
};

// Stem effect configuration - maps a stem's beat times to an effect
struct FStemEffectConfig
{
    TArray<double> BeatTimes;           // Beat times detected from this stem
    EStemEffectParam Effect = EStemEffectParam::None;  // Which effect to apply at these beats
    bool bEnabled = false;              // Whether this stem track is active
};

struct FBeatsyncProcessingParams
{
    FString AudioPath;
    // VideoPath is used when bIsMultiClip is false, VideoPaths when true
    FString VideoPath;
    TArray<FString> VideoPaths;
    FString OutputPath;
    bool bIsMultiClip = false;
    // BeatRate is a non-negative exponent: BeatDivisor = 2^BeatRate
    // BeatRate = 0 → every beat; BeatRate = 1 → every 2nd beat; BeatRate = 2 → every 4th beat, etc.
    // Valid range: [0, 3]. BeatRate must be clamped to this range so BeatDivisor = 1 << BeatRate does not overflow the expected range (see clamp to ClampedBeatRate and shift in BeatsyncProcessingTask.cpp).
    int32 BeatRate = 0;
    double AudioStart = 0.0;
    double AudioEnd = -1.0;
    FEffectsConfig EffectsConfig;
    // Per-clip speed ramps (slow-mo / speed-up). Disabled by default.
    FSpeedRampConfig SpeedConfig;
    // Analysis mode: determines which beat detection method to use
    EAnalysisModeParam AnalysisMode = EAnalysisModeParam::AIBeat;
    // Pre-analyzed beat times from the UI (user-edited markers)
    // If non-empty, these are used instead of re-analyzing the audio
    TArray<double> PreAnalyzedBeatTimes;
    double PreAnalyzedBPM = 0.0;

    // New Pro feature: Respect arrangement breaks using energy/segment analysis
    // When true, automatically thins or removes beats in low-energy drops and atmospheric sections
    bool bRespectBreaks = true;
    float BreakEnergyThreshold = 0.15f;  // Tune this for psytrance (lower = more aggressive at removing beats in breaks)

    // Stem effect configurations (Kick, Snare, Hi-Hat, Synth)
    // Each stem can have its own beat times and mapped effect
    FStemEffectConfig StemConfigs[4];

    // Output orientation: false = 1920x1080 landscape (default), true = 1080x1920 vertical/portrait (phones)
    bool bVerticalOutput = false;
};

struct FBeatsyncProcessingResult
{
    bool bSuccess = false;
    bool bAudioMuxFailed = false;  // True if video processing succeeded but audio muxing failed
    FString ErrorMessage;
    int32 BeatCount = 0;
    double BPM = 0.0;
    TArray<double> BeatTimes;
};

// Delegates for progress and completion callbacks.
// NOTE: these callbacks are marshaled to ENamedThreads::GameThread by the implementation
// and will be invoked on the Game Thread. Callers may safely touch UI/Engine objects
// from the callback.
DECLARE_DELEGATE_TwoParams(FOnBeatsyncProcessingProgress, float /*Progress*/, const FString& /*Status*/);
DECLARE_DELEGATE_OneParam(FOnBeatsyncProcessingComplete, const FBeatsyncProcessingResult& /*Result*/);

class FBeatsyncProcessingTask : public FNonAbandonableTask
{
    friend class FAsyncTask<FBeatsyncProcessingTask>;

public:
    FBeatsyncProcessingTask(const FBeatsyncProcessingParams& InParams,
                            FOnBeatsyncProcessingProgress InProgressDelegate,
                            FOnBeatsyncProcessingComplete InCompleteDelegate);
    ~FBeatsyncProcessingTask();

    void DoWork();
    void RequestCancel() {
        bCancelRequested.AtomicSet(true);
        BackendCancelFlag.store(1, std::memory_order_release);  // Signal backend to cancel
        if (SharedCancelFlag.IsValid()) {
            SharedCancelFlag->AtomicSet(true);
        }
    }
    bool IsCancelled() const { return bCancelRequested; }

    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FBeatsyncProcessingTask, STATGROUP_ThreadPoolAsyncTasks);
    }

private:
    FBeatsyncProcessingParams Params;
    FOnBeatsyncProcessingProgress OnProgress;
    FOnBeatsyncProcessingComplete OnComplete;
    FThreadSafeBool bCancelRequested;
    TSharedPtr<FThreadSafeBool> SharedCancelFlag;
    TSharedPtr<FThreadSafeBool> ProgressGuard;  // Guard for progress callbacks
    std::atomic<int> BackendCancelFlag{0};  // Cancel flag for backend C API (must be int, not bool)
    FThreadSafeBool bWorkCompleted;  // Set when DoWork finishes, used for destructor synchronization
    FEvent* WorkCompletedEvent = nullptr;  // Signaled when DoWork finishes, destructor waits on this
    void* Writer = nullptr;
    FCriticalSection WriterMutex;
    FString TempVideoPath;
    FString TempEffectsPath;

    void ReportProgress(float Progress, const FString& Status);
    bool HasAnyEffectsEnabled() const;

    // Helper to signal completion and cleanup - call before every return in DoWork
    void SignalWorkComplete()
    {
        bWorkCompleted.AtomicSet(true);
        if (WorkCompletedEvent)
        {
            WorkCompletedEvent->Trigger();
        }
    }
};
