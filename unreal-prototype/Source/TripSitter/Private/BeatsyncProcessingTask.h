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
    // Must be >= 0
    int32 BeatRate = 0;
    double AudioStart = 0.0;
    double AudioEnd = -1.0;
    FEffectsConfig EffectsConfig;
    // Analysis mode: determines which beat detection method to use
    EAnalysisModeParam AnalysisMode = EAnalysisModeParam::AIBeat;
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
    void* Writer = nullptr;

    // Guard used by progress callback to ensure we don't invoke callbacks on destroyed task
    TSharedPtr<FThreadSafeBool> ProgressGuard;

    void ReportProgress(float Progress, const FString& Status);
    bool HasAnyEffectsEnabled() const;
};
