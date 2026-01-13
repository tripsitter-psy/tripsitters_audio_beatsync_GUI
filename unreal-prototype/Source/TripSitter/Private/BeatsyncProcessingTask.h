#pragma once

#include "CoreMinimal.h"
#include "Async/AsyncWork.h"
#include "HAL/ThreadSafeBool.h"
#include "BeatsyncLoader.h"

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
};

struct FBeatsyncProcessingResult
{
    bool bSuccess = false;
    FString ErrorMessage;
    int32 BeatCount = 0;
    double BPM = 0.0;
    TArray<double> BeatTimes;
};

// Delegates for progress and completion callbacks (called on worker thread - marshal to game thread if needed)
DECLARE_DELEGATE_TwoParams(FOnBeatsyncProcessingProgress, float /*Progress*/, const FString& /*Status*/);
DECLARE_DELEGATE_OneParam(FOnBeatsyncProcessingComplete, const FBeatsyncProcessingResult& /*Result*/);

class FBeatsyncProcessingTask : public FNonAbandonableTask
{
    friend class FAsyncTask<FBeatsyncProcessingTask>;

public:
    FBeatsyncProcessingTask(const FBeatsyncProcessingParams& InParams,
                            FOnBeatsyncProcessingProgress InProgressDelegate,
                            FOnBeatsyncProcessingComplete InCompleteDelegate);

    void DoWork();
    void RequestCancel() { bCancelRequested.AtomicSet(true); }
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

    void ReportProgress(float Progress, const FString& Status);
    bool HasAnyEffectsEnabled() const;
};
