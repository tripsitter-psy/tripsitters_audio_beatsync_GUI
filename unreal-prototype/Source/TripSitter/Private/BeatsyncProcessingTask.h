#pragma once

#include "CoreMinimal.h"
#include "Async/AsyncWork.h"
#include "HAL/ThreadSafeBool.h"
#include "BeatsyncLoader.h"

struct FBeatsyncProcessingParams
{
    FString AudioPath;
    FString VideoPath;
    TArray<FString> VideoPaths;
    FString OutputPath;
    bool bIsMultiClip = false;
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
