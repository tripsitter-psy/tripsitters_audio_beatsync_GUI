// TripSitter - render time estimator
#include "RenderEtaEstimator.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformTime.h"
#include "HAL/FileManager.h"

// Units per second on an unknown machine. Deliberately conservative (slow):
// an ETA that shrinks as the real rate is measured reads better than one that
// keeps growing. Units per stage:
//   upscale_2x / upscale_4x : source megapixel-frames run through the model
//   normalize               : source seconds x output megapixels (NVENC/x264 re-encode)
//   cut                     : backend cost units (1 = one stream-copied clip)
//   effects                 : output seconds x output megapixels
//   mux                     : output seconds (stream copy)
double FRenderEtaEstimator::DefaultRate(const FString& RateKey)
{
    if (RateKey == TEXT("upscale_2x")) return 1.5;
    if (RateKey == TEXT("upscale_4x")) return 0.4;
    if (RateKey == TEXT("normalize"))  return 6.0;
    if (RateKey == TEXT("cut"))        return 4.0;
    if (RateKey == TEXT("effects"))    return 8.0;
    if (RateKey == TEXT("mux"))        return 150.0;
    return 1.0;
}

FString FRenderEtaEstimator::CalibrationPath() const
{
    return FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("RenderCalibration.txt"));
}

void FRenderEtaEstimator::LoadCalibration()
{
    FScopeLock Guard(&Lock);
    LearnedRates.Empty();
    TArray<FString> Lines;
    if (FFileHelper::LoadFileToStringArray(Lines, *CalibrationPath()))
    {
        for (const FString& Line : Lines)
        {
            FString Key, Value;
            if (Line.Split(TEXT("="), &Key, &Value))
            {
                double Rate = FCString::Atod(*Value.TrimStartAndEnd());
                if (Rate > 0.0)
                {
                    LearnedRates.Add(Key.TrimStartAndEnd(), Rate);
                }
            }
        }
    }
}

void FRenderEtaEstimator::SaveCalibration()
{
    // Called with Lock held.
    TArray<FString> Lines;
    Lines.Add(TEXT("# TripSitter render throughput (units/s), learned from previous renders on this machine."));
    Lines.Add(TEXT("# Delete this file to reset the ETA calibration."));
    for (const auto& Pair : LearnedRates)
    {
        Lines.Add(FString::Printf(TEXT("%s=%.6f"), *Pair.Key, Pair.Value));
    }
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(CalibrationPath()), true);
    FFileHelper::SaveStringArrayToFile(Lines, *CalibrationPath());
}

void FRenderEtaEstimator::SetPlan(const TArray<FRenderEtaStage>& InStages)
{
    FScopeLock Guard(&Lock);
    Stages.Empty();
    for (const FRenderEtaStage& Def : InStages)
    {
        if (Def.Work <= 0.0) continue;
        FStageState S;
        S.Def = Def;
        if (S.Def.RateKey.IsEmpty()) S.Def.RateKey = S.Def.Key;
        const double* Learned = LearnedRates.Find(S.Def.RateKey);
        S.Rate = (Learned && *Learned > 0.0) ? *Learned : DefaultRate(S.Def.RateKey);
        Stages.Add(S);
    }
    CurrentStage = -1;
    bStarted = false;
    bFinished = false;
}

void FRenderEtaEstimator::Start()
{
    FScopeLock Guard(&Lock);
    PlanStartTime = FPlatformTime::Seconds();
    bStarted = true;
    bFinished = false;
    if (Stages.Num() > 0)
    {
        CurrentStage = 0;
        Stages[0].StartTime = PlanStartTime;
    }
}

void FRenderEtaEstimator::OnStageProgress(const FString& StageKey, double Progress)
{
    const double Now = FPlatformTime::Seconds();
    FScopeLock Guard(&Lock);
    if (!bStarted || bFinished) return;

    int32 Index = Stages.IndexOfByPredicate([&](const FStageState& S) { return S.Def.Key == StageKey; });
    if (Index == INDEX_NONE)
    {
        // Not in the plan: ignore (the plan and the backend disagree on the pipeline).
        return;
    }

    // Any earlier stage that is still open is finished now (stages run in order).
    for (int32 i = 0; i < Index; ++i)
    {
        FStageState& Prev = Stages[i];
        if (Prev.EndTime < 0.0)
        {
            Prev.Progress = 1.0;
            if (Prev.StartTime < 0.0) Prev.StartTime = Now;
            Prev.EndTime = Now;
        }
    }

    FStageState& S = Stages[Index];
    if (Index != CurrentStage)
    {
        CurrentStage = Index;
    }
    if (S.StartTime < 0.0) S.StartTime = Now;

    Progress = FMath::Clamp(Progress, 0.0, 1.0);
    if (Progress > S.Progress)
    {
        // Instantaneous progress rate since the previous event, blended into a
        // smoothed rate. The whole-stage average (Progress / elapsed) anchors the
        // blend so a single fast or slow clip does not swing the ETA.
        const double Elapsed = Now - S.StartTime;
        if (Elapsed > 0.5 && Progress >= 0.01)
        {
            const double Average = Progress / Elapsed;
            double Instant = Average;
            if (S.LastProgressTime > 0.0 && Now - S.LastProgressTime > 0.05)
            {
                Instant = (Progress - S.LastProgress) / (Now - S.LastProgressTime);
            }
            const double Blend = 0.7 * Average + 0.3 * Instant;
            S.MeasuredRate = (S.MeasuredRate > 0.0) ? (0.6 * S.MeasuredRate + 0.4 * Blend) : Blend;
        }
        S.LastProgress = Progress;
        S.LastProgressTime = Now;
        S.Progress = Progress;
    }
    if (Progress >= 1.0 && S.EndTime < 0.0)
    {
        S.EndTime = Now;
    }
}

void FRenderEtaEstimator::Finish(bool bSuccess)
{
    const double Now = FPlatformTime::Seconds();
    FScopeLock Guard(&Lock);
    if (!bStarted || bFinished) return;
    bFinished = true;
    if (!bSuccess) return;

    // Learn units/s from every stage that actually ran for a meaningful time.
    bool bChanged = false;
    for (FStageState& S : Stages)
    {
        if (S.StartTime < 0.0) continue;
        if (S.EndTime < 0.0) S.EndTime = Now;
        const double Duration = S.EndTime - S.StartTime;
        if (Duration < 1.0 || S.Progress < 0.5) continue;
        const double Observed = (S.Def.Work * S.Progress) / Duration;
        if (!FMath::IsFinite(Observed) || Observed <= 0.0) continue;
        double* Existing = LearnedRates.Find(S.Def.RateKey);
        // Weighted toward the new observation; a 50/50 blend converges within a
        // couple of renders while still smoothing out an unusual clip mix.
        const double Updated = Existing ? (0.5 * (*Existing) + 0.5 * Observed) : Observed;
        LearnedRates.Add(S.Def.RateKey, Updated);
        bChanged = true;
    }
    if (bChanged)
    {
        SaveCalibration();
    }
}

double FRenderEtaEstimator::EstimateStageTotalSeconds(const FStageState& S, double Now) const
{
    const double Apriori = (S.Rate > 0.0) ? S.Def.Work / S.Rate : 0.0;
    if (S.MeasuredRate > 0.0 && S.Progress >= 0.02)
    {
        const double Measured = 1.0 / S.MeasuredRate;
        // Trust the measurement progressively: fully after ~15% of the stage.
        const double Trust = FMath::Clamp((S.Progress - 0.02) / 0.13, 0.0, 1.0);
        return Trust * Measured + (1.0 - Trust) * Apriori;
    }
    return Apriori;
}

FRenderEtaEstimator::FStatus FRenderEtaEstimator::GetStatus() const
{
    const double Now = FPlatformTime::Seconds();
    FScopeLock Guard(&Lock);

    FStatus Status;
    Status.StageLabel.Empty();
    Status.EtaText.Empty();
    Status.OverallProgress = 0.0;
    Status.StageProgress = 0.0;
    Status.bMeasured = false;
    Status.RemainingSeconds = -1.0;

    if (!bStarted || bFinished || Stages.Num() == 0)
    {
        return Status;
    }

    double TotalSeconds = 0.0;   // estimated total of the whole plan
    double DoneSeconds = 0.0;    // estimated seconds' worth already completed
    double Remaining = 0.0;
    for (int32 i = 0; i < Stages.Num(); ++i)
    {
        const FStageState& S = Stages[i];
        const double StageTotal = EstimateStageTotalSeconds(S, Now);
        TotalSeconds += StageTotal;
        if (S.EndTime >= 0.0 || S.Progress >= 1.0)
        {
            DoneSeconds += StageTotal;
            continue;
        }
        if (i == CurrentStage && S.StartTime >= 0.0)
        {
            const double Elapsed = Now - S.StartTime;
            double StageRemaining = StageTotal - Elapsed;
            // Never claim the stage is finished while it is still running: keep
            // at least a few seconds on the clock when the estimate ran out.
            if (StageRemaining < 5.0) StageRemaining = 5.0;
            Remaining += StageRemaining;
            DoneSeconds += FMath::Min(Elapsed, StageTotal);
            Status.StageProgress = S.Progress;
            Status.bMeasured = S.MeasuredRate > 0.0 && S.Progress >= 0.02;
        }
        else
        {
            Remaining += StageTotal;
        }
    }

    if (CurrentStage >= 0 && CurrentStage < Stages.Num())
    {
        Status.StageLabel = Stages[CurrentStage].Def.Label;
    }
    if (TotalSeconds > 0.0)
    {
        Status.OverallProgress = FMath::Clamp(DoneSeconds / TotalSeconds, 0.0, 0.995);
    }
    Status.RemainingSeconds = Remaining;
    Status.EtaText = FormatRemaining(Remaining);
    return Status;
}

FString FRenderEtaEstimator::FormatRemaining(double Seconds)
{
    if (Seconds < 0.0) return FString();
    if (Seconds < 45.0) return TEXT("under a minute left");
    if (Seconds < 600.0)
    {
        // Round to the nearest 15 s below ten minutes so the countdown feels live.
        const int32 Rounded = FMath::RoundToInt(Seconds / 15.0) * 15;
        const int32 Min = Rounded / 60;
        const int32 Sec = Rounded % 60;
        if (Min == 0) return FString::Printf(TEXT("about %d s left"), Sec);
        if (Sec == 0) return FString::Printf(TEXT("about %d min left"), Min);
        return FString::Printf(TEXT("about %d min %02d s left"), Min, Sec);
    }
    if (Seconds < 3600.0)
    {
        return FString::Printf(TEXT("about %d min left"), FMath::RoundToInt(Seconds / 60.0));
    }
    const int32 TotalMin = FMath::RoundToInt(Seconds / 60.0);
    const int32 Hours = TotalMin / 60;
    const int32 Min = TotalMin % 60;
    if (Min == 0) return FString::Printf(TEXT("about %d h left"), Hours);
    return FString::Printf(TEXT("about %d h %d min left"), Hours, Min);
}
