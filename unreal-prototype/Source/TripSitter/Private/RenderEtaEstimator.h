// TripSitter - render time estimator
#pragma once

#include "CoreMinimal.h"

/**
 * One pipeline stage in the ETA plan.
 *
 * Work is an a-priori amount of work in stage-specific units (see
 * FRenderEtaEstimator::DefaultRate for what a unit is per stage). It only has
 * to be proportional to the real cost: the estimator measures the actual rate
 * (units per second) while the stage runs and remembers it between runs.
 */
struct FRenderEtaStage
{
    FString Key;      // backend stage name: "upscale", "normalize", "cut", "effects", "mux"
    FString RateKey;  // calibration key, e.g. "upscale_2x" (defaults to Key)
    FString Label;    // human readable, e.g. "Upscaling clips"
    double Work = 0.0;
};

/**
 * Estimates the time remaining for a video render.
 *
 * How it works:
 *  - Before the render, the processing task builds a plan: the ordered stages
 *    that will run and how much work each one is (frames to upscale, seconds to
 *    encode, clips to cut...). Divided by a per-stage rate this gives a first
 *    estimate. Rates come from a small calibration file written after every
 *    successful render on this machine, so the estimate before the first progress
 *    event is already based on how fast this GPU/CPU was last time.
 *  - While a stage runs, the backend reports that stage's own 0..1 progress.
 *    The estimator measures the elapsed time per unit of progress (smoothed),
 *    which replaces the a-priori guess for the current stage. Future stages
 *    keep using the calibrated rates.
 *  - GetStatus() is meant to be polled every second: the remaining time counts
 *    down between progress events instead of only changing when one arrives.
 *
 * Thread-safe: progress arrives from the backend worker thread, status is read
 * on the game thread.
 */
class FRenderEtaEstimator
{
public:
    /** Default units-per-second for a stage on an unknown machine. */
    static double DefaultRate(const FString& RateKey);

    /** Load the calibration file (per-machine learned rates). Safe to call without one. */
    void LoadCalibration();

    /** Install the plan. Must be called before Start(). */
    void SetPlan(const TArray<FRenderEtaStage>& InStages);

    /** Mark the start of the planned stages (after beat analysis). */
    void Start();

    /** Backend stage progress (any thread). */
    void OnStageProgress(const FString& StageKey, double Progress);

    /** Called once the render finished; on success the measured rates are persisted. */
    void Finish(bool bSuccess);

    struct FStatus
    {
        FString StageLabel;      // e.g. "Cutting to the beat"
        FString EtaText;         // e.g. "about 4 min left" (empty if unknown)
        double OverallProgress;  // 0..1 across all planned stages, work weighted
        double StageProgress;    // 0..1 of the current stage
        bool bMeasured;          // false while the ETA is still purely a-priori
        double RemainingSeconds; // raw estimate, -1 if unknown
    };

    /** Current status, thread-safe. */
    FStatus GetStatus() const;

    /** Formats seconds as "about 1 h 20 min left" / "about 40 s left" / "under a minute left". */
    static FString FormatRemaining(double Seconds);

private:
    struct FStageState
    {
        FRenderEtaStage Def;
        double Rate = 0.0;          // calibrated units/s for the a-priori estimate
        double Progress = 0.0;      // last reported 0..1
        double StartTime = -1.0;    // seconds (FPlatformTime) when the stage first reported
        double EndTime = -1.0;
        double MeasuredRate = 0.0;  // smoothed progress-per-second while running
        double LastProgressTime = -1.0;
        double LastProgress = 0.0;
    };

    FString CalibrationPath() const;
    void SaveCalibration();
    double EstimateStageTotalSeconds(const FStageState& S, double Now) const;

    mutable FCriticalSection Lock;
    TArray<FStageState> Stages;
    TMap<FString, double> LearnedRates;
    int32 CurrentStage = -1;
    double PlanStartTime = -1.0;
    bool bStarted = false;
    bool bFinished = false;
};
