#pragma once

#include "CoreMinimal.h"

struct FBeatGrid
{
    TArray<double> Beats;
    double BPM = 0.0;
    double Duration = 0.0;
};

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

    double EffectStartTime = 0.0;   // Start time for effects (0 = from beginning)
    double EffectEndTime = -1.0;    // End time for effects (-1 = to end)
};
