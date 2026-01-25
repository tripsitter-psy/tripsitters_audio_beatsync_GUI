#pragma once

#include "CoreMinimal.h"

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

struct FBeatGrid
{
    TArray<double> Beats;
    double BPM = 0.0;
    double Duration = 0.0;
};

struct FEffectsConfig
{
    bool bEnableTransitions = false;
    ETransitionType TransitionType = ETransitionType::Fade;
    float TransitionDuration = 0.5f;

    bool bEnableColorGrade = false;
    EColorPreset ColorPreset = EColorPreset::Warm;

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
