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

    /** Beat divisor for effects (must be >= 1 to prevent division by zero).
     *  Value of 1 = every beat, 2 = every other beat, etc.
     *  Use GetEffectBeatDivisor() to get a validated value. */
    int32 EffectBeatDivisor = 1;

    /** Returns EffectBeatDivisor clamped to >= 1 to prevent division by zero. */
    int32 GetEffectBeatDivisor() const { return FMath::Max(1, EffectBeatDivisor); }

    double EffectStartTime = 0.0;   // Start time for effects (0 = from beginning)
    double EffectEndTime = -1.0;    // End time for effects (-1 = to end)
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
};

// AI analysis result
struct FAIResult
{
    TArray<double> Beats;
    TArray<double> Downbeats;
    double BPM = 0.0;
    double Duration = 0.0;
};
