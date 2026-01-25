#pragma once

#include "CoreMinimal.h"
#include "BeatsyncTypes.generated.h"

/**
 * Video transition types for beat-synced effects
 */
UENUM(BlueprintType)
enum class ETransitionType : uint8
{
    Fade UMETA(DisplayName = "Fade"),
    Wipe UMETA(DisplayName = "Wipe"),
    Dissolve UMETA(DisplayName = "Dissolve"),
    Zoom UMETA(DisplayName = "Zoom")
};

/**
 * Color grading presets for video processing
 */
UENUM(BlueprintType)
enum class EColorPreset : uint8
{
    Warm UMETA(DisplayName = "Warm"),
    Cool UMETA(DisplayName = "Cool"),
    Vintage UMETA(DisplayName = "Vintage"),
    Vibrant UMETA(DisplayName = "Vibrant")
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

USTRUCT(BlueprintType)
struct FBeatGrid
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
    TArray<double> Beats;

    UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
    double BPM = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
    double Duration = 0.0;
};

USTRUCT(BlueprintType)
struct FEffectsConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Transitions")
    bool bEnableTransitions = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Transitions")
    ETransitionType TransitionType = ETransitionType::Fade;

    // Backend uses double, but Blueprint only supports float; default matches backend (0.3)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Transitions", meta = (ClampMin = "0.0", ClampMax = "2.0"))
    float TransitionDuration = 0.3f; // Backend: double, default 0.3

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Color")
    bool bEnableColorGrade = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Color")
    EColorPreset ColorPreset = EColorPreset::Warm;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Vignette")
    bool bEnableVignette = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Vignette", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float VignetteStrength = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects")
    bool bEnableBeatFlash = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float FlashIntensity = 0.3f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects")
    bool bEnableBeatZoom = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float ZoomIntensity = 0.04f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects", meta = (ClampMin = "1", ClampMax = "16"))
    int32 EffectBeatDivisor = 1;
};
