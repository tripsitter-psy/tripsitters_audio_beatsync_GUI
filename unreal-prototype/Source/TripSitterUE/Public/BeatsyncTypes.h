#pragma once

#include "CoreMinimal.h"
#include "BeatsyncTypes.generated.h"

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
    FString TransitionType = TEXT("fade");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Transitions", meta = (ClampMin = "0.0", ClampMax = "2.0"))
    float TransitionDuration = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Color")
    bool bEnableColorGrade = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Color")
    FString ColorPreset = TEXT("warm");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Vignette")
    bool bEnableVignette = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Vignette", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float VignetteStrength = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects")
    bool bEnableBeatFlash = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float FlashIntensity = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects")
    bool bEnableBeatZoom = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float ZoomIntensity = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|BeatEffects", meta = (ClampMin = "1", ClampMax = "16"))
    int32 EffectBeatDivisor = 1;
};
