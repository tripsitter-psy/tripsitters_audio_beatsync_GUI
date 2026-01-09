#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "BeatsyncSubsystem.h"
#include "BeatVisualizerWidget.generated.h"

/**
 * Widget for visualizing beat timestamps on a timeline
 * Can be used standalone or embedded in the main BeatSync widget
 */
UCLASS()
class TRIPSITTERUE_API UBeatVisualizerWidget : public UUserWidget
{
	GENERATED_BODY()

public:
	/** Set the beat data to visualize */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|Visualizer")
	void SetBeatData(const FBeatData& InBeatData);

	/** Set the current playback position */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|Visualizer")
	void SetPlaybackPosition(float TimeInSeconds);

	/** Set the visible time range */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|Visualizer")
	void SetVisibleRange(float StartTime, float EndTime);

	/** Get the beat index at a specific time */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|Visualizer")
	int32 GetBeatIndexAtTime(float TimeInSeconds) const;

	/** Get beat time at index */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|Visualizer")
	float GetBeatTimeAtIndex(int32 Index) const;

	// Visual Settings
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Visualizer|Appearance")
	FLinearColor BeatColor = FLinearColor(0.0f, 0.85f, 1.0f, 1.0f); // Cyan

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Visualizer|Appearance")
	FLinearColor DownbeatColor = FLinearColor(1.0f, 0.0f, 0.5f, 1.0f); // Pink

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Visualizer|Appearance")
	FLinearColor PlayheadColor = FLinearColor(1.0f, 1.0f, 1.0f, 1.0f); // White

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Visualizer|Appearance")
	FLinearColor BackgroundColor = FLinearColor(0.04f, 0.04f, 0.1f, 0.9f); // Dark blue

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Visualizer|Appearance")
	float BeatLineWidth = 2.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Visualizer|Appearance")
	float DownbeatLineWidth = 4.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Visualizer|Appearance")
	int32 BeatsPerMeasure = 4;

	// State
	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|Visualizer|State")
	FBeatData BeatData;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|Visualizer|State")
	float CurrentPlaybackTime = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|Visualizer|State")
	float VisibleStartTime = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|Visualizer|State")
	float VisibleEndTime = 10.0f;

	// Events
	DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnBeatClicked, int32, BeatIndex);

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Visualizer|Events")
	FOnBeatClicked OnBeatClicked;

	DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTimelineClicked, float, TimeInSeconds);

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Visualizer|Events")
	FOnTimelineClicked OnTimelineClicked;

protected:
	virtual int32 NativePaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry, const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements, int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const override;
	virtual FReply NativeOnMouseButtonDown(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent) override;

	float TimeToPixel(float Time, float Width) const;
	float PixelToTime(float Pixel, float Width) const;
};
