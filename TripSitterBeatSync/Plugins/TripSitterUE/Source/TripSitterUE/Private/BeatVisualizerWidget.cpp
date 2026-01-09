#include "BeatVisualizerWidget.h"
#include "Rendering/DrawElements.h"
#include "Styling/CoreStyle.h"

void UBeatVisualizerWidget::SetBeatData(const FBeatData& InBeatData)
{
	BeatData = InBeatData;

	// Auto-adjust visible range to show entire track
	if (BeatData.Duration > 0.0f)
	{
		VisibleStartTime = 0.0f;
		VisibleEndTime = BeatData.Duration;
	}
}

void UBeatVisualizerWidget::SetPlaybackPosition(float TimeInSeconds)
{
	CurrentPlaybackTime = FMath::Clamp(TimeInSeconds, 0.0f, BeatData.Duration);
}

void UBeatVisualizerWidget::SetVisibleRange(float StartTime, float EndTime)
{
	VisibleStartTime = FMath::Max(0.0f, StartTime);
	VisibleEndTime = FMath::Max(VisibleStartTime + 0.1f, EndTime);
}

int32 UBeatVisualizerWidget::GetBeatIndexAtTime(float TimeInSeconds) const
{
	if (BeatData.BeatTimestamps.Num() == 0)
	{
		return INDEX_NONE;
	}

	// Find the closest beat
	int32 ClosestIndex = INDEX_NONE;
	float ClosestDistance = TNumericLimits<float>::Max();

	for (int32 i = 0; i < BeatData.BeatTimestamps.Num(); ++i)
	{
		float Distance = FMath::Abs(BeatData.BeatTimestamps[i] - TimeInSeconds);
		if (Distance < ClosestDistance)
		{
			ClosestDistance = Distance;
			ClosestIndex = i;
		}
	}

	// Only return if within a reasonable threshold (0.1 seconds)
	if (ClosestDistance < 0.1f)
	{
		return ClosestIndex;
	}

	return INDEX_NONE;
}

float UBeatVisualizerWidget::GetBeatTimeAtIndex(int32 Index) const
{
	if (Index >= 0 && Index < BeatData.BeatTimestamps.Num())
	{
		return BeatData.BeatTimestamps[Index];
	}
	return -1.0f;
}

float UBeatVisualizerWidget::TimeToPixel(float Time, float Width) const
{
	float Range = VisibleEndTime - VisibleStartTime;
	if (Range <= 0.0f) return 0.0f;

	float NormalizedTime = (Time - VisibleStartTime) / Range;
	return NormalizedTime * Width;
}

float UBeatVisualizerWidget::PixelToTime(float Pixel, float Width) const
{
	if (Width <= 0.0f) return VisibleStartTime;

	float NormalizedPixel = Pixel / Width;
	float Range = VisibleEndTime - VisibleStartTime;
	return VisibleStartTime + (NormalizedPixel * Range);
}

int32 UBeatVisualizerWidget::NativePaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry, const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements, int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const
{
	const FVector2D LocalSize = AllottedGeometry.GetLocalSize();
	const float Width = LocalSize.X;
	const float Height = LocalSize.Y;

	// Draw background
	FSlateDrawElement::MakeBox(
		OutDrawElements,
		LayerId,
		AllottedGeometry.ToPaintGeometry(),
		FCoreStyle::Get().GetBrush("WhiteBrush"),
		ESlateDrawEffect::None,
		BackgroundColor
	);

	LayerId++;

	// Draw beat lines
	for (int32 i = 0; i < BeatData.BeatTimestamps.Num(); ++i)
	{
		float BeatTime = BeatData.BeatTimestamps[i];

		// Skip beats outside visible range
		if (BeatTime < VisibleStartTime || BeatTime > VisibleEndTime)
		{
			continue;
		}

		float X = TimeToPixel(BeatTime, Width);
		bool bIsDownbeat = (i % BeatsPerMeasure) == 0;

		FLinearColor LineColor = bIsDownbeat ? DownbeatColor : BeatColor;
		float LineWidth = bIsDownbeat ? DownbeatLineWidth : BeatLineWidth;

		TArray<FVector2D> LinePoints;
		LinePoints.Add(FVector2D(X, 0.0f));
		LinePoints.Add(FVector2D(X, Height));

		FSlateDrawElement::MakeLines(
			OutDrawElements,
			LayerId,
			AllottedGeometry.ToPaintGeometry(),
			LinePoints,
			ESlateDrawEffect::None,
			LineColor,
			true,
			LineWidth
		);
	}

	LayerId++;

	// Draw playhead
	if (CurrentPlaybackTime >= VisibleStartTime && CurrentPlaybackTime <= VisibleEndTime)
	{
		float PlayheadX = TimeToPixel(CurrentPlaybackTime, Width);

		TArray<FVector2D> PlayheadPoints;
		PlayheadPoints.Add(FVector2D(PlayheadX, 0.0f));
		PlayheadPoints.Add(FVector2D(PlayheadX, Height));

		FSlateDrawElement::MakeLines(
			OutDrawElements,
			LayerId,
			AllottedGeometry.ToPaintGeometry(),
			PlayheadPoints,
			ESlateDrawEffect::None,
			PlayheadColor,
			true,
			3.0f
		);
	}

	return LayerId;
}

FReply UBeatVisualizerWidget::NativeOnMouseButtonDown(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent)
{
	if (InMouseEvent.GetEffectingButton() == EKeys::LeftMouseButton)
	{
		FVector2D LocalPos = InGeometry.AbsoluteToLocal(InMouseEvent.GetScreenSpacePosition());
		float Width = InGeometry.GetLocalSize().X;

		float ClickedTime = PixelToTime(LocalPos.X, Width);
		int32 ClickedBeat = GetBeatIndexAtTime(ClickedTime);

		if (ClickedBeat != INDEX_NONE)
		{
			OnBeatClicked.Broadcast(ClickedBeat);
		}
		else
		{
			OnTimelineClicked.Broadcast(ClickedTime);
		}

		return FReply::Handled();
	}

	return Super::NativeOnMouseButtonDown(InGeometry, InMouseEvent);
}
