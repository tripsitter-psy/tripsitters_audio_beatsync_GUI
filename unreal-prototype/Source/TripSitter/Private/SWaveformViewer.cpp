#include "SWaveformViewer.h"
#include "Rendering/DrawElements.h"
#include "Styling/CoreStyle.h"
#include "Framework/Application/SlateApplication.h"
#include "Framework/MultiBox/MultiBoxBuilder.h"

void SWaveformViewer::Construct(const FArguments& InArgs)
{
	WaveformColor = InArgs._WaveformColor;
	BeatMarkerColor = InArgs._BeatMarkerColor;
	SelectionColor = InArgs._SelectionColor;
	HandleColor = InArgs._HandleColor;
	EffectRegionColor = InArgs._EffectRegionColor;
	OnSelectionChanged = InArgs._OnSelectionChanged;
	OnEffectRegionChanged = InArgs._OnEffectRegionChanged;
	OnBeatTimesChanged = InArgs._OnBeatTimesChanged;

	// Initialize stem beat data with colors and names
	StemBeats[0].Type = EStemType::Kick;
	StemBeats[0].Name = TEXT("Kick");
	StemBeats[0].Color = FLinearColor(0.1f, 0.1f, 0.1f, 0.9f);  // Black/dark gray
	StemBeats[0].bEnabled = true;

	StemBeats[1].Type = EStemType::Snare;
	StemBeats[1].Name = TEXT("Snare");
	StemBeats[1].Color = FLinearColor(0.2f, 0.4f, 1.0f, 0.9f);  // Blue
	StemBeats[1].bEnabled = true;

	StemBeats[2].Type = EStemType::HiHat;
	StemBeats[2].Name = TEXT("Hi-Hat");
	StemBeats[2].Color = FLinearColor(1.0f, 0.9f, 0.2f, 0.9f);  // Yellow
	StemBeats[2].bEnabled = true;

	StemBeats[3].Type = EStemType::Synth;
	StemBeats[3].Name = TEXT("Synth");
	StemBeats[3].Color = FLinearColor(1.0f, 0.2f, 0.2f, 0.9f);  // Red
	StemBeats[3].bEnabled = true;
}

void SWaveformViewer::SetWaveformData(const TArray<float>& InPeaks, double InDuration)
{
	WaveformPeaks = InPeaks;
	// Clear band data when using single-color mode
	BassPeaks.Empty();
	MidPeaks.Empty();
	HighPeaks.Empty();
	Duration = InDuration;
	// Reset selection to full track when new data loaded
	SelectionStart = 0.0;
	SelectionEnd = Duration;
	Invalidate(EInvalidateWidget::Paint);
}

void SWaveformViewer::SetWaveformBands(const TArray<float>& InBassPeaks, const TArray<float>& InMidPeaks,
                                        const TArray<float>& InHighPeaks, double InDuration)
{
	BassPeaks = InBassPeaks;
	MidPeaks = InMidPeaks;
	HighPeaks = InHighPeaks;
	// Clear single-color data when using band mode
	WaveformPeaks.Empty();
	Duration = InDuration;
	// Reset selection to full track when new data loaded
	SelectionStart = 0.0;
	SelectionEnd = Duration;
	Invalidate(EInvalidateWidget::Paint);
}

void SWaveformViewer::SetBeatTimes(const TArray<double>& InBeatTimes)
{
	BeatTimes = InBeatTimes;
	Invalidate(EInvalidateWidget::Paint);
}

void SWaveformViewer::AddBeatAtTime(double Time)
{
	if (Time < 0 || Time > Duration) return;

	// Insert in sorted order
	int32 InsertIndex = 0;
	for (int32 i = 0; i < BeatTimes.Num(); ++i)
	{
		if (BeatTimes[i] > Time)
		{
			InsertIndex = i;
			break;
		}
		InsertIndex = i + 1;
	}

	BeatTimes.Insert(Time, InsertIndex);

	// Notify listeners
	if (OnBeatTimesChanged.IsBound())
	{
		OnBeatTimesChanged.Execute(BeatTimes);
	}

	Invalidate(EInvalidateWidget::Paint);
}

void SWaveformViewer::RemoveBeatAtTime(double Time, double Tolerance)
{
	int32 ClosestIndex = -1;
	double ClosestDist = Tolerance;

	for (int32 i = 0; i < BeatTimes.Num(); ++i)
	{
		double Dist = FMath::Abs(BeatTimes[i] - Time);
		if (Dist < ClosestDist)
		{
			ClosestDist = Dist;
			ClosestIndex = i;
		}
	}

	if (ClosestIndex >= 0)
	{
		RemoveBeatAtIndex(ClosestIndex);
	}
}

void SWaveformViewer::RemoveBeatAtIndex(int32 Index)
{
	if (BeatTimes.IsValidIndex(Index))
	{
		BeatTimes.RemoveAt(Index);

		// Notify listeners
		if (OnBeatTimesChanged.IsBound())
		{
			OnBeatTimesChanged.Execute(BeatTimes);
		}

		Invalidate(EInvalidateWidget::Paint);
	}
}

int32 SWaveformViewer::FindBeatNearPixel(float X, float Width, float TolerancePixels) const
{
	if (Duration <= 0 || BeatTimes.Num() == 0) return -1;

	double ClickTime = PixelToTime(X, Width);
	double TimeTolerance = (TolerancePixels / Width) * (Duration / ZoomLevel);

	int32 ClosestIndex = -1;
	double ClosestDist = TimeTolerance;

	for (int32 i = 0; i < BeatTimes.Num(); ++i)
	{
		double Dist = FMath::Abs(BeatTimes[i] - ClickTime);
		if (Dist < ClosestDist)
		{
			ClosestDist = Dist;
			ClosestIndex = i;
		}
	}

	return ClosestIndex;
}

// Stem beat management
void SWaveformViewer::SetStemBeatTimes(EStemType Stem, const TArray<double>& InBeatTimes)
{
	int32 Index = static_cast<int32>(Stem);
	if (Index >= 0 && Index < 4)
	{
		StemBeats[Index].BeatTimes = InBeatTimes;
		Invalidate(EInvalidateWidget::Paint);
	}
}

void SWaveformViewer::ClearStemBeats(EStemType Stem)
{
	int32 Index = static_cast<int32>(Stem);
	if (Index >= 0 && Index < 4)
	{
		StemBeats[Index].BeatTimes.Empty();
		Invalidate(EInvalidateWidget::Paint);
	}
}

void SWaveformViewer::ClearAllStemBeats()
{
	for (int32 i = 0; i < 4; ++i)
	{
		StemBeats[i].BeatTimes.Empty();
	}
	Invalidate(EInvalidateWidget::Paint);
}

const TArray<double>& SWaveformViewer::GetStemBeatTimes(EStemType Stem) const
{
	int32 Index = static_cast<int32>(Stem);
	if (Index >= 0 && Index < 4)
	{
		return StemBeats[Index].BeatTimes;
	}
	static TArray<double> Empty;
	return Empty;
}

void SWaveformViewer::SetStemEnabled(EStemType Stem, bool bEnabled)
{
	int32 Index = static_cast<int32>(Stem);
	if (Index >= 0 && Index < 4)
	{
		StemBeats[Index].bEnabled = bEnabled;
		Invalidate(EInvalidateWidget::Paint);
	}
}

bool SWaveformViewer::IsStemEnabled(EStemType Stem) const
{
	int32 Index = static_cast<int32>(Stem);
	if (Index >= 0 && Index < 4)
	{
		return StemBeats[Index].bEnabled;
	}
	return false;
}

FString SWaveformViewer::GetStemName(EStemType Stem)
{
	switch (Stem)
	{
		case EStemType::Kick:  return TEXT("Kick");
		case EStemType::Snare: return TEXT("Snare");
		case EStemType::HiHat: return TEXT("Hi-Hat");
		case EStemType::Synth: return TEXT("Synth");
		default: return TEXT("Unknown");
	}
}

FLinearColor SWaveformViewer::GetStemColor(EStemType Stem)
{
	switch (Stem)
	{
		case EStemType::Kick:  return FLinearColor(0.1f, 0.1f, 0.1f, 0.9f);  // Black/dark
		case EStemType::Snare: return FLinearColor(0.2f, 0.4f, 1.0f, 0.9f);  // Blue
		case EStemType::HiHat: return FLinearColor(1.0f, 0.9f, 0.2f, 0.9f);  // Yellow
		case EStemType::Synth: return FLinearColor(1.0f, 0.2f, 0.2f, 0.9f);  // Red
		default: return FLinearColor::White;
	}
}

void SWaveformViewer::SetSelectionRange(double InStart, double InEnd)
{
	SelectionStart = FMath::Clamp(InStart, 0.0, Duration);
	SelectionEnd = FMath::Clamp(InEnd, SelectionStart, Duration);

	// Re-clamp all effect regions to stay within new selection bounds
	ClampEffectRegionsToSelection();

	Invalidate(EInvalidateWidget::Paint);
}

void SWaveformViewer::ZoomIn()
{
	SetZoomLevel(ZoomLevel * 1.5f);
}

void SWaveformViewer::ZoomOut()
{
	SetZoomLevel(ZoomLevel / 1.5f);
}

void SWaveformViewer::ZoomToFit()
{
	ZoomLevel = 1.0f;
	ScrollPosition = 0.0;
	Invalidate(EInvalidateWidget::Paint);
}

void SWaveformViewer::SetZoomLevel(float Level)
{
	float OldZoom = ZoomLevel;
	ZoomLevel = FMath::Clamp(Level, 1.0f, 50.0f);

	// Adjust scroll to keep center point stable
	if (OldZoom != ZoomLevel)
	{
		double MaxScroll = 1.0 - (1.0 / ZoomLevel);
		ScrollPosition = FMath::Clamp(ScrollPosition, 0.0, MaxScroll);
		Invalidate(EInvalidateWidget::Paint);
	}
}

void SWaveformViewer::SetScrollPosition(double Position)
{
	double MaxScroll = 1.0 - (1.0 / ZoomLevel);
	ScrollPosition = FMath::Clamp(Position, 0.0, MaxScroll);
	Invalidate(EInvalidateWidget::Paint);
}

// Effect region management
int32 SWaveformViewer::AddEffectRegion(const FString& EffectName, double StartTime, double EndTime, FLinearColor Color)
{
	// Constrain effect region to within track selection bounds
	double EffectiveSelEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;

	FEffectRegion Region;
	Region.EffectName = EffectName;
	Region.StartTime = FMath::Clamp(StartTime, SelectionStart, EffectiveSelEnd);
	Region.EndTime = FMath::Clamp(EndTime, Region.StartTime, EffectiveSelEnd);
	Region.Color = Color;
	Region.bEnabled = true;

	int32 Index = EffectRegions.Add(Region);
	Invalidate(EInvalidateWidget::Paint);
	return Index;
}

void SWaveformViewer::RemoveEffectRegion(int32 Index)
{
	if (EffectRegions.IsValidIndex(Index))
	{
		EffectRegions.RemoveAt(Index);
		if (SelectedEffectRegion == Index)
		{
			SelectedEffectRegion = -1;
		}
		else if (SelectedEffectRegion > Index)
		{
			SelectedEffectRegion--;
		}
		Invalidate(EInvalidateWidget::Paint);
	}
}

void SWaveformViewer::ClearEffectRegions()
{
	EffectRegions.Empty();
	SelectedEffectRegion = -1;
	Invalidate(EInvalidateWidget::Paint);
}

void SWaveformViewer::SetEffectRegionRange(int32 Index, double StartTime, double EndTime)
{
	if (EffectRegions.IsValidIndex(Index))
	{
		// Constrain effect region to within track selection bounds
		double EffectiveSelEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;
		EffectRegions[Index].StartTime = FMath::Clamp(StartTime, SelectionStart, EffectiveSelEnd);
		EffectRegions[Index].EndTime = FMath::Clamp(EndTime, EffectRegions[Index].StartTime, EffectiveSelEnd);
		Invalidate(EInvalidateWidget::Paint);
	}
}

void SWaveformViewer::ClampEffectRegionsToSelection()
{
	double EffectiveSelEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;

	for (FEffectRegion& Region : EffectRegions)
	{
		// Clamp start to be within selection
		Region.StartTime = FMath::Clamp(Region.StartTime, SelectionStart, EffectiveSelEnd);
		// Clamp end to be within selection (and after start)
		Region.EndTime = FMath::Clamp(Region.EndTime, Region.StartTime, EffectiveSelEnd);
	}
}

void SWaveformViewer::Clear()
{
	WaveformPeaks.Empty();
	BassPeaks.Empty();
	MidPeaks.Empty();
	HighPeaks.Empty();
	BeatTimes.Empty();
	EffectRegions.Empty();
	// Clear stem beats
	for (int32 i = 0; i < 4; ++i)
	{
		StemBeats[i].BeatTimes.Empty();
	}
	Duration = 0.0;
	SelectionStart = 0.0;
	SelectionEnd = -1.0;
	ZoomLevel = 1.0f;
	ScrollPosition = 0.0;
	SelectedEffectRegion = -1;
	Invalidate(EInvalidateWidget::Paint);
}

FVector2D SWaveformViewer::ComputeDesiredSize(float LayoutScaleMultiplier) const
{
	return FVector2D(400.0f, 150.0f);
}

void SWaveformViewer::GetVisibleTimeRange(double& OutStart, double& OutEnd) const
{
	if (Duration <= 0)
	{
		OutStart = 0.0;
		OutEnd = 0.0;
		return;
	}

	double VisibleDuration = Duration / ZoomLevel;
	OutStart = ScrollPosition * Duration;
	OutEnd = OutStart + VisibleDuration;
	OutEnd = FMath::Min(OutEnd, Duration);
}

double SWaveformViewer::PixelToTime(float X, float Width) const
{
	if (Width <= 0 || Duration <= 0) return 0.0;

	double VisibleStart, VisibleEnd;
	GetVisibleTimeRange(VisibleStart, VisibleEnd);
	double VisibleDuration = VisibleEnd - VisibleStart;

	return VisibleStart + (X / Width) * VisibleDuration;
}

float SWaveformViewer::TimeToPixel(double Time, float Width) const
{
	if (Duration <= 0) return 0.0f;

	double VisibleStart, VisibleEnd;
	GetVisibleTimeRange(VisibleStart, VisibleEnd);
	double VisibleDuration = VisibleEnd - VisibleStart;

	if (VisibleDuration <= 0) return 0.0f;

	return static_cast<float>(((Time - VisibleStart) / VisibleDuration) * Width);
}

bool SWaveformViewer::IsOverStartHandle(float X, float Width) const
{
	float HandleX = TimeToPixel(SelectionStart, Width);
	return FMath::Abs(X - HandleX) <= (HandleWidth + HandleHitPadding);
}

bool SWaveformViewer::IsOverEndHandle(float X, float Width) const
{
	double EffectiveEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;
	float HandleX = TimeToPixel(EffectiveEnd, Width);
	return FMath::Abs(X - HandleX) <= (HandleWidth + HandleHitPadding);
}

int32 SWaveformViewer::GetEffectRegionAtPosition(float X, float Width) const
{
	double Time = PixelToTime(X, Width);

	for (int32 i = EffectRegions.Num() - 1; i >= 0; --i)
	{
		const FEffectRegion& Region = EffectRegions[i];
		if (Time >= Region.StartTime && Time <= Region.EndTime)
		{
			return i;
		}
	}
	return -1;
}

bool SWaveformViewer::IsOverEffectStartHandle(int32 RegionIndex, float X, float Width) const
{
	if (!EffectRegions.IsValidIndex(RegionIndex)) return false;

	float HandleX = TimeToPixel(EffectRegions[RegionIndex].StartTime, Width);
	return FMath::Abs(X - HandleX) <= (HandleWidth + HandleHitPadding);
}

bool SWaveformViewer::IsOverEffectEndHandle(int32 RegionIndex, float X, float Width) const
{
	if (!EffectRegions.IsValidIndex(RegionIndex)) return false;

	float HandleX = TimeToPixel(EffectRegions[RegionIndex].EndTime, Width);
	return FMath::Abs(X - HandleX) <= (HandleWidth + HandleHitPadding);
}

void SWaveformViewer::ShowBeatContextMenu(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(MouseEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;
	ContextMenuTime = PixelToTime(LocalPos.X, Width);

	// Check if clicking near an existing beat
	int32 BeatIndex = FindBeatNearPixel(LocalPos.X, Width, 15.0f);

	FMenuBuilder MenuBuilder(true, nullptr);

	if (BeatIndex >= 0)
	{
		// Context menu for existing beat marker
		double BeatTime = BeatTimes[BeatIndex];
		MenuBuilder.BeginSection("BeatMarker", FText::FromString(FString::Printf(TEXT("Beat %d (%.2fs)"), BeatIndex + 1, BeatTime)));

		MenuBuilder.AddMenuEntry(
			FText::FromString(TEXT("Remove Beat Marker")),
			FText::FromString(TEXT("Delete this beat marker")),
			FSlateIcon(),
			FUIAction(FExecuteAction::CreateLambda([this, BeatIndex]()
			{
				RemoveBeatAtIndex(BeatIndex);
			}))
		);

		MenuBuilder.EndSection();
	}
	else
	{
		// Context menu to add new beat
		MenuBuilder.BeginSection("AddBeat", FText::FromString(FString::Printf(TEXT("At %.2fs"), ContextMenuTime)));

		MenuBuilder.AddMenuEntry(
			FText::FromString(TEXT("Add Beat Marker")),
			FText::FromString(TEXT("Add a beat marker at this position")),
			FSlateIcon(),
			FUIAction(FExecuteAction::CreateLambda([this]()
			{
				AddBeatAtTime(ContextMenuTime);
			}))
		);

		MenuBuilder.EndSection();
	}

	FWidgetPath WidgetPath;
	FSlateApplication::Get().GeneratePathToWidgetUnchecked(AsShared(), WidgetPath);

	FSlateApplication::Get().PushMenu(
		AsShared(),
		WidgetPath,
		MenuBuilder.MakeWidget(),
		MouseEvent.GetScreenSpacePosition(),
		FPopupTransitionEffect::ContextMenu
	);
}

void SWaveformViewer::ShowEffectContextMenu(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(MouseEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;
	ContextMenuTime = PixelToTime(LocalPos.X, Width);

	// Check if clicking on an existing region
	int32 RegionIndex = GetEffectRegionAtPosition(LocalPos.X, Width);

	FMenuBuilder MenuBuilder(true, nullptr);

	if (RegionIndex >= 0)
	{
		// Context menu for existing effect region
		SelectedEffectRegion = RegionIndex;
		const FEffectRegion& Region = EffectRegions[RegionIndex];

		MenuBuilder.BeginSection("EffectRegion", FText::FromString(Region.EffectName));

		MenuBuilder.AddMenuEntry(
			FText::FromString(TEXT("Remove Effect Region")),
			FText::FromString(TEXT("Delete this effect region")),
			FSlateIcon(),
			FUIAction(FExecuteAction::CreateLambda([this, RegionIndex]()
			{
				RemoveEffectRegion(RegionIndex);
			}))
		);

		MenuBuilder.EndSection();
	}
	else
	{
		// Context menu to add new effect regions
		MenuBuilder.BeginSection("AddEffect", FText::FromString(TEXT("Add Effect Region")));

		// Vignette
		MenuBuilder.AddMenuEntry(
			FText::FromString(TEXT("Vignette")),
			FText::FromString(TEXT("Add vignette effect region")),
			FSlateIcon(),
			FUIAction(FExecuteAction::CreateLambda([this]()
			{
				double EndTime = FMath::Min(ContextMenuTime + 5.0, Duration);
				AddEffectRegion(TEXT("Vignette"), ContextMenuTime, EndTime,
					FLinearColor(0.3f, 0.0f, 0.5f, 0.4f));
			}))
		);

		// Beat Flash
		MenuBuilder.AddMenuEntry(
			FText::FromString(TEXT("Beat Flash")),
			FText::FromString(TEXT("Add beat flash effect region")),
			FSlateIcon(),
			FUIAction(FExecuteAction::CreateLambda([this]()
			{
				double EndTime = FMath::Min(ContextMenuTime + 5.0, Duration);
				AddEffectRegion(TEXT("Beat Flash"), ContextMenuTime, EndTime,
					FLinearColor(1.0f, 0.8f, 0.0f, 0.4f));
			}))
		);

		// Beat Zoom
		MenuBuilder.AddMenuEntry(
			FText::FromString(TEXT("Beat Zoom")),
			FText::FromString(TEXT("Add beat zoom effect region")),
			FSlateIcon(),
			FUIAction(FExecuteAction::CreateLambda([this]()
			{
				double EndTime = FMath::Min(ContextMenuTime + 5.0, Duration);
				AddEffectRegion(TEXT("Beat Zoom"), ContextMenuTime, EndTime,
					FLinearColor(0.0f, 0.8f, 1.0f, 0.4f));
			}))
		);

		// Color Grade
		MenuBuilder.AddMenuEntry(
			FText::FromString(TEXT("Color Grade")),
			FText::FromString(TEXT("Add color grading effect region")),
			FSlateIcon(),
			FUIAction(FExecuteAction::CreateLambda([this]()
			{
				double EndTime = FMath::Min(ContextMenuTime + 5.0, Duration);
				AddEffectRegion(TEXT("Color Grade"), ContextMenuTime, EndTime,
					FLinearColor(0.0f, 1.0f, 0.5f, 0.4f));
			}))
		);

		MenuBuilder.EndSection();
	}

	FWidgetPath WidgetPath;
	FSlateApplication::Get().GeneratePathToWidgetUnchecked(AsShared(), WidgetPath);

	FSlateApplication::Get().PushMenu(
		AsShared(),
		WidgetPath,
		MenuBuilder.MakeWidget(),
		MouseEvent.GetScreenSpacePosition(),
		FPopupTransitionEffect::ContextMenu
	);
}

FReply SWaveformViewer::OnMouseButtonDown(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	if (Duration <= 0) return FReply::Unhandled();

	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(MouseEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;
	const float RulerHeight = 24.0f;

	if (MouseEvent.GetEffectingButton() == EKeys::LeftMouseButton)
	{
		// Shift+Click: Add beat marker at click position
		if (MouseEvent.IsShiftDown() && LocalPos.Y > RulerHeight)
		{
			double ClickTime = PixelToTime(LocalPos.X, Width);
			AddBeatAtTime(ClickTime);
			return FReply::Handled();
		}

		// Ctrl+Click: Remove beat marker near click position
		if (MouseEvent.IsControlDown() && LocalPos.Y > RulerHeight)
		{
			int32 BeatIndex = FindBeatNearPixel(LocalPos.X, Width, 15.0f);
			if (BeatIndex >= 0)
			{
				RemoveBeatAtIndex(BeatIndex);
			}
			return FReply::Handled();
		}

		// Check if clicking on a beat marker to drag it (no modifiers, tight 5px tolerance)
		if (!MouseEvent.IsShiftDown() && !MouseEvent.IsControlDown() && LocalPos.Y > RulerHeight)
		{
			int32 BeatIndex = FindBeatNearPixel(LocalPos.X, Width, 5.0f);
			if (BeatIndex >= 0)
			{
				CurrentDragMode = EDragMode::DragBeat;
				DragBeatIndex = BeatIndex;
				DragStartTime = BeatTimes[BeatIndex];
				LastMousePos = LocalPos;
				return FReply::Handled().CaptureMouse(SharedThis(this));
			}
		}

		// First check if we're over an effect region handle
		for (int32 i = 0; i < EffectRegions.Num(); ++i)
		{
			if (IsOverEffectStartHandle(i, LocalPos.X, Width))
			{
				CurrentDragMode = EDragMode::DragEffectStart;
				DragEffectIndex = i;
				SelectedEffectRegion = i;
				LastMousePos = LocalPos;
				Invalidate(EInvalidateWidget::Paint);
				return FReply::Handled().CaptureMouse(SharedThis(this));
			}
			else if (IsOverEffectEndHandle(i, LocalPos.X, Width))
			{
				CurrentDragMode = EDragMode::DragEffectEnd;
				DragEffectIndex = i;
				SelectedEffectRegion = i;
				LastMousePos = LocalPos;
				Invalidate(EInvalidateWidget::Paint);
				return FReply::Handled().CaptureMouse(SharedThis(this));
			}
		}

		// Check for selection handle hits
		if (IsOverStartHandle(LocalPos.X, Width))
		{
			CurrentDragMode = EDragMode::DragStart;
			LastMousePos = LocalPos;
			return FReply::Handled().CaptureMouse(SharedThis(this));
		}
		else if (IsOverEndHandle(LocalPos.X, Width))
		{
			CurrentDragMode = EDragMode::DragEnd;
			LastMousePos = LocalPos;
			return FReply::Handled().CaptureMouse(SharedThis(this));
		}
		else
		{
			// Check if clicking on an effect region (to select it)
			int32 RegionIndex = GetEffectRegionAtPosition(LocalPos.X, Width);
			if (RegionIndex >= 0)
			{
				SelectedEffectRegion = RegionIndex;
				Invalidate(EInvalidateWidget::Paint);
				return FReply::Handled();
			}

			// Clicking in empty space just deselects effect regions
			// (Use the pink handles to adjust selection range, or Shift+Click to add beats)
			SelectedEffectRegion = -1;
			Invalidate(EInvalidateWidget::Paint);
			return FReply::Handled();
		}
	}
	else if (MouseEvent.GetEffectingButton() == EKeys::RightMouseButton)
	{
		// Show context menu for adding/removing beat markers
		ShowBeatContextMenu(MyGeometry, MouseEvent);
		return FReply::Handled();
	}
	else if (MouseEvent.GetEffectingButton() == EKeys::MiddleMouseButton)
	{
		// Pan with middle mouse
		CurrentDragMode = EDragMode::Pan;
		LastMousePos = LocalPos;
		return FReply::Handled().CaptureMouse(SharedThis(this));
	}

	return FReply::Unhandled();
}

FReply SWaveformViewer::OnMouseButtonUp(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	if (CurrentDragMode != EDragMode::None)
	{
		// Fire callback if we were dragging an effect region
		if ((CurrentDragMode == EDragMode::DragEffectStart || CurrentDragMode == EDragMode::DragEffectEnd)
			&& DragEffectIndex >= 0 && EffectRegions.IsValidIndex(DragEffectIndex))
		{
			if (OnEffectRegionChanged.IsBound())
			{
				OnEffectRegionChanged.Execute(DragEffectIndex,
					EffectRegions[DragEffectIndex].StartTime,
					EffectRegions[DragEffectIndex].EndTime);
			}
		}

		// Fire callback if we were dragging a beat marker
		if (CurrentDragMode == EDragMode::DragBeat && DragBeatIndex >= 0)
		{
			// Re-sort beat times array to maintain order
			BeatTimes.Sort();

			// Notify listeners of the change
			if (OnBeatTimesChanged.IsBound())
			{
				OnBeatTimesChanged.Execute(BeatTimes);
			}

			Invalidate(EInvalidateWidget::Paint);
		}

		CurrentDragMode = EDragMode::None;
		DragEffectIndex = -1;
		DragBeatIndex = -1;
		return FReply::Handled().ReleaseMouseCapture();
	}
	return FReply::Unhandled();
}

FReply SWaveformViewer::OnMouseMove(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	if (Duration <= 0) return FReply::Unhandled();

	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(MouseEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;

	if (CurrentDragMode == EDragMode::DragStart)
	{
		double NewTime = PixelToTime(LocalPos.X, Width);
		double EffectiveEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;
		SelectionStart = FMath::Clamp(NewTime, 0.0, EffectiveEnd - 0.1);

		// Re-clamp effect regions when selection changes
		ClampEffectRegionsToSelection();

		if (OnSelectionChanged.IsBound())
		{
			OnSelectionChanged.Execute(SelectionStart, SelectionEnd);
		}
		Invalidate(EInvalidateWidget::Paint);
		LastMousePos = LocalPos;
		return FReply::Handled();
	}
	else if (CurrentDragMode == EDragMode::DragEnd)
	{
		double NewTime = PixelToTime(LocalPos.X, Width);
		SelectionEnd = FMath::Clamp(NewTime, SelectionStart + 0.1, Duration);

		// Re-clamp effect regions when selection changes
		ClampEffectRegionsToSelection();

		if (OnSelectionChanged.IsBound())
		{
			OnSelectionChanged.Execute(SelectionStart, SelectionEnd);
		}
		Invalidate(EInvalidateWidget::Paint);
		LastMousePos = LocalPos;
		return FReply::Handled();
	}
	else if (CurrentDragMode == EDragMode::Pan)
	{
		float DeltaX = LocalPos.X - LastMousePos.X;
		double TimeDelta = -DeltaX / Width * (Duration / ZoomLevel);
		double NewScroll = ScrollPosition + (TimeDelta / Duration);
		SetScrollPosition(NewScroll);
		LastMousePos = LocalPos;
		return FReply::Handled();
	}
	else if (CurrentDragMode == EDragMode::DragEffectStart && EffectRegions.IsValidIndex(DragEffectIndex))
	{
		double NewTime = PixelToTime(LocalPos.X, Width);
		double EndTime = EffectRegions[DragEffectIndex].EndTime;
		// Constrain effect region to within track selection bounds
		double MinBound = SelectionStart;
		EffectRegions[DragEffectIndex].StartTime = FMath::Clamp(NewTime, MinBound, EndTime - 0.1);
		Invalidate(EInvalidateWidget::Paint);
		LastMousePos = LocalPos;
		return FReply::Handled();
	}
	else if (CurrentDragMode == EDragMode::DragEffectEnd && EffectRegions.IsValidIndex(DragEffectIndex))
	{
		double NewTime = PixelToTime(LocalPos.X, Width);
		double StartTime = EffectRegions[DragEffectIndex].StartTime;
		// Constrain effect region to within track selection bounds
		double EffectiveSelEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;
		double MaxBound = EffectiveSelEnd;
		EffectRegions[DragEffectIndex].EndTime = FMath::Clamp(NewTime, StartTime + 0.1, MaxBound);
		Invalidate(EInvalidateWidget::Paint);
		LastMousePos = LocalPos;
		return FReply::Handled();
	}
	else if (CurrentDragMode == EDragMode::DragBeat && BeatTimes.IsValidIndex(DragBeatIndex))
	{
		double NewTime = PixelToTime(LocalPos.X, Width);
		// Clamp to valid range (within track duration)
		NewTime = FMath::Clamp(NewTime, 0.0, Duration);

		// Update the beat time directly (don't re-sort during drag for stability)
		BeatTimes[DragBeatIndex] = NewTime;

		Invalidate(EInvalidateWidget::Paint);
		LastMousePos = LocalPos;
		return FReply::Handled();
	}

	return FReply::Unhandled();
}

FReply SWaveformViewer::OnMouseWheel(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	if (Duration <= 0) return FReply::Unhandled();

	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(MouseEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;

	// Get time at mouse position before zoom
	double TimeAtMouse = PixelToTime(LocalPos.X, Width);

	// Apply zoom
	float ZoomDelta = MouseEvent.GetWheelDelta();
	float NewZoom = ZoomLevel * FMath::Pow(1.15f, ZoomDelta);
	NewZoom = FMath::Clamp(NewZoom, 1.0f, 50.0f);

	if (NewZoom != ZoomLevel)
	{
		ZoomLevel = NewZoom;

		// Adjust scroll to keep time at mouse position stable
		double VisibleDuration = Duration / ZoomLevel;
		double NewScrollTime = TimeAtMouse - (LocalPos.X / Width) * VisibleDuration;
		ScrollPosition = FMath::Clamp(NewScrollTime / Duration, 0.0, 1.0 - (1.0 / ZoomLevel));

		Invalidate(EInvalidateWidget::Paint);
	}

	return FReply::Handled();
}

FCursorReply SWaveformViewer::OnCursorQuery(const FGeometry& MyGeometry, const FPointerEvent& CursorEvent) const
{
	if (Duration <= 0) return FCursorReply::Unhandled();

	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(CursorEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;
	const float RulerHeight = 24.0f;

	// Check if hovering over a beat marker (in waveform area, not ruler, tight 5px tolerance)
	if (LocalPos.Y > RulerHeight)
	{
		int32 BeatIndex = FindBeatNearPixel(LocalPos.X, Width, 5.0f);
		if (BeatIndex >= 0)
		{
			return FCursorReply::Cursor(EMouseCursor::CardinalCross);  // Move cursor for dragging
		}
	}

	// Check effect region handles
	for (int32 i = 0; i < EffectRegions.Num(); ++i)
	{
		if (IsOverEffectStartHandle(i, LocalPos.X, Width) || IsOverEffectEndHandle(i, LocalPos.X, Width))
		{
			return FCursorReply::Cursor(EMouseCursor::ResizeLeftRight);
		}
	}

	// Then selection handles
	if (IsOverStartHandle(LocalPos.X, Width) || IsOverEndHandle(LocalPos.X, Width))
	{
		return FCursorReply::Cursor(EMouseCursor::ResizeLeftRight);
	}

	return FCursorReply::Cursor(EMouseCursor::Default);
}

int32 SWaveformViewer::OnPaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry,
	const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements,
	int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const
{
	const bool bEnabled = ShouldBeEnabled(bParentEnabled);
	const ESlateDrawEffect DrawEffects = bEnabled ? ESlateDrawEffect::None : ESlateDrawEffect::DisabledEffect;

	const FVector2D LocalSize = AllottedGeometry.GetLocalSize();
	const float Width = LocalSize.X;
	const float Height = LocalSize.Y;

	if (Width <= 0 || Height <= 0)
	{
		return LayerId;
	}

	// Draw background
	FSlateDrawElement::MakeBox(
		OutDrawElements,
		LayerId,
		AllottedGeometry.ToPaintGeometry(),
		FCoreStyle::Get().GetBrush("GenericWhiteBox"),
		DrawEffects,
		FLinearColor(0.02f, 0.02f, 0.05f, 0.9f) // Dark background
	);

	LayerId++;

	// Get visible time range
	double VisibleStart, VisibleEnd;
	GetVisibleTimeRange(VisibleStart, VisibleEnd);

	// Draw effect regions (below selection)
	for (int32 i = 0; i < EffectRegions.Num(); ++i)
	{
		const FEffectRegion& Region = EffectRegions[i];
		if (!Region.bEnabled) continue;

		float RegStartPx = TimeToPixel(Region.StartTime, Width);
		float RegEndPx = TimeToPixel(Region.EndTime, Width);

		RegStartPx = FMath::Max(0.0f, RegStartPx);
		RegEndPx = FMath::Min(Width, RegEndPx);

		if (RegEndPx > RegStartPx)
		{
			// Draw region background
			FLinearColor RegionColor = Region.Color;
			if (i == SelectedEffectRegion)
			{
				RegionColor.A = FMath::Min(1.0f, RegionColor.A + 0.2f);
			}

			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(RegEndPx - RegStartPx, Height),
					FSlateLayoutTransform(FVector2D(RegStartPx, 0))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				RegionColor
			);

			// Draw effect name label
			FSlateFontInfo FontInfo = FCoreStyle::Get().GetFontStyle("SmallFont");
			FSlateDrawElement::MakeText(
				OutDrawElements,
				LayerId + 1,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(RegEndPx - RegStartPx - 4, 16),
					FSlateLayoutTransform(FVector2D(RegStartPx + 2, 2))
				),
				Region.EffectName,
				FontInfo,
				DrawEffects,
				FLinearColor::White
			);
		}
	}

	LayerId += 2;

	// Draw selection region if set
	double EffectiveSelEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;
	if (Duration > 0 && SelectionStart < EffectiveSelEnd)
	{
		float SelStartPx = TimeToPixel(SelectionStart, Width);
		float SelEndPx = TimeToPixel(EffectiveSelEnd, Width);

		// Clamp to visible area
		SelStartPx = FMath::Max(0.0f, SelStartPx);
		SelEndPx = FMath::Min(Width, SelEndPx);

		if (SelEndPx > SelStartPx)
		{
			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(SelEndPx - SelStartPx, Height),
					FSlateLayoutTransform(FVector2D(SelStartPx, 0))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				SelectionColor
			);
		}
	}

	LayerId++;

	// Draw waveform - either frequency-band colored (Rekordbox style) or single color
	// Account for ruler height at top
	const float RulerHeight = 24.0f;
	const float WaveformHeight = Height - RulerHeight;
	const float CenterY = RulerHeight + WaveformHeight * 0.5f;
	const float MaxAmplitude = WaveformHeight * 0.45f;

	if (BassPeaks.Num() > 0 && Duration > 0)
	{
		// Frequency-band colored waveform (Rekordbox/Traktor style)
		// Draw in order: Bass (bottom), Mids (middle), Highs (top) - stacked
		const int32 NumPeaks = BassPeaks.Num();

		// Calculate which peaks are visible
		int32 StartPeak = FMath::Max(0, FMath::FloorToInt((VisibleStart / Duration) * NumPeaks) - 1);
		int32 EndPeak = FMath::Min(NumPeaks, FMath::CeilToInt((VisibleEnd / Duration) * NumPeaks) + 1);
		int32 VisiblePeakCount = EndPeak - StartPeak;

		// LOD: Calculate step size based on peaks per pixel
		// If we have more peaks than pixels, skip some (but take max in range)
		int32 PeaksPerPixel = FMath::Max(1, FMath::CeilToInt(static_cast<float>(VisiblePeakCount) / Width));
		int32 Step = FMath::Max(1, PeaksPerPixel / 2);  // Sample at 2x pixel rate for quality

		// Calculate bar width based on zoom and step
		float BarWidth = FMath::Max(1.0f, (Width * Step) / (VisiblePeakCount + 1));

		// Colors matching Rekordbox style
		const FLinearColor BassColor(0.9f, 0.2f, 0.2f, 0.9f);   // Red for bass
		const FLinearColor MidColor(0.2f, 0.8f, 0.9f, 0.85f);    // Cyan for mids
		const FLinearColor HighColor(0.95f, 0.95f, 0.95f, 0.8f); // White for highs

		TArray<FVector2D> BarPoints;

		for (int32 i = StartPeak; i < EndPeak; i += Step)
		{
			double PeakTime = (static_cast<double>(i) / NumPeaks) * Duration;
			float X = TimeToPixel(PeakTime, Width);

			if (X < -BarWidth || X > Width + BarWidth) continue;

			// When stepping, take max value in the range for accurate transient display
			float BassValue = 0.0f, MidValue = 0.0f, HighValue = 0.0f;
			int32 RangeEnd = FMath::Min(i + Step, EndPeak);
			for (int32 j = i; j < RangeEnd; ++j)
			{
				if (j < BassPeaks.Num()) BassValue = FMath::Max(BassValue, BassPeaks[j]);
				if (j < MidPeaks.Num()) MidValue = FMath::Max(MidValue, MidPeaks[j]);
				if (j < HighPeaks.Num()) HighValue = FMath::Max(HighValue, HighPeaks[j]);
			}
			BassValue = FMath::Clamp(BassValue, 0.0f, 1.0f);
			MidValue = FMath::Clamp(MidValue, 0.0f, 1.0f);
			HighValue = FMath::Clamp(HighValue, 0.0f, 1.0f);

			// Draw bass (largest, background)
			if (BassValue > 0.01f)
			{
				float BassOffset = BassValue * MaxAmplitude;
				BarPoints.Reset(2);
				BarPoints.Add(FVector2D(X, CenterY - BassOffset));
				BarPoints.Add(FVector2D(X, CenterY + BassOffset));
				FSlateDrawElement::MakeLines(
					OutDrawElements, LayerId, AllottedGeometry.ToPaintGeometry(),
					BarPoints, DrawEffects, BassColor, true, FMath::Max(1.0f, BarWidth * 0.9f)
				);
			}

			// Draw mids (medium, overlaid)
			if (MidValue > 0.01f)
			{
				float MidOffset = MidValue * MaxAmplitude * 0.85f;
				BarPoints.Reset(2);
				BarPoints.Add(FVector2D(X, CenterY - MidOffset));
				BarPoints.Add(FVector2D(X, CenterY + MidOffset));
				FSlateDrawElement::MakeLines(
					OutDrawElements, LayerId + 1, AllottedGeometry.ToPaintGeometry(),
					BarPoints, DrawEffects, MidColor, true, FMath::Max(1.0f, BarWidth * 0.7f)
				);
			}

			// Draw highs (smallest, on top)
			if (HighValue > 0.01f)
			{
				float HighOffset = HighValue * MaxAmplitude * 0.6f;
				BarPoints.Reset(2);
				BarPoints.Add(FVector2D(X, CenterY - HighOffset));
				BarPoints.Add(FVector2D(X, CenterY + HighOffset));
				FSlateDrawElement::MakeLines(
					OutDrawElements, LayerId + 2, AllottedGeometry.ToPaintGeometry(),
					BarPoints, DrawEffects, HighColor, true, FMath::Max(1.0f, BarWidth * 0.5f)
				);
			}
		}
		LayerId += 3;
	}
	else if (WaveformPeaks.Num() > 0 && Duration > 0)
	{
		// Single-color waveform (fallback/legacy mode)
		const int32 NumPeaks = WaveformPeaks.Num();

		int32 StartPeak = FMath::Max(0, FMath::FloorToInt((VisibleStart / Duration) * NumPeaks) - 1);
		int32 EndPeak = FMath::Min(NumPeaks, FMath::CeilToInt((VisibleEnd / Duration) * NumPeaks) + 1);
		int32 VisiblePeakCount = EndPeak - StartPeak;

		// LOD: Calculate step size based on peaks per pixel
		int32 PeaksPerPixel = FMath::Max(1, FMath::CeilToInt(static_cast<float>(VisiblePeakCount) / Width));
		int32 Step = FMath::Max(1, PeaksPerPixel / 2);

		float BarWidth = FMath::Max(1.0f, (Width * Step) / (VisiblePeakCount + 1));

		TArray<FVector2D> BarPoints;
		for (int32 i = StartPeak; i < EndPeak; i += Step)
		{
			double PeakTime = (static_cast<double>(i) / NumPeaks) * Duration;
			float X = TimeToPixel(PeakTime, Width);

			if (X < -BarWidth || X > Width + BarWidth) continue;

			// When stepping, take max value in the range for accurate transient display
			float PeakValue = 0.0f;
			int32 RangeEnd = FMath::Min(i + Step, EndPeak);
			for (int32 j = i; j < RangeEnd; ++j)
			{
				if (j < WaveformPeaks.Num())
				{
					PeakValue = FMath::Max(PeakValue, WaveformPeaks[j]);
				}
			}
			PeakValue = FMath::Clamp(PeakValue, 0.0f, 1.0f);
			float YOffset = PeakValue * MaxAmplitude;

			BarPoints.Reset(2);
			BarPoints.Add(FVector2D(X, CenterY - YOffset));
			BarPoints.Add(FVector2D(X, CenterY + YOffset));

			FSlateDrawElement::MakeLines(
				OutDrawElements, LayerId, AllottedGeometry.ToPaintGeometry(),
				BarPoints, DrawEffects, WaveformColor, true, FMath::Max(1.0f, BarWidth * 0.8f)
			);
		}
		LayerId++;
	}
	else
	{
		LayerId++;
	}

	// Draw beat grid with time ruler (RulerHeight defined above = 24.0f)
	if (BeatTimes.Num() > 0 && Duration > 0)
	{
		// Draw ruler background
		FSlateDrawElement::MakeBox(
			OutDrawElements,
			LayerId,
			AllottedGeometry.ToPaintGeometry(
				FVector2D(Width, RulerHeight),
				FSlateLayoutTransform(FVector2D(0, 0))
			),
			FCoreStyle::Get().GetBrush("GenericWhiteBox"),
			DrawEffects,
			FLinearColor(0.05f, 0.05f, 0.1f, 0.95f) // Darker ruler background
		);
		LayerId++;

		// All beats are yellow
		const FLinearColor BeatMarkerYellow(1.0f, 0.8f, 0.2f, 0.9f);

		TArray<FVector2D> MarkerPoints;
		FSlateFontInfo RulerFont = FCoreStyle::Get().GetFontStyle("SmallFont");

		for (int32 BeatIdx = 0; BeatIdx < BeatTimes.Num(); ++BeatIdx)
		{
			double BeatTime = BeatTimes[BeatIdx];
			if (BeatTime < VisibleStart - 1.0 || BeatTime > VisibleEnd + 1.0) continue;

			float X = TimeToPixel(BeatTime, Width);
			if (X < -10 || X > Width + 10) continue;

			int32 BeatNumber = BeatIdx + 1;  // 1-indexed beat number

			// Draw yellow line from top to bottom for every beat
			MarkerPoints.Reset(2);
			MarkerPoints.Add(FVector2D(X, 0));
			MarkerPoints.Add(FVector2D(X, Height));

			FSlateDrawElement::MakeLines(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(),
				MarkerPoints,
				DrawEffects,
				BeatMarkerYellow,
				true,
				2.0f
			);

			// Draw beat number in ruler for every beat
			FString BeatText = FString::Printf(TEXT("%d"), BeatNumber);
			FSlateDrawElement::MakeText(
				OutDrawElements,
				LayerId + 1,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(30, RulerHeight - 4),
					FSlateLayoutTransform(FVector2D(X + 3, 2))
				),
				BeatText,
				RulerFont,
				DrawEffects,
				BeatMarkerYellow
			);
		}
	}
	else
	{
		// No beats - still draw ruler background
		FSlateDrawElement::MakeBox(
			OutDrawElements,
			LayerId,
			AllottedGeometry.ToPaintGeometry(
				FVector2D(Width, RulerHeight),
				FSlateLayoutTransform(FVector2D(0, 0))
			),
			FCoreStyle::Get().GetBrush("GenericWhiteBox"),
			DrawEffects,
			FLinearColor(0.05f, 0.05f, 0.1f, 0.95f)
		);
	}

	LayerId += 2;

	// Draw stem beat markers (colored lines for each stem type)
	if (Duration > 0)
	{
		TArray<FVector2D> StemMarkerPoints;

		for (int32 StemIdx = 0; StemIdx < 4; ++StemIdx)
		{
			const FStemBeats& Stem = StemBeats[StemIdx];
			if (!Stem.bEnabled || Stem.BeatTimes.Num() == 0) continue;

			for (int32 BeatIdx = 0; BeatIdx < Stem.BeatTimes.Num(); ++BeatIdx)
			{
				double BeatTime = Stem.BeatTimes[BeatIdx];
				if (BeatTime < VisibleStart - 1.0 || BeatTime > VisibleEnd + 1.0) continue;

				float X = TimeToPixel(BeatTime, Width);
				if (X < -10 || X > Width + 10) continue;

				// Draw colored line from below ruler to bottom
				// Offset slightly based on stem index to avoid overlap
				float XOffset = (StemIdx - 1.5f) * 1.5f;

				StemMarkerPoints.Reset(2);
				StemMarkerPoints.Add(FVector2D(X + XOffset, RulerHeight));
				StemMarkerPoints.Add(FVector2D(X + XOffset, Height));

				FSlateDrawElement::MakeLines(
					OutDrawElements,
					LayerId,
					AllottedGeometry.ToPaintGeometry(),
					StemMarkerPoints,
					DrawEffects,
					Stem.Color,
					true,
					2.0f
				);
			}
		}
		LayerId++;
	}

	// Draw center line (in waveform area, below ruler)
	{
		const float WaveformCenterY = 24.0f + (Height - 24.0f) * 0.5f;  // Ruler height = 24
		TArray<FVector2D> CenterLinePoints;
		CenterLinePoints.Add(FVector2D(0, WaveformCenterY));
		CenterLinePoints.Add(FVector2D(Width, WaveformCenterY));

		FSlateDrawElement::MakeLines(
			OutDrawElements,
			LayerId,
			AllottedGeometry.ToPaintGeometry(),
			CenterLinePoints,
			DrawEffects,
			FLinearColor(0.3f, 0.3f, 0.3f, 0.5f),
			true,
			1.0f
		);
	}

	LayerId++;

	// Draw effect region handles
	for (int32 i = 0; i < EffectRegions.Num(); ++i)
	{
		const FEffectRegion& Region = EffectRegions[i];
		if (!Region.bEnabled) continue;

		float StartHandleX = TimeToPixel(Region.StartTime, Width);
		float EndHandleX = TimeToPixel(Region.EndTime, Width);

		FLinearColor RegionHandleColor = Region.Color;
		RegionHandleColor.A = 1.0f;
		if (i == SelectedEffectRegion)
		{
			RegionHandleColor = FLinearColor::White;
		}

		// Start handle
		if (StartHandleX >= -HandleWidth && StartHandleX <= Width + HandleWidth)
		{
			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(4, Height),
					FSlateLayoutTransform(FVector2D(StartHandleX - 2, 0))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				RegionHandleColor
			);
		}

		// End handle
		if (EndHandleX >= -HandleWidth && EndHandleX <= Width + HandleWidth)
		{
			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(4, Height),
					FSlateLayoutTransform(FVector2D(EndHandleX - 2, 0))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				RegionHandleColor
			);
		}
	}

	LayerId++;

	// Draw selection handles
	if (Duration > 0)
	{
		float StartHandleX = TimeToPixel(SelectionStart, Width);
		float EndHandleX = TimeToPixel(EffectiveSelEnd, Width);

		// Start handle (left)
		if (StartHandleX >= -HandleWidth && StartHandleX <= Width + HandleWidth)
		{
			// Handle bar
			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(HandleWidth, Height),
					FSlateLayoutTransform(FVector2D(StartHandleX - HandleWidth / 2, 0))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				HandleColor
			);

			// Handle triangle top
			TArray<FVector2D> TriTop;
			TriTop.Add(FVector2D(StartHandleX - HandleWidth, 0));
			TriTop.Add(FVector2D(StartHandleX + HandleWidth, 0));
			TriTop.Add(FVector2D(StartHandleX, HandleWidth * 1.5f));

			FSlateDrawElement::MakeLines(
				OutDrawElements,
				LayerId + 1,
				AllottedGeometry.ToPaintGeometry(),
				TriTop,
				DrawEffects,
				HandleColor,
				true,
				2.0f
			);
		}

		// End handle (right)
		if (EndHandleX >= -HandleWidth && EndHandleX <= Width + HandleWidth)
		{
			// Handle bar
			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(HandleWidth, Height),
					FSlateLayoutTransform(FVector2D(EndHandleX - HandleWidth / 2, 0))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				HandleColor
			);

			// Handle triangle bottom
			TArray<FVector2D> TriBottom;
			TriBottom.Add(FVector2D(EndHandleX - HandleWidth, Height));
			TriBottom.Add(FVector2D(EndHandleX + HandleWidth, Height));
			TriBottom.Add(FVector2D(EndHandleX, Height - HandleWidth * 1.5f));

			FSlateDrawElement::MakeLines(
				OutDrawElements,
				LayerId + 1,
				AllottedGeometry.ToPaintGeometry(),
				TriBottom,
				DrawEffects,
				HandleColor,
				true,
				2.0f
			);
		}
	}

	LayerId += 2;

	// Draw zoom indicator (top-right corner)
	if (ZoomLevel > 1.01f)
	{
		FString ZoomText = FString::Printf(TEXT("%.1fx"), ZoomLevel);
		FSlateFontInfo FontInfo = FCoreStyle::Get().GetFontStyle("SmallFont");

		FSlateDrawElement::MakeText(
			OutDrawElements,
			LayerId,
			AllottedGeometry.ToPaintGeometry(
				FVector2D(50, 20),
				FSlateLayoutTransform(FVector2D(Width - 55, 5))
			),
			ZoomText,
			FontInfo,
			DrawEffects,
			FLinearColor(0.8f, 0.8f, 0.8f, 0.7f)
		);
	}

	return LayerId;
}
