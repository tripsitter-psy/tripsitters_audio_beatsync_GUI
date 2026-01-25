#include "SEffectTimeline.h"
#include "SWaveformViewer.h"  // For FEffectRegion
#include "Rendering/DrawElements.h"
#include "Styling/CoreStyle.h"
#include "Framework/Application/SlateApplication.h"
#include "Framework/MultiBox/MultiBoxBuilder.h"

void SEffectTimeline::Construct(const FArguments& InArgs)
{
	BackgroundColor = InArgs._BackgroundColor;
	OnRegionChanged = InArgs._OnRegionChanged;
	OnRegionSelected = InArgs._OnRegionSelected;
	OnAddRegion = InArgs._OnAddRegion;
	OnRemoveRegion = InArgs._OnRemoveRegion;
}

void SEffectTimeline::SetEffectRegions(const TArray<FEffectRegion>& InRegions)
{
	EffectRegions = InRegions;
	Invalidate(EInvalidateWidget::Paint);
}

void SEffectTimeline::SetTimeParameters(double InDuration, float InZoomLevel, double InScrollPosition)
{
	Duration = InDuration;
	// Clamp ZoomLevel to a safe minimum to prevent division by zero
	ZoomLevel = FMath::Max(InZoomLevel, 0.001f);
	ScrollPosition = InScrollPosition;
	Invalidate(EInvalidateWidget::Paint);
}

void SEffectTimeline::SetSelectionRange(double InStart, double InEnd)
{
	SelectionStart = InStart;
	SelectionEnd = InEnd;
	Invalidate(EInvalidateWidget::Paint);
}

void SEffectTimeline::SelectRegion(int32 Index)
{
	SelectedRegionIndex = Index;
	Invalidate(EInvalidateWidget::Paint);
}

FVector2D SEffectTimeline::ComputeDesiredSize(float LayoutScaleMultiplier) const
{
	return FVector2D(400.0f, 50.0f);  // Compact height for effect timeline
}

void SEffectTimeline::GetVisibleTimeRange(double& OutStart, double& OutEnd) const
{
	if (Duration <= 0 || ZoomLevel <= 0)
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

double SEffectTimeline::PixelToTime(float X, float Width) const
{
	if (Width <= 0 || Duration <= 0) return 0.0;

	double VisibleStart, VisibleEnd;
	GetVisibleTimeRange(VisibleStart, VisibleEnd);
	double VisibleDuration = VisibleEnd - VisibleStart;

	return VisibleStart + (X / Width) * VisibleDuration;
}

float SEffectTimeline::TimeToPixel(double Time, float Width) const
{
	if (Duration <= 0) return 0.0f;

	double VisibleStart, VisibleEnd;
	GetVisibleTimeRange(VisibleStart, VisibleEnd);
	double VisibleDuration = VisibleEnd - VisibleStart;

	if (VisibleDuration <= 0) return 0.0f;

	return static_cast<float>(((Time - VisibleStart) / VisibleDuration) * Width);
}

int32 SEffectTimeline::GetRegionAtPosition(float X, float Width) const
{
	if (EffectRegions.Num() == 0) return -1;

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

bool SEffectTimeline::IsOverRegionStartHandle(int32 RegionIndex, float X, float Width) const
{
	if (!EffectRegions.IsValidIndex(RegionIndex)) return false;

	float HandleX = TimeToPixel(EffectRegions[RegionIndex].StartTime, Width);
	return FMath::Abs(X - HandleX) <= (HandleWidth + HandleHitPadding);
}

bool SEffectTimeline::IsOverRegionEndHandle(int32 RegionIndex, float X, float Width) const
{
	if (!EffectRegions.IsValidIndex(RegionIndex)) return false;

	float HandleX = TimeToPixel(EffectRegions[RegionIndex].EndTime, Width);
	return FMath::Abs(X - HandleX) <= (HandleWidth + HandleHitPadding);
}

void SEffectTimeline::ShowContextMenu(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(MouseEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;
	ContextMenuTime = PixelToTime(LocalPos.X, Width);

	// Check if clicking on an existing region
	int32 RegionIndex = GetRegionAtPosition(LocalPos.X, Width);

	FMenuBuilder MenuBuilder(true, nullptr);

	if (RegionIndex >= 0)
	{
		// Context menu for existing effect region
		SelectedRegionIndex = RegionIndex;
		const FEffectRegion& Region = EffectRegions[RegionIndex];

		MenuBuilder.BeginSection("EffectRegion", FText::FromString(Region.EffectName));

		MenuBuilder.AddMenuEntry(
			FText::FromString(TEXT("Remove Effect")),
			FText::FromString(TEXT("Delete this effect region")),
			FSlateIcon(),
			FUIAction(FExecuteAction::CreateLambda([this, RegionIndex]()
			{
				if (OnRemoveRegion.IsBound())
				{
					OnRemoveRegion.Execute(RegionIndex);
				}
			}))
		);

		MenuBuilder.EndSection();
	}
	else
	{
		// Context menu to add new effect regions
		MenuBuilder.BeginSection("AddEffect", FText::FromString(TEXT("Add Effect")));

		// Data-driven effect menu entries
		static const struct {
			const TCHAR* Name;
			const TCHAR* Tooltip;
		} EffectEntries[] = {
			{ TEXT("Vignette"), TEXT("Add vignette effect") },
			{ TEXT("Beat Flash"), TEXT("Add beat flash effect") },
			{ TEXT("Beat Zoom"), TEXT("Add beat zoom effect") },
			{ TEXT("Color Grade"), TEXT("Add color grading effect") },
			{ TEXT("Transition"), TEXT("Add transition effect") }
		};

		for (const auto& Entry : EffectEntries)
		{
			MenuBuilder.AddMenuEntry(
				FText::FromString(Entry.Name),
				FText::FromString(Entry.Tooltip),
				FSlateIcon(),
				FUIAction(FExecuteAction::CreateLambda([this, EffectName = FString(Entry.Name)]()
				{
					if (OnAddRegion.IsBound())
					{
						OnAddRegion.Execute(EffectName, ContextMenuTime);
					}
				}))
			);
		}

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

FReply SEffectTimeline::OnMouseButtonDown(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	// Duration check applies to all interactions
	if (Duration <= 0) return FReply::Unhandled();

	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(MouseEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;

	// Right-click context menu works even with no regions (to add new ones)
	if (MouseEvent.GetEffectingButton() == EKeys::RightMouseButton)
	{
		ShowContextMenu(MyGeometry, MouseEvent);
		return FReply::Handled();
	}

	// Left-click operations require existing regions
	if (EffectRegions.Num() == 0) return FReply::Unhandled();

	if (MouseEvent.GetEffectingButton() == EKeys::LeftMouseButton)
	{
		// Check handles first
		for (int32 i = 0; i < EffectRegions.Num(); ++i)
		{
			if (IsOverRegionStartHandle(i, LocalPos.X, Width))
			{
				CurrentDragMode = EDragMode::DragStart;
				DragRegionIndex = i;
				SelectedRegionIndex = i;
				LastMousePos = LocalPos;
				Invalidate(EInvalidateWidget::Paint);
				if (OnRegionSelected.IsBound())
				{
					OnRegionSelected.Execute(i);
				}
				return FReply::Handled().CaptureMouse(SharedThis(this));
			}
			else if (IsOverRegionEndHandle(i, LocalPos.X, Width))
			{
				CurrentDragMode = EDragMode::DragEnd;
				DragRegionIndex = i;
				SelectedRegionIndex = i;
				LastMousePos = LocalPos;
				Invalidate(EInvalidateWidget::Paint);
				if (OnRegionSelected.IsBound())
				{
					OnRegionSelected.Execute(i);
				}
				return FReply::Handled().CaptureMouse(SharedThis(this));
			}
		}

		// Check if clicking on a region body (to select or move)
		int32 RegionIndex = GetRegionAtPosition(LocalPos.X, Width);
		if (RegionIndex >= 0)
		{
			SelectedRegionIndex = RegionIndex;
			CurrentDragMode = EDragMode::DragMove;
			DragRegionIndex = RegionIndex;
			// Store offset from region start
			DragStartOffset = PixelToTime(LocalPos.X, Width) - EffectRegions[RegionIndex].StartTime;
			LastMousePos = LocalPos;
			Invalidate(EInvalidateWidget::Paint);
			if (OnRegionSelected.IsBound())
			{
				OnRegionSelected.Execute(RegionIndex);
			}
			return FReply::Handled().CaptureMouse(SharedThis(this));
		}

		// Clicked empty area - deselect
		SelectedRegionIndex = -1;
		Invalidate(EInvalidateWidget::Paint);
		if (OnRegionSelected.IsBound())
		{
			OnRegionSelected.Execute(-1);
		}
		return FReply::Handled();
	}

	return FReply::Unhandled();
}

FReply SEffectTimeline::OnMouseButtonUp(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	if (CurrentDragMode != EDragMode::None && DragRegionIndex >= 0)
	{
		if (EffectRegions.IsValidIndex(DragRegionIndex))
		{
			if (OnRegionChanged.IsBound())
			{
				OnRegionChanged.Execute(DragRegionIndex,
					EffectRegions[DragRegionIndex].StartTime,
					EffectRegions[DragRegionIndex].EndTime);
			}
		}

		CurrentDragMode = EDragMode::None;
		DragRegionIndex = -1;
		return FReply::Handled().ReleaseMouseCapture();
	}
	return FReply::Unhandled();
}

FReply SEffectTimeline::OnMouseMove(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent)
{
	if (EffectRegions.Num() == 0 || Duration <= 0) return FReply::Unhandled();

	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(MouseEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;

	if (CurrentDragMode == EDragMode::DragStart && EffectRegions.IsValidIndex(DragRegionIndex))
	{
		double NewTime = PixelToTime(LocalPos.X, Width);
		double EndTime = EffectRegions[DragRegionIndex].EndTime;
		double MinBound = FMath::Max(0.0, SelectionStart);
		NewTime = FMath::Clamp(NewTime, MinBound, EndTime - 0.1);

		// Need non-const access - fire delegate to update
		if (OnRegionChanged.IsBound())
		{
			OnRegionChanged.Execute(DragRegionIndex, NewTime, EndTime);
		}
		LastMousePos = LocalPos;
		return FReply::Handled();
	}
	else if (CurrentDragMode == EDragMode::DragEnd && EffectRegions.IsValidIndex(DragRegionIndex))
	{
		double NewTime = PixelToTime(LocalPos.X, Width);
		double StartTime = EffectRegions[DragRegionIndex].StartTime;
		double EffectiveSelEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;
		NewTime = FMath::Clamp(NewTime, StartTime + 0.1, EffectiveSelEnd);

		if (OnRegionChanged.IsBound())
		{
			OnRegionChanged.Execute(DragRegionIndex, StartTime, NewTime);
		}
		LastMousePos = LocalPos;
		return FReply::Handled();
	}
	else if (CurrentDragMode == EDragMode::DragMove && EffectRegions.IsValidIndex(DragRegionIndex))
	{
		double ClickTime = PixelToTime(LocalPos.X, Width);
		double NewStart = ClickTime - DragStartOffset;
		double RegionDuration = EffectRegions[DragRegionIndex].EndTime - EffectRegions[DragRegionIndex].StartTime;
		double NewEnd = NewStart + RegionDuration;

		// Clamp to selection bounds
		double MinBound = FMath::Max(0.0, SelectionStart);
		double EffectiveSelEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;

		// If the region is larger than the available selection range, snap to full range
		double AvailableRange = EffectiveSelEnd - MinBound;
		if (RegionDuration >= AvailableRange) {
			// Snap to available range
			NewStart = MinBound;
			NewEnd = EffectiveSelEnd;
		} else {
			if (NewStart < MinBound)
			{
				NewStart = MinBound;
				NewEnd = NewStart + RegionDuration;
			}
			if (NewEnd > EffectiveSelEnd)
			{
				NewEnd = EffectiveSelEnd;
				NewStart = NewEnd - RegionDuration;
			}

			// Final clamps to ensure values are within bounds
			NewStart = FMath::Clamp(NewStart, MinBound, EffectiveSelEnd - RegionDuration);
			NewEnd = FMath::Clamp(NewEnd, MinBound + RegionDuration, EffectiveSelEnd);
		}

		if (OnRegionChanged.IsBound())
		{
			OnRegionChanged.Execute(DragRegionIndex, NewStart, NewEnd);
		}
		LastMousePos = LocalPos;
		return FReply::Handled();
	}

	return FReply::Unhandled();
}

FCursorReply SEffectTimeline::OnCursorQuery(const FGeometry& MyGeometry, const FPointerEvent& CursorEvent) const
{
	if (EffectRegions.Num() == 0 || Duration <= 0) return FCursorReply::Unhandled();

	FVector2D LocalPos = MyGeometry.AbsoluteToLocal(CursorEvent.GetScreenSpacePosition());
	float Width = MyGeometry.GetLocalSize().X;

	for (int32 i = 0; i < EffectRegions.Num(); ++i)
	{
		if (IsOverRegionStartHandle(i, LocalPos.X, Width) || IsOverRegionEndHandle(i, LocalPos.X, Width))
		{
			return FCursorReply::Cursor(EMouseCursor::ResizeLeftRight);
		}
	}

	// Check if over region body
	int32 RegionIndex = GetRegionAtPosition(LocalPos.X, Width);
	if (RegionIndex >= 0)
	{
		return FCursorReply::Cursor(EMouseCursor::CardinalCross);  // Move cursor
	}

	return FCursorReply::Cursor(EMouseCursor::Default);
}

int32 SEffectTimeline::OnPaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry,
	const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements,
	int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const
{
	const bool bEnabled = ShouldBeEnabled(bParentEnabled);
	const ESlateDrawEffect DrawEffects = bEnabled ? ESlateDrawEffect::None : ESlateDrawEffect::DisabledEffect;

	const FVector2D LocalSize = AllottedGeometry.GetLocalSize();
	const float Width = LocalSize.X;
	const float Height = LocalSize.Y;

	if (Width <= 0 || Height <= 0) return LayerId;

	// Draw background
	FSlateDrawElement::MakeBox(
		OutDrawElements,
		LayerId,
		AllottedGeometry.ToPaintGeometry(),
		FCoreStyle::Get().GetBrush("GenericWhiteBox"),
		DrawEffects,
		BackgroundColor
	);
	LayerId++;

	// Draw "EFFECTS" label on left
	FSlateFontInfo LabelFont = FCoreStyle::Get().GetFontStyle("SmallFont");
	FSlateDrawElement::MakeText(
		OutDrawElements,
		LayerId,
		AllottedGeometry.ToPaintGeometry(
			FVector2D(60, Height),
			FSlateLayoutTransform(FVector2D(4, (Height - 12) / 2))
		),
		TEXT("EFFECTS"),
		LabelFont,
		DrawEffects,
		FLinearColor(0.5f, 0.5f, 0.6f, 0.8f)
	);
	LayerId++;

	// Draw selection range indicator
	if (Duration > 0)
	{
		double EffectiveSelEnd = (SelectionEnd < 0) ? Duration : SelectionEnd;
		float SelStartPx = TimeToPixel(SelectionStart, Width);
		float SelEndPx = TimeToPixel(EffectiveSelEnd, Width);

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
				FLinearColor(0.1f, 0.15f, 0.2f, 0.5f)  // Subtle selection highlight
			);
		}
	}
	LayerId++;

	// Draw effect regions
	if (EffectRegions.Num() > 0)
	{
		const float RegionTop = 4.0f;
		const float RegionHeight = Height - 8.0f;
		int MaxLayer = LayerId;
		for (int32 i = 0; i < EffectRegions.Num(); ++i)
		{
			const FEffectRegion& Region = EffectRegions[i];
			if (!Region.bEnabled) continue;

			float RegStartPx = TimeToPixel(Region.StartTime, Width);
			float RegEndPx = TimeToPixel(Region.EndTime, Width);

			RegStartPx = FMath::Max(0.0f, RegStartPx);
			RegEndPx = FMath::Min(Width, RegEndPx);

			if (RegEndPx <= RegStartPx) continue;

			// Region background
			FLinearColor RegionColor = Region.Color;
			if (i == SelectedRegionIndex)
			{
				RegionColor = FLinearColor::LerpUsingHSV(RegionColor, FLinearColor::White, 0.3f);
				RegionColor.A = 0.9f;
			}
			else
			{
				RegionColor.A = 0.7f;
			}

			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(RegEndPx - RegStartPx, RegionHeight),
					FSlateLayoutTransform(FVector2D(RegStartPx, RegionTop))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				RegionColor
			);
			MaxLayer = FMath::Max(MaxLayer, LayerId);

			// Region name
			if (RegEndPx - RegStartPx > 40)
			{
				FSlateDrawElement::MakeText(
					OutDrawElements,
					LayerId + 1,
					AllottedGeometry.ToPaintGeometry(
						FVector2D(RegEndPx - RegStartPx - 8, RegionHeight),
						FSlateLayoutTransform(FVector2D(RegStartPx + 4, RegionTop + 2))
					),
					Region.EffectName,
					LabelFont,
					DrawEffects,
					FLinearColor::White
				);
				MaxLayer = FMath::Max(MaxLayer, LayerId + 1);
			}

			// Handles
			FLinearColor HandleColor = (i == SelectedRegionIndex) ? FLinearColor::White : FLinearColor(0.9f, 0.9f, 0.9f, 0.8f);

			// Start handle
			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId + 2,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(HandleWidth, RegionHeight),
					FSlateLayoutTransform(FVector2D(RegStartPx - HandleWidth / 2, RegionTop))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				HandleColor
			);
			MaxLayer = FMath::Max(MaxLayer, LayerId + 2);

			// End handle
			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId + 2,
				AllottedGeometry.ToPaintGeometry(
					FVector2D(HandleWidth, RegionHeight),
					FSlateLayoutTransform(FVector2D(RegEndPx - HandleWidth / 2, RegionTop))
				),
				FCoreStyle::Get().GetBrush("GenericWhiteBox"),
				DrawEffects,
				HandleColor
			);
			MaxLayer = FMath::Max(MaxLayer, LayerId + 2);
		}
		LayerId = MaxLayer + 1;
	}

	// Draw border
	TArray<FVector2D> BorderPoints;
	BorderPoints.Add(FVector2D(0, 0));
	BorderPoints.Add(FVector2D(Width, 0));
	BorderPoints.Add(FVector2D(Width, Height));
	BorderPoints.Add(FVector2D(0, Height));
	BorderPoints.Add(FVector2D(0, 0));

	FSlateDrawElement::MakeLines(
		OutDrawElements,
		LayerId,
		AllottedGeometry.ToPaintGeometry(),
		BorderPoints,
		DrawEffects,
		FLinearColor(0.3f, 0.3f, 0.4f, 0.6f),
		true,
		1.0f
	);

	return LayerId + 1;
}
