#pragma once

#include "CoreMinimal.h"
#include "Widgets/SLeafWidget.h"
#include "Widgets/DeclarativeSyntaxSupport.h"

// Forward declare effect region from SWaveformViewer
struct FEffectRegion;

DECLARE_DELEGATE_ThreeParams(FOnEffectTimelineRegionChanged, int32 /*RegionIndex*/, double /*Start*/, double /*End*/);
DECLARE_DELEGATE_OneParam(FOnEffectTimelineRegionSelected, int32 /*RegionIndex*/);
DECLARE_DELEGATE_TwoParams(FOnEffectTimelineAddRegion, const FString& /*EffectName*/, double /*Time*/);
DECLARE_DELEGATE_OneParam(FOnEffectTimelineRemoveRegion, int32 /*RegionIndex*/);

/**
 * Separate effect timeline widget - shows effect regions in a compact horizontal strip
 * Syncs with waveform viewer zoom/scroll for aligned time display
 */
class SEffectTimeline : public SLeafWidget
{
public:
	SLATE_BEGIN_ARGS(SEffectTimeline)
		: _BackgroundColor(FLinearColor(0.04f, 0.04f, 0.08f, 0.95f))
	{}
		SLATE_ARGUMENT(FLinearColor, BackgroundColor)
		SLATE_EVENT(FOnEffectTimelineRegionChanged, OnRegionChanged)
		SLATE_EVENT(FOnEffectTimelineRegionSelected, OnRegionSelected)
		SLATE_EVENT(FOnEffectTimelineAddRegion, OnAddRegion)
		SLATE_EVENT(FOnEffectTimelineRemoveRegion, OnRemoveRegion)
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

	// Set effect regions data (reference to waveform viewer's effect regions)
	void SetEffectRegions(const TArray<FEffectRegion>* InRegions);

	// Set time parameters to sync with waveform viewer
	void SetTimeParameters(double InDuration, float InZoomLevel, double InScrollPosition);

	// Set selection range (for visual reference)
	void SetSelectionRange(double InStart, double InEnd);

	// Select an effect region
	void SelectRegion(int32 Index);
	int32 GetSelectedRegion() const { return SelectedRegionIndex; }

	// SWidget interface
	virtual int32 OnPaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry,
		const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements,
		int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const override;

	virtual FVector2D ComputeDesiredSize(float LayoutScaleMultiplier) const override;

	// Mouse interaction
	virtual FReply OnMouseButtonDown(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
	virtual FReply OnMouseButtonUp(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
	virtual FReply OnMouseMove(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
	virtual FCursorReply OnCursorQuery(const FGeometry& MyGeometry, const FPointerEvent& CursorEvent) const override;

protected:
	// Effect region data (not owned - reference to waveform viewer's data)
	const TArray<FEffectRegion>* EffectRegions = nullptr;

	// Time parameters (synced with waveform viewer)
	double Duration = 0.0;
	float ZoomLevel = 1.0f;
	double ScrollPosition = 0.0;
	double SelectionStart = 0.0;
	double SelectionEnd = -1.0;

	// Selection state
	int32 SelectedRegionIndex = -1;

	// Drag state
	enum class EDragMode { None, DragStart, DragEnd, DragMove };
	EDragMode CurrentDragMode = EDragMode::None;
	int32 DragRegionIndex = -1;
	double DragStartOffset = 0.0;
	FVector2D LastMousePos = FVector2D::ZeroVector;

	// Visual settings
	FLinearColor BackgroundColor;
	static constexpr float HandleWidth = 6.0f;
	static constexpr float HandleHitPadding = 4.0f;

	// Delegates
	FOnEffectTimelineRegionChanged OnRegionChanged;
	FOnEffectTimelineRegionSelected OnRegionSelected;
	FOnEffectTimelineAddRegion OnAddRegion;
	FOnEffectTimelineRemoveRegion OnRemoveRegion;

	// Context menu position
	double ContextMenuTime = 0.0;

	// Helper functions
	double PixelToTime(float X, float Width) const;
	float TimeToPixel(double Time, float Width) const;
	void GetVisibleTimeRange(double& OutStart, double& OutEnd) const;

	// Hit testing
	int32 GetRegionAtPosition(float X, float Width) const;
	bool IsOverRegionStartHandle(int32 RegionIndex, float X, float Width) const;
	bool IsOverRegionEndHandle(int32 RegionIndex, float X, float Width) const;

	// Context menu
	void ShowContextMenu(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent);
};
