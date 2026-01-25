#pragma once

#include "CoreMinimal.h"
#include "Widgets/SLeafWidget.h"
#include "Widgets/DeclarativeSyntaxSupport.h"

DECLARE_DELEGATE_TwoParams(FOnSelectionChanged, double /*Start*/, double /*End*/);
DECLARE_DELEGATE_ThreeParams(FOnEffectRegionChanged, int32 /*RegionIndex*/, double /*Start*/, double /*End*/);
DECLARE_DELEGATE_OneParam(FOnBeatTimesChanged, const TArray<double>& /*BeatTimes*/);

// Stem types for beat-synced effects
enum class EStemType : uint8
{
	Kick = 0,    // Kick drum - Black
	Snare = 1,   // Snare drum - Blue
	HiHat = 2,   // Open hi-hat - Yellow
	Synth = 3,   // Synth hit - Red
	Count = 4
};

// Stem beat data
struct FStemBeats
{
	EStemType Type = EStemType::Kick;
	FString Name;
	FLinearColor Color;
	TArray<double> BeatTimes;
	bool bEnabled = true;
};

// Effect region for applying effects to specific time ranges
struct FEffectRegion
{
    FGuid Id = FGuid::NewGuid();
	FString EffectName;
	double StartTime = 0.0;
	double EndTime = 0.0;
	FLinearColor Color = FLinearColor(0.5f, 0.0f, 1.0f, 0.3f); // Purple tint
	bool bEnabled = true;

    bool operator==(const FEffectRegion& Other) const
    {
        return Id == Other.Id;
    }
};

/**
 * Waveform visualization widget - displays audio peaks and beat markers
 * Matches the psychedelic theme from the wxWidgets version
 * Supports zoom and selection handles for choosing audio ranges
 * Right-click to add effect regions with in/out points
 */
class SWaveformViewer : public SLeafWidget
{
public:
	SLATE_BEGIN_ARGS(SWaveformViewer)
		: _WaveformColor(FLinearColor(0.0f, 0.851f, 1.0f)) // Neon Cyan
		, _BeatMarkerColor(FLinearColor(1.0f, 1.0f, 0.4f)) // Yellow
		, _SelectionColor(FLinearColor(0.0f, 0.2f, 0.3f))  // Dark cyan tint
		, _HandleColor(FLinearColor(1.0f, 0.0f, 0.5f))     // Hot pink
		, _EffectRegionColor(FLinearColor(0.5f, 0.0f, 1.0f, 0.3f)) // Purple
	{}
		SLATE_ARGUMENT(FLinearColor, WaveformColor)
		SLATE_ARGUMENT(FLinearColor, BeatMarkerColor)
		SLATE_ARGUMENT(FLinearColor, SelectionColor)
		SLATE_ARGUMENT(FLinearColor, HandleColor)
		SLATE_ARGUMENT(FLinearColor, EffectRegionColor)
		SLATE_EVENT(FOnSelectionChanged, OnSelectionChanged)
		SLATE_EVENT(FOnEffectRegionChanged, OnEffectRegionChanged)
		SLATE_EVENT(FOnBeatTimesChanged, OnBeatTimesChanged)
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

	// Set waveform data (peak values 0.0-1.0) - single color mode
	void SetWaveformData(const TArray<float>& InPeaks, double InDuration);

	// Set frequency-band waveform data (Rekordbox/Traktor style)
	// Bass: Red (20-200 Hz), Mids: Cyan (200-2000 Hz), Highs: White (2000+ Hz)
	void SetWaveformBands(const TArray<float>& InBassPeaks, const TArray<float>& InMidPeaks,
	                      const TArray<float>& InHighPeaks, double InDuration);

	// Check if we have frequency band data
	bool HasBandData() const { return BassPeaks.Num() > 0; }

	// Set beat times in seconds
	void SetBeatTimes(const TArray<double>& InBeatTimes);

	// Manual beat editing (main beat track)
	void AddBeatAtTime(double Time);
	void RemoveBeatAtTime(double Time, double Tolerance = 0.05);
	void RemoveBeatAtIndex(int32 Index);
	const TArray<double>& GetBeatTimes() const { return BeatTimes; }

    // Effect region management
    // Removed duplicate RemoveEffectRegion(int32 Index);
    void RemoveEffectRegionById(const FGuid& Id);

	// Stem beat management
	void SetStemBeatTimes(EStemType Stem, const TArray<double>& InBeatTimes);
	void ClearStemBeats(EStemType Stem);
	void ClearAllStemBeats();
	const TArray<double>& GetStemBeatTimes(EStemType Stem) const;
	void SetStemEnabled(EStemType Stem, bool bEnabled);
	bool IsStemEnabled(EStemType Stem) const;
	static FString GetStemName(EStemType Stem);
	static FLinearColor GetStemColor(EStemType Stem);

	// Selection range (in seconds)
	void SetSelectionRange(double InStart, double InEnd);
	double GetSelectionStart() const { return SelectionStart; }
	double GetSelectionEnd() const { return SelectionEnd; }
	double GetDuration() const { return Duration; }

	// Zoom controls
	void ZoomIn();
	void ZoomOut();
	void ZoomToFit();
	void SetZoomLevel(float Level);
	float GetZoomLevel() const { return ZoomLevel; }

	// Scroll/Pan position (0.0 = start, 1.0 = end when zoomed)
	void SetScrollPosition(double Position);
	double GetScrollPosition() const { return ScrollPosition; }

	// Effect regions - areas where specific effects are applied
	int32 AddEffectRegion(const FString& EffectName, double StartTime, double EndTime, FLinearColor Color);
	void RemoveEffectRegion(int32 Index);
	void ClearEffectRegions();
	void SetEffectRegionRange(int32 Index, double StartTime, double EndTime);
	const TArray<FEffectRegion>& GetEffectRegions() const { return EffectRegions; }
	int32 GetSelectedEffectRegion() const { return SelectedEffectRegion; }

	// Clear all data
	void Clear();

	// SWidget interface
	virtual int32 OnPaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry,
		const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements,
		int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const override;

	virtual FVector2D ComputeDesiredSize(float LayoutScaleMultiplier) const override;

	// Mouse input for selection handles and panning
	virtual FReply OnMouseButtonDown(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
	virtual FReply OnMouseButtonUp(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
	virtual FReply OnMouseMove(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
	virtual FReply OnMouseWheel(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent) override;
	virtual FCursorReply OnCursorQuery(const FGeometry& MyGeometry, const FPointerEvent& CursorEvent) const override;

protected:
	// Data - single color waveform
	TArray<float> WaveformPeaks;
	// Data - frequency band waveform (Rekordbox/Traktor style)
	TArray<float> BassPeaks;   // Low frequencies (20-200 Hz) - Red
	TArray<float> MidPeaks;    // Mid frequencies (200-2000 Hz) - Cyan/Blue
	TArray<float> HighPeaks;   // High frequencies (2000+ Hz) - White

	TArray<double> BeatTimes;        // Main beat track (yellow)
	FStemBeats StemBeats[4];         // Stem beat tracks (Kick, Snare, HiHat, Synth)
	double Duration = 0.0;
	double SelectionStart = 0.0;
	double SelectionEnd = -1.0; // -1 means full track

	// Zoom and scroll
	float ZoomLevel = 1.0f;        // 1.0 = fit all, 2.0 = 2x zoom, etc.
	double ScrollPosition = 0.0;  // 0.0-1.0 normalized scroll position

	// Colors
	FLinearColor WaveformColor;
	FLinearColor BeatMarkerColor;
	FLinearColor SelectionColor;
	FLinearColor HandleColor;
	FLinearColor EffectRegionColor;

	// Effect regions
	TArray<FEffectRegion> EffectRegions;
	int32 SelectedEffectRegion = -1;

	// Selection handle interaction
	enum class EDragMode { None, DragStart, DragEnd, Pan, DragEffectStart, DragEffectEnd, DragBeat };
	EDragMode CurrentDragMode = EDragMode::None;
	double DragStartTime = 0.0;
	FVector2D LastMousePos = FVector2D::ZeroVector;
	int32 DragEffectIndex = -1;
	int32 DragBeatIndex = -1;

	// Handle hit testing
	static constexpr float HandleWidth = 10.0f;
	static constexpr float HandleHitPadding = 5.0f;

	// Selection changed delegate
	FOnSelectionChanged OnSelectionChanged;
	FOnEffectRegionChanged OnEffectRegionChanged;
	FOnBeatTimesChanged OnBeatTimesChanged;

	// Context menu position
	double ContextMenuTime = 0.0;

	// Helper functions
	double PixelToTime(float X, float Width) const;
	float TimeToPixel(double Time, float Width) const;

	// Get visible time range based on zoom/scroll
	void GetVisibleTimeRange(double& OutStart, double& OutEnd) const;

	// Check if mouse is over a selection handle
	bool IsOverStartHandle(float X, float Width) const;
	bool IsOverEndHandle(float X, float Width) const;

	// Find beat marker near a pixel position (returns index or -1)
	int32 FindBeatNearPixel(float X, float Width, float TolerancePixels = 10.0f) const;

	// Check if mouse is over an effect region handle
	int32 GetEffectRegionAtPosition(float X, float Width) const;
	bool IsOverEffectStartHandle(int32 RegionIndex, float X, float Width) const;
	bool IsOverEffectEndHandle(int32 RegionIndex, float X, float Width) const;

	// Show context menu for beat markers (right-click on waveform)
	void ShowBeatContextMenu(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent);

	// Show context menu for adding effects (used by effect timeline)
	void ShowEffectContextMenu(const FGeometry& MyGeometry, const FPointerEvent& MouseEvent);

	// Clamp all effect regions to stay within current selection bounds
	void ClampEffectRegionsToSelection();
};
