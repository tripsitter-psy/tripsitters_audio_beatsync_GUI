#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "Engine/Texture2D.h"
#include "Blueprint/UserWidget.h"
#include "BeatsyncSubsystem.h"
#include "BeatSyncHUD.generated.h"

// Forward declaration
class UBeatsyncSubsystem;

// Button state struct
USTRUCT()
struct FUIButton
{
	GENERATED_BODY()

	FVector2D Position;
	FVector2D Size;
	FString Label;
	bool bIsHovered = false;
	bool bIsPressed = false;
};

// Slider state struct
USTRUCT()
struct FUISlider
{
	GENERATED_BODY()

	FVector2D Position;
	float Width = 200.0f;
	float Value = 0.5f;
	float MinValue = 0.0f;
	float MaxValue = 1.0f;
	FString Label;
	bool bIsDragging = false;
};

// Checkbox state struct
USTRUCT()
struct FUICheckbox
{
	GENERATED_BODY()

	FVector2D Position;
	FString Label;
	bool bIsChecked = false;
};

// Dropdown state struct
USTRUCT()
struct FUIDropdown
{
	GENERATED_BODY()

	FVector2D Position;
	float Width = 150.0f;
	TArray<FString> Options;
	int32 SelectedIndex = 0;
	bool bIsOpen = false;
	FString Label;
};

UCLASS()
class TRIPSITTERBEATSYNC_API ABeatSyncHUD : public AHUD
{
	GENERATED_BODY()

public:
	ABeatSyncHUD();

	virtual void BeginPlay() override;
	virtual void DrawHUD() override;

	UPROPERTY(EditDefaultsOnly, Category = "UI")
	UTexture2D* BackgroundTexture;

	UPROPERTY(EditDefaultsOnly, Category = "UI")
	UTexture2D* HeaderTexture;

	UPROPERTY(EditDefaultsOnly, Category = "UI")
	TSubclassOf<UUserWidget> MainWidgetClass;

	UPROPERTY(EditDefaultsOnly, Category = "UI")
	UFont* CustomFont;

	UPROPERTY(EditDefaultsOnly, Category = "UI")
	UFont* CustomFontLarge;

	// Scroll handling
	void ScrollUp();
	void ScrollDown();

private:
	UPROPERTY()
	UUserWidget* MainWidget;

	// Scroll state
	float ScrollOffset = 0.0f;
	float MaxScrollOffset = 800.0f;
	float ScrollSpeed = 50.0f;
	float ContentHeight = 1200.0f;

	// Mouse state
	FVector2D MousePosition;
	bool bMousePressed = false;
	bool bMouseJustPressed = false;
	bool bMouseJustReleased = false;
	bool bPrevMousePressed = false;

	// UI Colors (psychedelic theme)
	FLinearColor CyanColor = FLinearColor(0.0f, 0.85f, 1.0f, 1.0f);
	FLinearColor PurpleColor = FLinearColor(0.545f, 0.0f, 1.0f, 1.0f);
	FLinearColor PinkColor = FLinearColor(1.0f, 0.0f, 0.5f, 1.0f);
	FLinearColor OrangeColor = FLinearColor(1.0f, 0.533f, 0.0f, 1.0f);
	FLinearColor DarkBG = FLinearColor(0.02f, 0.02f, 0.04f, 0.9f);
	FLinearColor ButtonBG = FLinearColor(0.08f, 0.08f, 0.12f, 1.0f);
	FLinearColor ButtonHover = FLinearColor(0.12f, 0.12f, 0.18f, 1.0f);
	FLinearColor SliderBG = FLinearColor(0.05f, 0.05f, 0.08f, 1.0f);

	// === INPUT PANEL STATE ===
	FString AudioFilePath = TEXT("No file selected");
	FString VideoFilePath = TEXT("No file selected");
	FString StatusMessage = TEXT("");
	float StatusMessageTimer = 0.0f;

	// === ANALYSIS SETTINGS STATE ===
	float BPMValue = 120.0f;
	float SensitivityValue = 0.5f;
	int32 BeatDetectionMode = 0; // 0=Auto, 1=Manual, 2=Tap
	bool bAnalyzeOnLoad = true;

	// === VIDEO SETTINGS STATE ===
	int32 ResolutionIndex = 0; // 0=1080p, 1=720p, 2=4K
	int32 FramerateIndex = 0; // 0=30fps, 1=60fps, 2=24fps
	bool bMaintainAspect = true;

	// === VISUAL EFFECTS STATE ===
	float BrightnessValue = 1.0f;
	float ContrastValue = 1.0f;
	float SaturationValue = 1.0f;
	float VignetteValue = 0.0f;
	bool bEnableGlow = false;
	bool bEnableChromatic = false;

	// === BEAT EFFECTS STATE ===
	int32 BeatEffectType = 0; // 0=Flash, 1=Zoom, 2=Shake, 3=Color
	float BeatIntensity = 0.7f;
	bool bSyncToKick = true;
	bool bSyncToSnare = false;

	// === TRANSITIONS STATE ===
	int32 TransitionType = 0; // 0=Cut, 1=Fade, 2=Wipe, 3=Dissolve
	float TransitionDuration = 0.5f;
	bool bAutoTransitions = true;

	// === OUTPUT STATE ===
	FString OutputPath = TEXT("~/Desktop/output.mp4");
	int32 OutputFormat = 0; // 0=MP4, 1=MOV, 2=AVI

	// === PROGRESS STATE ===
	float ProcessingProgress = 0.0f;
	bool bIsProcessing = false;

	// === WAVEFORM STATE ===
	float WaveformZoom = 1.0f;           // 1.0 = full view, higher = more zoomed
	float WaveformScrollX = 0.0f;        // Horizontal scroll position (0-1)
	float VideoStartTime = 0.0f;         // Video selection start (normalized 0-1)
	float VideoEndTime = 1.0f;           // Video selection end (normalized 0-1)
	float EffectStartTime = 0.3f;        // Effect region start (normalized 0-1)
	float EffectEndTime = 0.7f;          // Effect region end (normalized 0-1)
	int32 DraggingHandle = 0;            // 0=none, 1=video start, 2=video end, 3=effect start, 4=effect end
	float TotalAudioDuration = 180.0f;   // Total duration in seconds (3 min placeholder)

	// Waveform panel bounds for scroll detection
	FVector2D WaveformPanelMin = FVector2D::ZeroVector;
	FVector2D WaveformPanelMax = FVector2D::ZeroVector;

	// Drawing helpers
	void DrawTitle(float InScrollOffset = 0.0f);
	void DrawPanel(float X, float Y, float Width, float Height, const FString& Title, FLinearColor TitleColor);
	void DrawScrollbar();
	void DrawAllPanels();

	// Control drawing methods
	bool DrawButton(float X, float Y, float Width, float Height, const FString& Label, FLinearColor AccentColor);
	float DrawSlider(float X, float Y, float Width, const FString& Label, float Value, float MinVal, float MaxVal, FLinearColor AccentColor);
	bool DrawCheckbox(float X, float Y, const FString& Label, bool bChecked, FLinearColor AccentColor);
	int32 DrawDropdown(float X, float Y, float Width, const FString& Label, const TArray<FString>& Options, int32 SelectedIdx, FLinearColor AccentColor);
	void DrawLabel(float X, float Y, const FString& Text, FLinearColor Color = FLinearColor::White);
	void DrawProgressBar(float X, float Y, float Width, float Height, float Progress, FLinearColor Color);

	// Panel content drawing
	void DrawInputPanel(float X, float Y, float Width);
	void DrawAnalysisPanel(float X, float Y, float Width);
	void DrawVideoSettingsPanel(float X, float Y, float Width);
	void DrawOutputPanel(float X, float Y, float Width);
	void DrawEffectsPanel(float X, float Y, float Width);
	void DrawBeatEffectsPanel(float X, float Y, float Width);
	void DrawTransitionsPanel(float X, float Y, float Width);
	void DrawProgressPanel(float X, float Y, float Width);
	void DrawVisualizerPanel(float X, float Y, float Width);

	// Utility
	bool IsPointInRect(FVector2D Point, FVector2D RectPos, FVector2D RectSize);
	void UpdateMouseState();

	// File dialogs
	FString OpenFileDialog(const FString& Title, const FString& DefaultPath, const FString& FileTypes);
	FString SaveFileDialog(const FString& Title, const FString& DefaultPath, const FString& DefaultFile, const FString& FileTypes);

	// Active slider tracking
	int32 ActiveSliderID = -1;
	float* ActiveSliderValuePtr = nullptr;

	// === AUDIO ANALYSIS STATE ===
	bool bHasRealWaveform = false;
	bool bIsAnalyzing = false;
	FString AnalysisStatus = TEXT("");

	// Cached waveform data for drawing (from subsystem)
	TArray<FWaveformSample> CachedWaveform;
	TArray<float> CachedBeatTimestamps;
	float CachedBPM = 120.0f;
	float CachedDuration = 180.0f;

	// Get the beatsync subsystem
	UBeatsyncSubsystem* GetBeatsyncSubsystem() const;

	// Trigger audio analysis
	void AnalyzeSelectedAudio();

	// Callbacks for analysis events
	UFUNCTION()
	void OnAnalysisComplete();

	UFUNCTION()
	void OnAnalysisError(const FString& Error);

	UFUNCTION()
	void OnAnalysisProgress(float Progress);
};
