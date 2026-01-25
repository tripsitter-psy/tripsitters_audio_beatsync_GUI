// TripSitter Main Widget - Engine Program version

#pragma once

#include "CoreMinimal.h"
#include "Fonts/SlateFontInfo.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/DeclarativeSyntaxSupport.h"
#include "Styling/SlateBrush.h"
#include "Brushes/SlateImageBrush.h"
#include "Brushes/SlateDynamicImageBrush.h"
#include "Async/AsyncWork.h"
#include "BeatsyncProcessingTask.h"
#include "SWaveformViewer.h"

class SEditableTextBox;
class SProgressBar;
class STextBlock;
class SCheckBox;
class SSlider;
class SImage;
class SEffectTimeline;
template<typename T> class SComboBox;
template<typename T> class SSpinBox;

// Configuration enums
enum class EBeatRate : uint8
{
	Every = 0,
	Every2nd = 1,
	Every4th = 2,
	Every8th = 3
};

enum class EAnalysisMode : uint8
{
	Energy = 0,
	AIBeat = 1,
	AIStems = 2,
	AudioFlux = 3,
	StemsFlux = 4
};

enum class EResolution : uint8
{
	HD1080 = 0,
	HD720 = 1,
	UHD4K = 2,
	QHD2K = 3
};

enum class EFPS : uint8
{
	FPS24 = 0,
	FPS30 = 1,
	FPS60 = 2
};

// Effect types that can be mapped to stems
enum class EStemEffect : uint8
{
	None = 0,
	Flash = 1,
	Zoom = 2,
	Vignette = 3,
	ColorGrade = 4
};


// Stem configuration for effect mapping
struct FStemConfig
{
	FString FilePath;
	EStemEffect Effect = EStemEffect::None;
	TArray<double> BeatTimes;
	bool bEnabled = false;
};

// Number of stems (should match EStemType::Count)

constexpr int32 STEM_COUNT = static_cast<int32>(EStemType::Count);

/**
 * TripSitter Main Widget - Slate implementation
 */
class STripSitterMainWidget : public SCompoundWidget
{
public:
	SLATE_BEGIN_ARGS(STripSitterMainWidget) {}
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

private:
	// Load theme assets (fonts, images)
	void LoadAssets();

	// Theme colors
	FLinearColor NeonCyan = FLinearColor(0.0f, 0.851f, 1.0f);
	FLinearColor NeonPurple = FLinearColor(0.545f, 0.0f, 1.0f);
	FLinearColor DarkBg = FLinearColor(0.039f, 0.039f, 0.102f);
	FLinearColor ControlBg = FLinearColor(0.078f, 0.078f, 0.157f);
	FLinearColor TextColor = FLinearColor(0.784f, 0.863f, 1.0f);
	FLinearColor HotPink = FLinearColor(1.0f, 0.0f, 0.502f);
	FLinearColor NeonGreen = FLinearColor(0.0f, 1.0f, 0.392f);

	// Theme fonts (Corpta custom font)
	FSlateFontInfo TitleFont;
	FSlateFontInfo HeadingFont;
	FSlateFontInfo ButtonFont;
	FSlateFontInfo ButtonFontSmall;
	FSlateFontInfo BodyFont;
	bool bCustomFontLoaded = false;

	// Theme brushes
	FSlateBrush WallpaperBrush;
	FSlateBrush TitleBrush;
	TSharedPtr<FSlateDynamicImageBrush> WallpaperImageBrush;
	TSharedPtr<FSlateDynamicImageBrush> TitleImageBrush;

	// File paths
	FString AudioPath;
	FString VideoPath;
	TArray<FString> VideoPaths;
	FString OutputPath;
	bool bIsMultiClip = false;

	// Processing state
	float Progress = 0.0f;
	FString StatusText = TEXT("Ready");
	FString ETAText = TEXT("");
	bool bIsProcessing = false;
	bool bAudioAnalyzed = false;
	double DetectedBPM = 0.0;
	double OriginalFirstBeatTime = 0.0;  // Anchor point for BPM recalculation
	TArray<double> AnalyzedBeatTimes;

	// Async processing task
	TUniquePtr<FAsyncTask<FBeatsyncProcessingTask>> ProcessingTask;

	// Configuration
	EBeatRate BeatRate = EBeatRate::Every;
	EAnalysisMode AnalysisMode = EAnalysisMode::AIBeat;
	EResolution Resolution = EResolution::HD1080;
	EFPS FPS = EFPS::FPS30;

	// Effects config
	bool bEnableVignette = false;
	bool bEnableBeatFlash = false;
	bool bEnableBeatZoom = false;
	bool bEnableColorGrade = false;
	bool bEnableTransitions = false;
	float FlashIntensity = 0.5f;
	float ZoomIntensity = 0.1f;
	float VignetteStrength = 0.3f;
	float TransitionDuration = 0.5f;
	EColorPreset ColorPreset = EColorPreset::Warm;
	ETransitionType TransitionType = ETransitionType::Fade;

	// Stem configurations (Kick, Snare, HiHat, Synth)
	TStaticArray<FStemConfig, STEM_COUNT> StemConfigs;
	TArray<TSharedPtr<FString>> StemEffectOptions;

	// Preview state
	double PreviewTimestamp = 0.0;
	int32 PreviewWidth = 0;
	int32 PreviewHeight = 0;
	TArray<uint8> PreviewPixelData;
	TSharedPtr<FSlateDynamicImageBrush> PreviewImageBrush;
	FSlateBrush PreviewBrush;

	// Selection range
	double SelectionStart = 0.0;
	double SelectionEnd = -1.0;
	double AudioDuration = 0.0;

	// UI Elements
	TSharedPtr<SEditableTextBox> AudioPathBox;
	TSharedPtr<SEditableTextBox> VideoPathBox;
	TSharedPtr<SEditableTextBox> OutputPathBox;
	TSharedPtr<SSpinBox<double>> BPMSpinBox;
	TSharedPtr<SProgressBar> ProgressBar;
	TSharedPtr<STextBlock> StatusTextBlock;
	TSharedPtr<STextBlock> ETATextBlock;
	TSharedPtr<STextBlock> BPMTextBlock;
	TSharedPtr<SWaveformViewer> WaveformViewer;
	TSharedPtr<SEffectTimeline> EffectTimeline;
	TSharedPtr<SImage> PreviewImage;

	// Dropdown options
	TArray<TSharedPtr<FString>> BeatRateOptions;
	TArray<TSharedPtr<FString>> AnalysisModeOptions;
	TArray<TSharedPtr<FString>> ResolutionOptions;
	TArray<TSharedPtr<FString>> FPSOptions;
	TArray<TSharedPtr<FString>> ColorPresetOptions;
	TArray<TSharedPtr<FString>> TransitionOptions;

	// Selected indices
	int32 BeatRateIndex = 0;
	int32 AnalysisModeIndex = 1;
	int32 ResolutionIndex = 0;
	int32 FPSIndex = 1;


	// Button handlers
	FReply OnBrowseAudioClicked();
	FReply OnBrowseVideoClicked();
	FReply OnBrowseVideoFolderClicked();
	FReply OnBrowseOutputClicked();
	FReply OnStartSyncClicked();
	FReply OnCancelClicked();
	FReply OnAnalyzeAudioClicked();
	FReply OnApplyBeatMarkersClicked();
	FReply OnPreviewFrameClicked();

	// BPM adjustment
	void RecalculateBeatsFromBPM(double NewBPM);
	FReply OnBPMHalfClicked();
	FReply OnBPMDoubleClicked();
	void OnBPMValueChanged(double NewValue);

	// Preview texture update
	void UpdatePreviewTexture(const TArray<uint8>& RGBData, int32 Width, int32 Height);

	// UI section builders
	TSharedRef<SWidget> CreateFileSection();
	TSharedRef<SWidget> CreateWaveformSection();
	TSharedRef<SWidget> CreateAnalysisSection();
	TSharedRef<SWidget> CreateEffectsSection();
	TSharedRef<SWidget> CreateTransitionsSection();
	TSharedRef<SWidget> CreateControlSection();
	TSharedRef<SWidget> CreateStemsSection();

	// Helper to scan folder for videos
	void ScanFolderForVideos(const FString& FolderPath);

	// Load waveform data from audio file
	void LoadWaveformFromAudio(const FString& FilePath);

	// Stem file handling
	FReply OnBrowseStemClicked(int32 StemIndex);
	void AnalyzeStemFile(int32 StemIndex);
	void UpdateStemBeatsInWaveform();

	// Processing callbacks
	void OnProcessingProgress(float InProgress, const FString& Status);
	void OnProcessingComplete(const FBeatsyncProcessingResult& Result);
};
