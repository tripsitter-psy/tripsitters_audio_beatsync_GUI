

// TripSitter Main Widget - Engine Program version
// Source: BeatSyncEditor/unreal-prototype/Source/TripSitter/Private/STripSitterMainWidget.h
// Sync script: BeatSyncEditor/scripts/sync_tripsitter_ue.ps1
// NOTE: Edit here directly for quick iteration, then sync back to repo with -ToRepo flag

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

// ...existing code...

class SEditableTextBox;
class SProgressBar;
class STextBlock;
class SCheckBox;
class SSlider;
class SImage;
class SWaveformViewer;
template<typename T> class SComboBox;

/**
 * TripSitter Main Widget - Slate implementation
 * Replicates the wxWidgets GUI in Unreal Engine
 */

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
    AIStems = 2
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

class STripSitterMainWidget : public SCompoundWidget
{
public:
	SLATE_BEGIN_ARGS(STripSitterMainWidget) {}
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

private:
	// File paths
	FString AudioPath;
	FString VideoPath;           // Single video path OR folder path
	TArray<FString> VideoPaths;  // Multiple video paths when folder is selected
	FString OutputPath;
	bool bIsMultiClip = false;   // True when a folder with multiple videos is selected

	// Processing config
	EBeatRate BeatRate = EBeatRate::Every;        // Every, Every 2nd, Every 4th, Every 8th
	EAnalysisMode AnalysisMode = EAnalysisMode::Energy;    // Energy, AI Beat, AI+Stems
	EResolution Resolution = EResolution::HD1080;      // 1080p, 720p, 4K, 2K
	EFPS FPS = EFPS::FPS30;             // 24, 30, 60

	// Effects
	bool bEnableVignette = false;
	bool bEnableBeatFlash = false;
	bool bEnableBeatZoom = false;
	bool bEnableColorGrade = false;
	float FlashIntensity = 0.5f;
	float ZoomIntensity = 0.5f;
	int32 ColorPreset = 0;

	// Transitions
	bool bEnableTransitions = false;
	int32 TransitionType = 0;
	float TransitionDuration = 0.5f;

	// Preview
	bool bPreviewMode = false;
	int32 PreviewBeats = 8;
	float PreviewTimestamp = 0.0f;

	// Progress
	float Progress = 0.0f;
	FString StatusText = TEXT("Ready");
	FString ETAText = TEXT("");
	bool bIsProcessing = false;

	// UI Elements
	TSharedPtr<SEditableTextBox> AudioPathBox;
	TSharedPtr<SEditableTextBox> VideoPathBox;
	TSharedPtr<SEditableTextBox> OutputPathBox;
	TSharedPtr<SProgressBar> ProgressBar;
	TSharedPtr<STextBlock> StatusTextBlock;
	TSharedPtr<STextBlock> ETATextBlock;
	TSharedPtr<SWaveformViewer> WaveformViewer;

	// Colors (Psychedelic theme - from PsychedelicTheme.h)
	// Primary: Neon Cyan (0, 217, 255)
	FLinearColor NeonCyan = FLinearColor(0.0f, 0.851f, 1.0f);
	// Secondary: Neon Purple (139, 0, 255)
	FLinearColor NeonPurple = FLinearColor(0.545f, 0.0f, 1.0f);
	// Background: Dark Blue-Black (10, 10, 26)
	FLinearColor DarkBg = FLinearColor(0.039f, 0.039f, 0.102f);
	// Surface: Dark Gray-Blue (20, 20, 40)
	FLinearColor ControlBg = FLinearColor(0.078f, 0.078f, 0.157f);
	// Text: Light Blue-White (200, 220, 255)
	FLinearColor TextColor = FLinearColor(0.784f, 0.863f, 1.0f);
	// Accent: Hot Pink (255, 0, 128)
	FLinearColor HotPink = FLinearColor(1.0f, 0.0f, 0.502f);
	// Success: Neon Green (0, 255, 100)
	FLinearColor NeonGreen = FLinearColor(0.0f, 1.0f, 0.392f);

	// Button handlers
	FReply OnBrowseAudioClicked();
	FReply OnBrowseVideoClicked();
	FReply OnBrowseVideoFolderClicked();
	FReply OnBrowseOutputClicked();
	FReply OnStartSyncClicked();
	FReply OnCancelClicked();
	FReply OnPreviewFrameClicked();

	// Helper to scan folder for video files
	void ScanFolderForVideos(const FString& FolderPath);

	// Helper to create styled section
	TSharedRef<SWidget> CreateFileSection();
	TSharedRef<SWidget> CreateWaveformSection();
	TSharedRef<SWidget> CreateAnalysisSection();
	TSharedRef<SWidget> CreateEffectsSection();
	TSharedRef<SWidget> CreateTransitionsSection();
	TSharedRef<SWidget> CreateControlSection();

	// Load waveform data from audio file
	void LoadWaveformFromAudio(const FString& FilePath);

	// Dropdown options
	TArray<TSharedPtr<FString>> BeatRateOptions;
	TArray<TSharedPtr<FString>> AnalysisModeOptions;
	TArray<TSharedPtr<FString>> ResolutionOptions;
	TArray<TSharedPtr<FString>> FPSOptions;
	TArray<TSharedPtr<FString>> ColorPresetOptions;
	TArray<TSharedPtr<FString>> TransitionOptions;

	// Background and title brushes
	FSlateBrush WallpaperBrush;
	FSlateBrush TitleBrush;
	FSlateBrush PreviewBrush;

	// Image brush storage (for Program target - no Engine dependency)
	TSharedPtr<FSlateDynamicImageBrush> WallpaperImageBrush;
	TSharedPtr<FSlateDynamicImageBrush> TitleImageBrush;
	TSharedPtr<FSlateDynamicImageBrush> PreviewImageBrush;
	TArray<uint8> PreviewPixelData;
	int32 PreviewWidth = 0;
	int32 PreviewHeight = 0;

	// Custom fonts (Corpta)
	FSlateFontInfo TitleFont;        // Large title font (28pt)
	FSlateFontInfo HeadingFont;      // Section headings (16pt)
	FSlateFontInfo ButtonFont;       // Button text (18pt bold)
	FSlateFontInfo ButtonFontSmall;  // Smaller button text (14pt)
	FSlateFontInfo BodyFont;         // Regular body text (12pt)
	bool bCustomFontLoaded = false;

	void LoadAssets();

	// Async processing
	TUniquePtr<FAsyncTask<FBeatsyncProcessingTask>> ProcessingTask;
	void OnProcessingProgress(float Progress, const FString& Status);
	void OnProcessingComplete(const FBeatsyncProcessingResult& Result);

	// Preview
	TSharedPtr<SImage> PreviewImage;
	void UpdatePreviewTexture(const TArray<uint8>& RGBData, int32 Width, int32 Height);
};
