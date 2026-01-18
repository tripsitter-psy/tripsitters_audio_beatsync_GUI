

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
    AIStems = 2,
    AudioFlux = 3,          // Signal processing (CPU only)
    StemsFlux = 4           // Stems + AudioFlux hybrid (best quality)
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

/**
 * File paths state for TripSitter widget
 */
struct FTripSitterFilePaths
{
    FString AudioPath;
    FString VideoPath;           // Single video path OR folder path
    TArray<FString> VideoPaths;  // Multiple video paths when folder is selected
    FString OutputPath;
    bool bIsMultiClip = false;   // True when a folder with multiple videos is selected

    /** Scan folder for video files and populate VideoPaths */
    void ScanFolderForVideos(const FString& FolderPath);
};

/**
 * Effects configuration for video processing
 */
struct FTripSitterEffectsConfig
{
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
};

/**
 * Processing configuration and state
 */
struct FTripSitterProcessingState
{
    // Config
    EBeatRate BeatRate = EBeatRate::Every;
    EAnalysisMode AnalysisMode = EAnalysisMode::Energy;
    EResolution Resolution = EResolution::HD1080;
    EFPS FPS = EFPS::FPS30;

    // Progress state
    float Progress = 0.0f;
    FString StatusText = TEXT("Ready");
    FString ETAText = TEXT("");
    bool bIsProcessing = false;

    // Analysis results
    bool bAudioAnalyzed = false;
    double DetectedBPM = 0.0;
    TArray<double> AnalyzedBeatTimes;

    // Async processing task
    TUniquePtr<FAsyncTask<FBeatsyncProcessingTask>> ProcessingTask;

    /** Reset progress state to initial values */
    void ResetProgress();
};

/**
 * Preview state for video frame preview
 */
struct FTripSitterPreviewState
{
    bool bPreviewMode = false;
    int32 PreviewBeats = 8;
    float PreviewTimestamp = 0.0f;

    // Preview image data
    TSharedPtr<FSlateDynamicImageBrush> PreviewImageBrush;
    TArray<uint8> PreviewPixelData;
    int32 PreviewWidth = 0;
    int32 PreviewHeight = 0;

    /** Update preview texture from RGB data */
    void UpdatePreviewTexture(const TArray<uint8>& RGBData, int32 Width, int32 Height);
};

/**
 * Theme and style configuration (Psychedelic theme)
 */
struct FTripSitterTheme
{
    // Colors
    FLinearColor NeonCyan = FLinearColor(0.0f, 0.851f, 1.0f);      // Primary: (0, 217, 255)
    FLinearColor NeonPurple = FLinearColor(0.545f, 0.0f, 1.0f);    // Secondary: (139, 0, 255)
    FLinearColor DarkBg = FLinearColor(0.039f, 0.039f, 0.102f);    // Background: (10, 10, 26)
    FLinearColor ControlBg = FLinearColor(0.078f, 0.078f, 0.157f); // Surface: (20, 20, 40)
    FLinearColor TextColor = FLinearColor(0.784f, 0.863f, 1.0f);   // Text: (200, 220, 255)
    FLinearColor HotPink = FLinearColor(1.0f, 0.0f, 0.502f);       // Accent: (255, 0, 128)
    FLinearColor NeonGreen = FLinearColor(0.0f, 1.0f, 0.392f);     // Success: (0, 255, 100)

    // Fonts (Corpta custom font)
    FSlateFontInfo TitleFont;        // Large title font (28pt)
    FSlateFontInfo HeadingFont;      // Section headings (16pt)
    FSlateFontInfo ButtonFont;       // Button text (18pt bold)
    FSlateFontInfo ButtonFontSmall;  // Smaller button text (14pt)
    FSlateFontInfo BodyFont;         // Regular body text (12pt)
    bool bCustomFontLoaded = false;

    // Brushes
    FSlateBrush WallpaperBrush;
    FSlateBrush TitleBrush;
    FSlateBrush PreviewBrush;
    TSharedPtr<FSlateDynamicImageBrush> WallpaperImageBrush;
    TSharedPtr<FSlateDynamicImageBrush> TitleImageBrush;

    /** Load theme assets (fonts, images) */
    void LoadAssets();
};

class STripSitterMainWidget : public SCompoundWidget
{
public:
	SLATE_BEGIN_ARGS(STripSitterMainWidget) {}
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

private:
	// Extracted state components
	FTripSitterFilePaths FilePaths;
	FTripSitterEffectsConfig EffectsConfig;
	FTripSitterProcessingState ProcessingState;
	FTripSitterPreviewState PreviewState;
	FTripSitterTheme Theme;

	// UI Elements (widget references must stay in main class)
	TSharedPtr<SEditableTextBox> AudioPathBox;
	TSharedPtr<SEditableTextBox> VideoPathBox;
	TSharedPtr<SEditableTextBox> OutputPathBox;
	TSharedPtr<SProgressBar> ProgressBar;
	TSharedPtr<STextBlock> StatusTextBlock;
	TSharedPtr<STextBlock> ETATextBlock;
	TSharedPtr<STextBlock> BPMTextBlock;
	TSharedPtr<SWaveformViewer> WaveformViewer;
	TSharedPtr<SImage> PreviewImage;

	// Dropdown options (UI state)
	TArray<TSharedPtr<FString>> BeatRateOptions;
	TArray<TSharedPtr<FString>> AnalysisModeOptions;
	TArray<TSharedPtr<FString>> ResolutionOptions;
	TArray<TSharedPtr<FString>> FPSOptions;
	TArray<TSharedPtr<FString>> ColorPresetOptions;
	TArray<TSharedPtr<FString>> TransitionOptions;

	// Button handlers (event wiring)
	FReply OnBrowseAudioClicked();
	FReply OnBrowseVideoClicked();
	FReply OnBrowseVideoFolderClicked();
	FReply OnBrowseOutputClicked();
	FReply OnStartSyncClicked();
	FReply OnCancelClicked();
	FReply OnPreviewFrameClicked();
	FReply OnAnalyzeAudioClicked();

	// UI section builders (layout)
	TSharedRef<SWidget> CreateFileSection();
	TSharedRef<SWidget> CreateWaveformSection();
	TSharedRef<SWidget> CreateAnalysisSection();
	TSharedRef<SWidget> CreateEffectsSection();
	TSharedRef<SWidget> CreateTransitionsSection();
	TSharedRef<SWidget> CreateControlSection();

	// Load waveform data from audio file
	void LoadWaveformFromAudio(const FString& FilePath);

	// Processing callbacks
	void OnProcessingProgress(float InProgress, const FString& Status);
	void OnProcessingComplete(const FBeatsyncProcessingResult& Result);
};
