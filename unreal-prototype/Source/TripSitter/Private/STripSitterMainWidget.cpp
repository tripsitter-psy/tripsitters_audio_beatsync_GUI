#include "STripSitterMainWidget.h"
#include "BeatsyncLoader.h"
#include "BeatsyncProcessingTask.h"
#include "SWaveformViewer.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Input/SSlider.h"
#include "Widgets/Input/SComboBox.h"
#include "Widgets/Input/SSpinBox.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SSpacer.h"
#include "Widgets/Layout/SScaleBox.h"
#include "Widgets/Notifications/SProgressBar.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Images/SImage.h"
#include "Framework/Application/SlateApplication.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"
#include "Async/Async.h"

// Image loading - use IImageWrapper for Program target compatibility
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"
#include "Rendering/SlateRenderer.h"
#include "Brushes/SlateImageBrush.h"

#if WITH_EDITOR
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#else
// Windows native file dialog for standalone builds
#if PLATFORM_WINDOWS
#include "Windows/AllowWindowsPlatformTypes.h"
#include <commdlg.h>
#include <shlobj.h>
#include "Windows/HideWindowsPlatformTypes.h"
#endif
#endif

#define LOCTEXT_NAMESPACE "TripSitterMainWidget"

// Helper to load PNG and create brush with actual pixel data for standalone programs
static bool LoadPngToRawData(const FString& FilePath, TArray<uint8>& OutPixelData, int32& OutWidth, int32& OutHeight)
{
	TArray<uint8> RawFileData;
	if (!FFileHelper::LoadFileToArray(RawFileData, *FilePath))
	{
		UE_LOG(LogTemp, Warning, TEXT("Failed to load file: %s"), *FilePath);
		return false;
	}

	IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
	TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);

	if (!ImageWrapper.IsValid() || !ImageWrapper->SetCompressed(RawFileData.GetData(), RawFileData.Num()))
	{
		UE_LOG(LogTemp, Warning, TEXT("Failed to decompress PNG: %s"), *FilePath);
		return false;
	}

	if (!ImageWrapper->GetRaw(ERGBFormat::BGRA, 8, OutPixelData))
	{
		UE_LOG(LogTemp, Warning, TEXT("Failed to get raw pixel data: %s"), *FilePath);
		return false;
	}

	OutWidth = ImageWrapper->GetWidth();
	OutHeight = ImageWrapper->GetHeight();
	UE_LOG(LogTemp, Log, TEXT("Loaded PNG %dx%d: %s"), OutWidth, OutHeight, *FilePath);
	return true;
}

void STripSitterMainWidget::LoadAssets()
{
	// For Program target, resources are relative to the executable
	// First try: Next to executable in Resources folder
	// Second try: In Engine/Source/Programs/TripSitter/Resources (dev builds)
	FString ExeDir = FPaths::GetPath(FPlatformProcess::ExecutablePath());
	FString ResourceDir = FPaths::Combine(ExeDir, TEXT("Resources"));

	// If not found next to exe, try the source location (for dev builds)
	if (!FPaths::DirectoryExists(ResourceDir))
	{
		ResourceDir = FPaths::Combine(ExeDir, TEXT(".."), TEXT(".."), TEXT("Source"), TEXT("Programs"), TEXT("TripSitter"), TEXT("Resources"));
		ResourceDir = FPaths::ConvertRelativePathToFull(ResourceDir);
	}

	FString WallpaperPath = FPaths::Combine(ResourceDir, TEXT("wallpaper.png"));
	FString TitlePath = FPaths::Combine(ResourceDir, TEXT("TitleHeader.png"));
	FString FontPath = FPaths::Combine(ResourceDir, TEXT("Corpta.otf"));

	UE_LOG(LogTemp, Log, TEXT("TripSitter: Looking for resources in %s"), *ResourceDir);

	// Get the Slate renderer to create dynamic textures
	FSlateRenderer* Renderer = FSlateApplication::Get().GetRenderer();

	// Load wallpaper PNG and register with renderer
	if (FPaths::FileExists(WallpaperPath))
	{
		TArray<uint8> PixelData;
		int32 Width = 0, Height = 0;
		if (LoadPngToRawData(WallpaperPath, PixelData, Width, Height))
		{
			// Create unique name for the texture resource
			FName BrushName = FName(*FString::Printf(TEXT("TripSitterWallpaper_%d"), FMath::Rand()));

			// Register the pixel data with the Slate renderer
			if (Renderer && Renderer->GenerateDynamicImageResource(BrushName, Width, Height, PixelData))
			{
				// Create brush that references the registered texture
				WallpaperImageBrush = MakeShareable(new FSlateDynamicImageBrush(
					BrushName,
					FVector2D(Width, Height)
				));

				if (WallpaperImageBrush.IsValid())
				{
					WallpaperBrush = *WallpaperImageBrush;
					WallpaperBrush.DrawAs = ESlateBrushDrawType::Image;
					WallpaperBrush.Tiling = ESlateBrushTileType::NoTile;
					UE_LOG(LogTemp, Log, TEXT("TripSitter: Created wallpaper brush %dx%d"), Width, Height);
				}
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("TripSitter: Failed to register wallpaper texture with renderer"));
			}
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: Wallpaper not found at %s"), *WallpaperPath);
	}

	// Load title header PNG
	if (FPaths::FileExists(TitlePath))
	{
		TArray<uint8> PixelData;
		int32 Width = 0, Height = 0;
		if (LoadPngToRawData(TitlePath, PixelData, Width, Height))
		{
			FName BrushName = FName(*FString::Printf(TEXT("TripSitterTitle_%d"), FMath::Rand()));

			if (Renderer && Renderer->GenerateDynamicImageResource(BrushName, Width, Height, PixelData))
			{
				TitleImageBrush = MakeShareable(new FSlateDynamicImageBrush(
					BrushName,
					FVector2D(Width, Height)
				));

				if (TitleImageBrush.IsValid())
				{
					TitleBrush = *TitleImageBrush;
					TitleBrush.DrawAs = ESlateBrushDrawType::Image;
					TitleBrush.Tiling = ESlateBrushTileType::NoTile;
					UE_LOG(LogTemp, Log, TEXT("TripSitter: Created title brush %dx%d"), Width, Height);
				}
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("TripSitter: Failed to register title texture with renderer"));
			}
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: Title header not found at %s"), *TitlePath);
	}

	// Load custom Corpta font from disk
	if (FPaths::FileExists(FontPath))
	{
		// For runtime font loading, we need to create a composite font with the file path
		// The font path must be an absolute path for FSlateFontInfo to load it
		FString AbsFontPath = FPaths::ConvertRelativePathToFull(FontPath);

		// Create FSlateFontInfo with the font file path - Slate will load it at runtime
		TitleFont = FSlateFontInfo(AbsFontPath, 28);
		HeadingFont = FSlateFontInfo(AbsFontPath, 16);
		ButtonFont = FSlateFontInfo(AbsFontPath, 18);
		ButtonFontSmall = FSlateFontInfo(AbsFontPath, 14);
		BodyFont = FSlateFontInfo(AbsFontPath, 12);
		bCustomFontLoaded = true;
		UE_LOG(LogTemp, Log, TEXT("TripSitter: Loaded custom font from %s"), *AbsFontPath);
	}
	else
	{
		// Fallback to default fonts
		TitleFont = FCoreStyle::GetDefaultFontStyle("Bold", 28);
		HeadingFont = FCoreStyle::GetDefaultFontStyle("Bold", 16);
		ButtonFont = FCoreStyle::GetDefaultFontStyle("Bold", 18);
		ButtonFontSmall = FCoreStyle::GetDefaultFontStyle("Bold", 14);
		BodyFont = FCoreStyle::GetDefaultFontStyle("Regular", 12);
		bCustomFontLoaded = false;
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: Custom font not found at %s, using defaults"), *FontPath);
	}
}

void STripSitterMainWidget::Construct(const FArguments& InArgs)
{
	// Load wallpaper and title assets
	LoadAssets();

	// Initialize dropdown options
	BeatRateOptions.Add(MakeShared<FString>(TEXT("Every Beat")));
	BeatRateOptions.Add(MakeShared<FString>(TEXT("Every 2nd Beat")));
	BeatRateOptions.Add(MakeShared<FString>(TEXT("Every 4th Beat")));
	BeatRateOptions.Add(MakeShared<FString>(TEXT("Every 8th Beat")));

	AnalysisModeOptions.Add(MakeShared<FString>(TEXT("Energy (Fast)")));
	AnalysisModeOptions.Add(MakeShared<FString>(TEXT("AI Beat Detection")));
	AnalysisModeOptions.Add(MakeShared<FString>(TEXT("AI + Stem Separation (Best)")));

	ResolutionOptions.Add(MakeShared<FString>(TEXT("1920x1080 (Full HD)")));
	ResolutionOptions.Add(MakeShared<FString>(TEXT("1280x720 (HD)")));
	ResolutionOptions.Add(MakeShared<FString>(TEXT("3840x2160 (4K)")));
	ResolutionOptions.Add(MakeShared<FString>(TEXT("2560x1440 (2K)")));

	FPSOptions.Add(MakeShared<FString>(TEXT("24 fps (Cinematic)")));
	FPSOptions.Add(MakeShared<FString>(TEXT("30 fps (Standard)")));
	FPSOptions.Add(MakeShared<FString>(TEXT("60 fps (Smooth)")));

	ColorPresetOptions.Add(MakeShared<FString>(TEXT("Warm")));
	ColorPresetOptions.Add(MakeShared<FString>(TEXT("Cool")));
	ColorPresetOptions.Add(MakeShared<FString>(TEXT("Vintage")));
	ColorPresetOptions.Add(MakeShared<FString>(TEXT("Vibrant")));

	TransitionOptions.Add(MakeShared<FString>(TEXT("Fade")));
	TransitionOptions.Add(MakeShared<FString>(TEXT("Dissolve")));
	TransitionOptions.Add(MakeShared<FString>(TEXT("Wipe")));
	TransitionOptions.Add(MakeShared<FString>(TEXT("Zoom")));

	// Initialize beatsync backend
	FBeatsyncLoader::Initialize();

	// Build UI with wallpaper background
	ChildSlot
	[
		SNew(SOverlay)
		// Solid black background layer (ensures nothing shows through)
		+ SOverlay::Slot()
		[
			SNew(SImage)
			.ColorAndOpacity(FLinearColor::Black)
		]
		// Wallpaper background layer
		+ SOverlay::Slot()
		[
			SNew(SScaleBox)
			.Stretch(EStretch::ScaleToFill)
			[
				SNew(SImage)
				.Image(WallpaperImageBrush.IsValid() ? &WallpaperBrush : nullptr)
				.ColorAndOpacity(FLinearColor::White)
			]
		]
		// Content layer
		+ SOverlay::Slot()
		[
			SNew(SBox)
			.MinDesiredWidth(900)
			.MinDesiredHeight(700)
			[
				SNew(SBorder)
				.BorderBackgroundColor(FLinearColor(0.039f, 0.039f, 0.102f, 0.7f))
				.Padding(10)
				[
					SNew(SScrollBox)
					+ SScrollBox::Slot()
					[
						SNew(SVerticalBox)

						// Title Image (constrained height) or Text fallback
						+ SVerticalBox::Slot()
						.AutoHeight()
						.HAlign(HAlign_Center)
						.Padding(0, 5)
						[
							TitleImageBrush.IsValid() ?
							StaticCastSharedRef<SWidget>(
								SNew(SBox)
								.MaxDesiredWidth(500)
								.MaxDesiredHeight(120)
								[
									SNew(SScaleBox)
									.Stretch(EStretch::ScaleToFit)
									[
										SNew(SImage)
										.Image(&TitleBrush)
									]
								]
							)
							:
							StaticCastSharedRef<SWidget>(
								SNew(SVerticalBox)
								+ SVerticalBox::Slot()
								.AutoHeight()
								[
									SNew(STextBlock)
									.Text(FText::FromString(TEXT("MTV TRIP SITTER")))
									.Font(TitleFont)
									.ColorAndOpacity(NeonCyan)
									.Justification(ETextJustify::Center)
								]
								+ SVerticalBox::Slot()
								.AutoHeight()
								[
									SNew(STextBlock)
									.Text(FText::FromString(TEXT("Audio Beat Sync Editor")))
									.Font(ButtonFontSmall)
									.ColorAndOpacity(NeonPurple)
									.Justification(ETextJustify::Center)
								]
							)
						]

						+ SVerticalBox::Slot()
						.AutoHeight()
						.Padding(0, 5)
						[
							SNew(SSeparator)
							.ColorAndOpacity(NeonCyan)
						]

					// File Selection Section
					+ SVerticalBox::Slot()
					.AutoHeight()
					.Padding(0, 15)
					[
						CreateFileSection()
					]

					+ SVerticalBox::Slot()
					.AutoHeight()
					[
						SNew(SSeparator)
						.ColorAndOpacity(NeonPurple)
					]

					// Waveform Visualizer Section
					+ SVerticalBox::Slot()
					.AutoHeight()
					.Padding(0, 15)
					[
						CreateWaveformSection()
					]

					+ SVerticalBox::Slot()
					.AutoHeight()
					[
						SNew(SSeparator)
						.ColorAndOpacity(NeonCyan)
					]

					// Analysis Section
					+ SVerticalBox::Slot()
					.AutoHeight()
					.Padding(0, 15)
					[
						CreateAnalysisSection()
					]

					+ SVerticalBox::Slot()
					.AutoHeight()
					[
						SNew(SSeparator)
						.ColorAndOpacity(NeonCyan)
					]

					// Effects Section
					+ SVerticalBox::Slot()
					.AutoHeight()
					.Padding(0, 15)
					[
						CreateEffectsSection()
					]

					+ SVerticalBox::Slot()
					.AutoHeight()
					[
						SNew(SSeparator)
						.ColorAndOpacity(NeonPurple)
					]

					// Transitions Section
					+ SVerticalBox::Slot()
					.AutoHeight()
					.Padding(0, 15)
					[
						CreateTransitionsSection()
					]

					+ SVerticalBox::Slot()
					.AutoHeight()
					[
						SNew(SSeparator)
						.ColorAndOpacity(NeonCyan)
					]

						// Control Section (Start/Cancel/Progress)
						+ SVerticalBox::Slot()
						.AutoHeight()
						.Padding(0, 20)
						[
							CreateControlSection()
						]
					]
				]
			]
		]
	];
}

TSharedRef<SWidget> STripSitterMainWidget::CreateFileSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 5)
		[
			SNew(STextBlock)
			.Text(FText::FromString(TEXT("FILE SELECTION")))
			.Font(HeadingFont)
			.ColorAndOpacity(NeonCyan)
		]

		// Audio File
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.VAlign(VAlign_Center)
			.Padding(0, 0, 10, 0)
			[
				SNew(SBox)
				.WidthOverride(120)
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Audio File:")))
					.ColorAndOpacity(FLinearColor::White)
				]
			]
			+ SHorizontalBox::Slot()
			.FillWidth(1.0f)
			.Padding(0, 0, 10, 0)
			[
				SAssignNew(AudioPathBox, SEditableTextBox)
				.HintText(FText::FromString(TEXT("Select audio file (.mp3, .wav, .flac)")))
				.BackgroundColor(ControlBg)
			]
			+ SHorizontalBox::Slot()
			.AutoWidth()
			[
				SNew(SButton)
				.Text(FText::FromString(TEXT("Browse...")))
				.OnClicked(this, &STripSitterMainWidget::OnBrowseAudioClicked)
			]
		]

		// Video File/Folder
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.VAlign(VAlign_Center)
			.Padding(0, 0, 10, 0)
			[
				SNew(SBox)
				.WidthOverride(120)
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Video Source:")))
					.ColorAndOpacity(FLinearColor::White)
				]
			]
			+ SHorizontalBox::Slot()
			.FillWidth(1.0f)
			.Padding(0, 0, 10, 0)
			[
				SAssignNew(VideoPathBox, SEditableTextBox)
				.HintText(FText::FromString(TEXT("Select video file or folder")))
				.BackgroundColor(ControlBg)
			]
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(0, 0, 5, 0)
			[
				SNew(SButton)
				.Text(FText::FromString(TEXT("File...")))
				.OnClicked(this, &STripSitterMainWidget::OnBrowseVideoClicked)
				.ToolTipText(FText::FromString(TEXT("Select a single video file")))
			]
			+ SHorizontalBox::Slot()
			.AutoWidth()
			[
				SNew(SButton)
				.Text(FText::FromString(TEXT("Folder...")))
				.OnClicked(this, &STripSitterMainWidget::OnBrowseVideoFolderClicked)
				.ToolTipText(FText::FromString(TEXT("Select a folder containing multiple videos")))
			]
		]

		// Output File
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.VAlign(VAlign_Center)
			.Padding(0, 0, 10, 0)
			[
				SNew(SBox)
				.WidthOverride(120)
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Output File:")))
					.ColorAndOpacity(FLinearColor::White)
				]
			]
			+ SHorizontalBox::Slot()
			.FillWidth(1.0f)
			.Padding(0, 0, 10, 0)
			[
				SAssignNew(OutputPathBox, SEditableTextBox)
				.HintText(FText::FromString(TEXT("Output video path (.mp4)")))
				.BackgroundColor(ControlBg)
			]
			+ SHorizontalBox::Slot()
			.AutoWidth()
			[
				SNew(SButton)
				.Text(FText::FromString(TEXT("Browse...")))
				.OnClicked(this, &STripSitterMainWidget::OnBrowseOutputClicked)
			]
		];
}

TSharedRef<SWidget> STripSitterMainWidget::CreateWaveformSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 5)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.AutoWidth()
			[
				SNew(STextBlock)
				.Text(FText::FromString(TEXT("WAVEFORM")))
				.Font(HeadingFont)
				.ColorAndOpacity(NeonCyan)
			]
			+ SHorizontalBox::Slot()
			.FillWidth(1.0f)
			[
				SNew(SSpacer)
			]
			// Zoom controls
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(5, 0)
			[
				SNew(SButton)
				.Text(FText::FromString(TEXT("-")))
				.OnClicked_Lambda([this]() {
					if (WaveformViewer.IsValid()) WaveformViewer->ZoomOut();
					return FReply::Handled();
				})
				.ToolTipText(FText::FromString(TEXT("Zoom Out")))
			]
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(2, 0)
			[
				SNew(SButton)
				.Text(FText::FromString(TEXT("Fit")))
				.OnClicked_Lambda([this]() {
					if (WaveformViewer.IsValid()) WaveformViewer->ZoomToFit();
					return FReply::Handled();
				})
				.ToolTipText(FText::FromString(TEXT("Zoom to Fit")))
			]
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(5, 0)
			[
				SNew(SButton)
				.Text(FText::FromString(TEXT("+")))
				.OnClicked_Lambda([this]() {
					if (WaveformViewer.IsValid()) WaveformViewer->ZoomIn();
					return FReply::Handled();
				})
				.ToolTipText(FText::FromString(TEXT("Zoom In")))
			]
		]

		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SBox)
			.HeightOverride(150)
			[
				SAssignNew(WaveformViewer, SWaveformViewer)
				.WaveformColor(NeonCyan)
				.BeatMarkerColor(FLinearColor(1.0f, 1.0f, 0.4f)) // Yellow
				.SelectionColor(FLinearColor(0.0f, 0.2f, 0.3f, 0.5f))
				.HandleColor(HotPink)
			]
		]

		// Duration and selection info
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 4)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			[
				SNew(STextBlock)
				.Text_Lambda([this]() {
					if (WaveformViewer.IsValid())
					{
						double Duration = WaveformViewer->GetDuration();
						if (Duration > 0)
						{
							int32 Minutes = FMath::FloorToInt(Duration / 60.0);
							int32 Seconds = FMath::FloorToInt(FMath::Fmod(Duration, 60.0));
							return FText::FromString(FString::Printf(TEXT("Duration: %d:%02d"), Minutes, Seconds));
						}
					}
					return FText::FromString(TEXT("No audio loaded"));
				})
				.ColorAndOpacity(TextColor)
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			.HAlign(HAlign_Right)
			[
				SNew(STextBlock)
				.Text_Lambda([this]() {
					if (WaveformViewer.IsValid() && WaveformViewer->GetDuration() > 0)
					{
						double Start = WaveformViewer->GetSelectionStart();
						double End = WaveformViewer->GetSelectionEnd();
						if (End < 0) End = WaveformViewer->GetDuration();

						auto FormatTime = [](double Time) {
							int32 Min = FMath::FloorToInt(Time / 60.0);
							double Sec = FMath::Fmod(Time, 60.0);
							return FString::Printf(TEXT("%d:%05.2f"), Min, Sec);
						};

						return FText::FromString(FString::Printf(TEXT("Selection: %s - %s (%.1fs)"),
							*FormatTime(Start), *FormatTime(End), End - Start));
					}
					return FText::FromString(TEXT(""));
				})
				.ColorAndOpacity(HotPink)
			]
		]

		// Help text - use default font since Corpta may not have all glyphs
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 2)
		[
			SNew(STextBlock)
			.Text(FText::FromString(TEXT("Scroll to zoom | Middle-click drag to pan | Right-click to add effect regions")))
			.Font(FCoreStyle::GetDefaultFontStyle("Italic", 9))
			.ColorAndOpacity(FLinearColor(0.5f, 0.5f, 0.6f))
		];
}

void STripSitterMainWidget::LoadWaveformFromAudio(const FString& FilePath)
{
	if (!FBeatsyncLoader::IsInitialized())
	{
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: Beatsync not initialized, cannot load waveform"));
		return;
	}

	void* Analyzer = FBeatsyncLoader::CreateAnalyzer();
	if (!Analyzer)
	{
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: Failed to create analyzer for waveform"));
		return;
	}

	TArray<float> Peaks;
	double Duration = 0.0;

	bool bSuccess = FBeatsyncLoader::GetWaveform(Analyzer, FilePath, Peaks, Duration);
	FBeatsyncLoader::DestroyAnalyzer(Analyzer);

	if (bSuccess && WaveformViewer.IsValid())
	{
		WaveformViewer->SetWaveformData(Peaks, Duration);
		UE_LOG(LogTemp, Log, TEXT("TripSitter: Loaded waveform with %d peaks, duration %.2fs"), Peaks.Num(), Duration);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: Failed to load waveform from %s"), *FilePath);
	}
}

TSharedRef<SWidget> STripSitterMainWidget::CreateAnalysisSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 5)
		[
			SNew(STextBlock)
			.Text(FText::FromString(TEXT("ANALYSIS OPTIONS")))
			.Font(HeadingFont)
			.ColorAndOpacity(NeonPurple)
		]

		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			// Beat Rate
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			.Padding(0, 0, 20, 0)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Beat Rate:")))
					.ColorAndOpacity(FLinearColor::White)
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(0, 4, 0, 0)
				[
					SNew(SComboBox<TSharedPtr<FString>>)
					.OptionsSource(&BeatRateOptions)
					.OnGenerateWidget_Lambda([](TSharedPtr<FString> Item) {
						return SNew(STextBlock).Text(FText::FromString(*Item));
					})
					.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Item, ESelectInfo::Type) {
						int32 Index = BeatRateOptions.Find(Item);
						if (Index != INDEX_NONE) {
							BeatRate = static_cast<EBeatRate>(Index);
						}
					})
					[
						SNew(STextBlock)
						.Text_Lambda([this]() {
							int32 Index = static_cast<int32>(BeatRate);
							return FText::FromString(BeatRateOptions.IsValidIndex(Index) ? *BeatRateOptions[Index] : TEXT("Select..."));
						})
					]
				]
			]
			// Analysis Mode
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Analysis Mode:")))
					.ColorAndOpacity(FLinearColor::White)
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(0, 4, 0, 0)
				[
					SNew(SComboBox<TSharedPtr<FString>>)
					.OptionsSource(&AnalysisModeOptions)
					.OnGenerateWidget_Lambda([](TSharedPtr<FString> Item) {
						return SNew(STextBlock).Text(FText::FromString(*Item));
					})
					.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Item, ESelectInfo::Type) {
						int32 Index = AnalysisModeOptions.Find(Item);
						if (Index != INDEX_NONE) {
							AnalysisMode = static_cast<EAnalysisMode>(Index);
						}
					})
					[
						SNew(STextBlock)
						.Text_Lambda([this]() {
							int32 Index = static_cast<int32>(AnalysisMode);
							return FText::FromString(AnalysisModeOptions.IsValidIndex(Index) ? *AnalysisModeOptions[Index] : TEXT("Select..."));
						})
					]
				]
			]
		]

		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			// Resolution
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			.Padding(0, 0, 20, 0)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Resolution:")))
					.ColorAndOpacity(FLinearColor::White)
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(0, 4, 0, 0)
				[
					SNew(SComboBox<TSharedPtr<FString>>)
					.OptionsSource(&ResolutionOptions)
					.OnGenerateWidget_Lambda([](TSharedPtr<FString> Item) {
						return SNew(STextBlock).Text(FText::FromString(*Item));
					})
					.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Item, ESelectInfo::Type) {
						int32 Index = ResolutionOptions.Find(Item);
						if (Index != INDEX_NONE) {
							Resolution = static_cast<EResolution>(Index);
						}
					})
					[
						SNew(STextBlock)
						.Text_Lambda([this]() {
							int32 Index = static_cast<int32>(Resolution);
							return FText::FromString(ResolutionOptions.IsValidIndex(Index) ? *ResolutionOptions[Index] : TEXT("Select..."));
						})
					]
				]
			]
			// FPS
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Frame Rate:")))
					.ColorAndOpacity(FLinearColor::White)
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(0, 4, 0, 0)
				[
					SNew(SComboBox<TSharedPtr<FString>>)
					.OptionsSource(&FPSOptions)
					.OnGenerateWidget_Lambda([](TSharedPtr<FString> Item) {
						return SNew(STextBlock).Text(FText::FromString(*Item));
					})
					.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Item, ESelectInfo::Type) {
						int32 Index = FPSOptions.Find(Item);
						if (Index != INDEX_NONE) {
							FPS = static_cast<EFPS>(Index);
						}
					})
					[
						SNew(STextBlock)
						.Text_Lambda([this]() {
							int32 Index = static_cast<int32>(FPS);
							return FText::FromString(FPSOptions.IsValidIndex(Index) ? *FPSOptions[Index] : TEXT("Select..."));
						})
					]
				]
			]
		];
}

TSharedRef<SWidget> STripSitterMainWidget::CreateEffectsSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 5)
		[
			SNew(STextBlock)
			.Text(FText::FromString(TEXT("VISUAL EFFECTS")))
			.Font(HeadingFont)
			.ColorAndOpacity(NeonCyan)
		]

		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			// Vignette
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(0, 0, 30, 0)
			[
				SNew(SCheckBox)
				.OnCheckStateChanged_Lambda([this](ECheckBoxState State) {
					bEnableVignette = (State == ECheckBoxState::Checked);
				})
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Vignette")))
					.ColorAndOpacity(FLinearColor::White)
				]
			]
			// Beat Flash
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(0, 0, 30, 0)
			[
				SNew(SCheckBox)
				.OnCheckStateChanged_Lambda([this](ECheckBoxState State) {
					bEnableBeatFlash = (State == ECheckBoxState::Checked);
				})
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Beat Flash")))
					.ColorAndOpacity(FLinearColor::White)
				]
			]
			// Beat Zoom
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(0, 0, 30, 0)
			[
				SNew(SCheckBox)
				.OnCheckStateChanged_Lambda([this](ECheckBoxState State) {
					bEnableBeatZoom = (State == ECheckBoxState::Checked);
				})
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Beat Zoom")))
					.ColorAndOpacity(FLinearColor::White)
				]
			]
			// Color Grade
			+ SHorizontalBox::Slot()
			.AutoWidth()
			[
				SNew(SCheckBox)
				.OnCheckStateChanged_Lambda([this](ECheckBoxState State) {
					bEnableColorGrade = (State == ECheckBoxState::Checked);
				})
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Color Grade")))
					.ColorAndOpacity(FLinearColor::White)
				]
			]
		]

		// Sliders
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			.Padding(0, 0, 20, 0)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Flash Intensity:")))
					.ColorAndOpacity(FLinearColor::White)
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(0, 4, 0, 0)
				[
					SNew(SSlider)
					.Value(FlashIntensity)
					.OnValueChanged_Lambda([this](float Value) {
						FlashIntensity = Value;
					})
				]
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Zoom Intensity:")))
					.ColorAndOpacity(FLinearColor::White)
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(0, 4, 0, 0)
				[
					SNew(SSlider)
					.Value(ZoomIntensity)
					.OnValueChanged_Lambda([this](float Value) {
						ZoomIntensity = Value;
					})
				]
			]
		];
}

TSharedRef<SWidget> STripSitterMainWidget::CreateTransitionsSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 5)
		[
			SNew(STextBlock)
			.Text(FText::FromString(TEXT("TRANSITIONS")))
			.Font(HeadingFont)
			.ColorAndOpacity(NeonPurple)
		]

		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.AutoWidth()
			.Padding(0, 0, 20, 0)
			.VAlign(VAlign_Center)
			[
				SNew(SCheckBox)
				.OnCheckStateChanged_Lambda([this](ECheckBoxState State) {
					bEnableTransitions = (State == ECheckBoxState::Checked);
				})
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Enable Transitions")))
					.ColorAndOpacity(FLinearColor::White)
				]
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.3f)
			.Padding(0, 0, 20, 0)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Type:")))
					.ColorAndOpacity(FLinearColor::White)
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(0, 4, 0, 0)
				[
					SNew(SComboBox<TSharedPtr<FString>>)
					.OptionsSource(&TransitionOptions)
					.OnGenerateWidget_Lambda([](TSharedPtr<FString> Item) {
						return SNew(STextBlock).Text(FText::FromString(*Item));
					})
					.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Item, ESelectInfo::Type) {
						TransitionType = TransitionOptions.Find(Item);
					})
					[
						SNew(STextBlock)
						.Text_Lambda([this]() {
							return FText::FromString(TransitionOptions.IsValidIndex(TransitionType) ? *TransitionOptions[TransitionType] : TEXT("Select..."));
						})
					]
				]
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.3f)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Duration (sec):")))
					.ColorAndOpacity(FLinearColor::White)
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(0, 4, 0, 0)
				[
					SNew(SSpinBox<float>)
					.MinValue(0.05f)
					.MaxValue(5.0f)
					.Value(TransitionDuration)
					.OnValueChanged_Lambda([this](float Value) {
						TransitionDuration = Value;
					})
				]
			]
		];
}

TSharedRef<SWidget> STripSitterMainWidget::CreateControlSection()
{
	return SNew(SVerticalBox)
		// Preview image (shows extracted video frame)
		+ SVerticalBox::Slot()
		.AutoHeight()
		.HAlign(HAlign_Center)
		.Padding(0, 10)
		[
			SNew(SBox)
			.MaxDesiredWidth(480)
			.MaxDesiredHeight(270)
			[
				SNew(SBorder)
				.BorderBackgroundColor(FLinearColor(0.1f, 0.1f, 0.15f, 1.0f))
				.Padding(2)
				[
					SAssignNew(PreviewImage, SImage)
					.Image(&PreviewBrush)
				]
			]
		]

		// Progress bar
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 5)
		[
			SAssignNew(ProgressBar, SProgressBar)
			.Percent(Progress)
			.FillColorAndOpacity(NeonCyan)
		]

		// Status & ETA
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 8)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.7f)
			[
				SAssignNew(StatusTextBlock, STextBlock)
				.Text(FText::FromString(StatusText))
				.ColorAndOpacity(NeonCyan)
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.3f)
			.HAlign(HAlign_Right)
			[
				SAssignNew(ETATextBlock, STextBlock)
				.Text(FText::FromString(ETAText))
				.ColorAndOpacity(NeonPurple)
			]
		]

		// Buttons
		+ SVerticalBox::Slot()
		.AutoHeight()
		.Padding(0, 15)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot()
			.FillWidth(0.5f)
			.Padding(0, 0, 10, 0)
			[
				SNew(SButton)
				.HAlign(HAlign_Center)
				.VAlign(VAlign_Center)
				.OnClicked(this, &STripSitterMainWidget::OnStartSyncClicked)
				[
					SNew(SBox)
					.HeightOverride(50)
					.VAlign(VAlign_Center)
					[
						SNew(STextBlock)
						.Text(FText::FromString(TEXT("START SYNC")))
						.Font(ButtonFont)
						.Justification(ETextJustify::Center)
					]
				]
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.25f)
			.Padding(0, 0, 10, 0)
			[
				SNew(SButton)
				.HAlign(HAlign_Center)
				.OnClicked(this, &STripSitterMainWidget::OnCancelClicked)
				.IsEnabled_Lambda([this]() { return bIsProcessing; })
				[
					SNew(SBox)
					.HeightOverride(50)
					.VAlign(VAlign_Center)
					[
						SNew(STextBlock)
						.Text(FText::FromString(TEXT("CANCEL")))
						.Font(ButtonFontSmall)
						.Justification(ETextJustify::Center)
					]
				]
			]
			+ SHorizontalBox::Slot()
			.FillWidth(0.25f)
			[
				SNew(SButton)
				.HAlign(HAlign_Center)
				.OnClicked(this, &STripSitterMainWidget::OnPreviewFrameClicked)
				[
					SNew(SBox)
					.HeightOverride(50)
					.VAlign(VAlign_Center)
					[
						SNew(STextBlock)
						.Text(FText::FromString(TEXT("PREVIEW")))
						.Font(ButtonFontSmall)
						.Justification(ETextJustify::Center)
					]
				]
			]
		];
}

FReply STripSitterMainWidget::OnBrowseAudioClicked()
{
#if WITH_EDITOR
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (DesktopPlatform)
	{
		TArray<FString> OutFiles;
		if (DesktopPlatform->OpenFileDialog(
			FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
			TEXT("Select Audio File"),
			TEXT(""),
			TEXT(""),
			TEXT("Audio Files (*.mp3;*.wav;*.flac)|*.mp3;*.wav;*.flac"),
			EFileDialogFlags::None,
			OutFiles))
		{
			if (OutFiles.Num() > 0)
			{
				AudioPath = OutFiles[0];
				AudioPathBox->SetText(FText::FromString(AudioPath));
				LoadWaveformFromAudio(AudioPath);
			}
		}
	}
	#if PLATFORM_WINDOWS
	// Windows native file dialog for standalone builds
	OPENFILENAMEW ofn;
	WCHAR szFile[MAX_PATH] = { 0 };
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFilter = L"Audio Files\0*.mp3;*.wav;*.flac\0All Files\0*.*\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrTitle = L"Select Audio File";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
	if (GetOpenFileNameW(&ofn))
	{
		AudioPath = FString(szFile);
		AudioPathBox->SetText(FText::FromString(AudioPath));
		LoadWaveformFromAudio(AudioPath);
	}
	#endif
	// Non-Windows platforms: no native file dialog
#endif
	return FReply::Handled();
}

FReply STripSitterMainWidget::OnBrowseVideoClicked()
{
#if WITH_EDITOR
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (DesktopPlatform)
	{
		TArray<FString> OutFiles;
		if (DesktopPlatform->OpenFileDialog(
			FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
			TEXT("Select Video File"),
			TEXT(""),
			TEXT(""),
			TEXT("Video Files (*.mp4;*.mov;*.avi)|*.mp4;*.mov;*.avi"),
			EFileDialogFlags::None,
			OutFiles))
		{
			if (OutFiles.Num() > 0)
			{
				VideoPath = OutFiles[0];
				VideoPaths.Empty();
				VideoPaths.Add(VideoPath);
				bIsMultiClip = false;
				VideoPathBox->SetText(FText::FromString(VideoPath));
			}
		}
	}
#else
	OPENFILENAMEW ofn;
	WCHAR szFile[MAX_PATH] = { 0 };
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFilter = L"Video Files\0*.mp4;*.mov;*.avi\0All Files\0*.*\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrTitle = L"Select Video File";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
	if (GetOpenFileNameW(&ofn))
	{
		VideoPath = FString(szFile);
		VideoPaths.Empty();
		VideoPaths.Add(VideoPath);
		bIsMultiClip = false;
		VideoPathBox->SetText(FText::FromString(VideoPath));
	}
#endif
	return FReply::Handled();
}

void STripSitterMainWidget::ScanFolderForVideos(const FString& FolderPath)
{
	VideoPaths.Empty();

	// Scan for video files in the folder
	TArray<FString> Extensions = { TEXT("*.mp4"), TEXT("*.mov"), TEXT("*.avi"), TEXT("*.mkv"), TEXT("*.webm") };

	for (const FString& Ext : Extensions)
	{
		TArray<FString> Files;
		IFileManager::Get().FindFiles(Files, *FPaths::Combine(FolderPath, Ext), true, false);

		for (const FString& File : Files)
		{
			VideoPaths.Add(FPaths::Combine(FolderPath, File));
		}
	}

	// Sort alphabetically for consistent ordering
	VideoPaths.Sort();

	bIsMultiClip = (VideoPaths.Num() > 1);

	UE_LOG(LogTemp, Log, TEXT("TripSitter: Found %d video files in folder %s"), VideoPaths.Num(), *FolderPath);
}

FReply STripSitterMainWidget::OnBrowseVideoFolderClicked()
{
#if WITH_EDITOR
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (DesktopPlatform)
	{
		FString OutFolder;
		if (DesktopPlatform->OpenDirectoryDialog(
			FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
			TEXT("Select Video Folder"),
			TEXT(""),
			OutFolder))
		{
			VideoPath = OutFolder;
			ScanFolderForVideos(OutFolder);

			if (VideoPaths.Num() > 0)
			{
				VideoPathBox->SetText(FText::FromString(FString::Printf(TEXT("%s (%d videos)"), *OutFolder, VideoPaths.Num())));
			}
			else
			{
				VideoPathBox->SetText(FText::FromString(TEXT("No video files found in folder")));
			}
		}
	}
#else
	// Windows native folder browser for standalone builds
	BROWSEINFOW bi = { 0 };
	bi.lpszTitle = L"Select Video Folder";
	bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
	LPITEMIDLIST pidl = SHBrowseForFolderW(&bi);
	if (pidl)
	{
		WCHAR szPath[MAX_PATH];
		if (SHGetPathFromIDListW(pidl, szPath))
		{
			FString OutFolder(szPath);
			VideoPath = OutFolder;
			ScanFolderForVideos(OutFolder);

			if (VideoPaths.Num() > 0)
			{
				VideoPathBox->SetText(FText::FromString(FString::Printf(TEXT("%s (%d videos)"), *OutFolder, VideoPaths.Num())));
			}
			else
			{
				VideoPathBox->SetText(FText::FromString(TEXT("No video files found in folder")));
			}
		}
		CoTaskMemFree(pidl);
	}
#endif
	return FReply::Handled();
}

FReply STripSitterMainWidget::OnBrowseOutputClicked()
{
#if WITH_EDITOR
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (DesktopPlatform)
	{
		TArray<FString> OutFiles;
		if (DesktopPlatform->SaveFileDialog(
			FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
			TEXT("Save Output Video"),
			TEXT(""),
			TEXT("output.mp4"),
			TEXT("MP4 Video (*.mp4)|*.mp4"),
			EFileDialogFlags::None,
			OutFiles))
		{
			if (OutFiles.Num() > 0)
			{
				OutputPath = OutFiles[0];
				OutputPathBox->SetText(FText::FromString(OutputPath));
			}
		}
	}
#else
	// Windows native save dialog for standalone builds
	OPENFILENAMEW ofn;
	WCHAR szFile[MAX_PATH] = { 0 };
	wcscpy_s(szFile, L"output.mp4");
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFilter = L"MP4 Video\0*.mp4\0All Files\0*.*\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrTitle = L"Save Output Video";
	ofn.lpstrDefExt = L"mp4";
	ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
	if (GetSaveFileNameW(&ofn))
	{
		OutputPath = FString(szFile);
		OutputPathBox->SetText(FText::FromString(OutputPath));
	}
#endif
	return FReply::Handled();
}

FReply STripSitterMainWidget::OnStartSyncClicked()
{
	if (AudioPath.IsEmpty() || OutputPath.IsEmpty())
	{
		StatusText = TEXT("Please select audio and output files first");
		StatusTextBlock->SetText(FText::FromString(StatusText));
		return FReply::Handled();
	}

	// Check for video source (either single file or folder with videos)
	if (VideoPaths.Num() == 0 && VideoPath.IsEmpty())
	{
		StatusText = TEXT("Please select a video file or folder");
		StatusTextBlock->SetText(FText::FromString(StatusText));
		return FReply::Handled();
	}

	if (!FBeatsyncLoader::IsInitialized())
	{
		StatusText = TEXT("ERROR: Backend not loaded");
		StatusTextBlock->SetText(FText::FromString(StatusText));
		return FReply::Handled();
	}

	// Don't start if already processing
	if (bIsProcessing)
	{
		return FReply::Handled();
	}

	bIsProcessing = true;
	StatusText = TEXT("Starting...");
	StatusTextBlock->SetText(FText::FromString(StatusText));
	Progress = 0.0f;
	ProgressBar->SetPercent(Progress);

	UE_LOG(LogTemp, Log, TEXT("TripSitter: Starting async sync - Audio: %s, Video: %s, Output: %s"), *AudioPath, *VideoPath, *OutputPath);

	// Build processing parameters
	FBeatsyncProcessingParams Params;
	Params.AudioPath = AudioPath;
	Params.VideoPath = VideoPath;
	Params.VideoPaths = VideoPaths;
	Params.OutputPath = OutputPath;
	Params.bIsMultiClip = bIsMultiClip;
	Params.BeatRate = static_cast<int32>(BeatRate);

	// Get selection range for audio trimming
	if (WaveformViewer.IsValid() && WaveformViewer->GetDuration() > 0)
	{
		Params.AudioStart = WaveformViewer->GetSelectionStart();
		double SelEnd = WaveformViewer->GetSelectionEnd();
		if (SelEnd > Params.AudioStart && SelEnd < WaveformViewer->GetDuration())
		{
			Params.AudioEnd = SelEnd;
		}
	}

	// Build effects config from UI settings
	Params.EffectsConfig.bEnableVignette = bEnableVignette;
	Params.EffectsConfig.VignetteStrength = 0.5f; // Default value
	Params.EffectsConfig.bEnableBeatFlash = bEnableBeatFlash;
	Params.EffectsConfig.FlashIntensity = FlashIntensity;
	Params.EffectsConfig.bEnableBeatZoom = bEnableBeatZoom;
	Params.EffectsConfig.ZoomIntensity = ZoomIntensity;
	Params.EffectsConfig.bEnableColorGrade = bEnableColorGrade;
	Params.EffectsConfig.bEnableTransitions = bEnableTransitions;
	Params.EffectsConfig.TransitionDuration = TransitionDuration;
	Params.EffectsConfig.EffectBeatDivisor = 1 << static_cast<int32>(BeatRate);

	// Map color preset index to string
	if (ColorPresetOptions.IsValidIndex(ColorPreset))
	{
		Params.EffectsConfig.ColorPreset = *ColorPresetOptions[ColorPreset];
		FString Lower = Params.EffectsConfig.ColorPreset.ToLower();
		Params.EffectsConfig.ColorPreset = Lower;
	}
	else
	{
		Params.EffectsConfig.ColorPreset = TEXT("warm");
	}

	// Map transition type index to string
	if (TransitionOptions.IsValidIndex(TransitionType))
	{
		Params.EffectsConfig.TransitionType = *TransitionOptions[TransitionType];
		FString Lower = Params.EffectsConfig.TransitionType.ToLower();
		Params.EffectsConfig.TransitionType = Lower;
	}
	else
	{
		Params.EffectsConfig.TransitionType = TEXT("fade");
	}

	// Create and start async task
	ProcessingTask = MakeUnique<FAsyncTask<FBeatsyncProcessingTask>>(
		Params,
		FOnBeatsyncProcessingProgress::CreateSP(this, &STripSitterMainWidget::OnProcessingProgress),
		FOnBeatsyncProcessingComplete::CreateSP(this, &STripSitterMainWidget::OnProcessingComplete)
	);

	ProcessingTask->StartBackgroundTask();

	return FReply::Handled();
}

void STripSitterMainWidget::OnProcessingProgress(float InProgress, const FString& Status)
{
	Progress = InProgress;
	StatusText = Status;

	if (ProgressBar.IsValid())
	{
		ProgressBar->SetPercent(Progress);
	}
	if (StatusTextBlock.IsValid())
	{
		StatusTextBlock->SetText(FText::FromString(StatusText));
	}
}

void STripSitterMainWidget::OnProcessingComplete(const FBeatsyncProcessingResult& Result)
{
	bIsProcessing = false;

	if (Result.bSuccess)
	{
		StatusText = TEXT("Complete!");
		Progress = 1.0f;
		UE_LOG(LogTemp, Log, TEXT("TripSitter: Video processing completed successfully"));

		// Update waveform viewer with beat markers
		if (WaveformViewer.IsValid() && Result.BeatTimes.Num() > 0)
		{
			WaveformViewer->SetBeatTimes(Result.BeatTimes);
		}
	}
	else
	{
		StatusText = FString::Printf(TEXT("ERROR: %s"), *Result.ErrorMessage);
		UE_LOG(LogTemp, Error, TEXT("TripSitter: Video processing failed - %s"), *Result.ErrorMessage);
	}

	if (StatusTextBlock.IsValid())
	{
		StatusTextBlock->SetText(FText::FromString(StatusText));
	}
	if (ProgressBar.IsValid())
	{
		ProgressBar->SetPercent(Progress);
	}

	ProcessingTask.Reset();
}

FReply STripSitterMainWidget::OnCancelClicked()
{
	if (ProcessingTask.IsValid() && !ProcessingTask->IsDone())
	{
		ProcessingTask->GetTask().RequestCancel();
		StatusText = TEXT("Cancelling...");
		StatusTextBlock->SetText(FText::FromString(StatusText));
	}
	else
	{
		bIsProcessing = false;
		StatusText = TEXT("Cancelled");
		if (StatusTextBlock.IsValid())
		{
			StatusTextBlock->SetText(FText::FromString(StatusText));
		}
		Progress = 0.0f;
		if (ProgressBar.IsValid())
		{
			ProgressBar->SetPercent(Progress);
		}
	}
	return FReply::Handled();
}

FReply STripSitterMainWidget::OnPreviewFrameClicked()
{
	UE_LOG(LogTemp, Log, TEXT("TripSitter: OnPreviewFrameClicked - Start"));

	// Get video path - prefer first video from list, fallback to single video
	FString PreviewVideoPath = VideoPaths.Num() > 0 ? VideoPaths[0] : VideoPath;

	UE_LOG(LogTemp, Log, TEXT("TripSitter: PreviewVideoPath = %s"), *PreviewVideoPath);

	if (PreviewVideoPath.IsEmpty())
	{
		StatusText = TEXT("Please select a video file first");
		if (StatusTextBlock.IsValid())
		{
			StatusTextBlock->SetText(FText::FromString(StatusText));
		}
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: No video path selected"));
		return FReply::Handled();
	}

	// Check if file exists
	if (!FPaths::FileExists(PreviewVideoPath))
	{
		StatusText = FString::Printf(TEXT("ERROR: Video file not found: %s"), *PreviewVideoPath);
		if (StatusTextBlock.IsValid())
		{
			StatusTextBlock->SetText(FText::FromString(StatusText));
		}
		UE_LOG(LogTemp, Error, TEXT("TripSitter: Video file does not exist: %s"), *PreviewVideoPath);
		return FReply::Handled();
	}

	if (!FBeatsyncLoader::IsInitialized())
	{
		StatusText = TEXT("ERROR: Backend not loaded - trying to initialize...");
		if (StatusTextBlock.IsValid())
		{
			StatusTextBlock->SetText(FText::FromString(StatusText));
		}
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: Backend not initialized, attempting init..."));

		// Try to initialize
		if (!FBeatsyncLoader::Initialize())
		{
			StatusText = TEXT("ERROR: Failed to load backend DLL");
			if (StatusTextBlock.IsValid())
			{
				StatusTextBlock->SetText(FText::FromString(StatusText));
			}
			UE_LOG(LogTemp, Error, TEXT("TripSitter: Failed to initialize BeatsyncLoader"));
			return FReply::Handled();
		}
	}

	// Get timestamp from waveform selection or use preview timestamp
	double Timestamp = PreviewTimestamp;
	if (WaveformViewer.IsValid() && WaveformViewer->GetDuration() > 0)
	{
		Timestamp = WaveformViewer->GetSelectionStart();
	}

	// Clamp timestamp to valid range
	if (Timestamp < 0.0)
	{
		Timestamp = 0.0;
	}

	StatusText = FString::Printf(TEXT("Extracting frame at %.2fs..."), Timestamp);
	if (StatusTextBlock.IsValid())
	{
		StatusTextBlock->SetText(FText::FromString(StatusText));
	}

	UE_LOG(LogTemp, Log, TEXT("TripSitter: Attempting to extract frame from %s at %.2fs"), *PreviewVideoPath, Timestamp);

	// Extract frame
	TArray<uint8> FrameData;
	int32 Width = 0;
	int32 Height = 0;

	bool bSuccess = FBeatsyncLoader::ExtractFrame(PreviewVideoPath, Timestamp, FrameData, Width, Height);

	UE_LOG(LogTemp, Log, TEXT("TripSitter: ExtractFrame returned success=%d, dataSize=%d, w=%d, h=%d"),
		bSuccess ? 1 : 0, FrameData.Num(), Width, Height);

	if (bSuccess && FrameData.Num() > 0 && Width > 0 && Height > 0)
	{
		// Verify the data size matches expectations
		int32 ExpectedSize = Width * Height * 3;
		if (FrameData.Num() >= ExpectedSize)
		{
			UE_LOG(LogTemp, Log, TEXT("TripSitter: Calling UpdatePreviewTexture..."));
			UpdatePreviewTexture(FrameData, Width, Height);
			StatusText = FString::Printf(TEXT("Preview: %dx%d at %.2fs"), Width, Height, Timestamp);
			UE_LOG(LogTemp, Log, TEXT("TripSitter: Extracted preview frame %dx%d at %.2fs"), Width, Height, Timestamp);
		}
		else
		{
			StatusText = FString::Printf(TEXT("ERROR: Frame data size mismatch (%d vs %d)"), FrameData.Num(), ExpectedSize);
			UE_LOG(LogTemp, Error, TEXT("TripSitter: Frame data size %d != expected %d"), FrameData.Num(), ExpectedSize);
		}
	}
	else
	{
		StatusText = TEXT("ERROR: Failed to extract frame");
		UE_LOG(LogTemp, Warning, TEXT("TripSitter: Failed to extract preview frame from %s (success=%d, dataSize=%d, w=%d, h=%d)"),
			*PreviewVideoPath, bSuccess ? 1 : 0, FrameData.Num(), Width, Height);
	}

	if (StatusTextBlock.IsValid())
	{
		StatusTextBlock->SetText(FText::FromString(StatusText));
	}

	UE_LOG(LogTemp, Log, TEXT("TripSitter: OnPreviewFrameClicked - Complete"));
	return FReply::Handled();
}

void STripSitterMainWidget::UpdatePreviewTexture(const TArray<uint8>& RGBData, int32 Width, int32 Height)
{
	UE_LOG(LogTemp, Log, TEXT("UpdatePreviewTexture: Start - %dx%d, data size=%d"), Width, Height, RGBData.Num());

	// Validate input
	if (Width <= 0 || Height <= 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("UpdatePreviewTexture: Invalid dimensions %dx%d"), Width, Height);
		return;
	}

	int32 ExpectedSize = Width * Height * 3;
	if (RGBData.Num() < ExpectedSize)
	{
		UE_LOG(LogTemp, Warning, TEXT("UpdatePreviewTexture: RGB data size %d < expected %d"), RGBData.Num(), ExpectedSize);
		return;
	}

	// Store dimensions
	PreviewWidth = Width;
	PreviewHeight = Height;

	// Convert RGB24 to BGRA32 for Slate
	int32 PixelCount = Width * Height;
	PreviewPixelData.SetNum(PixelCount * 4);
	const uint8* SrcData = RGBData.GetData();

	UE_LOG(LogTemp, Log, TEXT("UpdatePreviewTexture: Converting RGB24 to BGRA32 (%d pixels)"), PixelCount);

	for (int32 i = 0; i < PixelCount; ++i)
	{
		PreviewPixelData[i * 4 + 0] = SrcData[i * 3 + 2]; // B
		PreviewPixelData[i * 4 + 1] = SrcData[i * 3 + 1]; // G
		PreviewPixelData[i * 4 + 2] = SrcData[i * 3 + 0]; // R
		PreviewPixelData[i * 4 + 3] = 255;                 // A
	}

	UE_LOG(LogTemp, Log, TEXT("UpdatePreviewTexture: Getting Slate renderer..."));

	// Create dynamic brush from raw pixel data using Slate renderer
	FSlateRenderer* Renderer = FSlateApplication::Get().GetRenderer();
	if (!Renderer)
	{
		UE_LOG(LogTemp, Error, TEXT("UpdatePreviewTexture: FSlateRenderer is null!"));
		return;
	}

	FName BrushName = FName(*FString::Printf(TEXT("PreviewTexture_%d"), FMath::Rand()));
	UE_LOG(LogTemp, Log, TEXT("UpdatePreviewTexture: Generating dynamic image resource '%s'..."), *BrushName.ToString());

	if (Renderer->GenerateDynamicImageResource(BrushName, Width, Height, PreviewPixelData))
	{
		UE_LOG(LogTemp, Log, TEXT("UpdatePreviewTexture: Creating FSlateDynamicImageBrush..."));

		PreviewImageBrush = MakeShareable(new FSlateDynamicImageBrush(
			BrushName,
			FVector2D(Width, Height)
		));

		if (PreviewImageBrush.IsValid())
		{
			PreviewBrush = *PreviewImageBrush;
			PreviewBrush.ImageSize = FVector2D(Width, Height);
			PreviewBrush.DrawAs = ESlateBrushDrawType::Image;
			PreviewBrush.Tiling = ESlateBrushTileType::NoTile;

			// Update preview image if it exists
			if (PreviewImage.IsValid())
			{
				UE_LOG(LogTemp, Log, TEXT("UpdatePreviewTexture: Setting PreviewImage brush"));
				PreviewImage->SetImage(&PreviewBrush);
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("UpdatePreviewTexture: PreviewImage widget is not valid!"));
			}
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("UpdatePreviewTexture: Failed to create FSlateDynamicImageBrush"));
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("UpdatePreviewTexture: GenerateDynamicImageResource failed"));
	}

	UE_LOG(LogTemp, Log, TEXT("UpdatePreviewTexture: Complete"));
}

#undef LOCTEXT_NAMESPACE
