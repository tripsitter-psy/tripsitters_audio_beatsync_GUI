#include "BeatSyncHUD.h"
#include "Engine/Canvas.h"
#include "Engine/Texture2D.h"
#include "Blueprint/UserWidget.h"
#include "CanvasItem.h"
#include "GameFramework/PlayerController.h"
#include "Framework/Application/SlateApplication.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/GameInstance.h"

#if WITH_DESKTOP_PLATFORM
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#endif

// Platform-specific file dialog support for packaged builds
#if PLATFORM_MAC
#import <AppKit/AppKit.h>
#import <CoreFoundation/CoreFoundation.h>
#define HAS_NATIVE_MAC_DIALOGS 1
#elif PLATFORM_WINDOWS
#include "Windows/AllowWindowsPlatformTypes.h"
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#include "Windows/HideWindowsPlatformTypes.h"
#define HAS_NATIVE_WINDOWS_DIALOGS 1
#endif

ABeatSyncHUD::ABeatSyncHUD()
{
	// Load wallpaper texture
	static ConstructorHelpers::FObjectFinder<UTexture2D> WallpaperFinder(
		TEXT("Texture2D'/Game/UI/Textures/wallpaper.wallpaper'"));
	if (WallpaperFinder.Succeeded())
	{
		BackgroundTexture = WallpaperFinder.Object;
	}

	// Load header texture
	static ConstructorHelpers::FObjectFinder<UTexture2D> HeaderFinder(
		TEXT("Texture2D'/Game/UI/Textures/header.header'"));
	if (HeaderFinder.Succeeded())
	{
		HeaderTexture = HeaderFinder.Object;
	}

	// Use engine built-in fonts (custom font disabled for now)
	// To use custom font: import Corpta.otf in editor, create Font asset at /Game/UI/Fonts/Corpta_Font
	CustomFont = nullptr;
	CustomFontLarge = nullptr;
}

void ABeatSyncHUD::BeginPlay()
{
	Super::BeginPlay();

	// Create and add the main widget if class is set
	if (MainWidgetClass)
	{
		MainWidget = CreateWidget<UUserWidget>(GetWorld(), MainWidgetClass);
		if (MainWidget)
		{
			MainWidget->AddToViewport();
		}
	}

	// Set up input mode for UI application
	if (APlayerController* PC = GetOwningPlayerController())
	{
		// Show mouse cursor
		PC->bShowMouseCursor = true;
		PC->bEnableClickEvents = true;
		PC->bEnableMouseOverEvents = true;

		// Set input mode to Game and UI - this is crucial for mouse input in packaged builds
		FInputModeGameAndUI InputMode;
		InputMode.SetLockMouseToViewportBehavior(EMouseLockMode::DoNotLock);
		InputMode.SetHideCursorDuringCapture(false);
		PC->SetInputMode(InputMode);

		// Ensure we don't capture mouse on click
		PC->SetShowMouseCursor(true);
	}

	// Bind to beatsync subsystem events
	if (UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem())
	{
		Subsystem->OnAnalysisProgress.AddDynamic(this, &ABeatSyncHUD::OnAnalysisProgress);
		Subsystem->OnAnalysisComplete.AddDynamic(this, &ABeatSyncHUD::OnAnalysisComplete);
		Subsystem->OnError.AddDynamic(this, &ABeatSyncHUD::OnAnalysisError);
	}
}

UBeatsyncSubsystem* ABeatSyncHUD::GetBeatsyncSubsystem() const
{
	if (UGameInstance* GI = UGameplayStatics::GetGameInstance(GetWorld()))
	{
		return GI->GetSubsystem<UBeatsyncSubsystem>();
	}
	return nullptr;
}

void ABeatSyncHUD::AnalyzeSelectedAudio()
{
	if (AudioFilePath.IsEmpty() || AudioFilePath == TEXT("No file selected"))
	{
		StatusMessage = TEXT("Please select an audio file first");
		StatusMessageTimer = 3.0f;
		return;
	}

	UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem();
	if (!Subsystem)
	{
		StatusMessage = TEXT("BeatSync subsystem not available");
		StatusMessageTimer = 3.0f;
		return;
	}

	bIsAnalyzing = true;
	AnalysisStatus = FString::Printf(TEXT("Analyzing: %s"), *AudioFilePath.Right(40));
	StatusMessage = AnalysisStatus;
	StatusMessageTimer = 10.0f;

	Subsystem->AnalyzeAudioFile(AudioFilePath);
}

void ABeatSyncHUD::OnAnalysisProgress(float Progress)
{
	AnalysisStatus = FString::Printf(TEXT("Analyzing: %.0f%%"), Progress * 100.0f);
	StatusMessage = AnalysisStatus;
	StatusMessageTimer = 5.0f;
}

void ABeatSyncHUD::OnAnalysisComplete()
{
	bIsAnalyzing = false;

	UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem();
	if (Subsystem)
	{
		FBeatData Data = Subsystem->GetBeatData();

		// Cache the data for waveform drawing
		CachedWaveform = Data.WaveformData;
		CachedBeatTimestamps = Data.BeatTimestamps;
		CachedBPM = Data.BPM > 0 ? Data.BPM : 120.0f;
		CachedDuration = Data.Duration > 0 ? Data.Duration : 180.0f;

		// Update UI state
		BPMValue = CachedBPM;
		TotalAudioDuration = CachedDuration;
		bHasRealWaveform = CachedWaveform.Num() > 0;

		// Show debug info with version marker [v2] to confirm new code is running
		AnalysisStatus = FString::Printf(TEXT("[v2] Found %d beats at %.1f BPM (%.0fs) %s"),
			Data.BeatCount, Data.BPM, Data.Duration, *Data.DebugInfo);
		StatusMessage = AnalysisStatus;
		StatusMessageTimer = 5.0f;
	}
}

void ABeatSyncHUD::OnAnalysisError(const FString& Error)
{
	bIsAnalyzing = false;
	AnalysisStatus = FString::Printf(TEXT("Error: %s"), *Error);
	StatusMessage = AnalysisStatus;
	StatusMessageTimer = 5.0f;
}

void ABeatSyncHUD::ScrollUp()
{
	ScrollOffset = FMath::Max(0.0f, ScrollOffset - ScrollSpeed);
}

void ABeatSyncHUD::ScrollDown()
{
	ScrollOffset = FMath::Min(MaxScrollOffset, ScrollOffset + ScrollSpeed);
}

void ABeatSyncHUD::UpdateMouseState()
{
	APlayerController* PC = GetOwningPlayerController();
	if (PC)
	{
		float MouseX, MouseY;
		PC->GetMousePosition(MouseX, MouseY);
		MousePosition = FVector2D(MouseX, MouseY);

		bPrevMousePressed = bMousePressed;
		bMousePressed = PC->IsInputKeyDown(EKeys::LeftMouseButton);
		bMouseJustPressed = bMousePressed && !bPrevMousePressed;
		bMouseJustReleased = !bMousePressed && bPrevMousePressed;
	}
}

void ABeatSyncHUD::DrawHUD()
{
	Super::DrawHUD();

	if (!Canvas) return;

	// Update mouse state
	UpdateMouseState();

	// Handle mouse wheel/touchpad scroll
	APlayerController* PC = GetOwningPlayerController();
	if (PC)
	{
		// Check if mouse is over waveform panel
		bool bMouseInWaveformArea =
			(MousePosition.X >= WaveformPanelMin.X && MousePosition.X <= WaveformPanelMax.X) &&
			(MousePosition.Y >= WaveformPanelMin.Y && MousePosition.Y <= WaveformPanelMax.Y);

		// Check if Command key is held (for zoom mode)
		bool bCommandHeld = PC->IsInputKeyDown(EKeys::LeftCommand) || PC->IsInputKeyDown(EKeys::RightCommand);

		// For horizontal scroll when zoomed
		bool bMouseOverWaveform = (WaveformZoom > 1.0f) && bMouseInWaveformArea;

		float MaxWaveformScroll = 1.0f - (1.0f / WaveformZoom);
		float WaveformScrollStep = 0.05f / WaveformZoom;  // Finer control when more zoomed

		// Helper lambda to adjust scroll when zooming to keep view centered
		auto AdjustScrollForZoom = [this](float OldZoom, float NewZoom)
		{
			if (NewZoom != OldZoom && NewZoom > 1.0f)
			{
				float OldVisible = 1.0f / OldZoom;
				float NewVisible = 1.0f / NewZoom;
				float CenterPos = WaveformScrollX + OldVisible * 0.5f;
				WaveformScrollX = FMath::Clamp(CenterPos - NewVisible * 0.5f, 0.0f, FMath::Max(0.0f, 1.0f - NewVisible));
			}
			else if (NewZoom <= 1.0f)
			{
				WaveformScrollX = 0.0f;
			}
		};

		// Method 1: Check for key press events
		if (PC->WasInputKeyJustPressed(EKeys::MouseScrollUp))
		{
			if (bCommandHeld && bMouseInWaveformArea)
			{
				// Cmd+Scroll Up = Zoom In
				float OldZoom = WaveformZoom;
				WaveformZoom = FMath::Min(8.0f, WaveformZoom + 0.5f);
				AdjustScrollForZoom(OldZoom, WaveformZoom);
			}
			else if (bMouseOverWaveform)
			{
				WaveformScrollX = FMath::Max(0.0f, WaveformScrollX - WaveformScrollStep * 2.0f);
			}
			else
			{
				ScrollOffset = FMath::Max(0.0f, ScrollOffset - ScrollSpeed * 3.0f);
			}
		}
		if (PC->WasInputKeyJustPressed(EKeys::MouseScrollDown))
		{
			if (bCommandHeld && bMouseInWaveformArea)
			{
				// Cmd+Scroll Down = Zoom Out
				float OldZoom = WaveformZoom;
				WaveformZoom = FMath::Max(1.0f, WaveformZoom - 0.5f);
				AdjustScrollForZoom(OldZoom, WaveformZoom);
			}
			else if (bMouseOverWaveform)
			{
				WaveformScrollX = FMath::Min(MaxWaveformScroll, WaveformScrollX + WaveformScrollStep * 2.0f);
			}
			else
			{
				ScrollOffset = FMath::Min(MaxScrollOffset, ScrollOffset + ScrollSpeed * 3.0f);
			}
		}

		// Method 2: Also try IsInputKeyDown for continuous scroll
		if (PC->IsInputKeyDown(EKeys::MouseScrollUp))
		{
			if (bCommandHeld && bMouseInWaveformArea)
			{
				float OldZoom = WaveformZoom;
				WaveformZoom = FMath::Min(8.0f, WaveformZoom + 0.1f);
				AdjustScrollForZoom(OldZoom, WaveformZoom);
			}
			else if (bMouseOverWaveform)
			{
				WaveformScrollX = FMath::Max(0.0f, WaveformScrollX - WaveformScrollStep);
			}
			else
			{
				ScrollOffset = FMath::Max(0.0f, ScrollOffset - ScrollSpeed * 0.5f);
			}
		}
		if (PC->IsInputKeyDown(EKeys::MouseScrollDown))
		{
			if (bCommandHeld && bMouseInWaveformArea)
			{
				float OldZoom = WaveformZoom;
				WaveformZoom = FMath::Max(1.0f, WaveformZoom - 0.1f);
				AdjustScrollForZoom(OldZoom, WaveformZoom);
			}
			else if (bMouseOverWaveform)
			{
				WaveformScrollX = FMath::Min(MaxWaveformScroll, WaveformScrollX + WaveformScrollStep);
			}
			else
			{
				ScrollOffset = FMath::Min(MaxScrollOffset, ScrollOffset + ScrollSpeed * 0.5f);
			}
		}

		// Method 3: Try axis value as fallback (most responsive for touchpad)
		float WheelAxis = PC->GetInputAxisValue(FName("MouseWheelAxis"));
		if (FMath::Abs(WheelAxis) > 0.01f)
		{
			if (bCommandHeld && bMouseInWaveformArea)
			{
				// Cmd+Scroll = Zoom
				float OldZoom = WaveformZoom;
				WaveformZoom = FMath::Clamp(WaveformZoom + WheelAxis * 0.3f, 1.0f, 8.0f);
				AdjustScrollForZoom(OldZoom, WaveformZoom);
			}
			else if (bMouseOverWaveform)
			{
				WaveformScrollX = FMath::Clamp(WaveformScrollX - WheelAxis * WaveformScrollStep * 3.0f, 0.0f, MaxWaveformScroll);
			}
			else
			{
				ScrollOffset = FMath::Clamp(ScrollOffset - WheelAxis * ScrollSpeed * 2.0f, 0.0f, MaxScrollOffset);
			}
		}
	}

	float ScreenWidth = Canvas->ClipX;
	float ScreenHeight = Canvas->ClipY;

	// Draw wallpaper background (static, doesn't scroll)
	if (BackgroundTexture)
	{
		FCanvasTileItem TileItem(
			FVector2D(0, 0),
			BackgroundTexture->GetResource(),
			FVector2D(ScreenWidth, ScreenHeight),
			FLinearColor::White
		);
		TileItem.BlendMode = SE_BLEND_Opaque;
		Canvas->DrawItem(TileItem);
	}

	// Draw header image at top (scrolls with content)
	if (HeaderTexture)
	{
		float HeaderWidth = 600.0f;
		float HeaderHeight = 150.0f;
		float HeaderX = (ScreenWidth - HeaderWidth) / 2.0f;
		float HeaderY = 20.0f - ScrollOffset;

		// Only draw if visible
		if (HeaderY + HeaderHeight > 0 && HeaderY < ScreenHeight)
		{
			FCanvasTileItem HeaderItem(
				FVector2D(HeaderX, HeaderY),
				HeaderTexture->GetResource(),
				FVector2D(HeaderWidth, HeaderHeight),
				FLinearColor::White
			);
			HeaderItem.BlendMode = SE_BLEND_Translucent;
			Canvas->DrawItem(HeaderItem);
		}
	}

	// Draw title (scrolls with content)
	DrawTitle(ScrollOffset);

	// Draw scrollable content
	DrawAllPanels();

	// Draw scrollbar
	DrawScrollbar();

	// Debug: Show mouse position and input state in corner
	if (GEngine && GEngine->GetSmallFont())
	{
		UFont* DebugFont = GEngine->GetSmallFont();
		FString DebugText = FString::Printf(TEXT("Mouse: %.0f, %.0f | Click: %d | Scroll: %.0f/%.0f"),
			MousePosition.X, MousePosition.Y, bMousePressed ? 1 : 0, ScrollOffset, MaxScrollOffset);
		FCanvasTextItem DebugItem(FVector2D(10, ScreenHeight - 30), FText::FromString(DebugText), DebugFont, FLinearColor::Yellow);
		DebugItem.Scale = FVector2D(1.0f, 1.0f);
		Canvas->DrawItem(DebugItem);
	}
}

void ABeatSyncHUD::DrawAllPanels()
{
	if (!Canvas) return;

	float ScreenWidth = Canvas->ClipX;
	float ScreenHeight = Canvas->ClipY;

	float PanelWidth = 420.0f;
	float PanelSpacing = 30.0f;
	float LeftX = 40.0f;
	float RightX = ScreenWidth - PanelWidth - 40.0f;

	// Starting Y position (scrollable content starts here)
	float ContentStartY = 200.0f;
	float CurrentY = ContentStartY - ScrollOffset;

	// ===== LEFT COLUMN =====

	// Input Section (Cyan) - Audio & Video file selection
	float InputPanelHeight = 200.0f;
	DrawPanel(LeftX, CurrentY, PanelWidth, InputPanelHeight, TEXT("INPUT FILES"), CyanColor);
	DrawInputPanel(LeftX, CurrentY, PanelWidth);
	CurrentY += InputPanelHeight + PanelSpacing;

	// Analysis Settings (Purple)
	float AnalysisPanelHeight = 260.0f;
	DrawPanel(LeftX, CurrentY, PanelWidth, AnalysisPanelHeight, TEXT("ANALYSIS SETTINGS"), PurpleColor);
	DrawAnalysisPanel(LeftX, CurrentY, PanelWidth);
	CurrentY += AnalysisPanelHeight + PanelSpacing;

	// Video Settings (Cyan)
	float VideoPanelHeight = 200.0f;
	DrawPanel(LeftX, CurrentY, PanelWidth, VideoPanelHeight, TEXT("VIDEO SETTINGS"), CyanColor);
	DrawVideoSettingsPanel(LeftX, CurrentY, PanelWidth);
	CurrentY += VideoPanelHeight + PanelSpacing;

	// Output Section (Cyan)
	float OutputPanelHeight = 200.0f;
	DrawPanel(LeftX, CurrentY, PanelWidth, OutputPanelHeight, TEXT("OUTPUT"), CyanColor);
	DrawOutputPanel(LeftX, CurrentY, PanelWidth);
	CurrentY += OutputPanelHeight + PanelSpacing;

	// ===== RIGHT COLUMN =====
	float RightColumnY = ContentStartY - ScrollOffset;

	// Effects Section (Pink) - Color grading, vignette, etc.
	float EffectsPanelHeight = 320.0f;
	DrawPanel(RightX, RightColumnY, PanelWidth, EffectsPanelHeight, TEXT("VISUAL EFFECTS"), PinkColor);
	DrawEffectsPanel(RightX, RightColumnY, PanelWidth);
	RightColumnY += EffectsPanelHeight + PanelSpacing;

	// Beat Effects (Pink)
	float BeatPanelHeight = 260.0f;
	DrawPanel(RightX, RightColumnY, PanelWidth, BeatPanelHeight, TEXT("BEAT EFFECTS"), PinkColor);
	DrawBeatEffectsPanel(RightX, RightColumnY, PanelWidth);
	RightColumnY += BeatPanelHeight + PanelSpacing;

	// Transitions (Purple)
	float TransitionPanelHeight = 240.0f;
	DrawPanel(RightX, RightColumnY, PanelWidth, TransitionPanelHeight, TEXT("TRANSITIONS"), PurpleColor);
	DrawTransitionsPanel(RightX, RightColumnY, PanelWidth);
	RightColumnY += TransitionPanelHeight + PanelSpacing;

	// Track the bottom of both columns to place waveform below
	float LeftColumnBottom = CurrentY;  // CurrentY is at end of left column
	float RightColumnBottom = RightColumnY;
	float BottomPanelsY = FMath::Max(LeftColumnBottom, RightColumnBottom) + PanelSpacing;

	// ===== BOTTOM PANELS (SCROLLABLE) =====

	// Progress bar area (full width, scrolls with content)
	float ProgressPanelHeight = 70.0f;
	DrawPanel(LeftX, BottomPanelsY, ScreenWidth - 80.0f, ProgressPanelHeight, TEXT("PROGRESS"), PurpleColor);
	DrawProgressPanel(LeftX, BottomPanelsY, ScreenWidth - 80.0f);
	BottomPanelsY += ProgressPanelHeight + PanelSpacing;

	// Beat Visualizer / Waveform (Orange) - full width, scrolls with content
	float WaveformPanelHeight = 250.0f;
	DrawPanel(LeftX, BottomPanelsY, ScreenWidth - 80.0f, WaveformPanelHeight, TEXT("WAVEFORM"), OrangeColor);
	DrawVisualizerPanel(LeftX, BottomPanelsY, ScreenWidth - 80.0f);
	BottomPanelsY += WaveformPanelHeight + PanelSpacing;

	// Update content height for scroll limits - total content from top to bottom of waveform
	ContentHeight = BottomPanelsY + ScrollOffset + 50.0f;  // Add some padding at bottom
	MaxScrollOffset = FMath::Max(0.0f, ContentHeight - ScreenHeight + 100.0f);
}

// =============== CONTROL DRAWING METHODS ===============

bool ABeatSyncHUD::IsPointInRect(FVector2D Point, FVector2D RectPos, FVector2D RectSize)
{
	return Point.X >= RectPos.X && Point.X <= RectPos.X + RectSize.X &&
		   Point.Y >= RectPos.Y && Point.Y <= RectPos.Y + RectSize.Y;
}

void ABeatSyncHUD::DrawLabel(float X, float Y, const FString& Text, FLinearColor Color)
{
	if (!Canvas) return;

	UFont* Font = CustomFont ? CustomFont : GEngine->GetSmallFont();
	if (Font)
	{
		FCanvasTextItem TextItem(FVector2D(X, Y), FText::FromString(Text), Font, Color);
		TextItem.Scale = FVector2D(1.3f, 1.3f);
		TextItem.bOutlined = true;
		TextItem.OutlineColor = FLinearColor(0.0f, 0.0f, 0.0f, 0.5f);
		Canvas->DrawItem(TextItem);
	}
}

bool ABeatSyncHUD::DrawButton(float X, float Y, float Width, float Height, const FString& Label, FLinearColor AccentColor)
{
	if (!Canvas) return false;

	FVector2D Pos(X, Y);
	FVector2D Size(Width, Height);

	bool bHovered = IsPointInRect(MousePosition, Pos, Size);
	bool bPressed = bHovered && bMousePressed;
	bool bClicked = bHovered && bMouseJustReleased;

	// Button background - brighter when hovered, even brighter when pressed
	FLinearColor BG = ButtonBG;
	if (bPressed)
	{
		BG = FLinearColor(0.2f, 0.2f, 0.3f, 1.0f);
	}
	else if (bHovered)
	{
		BG = FLinearColor(0.15f, 0.15f, 0.22f, 1.0f);
	}
	Canvas->K2_DrawBox(Pos, Size, 1.0f, BG);

	// Hover glow effect - draw accent colored inner glow
	if (bHovered)
	{
		FLinearColor GlowColor = AccentColor;
		GlowColor.A = 0.3f;
		Canvas->K2_DrawBox(FVector2D(X + 2, Y + 2), FVector2D(Width - 4, Height - 4), 1.0f, GlowColor);
	}

	// Button border - full brightness when hovered
	float BorderThickness = bHovered ? 3.0f : 2.0f;
	FLinearColor BorderColor = bHovered ? AccentColor : FLinearColor(AccentColor.R, AccentColor.G, AccentColor.B, 0.5f);
	Canvas->K2_DrawLine(FVector2D(X, Y), FVector2D(X + Width, Y), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X, Y + Height), FVector2D(X + Width, Y + Height), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X, Y), FVector2D(X, Y + Height), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X + Width, Y), FVector2D(X + Width, Y + Height), BorderThickness, BorderColor);

	// Button label - accent color when hovered
	UFont* Font = CustomFont ? CustomFont : GEngine->GetSmallFont();
	if (Font)
	{
		float TextWidth, TextHeight;
		Canvas->TextSize(Font, Label, TextWidth, TextHeight);
		float TextX = X + (Width - TextWidth) / 2.0f;
		float TextY = Y + (Height - TextHeight) / 2.0f;

		FLinearColor TextColor = bHovered ? AccentColor : FLinearColor::White;
		FCanvasTextItem TextItem(FVector2D(TextX, TextY), FText::FromString(Label), Font, TextColor);
		TextItem.bOutlined = true;
		TextItem.OutlineColor = FLinearColor::Black;
		Canvas->DrawItem(TextItem);
	}

	return bClicked;
}

float ABeatSyncHUD::DrawSlider(float X, float Y, float Width, const FString& Label, float Value, float MinVal, float MaxVal, FLinearColor AccentColor)
{
	if (!Canvas) return Value;

	float SliderHeight = 10.0f;
	float HandleWidth = 16.0f;
	float HandleHeight = 22.0f;

	// Check if hovering over slider area
	float SliderY = Y + 22.0f;
	bool bSliderHovered = IsPointInRect(MousePosition, FVector2D(X - 10, SliderY - 5), FVector2D(Width + 20, HandleHeight + 10));

	// Draw label - highlight when hovered
	FLinearColor LabelColor = bSliderHovered ? AccentColor : FLinearColor::White;
	DrawLabel(X, Y, Label, LabelColor);

	// Value display
	FString ValueStr = FString::Printf(TEXT("%.1f"), Value);
	DrawLabel(X + Width - 50, Y, ValueStr, AccentColor);

	// Slider track background
	FLinearColor TrackBG = bSliderHovered ? FLinearColor(0.08f, 0.08f, 0.12f, 1.0f) : SliderBG;
	Canvas->K2_DrawBox(FVector2D(X, SliderY + (HandleHeight - SliderHeight) / 2), FVector2D(Width, SliderHeight), 1.0f, TrackBG);

	// Track border when hovered
	if (bSliderHovered)
	{
		FLinearColor BorderColor = AccentColor;
		BorderColor.A = 0.5f;
		float TrackY = SliderY + (HandleHeight - SliderHeight) / 2;
		Canvas->K2_DrawLine(FVector2D(X, TrackY), FVector2D(X + Width, TrackY), 1.0f, BorderColor);
		Canvas->K2_DrawLine(FVector2D(X, TrackY + SliderHeight), FVector2D(X + Width, TrackY + SliderHeight), 1.0f, BorderColor);
	}

	// Filled portion
	float NormalizedValue = (Value - MinVal) / (MaxVal - MinVal);
	float FilledWidth = Width * NormalizedValue;
	Canvas->K2_DrawBox(FVector2D(X, SliderY + (HandleHeight - SliderHeight) / 2), FVector2D(FilledWidth, SliderHeight), 1.0f, AccentColor);

	// Handle
	float HandleX = X + FilledWidth - HandleWidth / 2;
	FVector2D HandlePos(HandleX, SliderY);
	FVector2D HandleSize(HandleWidth, HandleHeight);

	// Handle color - white when hovered, glow effect
	FLinearColor HandleColor = bSliderHovered ? FLinearColor::White : FLinearColor(0.7f, 0.7f, 0.7f, 1.0f);
	Canvas->K2_DrawBox(HandlePos, HandleSize, 1.0f, HandleColor);

	// Handle border
	if (bSliderHovered)
	{
		Canvas->K2_DrawLine(FVector2D(HandleX, SliderY), FVector2D(HandleX + HandleWidth, SliderY), 2.0f, AccentColor);
		Canvas->K2_DrawLine(FVector2D(HandleX, SliderY + HandleHeight), FVector2D(HandleX + HandleWidth, SliderY + HandleHeight), 2.0f, AccentColor);
		Canvas->K2_DrawLine(FVector2D(HandleX, SliderY), FVector2D(HandleX, SliderY + HandleHeight), 2.0f, AccentColor);
		Canvas->K2_DrawLine(FVector2D(HandleX + HandleWidth, SliderY), FVector2D(HandleX + HandleWidth, SliderY + HandleHeight), 2.0f, AccentColor);
	}

	// Handle interaction
	if (bSliderHovered && bMousePressed)
	{
		float NewNormalized = FMath::Clamp((MousePosition.X - X) / Width, 0.0f, 1.0f);
		return MinVal + NewNormalized * (MaxVal - MinVal);
	}

	return Value;
}

bool ABeatSyncHUD::DrawCheckbox(float X, float Y, const FString& Label, bool bChecked, FLinearColor AccentColor)
{
	if (!Canvas) return bChecked;

	float BoxSize = 22.0f;
	FVector2D BoxPos(X, Y);
	FVector2D BoxSizeVec(BoxSize, BoxSize);

	// Check hover over entire checkbox area including label
	bool bHovered = IsPointInRect(MousePosition, FVector2D(X, Y), FVector2D(BoxSize + 200, BoxSize));
	bool bClicked = bHovered && bMouseJustReleased;

	// Checkbox background - glow when hovered
	FLinearColor BG = bChecked ? AccentColor : SliderBG;
	if (bHovered && !bChecked)
	{
		BG = FLinearColor(0.1f, 0.1f, 0.15f, 1.0f);
	}
	Canvas->K2_DrawBox(BoxPos, BoxSizeVec, 1.0f, BG);

	// Checkbox border - thicker and brighter when hovered
	float BorderThickness = bHovered ? 2.5f : 1.5f;
	FLinearColor BorderColor = bHovered ? FLinearColor::White : AccentColor;
	Canvas->K2_DrawLine(FVector2D(X, Y), FVector2D(X + BoxSize, Y), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X, Y + BoxSize), FVector2D(X + BoxSize, Y + BoxSize), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X, Y), FVector2D(X, Y + BoxSize), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X + BoxSize, Y), FVector2D(X + BoxSize, Y + BoxSize), BorderThickness, BorderColor);

	// Checkmark
	if (bChecked)
	{
		Canvas->K2_DrawLine(FVector2D(X + 4, Y + BoxSize / 2), FVector2D(X + BoxSize / 3 + 2, Y + BoxSize - 5), 3.0f, FLinearColor::White);
		Canvas->K2_DrawLine(FVector2D(X + BoxSize / 3 + 2, Y + BoxSize - 5), FVector2D(X + BoxSize - 4, Y + 5), 3.0f, FLinearColor::White);
	}

	// Label - highlight when hovered
	FLinearColor LabelColor = bHovered ? AccentColor : FLinearColor::White;
	DrawLabel(X + BoxSize + 10, Y + 3, Label, LabelColor);

	return bClicked ? !bChecked : bChecked;
}

int32 ABeatSyncHUD::DrawDropdown(float X, float Y, float Width, const FString& Label, const TArray<FString>& Options, int32 SelectedIdx, FLinearColor AccentColor)
{
	if (!Canvas || Options.Num() == 0) return SelectedIdx;

	float DropdownHeight = 36.0f;

	// Label - highlight when hovered
	float DropdownY = Y + 24.0f;
	FVector2D DropdownPos(X, DropdownY);
	FVector2D DropdownSize(Width, DropdownHeight);
	bool bHovered = IsPointInRect(MousePosition, DropdownPos, DropdownSize);

	FLinearColor LabelColor = bHovered ? AccentColor : FLinearColor::White;
	DrawLabel(X, Y, Label, LabelColor);

	// Background - brighter when hovered with glow
	FLinearColor BG = bHovered ? FLinearColor(0.15f, 0.15f, 0.22f, 1.0f) : ButtonBG;
	Canvas->K2_DrawBox(DropdownPos, DropdownSize, 1.0f, BG);

	// Inner glow when hovered
	if (bHovered)
	{
		FLinearColor GlowColor = AccentColor;
		GlowColor.A = 0.2f;
		Canvas->K2_DrawBox(FVector2D(X + 2, DropdownY + 2), FVector2D(Width - 4, DropdownHeight - 4), 1.0f, GlowColor);
	}

	// Border - thicker when hovered
	float BorderThickness = bHovered ? 2.5f : 1.5f;
	FLinearColor BorderColor = bHovered ? AccentColor : FLinearColor(AccentColor.R, AccentColor.G, AccentColor.B, 0.5f);
	Canvas->K2_DrawLine(FVector2D(X, DropdownY), FVector2D(X + Width, DropdownY), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X, DropdownY + DropdownHeight), FVector2D(X + Width, DropdownY + DropdownHeight), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X, DropdownY), FVector2D(X, DropdownY + DropdownHeight), BorderThickness, BorderColor);
	Canvas->K2_DrawLine(FVector2D(X + Width, DropdownY), FVector2D(X + Width, DropdownY + DropdownHeight), BorderThickness, BorderColor);

	// Selected option text - highlight when hovered
	FString SelectedText = Options.IsValidIndex(SelectedIdx) ? Options[SelectedIdx] : TEXT("Select...");
	FLinearColor TextColor = bHovered ? AccentColor : FLinearColor::White;
	DrawLabel(X + 12, DropdownY + 8, SelectedText, TextColor);

	// Arrow indicator - animate/highlight when hovered
	float ArrowX = X + Width - 28;
	float ArrowY = DropdownY + DropdownHeight / 2;
	FLinearColor ArrowColor = bHovered ? FLinearColor::White : AccentColor;
	float ArrowThickness = bHovered ? 3.0f : 2.0f;

	// Draw left/right arrows to indicate cycling
	// Left arrow <
	Canvas->K2_DrawLine(FVector2D(ArrowX - 8, ArrowY), FVector2D(ArrowX - 2, ArrowY - 5), ArrowThickness, ArrowColor);
	Canvas->K2_DrawLine(FVector2D(ArrowX - 8, ArrowY), FVector2D(ArrowX - 2, ArrowY + 5), ArrowThickness, ArrowColor);
	// Right arrow >
	Canvas->K2_DrawLine(FVector2D(ArrowX + 18, ArrowY), FVector2D(ArrowX + 12, ArrowY - 5), ArrowThickness, ArrowColor);
	Canvas->K2_DrawLine(FVector2D(ArrowX + 18, ArrowY), FVector2D(ArrowX + 12, ArrowY + 5), ArrowThickness, ArrowColor);

	// Show option count indicator
	FString CountText = FString::Printf(TEXT("%d/%d"), SelectedIdx + 1, Options.Num());
	DrawLabel(X + Width - 70, DropdownY + 8, CountText, FLinearColor(0.5f, 0.5f, 0.5f, 1.0f));

	// Cycle through options on click
	if (bHovered && bMouseJustReleased)
	{
		return (SelectedIdx + 1) % Options.Num();
	}

	return SelectedIdx;
}

void ABeatSyncHUD::DrawProgressBar(float X, float Y, float Width, float Height, float Progress, FLinearColor Color)
{
	if (!Canvas) return;

	// Background
	Canvas->K2_DrawBox(FVector2D(X, Y), FVector2D(Width, Height), 1.0f, SliderBG);

	// Progress fill
	float FillWidth = Width * FMath::Clamp(Progress, 0.0f, 1.0f);
	if (FillWidth > 0)
	{
		Canvas->K2_DrawBox(FVector2D(X, Y), FVector2D(FillWidth, Height), 1.0f, Color);
	}

	// Border
	Canvas->K2_DrawLine(FVector2D(X, Y), FVector2D(X + Width, Y), 1.0f, Color);
	Canvas->K2_DrawLine(FVector2D(X, Y + Height), FVector2D(X + Width, Y + Height), 1.0f, Color);
	Canvas->K2_DrawLine(FVector2D(X, Y), FVector2D(X, Y + Height), 1.0f, Color);
	Canvas->K2_DrawLine(FVector2D(X + Width, Y), FVector2D(X + Width, Y + Height), 1.0f, Color);
}

// =============== PANEL CONTENT DRAWING ===============

void ABeatSyncHUD::DrawInputPanel(float X, float Y, float Width)
{
	if (!Canvas) return;
	if (Y + 200 < 0 || Y > Canvas->ClipY) return; // Skip if off screen

	float ContentX = X + 15;
	float ContentY = Y + 40;
	float ButtonWidth = Width - 30;

	// Update status message timer
	if (StatusMessageTimer > 0.0f)
	{
		StatusMessageTimer -= GetWorld()->GetDeltaSeconds();
	}

	// Audio file section
	DrawLabel(ContentX, ContentY, TEXT("Audio File:"), CyanColor);
	ContentY += 20;

	// Truncate path if too long
	FString DisplayAudioPath = AudioFilePath;
	if (DisplayAudioPath.Len() > 30) DisplayAudioPath = TEXT("...") + DisplayAudioPath.Right(27);
	DrawLabel(ContentX, ContentY, DisplayAudioPath, FLinearColor(0.6f, 0.6f, 0.6f, 1.0f));
	ContentY += 22;

	// Browse and Analyze buttons side by side
	float HalfWidth = (ButtonWidth - 10) / 2;

	if (DrawButton(ContentX, ContentY, HalfWidth, 32, TEXT("Browse Audio..."), CyanColor))
	{
		FString Result = OpenFileDialog(
			TEXT("Select Audio File"),
			FPaths::GetProjectFilePath(),
			TEXT("Audio Files (*.mp3;*.wav;*.flac;*.aac;*.ogg)|*.mp3;*.wav;*.flac;*.aac;*.ogg|All Files (*.*)|*.*")
		);
		if (!Result.IsEmpty())
		{
			AudioFilePath = Result;
			StatusMessage = TEXT("Audio file selected! Click Analyze.");
			StatusMessageTimer = 3.0f;
			bHasRealWaveform = false; // Reset until analyzed
		}
		else
		{
			// Show message that file dialogs aren't available in packaged builds
			StatusMessage = TEXT("File dialogs not available in packaged build");
			StatusMessageTimer = 3.0f;
		}
	}

	// Analyze button
	FString AnalyzeText = bIsAnalyzing ? TEXT("Analyzing...") : TEXT("Analyze");
	FLinearColor AnalyzeColor = bIsAnalyzing ? OrangeColor : PurpleColor;
	if (DrawButton(ContentX + HalfWidth + 10, ContentY, HalfWidth, 32, AnalyzeText, AnalyzeColor))
	{
		if (!bIsAnalyzing)
		{
			AnalyzeSelectedAudio();
		}
	}
	ContentY += 45;

	// Video file section
	DrawLabel(ContentX, ContentY, TEXT("Video File:"), CyanColor);
	ContentY += 20;

	FString DisplayVideoPath = VideoFilePath;
	if (DisplayVideoPath.Len() > 30) DisplayVideoPath = TEXT("...") + DisplayVideoPath.Right(27);
	DrawLabel(ContentX, ContentY, DisplayVideoPath, FLinearColor(0.6f, 0.6f, 0.6f, 1.0f));
	ContentY += 22;

	if (DrawButton(ContentX, ContentY, ButtonWidth, 32, TEXT("Browse Video File..."), CyanColor))
	{
		FString Result = OpenFileDialog(
			TEXT("Select Video File"),
			FPaths::GetProjectFilePath(),
			TEXT("Video Files (*.mp4;*.mov;*.avi;*.mkv;*.webm)|*.mp4;*.mov;*.avi;*.mkv;*.webm|All Files (*.*)|*.*")
		);
		if (!Result.IsEmpty())
		{
			VideoFilePath = Result;
			StatusMessage = TEXT("Video file selected!");
			StatusMessageTimer = 3.0f;
		}
		else
		{
			StatusMessage = TEXT("Drag & drop files onto window (coming soon)");
			StatusMessageTimer = 3.0f;
		}
	}

	// Show status message if active
	if (StatusMessageTimer > 0.0f && !StatusMessage.IsEmpty())
	{
		ContentY += 40;
		FLinearColor MsgColor = OrangeColor;
		MsgColor.A = FMath::Min(1.0f, StatusMessageTimer);
		DrawLabel(ContentX, ContentY, StatusMessage, MsgColor);
	}
}

void ABeatSyncHUD::DrawAnalysisPanel(float X, float Y, float Width)
{
	if (!Canvas) return;
	if (Y + 260 < 0 || Y > Canvas->ClipY) return;

	float ContentX = X + 15;
	float ContentY = Y + 40;
	float ControlWidth = Width - 30;

	// BPM Slider
	BPMValue = DrawSlider(ContentX, ContentY, ControlWidth, TEXT("BPM"), BPMValue, 60.0f, 200.0f, PurpleColor);
	ContentY += 60;

	// Sensitivity Slider
	SensitivityValue = DrawSlider(ContentX, ContentY, ControlWidth, TEXT("Sensitivity"), SensitivityValue, 0.0f, 1.0f, PurpleColor);
	ContentY += 60;

	// Beat Detection Mode Dropdown
	TArray<FString> DetectionModes = { TEXT("Auto Detect"), TEXT("Manual BPM"), TEXT("Tap Tempo") };
	BeatDetectionMode = DrawDropdown(ContentX, ContentY, ControlWidth, TEXT("Detection Mode"), DetectionModes, BeatDetectionMode, PurpleColor);
	ContentY += 70;

	// Auto analyze checkbox
	bAnalyzeOnLoad = DrawCheckbox(ContentX, ContentY, TEXT("Analyze on file load"), bAnalyzeOnLoad, PurpleColor);
}

void ABeatSyncHUD::DrawVideoSettingsPanel(float X, float Y, float Width)
{
	if (!Canvas) return;
	if (Y + 200 < 0 || Y > Canvas->ClipY) return;

	float ContentX = X + 15;
	float ContentY = Y + 40;
	float ControlWidth = (Width - 50) / 2;

	// Resolution Dropdown
	TArray<FString> Resolutions = { TEXT("1920x1080"), TEXT("1280x720"), TEXT("3840x2160") };
	ResolutionIndex = DrawDropdown(ContentX, ContentY, ControlWidth, TEXT("Resolution"), Resolutions, ResolutionIndex, CyanColor);

	// Framerate Dropdown
	TArray<FString> Framerates = { TEXT("30 fps"), TEXT("60 fps"), TEXT("24 fps") };
	FramerateIndex = DrawDropdown(ContentX + ControlWidth + 20, ContentY, ControlWidth, TEXT("Framerate"), Framerates, FramerateIndex, CyanColor);

	ContentY += 80;

	// Maintain aspect ratio
	bMaintainAspect = DrawCheckbox(ContentX, ContentY, TEXT("Maintain aspect ratio"), bMaintainAspect, CyanColor);
}

void ABeatSyncHUD::DrawOutputPanel(float X, float Y, float Width)
{
	if (!Canvas) return;
	if (Y + 160 < 0 || Y > Canvas->ClipY) return;

	float ContentX = X + 15;
	float ContentY = Y + 35;
	float ButtonWidth = Width - 30;

	// Output path
	DrawLabel(ContentX, ContentY, TEXT("Output Path:"), CyanColor);
	ContentY += 18;

	FString DisplayPath = OutputPath;
	if (DisplayPath.Len() > 35) DisplayPath = TEXT("...") + DisplayPath.Right(32);
	DrawLabel(ContentX, ContentY, DisplayPath, FLinearColor(0.6f, 0.6f, 0.6f, 1.0f));
	ContentY += 20;

	if (DrawButton(ContentX, ContentY, ButtonWidth, 28, TEXT("Choose Output Location..."), CyanColor))
	{
		FString Result = SaveFileDialog(
			TEXT("Save Output Video"),
			FPaths::GetProjectFilePath(),
			TEXT("beatsync_output.mp4"),
			TEXT("MP4 Video (*.mp4)|*.mp4|MOV Video (*.mov)|*.mov|AVI Video (*.avi)|*.avi|All Files (*.*)|*.*")
		);
		if (!Result.IsEmpty())
		{
			OutputPath = Result;
		}
	}
	ContentY += 40;

	// Format dropdown
	TArray<FString> Formats = { TEXT("MP4 (H.264)"), TEXT("MOV (ProRes)"), TEXT("AVI") };
	OutputFormat = DrawDropdown(ContentX, ContentY, ButtonWidth, TEXT("Output Format"), Formats, OutputFormat, CyanColor);
}

void ABeatSyncHUD::DrawEffectsPanel(float X, float Y, float Width)
{
	if (!Canvas) return;
	if (Y + 320 < 0 || Y > Canvas->ClipY) return;

	float ContentX = X + 15;
	float ContentY = Y + 40;
	float ControlWidth = Width - 30;

	// Brightness
	BrightnessValue = DrawSlider(ContentX, ContentY, ControlWidth, TEXT("Brightness"), BrightnessValue, 0.0f, 2.0f, PinkColor);
	ContentY += 55;

	// Contrast
	ContrastValue = DrawSlider(ContentX, ContentY, ControlWidth, TEXT("Contrast"), ContrastValue, 0.0f, 2.0f, PinkColor);
	ContentY += 55;

	// Saturation
	SaturationValue = DrawSlider(ContentX, ContentY, ControlWidth, TEXT("Saturation"), SaturationValue, 0.0f, 2.0f, PinkColor);
	ContentY += 55;

	// Vignette
	VignetteValue = DrawSlider(ContentX, ContentY, ControlWidth, TEXT("Vignette"), VignetteValue, 0.0f, 1.0f, PinkColor);
	ContentY += 60;

	// Effect toggles
	bEnableGlow = DrawCheckbox(ContentX, ContentY, TEXT("Enable Glow"), bEnableGlow, PinkColor);
	ContentY += 35;
	bEnableChromatic = DrawCheckbox(ContentX, ContentY, TEXT("Chromatic Aberration"), bEnableChromatic, PinkColor);
}

void ABeatSyncHUD::DrawBeatEffectsPanel(float X, float Y, float Width)
{
	if (!Canvas) return;
	if (Y + 260 < 0 || Y > Canvas->ClipY) return;

	float ContentX = X + 15;
	float ContentY = Y + 40;
	float ControlWidth = Width - 30;

	// Effect Type
	TArray<FString> EffectTypes = { TEXT("Flash"), TEXT("Zoom Pulse"), TEXT("Shake"), TEXT("Color Shift") };
	BeatEffectType = DrawDropdown(ContentX, ContentY, ControlWidth, TEXT("Beat Effect Type"), EffectTypes, BeatEffectType, PinkColor);
	ContentY += 75;

	// Intensity
	BeatIntensity = DrawSlider(ContentX, ContentY, ControlWidth, TEXT("Intensity"), BeatIntensity, 0.0f, 1.0f, PinkColor);
	ContentY += 60;

	// Sync options
	bSyncToKick = DrawCheckbox(ContentX, ContentY, TEXT("Sync to Kick"), bSyncToKick, PinkColor);
	ContentY += 35;
	bSyncToSnare = DrawCheckbox(ContentX, ContentY, TEXT("Sync to Snare"), bSyncToSnare, PinkColor);
}

void ABeatSyncHUD::DrawTransitionsPanel(float X, float Y, float Width)
{
	if (!Canvas) return;
	if (Y + 240 < 0 || Y > Canvas->ClipY) return;

	float ContentX = X + 15;
	float ContentY = Y + 40;
	float ControlWidth = Width - 30;

	// Transition Type
	TArray<FString> TransitionTypes = { TEXT("Cut"), TEXT("Fade"), TEXT("Wipe"), TEXT("Dissolve") };
	TransitionType = DrawDropdown(ContentX, ContentY, ControlWidth, TEXT("Transition Type"), TransitionTypes, TransitionType, PurpleColor);
	ContentY += 75;

	// Duration
	TransitionDuration = DrawSlider(ContentX, ContentY, ControlWidth, TEXT("Duration (sec)"), TransitionDuration, 0.1f, 2.0f, PurpleColor);
	ContentY += 60;

	// Auto transitions
	bAutoTransitions = DrawCheckbox(ContentX, ContentY, TEXT("Auto transitions on beat"), bAutoTransitions, PurpleColor);
}

void ABeatSyncHUD::DrawProgressPanel(float X, float Y, float Width)
{
	if (!Canvas) return;

	float ContentX = X + 15;
	float ContentY = Y + 30;
	float BarWidth = Width - 200;

	// Progress bar
	DrawProgressBar(ContentX, ContentY, BarWidth, 16, ProcessingProgress, PurpleColor);

	// Percentage text
	FString ProgressText = FString::Printf(TEXT("%.0f%%"), ProcessingProgress * 100.0f);
	DrawLabel(ContentX + BarWidth + 10, ContentY, ProgressText, FLinearColor::White);

	// Process button
	FString ButtonText = bIsProcessing ? TEXT("Cancel") : TEXT("START PROCESSING");
	FLinearColor ButtonColor = bIsProcessing ? OrangeColor : CyanColor;
	if (DrawButton(ContentX + BarWidth + 70, ContentY - 5, 150, 26, ButtonText, ButtonColor))
	{
		bIsProcessing = !bIsProcessing;
		if (!bIsProcessing) ProcessingProgress = 0.0f;
	}

	// Simulate progress for demo
	if (bIsProcessing)
	{
		ProcessingProgress = FMath::Min(1.0f, ProcessingProgress + 0.001f);
		if (ProcessingProgress >= 1.0f)
		{
			bIsProcessing = false;
		}
	}
}

void ABeatSyncHUD::DrawVisualizerPanel(float X, float Y, float Width)
{
	if (!Canvas) return;

	float ContentX = X + 15;
	float ContentY = Y + 35;
	float VizWidth = Width - 130;  // Leave room for zoom controls
	float VizHeight = 120.0f;      // Taller like Rekordbox
	float CenterY = ContentY + VizHeight / 2.0f;

	// Store waveform bounds for touchpad/scroll detection
	WaveformPanelMin = FVector2D(ContentX, ContentY);
	WaveformPanelMax = FVector2D(ContentX + VizWidth, ContentY + VizHeight);

	// === ZOOM CONTROLS ===
	float ZoomX = X + Width - 110;
	float ZoomY = ContentY;

	DrawLabel(ZoomX, ZoomY - 18, TEXT("Zoom"), OrangeColor);

	if (DrawButton(ZoomX, ZoomY, 40, 28, TEXT("-"), OrangeColor))
	{
		WaveformZoom = FMath::Max(1.0f, WaveformZoom - 0.5f);
		// Adjust scroll to keep centered when zooming out
		float VisibleFrac = 1.0f / WaveformZoom;
		WaveformScrollX = FMath::Clamp(WaveformScrollX, 0.0f, 1.0f - VisibleFrac);
	}

	if (DrawButton(ZoomX + 50, ZoomY, 40, 28, TEXT("+"), OrangeColor))
	{
		WaveformZoom = FMath::Min(8.0f, WaveformZoom + 0.5f);
	}

	FString ZoomText = FString::Printf(TEXT("%.1fx"), WaveformZoom);
	DrawLabel(ZoomX + 12, ZoomY + 35, ZoomText, FLinearColor::White);

	// === SCROLL CONTROLS (only show when zoomed in) ===
	if (WaveformZoom > 1.0f)
	{
		float ScrollY = ZoomY + 55;
		DrawLabel(ZoomX, ScrollY, TEXT("Scroll"), FLinearColor(0.6f, 0.6f, 0.6f, 1.0f));

		float ScrollStep = 0.1f / WaveformZoom;  // Smaller steps when more zoomed
		float MaxScroll = 1.0f - (1.0f / WaveformZoom);

		if (DrawButton(ZoomX, ScrollY + 18, 40, 24, TEXT("<"), OrangeColor))
		{
			WaveformScrollX = FMath::Max(0.0f, WaveformScrollX - ScrollStep);
		}

		if (DrawButton(ZoomX + 50, ScrollY + 18, 40, 24, TEXT(">"), OrangeColor))
		{
			WaveformScrollX = FMath::Min(MaxScroll, WaveformScrollX + ScrollStep);
		}
	}

	FString BPMText = FString::Printf(TEXT("%.0f BPM"), BPMValue);
	float BPMTextY = (WaveformZoom > 1.0f) ? ZoomY + 105 : ZoomY + 55;
	DrawLabel(ZoomX, BPMTextY, BPMText, OrangeColor);

	// === WAVEFORM BACKGROUND ===
	Canvas->K2_DrawBox(FVector2D(ContentX, ContentY), FVector2D(VizWidth, VizHeight), 1.0f, SliderBG);

	// === SUBTLE CENTER LINE ===
	FLinearColor CenterLineColor = FLinearColor(0.15f, 0.15f, 0.18f, 0.5f);
	Canvas->K2_DrawLine(FVector2D(ContentX, CenterY), FVector2D(ContentX + VizWidth, CenterY), 1.0f, CenterLineColor);

	// === CALCULATE BEAT POSITIONS ===
	float SecondsPerBeat = 60.0f / BPMValue;
	int32 TotalBeats = FMath::FloorToInt(TotalAudioDuration / SecondsPerBeat);

	// === REKORDBOX-STYLE WAVEFORM WITH FREQUENCY COLORING ===
	// Red/Orange = Bass (low freq), Blue/Cyan = Highs, White = Mids
	int32 TotalSamples = bHasRealWaveform ? CachedWaveform.Num() : 1024;
	if (TotalSamples < 1) TotalSamples = 1024;
	float HalfHeight = VizHeight / 2.0f - 2.0f;

	// === APPLY ZOOM: Calculate visible window ===
	// WaveformZoom of 1.0 = show all, 2.0 = show half, etc.
	float VisibleFraction = 1.0f / WaveformZoom;
	float MaxScrollX = 1.0f - VisibleFraction;
	WaveformScrollX = FMath::Clamp(WaveformScrollX, 0.0f, FMath::Max(0.0f, MaxScrollX));

	// Calculate visible sample range
	int32 VisibleSampleCount = FMath::Max(1, FMath::CeilToInt(TotalSamples * VisibleFraction));
	int32 StartSample = FMath::FloorToInt(WaveformScrollX * TotalSamples);
	StartSample = FMath::Clamp(StartSample, 0, TotalSamples - VisibleSampleCount);
	int32 EndSample = FMath::Min(StartSample + VisibleSampleCount, TotalSamples);

	// How many pixels per sample when zoomed
	int32 NumSamplesToRender = EndSample - StartSample;
	float SampleWidth = VizWidth / FMath::Max(1, NumSamplesToRender);

	// Show indicator if using real vs simulated waveform
	FString WaveformType = bHasRealWaveform ? TEXT("REAL AUDIO") : TEXT("SIMULATED");
	DrawLabel(ContentX + VizWidth - 100, ContentY + VizHeight + 25, WaveformType,
		bHasRealWaveform ? FLinearColor::Green : FLinearColor(0.5f, 0.5f, 0.5f, 1.0f));

	for (int32 i = 0; i < NumSamplesToRender; i++)
	{
		int32 SampleIdx = StartSample + i;
		float NormPos = (float)SampleIdx / TotalSamples;
		float TimeInSeconds = NormPos * TotalAudioDuration;

		float LowFreq, MidFreq, HighFreq, TotalAmplitude;

		// === USE REAL WAVEFORM DATA IF AVAILABLE ===
		if (bHasRealWaveform && SampleIdx < CachedWaveform.Num())
		{
			const FWaveformSample& Sample = CachedWaveform[SampleIdx];
			LowFreq = Sample.LowFreq;
			MidFreq = Sample.MidFreq;
			HighFreq = Sample.HighFreq;
			TotalAmplitude = Sample.Amplitude;
		}
		else
		{
			// === SIMULATED FREQUENCY BANDS (fallback) ===
			float BeatPhase = FMath::Fmod(TimeInSeconds, SecondsPerBeat) / SecondsPerBeat;
			int32 BeatInMeasure = FMath::FloorToInt(FMath::Fmod(TimeInSeconds / SecondsPerBeat, 4.0f));

			// LOW FREQUENCIES (Kick, Bass)
			LowFreq = 0.0f;
			if (BeatPhase < 0.06f)
			{
				LowFreq = FMath::Pow(1.0f - BeatPhase / 0.06f, 0.3f);
			}
			if (BeatPhase < 0.4f)
			{
				LowFreq = FMath::Max(LowFreq, 0.5f * FMath::Pow(1.0f - BeatPhase / 0.4f, 1.5f));
			}

			// MID FREQUENCIES (Snare, Vocals, Synths)
			MidFreq = 0.0f;
			if ((BeatInMeasure == 1 || BeatInMeasure == 3) && BeatPhase < 0.1f)
			{
				MidFreq = 0.85f * FMath::Pow(1.0f - BeatPhase / 0.1f, 0.5f);
			}
			float SynthPhase = FMath::Fmod(TimeInSeconds * 2.0f, 1.0f);
			MidFreq = FMath::Max(MidFreq, 0.15f + 0.1f * FMath::Sin(SynthPhase * PI * 2.0f));

			// HIGH FREQUENCIES (Hi-hats, Cymbals)
			HighFreq = 0.0f;
			float HiHatPhase = FMath::Fmod(TimeInSeconds, SecondsPerBeat / 2.0f) / (SecondsPerBeat / 2.0f);
			if (HiHatPhase < 0.03f)
			{
				HighFreq = 0.6f * FMath::Pow(1.0f - HiHatPhase / 0.03f, 0.4f);
			}
			if ((BeatInMeasure == 0 || BeatInMeasure == 2) && BeatPhase > 0.45f && BeatPhase < 0.55f)
			{
				float OpenHHPhase = (BeatPhase - 0.45f) / 0.1f;
				HighFreq = FMath::Max(HighFreq, 0.4f * (1.0f - OpenHHPhase));
			}
			float SixteenthPhase = FMath::Fmod(TimeInSeconds, SecondsPerBeat / 4.0f) / (SecondsPerBeat / 4.0f);
			if (SixteenthPhase < 0.02f)
			{
				HighFreq = FMath::Max(HighFreq, 0.25f * (1.0f - SixteenthPhase / 0.02f));
			}

			// Add noise texture
			float NoisePhase = NormPos * 2000.0f + TimeInSeconds * 100.0f;
			float Noise1 = FMath::Sin(NoisePhase * 7.13f) * FMath::Sin(NoisePhase * 11.7f);
			float Noise2 = FMath::Sin(NoisePhase * 13.3f) * FMath::Sin(NoisePhase * 17.1f);
			float Noise3 = FMath::Sin(NoisePhase * 19.7f) * FMath::Sin(NoisePhase * 23.3f);

			LowFreq += FMath::Abs(Noise1) * 0.08f * (0.5f + LowFreq);
			MidFreq += FMath::Abs(Noise2) * 0.06f * (0.5f + MidFreq);
			HighFreq += FMath::Abs(Noise3) * 0.1f * (0.5f + HighFreq);

			TotalAmplitude = FMath::Clamp(LowFreq * 0.5f + MidFreq * 0.3f + HighFreq * 0.2f, 0.02f, 1.0f);
		}

		// Clamp values
		LowFreq = FMath::Clamp(LowFreq, 0.0f, 1.0f);
		MidFreq = FMath::Clamp(MidFreq, 0.0f, 1.0f);
		HighFreq = FMath::Clamp(HighFreq, 0.0f, 1.0f);
		TotalAmplitude = FMath::Clamp(TotalAmplitude, 0.02f, 1.0f);

		// Stereo variation
		float NoisePhase = NormPos * 2000.0f;
		float StereoNoise = FMath::Sin(NoisePhase * 3.7f) * 0.15f;
		float TopHeight = TotalAmplitude * HalfHeight * (1.0f + StereoNoise);
		float BottomHeight = TotalAmplitude * HalfHeight * (1.0f - StereoNoise * 0.5f);

		float BarX = ContentX + i * SampleWidth;

		// === REKORDBOX-STYLE COLOR MIXING ===
		// Red (bass) -> White (mids) -> Blue (highs)
		FLinearColor WaveColor;

		// Check selection regions first
		bool bInEffectRegion = (NormPos >= EffectStartTime && NormPos <= EffectEndTime);
		bool bInVideoRegion = (NormPos >= VideoStartTime && NormPos <= VideoEndTime);

		if (bInEffectRegion)
		{
			// Pink/Magenta for effect region
			float FreqBlend = HighFreq / FMath::Max(0.01f, LowFreq + MidFreq + HighFreq);
			WaveColor = FLinearColor(
				1.0f,
				0.2f + FreqBlend * 0.4f,
				0.6f + FreqBlend * 0.4f,
				1.0f
			);
		}
		else if (bInVideoRegion)
		{
			// Cyan-tinted for video region
			float FreqBlend = LowFreq / FMath::Max(0.01f, LowFreq + MidFreq + HighFreq);
			WaveColor = FLinearColor(
				FreqBlend * 0.5f,
				0.7f + MidFreq * 0.3f,
				0.9f + HighFreq * 0.1f,
				1.0f
			);
		}
		else
		{
			// REKORDBOX COLORS: Red (low) -> Orange -> White (mid) -> Cyan -> Blue (high)
			float TotalFreq = LowFreq + MidFreq + HighFreq + 0.001f;
			float LowRatio = LowFreq / TotalFreq;
			float MidRatio = MidFreq / TotalFreq;
			float HighRatio = HighFreq / TotalFreq;

			// Base color from frequency content
			float R = LowRatio * 1.0f + MidRatio * 0.95f + HighRatio * 0.3f;
			float G = LowRatio * 0.3f + MidRatio * 0.9f + HighRatio * 0.7f;
			float B = LowRatio * 0.1f + MidRatio * 0.85f + HighRatio * 1.0f;

			// Boost brightness for transients
			float Brightness = 0.7f + TotalAmplitude * 0.5f;
			R *= Brightness;
			G *= Brightness;
			B *= Brightness;

			WaveColor = FLinearColor(
				FMath::Clamp(R, 0.0f, 1.0f),
				FMath::Clamp(G, 0.0f, 1.0f),
				FMath::Clamp(B, 0.0f, 1.0f),
				1.0f
			);
		}

		// === DRAW LAYERED WAVEFORM (Rekordbox style - multiple passes for glow) ===
		float BarW = FMath::Max(1.0f, SampleWidth);

		// Outer glow (dimmer, wider)
		FLinearColor GlowColor = WaveColor;
		GlowColor.A = 0.15f;
		float GlowExtra = 2.0f;
		Canvas->K2_DrawBox(
			FVector2D(BarX - 0.5f, CenterY - TopHeight - GlowExtra),
			FVector2D(BarW + 1.0f, TopHeight + GlowExtra),
			1.0f, GlowColor
		);
		Canvas->K2_DrawBox(
			FVector2D(BarX - 0.5f, CenterY),
			FVector2D(BarW + 1.0f, BottomHeight + GlowExtra),
			1.0f, GlowColor
		);

		// Main waveform
		Canvas->K2_DrawBox(
			FVector2D(BarX, CenterY - TopHeight),
			FVector2D(BarW, TopHeight),
			1.0f, WaveColor
		);
		Canvas->K2_DrawBox(
			FVector2D(BarX, CenterY),
			FVector2D(BarW, BottomHeight),
			1.0f, WaveColor
		);

		// Bright center highlight for loud transients
		if (TotalAmplitude > 0.6f)
		{
			FLinearColor HighlightColor = FLinearColor::White;
			HighlightColor.A = (TotalAmplitude - 0.6f) * 1.5f;
			float HighlightHeight = TopHeight * 0.3f;
			Canvas->K2_DrawBox(
				FVector2D(BarX, CenterY - HighlightHeight),
				FVector2D(BarW, HighlightHeight * 2.0f),
				1.0f, HighlightColor
			);
		}
	}

	// === BEAT MARKERS (subtle, on top) ===
	for (int32 BeatIdx = 0; BeatIdx <= TotalBeats; BeatIdx++)
	{
		float BeatTime = BeatIdx * SecondsPerBeat;
		float BeatNormPos = BeatTime / TotalAudioDuration;
		float BeatX = ContentX + BeatNormPos * VizWidth;

		if (BeatX >= ContentX && BeatX <= ContentX + VizWidth)
		{
			bool bIsMeasure = (BeatIdx % 4 == 0);

			// Subtle beat line
			FLinearColor BeatLineColor = bIsMeasure ?
				FLinearColor(1.0f, 0.5f, 0.0f, 0.4f) :  // Orange for measures
				FLinearColor(0.5f, 0.5f, 0.5f, 0.2f);   // Gray for beats

			Canvas->K2_DrawLine(
				FVector2D(BeatX, ContentY + 2),
				FVector2D(BeatX, ContentY + VizHeight - 2),
				bIsMeasure ? 1.5f : 0.5f,
				BeatLineColor
			);

			// Measure number at top
			if (bIsMeasure && BeatIdx > 0)
			{
				int32 MeasureNum = BeatIdx / 4;
				FString MeasureText = FString::Printf(TEXT("%d"), MeasureNum);
				DrawLabel(BeatX - 4, ContentY + 3, MeasureText, FLinearColor(0.6f, 0.4f, 0.2f, 0.8f));
			}
		}
	}

	// === WAVEFORM BORDER (subtle frame) ===
	FLinearColor BorderColor = FLinearColor(0.2f, 0.2f, 0.25f, 0.5f);
	Canvas->K2_DrawLine(FVector2D(ContentX, ContentY), FVector2D(ContentX + VizWidth, ContentY), 1.0f, BorderColor);
	Canvas->K2_DrawLine(FVector2D(ContentX, ContentY + VizHeight), FVector2D(ContentX + VizWidth, ContentY + VizHeight), 1.0f, BorderColor);
	Canvas->K2_DrawLine(FVector2D(ContentX, ContentY), FVector2D(ContentX, ContentY + VizHeight), 1.0f, BorderColor);
	Canvas->K2_DrawLine(FVector2D(ContentX + VizWidth, ContentY), FVector2D(ContentX + VizWidth, ContentY + VizHeight), 1.0f, BorderColor);

	// === VIDEO SELECTION REGION (CYAN) ===
	float VideoStartX = ContentX + VideoStartTime * VizWidth;
	float VideoEndX = ContentX + VideoEndTime * VizWidth;

	// Draw selection overlay
	FLinearColor VideoOverlay = CyanColor;
	VideoOverlay.A = 0.15f;
	Canvas->K2_DrawBox(FVector2D(VideoStartX, ContentY), FVector2D(VideoEndX - VideoStartX, VizHeight), 1.0f, VideoOverlay);

	// Draw handles
	float HandleWidth = 8.0f;
	float HandleHeight = VizHeight + 10.0f;

	// Video start handle
	FVector2D VSHandlePos(VideoStartX - HandleWidth/2, ContentY - 5);
	FVector2D VSHandleSize(HandleWidth, HandleHeight);
	bool bVSHovered = IsPointInRect(MousePosition, VSHandlePos, VSHandleSize);
	Canvas->K2_DrawBox(VSHandlePos, VSHandleSize, 1.0f, bVSHovered ? FLinearColor::White : CyanColor);

	// Video end handle
	FVector2D VEHandlePos(VideoEndX - HandleWidth/2, ContentY - 5);
	FVector2D VEHandleSize(HandleWidth, HandleHeight);
	bool bVEHovered = IsPointInRect(MousePosition, VEHandlePos, VEHandleSize);
	Canvas->K2_DrawBox(VEHandlePos, VEHandleSize, 1.0f, bVEHovered ? FLinearColor::White : CyanColor);

	// === EFFECT SELECTION REGION (PINK) ===
	float EffectStartX = ContentX + EffectStartTime * VizWidth;
	float EffectEndX = ContentX + EffectEndTime * VizWidth;

	// Draw effect region lines
	Canvas->K2_DrawLine(FVector2D(EffectStartX, ContentY), FVector2D(EffectStartX, ContentY + VizHeight), 3.0f, PinkColor);
	Canvas->K2_DrawLine(FVector2D(EffectEndX, ContentY), FVector2D(EffectEndX, ContentY + VizHeight), 3.0f, PinkColor);

	// Effect start handle
	FVector2D ESHandlePos(EffectStartX - HandleWidth/2, ContentY + VizHeight - 15);
	FVector2D ESHandleSize(HandleWidth, 20);
	bool bESHovered = IsPointInRect(MousePosition, ESHandlePos, ESHandleSize);
	Canvas->K2_DrawBox(ESHandlePos, ESHandleSize, 1.0f, bESHovered ? FLinearColor::White : PinkColor);

	// Effect end handle
	FVector2D EEHandlePos(EffectEndX - HandleWidth/2, ContentY + VizHeight - 15);
	FVector2D EEHandleSize(HandleWidth, 20);
	bool bEEHovered = IsPointInRect(MousePosition, EEHandlePos, EEHandleSize);
	Canvas->K2_DrawBox(EEHandlePos, EEHandleSize, 1.0f, bEEHovered ? FLinearColor::White : PinkColor);

	// === HANDLE DRAGGING ===
	if (bMouseJustPressed)
	{
		if (bVSHovered) DraggingHandle = 1;
		else if (bVEHovered) DraggingHandle = 2;
		else if (bESHovered) DraggingHandle = 3;
		else if (bEEHovered) DraggingHandle = 4;
	}

	if (!bMousePressed)
	{
		DraggingHandle = 0;
	}

	if (DraggingHandle > 0 && bMousePressed)
	{
		float NewTime = FMath::Clamp((MousePosition.X - ContentX) / VizWidth, 0.0f, 1.0f);

		switch (DraggingHandle)
		{
			case 1: VideoStartTime = FMath::Min(NewTime, VideoEndTime - 0.05f); break;
			case 2: VideoEndTime = FMath::Max(NewTime, VideoStartTime + 0.05f); break;
			case 3: EffectStartTime = FMath::Min(NewTime, EffectEndTime - 0.02f); break;
			case 4: EffectEndTime = FMath::Max(NewTime, EffectStartTime + 0.02f); break;
		}
	}

	// === TIME LABELS ===
	float LabelY = ContentY + VizHeight + 8;

	// Video time label (cyan)
	float VideoSec = VideoStartTime * TotalAudioDuration;
	float VideoEndSec = VideoEndTime * TotalAudioDuration;
	FString VideoTimeStr = FString::Printf(TEXT("Video: %d:%02d - %d:%02d"),
		(int)(VideoSec/60), (int)FMath::Fmod(VideoSec, 60.0f),
		(int)(VideoEndSec/60), (int)FMath::Fmod(VideoEndSec, 60.0f));
	DrawLabel(ContentX, LabelY, VideoTimeStr, CyanColor);

	// Effect time label (pink)
	float EffectSec = EffectStartTime * TotalAudioDuration;
	float EffectEndSec = EffectEndTime * TotalAudioDuration;
	FString EffectTimeStr = FString::Printf(TEXT("Effect: %d:%02d - %d:%02d"),
		(int)(EffectSec/60), (int)FMath::Fmod(EffectSec, 60.0f),
		(int)(EffectEndSec/60), (int)FMath::Fmod(EffectEndSec, 60.0f));
	DrawLabel(ContentX + 250, LabelY, EffectTimeStr, PinkColor);
}

FString ABeatSyncHUD::OpenFileDialog(const FString& Title, const FString& DefaultPath, const FString& FileTypes)
{
#if WITH_DESKTOP_PLATFORM
	// Try DesktopPlatform first (available in Editor builds)
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (DesktopPlatform)
	{
		TArray<FString> OutFiles;
		void* ParentWindowHandle = FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
			? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
			: nullptr;

		bool bSuccess = DesktopPlatform->OpenFileDialog(
			ParentWindowHandle,
			Title,
			DefaultPath,
			TEXT(""),
			FileTypes,
			EFileDialogFlags::None,
			OutFiles
		);

		if (bSuccess && OutFiles.Num() > 0)
		{
			return OutFiles[0];
		}
	}
#endif

	// Platform-specific native file dialogs for packaged builds
#if PLATFORM_MAC
	// Native macOS file dialog using osascript via NSTask
	@autoreleasepool
	{
		// Build AppleScript
		NSMutableString* script = [NSMutableString stringWithString:@"POSIX path of (choose file with prompt \""];
		[script appendString:[NSString stringWithUTF8String:TCHAR_TO_UTF8(*Title)]];
		[script appendString:@"\")"];

		// Run osascript via NSTask
		NSTask* task = [[NSTask alloc] init];
		[task setExecutableURL:[NSURL fileURLWithPath:@"/usr/bin/osascript"]];
		[task setArguments:@[@"-e", script]];

		NSPipe* outputPipe = [NSPipe pipe];
		NSPipe* errorPipe = [NSPipe pipe];
		[task setStandardOutput:outputPipe];
		[task setStandardError:errorPipe];

		NSError* error = nil;
		[task launchAndReturnError:&error];

		if (!error)
		{
			[task waitUntilExit];

			NSData* outputData = [[outputPipe fileHandleForReading] readDataToEndOfFile];
			NSString* output = [[NSString alloc] initWithData:outputData encoding:NSUTF8StringEncoding];

			if (output && [output length] > 0)
			{
				// Trim whitespace/newlines
				output = [output stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
				if ([output length] > 0)
				{
					return FString(UTF8_TO_TCHAR([output UTF8String]));
				}
			}
		}
	}
#elif PLATFORM_WINDOWS
	// Native Windows file dialog
	OPENFILENAMEW ofn;
	wchar_t szFile[MAX_PATH] = {0};

	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFilter = L"All Files\0*.*\0Audio Files\0*.wav;*.mp3;*.flac\0Video Files\0*.mp4;*.mov;*.avi\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrTitle = *Title ? TCHAR_TO_WCHAR(*Title) : L"Open File";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

	if (GetOpenFileNameW(&ofn))
	{
		return FString(szFile);
	}
#endif

	return FString();
}

FString ABeatSyncHUD::SaveFileDialog(const FString& Title, const FString& DefaultPath, const FString& DefaultFile, const FString& FileTypes)
{
#if WITH_DESKTOP_PLATFORM
	// Try DesktopPlatform first (available in Editor builds)
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (DesktopPlatform)
	{
		TArray<FString> OutFiles;
		void* ParentWindowHandle = FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
			? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
			: nullptr;

		bool bSuccess = DesktopPlatform->SaveFileDialog(
			ParentWindowHandle,
			Title,
			DefaultPath,
			DefaultFile,
			FileTypes,
			EFileDialogFlags::None,
			OutFiles
		);

		if (bSuccess && OutFiles.Num() > 0)
		{
			return OutFiles[0];
		}
	}
#endif

	// Platform-specific native save dialogs for packaged builds
#if PLATFORM_MAC
	// Native macOS save dialog using osascript via NSTask
	@autoreleasepool
	{
		NSMutableString* script = [NSMutableString stringWithString:@"POSIX path of (choose file name with prompt \""];
		[script appendString:[NSString stringWithUTF8String:TCHAR_TO_UTF8(*Title)]];
		[script appendString:@"\" default name \""];
		[script appendString:[NSString stringWithUTF8String:TCHAR_TO_UTF8(*DefaultFile)]];
		[script appendString:@"\")"];

		NSTask* task = [[NSTask alloc] init];
		[task setExecutableURL:[NSURL fileURLWithPath:@"/usr/bin/osascript"]];
		[task setArguments:@[@"-e", script]];

		NSPipe* outputPipe = [NSPipe pipe];
		NSPipe* errorPipe = [NSPipe pipe];
		[task setStandardOutput:outputPipe];
		[task setStandardError:errorPipe];

		NSError* error = nil;
		[task launchAndReturnError:&error];

		if (!error)
		{
			[task waitUntilExit];

			NSData* outputData = [[outputPipe fileHandleForReading] readDataToEndOfFile];
			NSString* output = [[NSString alloc] initWithData:outputData encoding:NSUTF8StringEncoding];

			if (output && [output length] > 0)
			{
				output = [output stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
				if ([output length] > 0)
				{
					return FString(UTF8_TO_TCHAR([output UTF8String]));
				}
			}
		}
	}
#elif PLATFORM_WINDOWS
	// Native Windows save dialog
	OPENFILENAMEW ofn;
	wchar_t szFile[MAX_PATH] = {0};

	// Copy default filename
	if (!DefaultFile.IsEmpty())
	{
		FCString::Strcpy(szFile, MAX_PATH, *DefaultFile);
	}

	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFilter = L"All Files\0*.*\0Video Files\0*.mp4;*.mov;*.avi\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrTitle = *Title ? TCHAR_TO_WCHAR(*Title) : L"Save File";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;

	if (GetSaveFileNameW(&ofn))
	{
		return FString(szFile);
	}
#endif

	return FString();
}

void ABeatSyncHUD::DrawScrollbar()
{
	if (!Canvas || MaxScrollOffset <= 0.0f) return;

	float ScreenWidth = Canvas->ClipX;
	float ScreenHeight = Canvas->ClipY;

	// Scrollbar track
	float ScrollbarX = ScreenWidth - 20.0f;
	float ScrollbarY = 200.0f;
	float ScrollbarHeight = ScreenHeight - 400.0f;
	float ScrollbarWidth = 8.0f;

	// Draw track
	FLinearColor TrackColor(0.1f, 0.1f, 0.15f, 0.8f);
	Canvas->K2_DrawBox(FVector2D(ScrollbarX, ScrollbarY), FVector2D(ScrollbarWidth, ScrollbarHeight), 1.0f, TrackColor);

	// Draw thumb
	float ThumbHeight = FMath::Max(30.0f, ScrollbarHeight * (ScrollbarHeight / (ScrollbarHeight + MaxScrollOffset)));
	float ThumbY = ScrollbarY + (ScrollOffset / MaxScrollOffset) * (ScrollbarHeight - ThumbHeight);

	Canvas->K2_DrawBox(FVector2D(ScrollbarX, ThumbY), FVector2D(ScrollbarWidth, ThumbHeight), 1.0f, CyanColor);
}

void ABeatSyncHUD::DrawTitle(float InScrollOffset)
{
	if (!Canvas) return;

	FString Title = TEXT("MTV TRIP SITTER - BEATSYNC");

	UFont* Font = CustomFontLarge ? CustomFontLarge : GEngine->GetLargeFont();
	if (Font)
	{
		float TextWidth, TextHeight;
		float Scale = CustomFontLarge ? 2.0f : 1.5f;
		Canvas->TextSize(Font, Title, TextWidth, TextHeight);

		float X = (Canvas->ClipX - TextWidth * Scale) / 2.0f;
		float Y = 175.0f - InScrollOffset;

		// Only draw if visible
		if (Y + TextHeight * Scale > 0 && Y < Canvas->ClipY)
		{
			FCanvasTextItem TextItem(FVector2D(X, Y), FText::FromString(Title), Font, CyanColor);
			TextItem.Scale = FVector2D(Scale, Scale);
			TextItem.bOutlined = true;
			TextItem.OutlineColor = FLinearColor(0.0f, 0.0f, 0.0f, 0.3f); // Softer, more transparent outline
			TextItem.BlendMode = SE_BLEND_Translucent;
			Canvas->DrawItem(TextItem);
		}
	}
}

void ABeatSyncHUD::DrawPanel(float X, float Y, float Width, float Height, const FString& Title, FLinearColor TitleColor)
{
	if (!Canvas) return;

	// Skip if panel is completely off screen
	if (Y + Height < 0 || Y > Canvas->ClipY) return;

	// Draw dark semi-transparent background
	Canvas->K2_DrawBox(FVector2D(X, Y), FVector2D(Width, Height), 1.0f, DarkBG);

	// Add 30% colored tint overlay matching panel's accent color
	FLinearColor PanelTint = TitleColor;
	PanelTint.A = 0.3f;
	Canvas->K2_DrawBox(FVector2D(X, Y), FVector2D(Width, Height), 1.0f, PanelTint);

	// Draw glowing border
	float BorderThickness = 2.0f;
	FLinearColor BorderColor = TitleColor;
	BorderColor.A = 0.9f;

	// Top border
	Canvas->K2_DrawLine(FVector2D(X, Y), FVector2D(X + Width, Y), BorderThickness, BorderColor);
	// Bottom border
	Canvas->K2_DrawLine(FVector2D(X, Y + Height), FVector2D(X + Width, Y + Height), BorderThickness, BorderColor);
	// Left border
	Canvas->K2_DrawLine(FVector2D(X, Y), FVector2D(X, Y + Height), BorderThickness, BorderColor);
	// Right border
	Canvas->K2_DrawLine(FVector2D(X + Width, Y), FVector2D(X + Width, Y + Height), BorderThickness, BorderColor);

	// Draw title bar background
	FLinearColor TitleBG = TitleColor;
	TitleBG.A = 0.3f;
	Canvas->K2_DrawBox(FVector2D(X + 1, Y + 1), FVector2D(Width - 2, 25.0f), 1.0f, TitleBG);

	// Draw title text
	UFont* Font = CustomFont ? CustomFont : GEngine->GetSmallFont();
	if (Font)
	{
		FCanvasTextItem TextItem(FVector2D(X + 12, Y + 5), FText::FromString(Title), Font, FLinearColor::White);
		TextItem.Scale = FVector2D(1.1f, 1.1f);
		TextItem.bOutlined = true;
		TextItem.OutlineColor = FLinearColor::Black;
		Canvas->DrawItem(TextItem);
	}
}
