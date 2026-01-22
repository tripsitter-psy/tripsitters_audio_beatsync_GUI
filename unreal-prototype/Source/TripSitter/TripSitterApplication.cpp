#include "TripSitterApplication.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Layout/SBorder.h"
#include "Styling/CoreStyle.h"
#include "StandaloneRenderer.h"

#if PLATFORM_WINDOWS
#include "Windows/WindowsApplication.h"
#include "Windows/AllowWindowsPlatformTypes.h"
#include <Windows.h>
#include "Windows/HideWindowsPlatformTypes.h"
#endif

FTripSitterApplication::FTripSitterApplication()
{
}

bool FTripSitterApplication::Initialize()
{
    // Create platform application and renderer using StandaloneRenderer
    FSlateApplication::Create();

    // Initialize standalone renderer
    TSharedPtr<FSlateRenderer> Renderer = GetStandardStandaloneRenderer();
    if (!Renderer.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("TripSitterApplication: Failed to create standalone renderer."));
        FSlateApplication::Shutdown();
        return false;
    }
    FSlateApplication::Get().InitializeRenderer(Renderer.ToSharedRef());

    // Create main window
    MainWindow = CreateMainWindow();
    if (!MainWindow.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("TripSitterApplication: Failed to create main window."));
        FSlateApplication::Shutdown();
        return false;
    }

    // Show the window
    FSlateApplication::Get().AddWindow(MainWindow.ToSharedRef());

    return true;
}

int32 FTripSitterApplication::Run()
{
    // Main application loop
    while (!IsEngineExitRequested())
    {
        // Process Slate messages
        FSlateApplication::Get().PumpMessages();

        // Tick Slate
        FSlateApplication::Get().Tick();

        // Small delay to prevent 100% CPU usage
        FPlatformProcess::Sleep(0.01f);
    }

    return 0;
}

void FTripSitterApplication::Shutdown()
{
    // Destroy main window before tearing down Slate
    if (MainWindow.IsValid())
    {
        MainWindow->RequestDestroyWindow();
        MainWindow.Reset();
    }

    // Now shutdown Slate application
    FSlateApplication::Shutdown();
}

TSharedPtr<SWindow> FTripSitterApplication::CreateMainWindow()
{
    // Create main window
    TSharedPtr<SWindow> Window = SNew(SWindow)
        .Title(FText::FromString(TEXT("TripSitter Beat Sync Editor")))
        .ClientSize(FVector2D(1200, 800))
        .SupportsMaximize(true)
        .SupportsMinimize(true)
        .IsInitiallyMaximized(false);

    // Create a simple UI layout
    TSharedPtr<SWidget> Content = SNew(SBorder)
        .BorderBackgroundColor(FLinearColor::White)
        .Padding(20)
        [
            SNew(SVerticalBox)
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 0, 0, 20)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("TripSitter Beat Sync Editor")))
                .Font(FSlateFontInfo(FCoreStyle::GetDefaultFont(), 24))
                .ColorAndOpacity(FLinearColor::Black)
            ]
            + SVerticalBox::Slot()
            .FillHeight(1.0f)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Application initialized successfully!\n\nThis is a Program target application with Slate UI.")))
                .Font(FSlateFontInfo(FCoreStyle::GetDefaultFont(), 14))
                .ColorAndOpacity(FLinearColor::Black)
            ]
        ];

    Window->SetContent(Content.ToSharedRef());

    // Handle window close
    Window->SetOnWindowClosed(FOnWindowClosed::CreateSP(this, &FTripSitterApplication::OnWindowClosed));

    return Window;
}

void FTripSitterApplication::OnWindowClosed(const TSharedRef<SWindow>& Window)
{
    // Request application exit when main window is closed
    RequestEngineExit(TEXT("Main window closed"));
}
