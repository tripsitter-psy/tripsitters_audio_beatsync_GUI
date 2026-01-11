#include "TripSitterApplication.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Input/SButton.h"
#include "Styling/CoreStyle.h"

FTripSitterApplication::FTripSitterApplication()
{
}

bool FTripSitterApplication::Initialize()
{
    // Initialize Slate application
    SlateApp = FSlateApplication::Create();

    // Create main window
    MainWindow = CreateMainWindow();
    if (!MainWindow.IsValid())
    {
        return false;
    }

    // Show the window
    FSlateApplication::Get().AddWindow(MainWindow.ToSharedRef());

    return true;
}

int32 FTripSitterApplication::Run()
{
    // Main application loop
    while (!GIsRequestingExit)
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
    if (MainWindow.IsValid())
    {
        MainWindow->RequestDestroyWindow();
        MainWindow.Reset();
    }

    if (SlateApp.IsValid())
    {
        SlateApp.Reset();
    }
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
    Window->SetOnWindowClosed(FOnWindowClosed::CreateRaw(this, &FTripSitterApplication::OnWindowClosed));

    return Window;
}

void FTripSitterApplication::OnWindowClosed(const TSharedRef<SWindow>& Window)
{
    // Request application exit when main window is closed
    GIsRequestingExit = true;
}