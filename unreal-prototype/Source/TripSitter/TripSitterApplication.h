#pragma once

#include "CoreMinimal.h"
#include "TripSitter.h"  // Module API header for TRIPSITTER_API macro
#include "Framework/Application/SlateApplication.h"
#include "Widgets/SWindow.h"

/**
 * Main application class for TripSitter desktop application
 */
class TRIPSITTER_API FTripSitterApplication : public TSharedFromThis<FTripSitterApplication>
{
public:
    FTripSitterApplication();

    /** Initialize the application */
    bool Initialize();

    /** Run the application main loop */
    int32 Run();

    /** Shutdown the application */
    void Shutdown();

private:
    /** Create the main window */
    TSharedPtr<SWindow> CreateMainWindow();

    /** Handle window close requests */
    void OnWindowClosed(const TSharedRef<SWindow>& Window);

    /** Main application window */
    TSharedPtr<SWindow> MainWindow;

    /** Tracks whether this instance created/owns the Slate application */
    bool bOwnsSlateApplication = false;
};