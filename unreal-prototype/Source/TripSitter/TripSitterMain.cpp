// TripSitter - Program Entry Point
#include "CoreMinimal.h"
#include "RequiredProgramMainCPPInclude.h"
#include "Framework/Application/SlateApplication.h"
#include "StandaloneRenderer.h"
#include "Stats/StatsSystem.h"
#include "Private/STripSitterMainWidget.h"
#include "Private/BeatsyncLoader.h"

IMPLEMENT_APPLICATION(TripSitter, "TripSitter");

int RunTripSitter(const TCHAR* CommandLine)
{
    FTaskTagScope TaskTagScope(ETaskTag::EGameThread);

    // Initialize the engine
    int32 PreInitResult = GEngineLoop.PreInit(CommandLine);
    if (PreInitResult != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("GEngineLoop.PreInit failed with code %d"), PreInitResult);
        FPlatformMisc::RequestExit(true);
        return PreInitResult;
    }

    // Make sure all UObject classes are registered and default properties have been initialized
    ProcessNewlyLoadedUObjects();

    // Tell the module manager it may now process newly-loaded UObjects when new C++ modules are loaded
    FModuleManager::Get().StartProcessingNewlyLoadedObjects();

    // Initialize Slate as standalone application
    FSlateApplication::InitializeAsStandaloneApplication(GetStandardStandaloneRenderer());
    FSlateApplication::InitHighDPI(true);

    // Initialize the beatsync backend DLL
    if (!FBeatsyncLoader::Initialize())
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to initialize beatsync backend. Beat sync features will be disabled."));
    }

    // Create main window
    TSharedRef<SWindow> MainWindow = SNew(SWindow)
        .Title(FText::FromString(TEXT("TripSitter Beat Sync Editor")))
        .ClientSize(FVector2D(1400, 900))
        .SupportsMaximize(true)
        .SupportsMinimize(true)
        .IsInitiallyMaximized(false);

    // Create the main widget content
    MainWindow->SetContent(
        SNew(STripSitterMainWidget)
    );

    // Add window and show
    FSlateApplication::Get().AddWindow(MainWindow);

    // Main application loop
    while (!IsEngineExitRequested())
    {
        BeginExitIfRequested();

        FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread);
        UE::Stats::FStats::AdvanceFrame(false);
        FTSTicker::GetCoreTicker().Tick(FApp::GetDeltaTime());
        FSlateApplication::Get().PumpMessages();
        FSlateApplication::Get().Tick();
        FPlatformProcess::Sleep(0.01f);

        GFrameCounter++;
    }

    // Cleanup
    FBeatsyncLoader::Shutdown();
    FCoreDelegates::OnExit.Broadcast();
    FSlateApplication::Shutdown();
    FModuleManager::Get().UnloadModulesAtShutdown();

    GEngineLoop.AppPreExit();
    GEngineLoop.AppExit();

    return 0;
}
