// TripSitter - Beat Sync Editor (Standalone Slate Application)

#include "TripSitterApp.h"
#include "RequiredProgramMainCPPInclude.h"
#include "Framework/Application/SlateApplication.h"
#include "StandaloneRenderer.h"
#include "HAL/PlatformProcess.h"
#include "Misc/CoreDelegates.h"
#include "Async/TaskGraphInterfaces.h"

// Include TripSitter UI widget and backend loader
#include "STripSitterMainWidget.h"
#include "BeatsyncLoader.h"

IMPLEMENT_APPLICATION(TripSitter, "TripSitter");

int RunTripSitter(const TCHAR* CommandLine)
{
	// Initialize engine subsystems
	GEngineLoop.PreInit(CommandLine);

	// Initialize Slate as a standalone application (NOT a game!)
	FSlateApplication::InitializeAsStandaloneApplication(GetStandardStandaloneRenderer());

	// Initialize the beatsync backend DLL
	FBeatsyncLoader::Initialize();

	// Create main application window with TripSitter UI
	TSharedRef<SWindow> MainWindow = SNew(SWindow)
		.Title(FText::FromString(TEXT("TripSitter - Beat Sync Editor")))
		.ClientSize(FVector2D(1400, 900))
		.SupportsMinimize(true)
		.SupportsMaximize(true)
		.IsInitiallyMaximized(false)
		[
			SNew(STripSitterMainWidget)
		];

	FSlateApplication::Get().AddWindow(MainWindow);

	// Application main loop (matching SlateViewer pattern)
	while (!IsEngineExitRequested())
	{
		BeginExitIfRequested();

		FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread);
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
