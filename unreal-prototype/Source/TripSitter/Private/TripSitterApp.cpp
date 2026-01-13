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
	int32 PreInitResult = GEngineLoop.PreInit(CommandLine);
	if (PreInitResult != 0)
	{
		UE_LOG(LogTemp, Error, TEXT("GEngineLoop.PreInit failed with error code: %d"), PreInitResult);
		return PreInitResult;
	}

	// Initialize Slate as a standalone application (NOT a game!)
	FSlateApplication::InitializeAsStandaloneApplication(GetStandardStandaloneRenderer());

	// Initialize the beatsync backend DLL
	if (!FBeatsyncLoader::Initialize())
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to initialize beatsync backend. Beat sync features will be disabled."));
		// Continue with application startup but beatsync features will be unavailable
	}

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
	FBeatsyncLoader::Shutdown(); // If depends on Slate/subsystems, keep before Slate shutdown
	FSlateApplication::Shutdown();
	GEngineLoop.AppPreExit(); // Broadcasts OnExit internally
	FModuleManager::Get().UnloadModulesAtShutdown();
	GEngineLoop.AppExit();

	return 0;
}
